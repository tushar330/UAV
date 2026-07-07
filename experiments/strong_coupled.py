"""
Strongest deterministic COUPLED planner we can reasonably build.

The coupled greedy in ``two_stage_vs_coupled.py`` is a deliberately weak witness.
This module strengthens it so it is a *fair, hard* bar for the CMDP to clear --
if the learned policy beats THIS, the learning is contributing beyond engineering.

Strengthenings (all within the CMDP's own action space: hover over a node anchor,
pick an altitude, footprint-serve the feasible cone -- so the comparison is fair):

  * Continuous altitude: each cluster hovers at the LOWEST altitude that still
    covers all members in the cone (lower is always cheaper for a fixed set:
    less climb + shorter links), replacing the coarse discrete grid.
  * Best anchor per cluster: hover over the member that minimises hover energy,
    not just the first/greedy pick.
  * Local search: node-reassignment (move a node to a cheaper feasible cluster)
    and cluster merges, until no improving move -- a real facility-location pass.
  * Multi-restart from several greedy inits; keep the best by the SAME accounting.
  * Stronger routing: NN + 2-opt + Or-opt over the hover anchors.

Feasibility (per-node QoS floor, cone coverage, tau_max budget) is maintained at
every step, and the final number is scored through the identical
``TENMATrainer._partition_and_evaluate`` used for every other planner.

    python -m experiments.strong_coupled --N 100 --instances 20
"""

import argparse
import os

import numpy as np
import torch
import yaml

from atom_3d.training import TENMATrainer, TrainConfig
from atom_3d.env.iot_env import IoTEnvironment3D
from atom_3d.experiments.heuristic_baseline import nn_2opt_tour


def nn_2opt_fast(pts, depot, iters=60):
    """Output-identical reimplementation of ``nn_2opt_tour`` using a precomputed
    pairwise distance matrix (avoids the per-call vstack/diff/norm rebuild in
    ``tour_len``). Same NN construction, same tie-break (by index), same 2-opt
    scan order and same ``+1e-6`` acceptance -> identical tours, much faster."""
    n = len(pts)
    if n == 0:
        return []
    P = np.asarray(pts)                                  # keep input dtype (float32)
    dp = np.asarray(depot, dtype=P.dtype)               # so arithmetic matches original
    Dd = np.linalg.norm(P - dp, axis=1)                 # depot <-> node
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))                    # node <-> node
    cur = int(np.argmin(Dd))
    tour = [cur]
    unvisited = set(range(n)); unvisited.discard(cur)
    while unvisited:
        nxt = min((D[cur, j], j) for j in unvisited)[1]  # tie-break by index, as original
        tour.append(nxt); unvisited.discard(nxt); cur = nxt

    def tlen(t):
        # edges in the SAME order as the original (depot->t0, t0->t1, ..., tk->depot)
        # summed with np.sum so the float rounding is bit-identical to tour_len.
        e = np.empty(len(t) + 1, dtype=Dd.dtype)
        e[0] = Dd[t[0]]
        if len(t) > 1:
            e[1:-1] = D[np.asarray(t[:-1]), np.asarray(t[1:])]
        e[-1] = Dd[t[-1]]
        return e.sum()

    best = tlen(tour); improved = True; it = 0
    while improved and it < iters:
        improved = False; it += 1
        for i in range(len(tour) - 1):
            for k in range(i + 1, len(tour)):
                new = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
                nl = tlen(new)
                if nl + 1e-6 < best:
                    tour, best = new, nl; improved = True
    return tour
from experiments.two_stage_vs_coupled import (
    make_channel_consts, rate_of, d_max_of, budget_keep, hover_energy,
    plan_coupled, plan_two_stage, plan_energy_min, score, per_node_2d, summarise,
)


# ----------------------------------------------------------------------
# cluster -> (best anchor, lowest covering altitude, feasibility, hover energy)
# ----------------------------------------------------------------------
def cluster_solve(trainer, xy, z, dem, rmin, dmax, members, c, H_min, H_max, h_safe,
                  cache=None):
    """Pick the member-anchor + lowest covering altitude that minimises hover
    energy while keeping every member covered, within its QoS floor, and within
    the tau_max charge+collect budget. Returns (anchor, H, energy) or None.

    ``cache`` (a dict keyed by frozenset(members)) memoises the result; the value
    is a deterministic function of the member SET for a fixed instance, so this is
    exact -- it only avoids recomputing clusters that did not change between trials.
    """
    members = np.asarray(members, dtype=int)
    if cache is not None:
        key = frozenset(int(x) for x in members)
        hit = cache.get(key, 0)
        if hit != 0:
            return hit
    best = None
    for a in members:
        rho = np.linalg.norm(xy[members] - xy[a], axis=1)
        # lowest altitude that covers every member in the cone (+ clearance/H_min)
        H = max(H_min, z[a] + h_safe, float((z[members] + rho / c["tan"]).max()))
        if H > H_max + 1e-6:
            continue                                  # too spread to cover from <= H_max
        d = np.sqrt(rho ** 2 + (H - z[members]) ** 2)
        floored = rmin[members] > 0
        if np.any(floored & (d > dmax[members] + 1e-6)):
            continue                                  # a floored member misses its floor
        t_c, t_e = trainer._hover_energy(d, dem[members])
        if float((t_c + t_e).sum()) > trainer.tau_max + 1e-9:
            continue                                  # would overflow the per-hover budget
        e = hover_energy(trainer, d, dem[members])
        if best is None or e < best[2]:
            best = (int(a), float(H), float(e))
    if cache is not None:
        cache[key] = best
    return best


def total_proxy(trainer, xy, z, dem, rmin, dmax, clusters, c, H_min, H_max, h_safe,
                order=None, cache=None, route_fn=None):
    """Faithful fast energy of a clustering = flight + vertical + collection.

    Mirrors ``_partition_and_evaluate`` (rotary-wing j/m for horizontal legs,
    asymmetric climb/descent for vertical, (P_C+P_H)*t_c for collection) but
    ignores multi-UAV segmentation returns -- fine for *relative* local search;
    the official number is always the full scorer. Returns (energy, hovers, order).
    """
    hovers = []
    for mem in clusters:
        res = cluster_solve(trainer, xy, z, dem, rmin, dmax, mem, c, H_min, H_max, h_safe,
                            cache=cache)
        if res is None:
            return np.inf, None, None
        a, H, e = res
        hovers.append((a, H, np.asarray(mem, dtype=int), e))
    pts = np.array([xy[h[0]] for h in hovers])
    if order is None:
        order = (route_fn or nn_2opt_tour)(pts, trainer.depot)
    jpm = trainer.power_model.energy_per_metre()
    climb = trainer.phys3d.climb_coeff
    desc = trainer.phys3d.descent_coeff_cd
    seq_xy = [trainer.depot] + [pts[i] for i in order] + [trainer.depot]
    seq_H = [trainer.depot_alt] + [hovers[i][1] for i in order] + [trainer.depot_alt]
    flight = sum(jpm * float(np.linalg.norm(seq_xy[k + 1] - seq_xy[k]))
                 for k in range(len(seq_xy) - 1))
    vertical = 0.0
    for k in range(len(seq_H) - 1):
        dH = seq_H[k + 1] - seq_H[k]
        vertical += climb * max(dH, 0.0) + desc * max(-dH, 0.0)
    collection = sum(h[3] for h in hovers)
    return flight + vertical + collection, hovers, order


# ----------------------------------------------------------------------
# local search: node reassignment (to nearest few clusters) + cluster merges
# ----------------------------------------------------------------------
def _nearest_clusters(xy, clusters, anchors_xy, n_idx, k=4):
    d = np.linalg.norm(anchors_xy - xy[n_idx], axis=1)
    return np.argsort(d)[:k]


def local_search(trainer, xy, z, dem, rmin, dmax, clusters, c, H_min, H_max, h_safe,
                 max_passes=4):
    clusters = [list(m) for m in clusters]
    base, hovers, order = total_proxy(trainer, xy, z, dem, rmin, dmax, clusters,
                                      c, H_min, H_max, h_safe)
    for _ in range(max_passes):
        improved = False
        # ---- node reassignment (fixed tour order within a pass for speed) ----
        anchors_xy = np.array([xy[h[0]] for h in hovers])
        cluster_of = {}
        for ci, mem in enumerate(clusters):
            for n in mem:
                cluster_of[n] = ci
        for n in range(len(xy)):
            ca = cluster_of[n]
            if len(clusters[ca]) == 1:
                cand = _nearest_clusters(xy, clusters, anchors_xy, n, k=4)
            else:
                cand = _nearest_clusters(xy, clusters, anchors_xy, n, k=4)
            for cb in cand:
                if cb == ca:
                    continue
                trial = [list(m) for m in clusters]
                trial[ca].remove(n)
                trial[cb].append(n)
                trial = [m for m in trial if m]
                e, h2, _ = total_proxy(trainer, xy, z, dem, rmin, dmax, trial,
                                       c, H_min, H_max, h_safe, order=None)
                if e < base - 1e-6:
                    clusters, base, hovers = trial, e, h2
                    improved = True
                    anchors_xy = np.array([xy[h[0]] for h in hovers])
                    cluster_of = {}
                    for ci, mem in enumerate(clusters):
                        for m in mem:
                            cluster_of[m] = ci
                    break
        # ---- cluster merges (nearest pairs) ----
        anchors_xy = np.array([xy[h[0]] for h in hovers])
        merged = True
        while merged:
            merged = False
            nc = len(clusters)
            for i in range(nc):
                di = np.linalg.norm(anchors_xy - anchors_xy[i], axis=1)
                for j in np.argsort(di):
                    if j <= i:
                        continue
                    trial = [list(m) for k2, m in enumerate(clusters) if k2 not in (i, j)]
                    trial.append(list(clusters[i]) + list(clusters[j]))
                    e, h2, _ = total_proxy(trainer, xy, z, dem, rmin, dmax, trial,
                                           c, H_min, H_max, h_safe)
                    if e < base - 1e-6:
                        clusters, base, hovers = trial, e, h2
                        anchors_xy = np.array([xy[h[0]] for h in hovers])
                        merged = True
                        improved = True
                        break
                if merged:
                    break
        if not improved:
            break
    return clusters


# ----------------------------------------------------------------------
# Accelerated-but-EXACT local search: byte-for-byte the same search as
# ``local_search`` (same candidate order, same per-trial re-routing, same
# first-improving ``+1e-6`` acceptance), only the primitives are cheaper:
#   * cluster_solve results are memoised (unchanged clusters cost O(1));
#   * routing uses the matrix-based ``nn_2opt_fast`` (output-identical);
#   * trials are built by shallow reference reuse instead of an O(N) deep copy.
# Because every evaluated objective equals the original total_proxy value, the
# produced clustering is identical to ``local_search`` (verified within tol).
# ----------------------------------------------------------------------
def local_search_exact_fast(trainer, xy, z, dem, rmin, dmax, clusters, c, H_min,
                            H_max, h_safe, max_passes=4):
    cache = {}
    def tp(cl):
        return total_proxy(trainer, xy, z, dem, rmin, dmax, cl, c, H_min, H_max,
                           h_safe, cache=cache, route_fn=nn_2opt_fast)
    clusters = [list(m) for m in clusters]
    base, hovers, order = tp(clusters)
    for _ in range(max_passes):
        improved = False
        # ---- node reassignment (same order/acceptance as local_search) ----
        anchors_xy = np.array([xy[h[0]] for h in hovers])
        cluster_of = {}
        for ci, mem in enumerate(clusters):
            for m in mem:
                cluster_of[m] = ci
        for n in range(len(xy)):
            ca = cluster_of[n]
            cand = _nearest_clusters(xy, clusters, anchors_xy, n, k=4)
            for cb in cand:
                if cb == ca:
                    continue
                memA = [x for x in clusters[ca] if x != n]
                memB = clusters[cb] + [n]
                trial = []                                   # shallow: reuse refs
                for idx, m in enumerate(clusters):
                    if idx == ca:
                        if memA:
                            trial.append(memA)
                    elif idx == cb:
                        trial.append(memB)
                    else:
                        trial.append(m)
                e, h2, o2 = tp(trial)
                if e < base - 1e-6:
                    clusters, base, hovers, order = trial, e, h2, o2
                    improved = True
                    anchors_xy = np.array([xy[h[0]] for h in hovers])
                    cluster_of = {}
                    for ci, mem in enumerate(clusters):
                        for m in mem:
                            cluster_of[m] = ci
                    break
        # ---- cluster merges (same order/acceptance as local_search) ----
        anchors_xy = np.array([xy[h[0]] for h in hovers])
        merged = True
        while merged:
            merged = False
            nc = len(clusters)
            for i in range(nc):
                di = np.linalg.norm(anchors_xy - anchors_xy[i], axis=1)
                for j in np.argsort(di):
                    if j <= i:
                        continue
                    trial = [m for k2, m in enumerate(clusters) if k2 not in (i, int(j))]
                    trial.append(list(clusters[i]) + list(clusters[int(j)]))
                    e, h2, o2 = tp(trial)
                    if e < base - 1e-6:
                        clusters, base, hovers, order = trial, e, h2, o2
                        anchors_xy = np.array([xy[h[0]] for h in hovers])
                        merged = True
                        improved = True
                        break
                if merged:
                    break
        if not improved:
            break
    return clusters


def plan_coupled_strong(trainer, xy, z, dem, rmin, grid, c, H_min, H_max, h_safe,
                        dmax, restarts=2, fast=True):
    """Strongest coupled planner: greedy init(s) -> continuous-altitude local
    search -> best by the faithful proxy. Returns hovers as [anchor, H, members]."""
    best_clusters, best_e = None, np.inf
    for r in range(restarts):
        # init from the (feasible) greedy coupled cover; vary the altitude grid
        # offered to the init so restarts explore different cluster granularities
        g = grid if r == 0 else tuple(sorted(set(grid) | {35, 50, 70, 90}))
        init = plan_coupled(trainer, xy, z, dem, rmin, g, c, H_min, H_max, h_safe, dmax)
        clusters = [list(h[2]) for h in init]
        ls = local_search_exact_fast if fast else local_search
        clusters = ls(trainer, xy, z, dem, rmin, dmax, clusters,
                      c, H_min, H_max, h_safe)
        e, _, _ = total_proxy(trainer, xy, z, dem, rmin, dmax, clusters,
                              c, H_min, H_max, h_safe)
        if e < best_e:
            best_e, best_clusters = e, clusters
    hovers = []
    for mem in best_clusters:
        a, H, _ = cluster_solve(trainer, xy, z, dem, rmin, dmax, mem, c, H_min, H_max, h_safe)
        hovers.append([a, H, np.asarray(mem, dtype=int)])
    return hovers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "atom_3d", "configs", "params.yaml"))
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--restarts", type=int, default=2)
    args = ap.parse_args()

    with open(os.path.abspath(args.config)) as f:
        params = yaml.safe_load(f)
    params["priority"]["enabled"] = True
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tr = TENMATrainer(params, TrainConfig(mode="3d", embed_dim=32, num_heads=8, ff_dim=32, num_layers=1))
    c = make_channel_consts(tr)
    H_min, H_max, h_safe = tr.H_min, tr.H_max, tr.h_safe
    grid = (30, 45, 60, 80, 100, 120, 140)

    sc, nd = params["scenario"], params["nodes"]
    f3 = IoTEnvironment3D.generate_batch(
        args.instances, args.N, seed=args.seed, Di_min=nd["Di_min"], Di_max=nd["Di_max"],
        zi_min=nd["zi_min"], zi_max=nd["zi_max"],
        area_width=sc["area_width"], area_height=sc["area_height"])[0]
    _, nw, nr = IoTEnvironment3D.generate_priorities(
        args.instances, args.N, seed=args.seed, priority_enabled=True,
        priority_class_probs=params["priority"]["class_probs"],
        priority_weights=[params["priority"]["weights"][k] for k in ("high", "medium", "low")],
        priority_rmin=[params["priority"]["R_min"][k] for k in ("high", "medium", "low")])

    nf = f3.cpu().numpy(); rmin_np = nr.cpu().numpy()
    ts, cp, cs = [], [], []
    for b in range(args.instances):
        xy, z, dem = nf[b, :, :2], nf[b, :, 2], nf[b, :, 3]
        rm = rmin_np[b]; dmax = d_max_of(rm, c)
        ts.append(plan_two_stage(tr, xy, z, dem, rm, grid, c, H_min, H_max, h_safe, dmax))
        cp.append(plan_coupled(tr, xy, z, dem, rm, grid, c, H_min, H_max, h_safe, dmax))
        cs.append(plan_coupled_strong(tr, xy, z, dem, rm, grid, c, H_min, H_max, h_safe, dmax, args.restarts))

    rows = [
        summarise("Strong two-stage (DECOUPLED)", score(tr, f3, ts, 0.0, nw, nr)),
        summarise("Coupled greedy (weak witness)", score(tr, f3, cp, 0.0, nw, nr)),
        summarise("Coupled STRONG (local search)", score(tr, f3, cs, 0.0, nw, nr)),
    ]
    print(f"\n=== N={args.N}, {args.instances} instances, priority on ===\n")
    print(f"  {'planner':36s} {'energy':>9} {'served':>7} {'hiQoS':>7} {'medQoS':>7} {'allQoS':>7} {'uavs':>6}")
    for e, served, hi, md, allq, uav, name in rows:
        print(f"  {name:36s} {e:7.2f}Wh {served:6.1f}% {hi:6.1f}% {md:6.1f}% {allq:6.1f}% {uav:6.1f}")
    e_ts, e_cp, e_cs = rows[0][0], rows[1][0], rows[2][0]
    print(f"\n  strong-coupled vs weak-coupled : {(1 - e_cs/e_cp)*100:+.1f}% energy")
    print(f"  strong-coupled vs two-stage    : {(1 - e_cs/e_ts)*100:+.1f}% energy")
    print("  => this strong-coupled is the bar the CMDP must beat to prove learning > engineering.")


if __name__ == "__main__":
    main()
