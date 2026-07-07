"""
Premise test: is QoS-constrained UAV harvesting a *genuinely coupled* problem,
or does energy-min-then-repair already solve it?

We compare, on identical instances and through the SAME energy accounting
(``TENMATrainer._partition_and_evaluate``), four deterministic planners:

  1. 2D-AUTO            : NN+2opt, visit every node at the fixed altitude.
                          Meets QoS trivially (flies low) but pays max energy.
  2. 3D energy-min      : QoS-blind greedy footprint cover (high hovers).
                          Cheap but INFEASIBLE -- this is the regime our current
                          learned planner occupies (~85 m, ~0% high-QoS).
  3. Strong two-stage   : the adversary, given every advantage. Stage 1 covers
     (DECOUPLED)          everything energy-min from high hovers; every critical
                          node left below its floor is *moved* into a dedicated
                          low repair hover; repair hovers are CLUSTERED (several
                          criticals share one dive) and pick the shallowest
                          feasible altitude (min climb-back); ALL hovers (high +
                          low) are then routed by one combined NN+2opt tour so the
                          two passes never pay a separate-tour penalty.
  4. Coupled greedy     : the witness. ONE QoS-aware pass. When a hover must dive
     (JOINT)              for a critical node it greedily ABSORBS every still-
                          uncovered node inside that low footprint (amortises the
                          dive over neighbours); routine-only regions get cheap
                          high hovers. Same combined NN+2opt routing.

Decision rule:
  * coupled << two-stage, consistently  -> coupling is real and exploitable
                                           => the CMDP has headroom; implement it.
  * coupled ~= two-stage                -> no exploitable coupling
                                           => rethink the contribution.

Both feasible planners are verified to hit ~100% critical-QoS (else the energy
number is meaningless). Run e.g.:

    python -m experiments.two_stage_vs_coupled --N 100 --instances 20
"""

import argparse
import math
import os

import numpy as np
import torch
import yaml

from atom_3d.models.trajectory_decoder import DecodePlan
from atom_3d.training import TENMATrainer, TrainConfig
from atom_3d.env.iot_env import IoTEnvironment3D
from atom_3d.experiments.heuristic_baseline import nn_2opt_tour, _plan_from_steps


# ----------------------------------------------------------------------
# channel helpers (shared physics: rate and the QoS distance ceiling d_max)
# ----------------------------------------------------------------------
def make_channel_consts(trainer):
    return dict(B=trainer.bandwidth, beta=trainer.beta, sigma2=trainer.sigma2,
                P_T=trainer.P_T, alpha=trainer.path_loss_exp, tan=trainer.tan_theta)


def rate_of(d, c):
    snr = c["P_T"] * c["beta"] / (np.maximum(d, 1e-3) ** c["alpha"] * c["sigma2"])
    return c["B"] * np.log2(1.0 + snr)


def d_max_of(R_min, c):
    """Largest 3D distance whose rate still meets R_min (inf when R_min<=0)."""
    R_min = np.asarray(R_min, dtype=np.float64)
    out = np.full_like(R_min, np.inf)
    pos = R_min > 0
    spec = (2.0 ** (R_min[pos] / c["B"]) - 1.0)
    out[pos] = (c["P_T"] * c["beta"] / (c["sigma2"] * spec)) ** (1.0 / c["alpha"])
    return out


# ----------------------------------------------------------------------
# per-hover budgeted membership (respect tau_max so the scorer won't drop nodes)
# ----------------------------------------------------------------------
def budget_keep(trainer, d3d, dem, member_idx):
    """Keep nearest members whose cumulative charge+collect time fits tau_max."""
    if member_idx.size == 0:
        return member_idx
    t_c, t_e = trainer._hover_energy(d3d, dem)
    order = np.argsort(d3d)
    cum = np.cumsum((t_c + t_e)[order])
    keep = order[cum <= trainer.tau_max]
    if keep.size == 0:
        keep = order[:1]
    return member_idx[keep]


def hover_energy(trainer, d3d, dem):
    t_c, t_e = trainer._hover_energy(d3d, dem)          # t_e == 0 when WPT is off
    return (trainer.P_T + trainer.P_H) * float(t_e.sum()) + \
           (trainer.P_C + trainer.P_H) * float(t_c.sum())


# ----------------------------------------------------------------------
# feasible-altitude band for a hover at anchor a covering a candidate set
# ----------------------------------------------------------------------
def feasible_H(xy, z, rmin, a, members, c, H_min, H_max, h_safe, dmax):
    """Highest altitude that (a) covers every member in the cone and (b) meets
    every floored member's rate floor. Returns None if no feasible H exists."""
    rho = np.linalg.norm(xy[members] - xy[a], axis=1)
    # coverage lower bound: cone must reach each member, plus clearance/H_min
    H_lo = max(H_min, float((z[members] + np.maximum(rho / c["tan"], h_safe)).max()))
    # QoS upper bound: floored members cap H at z_i + sqrt(dmax_i^2 - rho_i^2)
    H_hi = H_max
    fl = members[rmin[members] > 0]
    if fl.size:
        rho_f = np.linalg.norm(xy[fl] - xy[a], axis=1)
        slack = dmax[fl] ** 2 - rho_f ** 2
        if np.any(slack <= 0):           # a floored member is too far horizontally
            return None
        H_hi = min(H_hi, float((z[fl] + np.sqrt(slack)).min()))
    if H_lo > H_hi + 1e-6:
        return None
    return H_hi                          # shallowest feasible dive == least climb


# ----------------------------------------------------------------------
# Planner 2: QoS-blind energy-min footprint cover (≈ current learned planner)
# ----------------------------------------------------------------------
def plan_energy_min(trainer, xy, z, dem, grid, c):
    """Greedy high-footprint set cover, tau-budgeted, ignoring QoS. Returns hovers."""
    N = len(xy)
    covered = np.zeros(N, bool)
    hovers = []
    while not covered.all():
        unc = np.where(~covered)[0]
        best = None
        for a in unc:
            rho = np.linalg.norm(xy - xy[a], axis=1)
            for H in grid:
                in_cone = (~covered) & (rho <= np.maximum(H - z, 0.0) * c["tan"])
                idx = np.where(in_cone)[0]
                if idx.size == 0:
                    continue
                d3d = np.sqrt(rho[idx] ** 2 + (H - z[idx]) ** 2)
                idx = budget_keep(trainer, d3d, dem[idx], idx)
                d3d = np.sqrt(rho[idx] ** 2 + (H - z[idx]) ** 2)
                e = hover_energy(trainer, d3d, dem[idx])
                key = (idx.size, -e)
                if best is None or key > best[0]:
                    best = (key, a, float(H), idx)
        _, a, H, idx = best
        hovers.append([a, H, idx])
        covered[idx] = True
    return hovers


# ----------------------------------------------------------------------
# Planner 3: STRONG two-stage (decoupled), every advantage
# ----------------------------------------------------------------------
def plan_two_stage(trainer, xy, z, dem, rmin, grid, c, H_min, H_max, h_safe, dmax):
    N = len(xy)
    hovers = plan_energy_min(trainer, xy, z, dem, grid, c)   # stage 1: high cover

    # find violators: members served below their floor at their hover altitude
    violators = []
    for h in hovers:
        a, H, idx = h
        rho = np.linalg.norm(xy[idx] - xy[a], axis=1)
        d3d = np.sqrt(rho ** 2 + (H - z[idx]) ** 2)
        bad = idx[(rmin[idx] > 0) & (rate_of(d3d, c) < rmin[idx])]
        violators.extend(bad.tolist())
    violators = np.array(sorted(set(violators)), dtype=int)

    # remove violators from their stage-1 hovers (no double service)
    for h in hovers:
        h[2] = h[2][~np.isin(h[2], violators)]
    hovers = [h for h in hovers if h[2].size > 0]

    # stage 2: cluster violators into shallow feasible low hovers (amortise dives)
    covered_v = np.zeros(N, bool)
    covered_v[~np.isin(np.arange(N), violators)] = True   # only violators to cover
    while not covered_v.all():
        unc = np.where(~covered_v)[0]
        best = None
        for a in unc:                      # anchor among remaining violators
            rho = np.linalg.norm(xy - xy[a], axis=1)
            # candidate members: still-uncovered violators in geometric reach
            cand = unc[rho[unc] <= dmax[unc] + 1e-9]
            H = feasible_H(xy, z, rmin, a, cand, c, H_min, H_max, h_safe, dmax)
            if H is None:
                # fall back to the anchor alone at its shallowest feasible H
                H = feasible_H(xy, z, rmin, a, np.array([a]), c, H_min, H_max, h_safe, dmax)
                cand = np.array([a])
            in_cone = rho[cand] <= np.maximum(H - z[cand], 0.0) * c["tan"]
            mem = cand[in_cone]
            d3d = np.sqrt(rho[mem] ** 2 + (H - z[mem]) ** 2)
            mem = mem[rate_of(d3d, c) >= rmin[mem]]      # keep only feasible members
            if mem.size == 0:
                mem = np.array([a])
            d3d = np.sqrt(np.linalg.norm(xy[mem] - xy[a], axis=1) ** 2 + (H - z[mem]) ** 2)
            mem = budget_keep(trainer, d3d, dem[mem], mem)
            e = hover_energy(trainer, np.sqrt(np.linalg.norm(xy[mem] - xy[a], axis=1) ** 2 + (H - z[mem]) ** 2), dem[mem])
            key = (mem.size, -e)
            if best is None or key > best[0]:
                best = (key, a, float(H), mem)
        _, a, H, mem = best
        hovers.append([a, H, mem])
        covered_v[mem] = True
    return hovers


# ----------------------------------------------------------------------
# Planner 4: COUPLED greedy (joint), one QoS-aware pass
# ----------------------------------------------------------------------
def plan_coupled(trainer, xy, z, dem, rmin, grid, c, H_min, H_max, h_safe, dmax):
    N = len(xy)
    covered = np.zeros(N, bool)
    hovers = []
    while not covered.all():
        unc = np.where(~covered)[0]
        best = None
        for a in unc:
            rho = np.linalg.norm(xy - xy[a], axis=1)
            if rmin[a] > 0:
                # anchor is critical -> hover is forced low; ABSORB every uncovered
                # node that is feasible at the shallowest altitude meeting a's floor.
                Hs = [feasible_H(xy, z, rmin, a, np.array([a]), c, H_min, H_max, h_safe, dmax)]
            else:
                Hs = list(grid)            # routine anchor: free to pick energy-min H
            for H in Hs:
                if H is None:
                    continue
                in_cone = (~covered) & (rho <= np.maximum(H - z, 0.0) * c["tan"])
                idx = np.where(in_cone)[0]
                if idx.size == 0:
                    continue
                d3d = np.sqrt(rho[idx] ** 2 + (H - z[idx]) ** 2)
                # drop members that would violate THEIR floor at this H
                ok = rate_of(d3d, c) >= rmin[idx]
                idx, d3d = idx[ok], d3d[ok]
                if idx.size == 0:
                    continue
                idx = budget_keep(trainer, d3d, dem[idx], idx)
                d3d = np.sqrt(rho[idx] ** 2 + (H - z[idx]) ** 2)
                e = hover_energy(trainer, d3d, dem[idx])
                key = (idx.size, -e)        # most nodes, then least hover energy
                if best is None or key > best[0]:
                    best = (key, a, float(H), idx)
        if best is None:                    # safety: serve anchor alone
            a = unc[0]
            H = feasible_H(xy, z, rmin, a, np.array([a]), c, H_min, H_max, h_safe, dmax) or H_min
            best = ((1, 0), a, float(H), np.array([a]))
        _, a, H, idx = best
        hovers.append([a, H, idx])
        covered[idx] = True
    return hovers


# ----------------------------------------------------------------------
# turn a hover list into a routed (anchors, altitudes, served) plan + score
# ----------------------------------------------------------------------
def score(trainer, node_features, hovers_per_b, R_min, nw, nr):
    B, N = node_features.shape[0], node_features.shape[1]
    T = max(len(h) for h in hovers_per_b)
    anchors = np.zeros((B, T), np.int64)
    altitudes = np.full((B, T), trainer.fixed_alt, np.float32)
    served = np.zeros((B, T, N), bool)
    for b, hovers in enumerate(hovers_per_b):
        # natural order; the canonical scorer applies the shared NN+2opt refinement
        for t, (a, H, idx) in enumerate(hovers):
            anchors[b, t] = a
            altitudes[b, t] = H
            served[b, t, idx] = True
    plan = _plan_from_steps(anchors, altitudes, served, B, T, N, trainer.device)
    return trainer._partition_and_evaluate(
        plan, node_features, R_min=R_min, use_service_altitude=False,
        node_weights=nw, node_rmin=nr, refine_routing=True)


def per_node_2d(trainer, node_features, nw, nr):
    """2D-AUTO: NN+2opt visiting every node individually at the fixed altitude."""
    nf = node_features.cpu().numpy()
    B, N = nf.shape[0], nf.shape[1]
    anchors = np.zeros((B, N), np.int64)
    served = np.zeros((B, N, N), bool)
    altitudes = np.full((B, N), trainer.fixed_alt, np.float32)
    # natural order; the canonical scorer applies the shared NN+2opt refinement
    for b in range(B):
        for node in range(N):
            anchors[b, node] = node
            served[b, node, node] = True
    plan = _plan_from_steps(anchors, altitudes, served, B, N, N, trainer.device)
    return trainer._partition_and_evaluate(plan, node_features, R_min=0.0,
                                           use_service_altitude=False,
                                           node_weights=nw, node_rmin=nr,
                                           refine_routing=True)


def summarise(name, m):
    e = m["energy"].mean() / 3600.0
    served = (1 - m["unserved_frac"].mean()) * 100
    hi = np.nanmean(m["high_qos_satisfaction"]) * 100
    md = np.nanmean(m["med_qos_satisfaction"]) * 100
    allq = m["priority_satisfaction"].mean() * 100
    uav = m["num_uavs"].mean()
    return e, served, hi, md, allq, uav, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "atom_3d", "configs", "params.yaml"))
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
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

    nf = f3.cpu().numpy()
    rmin_np = nr.cpu().numpy()

    em, ts, cp = [], [], []
    for b in range(args.instances):
        xy, z, dem = nf[b, :, :2], nf[b, :, 2], nf[b, :, 3]
        rm = rmin_np[b]
        dmax = d_max_of(rm, c)
        em.append(plan_energy_min(tr, xy, z, dem, grid, c))
        ts.append(plan_two_stage(tr, xy, z, dem, rm, grid, c, H_min, H_max, h_safe, dmax))
        cp.append(plan_coupled(tr, xy, z, dem, rm, grid, c, H_min, H_max, h_safe, dmax))

    rows = [
        summarise("2D-AUTO (per-node, 20 m)", per_node_2d(tr, f3, nw, nr)),
        summarise("3D energy-min (=current planner regime)", score(tr, f3, em, 0.0, nw, nr)),
        summarise("Strong two-stage (DECOUPLED)", score(tr, f3, ts, 0.0, nw, nr)),
        summarise("Coupled greedy (JOINT)", score(tr, f3, cp, 0.0, nw, nr)),
    ]

    print(f"\n=== N={args.N}, {args.instances} instances, priority on "
          f"(high {params['priority']['R_min']['high']/1e6:.0f}/med "
          f"{params['priority']['R_min']['medium']/1e6:.0f} Mbps) ===\n")
    print(f"  {'planner':40s} {'energy':>9} {'served':>7} {'hiQoS':>7} {'medQoS':>7} {'allQoS':>7} {'uavs':>6}")
    for e, served, hi, md, allq, uav, name in rows:
        print(f"  {name:40s} {e:7.2f}Wh {served:6.1f}% {hi:6.1f}% {md:6.1f}% {allq:6.1f}% {uav:6.1f}")

    e_ts = rows[2][0]; e_cp = rows[3][0]
    print(f"\n  --> coupled vs strong two-stage: {(1 - e_cp/e_ts)*100:+.1f}% energy "
          f"({'coupled cheaper' if e_cp < e_ts else 'NO gain'})")
    print("      (compare ONLY at equal ~100% hi/med QoS; energy-min row is the infeasible floor)")


if __name__ == "__main__":
    main()
