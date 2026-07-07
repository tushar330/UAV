"""Shared routing refinement (NN construction + 2-opt improvement).

Single source of truth for the deterministic tour refinement applied IDENTICALLY
to every evaluated solution — the learned policy and all deterministic baselines —
so that energy comparisons isolate clustering/altitude/assignment quality rather
than routing-heuristic luck (PROBLEM_FORMULATION Phase 1). Used by the canonical
scorer (``TENMATrainer._partition_and_evaluate`` under ``refine_routing=True``) and
by the baseline planners for their internal local search.
"""

import numpy as np


def nn_2opt_tour(pts: np.ndarray, depot: np.ndarray, iters: int = 60) -> list:
    """Nearest-neighbour construction + 2-opt improvement over node indices.

    Returns a list of indices into ``pts`` giving a depot-rooted visiting order.
    Deterministic in ``pts``/``depot`` (NN starts from the point nearest the depot),
    so re-running it on an already-ordered set is idempotent.
    """
    n = len(pts)
    if n == 0:
        return []
    # nearest-neighbour starting from the node closest to the depot
    unvisited = set(range(n))
    cur = int(np.argmin(np.linalg.norm(pts - depot, axis=1)))
    tour = [cur]; unvisited.discard(cur)
    while unvisited:
        d = [(np.linalg.norm(pts[cur] - pts[j]), j) for j in unvisited]
        _, nxt = min(d)
        tour.append(nxt); unvisited.discard(nxt); cur = nxt

    def tour_len(t):
        p = np.vstack([depot, pts[t], depot])
        return np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))

    best = tour_len(tour)
    improved = True
    it = 0
    while improved and it < iters:
        improved = False; it += 1
        for i in range(len(tour) - 1):
            for k in range(i + 1, len(tour)):
                new = tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]
                nl = tour_len(new)
                if nl + 1e-6 < best:
                    tour, best = new, nl; improved = True
    return tour
