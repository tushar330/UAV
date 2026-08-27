# FIGURE LIST

Reconciled with the repository on **2026-08-17**. The previous version of this
file described a figure set that no longer existed (it named scripts such as
`figure05_voi_accumulation.py` that were never written) — writing from it would
have produced a manuscript that contradicted the figures.

Each figure is produced by one independent script and saved to `results/`.

## Status key

- **IN** — in the manuscript, backed by real data on disk.
- **CUT** — not in the manuscript; renders a `PLACEHOLDER — synthetic data`
  stamp and must never be shown as a result.

---

## In the manuscript (9 figures)

| # | Script | Shows | Input |
|---|---|---|---|
| 1 | `figure01_environment.py` | City, buildings, IoT nodes by class, depot | `get_city()` |
| 2 | `figure02_method_overview.py` | Method schematic | — |
| 3 | `figure03_altitude_comparison.py` | Altitude vs mission progress, per method | `altitude_traces.pkl` |
| 5 | `figure05_qos_comparison.py` | **Per-class QoS satisfaction — headline result** | `qos_satisfaction.npz`, `overall_metrics.npz` |
| 6 | `figure06_energy_comparison.py` | Energy split (flight/hover/comm/total) | `energy_breakdown.npz` |
| 7 | `figure07_trajectories_3d.py` | 3D trajectories over the city | `trajectories.pkl` |
| 9 | `figure09_rate_cdf.py` | CDF of high-priority achieved rates | `high_rate_samples.npz` |
| 10 | `figure10_altitude_distribution.py` | Hover-altitude density per method | `hover_altitudes.npz` |
| 13 | `figure13_overall_table.py` | **Summary table, CIs and paired tests** | `overall_metrics.npz` |

Figures 5, 6 and 13 carry the contribution and must agree numerically — they
read the same export.

---

## Cut from the manuscript

| # | Script | Why cut |
|---|---|---|
| 4 | `figure04_coverage_comparison.py` | Placeholder only |
| 4 | `figure04_training_dynamics.py` | Needs a trained policy; the CMDP run is invalid |
| 8 | `figure08_pareto_tradeoff.py` | Its sweep data was serve-all regime and was deleted; regenerating costs ~2 h |
| 11 | `figure11_dual_convergence.py` | Needs a trained policy (dual trajectories) |
| 12 | `figure12_ablation.py` | Needs *learned* ablation variants |

Figures 4, 11 and 12 exist for a future learned-policy paper. They are not
deleted, only excluded — see `PAPER_STORY.md`.

---

## The method slots every figure shares

| key | label | role |
|---|---|---|
| `2d_auto` | 2D-AUTO | the base paper being extended — **primary baseline** |
| `two_stage` | Two-Stage (decoupled) | strongest 3D baseline — standard practice |
| `coupled_greedy` | Coupled-Greedy (ablation) | ours without the local search — **an ablation, not a baseline** |
| `atom3d` | ATOM-3D-VoI (Ours) | the proposed method |

`Blind-3D` (key `3d_gnn`) was **removed** on 2026-08-17: a degenerate control
scoring ~0–1% on every class, whose key falsely implied a learned GNN. Do not
reintroduce it.

Figures obtain labels from `results_data/labels.json` via `apply_labels()`, and
drop any slot missing from the data via `present_methods()` — so a figure can
never invent a curve for a method that did not run.
