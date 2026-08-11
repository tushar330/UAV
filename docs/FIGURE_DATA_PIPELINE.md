# Figure-data pipeline — how `atom_3d` results reach `paper_figures`

The two halves of this repo describe different environments:

* `atom_3d` generates **uniform-random** scenarios (`IoTEnvironment3D.generate_batch`).
* `paper_figures` draws a **frozen synthetic city** (`synthetic_city.py`, seed 42):
  500 nodes on rooftops across 7 districts, depot at (50, 50, 0).

Results produced on one cannot honestly be plotted as results on the other. This
pipeline bridges them in one direction: it evaluates the repo's planners **on the
paper's city** and writes their metrics in the schema the figure scripts already
read.

> **Rule:** no figure value is ever hand-written. Everything below is produced by
> `TENMATrainer._partition_and_evaluate` — the same energy accounting used by the
> learned policy and every baseline.

---

## Components

| File | Role |
|---|---|
| `atom_3d/env/city_adapter.py` | Loads the frozen city into ATOM-3D tensors; can re-draw it at other RNG seeds for confidence intervals. |
| `atom_3d/training/tenma_trainer.py` | `_partition_and_evaluate(..., collect_trace=True)` returns per-hover records, per-node achieved rates, and the energy split. Default **off** — training is unaffected. |
| `atom_3d/experiments/export_figure_data.py` | Runs the planners and writes `paper_figures/results_data/*`. |
| `atom_3d/experiments/run_train.py` | `--export-figure-data` publishes the training curves and dual trajectories. |
| `paper_figures/common_plot.py` | `apply_labels()` rewrites figure legends to name whatever actually produced the numbers. |

---

## Running it

```bash
# everything that needs no training (canonical seed-42 city)
python -m atom_3d.experiments.export_figure_data

# iterate quickly: plain coupled greedy instead of the ~23 min local search
python -m atom_3d.experiments.export_figure_data --fast

# real confidence intervals (multiplies runtime by the number of seeds)
python -m atom_3d.experiments.export_figure_data --city-seeds 42,43,44,45,46

# training-dependent figures (4 and 11)
python -m atom_3d.experiments.run_train --mode 3d --encoder attention --cmdp \
    --export-figure-data
```

Figures then render real data automatically — each script prefers its
`REAL_DATA_PATH` over its built-in placeholder and drops the
`PLACEHOLDER — synthetic data` stamp on its own. Run them from `paper_figures/`
(`synthetic_city.CITY_FILE` is a relative path).

---

## Method slots

The figures key on three slots. Each resolves to the strongest **real** source
available, and `results_data/labels.json` records what that was so a legend never
claims a trained policy produced a planner's numbers.

| Slot | Today (no checkpoints) | With a checkpoint |
|---|---|---|
| `2d_auto` | per-node NN+2-opt tour at a fixed altitude | unchanged (not a learned method) |
| `3d_gnn` | QoS-blind greedy footprint cover | `3d_gnn.pt` |
| `atom3d` | strong coupled planner (local search) | `3d_attention_priority_cmdp.pt` |

**Wiring the learned policy into a slot is not implemented yet** — the exporter
detects the checkpoint and prints a NOTE. Until then the labels say
"deterministic", which is the point.

---

## Modelling decisions worth knowing

**2D cruise altitude is derived, not chosen.** It is the lowest altitude clearing
the tallest structure with the spec's `h_safe`: `37.4 + 10 = 47.4 m`. Flying lower
would leave rooftop nodes above the UAV with a zero-radius coverage cone.

**Class floors come from the city, not the config.** Every `IoTNode` carries its
own `required_rate` (38 / 25 / 8 Mbps). `params.yaml` had drifted to 38 / 28 / 0,
so the exporter overrides `priority.R_min` **in memory** to match the city; the
YAML on disk is untouched, so no training run is affected. Class weights likewise
follow `DATA_SPEC.md` (3/2/1) rather than the config's 5/2/1.

**There is a hard rate ceiling.** Clearance forces `H ≥ z + h_safe`, so the
shortest possible link is 10 m, which yields **44.5 Mbps** at `α = 3`. Any QoS
floor above that is unsatisfiable by *any* planner. The Pareto sweep computes this
ceiling and drops infeasible sweep points rather than reporting a solver failure
as a result. It also means the high-priority 38 Mbps floor is only met from
`H ≈ z + h_safe` — the altitude law the trajectory figures show.

**Confidence intervals need multiple seeds.** The city is deterministic and so are
the planners, so a single realization has no spread; `ci95` is written as `0.0`
rather than invented. Pass several `--city-seeds` for a real interval.

---

## What is still placeholder, and why

| Figure | Status |
|---|---|
| 4 — training dynamics | needs a real `run_train --cmdp --export-figure-data` |
| 11 — dual convergence | same run |
| 12 — ablation | its four bars are named after *learned* components (`learned_alt`, `cmdp_uniform`); deterministic planners cannot fill them without mislabelling |

These keep their `PLACEHOLDER` stamp on purpose: it marks exactly which claims are
still outstanding.
