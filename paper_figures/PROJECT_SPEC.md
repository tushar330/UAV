# PROJECT

**Paper title**

ATOM-3D-VoI:
Criticality-Class-Aware 3D UAV Trajectory Planning
for Multi-UAV IoT Data Collection

---

## Goal

Generate every publication-quality figure used in the paper.

The figures must **NOT** fabricate experimental results.

They should visualize either

- the environment
- algorithms
- trajectories
- evaluation metrics

using **actual data whenever available**.

---

## Requirements

- Python 3.11
- Matplotlib
- NumPy
- SciPy
- **No seaborn.**
- IEEE publication style.
- 600 DPI.
- Transparent background only if useful.
- Each figure saved separately.
- **No subplot** unless explicitly requested.

---

## Folder structure

```
paper_figures/
    PROJECT_SPEC.md
    FIGURE_LIST.md
    DATA_SPEC.md
    CODE_RULES.md
    PAPER_STORY.md
    FILE_DEPENDENCIES.md
    common_style.py          # LOCKED: IEEE style + palette (COLORS) + save/panel
    synthetic_city.py        # LOCKED: builds/loads the synthetic environment
    common_plot.py           # shared, generic drawing helpers reused by figures
    figure01_environment.py
    figure02_blind3d.py
    figure03_atom3d_voi.py
    figure04_altitude_snr.py
    figure05_voi_accumulation.py
    figure06_energy_breakdown.py
    figure07_coverage_service.py
    figure08_priority_satisfaction.py
    figure09_training_curves.py
    figure10_method_comparison.py
    figure11_ablation.py
    figure12_pareto_energy_voi.py
    figure13_altitude_qos_regression.py
    synthetic_city.pkl       # generated cache (via get_city); do NOT reference by path
    results_data/            # experiment outputs, loaded from disk when present
    results/                 # rendered figures land here (created automatically)
```

Notes:
- `common_style.py` and `synthetic_city.py` are **locked infrastructure** —
  do not modify them unless explicitly requested.
- The city is always obtained via `get_city()` / `load_city()` from
  `synthetic_city.py`. No script references the `.pkl` path directly.
- The single output directory is `paper_figures/results/`. Do not use
  `output/`. `common_plot.py` creates `results/` automatically.

---

## Coding style

- Readable
- Modular
- Well documented
- No duplicate plotting code
- No magic numbers

---

## Important

- **Never invent experiment results.**
- Whenever experiment outputs exist, **load them from disk**
  (`results_data/*.pkl` / `*.npz` / `*.csv`).
- If they don't exist, create **clearly-labelled placeholder data**
  that can later be replaced. Every placeholder figure must stamp
  `PLACEHOLDER — synthetic data` in a corner so it is never mistaken
  for a real result.
- All numeric constants (rates, colors, altitudes, thresholds) live in
  `DATA_SPEC.md` and `common_style.py`, never hard-coded in a figure script.
