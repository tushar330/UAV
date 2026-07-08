# DATA SPEC

Single source of truth for every constant used across figures.

> **The Python implementation is the source of truth.** All values below
> are transcribed from `common_style.py` and `synthetic_city.py`. Figure
> scripts must import colors from `common_style.COLORS` and city data via
> `synthetic_city.get_city()` — never re-declare these constants.

---

## Criticality classes

Colors are the exact `COLORS` values in `common_style.py`. Counts, rates
and demand ranges are from `synthetic_city.py`.

| Class  | Color  | `COLORS` key | Hex       | Count (of 500) | Share | QoS rate floor | Demand    | Weight w |
|--------|--------|--------------|-----------|----------------|-------|----------------|-----------|----------|
| High   | Red    | `high`       | `#E53935` | 50             | 10%   | 38 Mbps        | 1.5 – 2.0 | 3        |
| Medium | Orange | `medium`     | `#FB8C00` | 100            | 20%   | 25 Mbps        | 0.8 – 1.5 | 2        |
| Low    | Green  | `low`        | `#4CAF50` | 350            | 70%   |  8 Mbps        | 0.2 – 0.8 | 1        |

- Required rates in code are stored in bits/s (`38e6`, `25e6`, `8e6`).
- QoS met for a node iff achieved rate ≥ its class rate floor.
- VoI contribution of a served node = `w · D · 1[QoS met]`, where `D` is
  data volume collected. (Weight `w` is a paper-side convention; the code
  stores per-node `demand`, not `w`.)

---

## Environment (synthetic city)

All values from `synthetic_city.py` (`SEED = 42`).

| Quantity            | Value / range        | Notes                             |
|---------------------|----------------------|-----------------------------------|
| Area                | 1000 m × 1000 m      | flat ground plane                 |
| Districts           | 7                    | Downtown, Commercial, Residential_A/B, Industrial, Hospital, PowerStation |
| Building height     | 5 – 40 m             | per-district range                |
| Buildings           | 126                  | 18 per district                   |
| Roads               | 14                   | main (grid) + secondary           |
| POIs                | 4                    | Hospital, Power, Industrial, Business; scene decoration, not serviced |
| Number of IoT nodes | 500                  | 50 High / 100 Med / 350 Low       |
| Depot               | 1, at (50, 50, 0)    | UAV start/return                  |
| Demo UAV            | (500, 500, 100), coverage radius 170 m | for illustrative figures |

Node placement, building footprints and the RNG seed are frozen inside the
pickle written by `synthetic_city.py`. Regenerate only via that file.
**Do not reference the pickle path directly** — obtain the city with
`get_city()` / `load_city()` (see "City loading" below).

---

## UAV / physics

| Quantity              | Value          | Notes                          |
|-----------------------|----------------|--------------------------------|
| Safe clearance h_safe | 10 m           | min gap above node/building    |
| Altitude range        | 20 – 120 m     | policy action bound            |
| Path-loss exponent α  | 3              | print β/d² only in text, use α=3 in code |
| Hover power           | model-defined  | see energy model               |
| Propulsion power      | speed-dependent| see energy model               |
| WPT                   | **OFF**        | decoupled; not in current runs |

> Note: WPT is currently disabled in the pipeline. Do not draw energy-
> harvesting arrows in trajectory figures unless WPT results exist on disk.

---

## Methods (labels + colors)

Method colors come from `common_style.COLORS`. Use these exact keys/colors
for consistency across figures.

| `COLORS` key | Label             | Hex       | Suggested linestyle |
|--------------|-------------------|-----------|---------------------|
| `blind`      | Blind-3D          | `#2E7D32` | dashed              |
| `ours`       | ATOM-3D-VoI (ours)| `#1565C0` | solid               |
| `baseline2d` | 2D baseline       | `#616161` | dashdot             |

> `Coupled-Greedy` (referenced in PAPER_STORY / FIGURE_LIST) does not yet
> have a dedicated color key in `common_style.py`. Add one there **only
> when explicitly requested**; until then reuse `baseline2d` for it.

## Scene element colors (from `common_style.COLORS`)

| `COLORS` key | Meaning              | Hex       |
|--------------|----------------------|-----------|
| `uav`        | UAV marker/path      | `#1976D2` |
| `building`   | building footprint   | `#9E9E9E` |
| `ground`     | ground plane         | `#ECEFF1` |
| `cone`       | coverage cone/circle | `#90CAF9` |

---

## Metrics glossary

| Symbol / name        | Definition                                        |
|----------------------|---------------------------------------------------|
| VoI                  | Σ over served nodes of `w · D · 1[QoS met]`       |
| Energy               | total Joules: propulsion + hover + communication  |
| High-prio sat.       | fraction of High-class nodes served at QoS        |
| QoS violation rate   | fraction of collection attempts below rate floor  |
| R² (altitude gate)   | fit of learned H vs (node height + h_safe)        |

---

## City loading

Every figure obtains the city through `synthetic_city.py` — never by
opening a pickle path:

```python
from synthetic_city import get_city   # builds if missing, else loads + caches
city = get_city()
```

`load_city()` is also exported for the explicit-load case. No figure script
may hard-code a `.pkl` filename or absolute path.

---

## Output directory

All figures save to `paper_figures/results/` (created automatically by
`common_plot.py`). Do not use `output/`.

---

## File formats consumed by figures

Result files (trajectories, metrics, logs) are loaded from disk when they
exist. Paths are relative to `paper_figures/`.

| File                          | Format | Produced by         |
|-------------------------------|--------|---------------------|
| synthetic_city.pkl            | pickle | synthetic_city.py (via `get_city`) |
| results_data/traj_*.pkl       | pickle | training/eval run   |
| results_data/voi_timeseries.pkl| pickle| eval run            |
| results_data/*.csv            | CSV    | eval/aggregation    |
| results_data/training_log.csv | CSV    | training run        |

If an input is missing, the figure script fabricates **labelled placeholder**
data (stamped `PLACEHOLDER — synthetic data`) with the same schema so real
outputs can be dropped in later without edits.
