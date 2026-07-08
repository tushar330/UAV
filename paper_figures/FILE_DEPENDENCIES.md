# FILE DEPENDENCIES

Tells you exactly which files depend on which, so functionality is reused
instead of recreated.

---

## Dependency graph

```
common_style.py        # IEEE rcParams, palette, 600-DPI save helper
        ↓
synthetic_city.py      # builds/loads data/synthetic_city.pkl
        ↓
common_plot.py         # shared plotting helpers (city map, 3D city, legends)
        ↓
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
```

### What each layer owns

| File               | Owns                                                        |
|--------------------|-------------------------------------------------------------|
| `common_style.py`  | rcParams, class/method colors, fonts, `save_figure()`       |
| `synthetic_city.py`| environment generation + `load_city()`                      |
| `common_plot.py`   | reusable draw helpers + `save_figure` (→ `results/`): `draw_ground`, `draw_roads`, `draw_buildings`, `draw_pois`, `draw_nodes`, `draw_depot`, `draw_demo_uav`, `draw_coverage_cone`, `draw_legend` |
| `figureNN_*.py`    | figure-specific composition only — no reusable primitives   |

### Direction of dependency (never reversed)

- Figures depend on `common_plot.py` → `synthetic_city.py` → `common_style.py`.
- `common_plot.py` may depend on `common_style.py` and `synthetic_city.py`.
- **No figure imports another figure.**
- `common_style.py` and `synthetic_city.py` depend on nothing in this repo.

---

## Rules

- **Never duplicate plotting code.**
- **Reuse helper functions.**
- **Do not copy code between figures.**
- Only add a helper function to `common_plot.py` when **at least two
  figures** need it. A one-off stays local to its figure script.
- If two figures start sharing copy-pasted code, that is the signal to
  lift it into `common_plot.py` — refactor, don't re-copy.
- Changing a `common_plot.py` helper must not silently alter a figure's
  meaning; if two callers need different behavior, parameterize the helper.
