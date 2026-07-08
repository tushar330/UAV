# FIGURE LIST

Thirteen figures. Each is produced by one independent script and saved
separately to `results/`. "Input" lists the file(s) loaded from disk; if
that file is absent the script generates clearly-labelled placeholder data.

---

## Figure 1 — Environment

**Script**  figure01_environment.py

**Input**   city via get_city()

**Output**  results/figure01_environment.png

**Contains**
- roads
- buildings
- POIs
- IoT nodes (colored by criticality class)
- depot
- UAV start marker
- coverage cone (footprint at operating altitude)

Top-down 2D city map. This is the canonical scene reused conceptually by
all trajectory figures.

---

## Figure 2 — Blind 3D baseline trajectory

**Script**  figure02_blind3d.py

**Input**   results_data/traj_blind3d.pkl, city via get_city()

**Output**  results/figure02_blind3d.png

**Contains**
- 3D city (extruded buildings)
- baseline UAV path (criticality-agnostic, fixed altitude)
- nodes colored by class
- served vs unserved node markers

Shows the naive baseline: visits nodes without regard to criticality class.

---

## Figure 3 — ATOM-3D-VoI trajectory (proposed)

**Script**  figure03_atom3d_voi.py

**Input**   results_data/traj_atom3d.pkl, city via get_city()

**Output**  results/figure03_atom3d_voi.png

**Contains**
- 3D city
- proposed UAV path with learned altitude profile
- altitude dips over high-criticality nodes
- served-node order annotations

The headline qualitative result. Same scene as Fig. 2 for direct visual
comparison.

---

## Figure 4 — Altitude vs SNR / achievable rate

**Script**  figure04_altitude_snr.py

**Input**   results_data/link_budget.csv (optional)

**Output**  results/figure04_altitude_snr.png

**Contains**
- achievable rate (Mbps) vs UAV altitude for each criticality class
- horizontal QoS rate floor lines per class (High/Med/Low)
- shaded feasible altitude band per class

Explains *why* altitude is constraint-conditioned: the QoS floor sets a
maximum serviceable altitude per class.

---

## Figure 5 — VoI accumulation over the mission

**Script**  figure05_voi_accumulation.py

**Input**   results_data/voi_timeseries.pkl

**Output**  results/figure05_voi_accumulation.png

**Contains**
- cumulative Value-of-Information vs mission time
- one curve per method (Blind-3D, coupled-greedy, ATOM-3D-VoI)
- markers at each successful QoS-satisfying collection event

VoI = Σ w · D · 1[QoS met]. Higher-and-earlier is better.

---

## Figure 6 — Energy breakdown

**Script**  figure06_energy_breakdown.py

**Input**   results_data/energy.csv

**Output**  results/figure06_energy_breakdown.png

**Contains**
- per-method energy split: propulsion / hover / communication
- grouped bar chart (methods on x-axis)

Supports the ~30% energy-advantage claim of 3D over 2D operation.

---

## Figure 7 — Coverage / service map

**Script**  figure07_coverage_service.py

**Input**   results_data/service_map.pkl, city via get_city()

**Output**  results/figure07_coverage_service.png

**Contains**
- top-down map
- per-node served/unserved status
- per-node achieved rate as marker size/color
- QoS-violation nodes flagged

Which nodes actually got served at spec, class by class.

---

## Figure 8 — Priority satisfaction rates

**Script**  figure08_priority_satisfaction.py

**Input**   results_data/satisfaction.csv

**Output**  results/figure08_priority_satisfaction.png

**Contains**
- fraction of nodes served at QoS, grouped by class (High/Med/Low)
- one bar group per method

The core fairness/priority result: high-criticality nodes should be
served preferentially.

---

## Figure 9 — CMDP training curves

**Script**  figure09_training_curves.py

**Input**   results_data/training_log.csv

**Output**  results/figure09_training_curves.png

**Contains**
- reward vs training step
- constraint (QoS-violation / budget) vs training step, with limit line
- smoothed mean ± std shading over seeds

Convergence evidence for the learned policy.

---

## Figure 10 — Method comparison (headline metrics)

**Script**  figure10_method_comparison.py

**Input**   results_data/summary_metrics.csv

**Output**  results/figure10_method_comparison.png

**Contains**
- grouped bars: total VoI, energy, high-priority satisfaction
- methods: Blind-3D, coupled-greedy, ATOM-3D-VoI

Single-glance comparison across the main baselines.

---

## Figure 11 — Ablation study

**Script**  figure11_ablation.py

**Input**   results_data/ablation.csv

**Output**  results/figure11_ablation.png

**Contains**
- metric deltas as components are removed:
  learned altitude, criticality weighting, coupled planning
- bars relative to full model

Shows each component's contribution.

---

## Figure 12 — Pareto front: energy vs VoI

**Script**  figure12_pareto_energy_voi.py

**Input**   results_data/pareto.csv

**Output**  results/figure12_pareto_energy_voi.png

**Contains**
- scatter of (energy, VoI) operating points
- Pareto-optimal frontier line
- each method's point labeled

Trade-off view: ATOM-3D-VoI should dominate baselines.

---

## Figure 13 — Altitude–QoS regression

**Script**  figure13_altitude_qos_regression.py

**Input**   results_data/altitude_qos.csv

**Output**  results/figure13_altitude_qos_regression.png

**Contains**
- scatter of learned altitude vs (node height + safe clearance)
- fitted line, R² annotation, per-class coloring

Validates the altitude-constraint hypothesis (H ≈ z + h_safe when QoS
active; R² ≈ 0.97 for high-priority nodes).
