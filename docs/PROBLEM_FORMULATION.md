# ATOM-3D / 3D-AUTO — Locked Problem Formulation

> Canonical spec. If code and this doc disagree, fix one of them deliberately — do not drift.
> Derived from base papers A (AUTO, WPT-IoT, 2D) and B (UAV-ISAC, 3D). Decisions ratified 2026-06-18.

## 1. One-line statement
Extend the 2D **AUTO** framework to **3D**: UAV altitude becomes a decision variable that
(i) widens the communication/charging footprint (Paper B's cone), (ii) weakens the link, and
(iii) costs a **new altitude-change propulsion energy**. Goal: collect **all** IoT data at
**minimum total UAV energy**, beating the fixed-altitude 2D baseline.

## 2. Novel contributions (what is actually new)

> **Headline (revised 2026-06-18 after the altitude-freeze ablation).** Simply making
> altitude a learned action does *not* help: with weak link penalties and cheap hover, the
> energy-optimal policy is "fly as high as allowed and cover everything," so the altitude never
> varies. Our real contribution is a mechanism that gives altitude a **genuine, demand-driven
> interior optimum**, so the altitude profile varies along the trajectory *for energy reasons*.

1. **Energy-optimal *service altitude* allocation (dive-to-serve) — the headline novelty.** For each
   service action the UAV chooses a hover/dive altitude `H_s` that balances three competing costs:
   (i) **coverage breadth** — higher `H_s` ⇒ wider footprint ⇒ more nodes per stop ⇒ less flight;
   (ii) **link quality** — lower `H_s` ⇒ stronger channel ⇒ faster WPT charging & data collection ⇒
   **less hover energy** (and hover is the *expensive* state, see §5); (iii) **altitude-change
   propulsion** — diving deeper costs more climb energy to return to cruise. Because WPT charge time
   grows ∝ `(H_s−z)²` while climb cost grows linearly in dive depth, `E_serve(H_s)` has a **provable
   interior minimum** `H_s*` that **depends on the served cluster's data/energy demand** (§5a). This
   is the altitude analog of Paper A's KKT *time* allocation — a new closed-form sub-problem neither
   base paper has. Heterogeneous demand ⇒ `H_s*` differs per service ⇒ the altitude genuinely varies.
2. **Realistic rotary-wing propulsion (speed + altitude) making the optimum real.** We replace the
   constant-`P^F` assumption with the standard rotary-wing **power–speed curve** (hover = the costly
   peak; energy-optimal cruise speed per leg) plus the **altitude-change (climb/descent) energy**.
   The physics is standard; using it is what makes hover expensive and thus activates novelty #1.
3. **Learned 3D coverage-vs-dive policy (multi-UAV).** The graph-attention/TENMA policy decides, per
   node, whether to fold it into a shallow shared footprint or serve it with a deeper efficient dive —
   a genuinely 3D, energy-driven grouping decision. Multi-UAV split + WPT retained from Paper A;
   3D mobility/coverage cone from Paper B.
4. **Energy-vs-QoS (R_min) Pareto frontier** as a secondary result (the QoS floor is one regime that
   also forces lower flight).

*What we are NOT claiming as novel:* the rotary-wing power model, the WPT/EH model, and the coverage
cone are all standard. The contribution is the **service-altitude allocation formulation, its
interior-optimum characterization, and the learned multi-UAV policy over it.*

## 3. Base-paper anchors
- Paper A energy model (eqs. 9–12): `E_j = E^F_j + E^C_j + E^T_j`, minimize `Σ_j E_j` (eq. 15).
  - `E^F_j = P^F · T^F_j`, `T^F_j = Σ_t ‖r_j[t]−r_j[t−1]‖ / v`  (HORIZONTAL only, constant v).
  - `E^T_j = (P^T+P^H)·T^E_j` (WPT+hover), `E^C_j = (P^C+P^H)·T^C_j` (collect+hover).
  - `T^E*,T^C*` via KKT (eqs. 17–18), Lambert-W. Linear EH model `E^R=η_L P^R T^E` (eq. 6).
- Paper B coverage cone (eq. 23e): `(x_u−x_k)²+(y_u−y_k)² ≤ (H_u·tanθ)²`.
- Paper B mobility (eqs. 23l–23r): position updates, speed/accel bounds, altitude bounds, cyclic.

## 4. Assumptions (locked)
- **A1** LoS channel `|g_{ij}|² = β/d_{ij}²`, `d` = full 3D distance.
- **A2** WPT retained: linear EH (Paper A eq. 6); KKT time allocation (eqs. 17–18) with 3D `d`.
- **A3** Coverage cone (Paper B 23e): node `i` served at hover `(x,y,H)` iff
  `ρ² ≤ ((H−z_i)·tanθ)²`, θ = half-beamwidth = **60°**.
- **A4** Serve-all: every node collected exactly once at a single hover (hard constraint).
- **A5** Rotary-wing energy: horizontal flight uses the **power–speed curve** `P(V)` at the
  energy-optimal cruise speed (§5) + vertical climb/descent term. Hover power `P^H = P_0 + P_i`
  (the *peak* of the curve) — hover is the most expensive state.
- **A6** Quasi-static hover; UAVs orthogonal (no inter-UAV interference); single depot; cyclic.
- **A7** Mobility bounds from Paper B (23l–23r) — `v_xy,v_z,a_xy,a_z,H_min,H_max`.
- **A8** Static, known nodes (`x_i,y_i,z_i,D_i`); number of UAVs `m` auto-set by decoder segmentation.
- **A9** ~~Constant `P^F`~~ **(superseded)** — replaced by the Zeng–Zhang rotary-wing `P(V)` model so
  that hover is correctly expensive and velocity is energy-meaningful (enables novelty §5a).

## 5. Energy model (rotary-wing; the novel terms in bold)
Per UAV `j`:
```
E_j      = E^horiz_j + E^vert_j + E^T_j + E^C_j
E^horiz_j = Σ_legs  P(V*)/V* · ‖Δl‖        V* = argmin_V P(V)/V  (energy-optimal cruise, NOVEL physics)
E^vert_j  = Σ_legs [ (m·g/η)·max(ΔH,0) + c_d·max(−ΔH,0) ]    ← NOVEL (gravitational climb + small descent)
E^T_j     = (P^T + P^H) · Σ T^E_i        (WPT + hover, KKT times, 3D d;  P^H = P_0+P_i = hover peak)
E^C_j     = (P^C + P^H) · Σ T^C_i        (collect + hover, KKT times, 3D d)
```
Rotary-wing power–speed curve (Zeng & Zhang 2019):
```
P(V) = P_0(1 + 3V²/U_tip²) + P_i(√(1 + V⁴/(4v_0⁴)) − V²/(2v_0²))^½ + ½ d_0 ρ s A V³
```
`P(0) = P_0 + P_i` is hover power (the peak). For a leg of length L the energy is `L·min_V P(V)/V`
(max-range speed) — so flying is *cheaper per metre* than hovering, the realistic regime.

### 5a. NOVEL — energy-optimal service altitude `H_s*` (the dive-to-serve allocation)
When a UAV serves a node/cluster `S` (demand `D_S`, required harvested energy `E_S`) from hover
altitude `H_s` directly above ground elevation `z_S`, the **per-service energy** is:
```
E_serve(H_s) = P^H·[ T^E(H_s) + T^C(H_s) ]  +  (c_climb + c_d)·(H_cruise − H_s)
                └ hover/charge: grows ∝ (H_s − z_S)² ┘   └ dive-and-return climb: grows with depth ┘
   s.t.  footprint covers S:   H_s ≥ z_S + r_S / tanθ        (r_S = cluster radius)
         clearance / bounds:   max(H_min, z_S + h_safe) ≤ H_s ≤ H_max
```
because WPT received power ∝ `1/(H_s−z_S)²` ⇒ `T^E ∝ (H_s−z_S)²` (with `c_climb = m·g/η`). Ignoring the
slowly-varying `T^C` term, `dE_serve/dH_s = 0` gives a closed-form interior optimum:
```
H_s*  ≈  z_S  +  (c_climb + c_d) · η_L·β·P^T / ( 2·P^H·E_S )         (clamped to the constraints above)
```
**Key property:** `H_s*` *decreases* (the UAV dives deeper) as the served demand `E_S` grows, and the
coverage constraint pushes it *up* for wide clusters. Heterogeneous `D_i` and local density ⇒ a
**different `H_s*` at every service** ⇒ the altitude profile varies along the trajectory as a direct
consequence of energy minimization. This is the altitude analog of Paper A's KKT time allocation:
the outer policy picks routing + clustering; the inner sub-problem solves `H_s*` (and `T^E,T^C`)
in closed form.
`m` = UAV mass, `g=9.81`, `η` = motor efficiency, `c_d` = descent dissipation coeff (c_d ≪ m·g/η).

## 6. Optimization problem
```
(P0-3D)  min_{A, m, R, T^E, T^C}   Σ_{j=1}^m E_j
   s.t.  coverage cone (A3) for every served node
         serve-all:  Σ_j Σ_t a_{ij}[t] = 1   ∀i
         clearance:  H_j[t] ≥ z_i + h_safe
         QoS:        R_{ij} ≥ R_min   for every served node      (Pareto knob)
         storage:    Σ a_{ij} D_i ≤ C_max          (Paper A 14)
         energy:     E_j ≤ E_max                   (Paper A 13)
         mobility:   Paper B (23l–23r)
         cyclic:     start = end = depot
```
Decomposition (Paper A, extended): **inner** = KKT time allocation `T^E,T^C` **and the new
service-altitude allocation `H_s*`** (§5a), both closed-form; **outer** = 3D trajectory + clustering +
association + energy-optimal cruise speed by ATOM-3D / TENMA.

## 7. Pareto frontier — energy vs communication quality
Serve-all is always enforced. Sweep the per-node QoS floor **`R_min`**; each value yields one
`(total energy, link quality)` point. Higher `R_min` ⇒ lower altitudes / more hovers ⇒ more energy.
Headline curve: *"energy price of communication quality in a 3D WPT-IoT system."*

## 8. RL / MDP mapping (ATOM-3D)
- **State** (per decode step, active UAV): `[h_sa, h_prev, C_t, E_t, H_t]` + served-mask.
- **Action** (both sampled, both enter REINFORCE log-prob):
  1. next **anchor** node (attention pointer over unvisited nodes),
  2. **altitude** `H ∈ [H_min,H_max]` via Gaussian head (squashed). *Today the altitude head gets
     NO reward gradient — this fixes it.*
- **Transition:** hover at anchor `(x,y,H)`; serve all unvisited nodes with `ρ≤(H−z)tanθ`,
  `R≥R_min`, fitting `τ_max`; update `C_t,E_t,H_t`; advance UAV when capacity/battery would breach.
- **Reward (terminal):** `R = −(Σ_j E_j)/E_scale − μ·(unserved fraction)`  (serve-all → minimize energy).
  Critic baseline as in Paper A TENMA.

## 9. New / changed config (see configs/params.yaml)
- `coverage: {beamwidth_deg: 60, h_safe: 10, tau_max: 30}`
- `qos: {R_min: 0.0, R_min_sweep: [...]}`
- `uav: {mass_kg, motor_efficiency, descent_coeff_cd}` for `E^vert`
- **`propulsion: {P0, Pi, U_tip, v0, d0, rho, s, A}`** — rotary-wing power–speed curve (§5).
  `P^H` is derived as `P0+Pi` (hover peak), superseding the standalone `P_hover`.
- **WPT demand** `E_S` (required harvested energy per node) for the `H_s*` allocation (§5a).
- `C_max` lowered so capacity/battery force multi-UAV segmentation.

## 10. Implementation queue
**Phase 1 — base 3D pipeline (DONE).**
1. `params.yaml` — §9 params, lower `C_max`.  ✅
2. `channel_model` — `covered/serves(ρ,H,z)` + 3D rate.  ✅
3. `uav_physics` — `E^vert` (climb + descent).  ✅
4. `trajectory_decoder` (3D) — stochastic altitude action + `alt_log_probs`; footprint mask.  ✅
5. `tenma_trainer._partition_and_evaluate` — footprint serve-all + `τ_max` + `R_min` + segmentation;
   reward = −energy; `alt_log_probs` in the gradient.  ✅
6. `reward_function` — −energy (3D) / energy-min (2D baseline).  ✅
7. `run_train.py` / `run_eval.py` — normalize, `--mode/--encoder`, R_min sweep.  ✅

> **Finding (ablation):** with constant `P^F` + cheap hover, altitude collapses to "fly highest";
> the dive-to-serve novelty (§2.1, §5a) is needed to make altitude vary. Phase 2 implements it.

**Phase 2 — service-altitude novelty (IN PROGRESS).**
8. `uav_physics` — Zeng–Zhang `P(V)` power–speed curve + energy-optimal cruise `V*`; `P^H = P0+Pi`.
9. `service_altitude` — closed-form `H_s*` allocation (§5a) under coverage + clearance constraints.
10. `tenma_trainer` — use `P(V*)` legs + `H_s*` per service in `_partition_and_evaluate`;
    re-run the altitude-freeze ablation to confirm the profile now varies and beats any fixed `H`.
```
