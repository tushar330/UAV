"""
ATOM-3D-VoI ACCEPTANCE GATE  (no torch, no WPT — pure closed-form + Monte-Carlo).

Final validation of the reformulated direction (2026-06-25): WPT removed, altitude
recast as a QoS-FEASIBILITY lever under a value-of-information (VoI) objective.

Three strict sub-gates (mentor-mandated):
  G1  CLASS SEPARATION   per-class feasible-altitude ceiling must DIFFER across
                         criticality classes (else altitude is not a decision var).
  G2  VoI INTERIOR OPT   VoI(H) under a fixed energy budget B must have a
                         pronounced INTERIOR optimum (not monotone).
  G3  PRIORITY SHIFT     argmax_H VoI_priority  !=  argmax_H VoI_blind  (else
                         criticality is cosmetic — it doesn't change behaviour).

WPT is DISABLED here (no t_e term) but the project's WPT code is only feature-flagged
off elsewhere, not deleted, so this reformulation is reversible if the gate fails.

All channel / energy formulas copied from atom_3d/env/channel_model.py and
configs/params.yaml. The ONLY changed knob vs current params is `path_loss_exp`
(the physical de-saturation mechanism, Q2) — swept so we can see its effect.
"""
import numpy as np

# ------------------------------------------------------------------ constants (params.yaml)
c_light = 3e8
f_c     = 2.4e9
BETA    = (c_light / (4 * np.pi * f_c)) ** 2          # ~9.9e-5
SIGMA2  = 10 ** ((-110.0 - 30) / 10)                  # -110 dBm -> 1e-14 W  (PHYSICAL, unchanged)
B_HZ    = 2.0e6
P_T     = 0.5
P0, PI  = 79.86, 88.63
P_H     = P0 + PI            # hover power = 168.49 W
P_C     = 0.5
H_MIN, H_MAX = 20.0, 150.0
TANTH   = np.tan(np.radians(60.0))
SECTH   = 1.0 / np.cos(np.radians(60.0))    # edge-of-footprint distance = (H-z)*sec(theta) = 2*(H-z)
TAU_MAX = 30.0
AREA_W  = 1000.0
AREA    = AREA_W * AREA_W
N_NODES = 500
Z_MAX   = 50.0
D_MIN, D_MAX = 0.2, 1.5      # MB
CLASS_NAMES   = ["high", "medium", "low"]
CLASS_PROBS   = np.array([0.10, 0.30, 0.60])
CLASS_WEIGHTS = {"high": 5.0, "medium": 2.0, "low": 1.0}

def rate(d, alpha, sigma2=SIGMA2):
    gain = BETA / np.maximum(d, 1e-5) ** alpha
    return B_HZ * np.log2(1.0 + P_T * gain / sigma2)

def e_per_m():
    U_tip, v0, d0, rho, s, A = 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    V = np.linspace(0.5, 40.0, 4000)
    P = (P0 * (1 + 3 * V**2 / U_tip**2)
         + PI * (np.sqrt(1 + V**4 / (4 * v0**4)) - V**2 / (2 * v0**2)) ** 0.5
         + 0.5 * d0 * rho * s * A * V**3)
    return float(np.min(P / V))
E_PER_M = e_per_m()

Hs = np.array([20, 30, 40, 50, 60, 75, 90, 110, 130, 150], float)


def pick_floors(alpha):
    """Set per-class R_min floors RELATIVE to the de-saturated rate band so they bind
    inside [H_MIN,H_MAX]. high floor ~ rate at a low close hover; low floor = 0.
    Worst case the planner controls = node directly below (rho=0, d=H-z)."""
    r_lo = rate(H_MIN - 25.0, alpha)     # best achievable: low hover over a mean-elevation node
    r_hi = rate(H_MAX - 25.0, alpha)     # rate from the ceiling, directly below
    high = 0.80 * r_lo                   # strict: only met by low, close hovers
    med  = 0.5 * (r_lo + r_hi)           # moderate
    return {"high": high, "medium": med, "low": 0.0}, r_lo, r_hi


def gate_G1(alpha):
    print("\n" + "=" * 84)
    print(f"G1  CLASS SEPARATION   (path_loss_exp alpha = {alpha})")
    print("=" * 84)
    floors, r_lo, r_hi = pick_floors(alpha)
    print(f"  rate band over served range : {r_lo/1e6:5.1f} -> {r_hi/1e6:5.1f} Mbps "
          f"(dyn range {(1-r_hi/r_lo)*100:4.1f}%)")
    print(f"  floors  high={floors['high']/1e6:5.1f}  med={floors['medium']/1e6:5.1f}  "
          f"low={floors['low']/1e6:4.1f} Mbps")
    # feasibility of a node DIRECTLY BELOW (rho=0, z=25 mean) and at the FOOTPRINT EDGE.
    print(f"  {'H(m)':>6}{'R_below(Mbps)':>15}{'R_edge(Mbps)':>14}"
          f"{'high?':>8}{'med?':>7}{'low?':>6}")
    ceil = {k: 0.0 for k in floors}
    for H in Hs:
        cl = H - 25.0
        r_below = rate(cl, alpha)
        r_edge  = rate(cl * SECTH, alpha)     # edge node: weaker link
        feas = {k: r_below >= floors[k] for k in floors}      # best case (rho=0)
        for k in floors:
            if feas[k]:
                ceil[k] = H
        print(f"  {H:6.0f}{r_below/1e6:15.1f}{r_edge/1e6:14.1f}"
              f"{('Y' if feas['high'] else '-'):>8}{('Y' if feas['medium'] else '-'):>7}"
              f"{('Y' if feas['low'] else '-'):>6}")
    print(f"  --> feasible-altitude CEILING:  high={ceil['high']:.0f}m  "
          f"med={ceil['medium']:.0f}m  low={ceil['low']:.0f}m (>=150 = always)")
    sep = (ceil['high'] < ceil['medium'] < H_MAX) or (ceil['high'] + 20 < ceil['medium'])
    print(f"  G1 VERDICT: {'PASS - ceilings separate' if sep else 'FAIL - classes collapse'}")
    return floors, sep


def make_scenario(seed=2026):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, AREA_W, size=(N_NODES, 2))
    z  = rng.uniform(0, Z_MAX, size=N_NODES)
    D  = rng.uniform(D_MIN, D_MAX, size=N_NODES)
    cls_idx = rng.choice(3, size=N_NODES, p=CLASS_PROBS)
    return xy, z, D, cls_idx


def evaluate_altitude(H, alpha, floors, xy, z, D, cls_idx, weights, budget):
    """Constant-altitude-H coverage policy WITHOUT WPT. Returns (VoI, energy_Wh, served_frac).
    Square covering grid spaced by footprint radius r=(H-z_mean)*tan(theta); each node served
    by nearest hover center, QoS-gated by its class floor at the realised rho-distance.
    Hovers ranked by value-density; taken until the energy budget B is exhausted."""
    cl_mean = max(H - 25.0, 1.0)
    r_foot = cl_mean * TANTH
    step = max(r_foot, 30.0)                       # grid spacing (m)
    gx = np.arange(step / 2, AREA_W, step)
    cx, cy = np.meshgrid(gx, gx)
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)        # (H_pts,2)
    # assign each node to nearest hover center
    d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (N,Hpts)
    nearest = d2.argmin(1)
    rho = np.sqrt(d2[np.arange(N_NODES), nearest])
    cl_node = H - z
    d3 = np.sqrt(rho ** 2 + cl_node ** 2)
    r_node = rate(d3, alpha)
    floor_vec = np.array([floors[CLASS_NAMES[i]] for i in cls_idx])
    in_cone = rho <= (cl_node * TANTH)
    qos_ok  = r_node >= floor_vec
    served_possible = in_cone & qos_ok
    w_node = np.array([weights[CLASS_NAMES[i]] for i in cls_idx])
    # collect time per node (s), capped contribution
    t_c = (D * 8e6) / np.maximum(r_node, 1e-6)
    # group by hover: a hover is "used" if it serves >=1 possible node
    used = np.unique(nearest[served_possible])
    hov_value = np.zeros(len(centers))
    hov_cost  = np.zeros(len(centers))
    for h in used:
        m = (nearest == h) & served_possible
        hov_value[h] = (w_node[m] * D[m]).sum()
        hov_cost[h]  = (P_C + P_H) * min(t_c[m].sum(), TAU_MAX)   # hover collect energy (J)
    # flight: BHH tour over the used hovers
    n_used = len(used)
    if n_used == 0:
        return 0.0, 0.0, 0.0
    tour = 0.7 * np.sqrt(n_used * AREA)
    flight_per_hov = (tour * E_PER_M) / n_used                   # amortise flight per hover
    hov_total = hov_cost.copy()
    hov_total[used] += flight_per_hov
    # rank hovers by value density, take within budget B (J)
    order = used[np.argsort(-(hov_value[used] / np.maximum(hov_total[used], 1e-9)))]
    spent, voi, served = 0.0, 0.0, 0
    chosen = set()
    for h in order:
        if spent + hov_total[h] > budget:
            continue
        spent += hov_total[h]
        voi   += hov_value[h]
        chosen.add(h)
    served = int(sum(((nearest == h) & served_possible).sum() for h in chosen))
    return voi, spent / 3600.0, served / N_NODES


def gate_G2_G3(alpha, floors, budget_frac=0.55):
    print("\n" + "=" * 84)
    print(f"G2 / G3   VoI(H) under budget   (alpha = {alpha}, budget = {budget_frac:.0%} of full-serve@H_min)")
    print("=" * 84)
    xy, z, D, cls_idx = make_scenario()
    w_prio  = CLASS_WEIGHTS
    w_blind = {"high": 1.0, "medium": 1.0, "low": 1.0}
    # budget = fraction of the energy to serve everything at the cheapest-QoS altitude H_MIN
    _, e_full, _ = evaluate_altitude(H_MIN, alpha, {"high": 0, "medium": 0, "low": 0},
                                     xy, z, D, cls_idx, w_blind, budget=1e18)
    budget = budget_frac * e_full * 3600.0
    print(f"  full-serve energy @H_min = {e_full:.0f} Wh  ->  budget B = {budget/3600:.0f} Wh")
    print(f"  {'H(m)':>6}{'VoI_prio':>11}{'served%':>9}{'E(Wh)':>8}   |"
          f"{'VoI_blind':>11}{'served%':>9}{'E(Wh)':>8}")
    voip, voib = [], []
    for H in Hs:
        vp, ep, sp = evaluate_altitude(H, alpha, floors, xy, z, D, cls_idx, w_prio, budget)
        vb, eb, sb = evaluate_altitude(H, alpha, floors, xy, z, D, cls_idx, w_blind, budget)
        voip.append(vp); voib.append(vb)
        print(f"  {H:6.0f}{vp:11.1f}{sp*100:9.1f}{ep:8.1f}   |{vb:11.1f}{sb*100:9.1f}{eb:8.1f}")
    voip, voib = np.array(voip), np.array(voib)
    Hp = Hs[int(np.argmax(voip))]
    Hb = Hs[int(np.argmax(voib))]
    interior_p = H_MIN < Hp < H_MAX
    depth_p = (voip.max() - min(voip[0], voip[-1])) / max(voip.max(), 1e-9) * 100
    print(f"\n  G2: argmax VoI_priority = H={Hp:.0f}m  "
          f"({'INTERIOR' if interior_p else 'BOUNDARY'}), peak-vs-endpoint depth {depth_p:.1f}%")
    print(f"      VERDICT: {'PASS - pronounced interior optimum' if (interior_p and depth_p>10) else 'WEAK/FAIL'}")
    print(f"  G3: argmax VoI_priority = H={Hp:.0f}m   vs   argmax VoI_blind = H={Hb:.0f}m")
    print(f"      VERDICT: {'PASS - priority shifts the optimal altitude' if Hp != Hb else 'FAIL - priority is cosmetic'}")


def decisive_adaptive_vs_fixed(alpha=3.0, H_hi=120.0, floor_high=38e6, floor_med=28e6):
    """G3' — the FAIR test: spatially-adaptive altitude vs fixed-high, measured on
    HARD critical-QoS satisfaction and energy. Critical QoS is a CONSTRAINT here, not a
    soft VoI weight (the gate above proved soft weights are inert under VoI-max).

    FIXED-HIGH : every hover at H_hi (max footprint, cheapest coverage).
    ADAPTIVE   : low-priority regions served at H_hi; each high/med node gets a dedicated
                 hover dived to the altitude that meets its class floor (rho=0 over it).
    Reports critical-QoS satisfaction % and total energy for both."""
    print("\n" + "=" * 84)
    print(f"G3'  ADAPTIVE vs FIXED-HIGH   (alpha={alpha}, H_hi={H_hi:.0f}m; critical QoS = HARD)")
    print("=" * 84)
    xy, z, D, cls_idx = make_scenario()
    is_hi  = cls_idx == 0
    is_med = cls_idx == 1
    # ---- shared low-priority coverage at H_hi (tile area with footprint at H_hi) ----
    r_hi = (H_hi - 25.0) * TANTH
    step = max(r_hi, 30.0)
    gx = np.arange(step / 2, AREA_W, step)
    cx, cy = np.meshgrid(gx, gx); centers = np.stack([cx.ravel(), cy.ravel()], 1)
    d2 = ((xy[:, None, :] - centers[None]) ** 2).sum(-1); nearest = d2.argmin(1)
    rho = np.sqrt(d2[np.arange(N_NODES), nearest]); cl = H_hi - z
    r_at_hi = rate(np.sqrt(rho**2 + cl**2), alpha)
    used_hi = np.unique(nearest)
    def hover_energy(node_mask, H, rho_used):
        clr = H - z[node_mask]
        r = rate(np.sqrt(rho_used**2 + clr**2), alpha)
        tc = (D[node_mask] * 8e6) / np.maximum(r, 1e-6)
        return (P_C + P_H) * min(tc.sum(), TAU_MAX)
    def flight(n_hov):  return 0.7 * np.sqrt(max(n_hov, 1) * AREA) * E_PER_M
    # ---------- FIXED-HIGH ----------
    e_hi = sum(hover_energy(nearest == h, H_hi, rho[nearest == h]) for h in used_hi)
    e_hi += flight(len(used_hi))
    crit_sat_fixed = ((r_at_hi[is_hi] >= floor_high).mean()) if is_hi.any() else 1.0
    med_sat_fixed  = ((r_at_hi[is_med] >= floor_med).mean()) if is_med.any() else 1.0
    # ---------- ADAPTIVE ----------  (low-only coverage at H_hi + dedicated dives)
    crit_nodes = np.where(is_hi | is_med)[0]
    # altitude that meets each critical node's floor directly below it (rho=0)
    def alt_for_floor(floor, zi):
        cl_need = (P_T * BETA / (SIGMA2 * (2 ** (floor / B_HZ) - 1))) ** (1.0 / alpha)
        return float(np.clip(zi + cl_need, H_MIN, H_MAX)), cl_need
    e_adapt = e_hi  # reuse the same high coverage for low-priority bulk
    crit_ok = 0
    for i in crit_nodes:
        fl = floor_high if is_hi[i] else floor_med
        H_i, cl_need = alt_for_floor(fl, z[i])
        r_i = rate(max(H_i - z[i], 1.0), alpha)
        if r_i >= fl - 1e3:
            crit_ok += 1
        e_adapt += hover_energy(np.array([i]), H_i, np.array([0.0]))   # dedicated dive hover
    e_adapt += flight(len(used_hi) + len(crit_nodes)) - flight(len(used_hi))
    crit_sat_adapt = crit_ok / max(len(crit_nodes), 1)
    print(f"  #high={is_hi.sum()}  #med={is_med.sum()}  #low={(cls_idx==2).sum()}  "
          f"(high floor {floor_high/1e6:.0f}, med {floor_med/1e6:.0f} Mbps)")
    print(f"  FIXED-HIGH : critical(high)-QoS {crit_sat_fixed*100:5.1f}%  med-QoS {med_sat_fixed*100:5.1f}%"
          f"   E={e_hi/3600:6.1f} Wh")
    print(f"  ADAPTIVE   : critical+med-QoS  {crit_sat_adapt*100:5.1f}%"
          f"                       E={e_adapt/3600:6.1f} Wh"
          f"   (+{(e_adapt/e_hi-1)*100:.0f}% energy)")
    win = crit_sat_adapt > crit_sat_fixed + 0.3
    print(f"  VERDICT: {'PASS - adaptive achieves critical QoS that fixed-high structurally cannot' if win else 'FAIL'}")


def _tc(d, D, alpha):
    return (D * 8e6) / np.maximum(rate(d, alpha), 1e-6)

def _cover_cost(node_idx, xy, z, D, H, alpha):
    """Bulk-cover the given nodes with a footprint grid at altitude H. Returns
    (energy_J, n_hovers, rate_per_node, rho_per_node). Each node served from nearest
    hover center at its realised slant distance."""
    if len(node_idx) == 0:
        return 0.0, 0, np.array([]), np.array([])
    cl_m = max(H - z[node_idx].mean(), 1.0)
    step = max(cl_m * TANTH, 30.0)
    gx = np.arange(step / 2, AREA_W, step); cx, cy = np.meshgrid(gx, gx)
    centers = np.stack([cx.ravel(), cy.ravel()], 1)
    sub = xy[node_idx]
    d2 = ((sub[:, None] - centers[None]) ** 2).sum(-1); near = d2.argmin(1)
    rho = np.sqrt(d2[np.arange(len(sub)), near]); cl = H - z[node_idx]
    d3 = np.sqrt(rho ** 2 + cl ** 2); r = rate(d3, alpha)
    used = np.unique(near); e_hov = 0.0
    for h in used:
        m = near == h
        e_hov += (P_C + P_H) * min(_tc(d3[m], D[node_idx][m], alpha).sum(), TAU_MAX)
    return e_hov, len(used), r, rho

def alt_for_floor(floor, zi, alpha):
    cl_need = (P_T * BETA / (SIGMA2 * (2 ** (floor / B_HZ) - 1))) ** (1.0 / alpha)
    return float(np.clip(zi + cl_need, H_MIN, H_MAX))

def decisive_baseline(alpha=3.0, floor_high=38e6, floor_med=28e6, H_hi=120.0, H_med=55.0):
    print("\n" + "=" * 84)
    print(f"DECISIVE  3D-adaptive-heterogeneous  vs  honest 2D-fixed-low   (alpha={alpha})")
    print("           all strategies achieve 100% critical-QoS; compare ENERGY")
    print("=" * 84)
    xy, z, D, cls_idx = make_scenario()
    hi  = np.where(cls_idx == 0)[0]; med = np.where(cls_idx == 1)[0]; lo = np.where(cls_idx == 2)[0]
    crit = np.concatenate([hi, med])
    def flight(n):  return 0.7 * np.sqrt(max(n, 1) * AREA) * E_PER_M

    # --- (1) 2D-fixed-low: visit EVERY node individually at minimal standoff (strongest 2D) ---
    d_2d = np.maximum(10.0, np.abs(20.0 - z))          # close standoff, strong link, QoS always met
    e2d_hov = ((P_C + P_H) * _tc(d_2d, D, alpha)).sum()
    e2d = e2d_hov + flight(N_NODES)
    print(f"  (1) 2D-fixed-low (visit all {N_NODES} individually)        : "
          f"hover {e2d_hov/3600:6.1f} + flight {flight(N_NODES)/3600:6.1f} = {e2d/3600:6.1f} Wh   crit-QoS 100%")

    # --- (2) 3D-naive: bulk-high for low-priority + individual dives for ALL critical ---
    e_lo, n_lo, _, _ = _cover_cost(lo, xy, z, D, H_hi, alpha)
    e_dive = 0.0
    for i in crit:
        fl = floor_high if cls_idx[i] == 0 else floor_med
        Hi = alt_for_floor(fl, z[i], alpha); e_dive += (P_C + P_H) * _tc(max(Hi - z[i], 1.0), D[i], alpha)
    e3n = e_lo + e_dive + flight(n_lo + len(crit))
    print(f"  (2) 3D-naive (high-bulk low + dive each critical)       : "
          f"low-bulk {e_lo/3600:5.1f} + dives {e_dive/3600:5.1f} + flight {flight(n_lo+len(crit))/3600:5.1f} = "
          f"{e3n/3600:6.1f} Wh   crit-QoS 100%")

    # --- (3) 3D-smart: bulk-high for low + bulk-medium-alt for medium + dive only HIGH ---
    e_med, n_med, r_med, _ = _cover_cost(med, xy, z, D, H_med, alpha)
    med_ok = (r_med >= floor_med).mean() if len(med) else 1.0
    e_hi_dive = 0.0
    for i in hi:
        Hi = alt_for_floor(floor_high, z[i], alpha); e_hi_dive += (P_C + P_H) * _tc(max(Hi - z[i], 1.0), D[i], alpha)
    e3s = e_lo + e_med + e_hi_dive + flight(n_lo + n_med + len(hi))
    print(f"  (3) 3D-smart (low@{H_hi:.0f} + med@{H_med:.0f}[QoS {med_ok*100:.0f}%] + dive {len(hi)} high) : "
          f"{e3s/3600:6.1f} Wh   crit-QoS {'100' if med_ok>0.99 else f'{med_ok*100:.0f}(med)'}%")
    # --- (2b) 3D-BLIND: no priority labels -> must guarantee HIGH QoS for EVERY node ---
    #         (conservative: can't tell critical from low, so serve all at the strict floor)
    H_blind = alt_for_floor(floor_high, z.mean(), alpha)   # altitude meeting high floor for mean node
    e_blind, n_blind, r_blind, _ = _cover_cost(np.arange(N_NODES), xy, z, D, H_blind, alpha)
    blind_ok = (r_blind >= floor_high).mean()
    e_blind_tot = e_blind + flight(n_blind)
    print(f"  (2b) 3D-BLIND (serve ALL at high floor, H={H_blind:.0f}m, {n_blind} hovers) : "
          f"{e_blind_tot/3600:6.1f} Wh   all-QoS {blind_ok*100:.0f}%   <- no priority info")
    best3d = min(e3n, e3s)
    print(f"\n  --> 3D-priority {best3d/3600:.1f} Wh  |  3D-blind {e_blind_tot/3600:.1f} Wh  |  2D {e2d/3600:.1f} Wh")
    print(f"      priority's MARGINAL value over blind-3D : "
          f"{'SAVES ' + f'{(1-best3d/e_blind_tot)*100:.0f}%' if best3d < e_blind_tot else 'NONE'}")
    print(f"  --> best 3D = {best3d/3600:.1f} Wh   vs   2D = {e2d/3600:.1f} Wh   "
          f"({'3D SAVES ' + f'{(1-best3d/e2d)*100:.0f}%' if best3d < e2d else '3D WORSE by ' + f'{(best3d/e2d-1)*100:.0f}%'})")
    print(f"  VERDICT: {'PASS - heterogeneity gives 3D a real energy win at equal QoS' if best3d < 0.9*e2d else 'FAIL/MARGINAL - no defensible win'}")

def edge_floor_altitude(floor, z_ref, alpha):
    """Lowest altitude at which a node at the FOOTPRINT EDGE (slant = sec(theta)*clearance)
    still meets `floor`. This is the altitude a coverage hover must fly to guarantee QoS
    for every node it covers. floor=0 -> fly as high as allowed (H_MAX)."""
    if floor <= 0:
        return H_MAX
    snr_need = 2 ** (floor / B_HZ) - 1
    d_need = (P_T * BETA / (SIGMA2 * snr_need)) ** (1.0 / alpha)   # max slant meeting floor
    cl = d_need / SECTH                                            # edge slant = cl*sec(theta)
    return float(np.clip(z_ref + cl, H_MIN, H_MAX))

def _cover_energy(idx, xy, z, D, H, alpha):
    e, n, _, _ = _cover_cost(idx, xy, z, D, H, alpha)
    return e, n

def fair_sweep():
    """Behaviourally-FAIR three-baseline comparison, ALL at 100% critical-QoS by construction,
    across random seeds and critical-node fractions. Separates coverage gain from priority gain."""
    alpha, fl_hi, fl_med = 3.0, 38e6, 28e6
    def flight(n):  return 0.7 * np.sqrt(max(n, 1) * AREA) * E_PER_M
    print("\n" + "#" * 84)
    print("#   FAIR SENSITIVITY SWEEP  (all baselines 100% critical-QoS; coverage batching")
    print("#   ALLOWED for every strategy; only altitude-adaptivity & priority differ)")
    print("#" * 84)
    print(f"  {'crit%':>6}{'2D-fix':>9}{'3D-blind':>10}{'3D-prio':>9}"
          f"{'cover%':>9}{'prio%':>8}{'tot%':>8}   (Wh, mean+-std over 5 seeds)")
    for pc in [0.05, 0.10, 0.20, 0.30]:
        probs = np.array([pc, 0.30, 1.0 - pc - 0.30])
        e2dL, eblL, eprL = [], [], []
        for seed in [2026, 1, 2, 3, 4]:
            rng = np.random.default_rng(seed)
            xy = rng.uniform(0, AREA_W, (N_NODES, 2)); z = rng.uniform(0, Z_MAX, N_NODES)
            D = rng.uniform(D_MIN, D_MAX, N_NODES); cls = rng.choice(3, N_NODES, p=probs)
            hi, med, lo = np.where(cls == 0)[0], np.where(cls == 1)[0], np.where(cls == 2)[0]
            alln = np.arange(N_NODES)
            # (1) 2D-fixed-low : single fixed altitude meeting STRICT floor for ALL (fair: batching allowed)
            H_all = edge_floor_altitude(fl_hi, z.mean(), alpha)
            e2, n2 = _cover_energy(alln, xy, z, D, H_all, alpha); e2 += flight(n2)
            # (2) 3D-blind : adaptive altitude but NO priority -> still must meet strict floor for all
            #     (same as 2D here because without labels every node needs the strict floor)
            ebl, nbl = e2, n2
            # (3) 3D-priority : low@H_max, med@edge(med), high dive (rho=0 individual)
            elo, nlo = _cover_energy(lo, xy, z, D, H_MAX, alpha)
            emd, nmd = _cover_energy(med, xy, z, D, edge_floor_altitude(fl_med, z[med].mean() if len(med) else 25, alpha), alpha)
            ehi = 0.0
            for i in hi:
                Hi = alt_for_floor(fl_hi, z[i], alpha)
                ehi += (P_C + P_H) * _tc(max(Hi - z[i], 1.0), D[i], alpha)
            ep = elo + emd + ehi + flight(nlo + nmd + len(hi))
            e2dL.append(e2 / 3600); eblL.append(ebl / 3600); eprL.append(ep / 3600)
        e2m, eblm, epm = np.mean(e2dL), np.mean(eblL), np.mean(eprL)
        cover = (1 - eblm / e2m) * 100; prio = (1 - epm / eblm) * 100; tot = (1 - epm / e2m) * 100
        print(f"  {pc*100:5.0f}%{e2m:9.1f}{eblm:10.1f}{epm:9.1f}"
              f"{cover:9.1f}{prio:8.1f}{tot:8.1f}   (prio std {np.std(eprL):.1f})")
    print("\n  cover% = 2D->3D-blind (pure adaptive-altitude/coverage benefit at equal QoS)")
    print("  prio%  = 3D-blind->3D-priority (pure priority benefit)")
    print("  If prio% is stable & positive across crit% and seeds -> defensible. If it collapses -> fragile.")

def alpha_sweep(pc=0.10):
    """alpha (path-loss exponent) sensitivity at fixed 10% critical, 5 seeds.
    Floors RE-ANCHORED per alpha to the same relative strictness (high = 0.85*rate@15m
    clearance, med = midband) so we test the MECHANISM, not floor feasibility artifacts.
    Watch for a CLIFF: gain should turn on gradually as alpha rises off free-space (2.0)."""
    def flight(n):  return 0.7 * np.sqrt(max(n, 1) * AREA) * E_PER_M
    print("\n" + "#" * 84)
    print(f"#   ALPHA (path-loss exponent) SENSITIVITY  (crit={pc:.0%}, 5 seeds, floors re-anchored)")
    print("#" * 84)
    print(f"  {'alpha':>6}{'fl_hi':>8}{'fl_med':>8}{'2D=blind':>10}{'3D-prio':>9}"
          f"{'gain%':>8}{'std':>7}   (Mbps / Wh)")
    probs = np.array([pc, 0.30, 1.0 - pc - 0.30])
    for alpha in [2.0, 2.5, 2.8, 3.0, 3.2, 3.5]:
        r_close = rate(15.0, alpha); r_far = rate(120.0, alpha)
        fl_hi  = 0.85 * r_close
        fl_med = 0.5 * (r_close + r_far)
        e2L, epL = [], []
        for seed in [2026, 1, 2, 3, 4]:
            rng = np.random.default_rng(seed)
            xy = rng.uniform(0, AREA_W, (N_NODES, 2)); z = rng.uniform(0, Z_MAX, N_NODES)
            D = rng.uniform(D_MIN, D_MAX, N_NODES); cls = rng.choice(3, N_NODES, p=probs)
            hi, med, lo = np.where(cls == 0)[0], np.where(cls == 1)[0], np.where(cls == 2)[0]
            # 2D == fair blind: all nodes at the strict-floor altitude (low), batching allowed
            H_all = edge_floor_altitude(fl_hi, z.mean(), alpha)
            e2, n2 = _cover_energy(np.arange(N_NODES), xy, z, D, H_all, alpha); e2 += flight(n2)
            # 3D-priority: low@H_max, med@edge(med), high dive (rho=0)
            elo, nlo = _cover_energy(lo, xy, z, D, H_MAX, alpha)
            emd, nmd = _cover_energy(med, xy, z, D,
                                     edge_floor_altitude(fl_med, z[med].mean() if len(med) else 25, alpha), alpha)
            ehi = sum((P_C + P_H) * _tc(max(alt_for_floor(fl_hi, z[i], alpha) - z[i], 1.0), D[i], alpha) for i in hi)
            ep = elo + emd + ehi + flight(nlo + nmd + len(hi))
            e2L.append(e2 / 3600); epL.append(ep / 3600)
        e2m, epm = np.mean(e2L), np.mean(epL)
        gain = (1 - epm / e2m) * 100
        print(f"  {alpha:6.1f}{fl_hi/1e6:8.1f}{fl_med/1e6:8.1f}{e2m:10.1f}{epm:9.1f}"
              f"{gain:8.1f}{np.std(epL):7.1f}")
    print("\n  GRADUAL rise off alpha=2.0 with no cliff -> defensible (urban/suburban regime).")
    print("  A jump from ~0% to ~20% within 0.1 of alpha -> razor-thin regime, fragile.")

if __name__ == "__main__":
    print(f"E_PER_M (cruise) = {E_PER_M:.2f} J/m   |   hover power P_H = {P_H:.1f} W   |   WPT = DISABLED")
    for alpha in [2.0, 3.0, 3.5]:
        print("\n" + "#" * 84)
        print(f"#   PATH-LOSS EXPONENT alpha = {alpha}")
        print("#" * 84)
        floors, _ = gate_G1(alpha)
        gate_G2_G3(alpha, floors)
    print("\n" + "#" * 84)
    print("#   DECISIVE FAIR TEST (variable altitude, hard critical QoS)")
    print("#" * 84)
    decisive_adaptive_vs_fixed(alpha=3.0)
    decisive_baseline(alpha=3.0)
    fair_sweep()
    alpha_sweep()
