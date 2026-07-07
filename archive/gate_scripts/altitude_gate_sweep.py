"""
ALTITUDE PHYSICS GATE  (no torch — pure closed-form from the project's own model).

Mentor-mandated acceptance gate (Q7): before ANY retraining, sweep H in [20,150] m
and check whether altitude actually moves communication quality and energy. If the
physics has no tradeoff, no RL objective can invent one.

Every formula below is copied from the verified code paths:
  * SNR / rate ........ channel_model.ChannelModel.compute_data_rate
  * WPT time t_e ...... tenma_trainer._hover_energy  (power_model branch)
  * collect time t_c .. tenma_trainer._hover_energy
  * E_serve(H) ........ service_altitude.ServiceAltitudeAllocator._serve_energy
  * hover power P_H ... propulsion P0+Pi  (RotaryWingPower.P_hover)
Constants are the literal values in atom_3d/configs/params.yaml.
"""
import numpy as np

# ----------------------------------------------------------------------------
# EXACT params.yaml constants
# ----------------------------------------------------------------------------
c_light   = 3e8
f_c       = 2.4e9
BETA      = (c_light / (4 * np.pi * f_c)) ** 2          # ~9.9e-5
SIGMA2    = 10 ** ((-110.0 - 30) / 10)                  # -110 dBm -> 1e-14 W
B_HZ      = 2.0e6
P_T       = 0.5
P_C       = 0.5
ETA_L     = 0.8
E_COEFF   = 2.5e-7          # J per MB harvested-energy demand
P0, PI    = 79.86, 88.63
P_H       = P0 + PI         # hover power = 168.49 W (the expensive peak)
C_CLIMB   = 2.0 * 9.81 / 0.5   # m*g/eta = 39.24 J/m
C_D       = 4.0
H_MIN, H_MAX = 20.0, 150.0
TANTH     = np.tan(np.radians(60.0))
TAU_MAX   = 30.0
Z_MEAN    = 25.0            # nodes zi ~ U(0,50)
D_MEAN    = 0.85           # demand ~ U(0.2,1.5) MB
C_MAX     = 150.0
N_NODES   = 500
AREA      = 1000.0 * 1000.0
R_MIN_HIGH = 4.2e7         # 42 Mbps critical floor

def snr(d, sigma2=SIGMA2):          return P_T * BETA / (d ** 2 * sigma2)
def rate(d, sigma2=SIGMA2):         return B_HZ * np.log2(1.0 + snr(d, sigma2))
def t_e(d, demand=D_MEAN):                                    # WPT charge time
    P_R = ETA_L * BETA * P_T / d ** 2
    return (E_COEFF * demand) / np.maximum(P_R, 1e-30)
def t_c(d, demand=D_MEAN, sigma2=SIGMA2):                     # collect time
    return (demand * 8e6) / np.maximum(rate(d, sigma2), 1e-6)

def rotary_max_range_energy_per_m():
    """min_V P(V)/V for the Zeng-Zhang curve -> J per metre of cruise flight."""
    U_tip, v0, d0, rho, s, A = 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    V = np.linspace(0.5, 40.0, 4000)
    P = (P0 * (1 + 3 * V ** 2 / U_tip ** 2)
         + PI * (np.sqrt(1 + V ** 4 / (4 * v0 ** 4)) - V ** 2 / (2 * v0 ** 2)) ** 0.5
         + 0.5 * d0 * rho * s * A * V ** 3)
    return float(np.min(P / V))
E_PER_M = rotary_max_range_energy_per_m()

# ============================================================================
print("=" * 78)
print("PART 1 — Per-node isolation sweep (node directly below, rho=0, d=H-z)")
print("        CURRENT params.yaml regime")
print("=" * 78)
print(f"{'H(m)':>6}{'d(m)':>7}{'SNR(dB)':>9}{'Rate(Mbps)':>12}"
      f"{'t_e(s)':>9}{'t_c(s)':>9}{'E_hover(J)':>12}{'E_serve(J)':>12}")
Hs = np.array([20, 30, 45, 60, 75, 90, 110, 130, 150], float)
d  = Hs - 0.0   # node at ground z=0 directly below -> d = H
rate_v, esrv_v = [], []
for H, dd in zip(Hs, d):
    s_db = 10 * np.log10(snr(dd))
    rt   = rate(dd) / 1e6
    te   = t_e(dd); tc = t_c(dd)
    e_hov = (P_T + P_H) * te + (P_C + P_H) * tc
    e_srv = e_hov + (C_CLIMB + C_D) * max(H_MAX - H, 0.0)   # dive-and-return (service_altitude)
    rate_v.append(rt); esrv_v.append(e_srv)
    print(f"{H:6.0f}{dd:7.0f}{s_db:9.1f}{rt:12.1f}{te:9.1f}{tc:9.2f}{e_hov:12.0f}{e_srv:12.0f}")
rate_v = np.array(rate_v); esrv_v = np.array(esrv_v)
rate_var = (rate_v.max() - rate_v.min()) / rate_v.max() * 100
argmin_H = Hs[int(np.argmin(esrv_v))]
print(f"\n  Rate dynamic range : {rate_var:4.1f}%  (gate needs >= 25%)")
print(f"  E_serve argmin     : H = {argmin_H:.0f} m  "
      f"({'INTERIOR' if H_MIN < argmin_H < H_MAX else 'BOUNDARY -> no interior optimum'})")
print(f"  Critical floor 42 Mbps met up to H ~ "
      f"{Hs[rate_v*1e6 >= R_MIN_HIGH].max() if np.any(rate_v*1e6>=R_MIN_HIGH) else 0:.0f} m")
print(f"  WPT charge time exceeds tau_max(30s) above H ~ "
      f"{Hs[np.array([t_e(x) for x in Hs])>TAU_MAX].min() if np.any(np.array([t_e(x) for x in Hs])>TAU_MAX) else 999:.0f} m")

# ============================================================================
print("\n" + "=" * 78)
print("PART 2 — Same sweep, DE-SATURATED link (Q2: best-case SNR=20 dB at H_min)")
print("=" * 78)
SIGMA2_NEW = P_T * BETA / (H_MIN ** 2 * 100.0)     # SNR(H_min)=100=20 dB
print(f"  required noise sigma^2 = {SIGMA2_NEW:.2e} W "
      f"({10*np.log10(SIGMA2_NEW)+30:.0f} dBm; was -110 dBm)")
print(f"{'H(m)':>6}{'SNR(dB)':>9}{'Rate(Mbps)':>12}{'t_c(s)':>9}")
rate2 = []
for H in Hs:
    dd = H
    s_db = 10 * np.log10(snr(dd, SIGMA2_NEW))
    rt = rate(dd, SIGMA2_NEW) / 1e6
    rate2.append(rt)
    print(f"{H:6.0f}{s_db:9.1f}{rt:12.2f}{t_c(dd, sigma2=SIGMA2_NEW):9.2f}")
rate2 = np.array(rate2)
print(f"\n  Rate dynamic range : {(rate2.max()-rate2.min())/rate2.max()*100:4.1f}%  "
      f"(gate needs >= 25%)  ->  ratio {rate2.max()/rate2.min():.1f}x")

# ============================================================================
print("\n" + "=" * 78)
print("PART 3 — Coverage-scenario total energy E(H) to serve ALL N=500 nodes")
print("        (first-order: footprint coverage vs per-node charge/collect)")
print("=" * 78)
def mission_energy(H, sigma2, decoupled_wpt):
    cl = max(H - Z_MEAN, 1.0)                       # clearance above mean node
    r  = cl * TANTH                                 # footprint radius
    foot = min(np.pi * r ** 2, AREA)
    d_eff = np.sqrt(cl ** 2 + (r ** 2) / 2.0)       # mean in-disk distance E[d]
    te = t_e(d_eff)
    if decoupled_wpt:                               # Q3: fixed per-node charge cost
        te = t_e(np.sqrt(50.0 ** 2))                # frozen at a 50 m reference
    tc = t_c(d_eff, sigma2=sigma2)
    tau_node = te + tc
    lam = N_NODES / AREA
    k = min(lam * foot, C_MAX / D_MEAN, np.floor(TAU_MAX / max(tau_node, 1e-6)))
    k = max(k, 1.0)
    n_hov = np.ceil(N_NODES / k)
    tour = 0.7 * np.sqrt(max(n_hov, 1) * AREA)      # BHH tour-length estimate
    e_flight = tour * E_PER_M
    e_hover_all = N_NODES * ((P_T + P_H) * te + (P_C + P_H) * tc)
    return (e_flight + e_hover_all) / 3600.0, n_hov, k   # Wh
for label, sig, dec in [("CURRENT (sat link, WPT~d^2)", SIGMA2, False),
                        ("FIXED   (de-sat + decoupled WPT)", SIGMA2_NEW, True)]:
    print(f"\n  {label}")
    print(f"  {'H(m)':>6}{'nodes/hover':>13}{'#hovers':>9}{'E_total(Wh)':>13}")
    Es = []
    for H in Hs:
        e, nh, k = mission_energy(H, sig, dec)
        Es.append(e)
        print(f"  {H:6.0f}{k:13.1f}{nh:9.0f}{e:13.1f}")
    Es = np.array(Es)
    am = Hs[int(np.argmin(Es))]
    print(f"    -> E argmin at H={am:.0f} m  "
          f"({'INTERIOR optimum' if H_MIN < am < H_MAX else 'BOUNDARY (fly-low/high)'})")

print("\n" + "=" * 78)
print("GATE VERDICT printed above: PART1 rate% and E-argmin = current regime;")
print("PART2/3 show whether the Q2+Q3 parameter changes create a real tradeoff.")
print("=" * 78)
