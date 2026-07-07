"""
STRICT-GATE SEARCH over path-loss exponent (Q2 mechanism = exponent, locked) with
WPT decoupled from altitude (Q3). No noise fudge (sigma2 stays -110 dBm), no mmWave.

Strict gate (mentor): rate variation > 50%, critical QoS achievable over a useful
altitude band, coverage-energy interior minimum with > 10% depth.

Comm link:  Rate(d) = B log2(1 + P_T*beta / (d^n * sigma2))     <- exponent n is the knob
WPT:        fixed per-node charge cost (decoupled from altitude)  <- Q3
Everything else = exact params.yaml.
"""
import numpy as np

c_light, f_c = 3e8, 2.4e9
BETA   = (c_light / (4 * np.pi * f_c)) ** 2
SIGMA2 = 1e-14                      # -110 dBm, UNCHANGED (no noise fudge)
B_HZ   = 2.0e6
P_T = P_C = 0.5
ETA_L  = 0.8
E_COEFF = 2.5e-7
P_H    = 79.86 + 88.63             # 168.49 W hover peak
C_CLIMB, C_D = 39.24, 4.0
H_MIN, H_MAX = 20.0, 150.0
TANTH  = np.tan(np.radians(60.0))
TAU_MAX = 30.0
Z_MEAN, D_MEAN = 25.0, 0.85
C_MAX, N_NODES = 150.0, 500

def rotary_E_per_m():
    U_tip, v0, d0, rho, s, A = 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    V = np.linspace(0.5, 40, 4000)
    P = (79.86*(1+3*V**2/U_tip**2)
         + 88.63*(np.sqrt(1+V**4/(4*v0**4))-V**2/(2*v0**2))**0.5
         + 0.5*d0*rho*s*A*V**3)
    return float(np.min(P/V))
E_PER_M = rotary_E_per_m()

# Q3: fixed per-node WPT energy — "charge at a low reference altitude", altitude-independent
T_E_FIX = E_COEFF * D_MEAN / (ETA_L * BETA * P_T / 30.0**2)   # t_e at d=30 m, n=2
E_WPT_FIX = (P_T + P_H) * T_E_FIX

def rate(d, n):  return B_HZ * np.log2(1.0 + P_T * BETA / (d**n * SIGMA2))
def snr_db(d, n): return 10*np.log10(P_T * BETA / (d**n * SIGMA2))

Hs = np.array([20,30,45,60,75,90,110,130,150], float)

def coverage_energy(H, n, area_side):
    area = area_side**2
    cl = max(H - Z_MEAN, 1.0)
    r  = cl * TANTH
    foot = min(np.pi*r**2, area)
    d_eff = np.sqrt(cl**2 + r**2/2)
    tc = (D_MEAN*8e6)/max(rate(d_eff, n), 1e-6)
    tau_node = T_E_FIX + tc
    lam = N_NODES/area
    k = max(min(lam*foot, C_MAX/D_MEAN, np.floor(TAU_MAX/max(tau_node,1e-6))), 1.0)
    n_hov = np.ceil(N_NODES/k)
    e_flight = 0.7*np.sqrt(n_hov*area)*E_PER_M
    e_hover  = N_NODES*(E_WPT_FIX + (P_C+P_H)*tc)
    return (e_flight + e_hover)/3600.0   # Wh

print(f"WPT fixed per-node energy (decoupled): {E_WPT_FIX:.0f} J  "
      f"(t_e={T_E_FIX:.1f}s @30m)\n")
print(f"{'n':>4}{'area':>6}{'SNR@20':>8}{'SNR@150':>8}{'Rate@20':>9}{'Rate@150':>10}"
      f"{'rateVar%':>9}{'E*H(m)':>8}{'Edepth%':>9}{'  gate'}")
for n in [2.6, 3.0, 3.2, 3.5]:
    for area_side in [1000.0, 2000.0]:
        r20, r150 = rate(20, n)/1e6, rate(150, n)/1e6
        rate_var = (r20 - r150)/r20*100
        Es = np.array([coverage_energy(H, n, area_side) for H in Hs])
        iH = int(np.argmin(Es))
        Hstar = Hs[iH]
        depth = (Es.max() - Es.min())/Es.max()*100
        interior = H_MIN < Hstar < H_MAX
        passes = (rate_var > 50) and interior and (depth > 10)
        tag = "PASS" if passes else ("rate<50" if rate_var<=50 else
              ("boundary" if not interior else "depth<10"))
        print(f"{n:>4.1f}{area_side:>6.0f}{snr_db(20,n):>8.1f}{snr_db(150,n):>8.1f}"
              f"{r20:>9.1f}{r150:>10.1f}{rate_var:>9.1f}{Hstar:>8.0f}{depth:>9.1f}  {tag}")

# ---- detailed look at the most promising regime ----
print("\nDetail — n=3.2, area=2000 m  (comm rate band -> re-tune R_min to this):")
n = 3.2
for H in Hs:
    d = H
    print(f"  H={H:5.0f}  SNR={snr_db(d,n):6.1f} dB   Rate={rate(d,n)/1e6:6.2f} Mbps   "
          f"E_cov={coverage_energy(H,n,2000.0):7.1f} Wh")
band = rate(60, n)/1e6
print(f"\n  Suggested critical R_min ~ Rate(H=60m) = {band:.1f} Mbps "
      f"(critical met only when UAV dives below ~60 m)")
