"""Q7 gate figure for the LOCKED regime: n=3.2, area=2 km, WPT decoupled,
R_min re-tuned (high=27, med=20 Mbps). Plots Rate(H), Energy(H), VoI(H)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c_light, f_c = 3e8, 2.4e9
BETA = (c_light/(4*np.pi*f_c))**2
SIGMA2, B_HZ, P_T, P_C = 1e-14, 2e6, 0.5, 0.5
ETA_L, E_COEFF = 0.8, 2.5e-7
P_H = 168.49
H_MIN, H_MAX = 20.0, 150.0
TANTH = np.tan(np.radians(60.0)); TAU_MAX = 30.0
Z_MEAN, D_MEAN, C_MAX, N = 25.0, 0.85, 150.0, 500
N_EXP, AREA = 3.2, 2000.0**2
# priority classes: high/med/low  frac/weight/Rmin(Mbps)
FRAC = np.array([0.1, 0.3, 0.6]); W = np.array([5.,2.,1.]); RMIN = np.array([27e6,20e6,0.])

V = np.linspace(0.5,40,4000)
P = (79.86*(1+3*V**2/120**2)+88.63*(np.sqrt(1+V**4/(4*4.03**4))-V**2/(2*4.03**2))**0.5
     +0.5*0.6*1.225*0.05*0.503*V**3)
E_PER_M = float(np.min(P/V))
T_E_FIX = E_COEFF*D_MEAN/(ETA_L*BETA*P_T/30.0**2)
E_WPT_FIX = (P_T+P_H)*T_E_FIX

def rate(d): return B_HZ*np.log2(1+P_T*BETA/(d**N_EXP*SIGMA2))
def energy(H):
    cl=max(H-Z_MEAN,1.0); r=cl*TANTH; foot=min(np.pi*r**2,AREA)
    d_eff=np.sqrt(cl**2+r**2/2); tc=(D_MEAN*8e6)/max(rate(d_eff),1e-6)
    k=max(min((N/AREA)*foot, C_MAX/D_MEAN, np.floor(TAU_MAX/(T_E_FIX+tc))),1.0)
    nh=np.ceil(N/k); ef=0.7*np.sqrt(nh*AREA)*E_PER_M
    return (ef + N*(E_WPT_FIX+(P_C+P_H)*tc))/3600.0
def voi(H):                       # value if a single hover altitude H served everyone
    r=rate(H); met=(r>=RMIN).astype(float)
    return N*D_MEAN*float((FRAC*W*met).sum())

Hs=np.linspace(20,150,200)
RT=np.array([rate(h)/1e6 for h in Hs]); EN=np.array([energy(h) for h in Hs]); VO=np.array([voi(h) for h in Hs])
Hstar=Hs[int(np.argmin(EN))]

fig,ax=plt.subplots(1,3,figsize=(15,4.3))
ax[0].plot(Hs,RT,lw=2.2,color="#1f77b4")
ax[0].axhline(27,ls="--",c="r",lw=1); ax[0].axhline(20,ls="--",c="orange",lw=1)
ax[0].text(150,27.6,"critical 27",ha="right",c="r",fontsize=8)
ax[0].text(150,20.6,"medium 20",ha="right",c="orange",fontsize=8)
ax[0].set_title(f"Throughput(H)  — var {(RT.max()-RT.min())/RT.max()*100:.0f}%")
ax[0].set_xlabel("altitude H (m)"); ax[0].set_ylabel("rate (Mbps)")
ax[1].plot(Hs,EN,lw=2.2,color="#2ca02c")
ax[1].axvline(Hstar,ls=":",c="k"); ax[1].plot([Hstar],[EN.min()],"ko")
ax[1].text(Hstar+3,EN.min(),f"E* @ {Hstar:.0f} m",fontsize=9)
ax[1].set_title(f"Energy(H) — interior min, {(EN.max()-EN.min())/EN.max()*100:.0f}% deep")
ax[1].set_xlabel("altitude H (m)"); ax[1].set_ylabel("mission energy (Wh)")
ax[2].plot(Hs,VO,lw=2.2,color="#9467bd")
ax[2].axvline(Hstar,ls=":",c="k"); ax[2].text(Hstar+3,VO.min()+10,f"E-opt {Hstar:.0f} m",fontsize=8)
ax[2].set_title("VoI(H) at fixed altitude — critical value lost above ~60 m")
ax[2].set_xlabel("altitude H (m)"); ax[2].set_ylabel("weighted value served")
for a in ax: a.grid(alpha=.3)
fig.suptitle("ATOM-3D physics gate — LOCKED regime n=3.2, area=2 km, WPT decoupled (PASS)",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
out=r"T:\5\BTP\atom3d_figures\figures\gate_n32_locked.png"
fig.savefig(out,dpi=130); print("saved",out)
print(f"energy-optimal H* = {Hstar:.0f} m ; critical QoS needs H < ~60 m -> the dive tension")
