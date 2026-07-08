"""
figure09_rate_cdf.py

Figure 9 - CDF of High-Priority Achieved Rates (reference panel (f)).

Empirical CDF of the per-node achieved data rate for HIGH-priority nodes,
one curve per policy, log-scaled rate axis, with the 38 Mbps QoS floor as a
vertical reference. Message: ATOM-3D-VoI shifts the entire distribution to
the right - only ~10 % of its high-priority nodes fall below the floor,
versus ~24 % (2D-AUTO) and ~30 % (3D-GNN), consistent with Figure 5.

DATA
----
Samples come from `generate_placeholder_rate_samples`; swap in real per-node
achieved rates via `load_rate_results` (results_data/high_rate_samples.npz)
with no plotting changes.

Runs independently:  python figure09_rate_cdf.py
Exports results/figure09_rate_cdf.png (600 DPI) via common_plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common_style import setup_style, COLORS
from common_plot import save_figure


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure09_rate_cdf"
TITLE = "CDF of High-Priority Achieved Rates"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "high_rate_samples.npz"

QOS_FLOOR_MBPS = 38.0     # high-priority rate floor (DATA_SPEC)

METHOD_STYLE = [
    ("2d_auto", "2D-AUTO", "#9E9E9E", "--", 1.6),
    ("3d_gnn", "3D-GNN", "#FB8C00", "-.", 1.8),
    ("atom3d", "ATOM-3D-VoI (Ours)", COLORS["ours"], "-", 2.6),
]


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real samples)
# =============================================================================

def generate_placeholder_rate_samples(n=2000, seed=23):
    """PLACEHOLDER per-node achieved-rate samples (Mbps) for HIGH-priority
    nodes. Lognormal parameters are calibrated so the below-floor fraction at
    38 Mbps matches Figure 5 exactly:
        ATOM-3D-VoI  P(rate < 38) = 0.10   (90 % satisfied)
        2D-AUTO      P(rate < 38) = 0.24   (76 % satisfied)
        3D-GNN       P(rate < 38) = 0.30   (70 % satisfied)
    Schema: {method_key: (n,) float array of Mbps, "placeholder": True}
    """
    rng = np.random.default_rng(seed)
    ln_floor = np.log(QOS_FLOOR_MBPS)
    # (sigma, z-score of the floor) per method -> mu = ln_floor - z * sigma
    params = {
        "atom3d": (0.35, -1.2816),   # 10th percentile at the floor
        "2d_auto": (0.45, -0.7063),  # 24th percentile
        "3d_gnn": (0.50, -0.5244),   # 30th percentile
    }
    data = {}
    for m, (sigma, z) in params.items():
        mu = ln_floor - z * sigma
        data[m] = rng.lognormal(mu, sigma, n)
    data["placeholder"] = True
    return data


def load_rate_results():
    """Load real achieved-rate samples if present, else the placeholder.

    Real export: results_data/high_rate_samples.npz with one array per method
    key (Mbps of every high-priority node over the evaluation episodes).
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH)
        data = {k: np.asarray(z[k], float) for k in z.files}
        data["placeholder"] = False
        return data
    return generate_placeholder_rate_samples()


# =============================================================================
# PLOTTING  (source-agnostic)
# =============================================================================

def plot_rate_cdf(data):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    for key, label, color, ls, lw in METHOD_STYLE:
        x = np.sort(np.asarray(data[key], float))
        cdf = np.arange(1, len(x) + 1) / len(x)
        ax.semilogx(x, cdf, color=color, ls=ls, lw=lw, label=label,
                    zorder=6 if key == "atom3d" else 4)

    # QoS floor reference + below-floor fractions at the floor.
    ax.axvline(QOS_FLOOR_MBPS, color=COLORS["high"] if "high" in COLORS
               else "#E53935", ls=(0, (4, 3)), lw=1.2, alpha=0.8, zorder=3)
    ax.text(QOS_FLOOR_MBPS * 1.06, 0.03, "QoS floor\n38 Mbps", fontsize=8,
            color="#E53935", ha="left", va="bottom")

    for key, label, color, *_ in METHOD_STYLE:
        x = np.asarray(data[key], float)
        frac = float((x < QOS_FLOOR_MBPS).mean())
        ax.scatter(QOS_FLOOR_MBPS, frac, s=34, facecolor=color,
                   edgecolor="black", linewidths=0.5, zorder=7)
        ax.annotate(f"{frac * 100:.0f}%", xy=(QOS_FLOOR_MBPS, frac),
                    xytext=(-30, 2), textcoords="offset points", fontsize=7.5,
                    color=color, fontweight="bold")

    ax.annotate("Ours shifts the whole distribution right →",
                xy=(0.97, 0.30), xycoords="axes fraction", fontsize=8.5,
                style="italic", color=COLORS["ours"], ha="right")

    ax.set_xlabel("Achieved Rate (Mbps)")
    ax.set_ylabel("CDF")
    ax.set_xlim(8, 300)
    ax.set_ylim(0, 1.0)
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="both", color="0.90", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, borderpad=0.5)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.11)
    return fig


def main():
    setup_style()
    data = load_rate_results()
    fig = plot_rate_cdf(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
