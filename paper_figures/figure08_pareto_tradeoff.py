"""
figure08_pareto_tradeoff.py

Figure 8 - Energy vs. High-Priority Satisfaction Trade-off (reference panel (e)).

One curve per policy, each traced over a sweep of operating points (energy
budget / trade-off weight). Up-left is better. Message: at ANY energy level
ATOM-3D-VoI delivers more high-priority satisfaction - its curve dominates -
and its default operating point (76 kJ, 90 %) sits far above the baselines'
defaults (70 kJ, 76 %) and (52 kJ, 70 %), consistent with Figs. 5-6.

DATA
----
All values come from `generate_placeholder_pareto_data`; swap in real sweep
results via `load_pareto_results` (results_data/pareto_sweep.npz) with no
plotting changes.

Runs independently:  python figure08_pareto_tradeoff.py
Exports results/figure08_pareto_tradeoff.png (600 DPI) via common_plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from common_style import setup_style, COLORS
from common_plot import save_figure


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure08_pareto_tradeoff"
TITLE = "Energy vs. High-Priority Satisfaction Trade-off"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "pareto_sweep.npz"

METHOD_STYLE = [
    ("2d_auto", "2D-AUTO", "#9E9E9E", "o", False),
    ("3d_gnn", "3D-GNN", "#FB8C00", "s", False),
    ("atom3d", "ATOM-3D-VoI (Ours)", COLORS["ours"], "^", True),
]


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real sweep results)
# =============================================================================

def generate_placeholder_pareto_data():
    """PLACEHOLDER trade-off sweeps. Schema (same as the real export):
        {
          "curves": {m: {"energy": [...kJ], "satisfaction": [...%],
                          "default": (energy, satisfaction)}},
          "placeholder": True,
        }
    The "default" point of each method matches Figures 5-6 exactly:
    2D-AUTO (70, 76), 3D-GNN (52, 70), ATOM-3D-VoI (76, 90).
    """
    curves = {
        "2d_auto": {
            "energy": [55, 62, 70, 78, 88],
            "satisfaction": [58, 68, 76, 80, 82],
            "default": (70, 76),
        },
        "3d_gnn": {
            "energy": [40, 46, 52, 60, 70],
            "satisfaction": [52, 62, 70, 74, 76],
            "default": (52, 70),
        },
        "atom3d": {
            "energy": [58, 66, 76, 86, 96],
            "satisfaction": [72, 83, 90, 93, 94],
            "default": (76, 90),
        },
    }
    return {"curves": curves, "placeholder": True}


def load_pareto_results():
    """Load real sweep results if present, else the placeholder.

    Real export: results_data/pareto_sweep.npz with, per method key,
    `<m>_energy`, `<m>_satisfaction` (sweep arrays) and `<m>_default` (2,).
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH, allow_pickle=True)
        curves = {}
        for m, *_ in METHOD_STYLE:
            curves[m] = {
                "energy": list(np.asarray(z[f"{m}_energy"], float)),
                "satisfaction": list(np.asarray(z[f"{m}_satisfaction"], float)),
                "default": tuple(np.asarray(z[f"{m}_default"], float)),
            }
        return {"curves": curves, "placeholder": False}
    return generate_placeholder_pareto_data()


# =============================================================================
# PLOTTING  (source-agnostic)
# =============================================================================

def plot_pareto(data):
    curves = data["curves"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    for key, label, color, marker, emph in METHOD_STYLE:
        c = curves[key]
        ax.plot(c["energy"], c["satisfaction"], color=color,
                lw=2.4 if emph else 1.6, marker=marker, ms=6 if emph else 5,
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=0.5, label=label, zorder=6 if emph else 4)
        # Star the default operating point (the configuration of Figs. 5-6).
        ex, sy = c["default"]
        ax.scatter(ex, sy, marker="*", s=260 if emph else 180,
                   facecolor=color, edgecolor="black", linewidths=0.8,
                   zorder=7 if emph else 5)

    # Annotate Ours' default point.
    ax.annotate("default operating point\n(76 kJ, 90 %)",
                xy=(76, 90), xytext=(80, 78.5), fontsize=8,
                color=COLORS["ours"],
                arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=1.0))

    # "Better" direction cue (up-left).
    ax.annotate("", xy=(46, 92), xytext=(56, 82),
                arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4))
    ax.text(50.5, 86.2, "Better\n(↑ satisfaction, ↓ energy)", fontsize=8,
            color="0.35", ha="left", va="center")

    ax.set_xlabel("Total Energy (kJ)")
    ax.set_ylabel("High-Priority QoS Satisfaction (%)")
    ax.set_xlim(35, 100)
    ax.set_ylim(45, 100)
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.88", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mlines.Line2D([], [], marker="*", ls="none", markersize=11,
                                 markerfacecolor="white", markeredgecolor="black",
                                 label="default operating point"))
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              framealpha=0.92, borderpad=0.5, labelspacing=0.4)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.11)
    return fig


def main():
    setup_style()
    data = load_pareto_results()
    fig = plot_pareto(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
