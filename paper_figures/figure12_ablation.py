"""
figure12_ablation.py

Figure 12 - Ablation of the ATOM-3D-VoI Components (reference panel (j)).

Bar chart of high-priority QoS satisfaction as the method's components are
added back one at a time. Message: each ingredient contributes, and the
learned altitude + CMDP combination provides the largest gain - the full
method reaches the 90 % of Figure 5.

DATA
----
Values come from `generate_placeholder_ablation_data`; swap in real ablation
results via `load_ablation_results` (results_data/ablation.npz) with no
plotting changes.

Runs independently:  python figure12_ablation.py
Exports results/figure12_ablation.png (600 DPI) via common_plot.
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

FIG_NAME = "figure12_ablation"
TITLE = "Ablation: Contribution of Each Component"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "ablation.npz"

# Variant order (left -> right = components added back), label, emphasized?
VARIANTS = [
    ("fixed_alt", "Fixed\nAltitude", False),
    ("learned_alt", "+ Learned\nAltitude", False),
    ("cmdp_uniform", "+ CMDP Duals\n(uniform weights)", False),
    ("full", "ATOM-3D-VoI\n(full: + VoI weights)", True),
]

NEUTRAL_COLOR = "#9E9E9E"


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real ablation results)
# =============================================================================

def generate_placeholder_ablation_data():
    """PLACEHOLDER high-priority QoS satisfaction (%) per ablation variant.

    Schema: {"values": {variant_key: percent}, "placeholder": True}
    Story: a fixed-altitude policy misses elevated critical nodes (71); a
    learned altitude head helps but is not priority-directed (79); adding the
    CMDP duals makes descents constraint-driven (84); the full VoI-weighted
    objective concentrates them on the critical class (90 - matches Fig. 5).
    """
    return {
        "values": {
            "fixed_alt": 71.0,
            "learned_alt": 79.0,
            "cmdp_uniform": 84.0,
            "full": 90.0,
        },
        "placeholder": True,
    }


def load_ablation_results():
    """Load real ablation results if present, else the placeholder.

    Real export: results_data/ablation.npz with `variants` (K,) str and
    `values` (K,) float (high-priority QoS satisfaction, percent).
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH, allow_pickle=True)
        variants = [str(v) for v in z["variants"]]
        vals = np.asarray(z["values"], float)
        return {"values": dict(zip(variants, vals)), "placeholder": False}
    return generate_placeholder_ablation_data()


# =============================================================================
# PLOTTING  (source-agnostic)
# =============================================================================

def plot_ablation(data):
    values = data["values"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    x = np.arange(len(VARIANTS))
    heights = [values[k] for k, *_ in VARIANTS]
    colors = [COLORS["ours"] if emph else NEUTRAL_COLOR
              for _, _, emph in VARIANTS]

    bars = ax.bar(x, heights, width=0.6, color=colors,
                  edgecolor=["black" if emph else "#777777"
                             for _, _, emph in VARIANTS],
                  linewidth=[1.2 if emph else 0.5 for _, _, emph in VARIANTS],
                  zorder=3)
    for rect, h, (_, _, emph) in zip(bars, heights, VARIANTS):
        ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold" if emph else "normal",
                    color=COLORS["ours"] if emph else "#333333")

    # Step arrows showing the incremental gains between variants.
    for i in range(len(x) - 1):
        gain = heights[i + 1] - heights[i]
        ax.annotate(f"+{gain:.0f}", xy=((x[i] + x[i + 1]) / 2,
                                        max(heights[i], heights[i + 1]) + 4.5),
                    ha="center", fontsize=8, color="0.35", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in VARIANTS], fontsize=9)
    ax.set_ylabel("High-Priority QoS Satisfaction (%)")
    ax.set_ylim(0, 100)
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.text(0.02, 0.965, "components added left → right",
            transform=ax.transAxes, fontsize=8, style="italic", color="0.4",
            va="top")

    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.13)
    return fig


def main():
    setup_style()
    data = load_ablation_results()
    fig = plot_ablation(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
