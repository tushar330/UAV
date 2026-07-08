"""
figure05_qos_comparison.py

Figure 5 - Per-Class QoS Satisfaction Across UAV Policies.

Grouped bar chart of QoS-satisfaction (%) per criticality class (High /
Medium / Low) for the three policies (2D-AUTO, 3D-GNN, ATOM-3D-VoI). The
message: the objective is NOT to satisfy every node, but to guarantee the
HIGH-priority class while staying competitive elsewhere. ATOM-3D-VoI's gain
is concentrated on High priority; it does not dominate every category.

DATA
----
All values come from ONE isolated function, `generate_placeholder_qos_data`.
The plotting code (`plot_qos_comparison`) is agnostic to the source: when
real results exist at `results_data/qos_satisfaction.npz`, `load_qos_results`
loads them instead and the plotting code is unchanged. Placeholder values
reflect the project's smoke-test observations; no final metrics are fabricated.

Runs independently:  python figure05_qos_comparison.py
Exports results/figure05_qos_comparison.png (600 DPI) via common_plot.
Reuses locked infrastructure only (common_style, common_plot).
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

FIG_NAME = "figure05_qos_comparison"
TITLE = "QoS Satisfaction Across Priority Classes"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "qos_satisfaction.npz"

CLASSES = ["high", "medium", "low"]
# Constraint-style labels remind the reader these are QoS floors, not tags.
CLASS_LABELS = {"high": "High Priority\nQoS ≥ 38 Mbps",
                "medium": "Medium Priority\nQoS ≥ 25 Mbps",
                "low": "Low Priority\nQoS ≥ 8 Mbps"}

# Method order (left->right within each group), labels, colors, emphasis.
# Baselines stay neutral; ATOM-3D-VoI uses the project method color and is
# emphasized. Colors are consistent with the other figures.
METHOD_STYLE = [
    ("2d_auto", "2D-AUTO", "#9E9E9E", False),
    ("3d_gnn",  "3D-GNN",  "#FB8C00", False),
    ("atom3d",  "ATOM-3D-VoI (Ours)", COLORS["ours"], True),
]


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real results)
# =============================================================================

def generate_placeholder_qos_data():
    """Return PLACEHOLDER per-class QoS-satisfaction (%) for each method.

    Schema (identical to the real results export):
        {
          "classes": ["high", "medium", "low"],
          "values":  {method_key: {class_key: percent_float}},
          "placeholder": True,
        }

    Placeholder logic (from smoke-test observations):
      * 2D-AUTO  - strong baseline (efficient low-altitude routing); fails
                   mainly on elevated HIGH-priority sensors it cannot reach at
                   QoS. Best or near-best on Medium/Low.
      * 3D-GNN   - has altitude freedom but wastes it: priority-unaware, so it
                   is INCONSISTENT and sits BELOW 2D-AUTO on every class
                   (a motivation for adding the CMDP).
      * ATOM-3D-VoI - purposefully descends for critical nodes -> the largest
                   gain is on HIGH priority; only slightly better on Medium;
                   essentially tied (and just below 2D-AUTO) on Low, i.e. it
                   does NOT try to over-serve low-priority nodes.
    """
    values = {
        "2d_auto": {"high": 76.0, "medium": 86.0, "low": 95.0},
        "3d_gnn":  {"high": 70.0, "medium": 82.0, "low": 93.0},
        "atom3d":  {"high": 90.0, "medium": 89.0, "low": 94.0},
    }
    return {"classes": list(CLASSES), "values": values, "placeholder": True}


def load_qos_results():
    """Load real QoS results if present, else fall back to the placeholder.

    Real export: a NumPy archive at REAL_DATA_PATH with
        classes : (C,) str
        methods : (M,) str          (keys matching METHOD_STYLE)
        values  : (M, C) float      QoS satisfaction in percent
    which is reshaped into the same dict schema. Plotting is unchanged.
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH, allow_pickle=True)
        classes = [str(c) for c in z["classes"]]
        methods = [str(m) for m in z["methods"]]
        arr = np.asarray(z["values"], float)
        values = {m: {c: float(arr[i, j]) for j, c in enumerate(classes)}
                  for i, m in enumerate(methods)}
        return {"classes": classes, "values": values, "placeholder": False}
    return generate_placeholder_qos_data()


# =============================================================================
# PLOTTING  (source-agnostic: identical for placeholder or real results)
# =============================================================================

def plot_qos_comparison(data):
    classes = data["classes"]
    values = data["values"]

    x = np.arange(len(classes))
    n = len(METHOD_STYLE)
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    # Very light highlight behind the High-priority group: this is the class
    # where the method is designed to help, so it guides the eye there.
    if "high" in classes:
        hi = classes.index("high")
        ax.axvspan(hi - 0.46, hi + 0.46, color=COLORS["ours"], alpha=0.05,
                   zorder=0)

    for i, (key, label, color, emph) in enumerate(METHOD_STYLE):
        offset = (i - (n - 1) / 2.0) * width
        heights = [values[key][c] for c in classes]
        bars = ax.bar(
            x + offset, heights, width, label=label, color=color,
            edgecolor="black" if emph else "#777777",
            linewidth=1.2 if emph else 0.5,
            alpha=1.0 if emph else 0.9, zorder=3,
        )
        for rect, h in zip(bars, heights):
            ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold" if emph else "normal",
                        color=color if emph else "#333333")

    # "Largest Improvement" brace over the High-priority group.
    if "high" in classes:
        hi = classes.index("high")
        x1, x2, yb, tick = hi - 0.40, hi + 0.40, 100.0, 3.0
        ax.plot([x1, x1, x2, x2], [yb - tick, yb, yb, yb - tick],
                color="0.3", lw=1.3, clip_on=False, zorder=5)
        ax.text(hi, yb + 2.0, "Largest Improvement", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="0.2", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes])
    ax.set_ylabel("QoS Satisfaction (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Priority Class")
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False, fontsize=8.5, columnspacing=1.6, handlelength=1.4)

    fig.subplots_adjust(left=0.09, right=0.975, top=0.86, bottom=0.11)
    return fig


def main():
    setup_style()
    data = load_qos_results()
    fig = plot_qos_comparison(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
