"""
figure06_energy_comparison.py

Figure 6 - Energy Consumption Across UAV Policies (companion to Figure 5).

Grouped bar chart of the energy breakdown (Flight / Hover / Communication /
Total) for the three policies. Message: energy is a COST, not the objective.
3D-GNN is the cheapest (it stays high and rarely descends) - which is exactly
why its critical-node QoS is the poorest (Fig. 5). ATOM-3D-VoI spends only a
modest increment over 2D-AUTO - split across Flight and Hover from its
selective descend-serve-ascend behaviour - and that increment buys the large
high-priority QoS gain of Fig. 5. Energy efficiency != QoS performance.

This figure deliberately matches Figure 5's visual style (legend order, bar
order, colors, spacing, typography) so the two read as companion figures.

DATA
----
All values come from ONE isolated function, `generate_placeholder_energy_data`.
The plotting code (`plot_energy_comparison`) is agnostic to the source: when
real results exist at `results_data/energy_breakdown.npz`, `load_energy_results`
loads them instead and the plotting code is unchanged. Placeholder values
reflect the project's smoke-test observations; no final metrics are fabricated.

Runs independently:  python figure06_energy_comparison.py
Exports results/figure06_energy_comparison.png (600 DPI) via common_plot.
Reuses locked infrastructure only (common_style, common_plot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common_style import setup_style, COLORS
from common_plot import save_figure


# =============================================================================
# CONSTANTS  (kept identical in spirit to figure05 for a companion look)
# =============================================================================

FIG_NAME = "figure06_energy_comparison"
TITLE = "Energy Consumption Across UAV Policies"
UNIT = "kJ"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "energy_breakdown.npz"

# The additive components (Total = sum of the three); shown as its own group.
COMPONENTS = ["flight", "hover", "comm"]
CATEGORIES = ["flight", "hover", "comm", "total"]
CATEGORY_LABELS = {
    "flight": "Flight\nEnergy",
    "hover": "Hover\nEnergy",
    "comm": "Communication\nEnergy",
    "total": "Total\nEnergy",
}

# Method order / labels / colors / emphasis - IDENTICAL to figure05.
METHOD_STYLE = [
    ("2d_auto", "2D-AUTO", "#9E9E9E", False),
    ("3d_gnn",  "3D-GNN",  "#FB8C00", False),
    ("atom3d",  "ATOM-3D-VoI (Ours)", COLORS["ours"], True),
]


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real results)
# =============================================================================

def generate_placeholder_energy_data():
    """Return PLACEHOLDER per-component energy (kJ) for each method.

    Schema (identical to the real results export):
        {
          "categories": ["flight","hover","comm","total"],
          "unit": "kJ",
          "values": {method_key: {category_key: kJ_float}},  # total = sum(comp)
          "placeholder": True,
        }

    Placeholder logic (from smoke-test observations):
      * 3D-GNN  - LOWEST flight, hover and total energy: it spends most of the
                  mission at relatively high altitude and performs few
                  purposeful descents. That saving is precisely why its
                  critical-node QoS is the lowest (Fig. 5).
      * 2D-AUTO - moderate total; fixed low-altitude operation with efficient
                  routing; good for ground sensors, weak on elevated critical ones.
      * ATOM-3D-VoI - total only slightly above 2D-AUTO and clearly above 3D-GNN;
                  the increment appears in BOTH flight and hover (selective
                  descend -> serve -> ascend). Communication stays similar.
    """
    components = {
        "2d_auto": {"flight": 40.0, "hover": 22.0, "comm": 8.0},
        "3d_gnn":  {"flight": 30.0, "hover": 14.0, "comm": 8.0},
        "atom3d":  {"flight": 43.0, "hover": 25.0, "comm": 8.0},
    }
    values = {}
    for m, comp in components.items():
        entry = dict(comp)
        entry["total"] = sum(comp[c] for c in COMPONENTS)
        values[m] = entry
    return {"categories": list(CATEGORIES), "unit": UNIT, "values": values,
            "placeholder": True}


def load_energy_results():
    """Load real energy results if present, else fall back to the placeholder.

    Real export: a NumPy archive at REAL_DATA_PATH with
        categories : (K,) str      (flight/hover/comm/total)
        methods    : (M,) str      (keys matching METHOD_STYLE)
        values     : (M, K) float  energy per component (kJ)
    reshaped into the same dict schema. Plotting is unchanged.
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH, allow_pickle=True)
        cats = [str(c) for c in z["categories"]]
        methods = [str(m) for m in z["methods"]]
        arr = np.asarray(z["values"], float)
        values = {m: {c: float(arr[i, j]) for j, c in enumerate(cats)}
                  for i, m in enumerate(methods)}
        return {"categories": cats, "unit": str(z["unit"]) if "unit" in z.files
                else UNIT, "values": values, "placeholder": False}
    return generate_placeholder_energy_data()


# =============================================================================
# PLOTTING  (source-agnostic; mirrors figure05 exactly)
# =============================================================================

def plot_energy_comparison(data):
    cats = data["categories"]
    values = data["values"]
    unit = data.get("unit", UNIT)

    x = np.arange(len(cats))
    n = len(METHOD_STYLE)
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    # Light highlight behind the Total group: the headline energy comparison.
    if "total" in cats:
        ti = cats.index("total")
        ax.axvspan(ti - 0.46, ti + 0.46, color=COLORS["ours"], alpha=0.05,
                   zorder=0)

    for i, (key, label, color, emph) in enumerate(METHOD_STYLE):
        offset = (i - (n - 1) / 2.0) * width
        heights = [values[key][c] for c in cats]
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

    # "Modest energy increase" brace over the Total group - the companion of
    # Figure 5's "Largest Improvement" brace (small cost <-> large QoS gain).
    if "total" in cats:
        ti = cats.index("total")
        x1, x2, yb, tick = ti - 0.40, ti + 0.40, 84.0, 3.0
        ax.plot([x1, x1, x2, x2], [yb - tick, yb, yb, yb - tick],
                color="0.3", lw=1.3, clip_on=False, zorder=5)
        ax.text(ti, yb + 2.0, "Modest energy increase", ha="center",
                va="bottom", fontsize=8.5, fontweight="bold", color="0.2",
                zorder=5)

    # Explicit note tying 3D-GNN's low energy to its low critical QoS (Fig. 5).
    ax.text(0.015, 0.93,
            "3D-GNN spends the least energy (stays high, rarely descends)\n"
            "→ which is why its critical-node QoS is lowest (see Fig. 5).",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            style="italic", color="0.35")

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats])
    ax.set_ylabel(f"Energy ({unit})")
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Energy Component")
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
    data = load_energy_results()
    fig = plot_energy_comparison(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
