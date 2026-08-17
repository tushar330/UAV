"""
figure06_energy_comparison.py

Figure 6 - Energy Consumption Across UAV Policies (companion to Figure 5).

Grouped bar chart of the energy breakdown (Flight / Hover / Communication /
Total) for the three policies. Message: energy is a COST, not the objective,
so this figure must always be read together with the critical-QoS result of
Figure 5 - the cheapest policy is only the better policy if it also meets the
critical floors.

The in-plot note that states which way that comparison came out is DERIVED
from the data at draw time (see `qos_note`), never hardcoded. An earlier
hardcoded version asserted the cheapest method had the poorest critical QoS;
once real results replaced the placeholders that sentence became false, and
the figure silently shipped a claim its own data contradicted.

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
from common_plot import save_figure, apply_labels, run_regime, present_methods


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
    ("2d_auto",        "2D-AUTO",                   "#9E9E9E", False),
    ("two_stage",      "Two-Stage (decoupled)",     "#FB8C00", False),
    ("coupled_greedy", "Coupled-Greedy (ablation)", "#64B5F6", False),
    ("atom3d",         "ATOM-3D-VoI (Ours)", COLORS["ours"], True),
]

# Labels describe whatever actually produced the numbers on disk (the
# exporter's results_data/labels.json); unchanged in placeholder mode.
METHOD_STYLE = apply_labels(METHOD_STYLE)


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
      * 2D-AUTO - moderate total; fixed low-altitude operation with efficient
                  routing; good for ground sensors, weak on elevated critical ones.
      * Two-Stage - pays twice: a high QoS-blind cover, then a second pass of
                  shallow repair hovers, so hover energy is the highest.
      * Coupled-Greedy - the coupling avoids the repair pass; cheaper than
                  Two-Stage without the local search's refinement.
      * ATOM-3D-VoI - total close to 2D-AUTO; the increment appears in BOTH
                  flight and hover (selective descend -> serve -> ascend).
                  Communication stays similar.

    Under the budget regime these totals are all near B by construction; the
    figure is read for the SPLIT between components, not for a winner.
    """
    components = {
        "2d_auto":        {"flight": 40.0, "hover": 22.0, "comm": 8.0},
        "two_stage":      {"flight": 38.0, "hover": 28.0, "comm": 8.0},
        "coupled_greedy": {"flight": 41.0, "hover": 24.0, "comm": 8.0},
        "atom3d":         {"flight": 43.0, "hover": 25.0, "comm": 8.0},
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


# High-priority QoS per method, used ONLY to phrase the in-plot note (the bars
# themselves never depend on it). Read from the same results_data archive that
# Figure 5 plots, so the two figures cannot disagree about which way the
# energy/QoS comparison came out.
QOS_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "qos_satisfaction.npz"


def load_high_qos():
    """{method_key: high-priority QoS %} from disk, or None if unavailable.

    Returning None is a normal outcome (placeholder mode, or an export that
    predates the QoS archive); the caller falls back to a note that claims
    nothing about QoS.
    """
    if not QOS_DATA_PATH.exists():
        return None
    z = np.load(QOS_DATA_PATH, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    if "high" not in classes:
        return None
    col = classes.index("high")
    arr = np.asarray(z["values"], float)
    return {str(m): float(arr[i, col]) for i, m in enumerate(z["methods"])}


def qos_note(values, high_qos):
    """Phrase the energy-vs-QoS note from the data, never from an assumption.

    Energy efficiency and QoS can come out either way, so the sentence is
    chosen by comparing them: whichever method is cheapest, the note reports
    what that actually bought or cost in critical-node QoS.
    """
    # Under an equal-budget comparison "who is cheapest" is not the story -
    # every method is given the same B, so the totals are equal by design and
    # the only thing that differs is the QoS that budget buys.
    budget = run_regime().get("energy_budget_kj")
    if budget and high_qos and {"atom3d", "2d_auto"} <= set(high_qos):
        totals = [values[k]["total"] for k, *_ in present_methods(METHOD_STYLE, values)]
        spread = (max(totals) - min(totals)) / max(max(totals), 1e-9) * 100.0
        ours = dict((k, lbl) for k, lbl, *_ in METHOD_STYLE)["atom3d"]
        return (f"Equal budget B = {float(budget):.0f} kJ: all methods spend within "
                f"{spread:.0f}% of each other.\n"
                f"At that energy {ours} meets {high_qos['atom3d']:.0f}% of critical-node "
                f"QoS vs {high_qos['2d_auto']:.0f}% for 2D-AUTO (Fig. 5).")

    cheapest_key, cheapest_label = min(
        ((k, lbl) for k, lbl, *_ in METHOD_STYLE), key=lambda m: values[m[0]]["total"])
    if not high_qos or cheapest_key not in high_qos:
        return (f"{cheapest_label} spends the least total energy.\n"
                "Read against Fig. 5 for what that costs in critical-node QoS.")
    best_key = max(high_qos, key=lambda k: high_qos[k])
    cheapest_qos = high_qos[cheapest_key]
    leads_on_qos = cheapest_qos >= high_qos[best_key] - 1e-9
    # Leading on QoS is not the same as satisfying it: a method can top the
    # field at 12% and still meet no critical floor. Only the first branch may
    # claim there is no trade-off.
    if leads_on_qos and cheapest_qos >= 99.0:
        return (f"{cheapest_label} spends the least energy AND meets every\n"
                f"critical floor ({cheapest_qos:.0f}%, Fig. 5) - there is no trade-off here.")
    if leads_on_qos:
        return (f"{cheapest_label} spends the least energy and leads on critical-node\n"
                f"QoS, but still meets only {cheapest_qos:.0f}% of it (Fig. 5).")
    return (f"{cheapest_label} spends the least energy (stays high, rarely descends)\n"
            f"→ but meets only {cheapest_qos:.0f}% of critical-node QoS (see Fig. 5).")


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

    style = present_methods(METHOD_STYLE, values)
    n = len(style)
    for i, (key, label, color, emph) in enumerate(style):
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

    # Axis range follows the data: real energies differ in magnitude from the
    # placeholder set, so a fixed limit would clip the tallest bars.
    peak = max(values[key][c] for key, *_ in style for c in cats)
    top = peak * 1.28                      # headroom for the brace + labels
    brace_y = peak * 1.10

    # Brace over the Total group. Under the budget regime energy is a CONTROLLED
    # VARIABLE, not an outcome: every method flies until the same B is spent, so
    # the residual gap is budget granularity (a hover cannot be flown in part),
    # not an energy saving. An earlier version read "3% less energy than 2D",
    # which invited the reader to treat a non-significant by-construction
    # difference as a result. State the control instead - it is what makes the
    # QoS comparison in Figure 5 a fair one.
    if "total" in cats and {"atom3d", "2d_auto"} <= set(values):
        ti = cats.index("total")
        ours, base = values["atom3d"]["total"], values["2d_auto"]["total"]
        delta = abs(ours - base) / base * 100.0 if base else 0.0
        verdict = f"equal budget: within {delta:.0f}% of 2D"
        x1, x2, tick = ti - 0.40, ti + 0.40, top * 0.033
        ax.plot([x1, x1, x2, x2],
                [brace_y - tick, brace_y, brace_y, brace_y - tick],
                color="0.3", lw=1.3, clip_on=False, zorder=5)
        ax.text(ti, brace_y + top * 0.02, verdict, ha="center",
                va="bottom", fontsize=8.5, fontweight="bold", color="0.2",
                zorder=5)

    # Note tying the cheapest method to its critical-QoS outcome (Fig. 5),
    # phrased from the data so it stays true whichever way the comparison goes.
    ax.text(0.015, 0.97, qos_note(values, load_high_qos()),
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            style="italic", color="0.35")

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in cats])
    ax.set_ylabel(f"Energy ({unit})")
    ax.set_ylim(0, top)
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
