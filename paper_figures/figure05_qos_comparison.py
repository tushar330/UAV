"""
figure05_qos_comparison.py

Figure 5 - Per-Class QoS Satisfaction Across UAV Policies.

Grouped bar chart of QoS-satisfaction (%) per criticality class (High /
Medium / Low) for the four policies (2D-AUTO, Two-Stage, Coupled-Greedy,
ATOM-3D-VoI). The
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
from common_plot import (save_figure, apply_labels, budget_footnote,
                         class_floors_mbps, present_methods)


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure05_qos_comparison"
TITLE = "QoS Satisfaction Across Priority Classes"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "qos_satisfaction.npz"
# Same export, richer: carries the 95% CI and the paired-test p-values that
# qos_satisfaction.npz does not store.
METRICS_PATH = Path(__file__).resolve().parent / "results_data" / "overall_metrics.npz"

# Significance thresholds for the paired test against the 2D baseline.
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
NS_MARK = "n.s."
REFERENCE_METHOD = "2d_auto"
OURS_METHOD = "atom3d"

CLASSES = ["high", "medium", "low"]
# Constraint-style labels remind the reader these are QoS floors, not tags.
CLASS_TITLES = {"high": "High Priority", "medium": "Medium Priority",
                "low": "Low Priority"}
# Fallback floors (Mbps) for placeholder mode; the real ones come from the
# export, so a recalibration cannot leave this axis captioned with stale values.
DEFAULT_FLOORS_MBPS = {"high": 38.0, "medium": 32.5, "low": 29.0}


def class_labels():
    floors = {**DEFAULT_FLOORS_MBPS, **class_floors_mbps()}
    return {c: f"{CLASS_TITLES[c]}\nQoS ≥ {floors[c]:g} Mbps" for c in CLASS_TITLES}

# Method order (left->right within each group), labels, colors, emphasis.
# Baselines stay neutral; ATOM-3D-VoI uses the project method color and is
# emphasized. Colors are consistent with the other figures.
# Colors encode the roles: baselines neutral (grey/orange), the proposed method
# in the project blue, and its ablation in a light tint of that same blue so a
# reader sees at a glance that Coupled-Greedy belongs to OUR family and is not
# an independent competitor.
METHOD_STYLE = [
    ("2d_auto",        "2D-AUTO",              "#9E9E9E", False),
    ("two_stage",      "Two-Stage (decoupled)", "#FB8C00", False),
    ("coupled_greedy", "Coupled-Greedy (ablation)", "#64B5F6", False),
    ("atom3d",         "ATOM-3D-VoI (Ours)",   COLORS["ours"], True),
]

# Labels describe whatever actually produced the numbers on disk (the
# exporter's results_data/labels.json); unchanged in placeholder mode.
# This figure pools every city seed, so 2D-AUTO is labelled with the altitude
# range it actually flew across them, not the canonical scene's single value.
METHOD_STYLE = apply_labels(METHOD_STYLE, aggregate=True)


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
      * Two-Stage - places hovers QoS-blind, then repairs violations from
                   below. Recovers much of the critical class but pays for the
                   repair pass, so it trails the coupled family.
      * Coupled-Greedy - the proposed coupling WITHOUT the local search; sits
                   between Two-Stage and the full method, which is what makes
                   it an ablation rather than a baseline.
      * ATOM-3D-VoI - purposefully descends for critical nodes -> the largest
                   gain is on HIGH priority; only slightly better on Medium;
                   essentially tied (and just below 2D-AUTO) on Low, i.e. it
                   does NOT try to over-serve low-priority nodes.
    """
    values = {
        "2d_auto":        {"high": 76.0, "medium": 86.0, "low": 95.0},
        "two_stage":      {"high": 82.0, "medium": 87.0, "low": 94.0},
        "coupled_greedy": {"high": 87.0, "medium": 88.0, "low": 94.0},
        "atom3d":         {"high": 90.0, "medium": 89.0, "low": 94.0},
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
        return {"classes": classes, "values": values, "placeholder": False,
                **load_qos_uncertainty(classes)}
    return generate_placeholder_qos_data()


def load_qos_uncertainty(classes):
    """Per-class 95% CI and paired-test p-values from the overall-metrics export.

    ``qos_satisfaction.npz`` carries only point estimates. Drawn without spread,
    a 76-vs-62 bar pair reads as a settled win when the paired test may not
    support it, so the same export's CI/p-values are loaded here and rendered as
    error bars plus significance marks. Returns empty dicts when the export
    predates these fields, in which case the bars are drawn plain.

    Kept local to this figure: FILE_DEPENDENCIES only lifts a helper into
    common_plot.py once at least two figures need it (Fig. 13 reads the archive
    directly for its own table layout).
    """
    if not METRICS_PATH.exists():
        return {"ci": {}, "pvalues": {}, "n_seeds": 0}
    z = np.load(METRICS_PATH, allow_pickle=True)
    keys = set(z.files)
    methods = [str(m) for m in z["methods"]]
    metrics = [str(k) for k in z["metrics"]]
    cis = np.asarray(z["ci95"], float)
    pvals = (np.asarray(z["pvalues"], float) if "pvalues" in keys
             else np.full(cis.shape, np.nan))
    n_seeds = int(z["n_seeds"]) if "n_seeds" in keys else 0

    ci = {m: {c: float(cis[i, metrics.index(c)])
              for c in classes if c in metrics}
          for i, m in enumerate(methods)}
    pv = {m: {c: float(pvals[i, metrics.index(c)])
              for c in classes if c in metrics}
          for i, m in enumerate(methods)}
    return {"ci": ci, "pvalues": pv, "n_seeds": n_seeds}


def sig_mark(p):
    """Significance marker for a paired-test p-value ('' when unavailable)."""
    if p is None or not np.isfinite(p):
        return ""
    for thresh, mark in SIG_LEVELS:
        if p < thresh:
            return mark
    return NS_MARK


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

    # Only plot methods this export actually produced.
    style = present_methods(METHOD_STYLE, values)
    n = len(style)
    ci = data.get("ci", {})
    for i, (key, label, color, emph) in enumerate(style):
        offset = (i - (n - 1) / 2.0) * width
        heights = [values[key][c] for c in classes]
        # Error bars whenever the export carries a real across-seed spread; a
        # single realization has none, and inventing one would be fabrication.
        errs = [ci.get(key, {}).get(c, 0.0) for c in classes]
        yerr = errs if any(e > 0 for e in errs) else None
        bars = ax.bar(
            x + offset, heights, width, label=label, color=color,
            edgecolor="black" if emph else "#777777",
            linewidth=1.2 if emph else 0.5,
            alpha=1.0 if emph else 0.9, zorder=3,
            yerr=yerr, capsize=3,
            error_kw={"ecolor": "0.25", "elinewidth": 0.9, "capthick": 0.9,
                      "zorder": 4},
        )
        for rect, h, e in zip(bars, heights, errs):
            ax.annotate(f"{h:.0f}", xy=(rect.get_x() + rect.get_width() / 2,
                                        h + (e if yerr else 0.0)),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold" if emph else "normal",
                        color=color if emph else "#333333")

    # Per-class significance of ours vs the 2D baseline, printed above each
    # group. Without these the eye reads every taller blue bar as a win; on this
    # data only the High class separates, while Medium/Low are a trend and a
    # tie. The figure must not imply a result the paired test does not support.
    pvalues = data.get("pvalues", {}).get(OURS_METHOD, {})
    peak = max(values[key][c] + ci.get(key, {}).get(c, 0.0)
               for key, *_ in style for c in classes)
    for j, c in enumerate(classes):
        mark = sig_mark(pvalues.get(c))
        if not mark:
            continue
        strong = mark != NS_MARK
        ax.text(x[j], peak + 4.0, mark, ha="center", va="bottom",
                fontsize=10 if strong else 8,
                fontweight="bold" if strong else "normal",
                color="0.15" if strong else "0.45", zorder=5)

    # Brace over the class with the largest improvement, labelled with the
    # measured gain rather than the bare word "Largest": the phrasing used to
    # imply a win on every class, which this data does not support.
    if OURS_METHOD in values and REFERENCE_METHOD in values:
        gains = {c: values[OURS_METHOD][c] - values[REFERENCE_METHOD][c]
                 for c in classes}
        best_c = max(gains, key=gains.get)
        bi = classes.index(best_c)
        x1, x2, yb, tick = bi - 0.40, bi + 0.40, peak + 12.0, 3.0
        ax.plot([x1, x1, x2, x2], [yb - tick, yb, yb, yb - tick],
                color="0.3", lw=1.3, clip_on=False, zorder=5)
        ax.text(bi, yb + 1.5, f"+{gains[best_c]:.0f} pts vs 2D",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="0.2", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([class_labels()[c] for c in classes])
    ax.set_ylabel("QoS Satisfaction (%)")
    # Headroom for the brace and its caption, without leaving dead space when
    # nothing reaches 100%.
    ax.set_ylim(0, max(peak + 24.0, 105.0))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Priority Class")
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              frameon=False, fontsize=8.5, columnspacing=1.6, handlelength=1.4)

    # A QoS percentage means different things in different regimes: under
    # serve-all with hard QoS constraints it is 100% by construction, under a
    # budget it reports how far the mission got. State which one this is.
    lines = [budget_footnote()]
    n_seeds = int(data.get("n_seeds") or 0)
    if any(sig_mark(p) for p in pvalues.values()):
        stat = ("Paired t-test vs 2D-AUTO: *** p<0.001, ** p<0.01, * p<0.05, "
                "n.s. not significant.")
        if n_seeds:
            stat += f" Bars are mean ± Student-t 95% CI over n = {n_seeds} seeds."
        lines.append(stat)
    footnote = "\n".join(t for t in lines if t)
    if footnote:
        ax.text(0.5, -0.155, footnote, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.5, style="italic", color="0.4")

    fig.subplots_adjust(left=0.09, right=0.975, top=0.86, bottom=0.17)
    return fig


def main():
    setup_style()
    data = load_qos_results()
    fig = plot_qos_comparison(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
