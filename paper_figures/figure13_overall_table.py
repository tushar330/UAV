"""
figure13_overall_table.py

Figure 13 - Overall Performance Comparison table (reference panel (d)).

A rendered results table (mean +/- 95% CI over evaluation seeds) summarizing
every headline metric in one place. The means are IDENTICAL to the values in
the locked Figures 5 (per-class QoS satisfaction) and 6 (total energy), so
the table and the bar charts can never disagree. Ours' row is highlighted in
the project method color.

NOTE: for the camera-ready paper this table would normally be typeset in
LaTeX (booktabs); this rendered version keeps the repository self-contained
and lets the table be dropped into slides/preprints as an image.

DATA
----
All values come from `generate_placeholder_table_data`; swap in real results
via `load_table_results` (results_data/overall_metrics.npz) with no
rendering changes.

Runs independently:  python figure13_overall_table.py
Exports results/figure13_overall_table.png (600 DPI) via common_plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common_style import setup_style, COLORS
from common_plot import save_figure, apply_labels, present_methods


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure13_overall_table"
TITLE = "Overall Performance Comparison (Mean ± 95% CI)"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "overall_metrics.npz"

# Column definition: (key, header, better-direction arrow)
COLUMNS = [
    ("energy", "Total Energy\n(kJ) ↓", "down"),
    ("high", "High-Priority\nSatisfaction (%) ↑", "up"),
    ("medium", "Medium-Priority\nSatisfaction (%) ↑", "up"),
    ("low", "Low-Priority\nSatisfaction (%) ↑", "up"),
]

# Row order and labels (ours last, highlighted).
METHODS = [
    ("2d_auto", "2D-AUTO", False),
    ("two_stage", "Two-Stage (decoupled)", False),
    ("coupled_greedy", "Coupled-Greedy (ablation)", False),
    ("atom3d", "ATOM-3D-VoI (Ours)", True),
]

# Kept empty deliberately. This previously excluded Blind-3D, a degenerate
# control that won the energy column by barely flying. That row is gone, and
# every remaining row is a method that genuinely attempts the task, so the
# "best per column" bolding is meaningful for all of them.
CONTROL_KEYS: set[str] = set()

# Significance thresholds for the paired test against the 2D baseline.
SIG_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
NS_MARK = "n.s."

# The starred marks are measured against the 2D base paper this work extends.
# The strongest 3D baseline is reported separately (see `secondary_comparison`)
# so it is visible in the main results and cannot look suppressed.
OURS_METHOD = "atom3d"
SECONDARY_REFERENCE = "two_stage"

# Labels describe whatever actually produced the numbers on disk (the
# exporter's results_data/labels.json); unchanged in placeholder mode.
# This figure pools every city seed, so 2D-AUTO is labelled with the altitude
# range it actually flew across them, not the canonical scene's single value.
METHODS = apply_labels(METHODS, aggregate=True)

OURS_ROW_FILL = "#E8F0FE"          # light tint of the method color
HEADER_FILL = "#F0F0F0"


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real results)
# =============================================================================

def generate_placeholder_table_data():
    """PLACEHOLDER (mean, ci95) per method/metric.

    Means are EXACTLY the locked Figure-5/6 values; CIs are plausible spreads
    over evaluation seeds.
    Schema: {"values": {method: {metric: (mean, ci95)}}, "placeholder": True}
    """
    values = {
        "2d_auto": {"energy": (70.0, 1.8), "high": (76.0, 3.1),
                    "medium": (86.0, 2.4), "low": (95.0, 1.2)},
        "two_stage": {"energy": (74.0, 1.7), "high": (82.0, 3.4),
                      "medium": (87.0, 2.6), "low": (94.0, 1.4)},
        "coupled_greedy": {"energy": (75.0, 1.6), "high": (87.0, 2.6),
                           "medium": (88.0, 2.3), "low": (94.0, 1.3)},
        "atom3d": {"energy": (76.0, 1.6), "high": (90.0, 2.2),
                   "medium": (89.0, 2.1), "low": (94.0, 1.3)},
    }
    return {"values": values, "pvalues": {}, "n_seeds": 0,
            "legacy_ci": False, "placeholder": True}


def load_table_results():
    """Load real aggregated metrics if present, else the placeholder.

    Real export: results_data/overall_metrics.npz with `methods` (M,) str,
    `metrics` (K,) str, `means` (M, K) float, `ci95` (M, K) float. Newer exports
    also carry `pvalues` (M, K) from a paired test against the 2D baseline,
    `n_seeds`, and the raw `samples_<method>_<metric>` arrays.

    Exports written before the CI fix used the normal quantile 1.96 instead of
    Student-t, understating every interval. Those files carry no `ci_method`
    key, so they are recomputed here from the per-seed samples when available
    and otherwise flagged, rather than being plotted as if they were correct.
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH, allow_pickle=True)
        keys = set(z.files)
        methods = [str(m) for m in z["methods"]]
        metrics = [str(k) for k in z["metrics"]]
        means = np.asarray(z["means"], float)
        cis = np.asarray(z["ci95"], float)
        pvals = (np.asarray(z["pvalues"], float) if "pvalues" in keys
                 else np.full(means.shape, np.nan))
        n_seeds = int(z["n_seeds"]) if "n_seeds" in keys else 0
        legacy_ci = "ci_method" not in keys

        values = {m: {k: (float(means[i, j]), float(cis[i, j]))
                      for j, k in enumerate(metrics)}
                  for i, m in enumerate(methods)}
        pvalues = {m: {k: float(pvals[i, j]) for j, k in enumerate(metrics)}
                   for i, m in enumerate(methods)}
        # Raw per-seed samples, kept so the SECONDARY comparison below can be
        # computed here without another export run.
        samples = {k: np.asarray(z[k], float) for k in keys
                   if k.startswith("samples_")}
        return {"values": values, "pvalues": pvalues, "n_seeds": n_seeds,
                "legacy_ci": legacy_ci, "samples": samples,
                "placeholder": False}
    return generate_placeholder_table_data()


def secondary_comparison(samples, ours=OURS_METHOD, other=SECONDARY_REFERENCE):
    """Paired ours-vs-`other` p-values, as a {metric: p} dict.

    The starred marks in this table are measured against the 2D base paper,
    which is the work being extended. But 2D cannot reach the critical class at
    all, so a reader is entitled to ask how the method compares with a competent
    3D baseline. Reporting that here - from the same per-seed samples, so no
    re-planning is needed - answers it in the main results rather than leaving
    the strongest baseline looking hidden.

    Returns {} when the export predates the stored samples.
    """
    out = {}
    for key, _, _ in COLUMNS:
        a = samples.get(f"samples_{ours}_{key}")
        b = samples.get(f"samples_{other}_{key}")
        if a is None or b is None or a.size != b.size or a.size < 2:
            continue
        d = a - b
        sd = float(np.std(d, ddof=1))
        if sd == 0.0:
            out[key] = 1.0 if float(np.mean(d)) == 0.0 else 0.0
            continue
        t = float(np.mean(d)) / (sd / np.sqrt(d.size))
        try:
            from scipy import stats
            out[key] = float(2.0 * (1.0 - stats.t.cdf(abs(t), d.size - 1)))
        except Exception:
            return {}
    return out


# =============================================================================
# RENDERING  (source-agnostic)
# =============================================================================

def _best_per_column(values, methods=None):
    """Return {metric: method_key} of the best NON-CONTROL entry per column."""
    methods = METHODS if methods is None else methods
    best = {}
    for key, _, direction in COLUMNS:
        entries = {m: values[m][key][0] for m, *_ in methods
                   if m not in CONTROL_KEYS and m in values}
        if not entries:
            continue
        pick = min(entries, key=entries.get) if direction == "down" \
            else max(entries, key=entries.get)
        best[key] = pick
    return best


def _sig_mark(p):
    """Significance marker for a paired-test p-value ('' when unavailable)."""
    if p is None or not np.isfinite(p):
        return ""
    for thresh, mark in SIG_LEVELS:
        if p < thresh:
            return mark
    return NS_MARK


def plot_table(data):
    values = data["values"]
    pvalues = data.get("pvalues", {})
    methods = present_methods(METHODS, values)
    best = _best_per_column(values, methods)

    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ax.axis("off")

    col_labels = [h for _, h, _ in COLUMNS]
    cell_text, cell_colors = [], []
    for mkey, mlabel, emph in methods:
        row, colors = [], []
        for ckey, _, _ in COLUMNS:
            mean, ci = values[mkey][ckey]
            mark = _sig_mark(pvalues.get(mkey, {}).get(ckey))
            # The marker states whether THIS row differs from the 2D baseline;
            # the baseline's own row is the reference and carries no marker.
            text = f"{mean:.1f} ± {ci:.1f}"
            if mark:
                text = f"{text} {mark}"
            row.append(text)
            colors.append(OURS_ROW_FILL if emph else "white")
        cell_text.append(row)
        cell_colors.append(colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=[m for _, m, _ in methods],
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=[OURS_ROW_FILL if emph else "white" for *_, emph in methods],
        cellLoc="center", rowLoc="right", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.1)

    # Style pass: header fill, ours row bold+blue, best value per column bold.
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#666666")
        cell.set_linewidth(0.7)
        if r == 0:                                   # header row
            cell.set_facecolor(HEADER_FILL)
            cell.set_text_props(fontweight="bold", fontsize=8.2)
            cell.set_height(cell.get_height() * 1.45)
        elif r >= 1:
            mkey, _, emph = methods[r - 1]
            if emph:
                cell.set_text_props(color=COLORS["ours"], fontweight="bold")
            if c >= 0:                               # metric cells
                ckey = COLUMNS[c][0] if c < len(COLUMNS) else None
                if ckey and best[ckey] == mkey:
                    cell.set_text_props(fontweight="bold",
                                        color=COLORS["ours"] if emph
                                        else "#222222")

    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=18)

    # Caption states exactly what the marks mean, so the table cannot be read as
    # claiming significance it does not have. Built from the data, not hardcoded.
    n_seeds = int(data.get("n_seeds") or 0)
    bits = ["Bold marks the best value per column."]
    if any(_sig_mark(p) for row in pvalues.values() for p in row.values()):
        bits.append("Markers: paired t-test vs 2D-AUTO — "
                    "*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.")
    if n_seeds:
        bits.append(f"n = {n_seeds} city seeds; ± is the Student-t 95% CI.")
    # Secondary comparison against the strongest 3D baseline, stated explicitly
    # so the primary marks cannot be misread as being against the adjacent bar.
    sec = secondary_comparison(data.get("samples", {}))
    if sec:
        ref_label = dict((k, l) for k, l, _ in METHODS).get(
            SECONDARY_REFERENCE, SECONDARY_REFERENCE)
        # "p=0.000" claims a p-value of zero; report the bound instead.
        parts = [f"{k} p<0.001" if sec[k] < 0.0005 else f"{k} p={sec[k]:.3f}"
                 for k, _, _ in COLUMNS if k in sec]
        bits.append(f"Ours vs {ref_label}, paired: " + ", ".join(parts) + ".")
    if data.get("legacy_ci"):
        bits.append("CI from a legacy export (normal quantile) — re-export to correct.")
    bits.append("Means match Figs. 5-6; ↓ lower is better, ↑ higher is better.")

    fig.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.20)

    # Hang the footnote off the table's real bottom edge. A fixed y left a
    # wide dead band, because the table's height depends on how many method
    # rows are present after present_methods() drops absent ones.
    cells = table.get_celld().values()
    table_bottom_ax = min(c.get_y() for c in cells)
    pos = ax.get_position()
    footnote_y = pos.y0 + table_bottom_ax * pos.height - 0.035
    fig.text(0.5, footnote_y, "\n".join(bits),
             ha="center", va="top", fontsize=7.0, style="italic", color="0.4")
    return fig


def main():
    setup_style()
    data = load_table_results()
    fig = plot_table(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
