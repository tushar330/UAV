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
from common_plot import save_figure


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
    ("3d_gnn", "3D-GNN", False),
    ("atom3d", "ATOM-3D-VoI (Ours)", True),
]

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
        "3d_gnn": {"energy": (52.0, 1.5), "high": (70.0, 3.6),
                   "medium": (82.0, 2.9), "low": (93.0, 1.5)},
        "atom3d": {"energy": (76.0, 1.6), "high": (90.0, 2.2),
                   "medium": (89.0, 2.1), "low": (94.0, 1.3)},
    }
    return {"values": values, "placeholder": True}


def load_table_results():
    """Load real aggregated metrics if present, else the placeholder.

    Real export: results_data/overall_metrics.npz with `methods` (M,) str,
    `metrics` (K,) str, `means` (M, K) float, `ci95` (M, K) float.
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH, allow_pickle=True)
        methods = [str(m) for m in z["methods"]]
        metrics = [str(k) for k in z["metrics"]]
        means = np.asarray(z["means"], float)
        cis = np.asarray(z["ci95"], float)
        values = {m: {k: (float(means[i, j]), float(cis[i, j]))
                      for j, k in enumerate(metrics)}
                  for i, m in enumerate(methods)}
        return {"values": values, "placeholder": False}
    return generate_placeholder_table_data()


# =============================================================================
# RENDERING  (source-agnostic)
# =============================================================================

def _best_per_column(values):
    """Return {metric: method_key} of the best entry per column."""
    best = {}
    for key, _, direction in COLUMNS:
        entries = {m: values[m][key][0] for m, *_ in METHODS}
        pick = min(entries, key=entries.get) if direction == "down" \
            else max(entries, key=entries.get)
        best[key] = pick
    return best


def plot_table(data):
    values = data["values"]
    best = _best_per_column(values)

    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    ax.axis("off")

    col_labels = [h for _, h, _ in COLUMNS]
    cell_text, cell_colors = [], []
    for mkey, mlabel, emph in METHODS:
        row, colors = [], []
        for ckey, _, _ in COLUMNS:
            mean, ci = values[mkey][ckey]
            row.append(f"{mean:.1f} ± {ci:.1f}")
            colors.append(OURS_ROW_FILL if emph else "white")
        cell_text.append(row)
        cell_colors.append(colors)

    table = ax.table(
        cellText=cell_text,
        rowLabels=[m for _, m, _ in METHODS],
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=[OURS_ROW_FILL if emph else "white" for *_, emph in METHODS],
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
            mkey, _, emph = METHODS[r - 1]
            if emph:
                cell.set_text_props(color=COLORS["ours"], fontweight="bold")
            if c >= 0:                               # metric cells
                ckey = COLUMNS[c][0] if c < len(COLUMNS) else None
                if ckey and best[ckey] == mkey:
                    cell.set_text_props(fontweight="bold",
                                        color=COLORS["ours"] if emph
                                        else "#222222")

    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=18)
    fig.text(0.5, 0.045,
             "Bold marks the best value per column. Means match Figs. 5-6; "
             "↓ lower is better, ↑ higher is better.",
             ha="center", fontsize=7.5, style="italic", color="0.4")

    fig.subplots_adjust(left=0.22, right=0.98, top=0.82, bottom=0.12)
    return fig


def main():
    setup_style()
    data = load_table_results()
    fig = plot_table(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
