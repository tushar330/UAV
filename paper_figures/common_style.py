"""
common_style.py

Shared plotting style for all paper figures.

IEEE/Elsevier publication style.
"""

from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# COLORS
# =============================================================================

COLORS = {
    "low": "#4CAF50",
    "medium": "#FB8C00",
    "high": "#E53935",

    "blind": "#2E7D32",
    "ours": "#1565C0",
    "baseline2d": "#616161",

    "uav": "#1976D2",
    "building": "#9E9E9E",
    "ground": "#ECEFF1",

    "cone": "#90CAF9"
}

# =============================================================================
# FIGURE SETTINGS
# =============================================================================

FIG_WIDTH = 15
FIG_HEIGHT = 10

DPI = 600

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# GLOBAL STYLE
# =============================================================================

def setup_style():

    plt.rcParams.update({

        "font.family": "Times New Roman",

        "font.size": 11,

        "axes.titlesize": 13,

        "axes.labelsize": 11,

        "axes.linewidth": 1.2,

        "xtick.labelsize": 10,

        "ytick.labelsize": 10,

        "legend.fontsize": 10,

        "figure.dpi": DPI,

        "savefig.dpi": DPI,

        "axes.grid": False,

        "grid.alpha": 0.25,

        "lines.linewidth": 2.0,

        "figure.facecolor": "white"

    })


# =============================================================================
# PANEL LABELS
# =============================================================================

def panel(ax, label):

    ax.text(

        -0.10,

        1.05,

        f"({label})",

        transform=ax.transAxes,

        fontsize=14,

        fontweight="bold",

        va="top"

    )


# =============================================================================
# SAVE
# =============================================================================

def save(fig, filename):

    png = OUTPUT_DIR / f"{filename}.png"
    pdf = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
