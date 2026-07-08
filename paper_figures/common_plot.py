"""
common_plot.py

Reusable, generic plotting helpers shared by every paper figure.

Design rules (see FILE_DEPENDENCIES.md / CODE_RULES.md):
  - Depends only on the locked infrastructure: `common_style` (COLORS) and
    `synthetic_city` (data types). Never modifies either.
  - Contains ONLY generic visualization primitives. No experiment-specific
    logic (no trajectories, metrics, VoI, RL, etc.) lives here.
  - A helper belongs here only when >= 2 figures need it.

Every helper draws onto a caller-supplied Matplotlib `ax`, so figures stay
independent and compose these primitives however they need.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from common_style import COLORS


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
#
# The single output directory for every generated figure is
# `paper_figures/results/`. It is resolved relative to THIS file (not the
# current working directory) so figures save to the same place regardless
# of where they are launched from. It is created automatically on import.

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# High-resolution export matches the IEEE style in common_style.py.
EXPORT_DPI = 600


# =============================================================================
# CLASS / PRIORITY HELPERS
# =============================================================================

# Fixed priority order for consistent legends and z-ordering across figures.
PRIORITY_ORDER = ("high", "medium", "low")

# Human-readable labels for legends.
PRIORITY_LABELS = {
    "high": "High priority",
    "medium": "Medium priority",
    "low": "Low priority",
}


def priority_color(priority: str) -> str:
    """Return the palette color for a node priority ('high'/'medium'/'low')."""
    return COLORS[priority]


# =============================================================================
# SAVE
# =============================================================================

def save_figure(fig, name: str, *, pdf: bool = True) -> Path:
    """Save a figure into `results/` as PNG (and, by default, PDF) at 600 DPI.

    Parameters
    ----------
    fig  : the Matplotlib Figure to save.
    name : base filename without extension, e.g. "figure01_environment".
    pdf  : also write a vector PDF alongside the PNG (default True).

    Returns the PNG path. Prints each saved path.
    """
    png_path = RESULTS_DIR / f"{name}.png"
    fig.savefig(png_path, dpi=EXPORT_DPI, bbox_inches="tight")
    print(f"Saved: {png_path}")

    if pdf:
        pdf_path = RESULTS_DIR / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    return png_path


# =============================================================================
# 2D TOP-DOWN SCENE PRIMITIVES
# =============================================================================
#
# All of the following draw onto a top-down (x, y) axis in metres. The caller
# is responsible for axis limits, aspect ratio and titles; these helpers only
# add scene geometry so they can be freely combined.


def draw_ground(ax, city) -> None:
    """Fill the map extent with the ground color and set an equal aspect."""
    ax.set_xlim(0, city.metadata.width)
    ax.set_ylim(0, city.metadata.height)
    ax.set_aspect("equal")
    ax.set_facecolor(COLORS["ground"])


def draw_roads(ax, city, *, main_lw: float = 2.4, secondary_lw: float = 1.0):
    """Draw the road network. Main roads are thicker than secondary roads."""
    for road in city.roads:
        lw = main_lw if road.road_type == "main" else secondary_lw
        ax.plot(
            [road.x1, road.x2],
            [road.y1, road.y2],
            color="white",
            linewidth=lw,
            solid_capstyle="round",
            zorder=1,
        )


def draw_buildings(ax, city, *, alpha: float = 0.9):
    """Draw building footprints as rectangles centred on (x, y)."""
    for b in city.buildings:
        rect = mpatches.Rectangle(
            (b.x - b.width / 2, b.y - b.depth / 2),
            b.width,
            b.depth,
            facecolor=COLORS["building"],
            edgecolor="none",
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(rect)


def draw_pois(ax, city, *, marker: str = "*", size: float = 260):
    """Mark points of interest with a star and a short text label."""
    for poi in city.pois:
        ax.scatter(
            poi.x,
            poi.y,
            marker=marker,
            s=size,
            facecolor="#FFD54F",
            edgecolor="black",
            linewidths=0.8,
            zorder=6,
        )
        ax.annotate(
            poi.name,
            (poi.x, poi.y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            zorder=6,
        )


def draw_nodes(ax, city, *, size: float = 14, edge: bool = False):
    """Scatter IoT nodes colored by criticality class.

    Draws in fixed priority order (high on top) so critical nodes are never
    hidden. Returns nothing; use `draw_legend` for the class legend.
    """
    for priority in PRIORITY_ORDER:
        pts = [n for n in city.nodes if n.priority == priority]
        if not pts:
            continue
        ax.scatter(
            [n.x for n in pts],
            [n.y for n in pts],
            s=size,
            color=priority_color(priority),
            edgecolors="black" if edge else "none",
            linewidths=0.3 if edge else 0.0,
            zorder=4 + PRIORITY_ORDER.index(priority),
            label=PRIORITY_LABELS[priority],
        )


def draw_depot(ax, city, *, size: float = 160):
    """Mark the depot with a distinct square marker."""
    ax.scatter(
        city.depot.x,
        city.depot.y,
        marker="s",
        s=size,
        facecolor=COLORS["uav"],
        edgecolor="black",
        linewidths=1.0,
        zorder=7,
        label="Depot",
    )


def draw_demo_uav(ax, city, *, size: float = 120, with_cone: bool = True):
    """Mark the demo UAV and, optionally, its coverage footprint."""
    ax.scatter(
        city.demo_uav.x,
        city.demo_uav.y,
        marker="^",
        s=size,
        facecolor=COLORS["uav"],
        edgecolor="black",
        linewidths=1.0,
        zorder=8,
        label="UAV",
    )
    if with_cone:
        draw_coverage_cone(
            ax,
            city.demo_uav.x,
            city.demo_uav.y,
            city.demo_uav.coverage_radius,
        )


def draw_coverage_cone(ax, x: float, y: float, radius: float, *, alpha: float = 0.25):
    """Draw a UAV coverage footprint (top-down projection of the cone).

    Generic: pass any centre and radius. Used for the demo UAV and for any
    figure that needs to show a footprint at a given position.
    """
    circle = mpatches.Circle(
        (x, y),
        radius,
        facecolor=COLORS["cone"],
        edgecolor=COLORS["uav"],
        linewidth=1.0,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(circle)
    return circle


# =============================================================================
# LEGEND
# =============================================================================

def draw_legend(ax, *, include_scene: bool = True, loc: str = "upper right"):
    """Add a legend for the standard scene elements.

    Builds explicit proxy handles so the legend is identical across figures
    regardless of draw order or which primitives were called.
    """
    handles = [
        mlines.Line2D(
            [], [],
            marker="o", linestyle="none",
            markerfacecolor=priority_color(p), markeredgecolor="none",
            markersize=7, label=PRIORITY_LABELS[p],
        )
        for p in PRIORITY_ORDER
    ]

    if include_scene:
        handles += [
            mlines.Line2D(
                [], [], marker="s", linestyle="none",
                markerfacecolor=COLORS["uav"], markeredgecolor="black",
                markersize=8, label="Depot",
            ),
            mlines.Line2D(
                [], [], marker="^", linestyle="none",
                markerfacecolor=COLORS["uav"], markeredgecolor="black",
                markersize=9, label="UAV",
            ),
            mpatches.Patch(
                facecolor=COLORS["building"], edgecolor="none",
                label="Building",
            ),
        ]

    ax.legend(handles=handles, loc=loc, framealpha=0.9)
    return handles
