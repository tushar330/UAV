"""
figure04_coverage_comparison.py

Figure 4 - Coverage vs. Altitude: Quality over Quantity.

Three side-by-side top-down panels of the SAME city region (identical
buildings, identical IoT nodes, identical UAV position). Only the UAV
altitude changes, which - through cone geometry - changes the communication
footprint radius and the link quality. The figure makes the physical chain

    Altitude  ->  Coverage Radius  ->  Communication Quality

immediately visible, and communicates that ATOM-3D-VoI's *smaller* footprint
is an intentional trade-off: it focuses strong links on the critical nodes to
meet their QoS floor, rather than covering many nodes weakly.

DATA
----
The per-method altitude / QoS / link-quality values are conceptual
PLACEHOLDERS, produced by a single isolated function
(`generate_placeholder_coverage_specs`) and later replaceable by experiment
outputs. Coverage radius is NOT arbitrary - it is derived physically from
altitude via cone geometry r = H * tan(theta). The city region, buildings,
node positions and UAV position are the REAL Figure-1 city (via get_city()).

Runs independently:  python figure04_coverage_comparison.py
Exports results/figure04_coverage_comparison.png (600 DPI) via common_plot.
Reuses locked infrastructure only (common_style, common_plot, synthetic_city).
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from common_style import setup_style, COLORS
from common_plot import (
    save_figure, draw_buildings, draw_nodes, draw_coverage_cone,
    priority_color, PRIORITY_LABELS,
)
from synthetic_city import get_city


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure04_coverage_comparison"
SUPTITLE = r"Altitude $\rightarrow$ Coverage Radius $\rightarrow$ Communication Quality"

# Cone half-beamwidth. tan(59.5 deg) ~ 1.7, i.e. a 170 m ground footprint at
# 100 m altitude - matching the demo UAV baked into synthetic_city.py.
HALF_BEAMWIDTH_DEG = 59.5

FOCUS_DISTRICT = "Hospital"       # critical (high-priority) cluster to focus on
WINDOW_MARGIN = 30.0              # m of padding beyond the largest footprint

# QoS status colors (distinct from the class palette on purpose).
QOS_COLOR = {"not": "#E53935", "partial": "#FB8C00", "ok": "#2E7D32"}


# =============================================================================
# PLACEHOLDER SPECS  (isolated - replace with experiment outputs later)
# =============================================================================

def generate_placeholder_coverage_specs():
    """Return the per-method conceptual specs + the cone tan(theta).

    Schema per method (what a real experiment export would fill in):
        key, label, color, altitude [m], qos (text), qos_color,
        link_quality in [0,1], oneliner.
    Coverage radius is DERIVED here from altitude (not stored arbitrarily).
    link_quality is a placeholder proxy for achieved link rate: it rises as
    altitude drops (shorter, stronger link), so ATOM-3D-VoI (low) has the
    strongest links and 2D-AUTO (high) the weakest.
    """
    tan_theta = math.tan(math.radians(HALF_BEAMWIDTH_DEG))
    base = [
        dict(key="2d_auto", label="2D-AUTO", color="#757575", altitude=100.0,
             qos="Not satisfied", qos_color=QOS_COLOR["not"], link_quality=0.30,
             oneliner="Large footprint\nLow link quality"),
        dict(key="3d_gnn", label="3D-GNN", color="#FB8C00", altitude=75.0,
             qos="Partially satisfied", qos_color=QOS_COLOR["partial"], link_quality=0.60,
             oneliner="Moderate footprint\nLimited adaptation"),
        dict(key="atom3d", label="ATOM-3D-VoI (Ours)", color=COLORS["ours"], altitude=35.0,
             qos="Satisfied", qos_color=QOS_COLOR["ok"], link_quality=1.00,
             oneliner="Focused footprint\nHigh QoS for critical nodes"),
    ]
    for s in base:
        s["radius"] = s["altitude"] * tan_theta      # r = H * tan(theta)
        s["placeholder"] = True
    return base, tan_theta


def focus_region(city):
    """UAV position = centroid of the real high-priority (Hospital) cluster."""
    hi = [n for n in city.nodes
          if n.district == FOCUS_DISTRICT and n.priority == "high"]
    ux = float(np.mean([n.x for n in hi]))
    uy = float(np.mean([n.y for n in hi]))
    return ux, uy


# =============================================================================
# PLOTTING  (unchanged whether specs are placeholder or real)
# =============================================================================

def _draw_links(ax, city, spec, ux, uy):
    """Strong links to critical nodes in the footprint; faint for the rest.

    Critical-link weight/opacity scale with link_quality, so the SAME critical
    node is served weakly at high altitude and strongly at low altitude.
    """
    q = spec["link_quality"]
    for n in city.nodes:
        if math.hypot(n.x - ux, n.y - uy) > spec["radius"]:
            continue
        if n.priority == "high":
            # Weight/opacity scale with link quality; weak links (high altitude)
            # are thin and dashed to read as unreliable.
            ax.plot([ux, n.x], [uy, n.y], color=COLORS["ours"],
                    lw=0.8 + 3.0 * q, alpha=0.30 + 0.65 * q,
                    linestyle="--" if q < 0.45 else "-",
                    zorder=5, solid_capstyle="round")
        else:
            ax.plot([ux, n.x], [uy, n.y], color="0.45", lw=0.5,
                    alpha=0.15, zorder=4)


def draw_panel(ax, city, spec, ux, uy, half):
    ax.set_facecolor("white")
    draw_buildings(ax, city, alpha=0.85)
    draw_coverage_cone(ax, ux, uy, spec["radius"], alpha=0.10)   # very light
    _draw_links(ax, city, spec, ux, uy)
    draw_nodes(ax, city, size=18, edge=True)
    ax.scatter(ux, uy, marker="^", s=130, facecolor=COLORS["uav"],
               edgecolor="black", linewidths=1.0, zorder=9)

    ax.set_xlim(ux - half, ux + half)
    ax.set_ylim(uy - half, uy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("0.7")

    is_ours = spec["key"] == "atom3d"
    ax.set_title(spec["label"], color=spec["color"], fontsize=11,
                 fontweight="bold" if is_ours else "normal")

    ax.text(0.035, 0.975,
            f"Altitude:  {spec['altitude']:.0f} m\n"
            f"Coverage radius:  {spec['radius']:.0f} m",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.92))
    ax.text(0.035, 0.805, f"Critical QoS:  {spec['qos']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            fontweight="bold", color=spec["qos_color"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    ax.text(0.5, -0.055, spec["oneliner"], transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color=spec["color"],
            fontweight="bold" if is_ours else "normal")


def _figure_legend(fig):
    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor=priority_color(p), markeredgecolor="black",
                      markeredgewidth=0.4, markersize=7, label=PRIORITY_LABELS[p])
        for p in ("high", "medium", "low")
    ]
    handles += [
        mlines.Line2D([], [], marker="^", linestyle="none",
                      markerfacecolor=COLORS["uav"], markeredgecolor="black",
                      markersize=9, label="UAV"),
        mlines.Line2D([], [], color=COLORS["ours"], lw=3.0,
                      label="Critical link (thicker = higher quality)"),
        mpatches.Patch(facecolor=COLORS["cone"], alpha=0.35,
                       edgecolor=COLORS["uav"], label="Coverage footprint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.5,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))


def make_figure(city, specs):
    half = max(s["radius"] for s in specs) + WINDOW_MARGIN
    ux, uy = focus_region(city)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
    for ax, spec in zip(axes, specs):
        draw_panel(ax, city, spec, ux, uy, half)

    fig.suptitle(SUPTITLE, fontsize=13, fontweight="bold", y=0.98)
    _figure_legend(fig)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.16, wspace=0.08)
    return fig


def main():
    setup_style()
    city = get_city()
    specs, _tan = generate_placeholder_coverage_specs()
    fig = make_figure(city, specs)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
