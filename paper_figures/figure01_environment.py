"""
figure01_environment.py

Figure 1 — Environment.

Introduces the synthetic smart city that every experiment in the paper is
evaluated on: roads, extruded 3D buildings, points of interest, criticality-
classed IoT nodes, the UAV depot, one demonstration UAV and its downlink
coverage cone.

Runs independently:

    python figure01_environment.py

produces `results/figure01_environment.png` (600 DPI) via
`common_plot.save_figure`.

Reuses the locked infrastructure and never modifies it:
  - `synthetic_city.get_city`      -> the frozen city (never regenerated here)
  - `common_style.setup_style`     -> IEEE rcParams
  - `common_style.COLORS`          -> palette (via common_plot re-exports)
  - `common_plot`                  -> priority colors/labels + save_figure

Note: `common_plot`'s draw_* primitives are 2D (top-down). This figure is a
3D perspective view, so the extruded geometry is composed locally; only the
palette, priority metadata and the save helper are shared.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

from common_style import setup_style, COLORS
from common_plot import (
    save_figure,
    priority_color,
    PRIORITY_ORDER,
    PRIORITY_LABELS,
)
from synthetic_city import get_city


# =============================================================================
# TUNABLES (kept here, not scattered as magic numbers)
# =============================================================================

FIG_NAME = "figure01_environment"

# Camera. A lowish elevation gives buildings a readable skyline profile.
VIEW_ELEV = 22
VIEW_AZIM = -60

# Vertical exaggeration: the map is 1000 m wide but buildings are <= 40 m and
# the UAV flies at 100 m. Compressing the z-box relative to x/y keeps the
# scene readable while still showing real altitude.
Z_LIMIT = 115
BOX_ASPECT = (1.0, 1.0, 0.5)

# The ground quad is forced behind every other artist to defeat Matplotlib's
# per-collection depth sorting (otherwise the large flat quad paints over
# foreground buildings).
GROUND_COLOR = "#ECECEC"
GROUND_SORT_ZPOS = -1.0e4

# IoT node marker size (~10% larger than the previous 7.0).
NODE_SIZE = 7.7

# Demo-UAV marker size (~20% smaller than the previous 150).
UAV_MARKER_SIZE = 120

# Coverage-cone fill opacity (slightly more visible).
CONE_ALPHA = 0.20

# POI star marker floats clear above the tallest buildings (<=40 m), with a
# thin stem back to the ground so it still reads as a ground location.
POI_MARKER_Z = 78
POI_LABEL_DZ = 8

# Roads are drawn as thin filled ribbons (not lines): flat lines z-fight with
# the ground surface in mplot3d and disappear. As polygons with a sort
# position just above the ground they render reliably in dark asphalt.
ROAD_COLOR = "#C2C6CA"    # light gray, visually secondary
ROAD_Z = 0.4
ROAD_HW_MAIN = 4.0        # half-width (m) of main roads (thinner appearance)
ROAD_HW_SECONDARY = 2.0   # half-width (m) of secondary roads
ROAD_SORT_ZPOS = -9.0e3   # above the ground (-1e4), below the buildings

# Light background box behind floating text labels so they stay legible over
# buildings without an opaque block.
LABEL_BBOX = dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="none", alpha=0.7)

# Face shading factors relative to the base building color (simple one-sided
# lighting so extruded blocks read as solid 3D volumes).
FACE_SHADE = {
    "top": 1.18,
    "x_far": 0.90,
    "x_near": 0.70,
    "y_far": 1.00,
    "y_near": 0.80,
}
BUILDING_EDGE = "#888888"

# Coverage-cone surface resolution.
CONE_THETA_STEPS = 48
CONE_HEIGHT_STEPS = 2

# District boundaries: dashed outline, no fill, kept visually secondary.
DISTRICT_EDGE = "#A6A6A6"
DISTRICT_LS = (0, (6, 4))
DISTRICT_LW = 0.8
DISTRICT_Z = 1.0
# Names float just above the low-rise rooflines so they stay readable inside
# each region without being buried by the taller downtown blocks.
DISTRICT_LABEL_Z = 16.0
DISTRICT_LABEL_COLOR = "#5A5A5A"
DISTRICT_LABEL_SIZE = 7.0
# Faint background box keeps district names legible over the ground/roads
# while staying subordinate to the buildings.
DISTRICT_LABEL_BBOX = dict(boxstyle="round,pad=0.1", facecolor="white",
                           edgecolor="none", alpha=0.55)


# =============================================================================
# COLOR HELPERS
# =============================================================================

def _shade(color: str, factor: float):
    """Multiply an RGB color by `factor`, clamped to [0, 1]. factor>1 lightens."""
    r, g, b = to_rgb(color)
    return (min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0))


# =============================================================================
# SCENE ELEMENTS (3D)
# =============================================================================

def draw_ground(ax, city) -> None:
    """Flat ground quad at z = 0 covering the whole map extent.

    `set_sort_zpos` pins it behind every other collection so it never paints
    over foreground buildings (a known Matplotlib 3D depth-sort limitation).
    """
    w, h = city.metadata.width, city.metadata.height
    quad = [[(0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0)]]
    ground = Poly3DCollection(quad, facecolor=GROUND_COLOR, edgecolor="none")
    ground.set_sort_zpos(GROUND_SORT_ZPOS)
    ax.add_collection3d(ground)


def _road_ribbon(road, half_width):
    """Return a flat rectangular quad (at z=ROAD_Z) centred on a road segment."""
    dx, dy = road.x2 - road.x1, road.y2 - road.y1
    length = np.hypot(dx, dy)
    # Unit normal in the ground plane, scaled to the road half-width.
    nx, ny = -dy / length * half_width, dx / length * half_width
    return [
        (road.x1 + nx, road.y1 + ny, ROAD_Z),
        (road.x2 + nx, road.y2 + ny, ROAD_Z),
        (road.x2 - nx, road.y2 - ny, ROAD_Z),
        (road.x1 - nx, road.y1 - ny, ROAD_Z),
    ]


def draw_roads(ax, city) -> None:
    """Road network as thin dark ribbons resting just above the ground."""
    quads = [
        _road_ribbon(
            r, ROAD_HW_MAIN if r.road_type == "main" else ROAD_HW_SECONDARY
        )
        for r in city.roads
    ]
    ribbons = Poly3DCollection(quads, facecolor=ROAD_COLOR, edgecolor="none")
    ribbons.set_sort_zpos(ROAD_SORT_ZPOS)
    ax.add_collection3d(ribbons)


def draw_districts(ax, city) -> None:
    """Dashed, unfilled district outlines drawn low and light so they read as
    a background layer beneath the buildings. District *names* are added
    separately by `draw_district_labels` (a 2D overlay) because in-scene 3D
    text is occluded by the surrounding buildings.
    """
    theta = np.linspace(0, 2 * np.pi, 120)
    for d in city.districts:
        ax.plot(
            d.center_x + d.radius * np.cos(theta),
            d.center_y + d.radius * np.sin(theta),
            np.full_like(theta, DISTRICT_Z),
            color=DISTRICT_EDGE, linestyle=DISTRICT_LS, linewidth=DISTRICT_LW,
            alpha=0.8,
        )


def draw_district_labels(ax, city) -> None:
    """Centre a district name on each region as a 2D overlay.

    The 3D centroid is projected to 2D with the current camera and annotated
    on top of the scene, so names stay readable regardless of building
    occlusion while remaining visually secondary (small, gray, faint box).
    Must be called after the view/limits are set.
    """
    proj = ax.get_proj()
    for d in city.districts:
        x2, y2, _ = proj3d.proj_transform(
            d.center_x, d.center_y, DISTRICT_LABEL_Z, proj
        )
        ax.annotate(
            d.name.replace("_", " "), xy=(x2, y2), xycoords="data",
            fontsize=DISTRICT_LABEL_SIZE, style="italic", ha="center",
            va="center", color=DISTRICT_LABEL_COLOR, bbox=DISTRICT_LABEL_BBOX,
            zorder=11,
        )


def _building_faces(b):
    """Return (faces, face_colors) for one extruded building box (no bottom)."""
    x0, x1 = b.x - b.width / 2, b.x + b.width / 2
    y0, y1 = b.y - b.depth / 2, b.y + b.depth / 2
    z0, z1 = 0.0, b.height
    base = COLORS["building"]

    faces = [
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # y-near
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # y-far
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # x-near
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # x-far
    ]
    colors = [
        _shade(base, FACE_SHADE["top"]),
        _shade(base, FACE_SHADE["y_near"]),
        _shade(base, FACE_SHADE["y_far"]),
        _shade(base, FACE_SHADE["x_near"]),
        _shade(base, FACE_SHADE["x_far"]),
    ]
    return faces, colors


def draw_buildings(ax, city) -> None:
    """Extrude every building footprint into a shaded 3D block."""
    all_faces, all_colors = [], []
    for b in city.buildings:
        faces, colors = _building_faces(b)
        all_faces.extend(faces)
        all_colors.extend(colors)

    blocks = Poly3DCollection(
        all_faces, facecolors=all_colors,
        edgecolor=BUILDING_EDGE, linewidths=0.15,
    )
    blocks.set_zorder(2)
    ax.add_collection3d(blocks)


def draw_nodes(ax, city, *, size=NODE_SIZE) -> None:
    """IoT nodes as a 3D scatter, colored by criticality class.

    Drawn in fixed priority order (high last / on top) so critical nodes stay
    visible. Colors follow DATA_SPEC via common_plot.priority_color.
    """
    for priority in PRIORITY_ORDER:
        pts = [n for n in city.nodes if n.priority == priority]
        if not pts:
            continue
        ax.scatter(
            [n.x for n in pts], [n.y for n in pts], [n.z for n in pts],
            s=size, color=priority_color(priority),
            edgecolors="black", linewidths=0.15, depthshade=True,
        )


def draw_pois(ax, city) -> None:
    """Star-mark the four POIs with a compact text label above each."""
    for poi in city.pois:
        # Thin stem anchoring the floating star to its ground location.
        ax.plot(
            [poi.x, poi.x], [poi.y, poi.y], [0, POI_MARKER_Z],
            color="#B08900", linestyle="-", linewidth=0.6, alpha=0.5,
        )
        ax.scatter(
            poi.x, poi.y, POI_MARKER_Z,
            marker="*", s=200, facecolor="#FFD54F",
            edgecolor="black", linewidths=0.6, depthshade=False,
        )
        ax.text(
            poi.x, poi.y, POI_MARKER_Z + POI_LABEL_DZ, poi.name,
            fontsize=7.5, ha="center", va="bottom", zorder=10, bbox=LABEL_BBOX,
        )


def draw_depot(ax, city) -> None:
    """Square marker + label at the depot."""
    d = city.depot
    ax.scatter(
        d.x, d.y, d.z, marker="s", s=130,
        facecolor=COLORS["uav"], edgecolor="black", linewidths=1.0,
        depthshade=False,
    )
    ax.text(d.x + 25, d.y, d.z + 10, "Depot", fontsize=8, ha="left",
            bbox=LABEL_BBOX)


def draw_coverage_cone(ax, x, y, z, radius) -> None:
    """Downlink coverage cone: apex at the UAV, circular footprint on ground."""
    theta = np.linspace(0, 2 * np.pi, CONE_THETA_STEPS)
    u = np.linspace(0, 1, CONE_HEIGHT_STEPS)          # 0 = apex, 1 = ground ring
    U, T = np.meshgrid(u, theta)
    X = x + U * radius * np.cos(T)
    Y = y + U * radius * np.sin(T)
    Z = z * (1 - U)
    ax.plot_surface(
        X, Y, Z, color=COLORS["cone"], alpha=CONE_ALPHA,
        linewidth=0, antialiased=True, shade=False,
    )
    # Footprint outline on the ground for clarity.
    ax.plot(
        x + radius * np.cos(theta), y + radius * np.sin(theta),
        np.zeros_like(theta),
        color=COLORS["uav"], linewidth=0.8, alpha=0.6,
    )


def draw_demo_uav(ax, city) -> None:
    """Demonstration UAV marker, a drop-line to the ground, and its cone."""
    u = city.demo_uav
    draw_coverage_cone(ax, u.x, u.y, u.z, u.coverage_radius)
    ax.plot([u.x, u.x], [u.y, u.y], [0, u.z],
            color=COLORS["uav"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax.scatter(
        u.x, u.y, u.z, marker="^", s=UAV_MARKER_SIZE,
        facecolor=COLORS["uav"], edgecolor="black", linewidths=1.0,
        depthshade=False,
    )
    ax.text(u.x, u.y, u.z + 8, "UAV", fontsize=8, ha="center", bbox=LABEL_BBOX)


# =============================================================================
# LEGEND
# =============================================================================

def build_legend(ax) -> None:
    """Legend split into three logical, titled groups across the top:
    Priority Nodes, Infrastructure and UAV.
    """
    priority_handles = [
        mlines.Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor=priority_color(p), markeredgecolor="black",
                      markeredgewidth=0.3, markersize=6, label=PRIORITY_LABELS[p])
        for p in PRIORITY_ORDER
    ]
    infra_handles = [
        mpatches.Patch(facecolor=COLORS["building"], edgecolor=BUILDING_EDGE,
                       label="Building"),
        mlines.Line2D([], [], marker="s", linestyle="none",
                      markerfacecolor=COLORS["uav"], markeredgecolor="black",
                      markersize=8, label="Depot"),
        mlines.Line2D([], [], marker="*", linestyle="none",
                      markerfacecolor="#FFD54F", markeredgecolor="black",
                      markersize=11, label="Point of interest"),
        mpatches.Patch(facecolor=ROAD_COLOR, edgecolor="none", label="Road"),
        mlines.Line2D([], [], color=DISTRICT_EDGE, linestyle=DISTRICT_LS,
                      linewidth=DISTRICT_LW, label="District"),
    ]
    uav_handles = [
        mlines.Line2D([], [], marker="^", linestyle="none",
                      markerfacecolor=COLORS["uav"], markeredgecolor="black",
                      markersize=9, label="UAV"),
        mpatches.Patch(facecolor=COLORS["cone"], alpha=0.35,
                       edgecolor=COLORS["uav"], label="Coverage cone"),
    ]

    common = dict(loc="upper left", fontsize=7.5, framealpha=0.9,
                  borderpad=0.5, handletextpad=0.4, labelspacing=0.35)
    groups = [
        ("Priority Nodes", priority_handles, (0.00, 1.00)),
        ("Infrastructure", infra_handles, (0.24, 1.00)),
        ("UAV", uav_handles, (0.46, 1.00)),
    ]
    for title, handles, anchor in groups:
        leg = ax.legend(handles=handles, title=title, bbox_to_anchor=anchor,
                        **common)
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(8)
        ax.add_artist(leg)   # keep every group (later ax.legend would replace)


# =============================================================================
# FIGURE ASSEMBLY
# =============================================================================

def make_figure(city):
    """Compose the full Figure 1 and return the Matplotlib Figure."""
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    # Scene, drawn back-to-front for sensible occlusion.
    draw_ground(ax, city)
    draw_roads(ax, city)
    draw_districts(ax, city)   # secondary layer, beneath the buildings
    draw_buildings(ax, city)
    draw_nodes(ax, city)
    draw_pois(ax, city)
    draw_depot(ax, city)
    draw_demo_uav(ax, city)

    # Camera and extents.
    w, h = city.metadata.width, city.metadata.height
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0, Z_LIMIT)
    ax.set_box_aspect(BOX_ASPECT)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    # Clean, publication-style axes.
    ax.set_xlabel("x (m)", labelpad=6)
    ax.set_ylabel("y (m)", labelpad=6)
    ax.set_zlabel("altitude (m)", labelpad=-2)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_zticks([0, 50, 100])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.6)

    # District names as a 2D overlay (needs the finalised camera projection).
    draw_district_labels(ax, city)

    build_legend(ax)

    # tight_layout mishandles 3D axes and clips the z-label / corner labels;
    # reserve explicit margins instead.
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.06, top=0.98)
    return fig


def main() -> None:
    setup_style()
    city = get_city()          # frozen city; never regenerated here
    fig = make_figure(city)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
