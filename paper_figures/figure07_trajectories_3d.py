"""
figure07_trajectories_3d.py

Figure 7 - 3D Trajectories in the Network (reference panel (a)).

3D view of the synthetic-city IoT nodes (colored by priority) with one
representative hover-point trajectory per policy. Message: ATOM-3D-VoI's
hover sequence dips low exactly over the high-priority clusters (Hospital,
Power Station), while 3D-GNN stays high everywhere and 2D-AUTO is pinned to
one fixed altitude.

DATA
----
Nodes are REAL (city via get_city()). The hover-point trajectories are
PLACEHOLDER, produced by `generate_placeholder_trajectories` and swapped for
real rollouts via `load_trajectory_results` (results_data/trajectories.pkl)
with no plotting changes.

Runs independently:  python figure07_trajectories_3d.py
Exports results/figure07_trajectories_3d.png (600 DPI) via common_plot.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from common_style import setup_style, COLORS
from common_plot import (save_figure, priority_color, method_labels,
                         PRIORITY_ORDER, PRIORITY_LABELS)
from synthetic_city import get_city


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure07_trajectories_3d"
TITLE = "3D Trajectories in the Network"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "trajectories.pkl"

# In-plot callouts sit over the trajectories, so they carry a soft white
# backing to stay readable at IEEE print size.
LABEL_BOX = dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78,
                 edgecolor="none")

VIEW_ELEV, VIEW_AZIM = 22, -60          # camera identical to Figure 1
Z_LIMIT = 155
BOX_ASPECT = (1.0, 1.0, 0.5)

# Method line styles (consistent palette with Figs. 3-6).
#
# `marker` is the per-hover dot size, or None to draw the path only. Real
# rollouts carry hundreds of hovers (2D-AUTO visits every node individually,
# ~500 of them), and at that count a marker per hover is a solid blob that
# hides the city and the other two paths -- so 2D-AUTO is drawn as a bare line
# and its constant altitude is conveyed by the reference plane instead.
METHOD_STYLE = {
    "2d_auto": dict(label="2D-AUTO", color="#9E9E9E", ls=":", lw=1.4, z=5,
                    marker=None),
    "3d_gnn": dict(label="3D-GNN", color="#FB8C00", ls="--", lw=1.8, z=6,
                   marker=20),
    "atom3d": dict(label="ATOM-3D-VoI (Ours)", color=COLORS["ours"],
                   ls="-", lw=2.0, z=7, marker=11),
}

# Labels must name whatever actually produced each curve (results_data/labels.json):
# the '3d_gnn' slot currently holds a deterministic greedy cover, NOT a trained
# GNN, and the legend must not claim otherwise.
for _key, _label in method_labels().items():
    if _key in METHOD_STYLE:
        METHOD_STYLE[_key]["label"] = _label


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real rollouts)
# =============================================================================

def generate_placeholder_trajectories():
    """PLACEHOLDER hover-point sequences per method (x, y, altitude).

    Schema (same as the real rollout export):
        { method_key: [(x, y, z), ...] }   # ordered hover points, depot first/last
    Behaviour encoded: Ours cruises ~90 m, dives to ~30 m over the Hospital
    (500,180) and Power Station (170,820) high-priority clusters, and serves
    medium clusters from moderate altitude. 3D-GNN wanders at ~80 m with no
    purposeful descents. 2D-AUTO holds its fixed 100 m altitude.
    """
    return {
        "atom3d": [
            (50, 50, 0), (150, 300, 88), (220, 500, 90), (170, 820, 31),
            (500, 760, 88), (500, 500, 55), (780, 500, 89), (780, 820, 52),
            (620, 300, 87), (500, 180, 30), (250, 120, 86), (50, 50, 0),
        ],
        "3d_gnn": [
            (50, 50, 0), (300, 300, 80), (600, 400, 83), (820, 640, 78),
            (560, 720, 82), (240, 700, 79), (420, 160, 81), (50, 50, 0),
        ],
        "2d_auto": [
            (50, 50, 0), (250, 250, 100), (520, 380, 100), (780, 600, 100),
            (520, 820, 100), (220, 640, 100), (50, 50, 0),
        ],
        "placeholder": True,
    }


def load_trajectory_results():
    """Load real rollout hover sequences if present, else the placeholder."""
    if REAL_DATA_PATH.exists():
        with open(REAL_DATA_PATH, "rb") as f:
            data = pickle.load(f)
        data.setdefault("placeholder", False)
        return data
    return generate_placeholder_trajectories()


def cruise_altitude(points):
    """The one altitude a fixed-altitude tour holds, or None if it holds none.

    Depot waypoints sit at ground level and every sortie return passes through
    them, so they are excluded before asking whether the rest of the tour is
    flat. Returns None when the path varies (i.e. it is not a fixed-altitude
    method), so the reference plane is only ever drawn where it is truthful.
    """
    if not points:
        return None
    z = np.asarray(points, float)[:, 2]
    airborne = z[z > 1e-6]
    if not len(airborne) or np.ptp(airborne) > 0.5:
        return None
    return float(np.median(airborne))


# =============================================================================
# PLOTTING  (source-agnostic)
# =============================================================================

def plot_trajectories(city, trajs):
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    # Real nodes, colored by priority (high drawn last / on top).
    for p in reversed(PRIORITY_ORDER):
        pts = [n for n in city.nodes if n.priority == p]
        ax.scatter([n.x for n in pts], [n.y for n in pts], [n.z for n in pts],
                   s=5, color=priority_color(p), depthshade=True, alpha=0.75)

    # 2D-AUTO's defining property is that it holds ONE altitude for the whole
    # sortie. In a 3D projection a horizontal path still sweeps up and down the
    # screen, so the constancy is invisible without a reference: this draws the
    # plane the tour lives in, taken from the data rather than assumed.
    cruise = cruise_altitude(trajs.get("2d_auto"))
    if cruise is not None:
        gx, gy = np.meshgrid(np.linspace(0, city.metadata.width, 2),
                             np.linspace(0, city.metadata.height, 2))
        ax.plot_surface(gx, gy, np.full_like(gx, cruise), zorder=1,
                        color=METHOD_STYLE["2d_auto"]["color"], alpha=0.13,
                        shade=False, linewidth=0)
        ax.text(city.metadata.width * 0.99, city.metadata.height * 0.03,
                cruise + 4, f"2D-AUTO holds {cruise:.0f} m", fontsize=7,
                color="0.35", ha="right", zorder=9, bbox=LABEL_BOX)

    # Trajectories (draw ours last so it sits on top). Hover markers are drawn
    # only where they stay readable -- see METHOD_STYLE["marker"].
    for key in ("2d_auto", "3d_gnn", "atom3d"):
        st = METHOD_STYLE[key]
        pts = np.array(trajs[key], float)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=st["color"],
                ls=st["ls"], lw=st["lw"], zorder=st["z"])
        if st["marker"]:
            ax.scatter(pts[1:-1, 0], pts[1:-1, 1], pts[1:-1, 2], s=st["marker"],
                       color=st["color"], edgecolor="black", linewidths=0.3,
                       depthshade=False, zorder=st["z"])

    # Call out Ours' deepest descent. The earlier version numbered every hover
    # and dropped a ground line under each one below 45 m; on real rollouts
    # (88 hovers, most of them dives) that produced ~86 overlapping labels and
    # a picket fence of verticals. One callout carries the same message.
    # Multi-UAV routes return to the depot between sorties, so the interior of
    # the path contains ground-level depot waypoints. Those are not dives and
    # must be excluded, or the callout lands on the depot at 0 m.
    ours = np.array(trajs["atom3d"], float)
    depot_xy = ours[0, :2]
    interior = ours[1:-1]
    interior = interior[np.linalg.norm(interior[:, :2] - depot_xy, axis=1) > 1.0]
    if len(interior):
        lo = interior[int(np.argmin(interior[:, 2]))]
        ax.plot([lo[0], lo[0]], [lo[1], lo[1]], [0, lo[2]], color=COLORS["ours"],
                ls=":", lw=1.1, alpha=0.9, zorder=9)
        ax.text(lo[0], lo[1], lo[2] + 14,
                f"lowest dive {lo[2]:.0f} m", fontsize=7,
                color=COLORS["ours"], fontweight="bold", ha="center",
                zorder=9, bbox=LABEL_BOX)

    # Depot marker.
    ax.scatter(*ours[0], marker="s", s=90, facecolor=COLORS["uav"],
               edgecolor="black", linewidths=0.8, depthshade=False, zorder=8)
    ax.text(ours[0, 0] + 55, ours[0, 1] + 30, 14, "Depot", fontsize=7.5,
            ha="left", zorder=9, bbox=LABEL_BOX)

    # Camera / axes (Figure-1 conventions).
    ax.set_xlim(0, city.metadata.width)
    ax.set_ylim(0, city.metadata.height)
    ax.set_zlim(0, Z_LIMIT)
    ax.set_box_aspect(BOX_ASPECT)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel("x (m)", labelpad=4)
    ax.set_ylabel("y (m)", labelpad=4)
    ax.set_zlabel("altitude (m)", labelpad=-2)
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_yticks([0, 250, 500, 750, 1000])
    ax.set_zticks([0, 50, 100, 150])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))

    # Legend: priorities + methods.
    handles = [
        mlines.Line2D([], [], marker="o", ls="none", markersize=5,
                      markerfacecolor=priority_color(p), markeredgecolor="none",
                      label=PRIORITY_LABELS[p])
        for p in PRIORITY_ORDER
    ] + [
        mlines.Line2D([], [], color=st["color"], ls=st["ls"],
                      lw=max(st["lw"], 1.8), label=st["label"])
        for st in (METHOD_STYLE["2d_auto"], METHOD_STYLE["3d_gnn"],
                   METHOD_STYLE["atom3d"])
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.97),
              fontsize=7.5, framealpha=0.92, borderpad=0.5, labelspacing=0.35)

    ax.set_title(TITLE + "\n(hover-point sequences: Ours varies altitude, "
                 "Blind-3D stays high, 2D-AUTO is pinned to one altitude)",
                 fontsize=10, pad=0)
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.05, top=0.97)
    return fig


def main():
    setup_style()
    city = get_city()
    trajs = load_trajectory_results()
    fig = plot_trajectories(city, trajs)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
