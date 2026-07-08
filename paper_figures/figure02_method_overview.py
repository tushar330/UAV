"""
figure02_method_overview.py

Figure 2 - Overview of the Proposed ATOM-3D-VoI Framework.

A conceptual methodology figure (NO experimental results, NO fabricated
metrics). Two panels plus a workflow ribbon:

  (a) Physical mechanism, drawn as a side-elevation of a real region of the
      Figure-1 city (the Hospital district): actual buildings, actual node
      locations and priority colors, with the UAV, its adaptive altitude
      profile, coverage cone and communication links overlaid. Side elevation
      is used because altitude adaptation is far clearer than in 3D.

  (b) The learning method as a Constrained MDP, mirroring the actual
      architecture: State -> Graph Encoder (multi-head self-attention) ->
      Trajectory Decoder -> Altitude Head -> CMDP constraint layer -> Action
      -> Environment -> Reward -> Policy Update.

Runs independently:  python figure02_method_overview.py
Exports results/figure02_method_overview.png (600 DPI) via common_plot.

Reuses the locked infrastructure only (common_style, common_plot,
synthetic_city); the city is loaded with get_city() and never regenerated.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import (
    Rectangle, Polygon, Ellipse, FancyBboxPatch, FancyArrowPatch,
)

from common_style import setup_style, COLORS, panel
from common_plot import save_figure, priority_color, PRIORITY_LABELS
from synthetic_city import get_city


# =============================================================================
# TUNABLES
# =============================================================================

FIG_NAME = "figure02_method_overview"
SUPTITLE = "Overview of the Proposed ATOM-3D-VoI Framework"
CONCEPT_NOTE = "Conceptual schematic - illustrative, not experimental results"

FOCUS_DISTRICT = "Hospital"     # region shown in the side-elevation panel

# Panel (a) altitudes (m). Conceptual UAV profile, not a measured trajectory.
CRUISE_ALT = 90.0
SAFE_CLEARANCE = 12.0           # hover just above the tallest local rooftop
XPAD = 18.0                     # horizontal padding around the district (m)
Z_TOP = 108.0                   # altitude-axis top

UAV_COLOR = COLORS["uav"]
LINK_COLOR = COLORS["uav"]

# Panel (b) block-diagram palette. Accents deliberately avoid the class
# red/orange/green so nothing is confused with a priority level.
METHOD_FILL, METHOD_EDGE = "#E8F0FE", COLORS["ours"]     # encoder/decoder/heads
STATE_FILL, STATE_EDGE = "#F0F0F0", "#616161"
ACTION_FILL, ACTION_EDGE = "#D6E4FF", "#0D47A1"
ENV_FILL, ENV_EDGE = "#ECEFF1", "#546E7A"
REWARD_FILL, REWARD_EDGE = "#E0F2F1", "#00796B"          # teal
UPDATE_FILL, UPDATE_EDGE = "#F3E5F5", "#6A1B9A"          # purple
ARROW_COLOR = "#555555"


# =============================================================================
# PANEL (a) - SIDE-ELEVATION MECHANISM (real Hospital-district geometry)
# =============================================================================

def _region(city):
    """Return the focus district plus its buildings and nodes (all real)."""
    district = next(d for d in city.districts if d.name == FOCUS_DISTRICT)
    buildings = [b for b in city.buildings if b.district == FOCUS_DISTRICT]
    nodes = [n for n in city.nodes if n.district == FOCUS_DISTRICT]
    return district, buildings, nodes


def _uav_profile(x_lo, x_hi, x_center, hover_alt):
    """Piecewise adaptive-altitude path: cruise -> descend -> hover -> climb."""
    xs = [x_lo, x_center - 55, x_center - 18, x_center + 18, x_center + 55, x_hi]
    zs = [CRUISE_ALT, CRUISE_ALT, hover_alt, hover_alt, CRUISE_ALT, CRUISE_ALT]
    return np.array(xs), np.array(zs)


def draw_mechanism_panel(ax, city):
    """Side elevation of the Hospital district with the UAV overlay."""
    _, buildings, nodes = _region(city)

    x_lo = min(b.x - b.width / 2 for b in buildings) - XPAD
    x_hi = max(b.x + b.width / 2 for b in buildings) + XPAD
    hover_alt = max(b.height for b in buildings) + SAFE_CLEARANCE
    x_center = float(np.mean([n.x for n in nodes]))

    # Ground.
    ax.axhline(0, color="#9E9E9E", lw=1.2, zorder=1)

    # Buildings as a real skyline (side projection onto x).
    for b in buildings:
        ax.add_patch(Rectangle(
            (b.x - b.width / 2, 0), b.width, b.height,
            facecolor=COLORS["building"], edgecolor="#888888",
            linewidth=0.4, alpha=0.9, zorder=2,
        ))

    # Coverage cone from the hover apex down to a ground footprint.
    footprint_r = max(abs(n.x - x_center) for n in nodes) + 10
    cone = Polygon(
        [(x_center, hover_alt),
         (x_center - footprint_r, 0.0),
         (x_center + footprint_r, 0.0)],
        closed=True, facecolor=COLORS["cone"], edgecolor=UAV_COLOR,
        linewidth=0.8, alpha=0.22, zorder=3,
    )
    ax.add_patch(cone)

    # Communication links from the UAV to each served (in-footprint) node.
    for n in nodes:
        if abs(n.x - x_center) <= footprint_r:
            ax.plot([x_center, n.x], [hover_alt, n.z],
                    color=LINK_COLOR, lw=0.7, ls=(0, (3, 2)),
                    alpha=0.7, zorder=4)

    # Real IoT nodes (Hospital district -> high priority) on the rooftops.
    for n in nodes:
        ax.scatter(n.x, n.z, s=34, color=priority_color(n.priority),
                   edgecolors="black", linewidths=0.4, zorder=6)

    # Adaptive UAV path with a plain label beside the constant-altitude segments.
    ax.text(x_lo + 4, CRUISE_ALT + 2, "Cruise altitude", fontsize=7,
            color=UAV_COLOR, va="bottom")
    xs, zs = _uav_profile(x_lo, x_hi, x_center, hover_alt)
    ax.plot(xs, zs, color=UAV_COLOR, lw=2.2, zorder=7)
    ax.scatter(x_center, hover_alt, marker="^", s=150, facecolor=UAV_COLOR,
               edgecolor="black", linewidths=1.0, zorder=8)
    ax.text(x_center, hover_alt + 4, "UAV", fontsize=8, ha="center",
            va="bottom", zorder=8)

    # Sparse annotations (kept minimal to avoid clutter).
    ax.annotate("adaptive descent\nto meet QoS floor",
                xy=(x_center - 30, (CRUISE_ALT + hover_alt) / 2),
                xytext=(x_lo + 6, hover_alt + 26), fontsize=7, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8))
    # Subtle red dashed ellipse around the critical (high-priority) cluster.
    nx = [n.x for n in nodes]
    nz = [n.z for n in nodes]
    ax.add_patch(Ellipse(
        (float(np.mean(nx)), float(np.mean(nz))),
        (max(nx) - min(nx)) + 26, (max(nz) - min(nz)) + 16,
        facecolor="none", edgecolor=priority_color("high"),
        linestyle="--", linewidth=1.0, alpha=0.7, zorder=6))
    ax.text(x_center + 14, hover_alt + 22, "high-priority nodes",
            fontsize=7, color=priority_color("high"), ha="left")
    ax.text(x_center + footprint_r * 0.55, 6, "coverage cone", fontsize=7,
            color=UAV_COLOR, ha="left")
    ax.annotate("communication links",
                xy=(x_center - footprint_r * 0.45,
                    (hover_alt + min(n.z for n in nodes)) / 2),
                xytext=(x_lo + 6, 44), fontsize=7, color=LINK_COLOR,
                arrowprops=dict(arrowstyle="->", color=LINK_COLOR, lw=0.8))

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, Z_TOP)
    ax.set_xlabel("Mission Progress")
    ax.set_ylabel("altitude (m)")
    ax.set_title("Priority-aware adaptive-altitude data collection\n"
                 f"({FOCUS_DISTRICT} district of the Fig. 1 city)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


# =============================================================================
# PANEL (b) - CMDP PIPELINE (mirrors the real architecture)
# =============================================================================

def _box(ax, cx, cy, w, h, text, fill, edge, *, bold_title=False, fontsize=7.2):
    """Draw a rounded box centred at (cx, cy); return its center/size."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=3,
    ))
    weight = "bold" if bold_title else "normal"
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, linespacing=1.25, fontweight=weight)
    return dict(cx=cx, cy=cy, w=w, h=h)


def _arrow(ax, p_from, p_to, *, rad=0.0, label=None, color=ARROW_COLOR):
    """Arrow between two box dicts (edge-to-edge), optionally curved/labelled."""
    ax.add_patch(FancyArrowPatch(
        p_from, p_to, arrowstyle="-|>", mutation_scale=11,
        lw=1.2, color=color, zorder=2,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    ))
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx, my, label, fontsize=6.5, color=color, ha="center",
                va="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))


def draw_cmdp_panel(ax):
    """Constrained-MDP block diagram reflecting the ATOM-3D-VoI architecture."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Objective banner.
    ax.text(0.5, 0.985,
            r"maximize  $E\!\left[\sum_i w_i D_i\,\mathbf{1}[\mathrm{QoS}]\right]$"
            r"   s.t.  per-class QoS satisfied,  energy $\leq E_{\max}$",
            ha="center", va="top", fontsize=7.6,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFDF3", ec="#C9B458"))

    bw, bh = 0.42, 0.092
    lx = 0.27                     # left column (policy) centre
    rx = 0.75                     # right column (environment loop) centre

    # Left column: State -> Encoder -> Decoder -> Altitude Head -> CMDP -> Action
    left = [
        ("State  $s_t$\nnode graph: $x_i, z_i$, priority $w_i$, demand $D_i$;\n"
         "remaining battery", STATE_FILL, STATE_EDGE, False),
        ("Graph Encoder\nmulti-head self-attention ($\\times L$)",
         METHOD_FILL, METHOD_EDGE, True),
        ("Trajectory Decoder\npointer attention $\\rightarrow$ next anchor",
         METHOD_FILL, METHOD_EDGE, True),
        ("Altitude Head\nGaussian $H \\in [H_{\\min}, H_{\\max}]$",
         METHOD_FILL, METHOD_EDGE, True),
        ("CMDP Constraint Layer\nper-class QoS Lagrangian duals $\\lambda$",
         METHOD_FILL, METHOD_EDGE, True),
        ("Action  $a_t$\n(next node, altitude $H$)", ACTION_FILL, ACTION_EDGE, True),
    ]
    ys = np.linspace(0.87, 0.10, len(left))
    boxes = [_box(ax, lx, y, bw, bh, t, f, e, bold_title=b)
             for (t, f, e, b), y in zip(left, ys)]
    for a, c in zip(boxes[:-1], boxes[1:]):        # downward arrows
        _arrow(ax, (a["cx"], a["cy"] - bh / 2), (c["cx"], c["cy"] + bh / 2))

    # Right column: Environment -> Reward -> Policy Update (upward loop).
    env = _box(ax, rx, ys[-1], bw, bh,
               "Wireless Environment\nLoS Channel  ·  Rotary-wing Energy\n"
               "Footprint Coverage", ENV_FILL, ENV_EDGE)
    reward = _box(ax, rx, 0.40, bw, bh,
                  r"Reward  $r_t=\sum_i w_i D_i\cdot 1[\mathrm{QoS}]$"
                  "\n" r"$\mathbf{Cost\ Constraint}$: energy $\leq E_{\max}$",
                  REWARD_FILL, REWARD_EDGE)
    update = _box(ax, rx, 0.66, bw, bh,
                  "Policy Update\nprimal-dual actor-critic\n(critic baseline, dual ascent)",
                  UPDATE_FILL, UPDATE_EDGE)

    action = boxes[-1]
    _arrow(ax, (action["cx"] + bw / 2, action["cy"]),
           (env["cx"] - bw / 2, env["cy"]), label="execute")
    _arrow(ax, (env["cx"], env["cy"] + bh / 2), (reward["cx"], reward["cy"] - bh / 2))
    _arrow(ax, (reward["cx"], reward["cy"] + bh / 2),
           (update["cx"], update["cy"] - bh / 2))
    # Learning loop back to the encoder.
    _arrow(ax, (update["cx"] - bw / 2, update["cy"]),
           (boxes[1]["cx"] + bw / 2, boxes[1]["cy"]),
           rad=-0.32, label="gradient / dual update", color=UPDATE_EDGE)

    ax.set_title("ATOM-3D-VoI as a Constrained MDP", fontsize=9)


# =============================================================================
# WORKFLOW RIBBON (data-collection workflow, spans both panels)
# =============================================================================

WORKFLOW_STEPS = [
    "Observe", "Select node + altitude", "Fly & hover",
    "Collect if QoS met", "Update budget", "Repeat / return to depot",
]


def draw_workflow_ribbon(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(WORKFLOW_STEPS)
    slot = 1.0 / n
    cw = slot * 0.82
    for i, step in enumerate(WORKFLOW_STEPS):
        cx = (i + 0.5) * slot
        ax.add_patch(FancyBboxPatch(
            (cx - cw / 2, 0.18), cw, 0.64,
            boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=METHOD_FILL, edgecolor=METHOD_EDGE, linewidth=1.1))
        ax.text(cx, 0.5, step, ha="center", va="center", fontsize=7)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (cx + cw / 2, 0.5), ((i + 1) * slot + 0.09 * slot, 0.5),
                arrowstyle="-|>", mutation_scale=10, lw=1.1, color=ARROW_COLOR))


# =============================================================================
# FIGURE ASSEMBLY
# =============================================================================

def make_figure(city):
    fig = plt.figure(figsize=(11.4, 6.2))
    gs = GridSpec(2, 2, height_ratios=[6.0, 0.7], width_ratios=[1.12, 1.0],
                  hspace=0.28, wspace=0.24, figure=fig)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, :])

    draw_mechanism_panel(ax_a, city)
    draw_cmdp_panel(ax_b)
    draw_workflow_ribbon(ax_w)

    panel(ax_a, "a")
    panel(ax_b, "b")

    fig.suptitle(SUPTITLE, fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.995, 0.005, CONCEPT_NOTE, ha="right", va="bottom",
             fontsize=6.5, style="italic", color="#888888")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.03, top=0.90)
    return fig


def main():
    setup_style()
    city = get_city()
    fig = make_figure(city)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
