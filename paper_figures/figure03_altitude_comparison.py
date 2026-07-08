"""
figure03_altitude_comparison.py

Figure 3 - Adaptive Altitude Behaviour Across UAV Policies.

Altitude vs. mission progress for the three policies (2D-AUTO, 3D-GNN,
ATOM-3D-VoI). The story must land in <5 s: only ATOM-3D-VoI intentionally
descends at high-priority service events (its dives line up with the red
critical-event dots and the faintly shaded serving intervals) to satisfy the
high-priority 38 Mbps QoS floor, while 3D-GNN merely wobbles and 2D-AUTO is
flat.

DATA
----
All trace/event data is produced by a SINGLE isolated function,
`generate_placeholder_altitude_traces`, and consumed by the plotting code
through a fixed schema (see `load_altitude_traces`). When real experiment
outputs exist at `results_data/altitude_traces.pkl`, they are loaded instead
and the plotting code is used unchanged. No final metrics are fabricated.

Runs independently:  python figure03_altitude_comparison.py
Exports results/figure03_altitude_comparison.png (600 DPI) via common_plot.
Reuses locked infrastructure only (common_style, common_plot).
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from common_style import setup_style, COLORS
from common_plot import save_figure, priority_color


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure03_altitude_comparison"
TITLE = "Adaptive Altitude Behaviour Across UAV Policies"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "altitude_traces.pkl"

# Physical altitude constants (m) for the placeholder policies.
CRUISE_ALT = 90.0          # ATOM-3D-VoI nominal cruise
HOVER_ALT = 30.0           # ATOM-3D-VoI nominal dive-to-serve altitude
FIXED_2D_ALT = 100.0       # 2D-AUTO constant altitude
GNN_MEAN_ALT = 80.0        # 3D-GNN mean altitude (small oscillation)

# Per-class QoS rate floors (Mbps) - the WHY behind the dives (see DATA_SPEC).
QOS_RATE = {"high": 38, "medium": 25, "low": 8}

# Visual hierarchy: Ours dominant, 3D-GNN second, 2D-AUTO least.
# (key, label, color, linewidth, alpha, z-order)
METHOD_STYLE = [
    ("atom3d", "ATOM-3D-VoI (Ours)", COLORS["ours"], 3.0, 1.00, 6),
    ("3d_gnn", "3D-GNN",             "#FB8C00",       1.9, 0.95, 5),
    ("2d_auto", "2D-AUTO",           "#9E9E9E",       1.1, 0.90, 4),
]

DOT_Y = -5.0               # priority dots sit just BELOW the x-axis


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real outputs)
# =============================================================================

def _smoothstep(t):
    """Cubic smoothstep on [0, 1] (clamped), for physically smooth edges."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _dive_well(x, center, plateau_hw, trans_down, trans_up):
    """Smooth flat-bottomed 'well' in [0,1] with independent descent/ascent
    widths (a real dive descends a little slower than it climbs)."""
    left_edge, right_edge = center - plateau_hw, center + plateau_hw
    well = np.ones_like(x)
    left = x < left_edge
    right = x > right_edge
    well[left] = 1.0 - _smoothstep((left_edge - x[left]) / trans_down)
    well[right] = 1.0 - _smoothstep((x[right] - right_edge) / trans_up)
    return well


def generate_placeholder_altitude_traces(n=1400, seed=7):
    """Return physically-plausible PLACEHOLDER traces + events (one function).

    Schema (identical to what the real experiment export must provide):
        {
          "progress": (n,) float   - mission progress in percent [0, 100],
          "traces":   {"2d_auto":(n,), "3d_gnn":(n,), "atom3d":(n,)} altitude m,
          "events":   [ {"progress": float, "priority": "high|medium|low"} ],
          "critical_intervals": [ (start, end) ]  serving-a-high-priority ranges,
          "placeholder": True,
        }
    Behaviour encoded (paper formulation): the high-priority 38 Mbps floor can
    only be met from low altitude, so ATOM-3D-VoI dives at the HIGH events;
    medium/low floors are met from cruise, so no dive there. Each dive is
    slightly different (depth, dwell, descent rate) so it reads like a real UAV.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 100.0, n)

    events = [
        {"progress": 12.0, "priority": "high"},
        {"progress": 26.0, "priority": "medium"},
        {"progress": 38.0, "priority": "high"},
        {"progress": 50.0, "priority": "low"},
        {"progress": 66.0, "priority": "high"},
        {"progress": 78.0, "priority": "medium"},
        {"progress": 88.0, "priority": "high"},
    ]

    # --- 2D-AUTO: perfectly constant altitude ---
    auto = np.full_like(x, FIXED_2D_ALT)

    # --- 3D-GNN: small, smooth oscillation (+-4-5 m), unrelated to events ---
    gnn = (GNN_MEAN_ALT
           + 2.6 * np.sin(2 * np.pi * x / 41.0 + 0.6)
           + 1.6 * np.sin(2 * np.pi * x / 17.0 + 1.3)
           + 1.0 * np.sin(2 * np.pi * x / 67.0 + rng.uniform(0, 2 * np.pi)))

    # --- ATOM-3D-VoI: cruise, diving to serve each HIGH event (each unique) ---
    reduction = np.zeros_like(x)
    critical_intervals = []
    for e in events:
        if e["priority"] != "high":
            continue
        c = e["progress"]
        hover = HOVER_ALT + rng.uniform(-1.5, 4.0)          # ~28.5 - 34 m
        plateau_hw = rng.uniform(2.5, 5.0)                  # different dwell
        trans_down = rng.uniform(6.5, 8.5)                  # smooth descent
        trans_up = rng.uniform(5.0, 7.0)                    # slightly faster climb
        well = _dive_well(x, c, plateau_hw, trans_down, trans_up)
        reduction = np.maximum(reduction, (CRUISE_ALT - hover) * well)
        critical_intervals.append((c - (plateau_hw + 2.5), c + (plateau_hw + 2.5)))
    atom = CRUISE_ALT - reduction
    # tiny smooth ripple so cruise/hover are not dead-flat (still readable)
    atom = atom + 0.7 * np.sin(2 * np.pi * x / 21.0 + 0.4)

    return {
        "progress": x,
        "traces": {"2d_auto": auto, "3d_gnn": gnn, "atom3d": atom},
        "events": events,
        "critical_intervals": critical_intervals,
        "placeholder": True,
    }


def load_altitude_traces():
    """Load real traces if present, else fall back to the placeholder.

    To use real experiment output, write a pickle at REAL_DATA_PATH following
    the schema documented in `generate_placeholder_altitude_traces`. Nothing
    in the plotting code below needs to change.
    """
    if REAL_DATA_PATH.exists():
        with open(REAL_DATA_PATH, "rb") as f:
            data = pickle.load(f)
        data.setdefault("placeholder", False)
        return data
    return generate_placeholder_altitude_traces()


# =============================================================================
# PLOTTING  (unchanged whether data is placeholder or real)
# =============================================================================

def _plot_traces(ax, data, *, scale=1.0):
    x = data["progress"]
    for key, label, color, lw, alpha, z in METHOD_STYLE:
        ax.plot(x, data["traces"][key], color=color, lw=lw * scale, alpha=alpha,
                zorder=z, solid_capstyle="round", label=label)


def _shade_serving(ax, data, *, alpha=0.048):
    """Very light background shading over high-priority serving intervals."""
    for (s, e) in data["critical_intervals"]:
        ax.axvspan(s, e, color=priority_color("high"), alpha=alpha, zorder=0)


def _draw_events(ax, data):
    """Priority dots just BELOW the axis + faint guides at high events."""
    for ev in data["events"]:
        c, pr = ev["progress"], ev["priority"]
        ax.scatter(c, DOT_Y, s=40, color=priority_color(pr), edgecolors="black",
                   linewidths=0.4, zorder=9, clip_on=False)
        if pr == "high":
            ax.plot([c, c], [DOT_Y, CRUISE_ALT], color=priority_color("high"),
                    lw=0.7, ls=(0, (2, 3)), alpha=0.30, zorder=2, clip_on=False)


def _annotate_one_descent(ax, data):
    """Label a single representative adaptive descent with the QoS reason."""
    c = [e["progress"] for e in data["events"] if e["priority"] == "high"][1]
    xs, ys = data["progress"], data["traces"]["atom3d"]
    xp = c - 6.0
    yp = float(np.interp(xp, xs, ys))
    ax.annotate("Adaptive descent\nto satisfy 38 Mbps QoS",
                xy=(xp, yp), xytext=(c - 27, 74),
                fontsize=8, color=COLORS["ours"], ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=1.0))


def _legends(ax):
    """Compact method legend (hierarchy order) + priority key WITH QoS rates."""
    method_handles = [
        mlines.Line2D([], [], color=color, lw=max(lw, 2.0), label=label)
        for (_key, label, color, lw, _a, _z) in METHOD_STYLE
    ]
    leg1 = ax.legend(handles=method_handles, loc="upper right",
                     bbox_to_anchor=(0.995, 0.995), framealpha=0.92,
                     fontsize=7.5, title="Policy", title_fontsize=7.5,
                     handlelength=1.6, borderpad=0.5, labelspacing=0.35)
    leg1.get_title().set_fontweight("bold")
    ax.add_artist(leg1)

    dot_handles = [
        mlines.Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor=priority_color(p), markeredgecolor="black",
                      markeredgewidth=0.4, markersize=6,
                      label=f"{p.capitalize()}  ({QOS_RATE[p]} Mbps)")
        for p in ("high", "medium", "low")
    ]
    leg2 = ax.legend(handles=dot_handles, loc="upper left",
                     bbox_to_anchor=(0.005, 0.995), framealpha=0.92,
                     fontsize=7, title="Served-node priority (QoS floor)",
                     title_fontsize=7, borderpad=0.5, labelspacing=0.35)
    leg2.get_title().set_fontweight("bold")


def make_figure(data):
    fig, ax = plt.subplots(figsize=(9.2, 4.9))

    _shade_serving(ax, data)
    _plot_traces(ax, data)
    _draw_events(ax, data)
    _annotate_one_descent(ax, data)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Mission Progress (%)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.85", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", pad=14)   # push tick labels below the dot row

    _legends(ax)

    fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.16)
    return fig


def main():
    setup_style()
    data = load_altitude_traces()
    fig = make_figure(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
