"""
figure10_altitude_distribution.py

Figure 10 - Hover-Altitude Distribution (reference panel (g)).

Kernel-density estimate of the hover/service altitudes chosen by each policy
over the mission. Message: 2D-AUTO is a spike at its fixed altitude; 3D-GNN
concentrates in one high-altitude band (it never purposefully descends);
ATOM-3D-VoI is BIMODAL - a cruise mode near 90 m plus a distinct low-altitude
service mode inside the critical-service band, matching the dive behaviour of
Figure 3.

DATA
----
Samples come from `generate_placeholder_altitude_samples`; swap in real hover
altitudes via `load_altitude_samples` (results_data/hover_altitudes.npz) with
no plotting changes.

Runs independently:  python figure10_altitude_distribution.py
Exports results/figure10_altitude_distribution.png (600 DPI) via common_plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from common_style import setup_style, COLORS
from common_plot import save_figure, priority_color, apply_labels


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure10_altitude_distribution"
TITLE = "Hover-Altitude Distribution Across Policies"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "hover_altitudes.npz"

# Altitudes from which the 38 Mbps high-priority floor is satisfiable
# (consistent with the ~30 m dives of Figures 3 and 7).
SERVICE_BAND = (25.0, 40.0)

METHOD_STYLE = [
    ("2d_auto", "2D-AUTO", "#9E9E9E", "--", 1.6),
    ("3d_gnn", "3D-GNN", "#FB8C00", "-.", 1.8),
    ("atom3d", "ATOM-3D-VoI (Ours)", COLORS["ours"], "-", 2.6),
]

# Labels describe whatever actually produced the numbers on disk (the
# exporter's results_data/labels.json); unchanged in placeholder mode.
METHOD_STYLE = apply_labels(METHOD_STYLE)


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real hover logs)
# =============================================================================

def generate_placeholder_altitude_samples(n=1200, seed=31):
    """PLACEHOLDER hover-altitude samples (m) per policy.

    Schema: {method_key: (n,) float array, "placeholder": True}
    Consistent with Figure 3's traces: 2D-AUTO fixed at 100 m (tiny actuator
    jitter); 3D-GNN one band around 80 m; Ours a 65/35 mixture of a ~90 m
    cruise mode and a ~31 m critical-service mode inside SERVICE_BAND.
    """
    rng = np.random.default_rng(seed)
    n_dive = int(0.35 * n)
    ours = np.concatenate([
        rng.normal(90.0, 2.2, n - n_dive),          # cruise mode
        rng.normal(31.0, 2.6, n_dive),              # dive-to-serve mode
    ])
    return {
        # small actuator/wind jitter so the fixed altitude reads as a narrow
        # peak without crushing the shared density scale
        "2d_auto": rng.normal(100.0, 2.5, n),
        "3d_gnn": rng.normal(80.0, 4.0, n),
        "atom3d": ours,
        "placeholder": True,
    }


def load_altitude_samples():
    """Load real hover-altitude logs if present, else the placeholder.

    Real export: results_data/hover_altitudes.npz with one array per method
    key (altitude of every hover/service event across evaluation episodes).
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH)
        data = {k: np.asarray(z[k], float) for k in z.files}
        data["placeholder"] = False
        return data
    return generate_placeholder_altitude_samples()


# =============================================================================
# PLOTTING  (source-agnostic)
# =============================================================================

def plot_altitude_distribution(data):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    # The evaluation grid spans the observed altitudes: the placeholder-era
    # fixed 0-120 m window clipped real cruise altitudes at the config's H_max.
    hi = max(float(np.max(np.asarray(data[k], float))) for k, *_ in METHOD_STYLE)
    grid = np.linspace(0, hi * 1.10, 600)

    peak_density = 0.0
    ours_density = None
    for key, label, color, ls, lw in METHOD_STYLE:
        samples = np.asarray(data[key], float)
        kde = gaussian_kde(samples, bw_method=0.18)
        dens = kde(grid)
        peak_density = max(peak_density, float(dens.max()))
        if key == "atom3d":
            ours_density = dens
        ax.plot(grid, dens, color=color, ls=ls, lw=lw, label=label,
                zorder=6 if key == "atom3d" else 4)
        ax.fill_between(grid, dens, color=color,
                        alpha=0.15 if key == "atom3d" else 0.08, zorder=2)

    # Critical-service altitude band (label placed high, clear of the curves).
    ax.axvspan(*SERVICE_BAND, color=priority_color("high"), alpha=0.07,
               zorder=0)
    ax.text(np.mean(SERVICE_BAND), peak_density * 1.12, "critical-service\naltitudes",
            fontsize=7.5, color=priority_color("high"), ha="center",
            va="top")

    # Point out Ours' two modes, located from the actual density rather than
    # from hard-coded coordinates, so the arrows follow the data. Genuine local
    # maxima are used: splitting the range at a fixed altitude would land the
    # upper arrow on the dive mode's shoulder instead of the cruise mode.
    if ours_density is not None:
        peaks, props = find_peaks(ours_density, prominence=peak_density * 0.05)
        if peaks.size >= 2:
            order = np.argsort(props["prominences"])[::-1][:2]
            lo_i, hi_i = sorted(peaks[order])
            dive_x, dive_y = float(grid[lo_i]), float(ours_density[lo_i])
            cruise_x, cruise_y = float(grid[hi_i]), float(ours_density[hi_i])
            ax.annotate("dive-to-serve mode\n(only Ours)", xy=(dive_x, dive_y),
                        xytext=(dive_x + hi * 0.06, dive_y + peak_density * 0.22),
                        fontsize=8, color=COLORS["ours"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=1.0))
            ax.annotate("cruise mode", xy=(cruise_x, cruise_y),
                        xytext=(cruise_x - hi * 0.22, cruise_y + peak_density * 0.16),
                        fontsize=8, color=COLORS["ours"],
                        arrowprops=dict(arrowstyle="->", color=COLORS["ours"], lw=1.0))

    ax.set_xlabel("Hover Altitude (m)")
    ax.set_ylabel("Density")
    # Limits follow the observed hover altitudes and KDE peak; the
    # placeholder-era fixed limits clipped both.
    ax.set_xlim(0, hi * 1.10)
    ax.set_ylim(0, max(peak_density * 1.25, 1e-3))
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, borderpad=0.5)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.11)
    return fig


def main():
    setup_style()
    data = load_altitude_samples()
    fig = plot_altitude_distribution(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
