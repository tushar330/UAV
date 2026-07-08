"""
figure11_dual_convergence.py

Figure 11 - Dual-Variable Convergence of the CMDP (reference panel (h)).

Evolution of the per-class Lagrangian dual multipliers (lambda_high,
lambda_medium) during training. Message: the duals rise while the class
constraints are violated, then settle automatically at the level needed to
enforce each QoS floor - no manual reward-weight tuning. lambda_high settles
higher than lambda_medium because the 38 Mbps floor is harder to satisfy.
Stabilization coincides with the convergence epoch of Figure 4.

DATA
----
Curves come from `generate_placeholder_dual_data`; swap in real dual logs via
`load_dual_results` (results_data/dual_variables.npz) with no plotting changes.

Runs independently:  python figure11_dual_convergence.py
Exports results/figure11_dual_convergence.png (600 DPI) via common_plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common_style import setup_style, COLORS
from common_plot import save_figure, priority_color


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure11_dual_convergence"
TITLE = "Dual-Variable Convergence of the CMDP"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "dual_variables.npz"

N_EPOCHS = 300
CONVERGENCE_EPOCH = 130       # same guide line as Figure 4

DUAL_STYLE = [
    ("lambda_high", r"$\lambda_{\mathrm{high}}$  (QoS $\geq$ 38 Mbps)",
     priority_color("high"), "-"),
    ("lambda_medium", r"$\lambda_{\mathrm{medium}}$  (QoS $\geq$ 25 Mbps)",
     priority_color("medium"), "--"),
]


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real dual logs)
# =============================================================================

def generate_placeholder_dual_data(n=N_EPOCHS, seed=17):
    """PLACEHOLDER dual-variable trajectories from primal-dual training.

    Schema: {"epoch": (n,), "lambda_high": (n,), "lambda_medium": (n,),
             "convergence_epoch": int, "placeholder": True}
    Shape: rapid dual ascent while the constraint is violated, overshoot,
    then settling. lambda_high stabilizes higher (harder floor); noise decays
    as the violations shrink.
    """
    rng = np.random.default_rng(seed)
    e = np.arange(n)

    def dual(final, rise, overshoot, noise0):
        base = final * (1.0 - np.exp(-e / rise))
        bump = overshoot * (e / 45.0) * np.exp(-e / 45.0)   # early overshoot
        noise = rng.normal(0, noise0 * np.exp(-e / 90.0) + 0.01, n)
        return np.maximum(base + bump + noise, 0.0)

    return {
        "epoch": e,
        "lambda_high": dual(final=1.80, rise=38.0, overshoot=0.85, noise0=0.10),
        "lambda_medium": dual(final=0.85, rise=48.0, overshoot=0.40, noise0=0.06),
        "convergence_epoch": CONVERGENCE_EPOCH,
        "placeholder": True,
    }


def load_dual_results():
    """Load real dual logs if present, else the placeholder.

    Real export: results_data/dual_variables.npz with epoch, lambda_high,
    lambda_medium (from the trainer's LagrangianDuals state per epoch).
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH)
        data = {k: np.asarray(z[k]) for k in z.files}
        data.setdefault("convergence_epoch", CONVERGENCE_EPOCH)
        data["placeholder"] = False
        return data
    return generate_placeholder_dual_data()


# =============================================================================
# PLOTTING  (source-agnostic)
# =============================================================================

def plot_dual_convergence(data):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    e = data["epoch"]

    for key, label, color, ls in DUAL_STYLE:
        ax.plot(e, np.asarray(data[key], float), color=color, ls=ls, lw=2.0,
                label=label)

    conv = int(data["convergence_epoch"])
    ax.axvline(conv, color="0.5", ls=(0, (5, 4)), lw=1.0, alpha=0.7, zorder=1)
    ax.text(conv + 4, 0.06, "approx. convergence (Fig. 4)", rotation=90,
            fontsize=7, color="0.45", va="bottom")

    ax.annotate("duals adapt automatically to enforce each QoS floor -\n"
                "no manual reward-weight tuning",
                xy=(0.98, 0.20), xycoords="axes fraction", ha="right",
                fontsize=8.5, style="italic", color="0.35")
    ax.annotate(r"$\lambda_{\mathrm{high}} > \lambda_{\mathrm{medium}}$: "
                "the 38 Mbps floor is harder to satisfy",
                xy=(0.98, 0.10), xycoords="axes fraction", ha="right",
                fontsize=8, color="0.35")

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel(r"Dual Variable $\lambda$")
    ax.set_xlim(0, int(e[-1]))
    ax.set_ylim(bottom=0)
    ax.set_title(TITLE, fontsize=12, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.90", lw=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92, borderpad=0.5)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.11)
    return fig


def main():
    setup_style()
    data = load_dual_results()
    fig = plot_dual_convergence(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
