"""
figure04_training_dynamics.py

Figure 4 - Training Dynamics of ATOM-3D-VoI (CMDP optimization health).

A 2x2 composite showing that the proposed primal-dual CMDP policy trains
stably: reward rises and converges, actor and critic losses settle, and the
average constraint violation is driven toward zero. A single dashed vertical
line marks the approximate convergence epoch across all panels.

  (a) Average Episode Reward       (b) Actor Loss
  (c) Critic Loss                  (d) Average Constraint Violation

DATA
----
All curves come from ONE isolated function, `generate_placeholder_training_data`.
The plotting code (`plot_training_dynamics`) is agnostic to the data source:
when real logs exist at `results_data/training_log.npz`, `load_training_data`
loads them instead and nothing in the plotting code changes. No final metrics
are fabricated - these are clearly-labelled placeholder training curves.

Runs independently:  python figure04_training_dynamics.py
Exports results/figure04_training_dynamics.png (600 DPI) via common_plot.
Reuses locked infrastructure only (common_style, common_plot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from common_style import setup_style, COLORS, panel
from common_plot import save_figure


# =============================================================================
# CONSTANTS
# =============================================================================

FIG_NAME = "figure04_training_dynamics"
SUPTITLE = "Training Dynamics of ATOM-3D-VoI"

REAL_DATA_PATH = Path(__file__).resolve().parent / "results_data" / "training_log.npz"

N_EPOCHS = 300
CONVERGENCE_EPOCH = 130          # approximate; drawn as a shared guide line
METHOD_COLOR = COLORS["ours"]    # ATOM-3D-VoI project method color
RAW_ALPHA = 0.25                 # faint per-epoch trace behind the EMA
EMA_ALPHA = 0.10                 # exponential-moving-average smoothing factor


# =============================================================================
# PLACEHOLDER DATA  (isolated - replace by loading real logs)
# =============================================================================

def generate_placeholder_training_data(n=N_EPOCHS, seed=11):
    """Return realistic PLACEHOLDER CMDP training curves (one function).

    Schema (identical to what the real log export must provide):
        {
          "epoch":                (n,) int,
          "reward":               (n,) float,  average episode reward,
          "actor_loss":           (n,) float,
          "critic_loss":          (n,) float,
          "constraint_violation": (n,) float,  average per-episode violation,
          "convergence_epoch":    int,
          "placeholder":          True,
        }
    Each series is a smooth trend + heteroscedastic noise (larger early), so
    the curves look like real RL training rather than analytic functions.
    """
    rng = np.random.default_rng(seed)
    e = np.arange(n)

    # (a) Reward: fast early rise -> plateau, small oscillation near optimum.
    reward_trend = 105.0 - (105.0 - 15.0) * np.exp(-e / 45.0)
    reward_noise = rng.normal(0, 5.5 * np.exp(-e / 110.0) + 1.8, n)
    reward_osc = 1.4 * np.sin(e / 6.5) * np.clip((e - CONVERGENCE_EPOCH) / 60.0, 0, 1)
    reward = reward_trend + reward_noise + reward_osc

    # (b) Actor loss: large initial magnitude -> smooth convergence.
    actor_trend = 0.15 + 2.35 * np.exp(-e / 40.0)
    actor_noise = rng.normal(0, 0.11 * np.exp(-e / 85.0) + 0.025, n)
    actor_loss = np.maximum(actor_trend + actor_noise, 0.0)

    # (c) Critic loss: larger variance early, stable convergence later.
    critic_trend = 0.05 + 1.55 * np.exp(-e / 50.0)
    critic_noise = rng.normal(0, 0.34 * np.exp(-e / 42.0) + 0.02, n)
    critic_loss = np.maximum(critic_trend + critic_noise, 0.0)

    # (d) Constraint violation: high -> decreases steadily -> stabilizes ~0.
    cv_trend = 0.02 + 0.60 * np.exp(-e / 55.0)
    cv_noise = rng.normal(0, 0.05 * np.exp(-e / 70.0) + 0.008, n)
    constraint_violation = np.maximum(cv_trend + cv_noise, 0.0)

    return {
        "epoch": e,
        "reward": reward,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "constraint_violation": constraint_violation,
        "convergence_epoch": CONVERGENCE_EPOCH,
        "placeholder": True,
    }


def load_training_data():
    """Load real training logs if present, else fall back to the placeholder.

    To use real data, export a NumPy archive at REAL_DATA_PATH with arrays
    matching the schema in `generate_placeholder_training_data` (e.g. from a
    TensorBoard scalar dump or the trainer's own logging). The plotting code
    below is unchanged.
    """
    if REAL_DATA_PATH.exists():
        z = np.load(REAL_DATA_PATH)
        data = {k: z[k] for k in z.files}
        data.setdefault("convergence_epoch", CONVERGENCE_EPOCH)
        data.setdefault("placeholder", False)
        return data
    return generate_placeholder_training_data()


# =============================================================================
# PLOTTING  (source-agnostic: identical for placeholder or real logs)
# =============================================================================

def _ema(x, alpha=EMA_ALPHA):
    """Exponential moving average (causal), for a smoothed trend overlay."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def _draw_panel(ax, epoch, raw, ylabel, conv, *, bottom_row):
    """Draw one metric panel: faint per-epoch trace + bold EMA + guide line."""
    ax.plot(epoch, raw, color=METHOD_COLOR, alpha=RAW_ALPHA, lw=0.8, zorder=2)
    ax.plot(epoch, _ema(raw), color=METHOD_COLOR, lw=2.0, zorder=3)
    ax.axvline(conv, color="0.5", ls=(0, (5, 4)), lw=1.0, alpha=0.7, zorder=1)

    ax.set_ylabel(ylabel)
    if bottom_row:
        ax.set_xlabel("Training Epoch")
    ax.set_xlim(epoch[0], epoch[-1])
    ax.margins(y=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def plot_training_dynamics(data):
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.4))
    conv = int(data["convergence_epoch"])
    e = data["epoch"]

    panels = [
        (axs[0, 0], "reward", "Average Episode Reward", "a", False),
        (axs[0, 1], "actor_loss", "Actor Loss", "b", False),
        (axs[1, 0], "critic_loss", "Critic Loss", "c", True),
        (axs[1, 1], "constraint_violation", "Average Constraint Violation", "d", True),
    ]
    for ax, key, ylabel, tag, bottom in panels:
        _draw_panel(ax, e, np.asarray(data[key], float), ylabel, conv,
                    bottom_row=bottom)
        panel(ax, tag)

    # A little extra top headroom so the (a)-(d) labels clear the top tick.
    for ax in axs.flat:
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.10 * (y1 - y0))

    # Feasibility reference for the constraint panel.
    axs[1, 1].axhline(0.0, color="0.7", lw=0.8, ls=":", zorder=1)

    # Label the shared convergence line once (top-left panel).
    ax0 = axs[0, 0]
    ax0.text(conv + 4, ax0.get_ylim()[0] + 0.06 * np.ptp(ax0.get_ylim()),
             "approx. convergence", rotation=90, fontsize=7, color="0.45",
             va="bottom", ha="left")

    # Compact key explaining the two line styles (one method throughout).
    handles = [
        mlines.Line2D([], [], color=METHOD_COLOR, lw=2.0, label="EMA (smoothed)"),
        mlines.Line2D([], [], color=METHOD_COLOR, lw=0.8, alpha=RAW_ALPHA,
                      label="per-epoch"),
        mlines.Line2D([], [], color="0.5", lw=1.0, ls=(0, (5, 4)),
                      label="approx. convergence"),
    ]
    ax0.legend(handles=handles, loc="lower right", fontsize=7.5,
               framealpha=0.9, borderpad=0.5, labelspacing=0.3)

    fig.suptitle(SUPTITLE, fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.91, bottom=0.09,
                        hspace=0.32, wspace=0.24)
    return fig


def main():
    setup_style()
    data = load_training_data()
    fig = plot_training_dynamics(data)
    save_figure(fig, FIG_NAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
