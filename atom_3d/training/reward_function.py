"""
Reward functions for ATOM-3D / 2D-AUTO training.

Spec §8: serve-all is the goal, so the base reward is **negative total energy**
(lower energy => higher reward) with a penalty for any node left unserved::

    R = -(Σ_j E_j) / E_scale  -  mu * (unserved fraction)

ATOM-3D-Priority adds a **priority-weighted quality/freshness penalty** so that
critical sensors (strict per-class ``R_min``) are served with a strong link and
early, while low-priority sensors never force the UAV down::

    R = -(Σ_j E_j)/E_scale  -  mu*unserved  -  lambda_priority * priority_penalty

The penalty is normalised to ~[0,1] (see :func:`priority_penalty`) so it is
comparable to the unserved fraction. With ``lambda_priority = 0`` (default) the
reward is identical to the energy-minimisation form, so the 2D baseline and any
priority-off run are unaffected.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardConfig:
    """Knobs for shaping the reward."""
    energy_scale: float = 1.0e4    # E_scale — normalises Joules into an O(1) reward
    unserved_penalty: float = 5.0  # mu — penalty per unit unserved fraction
    lambda_priority: float = 0.0   # weight on the priority penalty (0 => priority off)
    freshness_coeff: float = 0.0   # weight on the freshness sub-term inside the penalty


def priority_penalty(
    weights: np.ndarray,
    rmin: np.ndarray,
    served_mask: np.ndarray,
    rates: np.ndarray,
    serve_order: np.ndarray = None,
    freshness_coeff: float = 0.0,
) -> float:
    """Weighted quality/freshness shortfall for one instance, normalised to ~[0,1].

    A node "misses" when it is left unserved, or served below its class rate floor.
    Misses are weighted by node importance ``w_i`` and normalised by Σ w_i, so a
    missed *critical* node hurts far more than a missed low-priority one.

    Args:
        weights:     (N,) per-node importance w_i (>= 0).
        rmin:        (N,) per-node QoS floor (bits/s); 0 => no floor (never a quality miss).
        served_mask: (N,) bool — True if the node's data was collected.
        rates:       (N,) achieved data rate (bits/s) at the serving hover
                     (value ignored for unserved nodes).
        serve_order: (N,) normalised collection order in [0,1] (0 = collected first),
                     or None to skip the freshness term.
        freshness_coeff: weight on the freshness sub-term.

    Returns:
        Scalar penalty in ~[0, 1 + freshness_coeff]; 0.0 when all weights are 0.
    """
    w = np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    if wsum <= 0:
        return 0.0

    served = np.asarray(served_mask, dtype=bool)
    rmin = np.asarray(rmin, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)

    miss = np.zeros_like(w)
    miss[~served] = 1.0  # unserved node = full miss
    # quality shortfall for served nodes that carry a positive rate floor
    has_floor = served & (rmin > 0)
    shortfall = np.clip((rmin - rates) / np.maximum(rmin, 1e-9), 0.0, 1.0)
    miss[has_floor] = shortfall[has_floor]

    penalty = float((w * miss).sum() / wsum)

    if serve_order is not None and freshness_coeff > 0:
        order = np.asarray(serve_order, dtype=np.float64)
        fr = np.where(served, order, 0.0)
        penalty += freshness_coeff * float((w * fr).sum() / wsum)

    return penalty


def energy_reward(
    total_energy_j: float,
    unserved_fraction: float = 0.0,
    priority_pen: float = 0.0,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Reward for one instance (3D and 2D share this form).

    Args:
        total_energy_j:    Σ_j E_j across all UAVs (Joules).
        unserved_fraction: fraction of nodes never collected (0 when serve-all holds).
        priority_pen:      priority-weighted quality/freshness penalty (see
                           :func:`priority_penalty`); 0 when priority is off.
        config:            scaling / penalty configuration.

    Returns:
        Scalar reward (higher is better).
    """
    return (
        -(total_energy_j / config.energy_scale)
        - config.unserved_penalty * unserved_fraction
        - config.lambda_priority * priority_pen
    )


def reward_to_minimize_energy(
    total_energy_j: float,
    unserved_fraction: float = 0.0,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Explicit alias for the Paper-A 2D baseline (identical energy-min objective)."""
    return energy_reward(total_energy_j, unserved_fraction, 0.0, config)
