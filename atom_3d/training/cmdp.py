"""Primal-dual CMDP scaffolding for the frozen QoS-constrained energy problem.

The optimization problem is UNCHANGED (PROBLEM_FORMULATION §6): minimize total UAV
energy subject to the per-class QoS constraint ``R_{ij} >= R_min``. This module only
provides the *solver* machinery — it does not alter the objective, the scorer, or the
constraint.

Formulation mapping (standard constrained-MDP primal-dual):
  * constraint per QoS class c:   E[ satisfaction_c ] >= target
        <=>   g_c = target - satisfaction_c <= 0
    where satisfaction_c is the fraction of class-c floored nodes meeting their R_min
    (exactly the scorer's existing per-class metric — no new measurement, no new
    constraint).
  * Lagrangian (minimization):     L = E[energy] + Σ_c λ_c · E[g_c],   λ_c >= 0
    -> primal reward (maximize):   R = -energy/scale - μ·unserved - Σ_c λ_c · g_c
  * dual ascent (projected):       λ_c <- clip( λ_c + η_λ · E[g_c], 0, λ_max )

When the constraint is violated (satisfaction_c < target) g_c > 0, so λ_c rises and
the policy is pushed to serve class c better; once satisfied, g_c <= 0 drives λ_c back
toward 0. This replaces the FIXED ``reward.lambda_priority`` penalty with multipliers
the dual step learns. Nothing here is tuned — defaults are placeholders for Colab.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class CMDPConfig:
    """Primal-dual solver hyperparameters (UNTUNED defaults; set on Colab)."""
    enabled: bool = False
    dual_lr: float = 0.05               # η_λ — dual-ascent step size
    dual_init: float = 0.0              # initial multiplier λ_c(0)
    dual_max: float = 100.0             # projection clamp: λ_c ∈ [0, dual_max]
    target_satisfaction: float = 1.0    # per-class QoS satisfaction target (formulation: 1.0)

    @classmethod
    def from_params(cls, params: dict, enabled: bool) -> "CMDPConfig":
        c = dict(params.get("cmdp", {}) or {})
        return cls(
            enabled=enabled,
            dual_lr=float(c.get("dual_lr", 0.05)),
            dual_init=float(c.get("dual_init", 0.0)),
            dual_max=float(c.get("dual_max", 100.0)),
            target_satisfaction=float(c.get("target_satisfaction", 1.0)),
        )


class LagrangianDuals:
    """Per-class QoS dual multipliers λ_c >= 0 with projected dual ascent.

    ``class_names`` lists only the QoS-CONSTRAINED classes (those with a positive
    rate floor); a class without a floor has no constraint and therefore no dual.
    """

    def __init__(self, class_names: List[str], cfg: CMDPConfig):
        self.class_names = list(class_names)
        self.cfg = cfg
        self.lmbda: Dict[str, float] = {c: float(cfg.dual_init) for c in self.class_names}

    # -- constraint violation g_c = target - satisfaction_c --------------------
    def violation(self, satisfaction: Dict[str, float]) -> Dict[str, float]:
        """g_c for each class from a {class: satisfaction} map.

        A NaN/None satisfaction (instance/batch had no class-c node) means the
        constraint is inactive there -> violation 0 (no dual contribution).
        """
        out: Dict[str, float] = {}
        for c in self.class_names:
            s = satisfaction.get(c, np.nan)
            out[c] = 0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) \
                else float(self.cfg.target_satisfaction - s)
        return out

    # -- projected dual ascent: λ_c <- clip(λ_c + η·g_c, 0, λ_max) --------------
    def ascent(self, mean_violation: Dict[str, float]) -> None:
        for c in self.class_names:
            g = float(mean_violation.get(c, 0.0))
            stepped = self.lmbda[c] + self.cfg.dual_lr * g
            self.lmbda[c] = float(min(max(stepped, 0.0), self.cfg.dual_max))

    def as_dict(self) -> Dict[str, float]:
        return dict(self.lmbda)

    # -- checkpoint round-trip --------------------------------------------------
    def state_dict(self) -> dict:
        return {"lmbda": dict(self.lmbda), "class_names": list(self.class_names)}

    def load_state_dict(self, sd: dict) -> None:
        for c, v in (sd.get("lmbda", {}) or {}).items():
            if c in self.lmbda:
                self.lmbda[c] = float(v)
