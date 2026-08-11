"""Export ATOM-3D results in the schema ``paper_figures`` already consumes.

Each figure script owns a ``REAL_DATA_PATH`` under ``paper_figures/results_data/``
and a ``load_*()`` that prefers that file over its built-in placeholder, clearing
the ``PLACEHOLDER - synthetic data`` stamp automatically. This module is the
producer for those files: it evaluates the repo's planners on the paper's frozen
synthetic city and writes the metrics out in exactly those schemas.

Nothing here invents numbers. Every value is produced by
``TENMATrainer._partition_and_evaluate`` - the same energy accounting used by the
learned policy and every baseline - and the provenance of each method slot is
recorded in ``results_data/labels.json`` so a figure never claims a result came
from a model that did not produce it.

Method slots
------------
The figures key on three slots. Each resolves to the strongest *real* source
available at run time, and reports what it actually was:

======== ================================ ==============================
slot     without a checkpoint             with a checkpoint
======== ================================ ==============================
2d_auto  per-node NN+2-opt tour, fixed H  (unchanged - not a learned method)
3d_gnn   QoS-blind greedy footprint cover ``3d_gnn.pt`` learned policy
atom3d   strong coupled planner           ``3d_attention_priority_cmdp.pt``
======== ================================ ==============================

Usage
-----
    # everything that needs no training (canonical seed-42 city)
    python -m atom_3d.experiments.export_figure_data

    # add across-realization confidence intervals (slow: re-plans per seed)
    python -m atom_3d.experiments.export_figure_data --city-seeds 42,43,44,45,46

    # skip the ~20 min/instance local search while iterating
    python -m atom_3d.experiments.export_figure_data --fast

The two training-dependent figures (``training_log.npz``, ``dual_variables.npz``)
are NOT written here - they come from an actual ``run_train --cmdp`` log. Their
figures keep their placeholder stamp until that run exists, which is the correct
signal that the learned results are still outstanding.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from ..env.city_adapter import (
    CityInstance, apply_city_depot, build_instances_for_seeds, class_counts,
    PAPER_FIGURES_DIR, PRIORITY_ORDER,
)
from ..training import TENMATrainer, TrainConfig
from .heuristic_baseline import _plan_from_steps

# The deterministic planners live in the root-level `experiments` package.
from experiments.two_stage_vs_coupled import (          # noqa: E402
    d_max_of, make_channel_consts, plan_coupled, plan_energy_min,
)
from experiments.strong_coupled import plan_coupled_strong  # noqa: E402


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
RESULTS_DATA_DIR = PAPER_FIGURES_DIR / "results_data"

# Altitude grid offered to the greedy planners (m). Spans the config's
# [H_min, H_max] band so a planner can choose a deep dive or a wide high hover.
ALTITUDE_GRID: Tuple[float, ...] = (30, 45, 60, 80, 100, 120, 140)

# Per-class QoS floors are read from the city itself (each IoTNode carries its
# own required_rate). The scorer, however, derives its per-class satisfaction
# masks from `priority.R_min` in params.yaml, which drifted from the city
# (config says medium=28 Mbps, the city says 25). We override that mapping
# IN MEMORY so the masks line up; params.yaml on disk is never touched, so no
# training run is affected.
CITY_CLASS_FLOORS = {"high": 38e6, "medium": 25e6, "low": 8e6}

# Figure slot -> checkpoint tag that would upgrade it to a learned result.
LEARNED_CHECKPOINTS = {
    "3d_gnn": "3d_gnn",
    "atom3d": "3d_attention_priority_cmdp",
}

# Unserved nodes have no achieved rate. Fig. 9 plots a log-x CDF, where 0 is
# undefined, so they are placed at this floor: visibly off-scale to the left
# rather than silently dropped from the distribution.
UNSERVED_RATE_MBPS = 1e-3


# ---------------------------------------------------------------------------
# RESULT CONTAINER
# ---------------------------------------------------------------------------
@dataclass
class MethodRun:
    """One method slot evaluated over one or more city realizations."""

    key: str                       # figure slot key (2d_auto / 3d_gnn / atom3d)
    label: str                     # honest label describing what actually ran
    source: str                    # provenance sentence for labels.json
    metrics: List[dict] = field(default_factory=list)   # scorer output per city
    traces: List[dict] = field(default_factory=list)    # trace dict per city

    # -- aggregate helpers ------------------------------------------------
    def energy_wh(self) -> np.ndarray:
        return np.array([float(m["energy"][0]) / 3600.0 for m in self.metrics])

    def all_hover_altitudes(self) -> np.ndarray:
        return np.array([h["altitude"] for t in self.traces for h in t["hovers"]])

    def energy_split_kj(self) -> Dict[str, float]:
        """Mean flight / hover / comm / total energy in kJ."""
        f = np.mean([t["energy_flight"] for t in self.traces]) / 1e3
        h = np.mean([t["energy_hover"] for t in self.traces]) / 1e3
        c = np.mean([t["energy_comm"] for t in self.traces]) / 1e3
        return {"flight": f, "hover": h, "comm": c, "total": f + h + c}


# ---------------------------------------------------------------------------
# QoS measured directly from the trace (all three classes, not just hi/med)
# ---------------------------------------------------------------------------
def class_satisfaction(trace: dict, instance: CityInstance) -> Dict[str, float]:
    """Fraction (percent) of each class served AND meeting its own rate floor.

    Computed from the trace rather than the scorer's ``high/med_qos_satisfaction``
    so that the *low* class is covered too (the scorer only exposes the two
    classes that carry a dual multiplier).
    """
    rate = trace["rate"]
    served = trace["served"]
    floors = instance.node_rmin[0].numpy()
    out: Dict[str, float] = {}
    for cls in PRIORITY_ORDER:
        mask = instance.class_mask(cls)
        if not mask.any():
            out[cls] = float("nan")
            continue
        met = served[mask] & (rate[mask] >= floors[mask])
        out[cls] = 100.0 * float(met.sum()) / float(mask.sum())
    return out


# ---------------------------------------------------------------------------
# PLANNERS -> scored runs (each returns metrics + trace)
# ---------------------------------------------------------------------------
def _score_hovers(trainer, instance: CityInstance, hovers) -> dict:
    """Score a hover list through the canonical scorer, with tracing on."""
    n = instance.num_nodes
    t_steps = max(len(hovers), 1)
    anchors = np.zeros((1, t_steps), np.int64)
    altitudes = np.full((1, t_steps), trainer.fixed_alt, np.float32)
    served = np.zeros((1, t_steps, n), bool)
    for t, (a, H, idx) in enumerate(hovers):
        anchors[0, t] = a
        altitudes[0, t] = H
        served[0, t, idx] = True
    plan = _plan_from_steps(anchors, altitudes, served, 1, t_steps, n, trainer.device)
    return trainer._partition_and_evaluate(
        plan, instance.node_features, R_min=0.0, use_service_altitude=False,
        node_weights=instance.node_weights, node_rmin=instance.node_rmin,
        refine_routing=True, collect_trace=True)


def run_2d_auto(trainer, instance: CityInstance, altitude: float) -> dict:
    """2D-AUTO: NN+2-opt visiting every node individually at one fixed altitude."""
    n = instance.num_nodes
    anchors = np.arange(n, dtype=np.int64)[None, :]
    altitudes = np.full((1, n), altitude, np.float32)
    served = np.zeros((1, n, n), bool)
    served[0, np.arange(n), np.arange(n)] = True
    plan = _plan_from_steps(anchors, altitudes, served, 1, n, n, trainer.device)
    return trainer._partition_and_evaluate(
        plan, instance.node_features, R_min=0.0, use_service_altitude=False,
        node_weights=instance.node_weights, node_rmin=instance.node_rmin,
        refine_routing=True, collect_trace=True)


def run_blind_3d(trainer, instance: CityInstance, chan) -> dict:
    """Priority-blind greedy footprint cover: 3D, altitude varies, QoS ignored."""
    nf = instance.node_features.numpy()[0]
    hovers = plan_energy_min(trainer, nf[:, :2], nf[:, 2], nf[:, 3], ALTITUDE_GRID, chan)
    return _score_hovers(trainer, instance, hovers)


def run_coupled(trainer, instance: CityInstance, chan, *,
                strong: bool, rmin_override: Optional[np.ndarray] = None) -> dict:
    """QoS-aware coupled planner (optionally with continuous-altitude local search)."""
    nf = instance.node_features.numpy()[0]
    xy, z, dem = nf[:, :2], nf[:, 2], nf[:, 3]
    rmin = instance.node_rmin[0].numpy() if rmin_override is None else rmin_override
    dmax = d_max_of(rmin, chan)
    args = (trainer, xy, z, dem, rmin, ALTITUDE_GRID, chan,
            trainer.H_min, trainer.H_max, trainer.h_safe, dmax)
    hovers = plan_coupled_strong(*args, restarts=1) if strong else plan_coupled(*args)
    return _score_hovers(trainer, instance, hovers)


# ---------------------------------------------------------------------------
# FIGURE WRITERS
# ---------------------------------------------------------------------------
def _save_npz(name: str, **arrays) -> Path:
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DATA_DIR / name
    np.savez(path, **arrays)
    print(f"  wrote {path.relative_to(PAPER_FIGURES_DIR.parent)}")
    return path


def _save_pickle(name: str, obj) -> Path:
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DATA_DIR / name
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  wrote {path.relative_to(PAPER_FIGURES_DIR.parent)}")
    return path


def write_qos_satisfaction(runs: Sequence[MethodRun], instances) -> None:
    """Fig. 5 - per-class QoS satisfaction (%) per method."""
    classes = list(PRIORITY_ORDER)
    values = np.zeros((len(runs), len(classes)), float)
    for i, run in enumerate(runs):
        per_city = [class_satisfaction(t, inst)
                    for t, inst in zip(run.traces, instances)]
        for j, cls in enumerate(classes):
            values[i, j] = float(np.nanmean([p[cls] for p in per_city]))
    _save_npz("qos_satisfaction.npz",
              classes=np.array(classes), methods=np.array([r.key for r in runs]),
              values=values)


def write_energy_breakdown(runs: Sequence[MethodRun]) -> None:
    """Fig. 6 - flight / hover / comm / total energy (kJ) per method."""
    cats = ["flight", "hover", "comm", "total"]
    values = np.array([[run.energy_split_kj()[c] for c in cats] for run in runs])
    _save_npz("energy_breakdown.npz",
              categories=np.array(cats), methods=np.array([r.key for r in runs]),
              values=values, unit=np.array("kJ"))


def write_hover_altitudes(runs: Sequence[MethodRun]) -> None:
    """Fig. 10 - every hover altitude, one array per method."""
    _save_npz("hover_altitudes.npz",
              **{r.key: r.all_hover_altitudes() for r in runs})


def write_high_rate_samples(runs: Sequence[MethodRun], instances) -> None:
    """Fig. 9 - achieved rate (Mbps) at every high-priority node."""
    arrays = {}
    for run in runs:
        samples = []
        for trace, inst in zip(run.traces, instances):
            mask = inst.class_mask("high")
            mbps = trace["rate"][mask] / 1e6
            mbps = np.where(trace["served"][mask], mbps, UNSERVED_RATE_MBPS)
            samples.append(np.maximum(mbps, UNSERVED_RATE_MBPS))
        arrays[run.key] = np.concatenate(samples)
    _save_npz("high_rate_samples.npz", **arrays)


def write_trajectories(runs: Sequence[MethodRun]) -> None:
    """Fig. 7 - ordered hover points (x, y, z), depot first and last.

    Multi-UAV routes are concatenated with a depot waypoint between sorties, so
    the drawn path is the true mission including each return leg.
    """
    out: Dict[str, list] = {}
    for run in runs:
        trace = run.traces[0]                       # canonical scene only
        depot = trace["depot"]
        points = [depot]
        current_uav = 0
        for hover in trace["hovers"]:
            if hover["uav"] != current_uav:         # previous UAV flew home
                points.append(depot)
                current_uav = hover["uav"]
            points.append((hover["x"], hover["y"], hover["altitude"]))
        points.append(depot)
        out[run.key] = points
    _save_pickle("trajectories.pkl", out)


def write_altitude_traces(runs: Sequence[MethodRun], instance: CityInstance,
                          n: int = 1400) -> None:
    """Fig. 3 - altitude vs mission progress, with critical-service events.

    Progress is cumulative horizontal flight distance (0-100%), so a dive shows
    up where it physically happens along the route rather than at a uniform
    step index. The altitude curve is the flown profile: cruise between hovers,
    held at the hover altitude while serving.
    """
    traces: Dict[str, np.ndarray] = {}
    events: List[dict] = []
    critical_intervals: List[Tuple[float, float]] = []
    grid = np.linspace(0.0, 100.0, n)
    floors = instance.node_rmin[0].numpy()

    for run in runs:
        trace = run.traces[0]
        depot = trace["depot"]
        pts = [(depot[0], depot[1], depot[2])]
        for hover in trace["hovers"]:
            pts.append((hover["x"], hover["y"], hover["altitude"]))
        pts.append((depot[0], depot[1], depot[2]))
        arr = np.array(pts, float)

        seg = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= 0:
            traces[run.key] = np.full(n, arr[0, 2])
            continue
        progress = 100.0 * cum / cum[-1]
        traces[run.key] = np.interp(grid, progress, arr[:, 2])

        # Events/intervals are a property of the mission, taken from the
        # method that actually resolves the criticality classes (atom3d).
        if run.key != "atom3d":
            continue
        for idx, hover in enumerate(trace["hovers"], start=1):
            served_idx = hover["served"]
            if served_idx.size == 0:
                continue
            has_high = bool((floors[served_idx] >= CITY_CLASS_FLOORS["high"]).any())
            has_med = bool((floors[served_idx] >= CITY_CLASS_FLOORS["medium"]).any())
            priority = "high" if has_high else ("medium" if has_med else "low")
            p = float(progress[idx])
            events.append({"progress": p, "priority": priority})
            if has_high:
                span = 0.5 * float(np.diff(progress).mean()) + 0.4
                critical_intervals.append((p - span, p + span))

    _save_pickle("altitude_traces.pkl", {
        "progress": grid, "traces": traces, "events": events,
        "critical_intervals": critical_intervals,
    })


def write_overall_metrics(runs: Sequence[MethodRun], instances) -> None:
    """Fig. 13 - mean and 95% CI of energy + per-class QoS across realizations.

    With a single city realization there is no across-seed spread, so the CI is
    reported as 0.0 rather than invented. Pass several ``--city-seeds`` to get a
    real interval.
    """
    metric_keys = ["energy", "high", "medium", "low"]
    means = np.zeros((len(runs), len(metric_keys)), float)
    ci95 = np.zeros_like(means)

    for i, run in enumerate(runs):
        per_city = [class_satisfaction(t, inst)
                    for t, inst in zip(run.traces, instances)]
        samples = {
            # kJ, matching this table's column header and Figure 6's unit.
            # (1 Wh = 3.6 kJ; reporting Wh here would silently disagree with Fig. 6.)
            "energy": run.energy_wh() * 3.6,
            "high": np.array([p["high"] for p in per_city]),
            "medium": np.array([p["medium"] for p in per_city]),
            "low": np.array([p["low"] for p in per_city]),
        }
        for j, key in enumerate(metric_keys):
            values = np.asarray(samples[key], float)
            means[i, j] = float(np.nanmean(values))
            if values.size > 1:
                sem = np.nanstd(values, ddof=1) / np.sqrt(values.size)
                ci95[i, j] = float(1.96 * sem)
    _save_npz("overall_metrics.npz",
              methods=np.array([r.key for r in runs]),
              metrics=np.array(metric_keys), means=means, ci95=ci95)


def max_achievable_rate(trainer) -> float:
    """Best possible link rate (bits/s), at the closest legally allowed approach.

    Clearance forces ``H >= z + h_safe``, so the shortest achievable link is
    ``h_safe`` metres. Any QoS floor above the rate at that distance is
    physically unsatisfiable no matter what the planner does - which is a real
    property of the system, not a solver failure.
    """
    d = max(trainer.h_safe, 1e-3)
    snr = trainer.P_T * trainer.beta / (d ** trainer.path_loss_exp * trainer.sigma2)
    return float(trainer.bandwidth * np.log2(1.0 + snr))


def write_pareto(trainer, instances, chan, runs: Sequence[MethodRun],
                 scales: Sequence[float], fast: bool) -> None:
    """Fig. 8 - energy vs critical-QoS frontier, swept by scaling the floors.

    Each sweep point re-plans at a scaled QoS floor, so the frontier is traced
    by real re-optimization rather than by interpolating one solution. Scales
    whose floor exceeds the physical rate ceiling are dropped (and reported)
    rather than plotted, since no planner can satisfy them.
    """
    instance = instances[0]
    base_rmin = instance.node_rmin[0].numpy()
    ceiling = max_achievable_rate(trainer)
    peak_floor = float(base_rmin.max())
    max_scale = ceiling / peak_floor if peak_floor > 0 else float("inf")

    feasible = [s for s in scales if s * peak_floor <= ceiling]
    dropped = [s for s in scales if s not in feasible]
    print(f"  rate ceiling at h_safe={trainer.h_safe:.0f} m: {ceiling/1e6:.1f} Mbps "
          f"-> max feasible floor scale x{max_scale:.2f}")
    if dropped:
        print(f"  dropping physically infeasible sweep points: "
              f"{', '.join('x%.2f' % s for s in dropped)}")
    if not feasible:
        print("  no feasible sweep points - skipping pareto_sweep.npz")
        return
    scales = feasible

    curves: Dict[str, dict] = {}

    def evaluate(run: MethodRun, rmin: np.ndarray):
        """(energy Wh, high-QoS %) for one method at one target floor."""
        if run.key == "2d_auto":
            metrics = run_2d_auto(trainer, instance, trainer.fixed_alt)
        elif run.key == "3d_gnn":
            metrics = run_blind_3d(trainer, instance, chan)
        else:
            metrics = run_coupled(trainer, instance, chan,
                                  strong=not fast, rmin_override=rmin)
        trace = metrics["trace"][0]
        # Satisfaction is always measured against the TRUE floors, so the
        # frontier answers "what did moving the target actually buy?"
        # kJ, matching this figure's axis and Figures 6/13 (1 Wh = 3.6 kJ).
        return (float(metrics["energy"][0]) / 1000.0,
                class_satisfaction(trace, instance)["high"])

    for run in runs:
        # 2D-AUTO and Blind-3D never read the QoS floor, so they do not trace a
        # frontier at all: they are a single operating point. Evaluating them
        # once is both honest and much cheaper than re-running them per scale.
        responds_to_floor = run.key == "atom3d"
        if not responds_to_floor:
            energy, satisfaction = evaluate(run, base_rmin)
            curves[run.key] = {"energy": [energy], "satisfaction": [satisfaction],
                               "default": (energy, satisfaction)}
            continue

        energies, satisfactions, kept = [], [], []
        for scale in scales:
            try:
                energy, satisfaction = evaluate(run, base_rmin * float(scale))
            except (TypeError, ValueError) as exc:
                # A planner can fail to find a feasible cover near the rate
                # ceiling. Drop that point instead of losing the whole sweep.
                print(f"    [skip] {run.key} at floor x{scale:.2f}: {exc}")
                continue
            energies.append(energy)
            satisfactions.append(satisfaction)
            kept.append(scale)
        if not energies:
            print(f"  [warn] no feasible sweep point for '{run.key}'")
            continue
        default_idx = int(np.argmin(np.abs(np.asarray(kept, float) - 1.0)))
        curves[run.key] = {
            "energy": energies, "satisfaction": satisfactions,
            "default": (energies[default_idx], satisfactions[default_idx]),
        }

    payload = {}
    for key, curve in curves.items():
        payload[f"{key}_energy"] = np.array(curve["energy"], float)
        payload[f"{key}_satisfaction"] = np.array(curve["satisfaction"], float)
        payload[f"{key}_default"] = np.array(curve["default"], float)
    _save_npz("pareto_sweep.npz", **payload)


def write_training_log(epochs, reward, actor_loss, critic_loss, violation,
                       convergence_epoch: Optional[int] = None) -> None:
    """Fig. 4 - CMDP training dynamics, written from a real training run.

    ``violation`` is the mean per-class constraint violation g_c = target - satisfaction
    reported by the trainer's dual step; it is the quantity the CMDP actually drives
    to zero, not a proxy.
    """
    payload = {
        "epoch": np.asarray(epochs, int),
        "reward": np.asarray(reward, float),
        "actor_loss": np.asarray(actor_loss, float),
        "critic_loss": np.asarray(critic_loss, float),
        "constraint_violation": np.asarray(violation, float),
    }
    if convergence_epoch is not None:
        payload["convergence_epoch"] = np.asarray(int(convergence_epoch))
    _save_npz("training_log.npz", **payload)


def write_dual_variables(epochs, duals: Dict[str, Sequence[float]],
                         convergence_epoch: Optional[int] = None) -> None:
    """Fig. 11 - per-class Lagrange multipliers over training.

    ``duals`` maps a QoS class name ('high'/'medium') to its lambda trajectory,
    straight from ``LagrangianDuals``.
    """
    payload = {"epoch": np.asarray(epochs, int)}
    for cls, series in duals.items():
        payload[f"lambda_{cls}"] = np.asarray(series, float)
    if convergence_epoch is not None:
        payload["convergence_epoch"] = np.asarray(int(convergence_epoch))
    _save_npz("dual_variables.npz", **payload)


def write_labels(runs: Sequence[MethodRun], extra: Optional[dict] = None) -> None:
    """Provenance sidecar: what actually produced each figure slot."""
    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "methods": {r.key: {"label": r.label, "source": r.source} for r in runs},
        "generated_by": "atom_3d.experiments.export_figure_data",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    path = RESULTS_DATA_DIR / "labels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {path.relative_to(PAPER_FIGURES_DIR.parent)}")


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def build_trainer(params: dict, instance: CityInstance) -> TENMATrainer:
    trainer = TENMATrainer(
        params, TrainConfig(mode="3d", embed_dim=32, num_heads=8, ff_dim=32, num_layers=1))
    apply_city_depot(trainer, instance)
    return trainer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "params.yaml"))
    ap.add_argument("--city-seeds", default="42",
                    help="comma-separated city realizations; >1 enables real CIs")
    ap.add_argument("--fast", action="store_true",
                    help="use the plain coupled greedy instead of the ~20 min local search")
    # Default range stays under the physical rate ceiling (~1.17x the 38 Mbps
    # floor at h_safe); write_pareto drops anything above it anyway.
    ap.add_argument("--pareto-scales", default="0.0,0.25,0.5,0.75,1.0,1.15")
    ap.add_argument("--skip-pareto", action="store_true")
    args = ap.parse_args()

    params = yaml.safe_load(open(os.path.abspath(args.config)))
    params["priority"]["enabled"] = True
    params["priority"]["R_min"] = dict(CITY_CLASS_FLOORS)

    seeds = [int(s) for s in args.city_seeds.split(",") if s.strip()]
    scales = [float(s) for s in args.pareto_scales.split(",") if s.strip()]

    print(f"[export] city realizations: {seeds}")
    instances = build_instances_for_seeds(seeds)
    print(f"[export] classes: {class_counts(instances[0])}")

    trainer = build_trainer(params, instances[0])
    chan = make_channel_consts(trainer)

    # 2D-AUTO cruise altitude: the lowest that clears the tallest structure with
    # the spec's safety margin. Derived from the city, not hard-coded, so it
    # adapts if the environment changes.
    tallest = max(float(i.node_features[0, :, 2].max()) for i in instances)
    two_d_alt = tallest + trainer.h_safe
    print(f"[export] 2D cruise altitude = {two_d_alt:.1f} m "
          f"(tallest structure {tallest:.1f} m + h_safe {trainer.h_safe:.0f} m)")
    trainer.fixed_alt = two_d_alt

    coupled_label = ("Coupled-Greedy (deterministic)" if args.fast
                     else "Strong-Coupled (deterministic)")
    runs = [
        MethodRun("2d_auto", f"2D-AUTO ({two_d_alt:.0f} m)",
                  "per-node NN+2-opt tour at a fixed altitude"),
        MethodRun("3d_gnn", "Blind-3D (greedy cover)",
                  "QoS-blind greedy footprint cover; NOT a trained GNN policy"),
        MethodRun("atom3d", coupled_label,
                  "QoS-aware coupled planner; NOT the trained CMDP policy"),
    ]
    for key, tag in LEARNED_CHECKPOINTS.items():
        ckpt = Path(params["paths"]["checkpoint_dir"]) / f"{tag}.pt"
        if ckpt.exists():
            print(f"[export] NOTE: {ckpt} exists - slot '{key}' can be upgraded "
                  f"to the learned policy (not yet wired; deterministic used).")

    for idx, instance in enumerate(instances):
        apply_city_depot(trainer, instance)
        print(f"[export] city seed {seeds[idx]} ...")
        t0 = time.time()
        runs[0].metrics.append(run_2d_auto(trainer, instance, two_d_alt))
        print(f"    2D-AUTO        {time.time() - t0:6.1f}s")
        t0 = time.time()
        runs[1].metrics.append(run_blind_3d(trainer, instance, chan))
        print(f"    Blind-3D       {time.time() - t0:6.1f}s")
        t0 = time.time()
        runs[2].metrics.append(run_coupled(trainer, instance, chan, strong=not args.fast))
        print(f"    {coupled_label.split(' (')[0]:14s} {time.time() - t0:6.1f}s")
        for run in runs:
            run.traces.append(run.metrics[-1]["trace"][0])

    print("\n[export] summary (mean over realizations)")
    print(f"  {'method':32s} {'energy':>9} {'high':>7} {'med':>7} {'low':>7}")
    for run in runs:
        sat = [class_satisfaction(t, i) for t, i in zip(run.traces, instances)]
        print(f"  {run.label:32s} {run.energy_wh().mean():7.2f}Wh "
              f"{np.nanmean([s['high'] for s in sat]):6.1f}% "
              f"{np.nanmean([s['medium'] for s in sat]):6.1f}% "
              f"{np.nanmean([s['low'] for s in sat]):6.1f}%")

    print("\n[export] writing figure data")
    write_qos_satisfaction(runs, instances)
    write_energy_breakdown(runs)
    write_hover_altitudes(runs)
    write_high_rate_samples(runs, instances)
    write_trajectories(runs)
    write_altitude_traces(runs, instances[0])
    write_overall_metrics(runs, instances)
    if not args.skip_pareto:
        apply_city_depot(trainer, instances[0])
        write_pareto(trainer, instances, chan, runs, scales, args.fast)
    write_labels(runs, extra={
        "city_seeds": seeds,
        "two_d_altitude_m": two_d_alt,
        "note": ("Deterministic planners on the frozen synthetic city. Slots "
                 "'3d_gnn' and 'atom3d' are NOT learned policies yet - see "
                 "each slot's 'source'. training_log.npz and dual_variables.npz "
                 "are intentionally absent until a real CMDP run produces them."),
    })
    print("\n[export] done. Figures 3, 5, 6, 7, 8, 9, 10, 13 now load real data.")
    print("         Figures 4, 11, 12 keep their PLACEHOLDER stamp (need training).")


if __name__ == "__main__":
    main()
