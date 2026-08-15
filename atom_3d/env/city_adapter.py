"""Adapter: the paper's synthetic city -> ATOM-3D evaluation tensors.

``paper_figures/synthetic_city.py`` owns the frozen environment the paper's
figures describe (500 nodes on buildings across 7 districts, seeded at 42).
The ATOM-3D pipeline, by contrast, generates uniform-random scenarios via
``IoTEnvironment3D.generate_batch``. Those are two different environments, so
results produced on one cannot be plotted as results on the other.

This module bridges them **in one direction only**: it loads the city and
converts it into exactly the tensors the trainer/scorer already consume, so
every planner in this repo can be evaluated on the paper's environment and its
metrics exported for the figures.

Nothing here modifies the city. ``synthetic_city.py`` is locked infrastructure
(``paper_figures/CODE_RULES.md``); we import and read it, never write it.

Two quirks of the locked file that are handled here:

* ``CITY_FILE`` is a *relative* ``Path("synthetic_city.pkl")``, so the module's
  own ``get_city()`` only resolves when the process cwd is ``paper_figures/``.
  We pass an absolute path to ``load_city()`` instead of relying on cwd.
* The pickle stores ``City``/``IoTNode`` dataclass instances, so unpickling
  requires the defining module to be importable under the exact name
  ``synthetic_city``. We register it in ``sys.modules`` under that name.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# --- repo layout -----------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES_DIR = _REPO_ROOT / "paper_figures"
_SYNTHETIC_CITY_PY = PAPER_FIGURES_DIR / "synthetic_city.py"
_CITY_PICKLE = PAPER_FIGURES_DIR / "synthetic_city.pkl"

# Criticality-class importance w_i used by the paper's VoI definition
# (VoI = sum over served nodes of w * D * 1[QoS met], see DATA_SPEC.md).
#
# NOTE — a real drift between the two halves of the repo: DATA_SPEC.md defines
# w = 3/2/1 while configs/params.yaml defines priority.weights = 5/2/1. The
# figures are the consumer here, so the figure-side convention wins and is
# stated explicitly rather than silently inherited from the training config.
CLASS_WEIGHTS: Dict[str, float] = {"high": 3.0, "medium": 2.0, "low": 1.0}

# Per-class rate floors default to each IoTNode's own ``required_rate`` from the
# pickle. They can be overridden per class (see ``class_floors``) because the
# pickle's 38/25/8 Mbps do not all bind on this geometry: at alpha=3 those map
# to reach 21.1 / 94.9 / 690.9 m, and on a 1000 m map a 690 m reach can never
# fail. A floor that cannot fail measures coverage, not quality of service, so
# with the shipped values only the critical class actually exercises a QoS
# constraint. Overriding here keeps `synthetic_city.py` and the pickle locked.

PRIORITY_ORDER = ("high", "medium", "low")


@dataclass
class CityInstance:
    """One evaluation instance built from the paper's synthetic city.

    Shapes match what ``TENMATrainer`` expects for a batch of size 1, so this
    can be passed straight into ``evaluate`` / ``_partition_and_evaluate`` or
    into any planner in ``atom_3d.experiments.heuristic_baseline``.
    """

    node_features: torch.Tensor      # (1, N, 4) float32 — [x, y, z, demand_MB]
    node_rmin: torch.Tensor          # (1, N) float32 — per-node QoS floor (bits/s)
    node_weights: torch.Tensor       # (1, N) float32 — per-node importance w_i
    priority: np.ndarray             # (N,) str — "high" / "medium" / "low"
    depot_xy: np.ndarray             # (2,) float32 — depot horizontal position
    depot_z: float                   # depot altitude (m)
    city: object                     # the raw City dataclass (buildings, roads, ...)

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[1])

    def class_mask(self, priority: str) -> np.ndarray:
        """(N,) bool mask selecting one criticality class."""
        return self.priority == priority


# ---------------------------------------------------------------------------
# loading the locked module + pickle
# ---------------------------------------------------------------------------
def _import_synthetic_city():
    """Import ``paper_figures/synthetic_city.py`` under its own module name.

    Registering it as ``synthetic_city`` (not a dotted path) is required: the
    pickle references that module name for its dataclasses, so any other name
    makes ``pickle.load`` fail with ``ModuleNotFoundError``.
    """
    if "synthetic_city" in sys.modules:
        return sys.modules["synthetic_city"]
    if not _SYNTHETIC_CITY_PY.exists():
        raise FileNotFoundError(
            f"cannot find the locked city generator at {_SYNTHETIC_CITY_PY}"
        )
    spec = importlib.util.spec_from_file_location("synthetic_city", _SYNTHETIC_CITY_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["synthetic_city"] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _cwd(path: Path):
    """Temporarily chdir — needed only when the city must be *built*."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def load_city_object():
    """Return the ``City`` dataclass, building the pickle only if absent.

    Prefers an absolute-path load so the caller's cwd is irrelevant. Falls back
    to the locked module's own ``get_city()`` (which needs cwd=paper_figures)
    exactly once, to generate the pickle the first time.
    """
    sc = _import_synthetic_city()
    if _CITY_PICKLE.exists():
        return sc.load_city(_CITY_PICKLE)
    with _cwd(PAPER_FIGURES_DIR):
        return sc.get_city()


# ---------------------------------------------------------------------------
# city -> tensors
# ---------------------------------------------------------------------------
def build_instance(
    city=None,
    *,
    class_weights: Optional[Dict[str, float]] = None,
    class_floors: Optional[Dict[str, float]] = None,
    recenter_on_depot: bool = False,
) -> CityInstance:
    """Convert the city into ATOM-3D evaluation tensors.

    Args:
        city: a preloaded ``City``; loaded via :func:`load_city_object` if None.
        class_weights: override for the per-class importance w_i.
        class_floors: {class: bits/s} replacing the pickle's per-node
            ``required_rate``. Use when the shipped floors do not all bind on
            this geometry (see the note above); None keeps the pickle's values.
        recenter_on_depot: shift x/y so the depot sits at the origin. Off by
            default — the scorer takes the depot position explicitly (see
            :func:`apply_city_depot`), so keeping the city's own [0, 1000]
            coordinates means exported trajectories overlay the figure maps
            without any inverse transform.

    Returns:
        A :class:`CityInstance` with batch dimension 1.
    """
    if city is None:
        city = load_city_object()
    weights = dict(CLASS_WEIGHTS if class_weights is None else class_weights)

    nodes = list(city.nodes)
    n = len(nodes)

    xy = np.array([[nd.x, nd.y] for nd in nodes], dtype=np.float32)      # (N, 2)
    z = np.array([nd.z for nd in nodes], dtype=np.float32)               # (N,)
    demand = np.array([nd.demand for nd in nodes], dtype=np.float32)     # (N,)
    priority = np.array([nd.priority for nd in nodes], dtype=object)
    if class_floors is None:
        rmin = np.array([nd.required_rate for nd in nodes], dtype=np.float32)
    else:
        missing = sorted({p for p in priority.tolist()} - set(class_floors))
        if missing:
            raise ValueError(f"class_floors is missing classes: {missing}")
        rmin = np.array([class_floors[p] for p in priority.tolist()],
                        dtype=np.float32)

    unknown = sorted({p for p in priority.tolist()} - set(weights))
    if unknown:
        raise ValueError(f"city contains priority classes with no weight: {unknown}")
    w = np.array([weights[p] for p in priority.tolist()], dtype=np.float32)

    depot_xy = np.array([city.depot.x, city.depot.y], dtype=np.float32)
    depot_z = float(city.depot.z)

    if recenter_on_depot:
        xy = xy - depot_xy
        depot_xy = np.zeros(2, dtype=np.float32)

    # (1, N, 4) — [x, y, z, demand], the 3D feature layout the encoder/scorer use
    features = np.concatenate(
        [xy, z[:, None], demand[:, None]], axis=1
    ).astype(np.float32)[None, ...]

    return CityInstance(
        node_features=torch.from_numpy(features),
        node_rmin=torch.from_numpy(rmin[None, :]),
        node_weights=torch.from_numpy(w[None, :]),
        priority=priority,
        depot_xy=depot_xy,
        depot_z=depot_z,
        city=city,
    )


def build_instances_for_seeds(
    seeds: List[int], **kwargs
) -> List[CityInstance]:
    """Independent city realizations, one per RNG seed.

    The canonical city is frozen at ``synthetic_city.SEED = 42`` and cached in
    the pickle; a single deterministic instance gives deterministic planners
    zero spread, so an aggregate table could not report a confidence interval.
    Re-drawing the city at other seeds yields *statistically identical* cities
    (same district layout rules, same 50/100/350 class split, same demand and
    rate-floor distributions) with different node placements — which is a
    legitimate source of across-realization variance.

    ``synthetic_city`` is locked, so we do not edit it: we rebind its
    module-level ``RNG`` for the duration of each generation and restore it
    afterwards. Seed 42 is served from the cached pickle so the canonical
    scene used by the qualitative figures is bit-identical to Figure 1.
    """
    sc = _import_synthetic_city()
    original_rng = sc.RNG
    out: List[CityInstance] = []
    try:
        for seed in seeds:
            if int(seed) == int(sc.SEED):
                city = load_city_object()          # canonical cached scene
            else:
                sc.RNG = np.random.default_rng(int(seed))
                city = sc.generate_city()
                sc.verify_city(city)               # asserts the 50/100/350 split
            out.append(build_instance(city, **kwargs))
    finally:
        sc.RNG = original_rng
    return out


def apply_city_depot(trainer, instance: CityInstance) -> None:
    """Point a ``TENMATrainer``'s depot at the city's depot.

    The trainer reads ``scenario.data_center`` from params.yaml at construction
    time (origin, z=20). Evaluating on the city requires its real depot at
    (50, 50, 0), so we set it explicitly instead of editing the shared config —
    which would silently change every training run too.
    """
    trainer.depot = np.asarray(instance.depot_xy, dtype=np.float32).copy()
    trainer.depot_alt = float(instance.depot_z)


def class_counts(instance: CityInstance) -> Dict[str, int]:
    """Node count per criticality class (sanity check against DATA_SPEC)."""
    return {p: int((instance.priority == p).sum()) for p in PRIORITY_ORDER}


def tallest_structure(instance: CityInstance) -> float:
    """Highest node elevation in the city (m).

    Used to pick a 2D baseline cruise altitude that can physically see every
    node: nodes sit on rooftops, so a fixed altitude below the tallest building
    would give some nodes a zero-radius coverage cone.
    """
    return float(instance.node_features[0, :, 2].max().item())
