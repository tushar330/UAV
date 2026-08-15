"""Regression tests for the altitude head.

These exist because a 500-epoch CMDP run was invalidated by a defect neither the
loss curves nor a smoke test could reveal: the altitude head was fed only the
pre-selection glimpse, so it could not see WHICH anchor had been chosen. Since
the anchor is always served at its own hover with the UAV directly overhead, its
achieved rate depends only on ``H - z_anchor`` -- a head blind to ``z_anchor``
cannot satisfy a per-node rate floor. It collapsed to a constant
H = 85.5 +/- 0.14 m and scored 0% high-priority QoS, while training loss and
gradients all looked healthy.

Run:  python tests/test_altitude_head.py        (no pytest needed)
  or: python -m pytest tests/test_altitude_head.py -q
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atom_3d.models.trajectory_decoder import TrajectoryDecoder

EMBED, HEADS, B, N = 32, 4, 4, 12
H_MIN, H_MAX, H_SAFE = 20.0, 150.0, 10.0


def make_decoder():
    torch.manual_seed(0)
    return TrajectoryDecoder(embed_dim=EMBED, num_heads=HEADS, dim_3d=True,
                             H_min=H_MIN, H_max=H_MAX, h_safe=H_SAFE)


def make_inputs(seed=1, z=None):
    g = torch.Generator().manual_seed(seed)
    h_nodes = torch.randn(B, N, EMBED, generator=g)
    h_graph = torch.randn(B, EMBED, generator=g)
    node_xy = torch.rand(B, N, 2, generator=g) * 1000.0
    node_z = (torch.rand(B, N, generator=g) * 50.0) if z is None else z
    node_demand = torch.rand(B, N, generator=g) + 0.2
    return h_nodes, h_graph, node_xy, node_z, node_demand


def test_altitude_head_sees_the_chosen_anchor():
    """alt_mean must consume the anchor embedding as well as the context.

    This is the structural property whose absence caused the collapse. Checking
    the input width pins it directly, independently of any training outcome.
    """
    dec = make_decoder()
    assert dec.alt_mean[0].in_features == 2 * EMBED, (
        "altitude head takes only the pre-anchor context; it cannot condition "
        "on the chosen anchor and so cannot express H as a function of z_anchor")


def test_altitude_tracks_anchor_elevation():
    """Altitude must respond to the served node's elevation, not be constant.

    Run the same decoder on two cities differing ONLY in elevation. If chosen
    altitudes are identical, altitude is not a function of z at all -- the
    failure mode that produced a constant 85.5 m.
    """
    dec = make_decoder()
    flat = dec(*make_inputs(z=torch.full((B, N), 5.0)), greedy=True).altitudes
    tall = dec(*make_inputs(z=torch.full((B, N), 45.0)), greedy=True).altitudes
    # Rollout length varies with the plan, so compare the first hover, which
    # both runs always have, rather than whole tensors of different width.
    assert not torch.allclose(flat[:, 0], tall[:, 0], atol=1e-3), (
        "altitude identical over 5 m and 45 m nodes: the head ignores elevation")
    # And it must move the right way: a 40 m taller node forces a higher hover,
    # since the band starts at z_anchor + h_safe.
    assert (tall[:, 0] > flat[:, 0]).all(), (
        "altitude did not rise for taller nodes")


def test_altitude_clears_the_served_node():
    """Every hover must clear its own anchor by h_safe and respect H_max.

    The parametrization is supposed to make the closest legal approach reachable
    by construction rather than something the policy must find by luck.
    """
    dec = make_decoder()
    h_nodes, h_graph, node_xy, node_z, node_demand = make_inputs(z=None)
    plan = dec(h_nodes, h_graph, node_xy, node_z, node_demand, greedy=True)
    z_anchor = node_z.gather(1, plan.anchors)
    active = plan.step_active
    clearance = (plan.altitudes - z_anchor)[active]
    assert (clearance >= H_SAFE - 1e-3).all(), (
        f"hover below the safety margin: min clearance {clearance.min():.2f} m")
    assert (plan.altitudes[active] <= H_MAX + 1e-3).all()


def test_altitude_head_receives_gradient():
    """The score-function path must actually reach alt_mean.

    An earlier bug reparametrized the sample so log_prob was constant in the
    mean, zeroing this gradient. Sampling (greedy=False) is required: the greedy
    path takes the mean itself.
    """
    dec = make_decoder()
    plan = dec(*make_inputs(), greedy=False)
    plan.log_p_alt.sum().backward()
    grad = dec.alt_mean[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0, (
        "no gradient reached alt_mean; the altitude action is not being learned")


def test_frozen_altitude_ablation_still_constant():
    """The freeze-altitude ablation must stay exactly constant."""
    torch.manual_seed(0)
    dec = TrajectoryDecoder(embed_dim=EMBED, num_heads=HEADS, dim_3d=True,
                            H_min=H_MIN, H_max=H_MAX, h_safe=H_SAFE,
                            freeze_altitude=70.0)
    plan = dec(*make_inputs(), greedy=True)
    assert torch.allclose(plan.altitudes, torch.full_like(plan.altitudes, 70.0))


def _main():
    """Minimal runner so the suite works in an environment without pytest."""
    failures = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                      # noqa: BLE001 - report all
            failures.append(name)
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
