"""
Evaluation / experiment driver for ATOM-3D.

Produces the three headline results from the approach doc (§6):

* ``--compare``  : 3D vs 2D (and optionally the GNN ablation) on the same node
  layouts — the energy headline + the "attention helps" ablation.
* ``--pareto``   : sweep the per-node QoS floor ``R_min`` for the 3D model and
  trace the energy-vs-quality frontier (§7).
* ``--plot``     : draw one instance's 3D routes + altitude profile.

Each requested checkpoint is loaded from ``<checkpoint_dir>/<mode>_<encoder>.pt``.
Models are evaluated greedily on a fixed set of instances for a fair comparison.
"""

import argparse
import os

import numpy as np
import torch
import yaml

from ..training import TENMATrainer, TrainConfig
from ..env.iot_env import IoTEnvironment2D, IoTEnvironment3D
from ..utils.visualization import TrajectoryVisualizer


def load_params(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_batch(mode, batch_size, N, params, seed):
    sc = params["scenario"]; nd = params["nodes"]
    kw = dict(seed=seed, Di_min=nd["Di_min"], Di_max=nd["Di_max"],
              area_width=sc["area_width"], area_height=sc["area_height"])
    if mode == "3d":
        kw.update(zi_min=nd["zi_min"], zi_max=nd["zi_max"])
        return IoTEnvironment3D.generate_batch(batch_size, N, **kw)[0]
    return IoTEnvironment2D.generate_batch(batch_size, N, **kw)[0]


def make_priorities(batch_size, N, params, seed):
    """Per-node (weights, R_min) for priority evaluation (3D)."""
    pc = params["priority"]
    _, w, r = IoTEnvironment3D.generate_priorities(
        batch_size, N, seed=seed, priority_enabled=True,
        priority_class_probs=pc["class_probs"],
        priority_weights=[pc["weights"]["high"], pc["weights"]["medium"], pc["weights"]["low"]],
        priority_rmin=[pc["R_min"]["high"], pc["R_min"]["medium"], pc["R_min"]["low"]],
    )
    return w, r


def build_trainer(params, mode, encoder, embed_dim, num_layers, device, priority_feature=False):
    cfg = TrainConfig(
        mode=mode, encoder=encoder, embed_dim=embed_dim,
        num_heads=params["encoder"]["H_heads"], ff_dim=embed_dim,
        num_layers=num_layers, dropout=params["encoder"]["dropout"],
        priority_feature=priority_feature, device=device,
    )
    return TENMATrainer(params, cfg)


def load_if_available(trainer, params, mode, encoder, device):
    return load_tag(trainer, params, f"{mode}_{encoder}", device)


def load_tag(trainer, params, tag, device):
    """Load ``<checkpoint_dir>/<tag>.pt`` if present; warn + keep random weights otherwise."""
    path = os.path.join(params["paths"]["checkpoint_dir"], f"{tag}.pt")
    if os.path.exists(path):
        trainer.load_state_dict(torch.load(path, map_location=device))
        print(f"[load] {path}")
        return True
    print(f"[warn] no checkpoint at {path} - using randomly initialised weights")
    return False


def main():
    ap = argparse.ArgumentParser(description="Evaluate / compare ATOM-3D models")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "params.yaml"))
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ckpt-dir", default=None, help="override checkpoint dir to load from")
    ap.add_argument("--compare", action="store_true", help="3D vs 2D vs GNN energy table")
    ap.add_argument("--pareto", action="store_true", help="R_min sweep -> energy/QoS frontier")
    ap.add_argument("--plot", action="store_true", help="plot one 3D instance's routes")
    ap.add_argument("--priority", action="store_true",
                    help="priority-aware eval: priority-3D vs blind-3D vs 2D + critical-floor frontier")
    ap.add_argument("--cmdp", action="store_true",
                    help="load the CMDP-trained priority model (tag *_priority_cmdp) instead of the "
                         "fixed-lambda *_priority model; only affects which priority checkpoint is loaded")
    args = ap.parse_args()

    params = load_params(os.path.abspath(args.config))
    if args.ckpt_dir is not None:
        params["paths"]["checkpoint_dir"] = args.ckpt_dir
    N = args.N if args.N is not None else params["scenario"]["N"]
    n_inst = args.instances if args.instances is not None else params["training"]["num_eval_instances"]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    viz = TrajectoryVisualizer(params["paths"]["figure_dir"])

    # default action: do everything except the opt-in priority eval
    if not (args.compare or args.pareto or args.plot or args.priority):
        args.compare = args.pareto = args.plot = True

    # ---------------- comparison table ----------------
    if args.compare:
        print("\n=== Energy comparison (greedy, same layouts) ===")
        rows = [("3d", "attention"), ("2d", "attention"), ("3d", "gnn")]
        results = {}
        for mode, enc in rows:
            trainer = build_trainer(params, mode, enc, args.embed_dim, args.num_layers, args.device)
            load_if_available(trainer, params, mode, enc, args.device)
            batch = make_batch(mode, n_inst, N, params, seed=args.seed)
            ev = trainer.evaluate(batch, R_min=params["qos"]["R_min"], greedy=True)
            results[(mode, enc)] = ev
            print(f"  {mode.upper():3s}/{enc:9s} | energy {ev['energy_wh']:8.2f} Wh | "
                  f"uavs {ev['num_uavs']:5.1f} | served {(1-ev['unserved_frac'])*100:5.1f}%")
        a3 = results[("3d", "attention")]["energy_wh"]
        a2 = results[("2d", "attention")]["energy_wh"]
        if a2 > 0:
            print(f"  --> 3D saves {(1 - a3/a2)*100:.1f}% energy vs 2D baseline")

    # ---------------- Pareto frontier ----------------
    if args.pareto:
        print("\n=== Energy vs QoS (R_min) frontier — 3D model ===")
        trainer = build_trainer(params, "3d", "attention", args.embed_dim, args.num_layers, args.device)
        load_if_available(trainer, params, "3d", "attention", args.device)
        batch = make_batch("3d", n_inst, N, params, seed=args.seed)
        sweep = params["qos"]["R_min_sweep"]
        energies, served_pct, rmins = [], [], []
        for rmin in sweep:
            ev = trainer.evaluate(batch, R_min=float(rmin), greedy=True)
            energies.append(ev["energy_wh"])
            served_pct.append((1 - ev["unserved_frac"]) * 100)
            rmins.append(rmin)
            print(f"  R_min {rmin:>9.1f} | energy {ev['energy_wh']:8.2f} Wh | "
                  f"served {served_pct[-1]:5.1f}%")
        try:
            viz.plot_pareto_curve(served_pct, energies, rmins, filename="pareto_rmin.png")
        except Exception as e:
            print(f"[warn] pareto plot failed: {e}")

    # ---------------- priority-aware evaluation ----------------
    if args.priority:
        print("\n=== Priority-aware evaluation (same priority labels) ===")
        w, r = make_priorities(n_inst, N, params, seed=args.seed)
        # priority model tag: CMDP (dual-managed) or the fixed-lambda variant.
        pri_tag = "3d_attention_priority_cmdp" if args.cmdp else "3d_attention_priority"
        # priority-3D (trained with --priority/--cmdp), priority-blind 3D (energy-only), 2D baseline
        # (label, mode, encoder, checkpoint tag, priority_feature)
        specs = [("priority 3D", "3d", "attention", pri_tag, True),
                 ("blind 3D",    "3d", "attention", "3d_attention", False),
                 ("2D",          "2d", "attention", "2d_attention", False)]
        pri_batch_3d = make_batch("3d", n_inst, N, params, seed=args.seed)
        # 2D batch is the SAME instances as 3D (drop the elevation column) so the
        # three baselines are compared on identical x/y/demand layouts.
        pri_batch_2d = pri_batch_3d[..., [0, 1, 3]].contiguous()
        print(f"  {'model':11s} | {'energy':>9} | {'served':>7} | {'data MB':>8} | "
              f"{'high-QoS':>8} | {'med-QoS':>7} | {'allQoS':>7} | penalty")
        for label, mode, enc, tag, pf in specs:
            trainer = build_trainer(params, mode, enc, args.embed_dim, args.num_layers,
                                    args.device, priority_feature=pf)
            load_tag(trainer, params, tag, args.device)
            bb = pri_batch_3d if mode == "3d" else pri_batch_2d
            # qos_decode=pf: only the priority-aware model gates its decoder on the
            # floors (as it was trained); blind-3D/2D run their ungated policy and we
            # MEASURE their QoS satisfaction against the same floors.
            ev = trainer.evaluate(bb, R_min=0.0, greedy=True, node_weights=w, node_rmin=r,
                                  qos_decode=pf)
            print(f"  {label:11s} | {ev['energy_wh']:7.2f}Wh | "
                  f"{(1-ev['unserved_frac'])*100:6.1f}% | {ev['data_collected']:8.1f} | "
                  f"{ev['high_qos_satisfaction']*100:7.1f}% | {ev['med_qos_satisfaction']*100:6.1f}% | "
                  f"{ev['priority_satisfaction']*100:6.1f}% | {ev['priority_penalty']:.3f}")

        # critical-floor frontier: tighten the QoS floors and watch energy vs QoS-met
        print("\n  -- critical-floor frontier (priority-3D model) --")
        trainer = build_trainer(params, "3d", "attention", args.embed_dim, args.num_layers,
                                args.device, priority_feature=True)
        load_tag(trainer, params, pri_tag, args.device)
        for scale in [0.0, 0.5, 1.0, 1.5]:
            ev = trainer.evaluate(pri_batch_3d, R_min=0.0, greedy=True,
                                  node_weights=w, node_rmin=r * scale)
            print(f"  floor x{scale:>3.1f} | energy {ev['energy_wh']:8.2f} Wh | "
                  f"critical-QoS met {ev['priority_satisfaction']*100:5.1f}%")

    # ---------------- trajectory plot ----------------
    if args.plot:
        # When --priority is also set, plot the priority model faithfully (encoder
        # sees priority + decoder applies per-node QoS gating); else the energy-only model.
        plot_priority = args.priority
        tag = ("3d_attention_priority_cmdp" if args.cmdp else "3d_attention_priority") \
            if plot_priority else "3d_attention"
        print(f"\n=== Plotting one 3D instance ({'priority' if plot_priority else 'energy-only'} model) ===")
        trainer = build_trainer(params, "3d", "attention", args.embed_dim, args.num_layers,
                                args.device, priority_feature=plot_priority)
        load_tag(trainer, params, tag, args.device)
        N_plot = min(N, 60)
        batch = make_batch("3d", 1, N_plot, params, seed=args.seed)
        if plot_priority:
            pw, pr = make_priorities(1, N_plot, params, seed=args.seed)
            routes, altitudes = trainer.extract_routes(batch, R_min=0.0, node_weights=pw, node_rmin=pr)
        else:
            routes, altitudes = trainer.extract_routes(batch, R_min=params["qos"]["R_min"])
        nf = batch[0].cpu().numpy()
        try:
            viz.plot_3d_trajectories(nf[:, :2], nf[:, 2], routes, altitudes,
                                     trainer.depot, filename="routes_3d.png")
            viz.plot_altitude_profile(altitudes, filename="altitude_profile.png")
        except Exception as e:
            print(f"[warn] trajectory plot failed: {e}")

    print("\n[done] evaluation complete")


if __name__ == "__main__":
    main()
