"""
Training driver for ATOM-3D / 2D-AUTO.

Examples
--------
    # train the 3D attention policy (headline model)
    python -m atom_3d.experiments.run_train --mode 3d --encoder attention

    # train the 2D Paper-A baseline
    python -m atom_3d.experiments.run_train --mode 2d --encoder attention

    # train the GNN ablation (attention -> plain message passing)
    python -m atom_3d.experiments.run_train --mode 3d --encoder gnn

Checkpoints are written to ``<checkpoint_dir>/<mode>_<encoder>.pt`` so the eval
driver can load 2D, 3D and GNN models side by side.
"""

import argparse
import os

import numpy as np
import torch
import yaml

from ..training import TENMATrainer, TrainConfig
from ..env.iot_env import IoTEnvironment2D, IoTEnvironment3D
from ..utils.visualization import TrajectoryVisualizer


def load_params(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_batch(mode: str, batch_size: int, N: int, params: dict, seed=None):
    sc = params["scenario"]; nd = params["nodes"]
    kw = dict(seed=seed, Di_min=nd["Di_min"], Di_max=nd["Di_max"],
              area_width=sc["area_width"], area_height=sc["area_height"])
    if mode == "3d":
        kw.update(zi_min=nd["zi_min"], zi_max=nd["zi_max"])
        return IoTEnvironment3D.generate_batch(batch_size, N, **kw)[0]
    return IoTEnvironment2D.generate_batch(batch_size, N, **kw)[0]


def make_priorities(mode: str, batch_size: int, N: int, params: dict, enabled: bool, seed=None):
    """Per-node (weights, R_min) for a batch, or (None, None) when priority is off.

    Priority is a 3D-only extension; in 2D or when disabled we return no arrays so
    the trainer behaves exactly as the energy-only baseline.
    """
    if mode != "3d" or not enabled:
        return None, None
    pc = params["priority"]
    _, w, r = IoTEnvironment3D.generate_priorities(
        batch_size, N, seed=seed, priority_enabled=True,
        priority_class_probs=pc["class_probs"],
        priority_weights=[pc["weights"]["high"], pc["weights"]["medium"], pc["weights"]["low"]],
        priority_rmin=[pc["R_min"]["high"], pc["R_min"]["medium"], pc["R_min"]["low"]],
    )
    return w, r


def main():
    ap = argparse.ArgumentParser(description="Train ATOM-3D / 2D-AUTO policy")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "params.yaml"))
    ap.add_argument("--mode", choices=["2d", "3d"], default="3d")
    ap.add_argument("--encoder", choices=["attention", "gnn"], default="attention")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--N", type=int, default=None, help="nodes per instance (default: config scenario.N)")
    ap.add_argument("--R-min", type=float, default=None, help="per-node QoS floor (default: config qos.R_min)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr-actor", type=float, default=None, help="override actor LR (default: config)")
    ap.add_argument("--lr-critic", type=float, default=None, help="override critic LR (default: config)")
    ap.add_argument("--entropy-coef", type=float, default=0.0, help="entropy bonus for exploration")
    ap.add_argument("--freeze-altitude", type=float, default=None,
                    help="3D ablation: hover at this single altitude (disables the altitude head)")
    ap.add_argument("--priority", action="store_true",
                    help="enable priority-aware training (per-class R_min gate + weighted penalty)")
    ap.add_argument("--cmdp", action="store_true",
                    help="primal-dual CMDP: per-class QoS dual multipliers replace the fixed "
                         "lambda_priority penalty (implies --priority; hyperparams from config 'cmdp')")
    ap.add_argument("--lambda-priority", type=float, default=None,
                    help="override reward.lambda_priority (weight on the priority penalty)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="override checkpoint output dir (use a scratch dir for smoke tests so "
                         "real checkpoints are never overwritten)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-instances", type=int, default=None)
    args = ap.parse_args()

    params = load_params(os.path.abspath(args.config))
    tcfg = params["training"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else tcfg["batch_size"]
    N = args.N if args.N is not None else params["scenario"]["N"]
    R_min = args.R_min if args.R_min is not None else params["qos"]["R_min"]
    seed = args.seed if args.seed is not None else tcfg["seed"]
    n_eval = args.eval_instances if args.eval_instances is not None else tcfg["num_eval_instances"]

    torch.manual_seed(seed); np.random.seed(seed)

    # Priority-aware training (3D only). CLI overrides the config switch / weight.
    # CMDP needs the per-class QoS floors + satisfaction, so it implies priority.
    cmdp_on = (args.mode == "3d") and args.cmdp
    priority_on = (args.mode == "3d") and (args.priority or cmdp_on or params.get("priority", {}).get("enabled", False))
    if args.lambda_priority is not None:
        params["reward"]["lambda_priority"] = args.lambda_priority
    # The fixed-lambda penalty is irrelevant under CMDP (dual multipliers manage QoS).
    if priority_on and not cmdp_on and float(params["reward"].get("lambda_priority", 0.0)) <= 0.0:
        print("[warn] priority enabled but reward.lambda_priority<=0 — the penalty will not "
              "shape training; pass --lambda-priority > 0 (e.g. 1.0).")

    lr_actor = args.lr_actor if args.lr_actor is not None else tcfg["lr_actor"]
    lr_critic = args.lr_critic if args.lr_critic is not None else tcfg["lr_critic"]
    train_cfg = TrainConfig(
        mode=args.mode, encoder=args.encoder,
        embed_dim=args.embed_dim, num_heads=params["encoder"]["H_heads"],
        ff_dim=args.embed_dim, num_layers=args.num_layers,
        dropout=params["encoder"]["dropout"],
        lr_actor=lr_actor, lr_critic=lr_critic,
        lr_decay=tcfg["lr_decay"], max_grad_norm=tcfg["max_grad_norm"],
        entropy_coef=args.entropy_coef,
        freeze_altitude=args.freeze_altitude,
        priority_feature=priority_on,
        cmdp=cmdp_on,
        device=args.device,
    )
    trainer = TENMATrainer(params, train_cfg)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else params["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"{args.mode}_{args.encoder}"
    if args.freeze_altitude is not None:
        tag += f"_frozen{int(args.freeze_altitude)}"
    if priority_on:
        tag += "_priority"   # keep priority models separate from the energy-only ones
    if cmdp_on:
        tag += "_cmdp"       # keep CMDP (dual-managed) models separate from fixed-lambda ones
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    start_epoch = 1
    if args.resume and os.path.exists(ckpt_path):
        blob = torch.load(ckpt_path, map_location=args.device)
        trainer.load_state_dict(blob)   # accepts both the epoch-wrapped and bare formats
        # New checkpoints carry "epoch"; legacy bare ones don't -> warm start at 1.
        start_epoch = int(blob.get("epoch", 0)) + 1 if isinstance(blob, dict) else 1
        print(f"[resume] loaded {ckpt_path} @ epoch {start_epoch - 1} -> continuing at {start_epoch}")

    # Fixed evaluation set (same instances every epoch for a stable curve).
    eval_batch = make_batch(args.mode, n_eval, N, params, seed=seed + 9999)
    eval_w, eval_r = make_priorities(args.mode, n_eval, N, params, priority_on, seed=seed + 9999)

    history = {"reward": [], "actor_loss": [], "critic_loss": [],
               "avg_data": [], "avg_energy": []}

    qos_mode = ("cmdp(dual)" if cmdp_on
                else "on(lambda=%.2f)" % params["reward"].get("lambda_priority", 0.0) if priority_on
                else "off")
    print(f"[train] mode={args.mode} encoder={args.encoder} N={N} "
          f"batch={batch_size} epochs={epochs} R_min={R_min} device={args.device} "
          f"priority={qos_mode}")

    for epoch in range(start_epoch, epochs + 1):
        batch = make_batch(args.mode, batch_size, N, params, seed=seed + epoch)
        b_w, b_r = make_priorities(args.mode, batch_size, N, params, priority_on, seed=seed + epoch)
        stats = trainer.train_step(batch, R_min=R_min, node_weights=b_w, node_rmin=b_r)
        trainer.sched_actor.step(); trainer.sched_critic.step()

        if epoch % tcfg["eval_interval"] == 0 or epoch == 1:
            ev = trainer.evaluate(eval_batch, R_min=R_min, greedy=True,
                                  node_weights=eval_w, node_rmin=eval_r)
            history["reward"].append(ev["reward"])
            history["actor_loss"].append(stats["actor_loss"])
            history["critic_loss"].append(stats["critic_loss"])
            history["avg_data"].append(ev["data_collected"])
            history["avg_energy"].append(ev["energy"])
            pri = f" | priorityPen {ev['priority_penalty']:.3f}" if priority_on else ""
            if cmdp_on and "duals" in stats:
                lam = " ".join(f"lam_{c}={v:.3f}" for c, v in stats["duals"].items())
                hi = ev.get("high_qos_satisfaction", float("nan"))
                md = ev.get("med_qos_satisfaction", float("nan"))
                pri += f" | hiQoS {hi*100:.0f}% medQoS {md*100:.0f}% | {lam}"
            print(f"  epoch {epoch:4d} | reward {ev['reward']:.3f} | "
                  f"energy {ev['energy_wh']:.2f} Wh | uavs {ev['num_uavs']:.1f} | "
                  f"unserved {ev['unserved_frac']*100:.1f}%{pri} | "
                  f"actorL {stats['actor_loss']:.2f} criticL {stats['critic_loss']:.2f}")

        if epoch % tcfg["checkpoint_interval"] == 0 or epoch == epochs:
            torch.save({"sd": trainer.state_dict(), "epoch": epoch}, ckpt_path)

    torch.save({"sd": trainer.state_dict(), "epoch": epochs}, ckpt_path)
    print(f"[done] checkpoint saved to {ckpt_path}")

    # Training curves
    if history["reward"]:
        viz = TrajectoryVisualizer(params["paths"]["figure_dir"])
        try:
            viz.plot_training_curves(history, filename=f"training_{tag}.png")
        except Exception as e:
            print(f"[warn] could not plot training curves: {e}")


if __name__ == "__main__":
    main()
