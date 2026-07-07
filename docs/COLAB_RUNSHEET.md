# Colab Run-Sheet — ATOM-3D-Priority (CMDP)

Exact commands to train and evaluate the priority models on a Colab GPU. Copy each block into
its own Colab cell.

> **Golden rule:** every model (2D, 3D-blind, 3D-CMDP, GNN) must be trained at the **same**
> `--embed-dim` and `--num-layers` (and the config's `H_heads`), and evaluated at those same
> dims, or `run_eval` fails with a size-mismatch. The priority/CMDP model auto-uses a 5-feature
> encoder and the others 4 — that difference is handled for you; only embed-dim / layers / heads
> must match. **N must also be identical across train, eval, and the deterministic baselines.**

> **Headline solver = primal-dual CMDP** (`--cmdp`): per-class QoS dual multipliers replace the
> old fixed `--lambda-priority` penalty. The fixed-lambda path is kept only as a legacy ablation
> (Section 4).

> **⚠️ ALTITUDE-GRADIENT FIX (2026-07-02) — retrain everything 3D from scratch.**
> `trajectory_decoder.py` previously sampled the altitude with a non-detached `rsample()`,
> which made the altitude head's REINFORCE gradient identically zero: every pre-fix 3D
> checkpoint has an *untrained* altitude head (stuck at the ~85 m sigmoid init), and ones
> saved under the (now-reverted) Option D optimizer split won't even load. Before running:
> 1. **Sync the fixed code to Drive** — `trajectory_decoder.py` must contain `dist.sample()`
>    (not `rsample()`) in the altitude action, and `tenma_trainer.py` must have the single
>    Adam + ExponentialLR actor optimizer (no altitude param group). Verify in Colab:
>    `!grep -n "dist.sample()" atom_3d/models/trajectory_decoder.py` — must print a match.
> 2. **Delete or archive all old `checkpoints/3d_*.pt` (incl. GNN-3D)** and do **NOT** pass
>    `--resume` into them. The 2D checkpoint is unaffected (no altitude head) and may be kept.
> 3. Sanity signal that the fix is live: in the 3D training logs, greedy hover altitude must
>    MOVE within the first ~50 epochs (in a no-QoS smoke it climbs from ~85 m; under CMDP the
>    per-class pattern is the Section 3 hypothesis). A flat ~81–85 m altitude = stale code.

---

## 0. Runtime + setup

1. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 15 GB works at N=200; use
   L4 24 GB / A100 40 GB for N=500 — see the OOM note at the bottom).

```python
# Cell 1 — confirm GPU + reduce allocator fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

```python
# Cell 2 — mount Drive and cd into the repo root (the folder that CONTAINS the `atom_3d` package)
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/atom_3d      # <-- adjust to where you put the project
!ls atom_3d/configs/params.yaml          # sanity: should print the path, not an error
```

```python
# Cell 3 — dependencies (Colab already ships torch + CUDA)
!pip -q install "numpy>=1.24" "scipy>=1.10" "pyyaml>=6.0" "matplotlib>=3.7" "tqdm>=4.65"
```

> **Memory-safe defaults below: `--N 200 --embed-dim 256 --num-layers 3`.** This is ~12× less
> activation memory than the old `N=500 / embed-512 / batch-64` recipe that OOM'd a T4 (N² term
> 6.3× smaller, embed term 2× smaller). If you have an L4/A100, you can raise to `--N 500`
> and/or `--embed-dim 512` — keep them identical across ALL cells.

---

## 1. Train the models

```python
# Cell 4 — 2D baseline (Paper A). 2D is the memory hog (N sequential decode steps); keep batch 32.
!python -m atom_3d.experiments.run_train --mode 2d --encoder attention \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 32 --device cuda
```

```python
# Cell 5 — 3D blind (energy-only; the priority-BLIND baseline)
!python -m atom_3d.experiments.run_train --mode 3d --encoder attention \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda
```

```python
# Cell 6 — 3D CMDP (the HEADLINE model). Per-class QoS duals; writes 3d_attention_priority_cmdp.pt
# Watch the log: lam_high / lam_medium rise while hiQoS/medQoS < target, then plateau as the
# policy learns to meet the floors (dual convergence). CMDP hyperparams come from config 'cmdp'.
!python -m atom_3d.experiments.run_train --mode 3d --encoder attention --cmdp \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda
```

Checkpoints land in `checkpoints/`: `2d_attention.pt`, `3d_attention.pt`,
`3d_attention_priority_cmdp.pt`. Training curves go to `figures/`. Re-run any cell with
`--resume` to continue from its checkpoint (the CMDP cell also restores the dual multipliers) —
but ONLY into checkpoints created *after* the 2026-07-02 altitude-gradient fix; never resume a
pre-fix 3D checkpoint (untrained altitude head; Option D optimizer state fails to load).

---

## 2. Evaluate (headline + baselines)

Use the **same** `--N` / `--embed-dim` / `--num-layers` as training.

```python
# Cell 7 — PRIORITY headline: CMDP-3D vs blind-3D vs 2D + critical-floor frontier
# --cmdp makes run_eval load the *_priority_cmdp checkpoint for the priority-3D row.
!python -m atom_3d.experiments.run_eval --priority --cmdp \
    --N 200 --instances 100 --embed-dim 256 --num-layers 3 --device cuda
```

Read Cell 7 at **equal critical-QoS**:
- **2D** meets `high-QoS` highly but at the HIGHEST `energy` (can't dive/climb; fails the tallest
  high-priority sensors — see the AUTO audit).
- **blind 3D** is cheap but `high-QoS` ~0% (ignores priority).
- **CMDP 3D** should reach `high-QoS` ~100% at `energy` BELOW 2D — meeting critical QoS that
  blind-3D cannot, more cheaply than 2D. That three-way contrast IS the contribution.

```python
# Cell 8 — deterministic baselines through the SAME corrected scorer (CPU-side; match --N).
# strong_coupled local search is slow at N=200 (minutes); reduce --instances if needed.
!python -m experiments.two_stage_vs_coupled --N 200 --instances 20 --seed 2026
!python -m experiments.strong_coupled       --N 200 --instances 20 --seed 2026 --restarts 2
```

```python
# Cell 9 — (sanity) energy table 3D-blind vs 2D vs GNN at UNEQUAL QoS (secondary, not the headline)
!python -m atom_3d.experiments.run_eval --compare \
    --N 200 --instances 100 --embed-dim 256 --num-layers 3 --device cuda
```

```python
# Cell 10 — plot one CMDP trajectory (altitude tracks sensor elevation for critical nodes)
!python -m atom_3d.experiments.run_eval --priority --cmdp --plot \
    --N 200 --instances 20 --embed-dim 256 --num-layers 3 --device cuda
```

---

## 3. After training — altitude-law validation (working hypothesis)

Once the CMDP checkpoint exists, re-run the per-class `H = a·z + b` fit (slope / intercept / R²
and the clearance `H − z` distribution) on the **trained CMDP policy's** hovers, broken down by
High / Medium / Low priority, and compare to Strong-Coupled. If both solvers independently land on
`H ≈ z + h_safe` (high R²) for high-priority nodes, that is a headline finding. (Analysis script
template: the session's `audit_altitude_law.py`; swap the planner for a `run_eval`/policy rollout.
Report PER-CLASS, never pooled — the pooled fit is a Simpson's-paradox artifact.)

---

## 4. (Legacy / optional) Fixed-lambda priority + lambda sweep

Superseded by CMDP; keep only if you want the fixed-multiplier ablation. Tag = `*_priority`.

```python
# Fixed-lambda priority model (legacy): lambda must be O(20-50) to shape QoS at this energy scale.
!python -m atom_3d.experiments.run_train --mode 3d --encoder attention --priority --lambda-priority 20 \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda
# Evaluate WITHOUT --cmdp so it loads the *_priority (fixed-lambda) checkpoint:
!python -m atom_3d.experiments.run_eval --priority \
    --N 200 --instances 100 --embed-dim 256 --num-layers 3 --device cuda
```

```python
# GNN encoder ablation (attention -> plain message passing); energy-only, same dims.
!python -m atom_3d.experiments.run_train --mode 3d --encoder gnn \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda
```

---

## Notes & gotchas

- **OOM is the #1 issue, and N is the dominant lever (N², not batch).** 2D visits every node
  one-by-one → N sequential attention decode steps, each re-projecting all N embeddings → the
  retained autograd graph scales ~ N · steps · batch ≈ N²·batch (3D is similar via the footprint
  rollout). So lowering batch barely helps; **lower N** (quadratic) and/or **embed-dim** (linear).
  The failing run was `N=500 / embed-512 / batch-64`; the safe recipe here is `N=200 / embed-256`,
  2D batch 32 / 3D batch 64, plus `expandable_segments:True` (Cell 1). To keep `N=500`, use an
  L4 (24 GB) or A100 (40 GB) runtime. **All models AND the deterministic baselines must use the
  same N** for a fair comparison.
- **Dims must match** between every `run_train` and `run_eval` for the same model, or loading
  fails with a size-mismatch. Batch size does NOT change saved weights, so different batch sizes
  per model stay comparable.
- **CMDP eval tag:** `run_eval --priority --cmdp` loads `3d_attention_priority_cmdp.pt`; without
  `--cmdp` it loads the fixed-lambda `3d_attention_priority.pt`. Use `--cmdp` for the headline.
- **Channel regime:** `params.yaml` has `path_loss_exponent: 3.0` (de-saturated urban/suburban)
  and `wpt_enabled: false` (WPT is dormant, not part of this contribution). Per-class floors are
  high 38 / med 28 / low 0 Mbps. Do not revert these for the priority runs.
- **Checkpoints persist** under `checkpoints/` on the mounted Drive across sessions.
- **Quick smoke** before the long runs: append `--N 30 --epochs 5 --batch-size 8 --ckpt-dir /tmp/smoke`
  to a train cell so it finishes in seconds without touching real checkpoints.
- **Honesty reminder:** the locked claim is *priority-aware 3D meets critical (high-class) QoS at
  LOWER energy than the 2D baseline and at higher critical-QoS than priority-blind 3D* — a
  CONDITIONAL (equal-QoS) energy claim, plus the energy↔QoS frontier. NOT the falsified
  unconditional "3D saves energy" story. The constraint-conditioned altitude law (Section 3) is a
  WORKING HYPOTHESIS pending the post-training validation, not yet a paper claim.
```
