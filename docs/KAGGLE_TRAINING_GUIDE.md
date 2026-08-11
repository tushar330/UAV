# Kaggle Training Guide — ATOM-3D-VoI

End-to-end: train the models on a Kaggle GPU, export the paper figure data, and
pull everything back. For the deeper background on each flag and the failure
modes behind them, see [`COLAB_RUNSHEET.md`](COLAB_RUNSHEET.md); for how results
reach the figures, see [`FIGURE_DATA_PIPELINE.md`](FIGURE_DATA_PIPELINE.md).

Code source of truth: **https://github.com/tushar330/UAV** (`main`).

---

## The three rules that break runs

1. **Every model must use the same `--embed-dim`, `--num-layers`, `--N`** (and the
   config's `H_heads`) across training *and* evaluation, or loading fails with a
   size mismatch. Batch size may differ freely — it does not change saved weights.
2. **`N` is the memory lever, not batch size.** The decode is `N` sequential steps
   over `N` embeddings, so the retained autograd graph scales ≈ `N²·batch`.
   `N=500 / embed-512` OOMs a 16 GB card; `N=200 / embed-256` fits.
3. **Never `--resume` a 3D checkpoint made before 2026-07-02.** Those have an
   untrained altitude head (the `rsample()` bug zeroed its gradient). Verify the
   fix is live before training — Cell 2 below does this.

---

## 0. Runtime

* Settings → Accelerator → **GPU P100** (16 GB). T4 x2 also works, but the code is
  single-GPU, so P100 is the better pick.
* Settings → **Internet ON** (needed for the clone; requires a phone-verified account).
* Quota: ~30 GPU-h/week, 12 h max per session. Use `--resume` to span sessions.

```python
# Cell 1 — confirm GPU + reduce allocator fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
print("CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

```python
# Cell 2 — clone and verify the altitude-gradient fix is present
%cd /kaggle/working
!git clone https://github.com/tushar330/UAV.git
%cd UAV
!grep -n "dist.sample()" atom_3d/models/trajectory_decoder.py   # MUST print a match
```

If that grep prints nothing, you are on stale code — stop and re-clone.

```python
# Cell 3 — dependencies (Kaggle ships torch/CUDA/numpy/scipy/matplotlib)
!pip -q install "pyyaml>=6.0" "tqdm>=4.65"
```

> **Persistence:** `/kaggle/working` lives only as long as the session, and is
> captured into the notebook Output when you **Save Version**. Download
> `checkpoints/*.pt` periodically, or Save Version at the end — otherwise a dead
> session loses the run.

---

## 1. Smoke test first (30 seconds)

Never start a 12-hour run without this. It writes to a scratch directory, so real
checkpoints are untouched.

```python
!python -m atom_3d.experiments.run_train --mode 3d --encoder attention --cmdp \
    --N 30 --epochs 5 --batch-size 8 --embed-dim 32 --num-layers 1 \
    --ckpt-dir /tmp/smoke --device cuda
```

Healthy output: `lam_high` / `lam_medium` **rise** from 0 while `hiQoS` is below
target. That is the dual ascent working.

---

## 2. Train

Same dims everywhere. ~500 epochs each.

```python
# Cell 4 — 2D baseline (Paper A). 2D is the memory hog: N sequential decode steps.
!python -m atom_3d.experiments.run_train --mode 2d --encoder attention \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 32 --device cuda
```

```python
# Cell 5 — 3D blind (energy-only; the priority-BLIND baseline)
!python -m atom_3d.experiments.run_train --mode 3d --encoder attention \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda
```

```python
# Cell 6 — 3D CMDP (the HEADLINE model) + publish Figures 4 and 11
!python -m atom_3d.experiments.run_train --mode 3d --encoder attention --cmdp \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda \
    --export-figure-data
```

`--export-figure-data` writes `paper_figures/results_data/training_log.npz` and
`dual_variables.npz`, which is what makes Figure 4 (training dynamics) and
Figure 11 (dual convergence) render this run instead of placeholders. **Only the
CMDP run produces duals**, so only Cell 6 needs the flag.

Checkpoints land in `checkpoints/` as `2d_attention.pt`, `3d_attention.pt`,
`3d_attention_priority_cmdp.pt`. Add `--resume` to any cell to continue (the CMDP
cell restores the dual multipliers too).

**What to watch:** greedy hover altitude must *move* within the first ~50 epochs.
A flat ~81–85 m means the altitude head is not learning — you are on stale code.

Optional ablation, same dims:

```python
# GNN encoder ablation (attention -> plain message passing)
!python -m atom_3d.experiments.run_train --mode 3d --encoder gnn \
    --N 200 --embed-dim 256 --num-layers 3 --epochs 500 --batch-size 64 --device cuda
```

---

## 3. Evaluate

Use the **same** `--N` / `--embed-dim` / `--num-layers` as training.

```python
# Headline: CMDP-3D vs blind-3D vs 2D, plus the critical-floor frontier
!python -m atom_3d.experiments.run_eval --priority --cmdp \
    --N 200 --instances 100 --embed-dim 256 --num-layers 3 --device cuda
```

Read that table **at equal critical QoS**:

* **2D** — meets high-QoS only from low altitude, at the highest energy.
* **blind 3D** — cheap, but high-QoS ≈ 0% (it ignores priority).
* **CMDP 3D** — should reach high-QoS ≈ 100% at energy *below* 2D. That three-way
  contrast is the contribution.

```python
# Deterministic baselines through the SAME scorer (CPU-side; match --N)
!python -m experiments.two_stage_vs_coupled --N 200 --instances 20 --seed 2026
!python -m experiments.strong_coupled       --N 200 --instances 20 --seed 2026 --restarts 2
```

`strong_coupled` is slow (minutes at N=200; ~23 min at N=500). Reduce
`--instances` if needed.

---

## 4. Export the figure data

The deterministic figure data does not need a GPU and can be run locally, but if
you are already on Kaggle:

```python
!python -m atom_3d.experiments.export_figure_data
```

Add `--fast` to skip the ~23 min local search while iterating, or
`--city-seeds 42,43,44,45,46` for real confidence intervals (multiplies runtime).

This writes `paper_figures/results_data/`, after which every figure script renders
real data and drops its `PLACEHOLDER` stamp automatically.

---

## 5. Get the results out

```python
# Bundle everything worth keeping into one downloadable archive
!tar czf /kaggle/working/atom3d_results.tar.gz \
    checkpoints/*.pt paper_figures/results_data figures/*.png
!ls -lh /kaggle/working/atom3d_results.tar.gz
```

Download it from the file browser (or Save Version). Locally:

```bash
tar xzf atom3d_results.tar.gz
cd paper_figures && for f in figure*.py; do python "$f"; done
```

Figures must be run from `paper_figures/` — `synthetic_city.CITY_FILE` is a
relative path.

To continue training in a **new** session: re-run Cells 1–3, re-upload the `.pt`
files into `checkpoints/` (Add Input → your saved notebook output, then
`!cp /kaggle/input/<slug>/UAV/checkpoints/*.pt checkpoints/`), and re-run the
train cell with `--resume`.

---

## 6. Wiring the trained policy into the figures

Currently the `atom3d` and `3d_gnn` figure slots hold **deterministic planners**,
and `results_data/labels.json` says so. Once you have
`3d_attention_priority_cmdp.pt`, the exporter detects it and prints a NOTE — but
using it for those slots is **not implemented yet**. Until it is, do not relabel
those curves as learned results.

---

## Config notes

`atom_3d/configs/params.yaml` is set for the priority contribution:

* `path_loss_exponent: 3.0` — de-saturated urban/suburban. At α=2 the SNR sits in
  deep log-saturation and altitude barely moves achievable rate, so the QoS floors
  never bind.
* `wpt_enabled: false` — WPT is dormant, not part of this contribution. This also
  disables the closed-form `H_s*` override so the decoder's sampled altitude
  actually drives the reward.

Do not revert these for priority runs.

---

## Honesty reminder

The locked claim is conditional: *priority-aware 3D meets critical (high-class)
QoS at **lower energy than the 2D baseline** and at **higher critical-QoS than
priority-blind 3D**,* plus the energy↔QoS frontier. That is **not** the falsified
unconditional "3D saves energy" story. The constraint-conditioned altitude law is
a working hypothesis pending post-training validation, not yet a paper claim.
