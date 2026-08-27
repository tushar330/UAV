# Session handoff — decoder fix, budget regime, recalibrated floors

**Date:** 2026-08-16 · **Branch:** `figures/budget-regime-and-decoder-fix` @ `28e029a`
(pushed to `tushar330/UAV`; **not yet merged to `main`**)
**Working tree:** clean except the two untracked report binaries.

---

## 1. Headline — the locked claim is now demonstrated

Deterministic export, **5 city realizations** (seeds 42–46), recalibrated floors,
equal budget `B = 0.65 × 2D-AUTO's full-city cost = 82.7 kJ`:

| method | energy | high | med | low |
|---|---|---|---|---|
| 2D-AUTO (base paper) | 82.0 ± 0.3 kJ | 1.6 ± 3.1% | 62.2 ± 10.8% | 68.5 ± 8.0% |
| Blind-3D (control) | 73.8 ± 8.5 kJ | 0.0% | 0.8% | 1.3% |
| **Strong-Coupled (ours)** | **79.6 ± 2.1 kJ** | **68.8 ± 12.5%** | **76.4 ± 7.2%** | **77.4 ± 13.6%** |

**Better QoS on every class at 2.9% less energy than 2D-AUTO.** Figures 5, 6, 7,
13 render this and agree with each other.

> A 2-seed run with `--fast` showed medium regressing (73 vs 75). That was the
> greedy planner, not the method — Strong-Coupled recovers it. **Never quote
> `--fast` numbers as results.**

---

## 2. The invalidated training run (why every checkpoint was thrown away)

The 500-epoch CMDP run pushed on 2026-08-15 scored **0% high-priority QoS**.
Losses and gradients looked healthy the whole way; the tell was in the duals.

**Root cause (structural, not hyperparameters):** the altitude head read only the
pre-selection glimpse, so it never saw *which anchor was chosen*. The anchor is
always served at its own hover with the UAV directly overhead, so its rate
depends only on `H − z_anchor` — a head blind to `z_anchor` cannot satisfy a
per-node floor. It collapsed to a constant **H = 85.5 ± 0.14 m**.

**Second defect, found only by smoke-testing the fix:** conditioning alone did
nothing (still 0% after 150 epochs at both 1e-5 and 1e-3). The feasible band sits
**3.4–5.0 σ** below the initial mean, so the policy never sampled it and never
saw the reward for diving. Fixed by biasing the head's output to `−3.5` so every
hover *starts* 13–19 m above its anchor, inside the critical floor.

After both fixes: **hiQoS 0% → 95%**, and `λ_medium` is driven to 0 rather than
growing — the primal–dual mechanism working in both directions for the first time.

`load_state_dict` now **refuses pre-fix checkpoints by name**. Do not try to
resurrect `archive/` or the old `checkpoints/*.pt`.

---

## 3. What changed, and why each mattered

| # | Fix | Consequence if it had shipped |
|---|---|---|
| 1 | Altitude head conditioned on the chosen anchor | 0% high-QoS |
| 2 | Head initialised inside the feasible band (`INIT_ALT_LOGIT = −3.5`) | 0% high-QoS even with #1 |
| 3 | Floors recalibrated **38 / 32.5 / 29 Mbps**; `params.yaml` matched | 70% of nodes trained with no QoS constraint |
| 4 | `_normalize` centres positions per instance | Half the city outside the encoder's trained input range |
| 5 | `run_train --city` | Trained on uniform-random, scored on clustered cities |
| 6 | 2D altitude computed per realization | Baseline penalised (seed 42: 8% → 0%) |
| 7 | `--learned-atom3d CKPT` | A trained policy had **no route into the figures at all** |
| 8 | `route_order()` truncates along the flown tour | 2D fit 213 nodes instead of 326 — margin inflated by an artifact |

**Why the floors were wrong:** the shipped 38/25/8 Mbps map to reach
21.1 / 94.9 / **690.9** m. On a 1000 m map a 690 m reach can never fail, so the
low bar measured *coverage*, not QoS, and only the critical class exercised a
constraint. Rate is compressed (`~log(1/d³)`), so the usable band is only
25–44 Mbps; the new floors space the classes evenly across it (reach 21/40/60 m).

---

## 4. Formulation change: budget-constrained (Q4/Q6 un-frozen)

Under serve-all with QoS as hard constraints, **every feasible plan scores 100%
by construction** — QoS carried no comparative information. With a mission
budget every method flies its own route until `B` is spent, so energy is
comparable *by construction* and QoS is the comparison. This is what makes
"better QoS at comparable energy" a measurable claim.

`labels.json` records `regime`, `energy_budget_kj` and `class_floors_bps`;
`common_plot.run_regime()` / `budget_footnote()` / `class_floors_mbps()` surface
them so a figure can never be captioned with a regime or floor its data was not
measured against.

---

## 5. Next step — training

```bash
git checkout main && git merge --ff-only figures/budget-regime-and-decoder-fix
git push origin main          # Kaggle clones main
```

```bash
python -m atom_3d.experiments.run_train --mode 3d --cmdp --city --epochs 500 --lr-actor 1e-4
```

```bash
python -m atom_3d.experiments.export_figure_data \
    --budget-frac-of-2d 0.65 --city-seeds 42,43,44,45,46 \
    --learned-atom3d checkpoints/3d_attention_priority_cmdp.pt
```

**The bar to beat is §1** — same planner slot, same budget, so it is a direct
comparison.

**Kill criterion:** if constraint violation is still flat at epoch ~100, stop.
That was the tell last time and it never recovered in 500 epochs.

**`lr_actor` is unresolved.** The config's `1e-5` moved the altitude head's
output bias by 3×10⁻⁵ over 150 epochs; `1e-3` moved it 200× more but energy
drifted *up*. `1e-4` above is a guess — a short 3-way sweep beats committing blind.

**Expect the 95% at epoch 1 to be the initialisation, not learning.** What
training must buy is *energy reduction while holding QoS*. If it ends at ~95% QoS
and the same energy it started with, it has learned nothing useful even though
the curve looks fine.

---

## 6. Open items

1. **Figure 8 is stale.** `pareto_sweep.npz` still holds serve-all-regime data
   (this run used `--skip-pareto`). Do **not** show it beside Figures 5/6/13.
   Regenerating is ~2 h because each sweep point re-plans with Strong-Coupled.
2. **Figures 4, 11, 12 keep the PLACEHOLDER stamp.** They need a trained policy.
   Figure 12's ablation bars need *learned* variants (`fixed_alt`, `learned_alt`,
   `cmdp_uniform`, `full`) and cannot be filled by deterministic planners.
3. **`target_satisfaction: 1.0`** means `λ_high` creeps up forever at anything
   short of perfect, so Figure 11 will not show a settling dual. Consider
   0.95–0.98 if you want that figure to show convergence.
4. **Only `high` and `medium` get duals.** `low` now has a floor but no dual and
   no `low_qos_satisfaction` metric in the scorer; it is pressured through the
   serve-all penalty (μ=5) instead. Adding a third dual means touching the
   deliberately frozen scorer.
5. **Figure 13 bolds Blind-3D as "best" on energy** (73.8 kJ) because it is
   lowest. It is a degenerate control that underspends the budget and scores ~0
   everywhere. Cosmetic, but exclude the control from the bolding rule.
6. **Class mix still differs**: `params.yaml class_probs` is 10/30/60, the city
   is 10/20/70. Only matters for the uniform-random path; `--city` sidesteps it.

---

## 7. Landmines (kept from the previous handoff, still true)

* **`run_train` writes `figures/training_*.png` from config** and overwrites a
  committed artifact on every smoke run. `git checkout -- figures/` afterwards.
* **`synthetic_city.CITY_FILE` is a relative path** — figure scripts must run
  with cwd = `paper_figures/`. The adapter passes an absolute path instead.
* **Unit trap:** Figs 6/8/13 report **kJ**; `MethodRun.energy_wh()` returns Wh.
  Convert (×3.6). This bug shipped once already.
* **`--fast` is for iteration only.** Its `atom3d` row is a different, weaker
  planner and its numbers are not results.
* **Background jobs inherit the shell's cwd.** One export died instantly with
  `ModuleNotFoundError` because cwd had drifted to `paper_figures/`. Pass
  `PYTHONPATH` explicitly.

---

## 8. New tooling worth knowing

* **`--hover-cache PATH`** — pickles built hover plans keyed by
  (method, seed, planner strength, 2D altitude). The Strong-Coupled plan costs
  ~25–38 min per seed and does not depend on the budget, so budget and figure
  iterations become seconds instead of hours.
* **`--priority-schedule`** — serves the critical class first under budget.
  Yields 100/66/72 vs 2D's 2/45/42: strongest critical result, but drives high-QoS
  to exactly 100% and starves bulk coverage. Kept as a secondary result.
* **`tests/test_altitude_head.py`** — the repo's first tests (6, all passing).
  Runs without pytest: `python tests/test_altitude_head.py`. Pins the two
  properties whose absence cost a 500-epoch run.

---

## 9. Non-negotiable (unchanged)

`CLAUDE.md`: *"Never fabricate experimental results. Load results from disk."*

The `PLACEHOLDER — synthetic data` stamp enforces this; Figures 4, 11, 12 keep it
**on purpose**. `results_data/labels.json` is the source of truth for what
produced each curve — never relabel a deterministic planner as a learned policy.
The mock figures in the report are **not reproducible targets**: they show
2D-AUTO at 76% high-QoS, which the physics forbids (at its minimum safe altitude
only ~8% of high-priority nodes are within the 21.1 m the 38 Mbps floor allows).

---

## 10. Deliberately untracked

```
ATOM-3D-VoI_Research_Report.docx   5.7 MB
ATOM-3D-VoI_Research_Report.pdf    1.1 MB
```
Both are now **out of date** — they predate the budget regime and the floor
recalibration.
