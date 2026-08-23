# Report handoff — what to write, what to quote, what not to claim

**For:** whoever is writing the BTP report.
**Date:** 2026-08-23 · **Branch:** `figures/budget-regime-and-decoder-fix` @ `1ea13c2`
(pushed to `tushar330/UAV`; **not yet merged to `main`**)
**Figures:** all 14 scripts run clean; `paper_figures/results/` is current and reproducible.

`SESSION_HANDOFF.md` is the *developer* handoff and is now partly stale — it quotes a
5-seed run and a `Blind-3D` method that was removed. Numbers below supersede it.

---

## 1. The one-sentence claim

> The base paper plans in 2D at a fixed altitude. We extend it to 3D, choosing hover
> placement and altitude jointly under each node's QoS rate floor, and under an equal
> energy budget this delivers dramatically better criticality-aware service.

**This is a deterministic-planner paper. Do not describe anything as learned, trained,
or reinforcement learning.** No trained policy produces any number in the report. The
method is a coupled planner: greedy cover → continuous-altitude solve → local search →
route. Figure 2(b) draws exactly this.

---

## 2. The numbers to quote

Source: `paper_figures/results_data/overall_metrics.npz`, **n = 20 city seeds (42–61)**,
Student-t 95 % CI, paired t-test vs 2D-AUTO. These are what Figures 5, 6 and 13 render —
they agree with each other by construction, so quote from here and they will match.

| method | energy (kJ) | high (%) | medium (%) | low (%) |
|---|---|---|---|---|
| 2D-AUTO (base paper) | 81.5 ± 0.5 | 0.6 ± 0.9 | 61.0 ± 5.1 | 66.7 ± 4.2 |
| Two-Stage (decoupled) | 79.4 ± 1.0 | 54.9 ± 5.7 | 66.8 ± 5.2 | 52.7 ± 5.4 |
| Coupled-Greedy (ablation) | 80.1 ± 1.2 | 54.9 ± 5.6 | 71.5 ± 5.1 | 64.9 ± 6.1 |
| **Strong-Coupled (ours)** | **78.5 ± 1.5** | **70.2 ± 7.3** | **80.8 ± 3.8** | **74.2 ± 6.2** |

**Headline:** critical-node QoS **0.6 % → 70.2 %** at an equal energy budget.

**Secondary (ours vs Two-Stage, paired, n = 20)** — significant on all three classes at
statistically indistinguishable energy:

| class | Two-Stage | Ours | diff | t p | Wilcoxon p |
|---|---|---|---|---|---|
| high | 54.9 | 70.2 | +15.3 | 0.0009 | 0.0015 |
| medium | 66.8 | 80.8 | +14.1 | 0.0013 | 0.0010 |
| low | 52.7 | 74.2 | +21.5 | <0.0001 | 0.0001 |
| energy | 79.4 kJ | 78.5 kJ | −0.9 | 0.33 | 0.50 |

### Setup constants

- **QoS floors:** high **38**, medium **32.5**, low **29** Mbps (reach 21 / 40 / 60 m).
- **Energy budget:** `B = 82.65 kJ` (= 0.65 × 2D-AUTO's full-city cost). Say "83 kJ".
- **Regime:** budget-constrained. Every method flies until `B` is spent, so energy is
  comparable *by construction* and the comparison is the QoS that budget buys.
  **QoS below 100 % means the budget ran out, not that a constraint was violated** —
  say this explicitly somewhere, or a reader will read it as constraint violation.
- **2D-AUTO's altitude:** it picks one altitude *per scene*; across the 20 seeds that is
  **46–50 m**. Figures that pool seeds say "2D-AUTO (46–50 m)"; Figures 3 and 7 draw the
  single canonical scene and correctly say "(47 m)". Don't "fix" this inconsistency.

---

## 3. Which figures go in the report

**Use these nine.** Everything else must never appear as a result.

| # | File | What it shows |
|---|---|---|
| 1 | `figure01_environment` | The city: buildings, nodes by class, depot |
| 2 | `figure02_method_overview` | (a) the physical mechanism, (b) the planner pipeline |
| 3 | `figure03_altitude_comparison` | Altitude vs mission progress — the visual thesis |
| 5 | `figure05_qos_comparison` | **Per-class QoS — THE RESULT** |
| 6 | `figure06_energy_comparison` | Energy split — read as a *control*, not a result |
| 7 | `figure07_trajectories_3d` | 3D routes over the city |
| 9 | `figure09_rate_cdf` | CDF of high-priority achieved rates |
| 10 | `figure10_altitude_distribution` | Where each policy hovers |
| 13 | `figure13_overall_table` | Summary table, CIs, paired tests |

**Never show these five.** They now carry a red `PLACEHOLDER — synthetic data` or
`NOT A MANUSCRIPT RESULT` watermark for exactly this reason:

- `figure04_coverage_comparison` — invented numbers
- `figure04_training_dynamics` — needs a trained policy; the CMDP run is invalid
- `figure08_pareto_tradeoff` — its sweep data was the old serve-all regime
- `figure11_dual_convergence` — needs a trained policy
- `figure12_ablation` — invented numbers ("+ Learned Altitude", "+ CMDP Duals")

If you see a watermark on a figure you were about to paste, that is the stamp doing its
job. Don't crop it out — take the figure out instead.

### Reading order for the results section

```
2D fixed altitude cannot reach critical nodes   (Fig. 1, 3)
        ↓
3D: descend selectively, guided by QoS floors   (Fig. 3, 7, 10)
        ↓
Critical-node QoS rises from 0.6% to 70.2%      (Fig. 5, 9)
        ↓
...at the SAME energy budget                    (Fig. 6)
        ↓
...and beats the natural decoupled 3D approach  (Fig. 5, 13)
```

---

## 4. Claims the data does NOT support

These are the ones a reviewer will catch. Each is contradicted by our own figures.

1. **Do not claim an energy saving.** Energy is equalised by construction. The residual
   3 kJ gap is hover granularity, not efficiency.
2. **Do not claim "coupling beats decoupling" as the mechanism.** At n = 20,
   Coupled-Greedy and Two-Stage tie on the critical class at *exactly* 54.9 %
   (diff +0.00, p = 1.000). Coupling alone buys nothing there. The entire critical-QoS
   advantage (+15.3) comes from the **continuous-altitude local search** — state that as
   the mechanism. The ablation supports it precisely.
3. **Do not lead with low-priority vs 2D-AUTO.** It is marginal (+7.5, t p = 0.023,
   Wilcoxon p = 0.0486 — right at the line). Safe phrasing against 2D is *"without
   degrading low-priority service"*. The large low-priority gain (+21.5) is against
   **Two-Stage**, and that one is robust.
4. **Do not describe any altitude as learned.** See §1.
5. **Keep Two-Stage in the main results.** It is the strongest baseline. A reviewer who
   discovers a decoupled 3D method reaches 55 % while the report only showed 2D's 0.6 %
   will read it as baseline-hiding.
6. **Report the local search's cost honestly** — ~30 min per city at N = 500. It is
   load-bearing, not a refinement.

---

## 5. Figure changes you may need to reflect in the text

Fixed on 2026-08-23. If the draft was written before this, check these passages:

- **Figure 2(b) was redrawn.** It previously showed a primal-dual actor-critic CMDP
  (graph encoder, pointer decoder, Gaussian altitude head, Lagrangian duals). It now
  shows the deterministic planner. **If your method section describes an encoder/decoder
  architecture, it no longer matches the figure** — and the figure is the correct one.
  ⚠️ *Open question for the team: if the report deliberately presents the CMDP as future
  work, say so and the panel can come back clearly labelled as a planned extension.*
- **QoS floor labels** were showing the pre-recalibration 38 / 25 / 8 Mbps. Now
  38 / 32.5 / 29 everywhere. If the text quotes 25 or 8 Mbps, it is wrong.
- **Figure 7's subtitle** used to name `Blind-3D`, a method removed on 2026-08-17. If
  the text mentions Blind-3D anywhere, delete it.
- **Figure 10** was unreadable (one curve's scale flattened the rest) and now shows the
  real story: ours hovers inside the 25–40 m critical-service band while both 3D
  baselines sit near 68 m.
- **Figure 9** now labels its plateau — roughly 30 % of our high-priority nodes and 45 %
  of the baselines' get essentially zero rate (budget exhausted). Don't describe those
  as "served at a low rate".

---

## 6. Regenerating a figure

Every script is standalone and reads from `paper_figures/results_data/`:

```bash
cd paper_figures && python figure05_qos_comparison.py
```

Writes a 600 DPI PNG **and** a vector PDF into `paper_figures/results/`. **Use the PDF
in the report** — it stays sharp at any size.

Regenerating the *data* (not needed for the report) is a long job and would shift `B`:

```bash
python -m atom_3d.experiments.export_figure_data --budget-frac-of-2d 0.65 \
    --city-seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
```

Before editing any figure script, read the specs in `paper_figures/` — `PROJECT_SPEC.md`,
`FIGURE_LIST.md`, `DATA_SPEC.md`, `CODE_RULES.md`, `PAPER_STORY.md`, `FILE_DEPENDENCIES.md`.
`PAPER_STORY.md` is the authority on what may be claimed.

---

## 7. Open items

1. **Figure 2(b) framing** — deterministic planner (current) vs CMDP-as-future-work. Needs
   a decision from whoever owns the method section.
2. **`two_d_altitude_m` in `labels.json`** is seed 42's altitude, used for both the label
   *and* the budget `B`. The figures now work around it for labelling. Changing it
   properly means re-exporting and shifts `B` — probably not worth it before submission.
3. **Branch is not merged to `main`** and there is no PR open.
4. The report binaries in the repo root (`ATOM-3D-VoI_Research_Report.docx/.pdf`,
   `RESEARCH_OVERVIEW.pdf`, `btp_report.zip`) are untracked and not in git.
