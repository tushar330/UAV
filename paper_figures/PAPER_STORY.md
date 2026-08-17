# PAPER STORY

Rewritten **2026-08-17** to match what the figures actually show. The previous
version described a *learned* contribution ("Adaptive Altitude is learned, not
hand-tuned") and a "~30% cheaper than 2D" energy claim. Both are false for this
manuscript: no trained policy is used, and under the budget regime energy is a
controlled variable rather than a result. Writing from the old version would
have produced claims the figures contradict.

---

## The one-sentence claim

> The base paper plans in 2D at a fixed altitude. We extend it to 3D, choosing
> hover placement and altitude jointly under each node's QoS rate floor, and
> under an equal energy budget this delivers dramatically better
> criticality-aware service.

**This is a deterministic-planner paper.** No reinforcement learning is claimed
anywhere. See the `atom3d-deterministic-paper-scope` decision record.

---

## The chain the figures must communicate

```
2D fixed altitude cannot reach critical nodes   (Fig. 1, 3)
        ↓
3D: descend selectively, guided by QoS floors   (Fig. 3, 7, 10)
        ↓
Critical-node QoS rises from 0.8% to 67.4%      (Fig. 5, 9)
        ↓
...at the SAME energy budget                    (Fig. 6)
        ↓
...and beats the natural decoupled 3D approach  (Fig. 5, 13)
```

---

## Reading order

**Fig. 1 — Environment.** Nodes are not equal: some are critical.

**Fig. 2 — Method overview.** How placement and altitude are decided together.

**Fig. 3 — Altitude vs mission progress.** The visual thesis: ours dives over
critical nodes; Two-Stage repairs only after the fact; 2D-AUTO is a flat line.

**Fig. 5 — Per-class QoS. THE RESULT.** High 0.8 → 67.4% (p<0.001, 10/10 seeds),
medium 61.6 → 78.2% (p=0.0016). Low priority is *not* significantly different
and must not be claimed as a win.

**Fig. 6 — Energy.** Read as a control, not a result: every method flies until
the same budget B is spent. This is what makes Fig. 5 a fair comparison.

**Fig. 7 — Trajectories** and **Fig. 9 — Rate CDF** and **Fig. 10 — Altitude
distribution.** The mechanism: ours is bimodal (cruise + dive-to-serve), which
shifts the high-priority rate distribution across the floor.

**Fig. 13 — Summary table.** Every number with a Student-t 95% CI and a paired
test, plus the explicit ours-vs-Two-Stage comparison.

---

## Guardrails — claims the data does NOT support

- **Do not claim an energy saving.** Energy is equalised by construction. The
  residual gap is hover granularity (both methods underspend B; ours underspends
  more because its hovers are chunkier).
- **Do not claim a win on low priority.** p=0.16 vs 2D-AUTO. Say "without
  degrading" instead.
- **Do not claim "coupling beats decoupling" as the mechanism.** Coupled-Greedy
  (51.6% high) does *not* beat Two-Stage (55.0%). The gain comes from coupling
  **plus the continuous-altitude local search**; the ablation proves this, and
  claiming otherwise is contradicted by our own Fig. 5.
- **Report the local search's cost honestly** (~30 min per city at N=500). It is
  load-bearing, not a refinement.
- **Do not describe any altitude as learned.** Every checkpoint to date has its
  altitude head still at initialisation.
- **Two-Stage must stay in the main results.** It is the strongest baseline; a
  reviewer who discovers a decoupled 3D method reaches 55% while the paper only
  showed 2D's 0.8% will read it as baseline-hiding.

---

## Secondary result

Ours vs Two-Stage, paired, n=10: high +12.4 (p=0.033, Wilcoxon p=0.063),
low +22.2 (p=0.0006), medium +5.0 (n.s.), energy comparable (n.s.). Only low
priority is robustly significant; high is borderline and the per-seed
differences include four exact ties. **More seeds would firm this up** — say so
rather than overstating it.
