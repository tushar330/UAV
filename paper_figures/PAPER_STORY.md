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
- **Low priority vs 2D-AUTO is only marginal** (+7.5, t p=0.023, Wilcoxon
  p=0.0486 — right at the line at n=20). Do not lead with it. "Without
  degrading low-priority service" remains the safe phrasing against 2D. The
  large low-priority gain (+21.5) is against **Two-Stage**, and that one is
  robust.
- **Do not claim "coupling beats decoupling" as the mechanism.** At n=20
  Coupled-Greedy and Two-Stage tie on the critical class at **exactly 54.9%
  (diff +0.00, p=1.000)**. Coupling alone buys nothing there; the entire
  critical-QoS advantage (+15.3) comes from the **continuous-altitude local
  search**. State the refinement as the mechanism — the ablation supports that
  precisely, and the alternative claim is contradicted by our own Fig. 5.
- **Report the local search's cost honestly** (~30 min per city at N=500). It is
  load-bearing, not a refinement.
- **Do not describe any altitude as learned.** Every checkpoint to date has its
  altitude head still at initialisation.
- **Two-Stage must stay in the main results.** It is the strongest baseline; a
  reviewer who discovers a decoupled 3D method reaches 55% while the paper only
  showed 2D's 0.8% will read it as baseline-hiding.

---

## Secondary result — SETTLED at n=20 (2026-08-18)

Ours vs Two-Stage, paired, n=20 — **significant on all three QoS classes at
statistically indistinguishable energy**:

| class | Two-Stage | Ours | diff | t p | Wilcoxon p | wins |
|---|---|---|---|---|---|---|
| high | 54.9% | 70.2% | +15.3 | 0.0009 | 0.0015 | 14/20 |
| medium | 66.8% | 80.8% | +14.1 | 0.0013 | 0.0010 | 19/20 |
| low | 52.7% | 74.2% | +21.5 | <0.0001 | 0.0001 | 19/20 |
| energy | 79.4 kJ | 78.5 kJ | −0.9 | 0.33 | 0.50 | — |

The borderline n=10 result (high p=0.033/0.063) resolved cleanly. Both the
t-test and the rank test now agree on every class, so the earlier
outlier-sensitivity caveat no longer applies.
