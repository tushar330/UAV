# PAPER STORY

The figures must guide the reader through the proposed method as a single
coherent argument — not as 13 independent plots. Every figure exists to
advance one link in the causal chain below.

---

## The chain the paper must communicate

```
Priority Awareness
        ↓
Adaptive Altitude
        ↓
Higher Critical QoS
        ↓
Lower Energy
        ↓
Better Trade-off
```

---

## Reading order (mapped to FIGURE_LIST.md)

**Fig. 1 — Environment.**
Set the stage. A city of buildings, IoT nodes colored by criticality
class, a depot, a UAV. The reader learns nodes are *not equal*: some are
High-criticality.

**Fig. 2 — Blind-3D baseline.**
The UAV flies without priority awareness. Fixed altitude, criticality-
agnostic order. This is what "naive" looks like.

**Fig. 3 — Our CMDP approach (ATOM-3D-VoI).**
Same scene, but the UAV *intentionally descends* near critical nodes.
This is the visual thesis of the paper. → *Priority Awareness → Adaptive Altitude.*

**Fig. 4 — Altitude vs SNR / QoS floors.**
*Why* descending matters: each class has a QoS rate floor that caps the
serviceable altitude. High-criticality → tighter altitude band.
This is the mechanism behind Fig. 3.

**Fig. 13 — Altitude–QoS regression.** *(read alongside Fig. 4)*
Quantitative proof the learned policy obeys the mechanism:
H ≈ node height + h_safe when QoS is active (R² ≈ 0.97, High class).
→ *Adaptive Altitude is learned, not hand-tuned.*

**Fig. 5 — VoI accumulation.**
Because it descends for critical nodes, our method accrues Value-of-
Information faster and higher. → *Higher Critical QoS.*

**Fig. 7 — Coverage / service map** and **Fig. 8 — Priority satisfaction.**
Where and to whom that VoI goes: High-criticality nodes are served at QoS
far more often than under Blind-3D. → *Higher Critical QoS, made concrete.*

**Fig. 6 — Energy breakdown.**
Adaptive altitude is not paid for with extra energy — the 3D policy is
~30% cheaper than 2D operation. → *Lower Energy.*

**Fig. 10 — Method comparison** and **Fig. 12 — Pareto (energy vs VoI).**
Put it together: our method dominates the baselines on the energy/VoI
frontier. → *Better Trade-off.*

**Fig. 9 — CMDP training curves** and **Fig. 11 — Ablation.**
Supporting evidence: the policy converges under constraints, and each
component (learned altitude, criticality weighting, coupled planning)
measurably contributes.

---

## Guardrails for the narrative

- Figures 2 and 3 **must** use the identical scene so the descent is
  visually obvious side by side.
- The same class colors and method colors (see DATA_SPEC.md) are used in
  every figure so the reader never re-learns the legend.
- No figure should introduce a claim the chain above doesn't need. If a
  figure doesn't advance a link, question whether it belongs.
- If the paper's figure numbering is later reordered, update this file
  and FIGURE_LIST.md together — they must never disagree.
