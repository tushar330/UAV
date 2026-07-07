# ATOM-3D-Priority — Project Approach (Beginner-Friendly)

*Plain-language plan for the priority direction. For the energy/altitude background and the
honest negative result that led here, see `PROGRESS_AND_FINDINGS.md` and `APPROACH_SIMPLE.md`.
For the rigorous math, see `PROBLEM_FORMULATION.md`.*

---

## 0. Why we pivoted to this

Our original BTP tried to prove "flying in 3D (changing height) saves battery energy." We built the
whole system and tested it honestly — and found it **does not**: for pure energy, the best choice is
always to **fly low** (wireless charging is fastest close to the sensor). The one 3D result that
**did** survive: height is a real lever for **communication quality** — demanding a higher data rate
forces the drone lower, which costs energy. That is the foundation we build on here.

**New direction:** not all sensors matter equally. A **fire/gas sensor is critical**; a **logging
sensor is not**. We make the drones serve **critical sensors with a stronger link, lower, and
earlier**, and sweep up unimportant sensors **cheaply from high up**. So the drone's **height rises
and falls based on how critical each sensor is.** We keep almost all the code we already built.

---

## 1. The story in one paragraph

Many drones fly over a field of IoT sensors. Each drone flies to a spot, **wirelessly charges** the
sensors there, **collects their data**, moves on, and returns to base. Each sensor has a **priority**
(high / medium / low). High-priority sensors **demand a strong link** (high data rate), so the drone
**dives low** over them; low-priority sensors tolerate a weak link, so the drone **stays high** and
covers many at once. An AI plans this for all drones together. We then measure the trade-off between
**total battery energy** and **how well critical data is served** — a trade-off a fixed-height system
cannot navigate.

---

## 2. What "priority" means here

| Class | Example sensor | Weight `w` | Quality floor `R_min` | Urgency |
|---|---|---|---|---|
| High | Fire / gas leak | 5 | strict (high rate) | collect early |
| Medium | Air quality | 2 | moderate | moderate |
| Low | Logging / archival | 1 | loose (weak link OK) | can wait |

Two consequences:
1. **Quality:** high-priority sensors need a high minimum data rate `R_min`; low-priority ones don't.
2. **Timeliness:** high-priority data should be collected sooner (freshness).

> **Example.** 100 sensors in a 1 km field: 10 fire (high), 30 air-quality (medium), 60 logging
> (low). We want the 10 fire sensors served fast with a strong link — even if it costs more energy —
> and the 60 logging sensors swept up whenever it's cheap.

---

## 3. The key idea: **priority decides height** (our core mechanism)

Two simple physical facts:
- Coverage circle grows with height: **`r = (H − z)·tanθ`** (Paper B). Fly high → big circle → many
  sensors at once.
- Signal strength falls with distance: **`signal ∝ 1/distance²`**. Fly high → far → **weak link →
  slow, low-rate** charging/collection.

Connect to priority:
- **High-priority sensor → strict `R_min` → needs a strong link → drone must DIVE LOW** over it.
- **Low-priority sensors → loose `R_min` → drone STAYS HIGH** and scoops up many in one wide shot.

**So the drone's altitude undulates along the route, driven by the criticality of what it serves.**

> **Worked example (θ = 60°, tanθ ≈ 1.73).**
> - Over a cluster of 8 **logging** sensors → fly to H = 130 m → radius ≈ 225 m → all 8 inside → one
>   cheap hover (weak link is fine).
> - Over a single **fire** sensor → dive to H = 30 m → strong link → meets the strict rate → data
>   collected fast and fresh → climb back.
>
> A fixed-height (2D) drone can't do both — it either wastes effort over logging sensors or fails
> the quality bar over fire sensors. **That gap is our contribution.**

---

## 4. Where we stand vs. the literature (the honesty section)

The priority + WPT + multi-UAV *theme* is **not empty** — we will NOT claim to be first in the theme:

- **Closest prior work:** *AoI-Minimal Task Assignment and Trajectory Optimization in
  Multi-UAV-Assisted Wireless-Powered IoT Networks* (MDPI Drones, 2025) — multi-UAV + WPT +
  Age-of-Information (freshness) + k-means clustering + task assignment + trajectory. **But it uses
  fixed altitude, uniform freshness (no criticality classes), and a genetic-algorithm heuristic.**
- **Paper A (AUTO/ATOM/TENMA):** multi-UAV + WPT + attention-RL planner — but **fixed 20 m altitude,
  no priority.** This is exactly our 2D baseline.
- Across the whole multi-UAV WPT/AoI data-collection sub-field, the UAVs **fly at constant altitude**
  and optimize only hovering *locations*, speed, and associations.

**What is genuinely open (our three novelty pillars):**

| Pillar | Status in literature | Ours |
|---|---|---|
| Priority/freshness + WPT + multi-UAV | occupied (MDPI 2025) | reused |
| **Priority drives VARIABLE ALTITUDE** (coverage vs link-quality per criticality) | **open** — others fix altitude | **headline** |
| **Learned attention-RL planner** over priority + altitude | open — AoI-WPT space uses k-means + genetic heuristics | yes |
| **Energy ↔ priority-quality Pareto** navigated by altitude | open | analysis contribution |

**Framing guardrail:** do NOT define priority as pure freshness/AoI (that collides with MDPI 2025).
Define priority as **criticality CLASSES with per-class data-rate / QoS floors** (these are what
couple to altitude), with freshness as a secondary ordering. The rate-floor-per-class is exactly
what the AoI papers don't have and is what makes altitude matter.

---

## 5. Which papers we use, and what we take from each

| Piece we use | From | Why |
|---|---|---|
| Attention graph planner (ATOM) + RL training (TENMA) | **Paper A — AUTO** | AI that decides which drone visits which sensor, in what order |
| Multi-UAV split (assign sensors to drones, respect battery) | **Paper A — AUTO** | Many drones, not one |
| WPT charging + exact charge/collect timing (KKT formula) | **Paper A — AUTO** | Don't re-learn timing that has a formula |
| Height → coverage cone `r = (H−z)·tanθ` | **Paper B — UAV-ISAC** | Makes height meaningful |
| 3D movement limits (speed, climb rate, min/max height) | **Paper B — UAV-ISAC** | Realistic flight |
| Realistic rotary-wing power model (**hovering is expensive**) | **Zhao 2024 (final.pdf)** | Honest energy accounting |
| Cluster-based collection + heuristic baselines (PSO, GA, mTSP) | **Zhao 2024 (final.pdf)** | Methods to compare against |
| LoS/NLoS link realism + multi-UAV cooperation framing | **Bayerlein 2021 (MARL.pdf)** | Channel realism; drones dividing the field |
| What "priority/AoI/load-balancing" already did (related work we beat) | MDPI 2025; Poudel & Moh 2023; AoI surveys | Position against: they fix altitude, no WPT+criticality+learned combo |
| **Priority → altitude → WPT coupling + energy↔priority-quality Pareto via learned planner** | **Neither — ours** | The missing link |

---

## 6. System model (plain words)

- **Field:** flat 2D area, `N` static sensors; each has `(x, y)`, data amount `D`, priority class +
  weight `w`.
- **Drones:** `M` identical drones; start at base, fly a route of hovers, return. At each hover pick
  position `(x, y)` **and height `H`**; charge + collect every still-unserved sensor inside the
  coverage circle whose link clears its priority `R_min`.
- **Energy bill (per drone, summed):** `flying + climb/descent + charging + collecting`. Flight/hover
  power from the realistic rotary-wing curve (hovering is the most expensive thing). Charge/collect
  time from the WPT + KKT formula (farther = weaker link = longer = more energy).
- **Link:** strength falls with distance (`∝ 1/d²`), LoS/NLoS realism; sets data rate `R` which must
  clear the sensor's priority-dependent `R_min`.

---

## 7. What the planner does — step by step

```
Sensors (x, y, data, PRIORITY) ─► GRAPH ATTENTION ENCODER (Paper A) ─► smart features
                                            │
                                            ▼
   PRIORITY-AWARE MULTI-UAV SPLIT: share sensors so no drone is overloaded with too many
   "must-dive-low" critical stops  (balance the priority+altitude COST, not just sensor count)
                                            │
                                            ▼
   3D DECODER (per drone): pick next hover (x, y) + pick height H
       • critical sensor nearby  → choose LOW H  (strong link, meet strict R_min)
       • cluster of low-priority → choose HIGH H (cover many cheaply)
                                            │
                                            ▼
   At each hover: charge + collect every in-circle sensor that clears its priority R_min;
   high-priority sensors scheduled EARLIER (freshness)
                                            │
                                            ▼
   Add up energy: flying + CLIMB + charging + collecting
                                            │
                                            ▼
   REWARD = −(energy) − λ·Σ wᵢ·penaltyᵢ   (penalty for critical sensor served late or below
   its quality floor);  must serve ALL sensors
                                            │
                                            ▼
   TENMA reinforcement learning improves the planner over many random fields
```

New blocks vs. Paper A: **priority-aware split**, **priority-driven height choice** in the decoder,
and the **priority-weighted reward**.

---

## 8. What "good" means — the objective

We do **not** minimize energy alone (that just says "fly low everywhere"). We minimize a
**priority-weighted cost:**

```
Total cost = Energy  +  λ · Σ  wᵢ · penaltyᵢ
```
`penaltyᵢ` is large if a high-priority sensor was served **late** or **below its quality floor**,
small/zero for low-priority sensors served whenever convenient. `λ` is a knob: turn it up → spend
more energy to serve critical sensors better.

**Headline result = the Pareto curve:** sweep `λ` (or the high-priority `R_min`) → plot **energy vs.
priority-satisfaction.** Our height-varying 3D method should dominate the fixed-height 2D one (better
quality at the same energy, or less energy at the same quality) — a frontier 2D cannot reach.

---

## 9. Experiments

**Headline comparison** — our priority-aware 3D planner vs.
- priority-blind 3D planner (all sensors equal),
- 2D fixed-height priority planner (the published kind),
- heuristic baselines (PSO / Genetic / mTSP from Zhao 2024).

Metrics: total energy, **% high-priority sensors served within quality floor**, **average
freshness/age of high-priority data**, combined **Priority Satisfaction Score**.

**Pareto curve:** sweep high-priority `R_min` (or `λ`) → energy vs. priority-satisfaction frontier.

**Ablations:**
- Freeze altitude → can't meet critical quality without energy spikes ⇒ height helps.
- Remove priority-weighting → critical data late ⇒ priority logic helps.
- Replace attention with plain network → worse routing ⇒ AI planner helps.
- Remove priority from the multi-UAV split → one drone overloaded with critical dives ⇒ balancing helps.

---

## 10. What we reuse vs. build new

**Reuse (already in the codebase):** graph attention encoder + multi-UAV decoder; WPT + KKT timing;
realistic rotary-wing power model; coverage cone + 3D movement; LoS/NLoS channel + energy accounting;
heuristic-baseline harness. (~80%.)

**Build new (the actual research):** priority features per sensor (class + weight); per-class quality
floors `R_min(high/med/low)`; priority-weighted reward (energy + weighted lateness/age);
priority-aware multi-UAV split (balance the dive/quality cost); priority-satisfaction metrics + the
energy↔priority Pareto sweep. (~20%.)

---

## 11. One-sentence novelty statement

> *We make UAV altitude and the wireless-charging schedule respond to **sensor priority**: critical
> sensors pull the drone low for a strong, fast, fresh link while unimportant ones are swept up
> cheaply from high altitude, and we characterise the resulting **energy-vs-priority-quality
> trade-off** with a multi-UAV attention-RL planner — a coupling of priority, variable altitude, and
> WPT that the fixed-altitude AoI/WPT multi-UAV literature and the 2D priority literature do not
> provide.*

---

## 12. Honesty guardrails

- Do **not** claim "height saves energy" — that collapsed under testing.
- Claim: height + priority let the system **serve critical data better for a given energy budget**,
  and trade smoothly along a frontier a 2D system can't reach.
- Do **not** frame priority as pure AoI/freshness (collides with MDPI 2025) — use criticality classes
  with per-class quality floors.

---

## 13. Next steps

1. ~~Focused novelty search on priority + WPT + altitude~~ — **done**; gap confirmed (lead with the
   variable-altitude-as-priority-lever story + learned planner).
2. Lock the priority model (3 classes, weights, per-class `R_min`) in `configs/params.yaml`.
3. Add priority features + priority-weighted reward to the trainer.
4. Add the priority-aware multi-UAV split.
5. Run the headline comparison + Pareto sweep + ablations.
