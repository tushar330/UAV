# ATOM-3D: A Complete Guide to This Research

**Last updated:** 2026-08-18
**Purpose:** Read this one file and you can explain the entire project — what we
started from, what we built, every problem we hit, what we can honestly claim,
and what is left to do.

Language is kept deliberately plain. Formulas appear only where they matter.

---

# PART 1 — THE PROBLEM IN PLAIN WORDS

A city has **500 IoT sensors** spread over a 1 km x 1 km area. They sit on the
ground and on rooftops, so they are at different heights (0–50 m). Each holds a
small amount of data that must be collected.

A **drone (UAV)** flies from a depot, stops at chosen points, collects data by
radio, and returns. The drone has a **limited battery**.

Not all sensors are equally important. We split them into three
**criticality classes**:

| Class | Share | Example | Needs |
|---|---|---|---|
| **High** | 10% | Hospital, power station | Fast, reliable link |
| **Medium** | 20% | Traffic, industrial | Moderate link |
| **Low** | 70% | Environmental sensors | Basic link |

**The question this research answers:**

> Given a fixed battery budget, how should the drone fly — and at what heights —
> so that the *most important* sensors actually get served properly?

---

# PART 2 — THE BASE PAPER (WHAT WE STARTED FROM)

## 2.1 What the base paper did

The base paper is **AUTO** — a 2D UAV data-collection framework for IoT.

Its setup:

- The drone flies at **one fixed height** for the whole mission.
- It visits sensors and collects all their data ("serve-all").
- It uses a **graph attention neural network** to decide the visiting order.
- **Goal: minimise total energy** while collecting everything.

## 2.2 The base paper's energy model

Total energy is the sum over all drones `j`:

```
E_j  =  E^F_j   +   E^T_j    +   E^C_j
        flying      charging     collecting
```

where

```
E^F_j = P^F · T^F_j            (flying:     power x flight time)
E^T_j = (P^T + P^H) · T^E      (charging:   transmit + hover power)
E^C_j = (P^C + P^H) · T^C      (collecting: receive + hover power)
```

The paper's clever part was solving the **time allocation** — how long to charge
versus collect at each stop — in closed form using KKT conditions.

## 2.3 The key limitation we attack

**Everything happens at one altitude. Height is not a decision.**

This matters because of physics. Radio quality falls with distance, so:

- **Fly high** → you see many sensors at once (few stops, less flying), but
  every link is **weak**.
- **Fly low** → links are **strong**, but you see fewer sensors per stop, so you
  need more stops and more flying.

A single fixed altitude must be **high enough to clear the tallest building**.
In our city that forces **47 m**. At 47 m the link is too weak for high-priority
sensors — they can *never* be served properly, no matter how good the route is.

**That is the gap this research fills.**

---

# PART 3 — THE PHYSICS YOU NEED (ONLY 3 FORMULAS)

## Formula 1 — Data rate (how fast we can collect)

```
R = B · log2( 1 + (P_T · beta) / (d^alpha · sigma^2) )
```

- `d` = straight-line 3D distance from drone to sensor
- `alpha = 3` = path-loss exponent (urban)
- `B = 2 MHz`, `sigma^2` = noise power

**Plain meaning:** the further away, the slower. Distance hurts *cubed*.

## Formula 2 — Maximum serving distance (the QoS floor)

Turn Formula 1 around. If a sensor needs a minimum rate `R_min`, the drone must
come within:

```
d_max = [ (P_T · beta) / (sigma^2 · (2^(R_min/B) − 1)) ]^(1/alpha)
```

**This single formula drives the whole paper.** Plugging in our floors:

| Class | Required rate | Drone must be within |
|---|---|---|
| High | 38 Mbps | **21 m** |
| Medium | 32.5 Mbps | **40 m** |
| Low | 29 Mbps | **60 m** |

A drone stuck at 47 m **cannot** get within 21 m of a ground sensor. That is
exactly why the 2D baseline scores ~0% on high priority. It is not a bad
algorithm — it is **geometrically impossible**.

## Formula 3 — Coverage cone (how many sensors one stop can see)

```
rho^2  <=  ((H − z_i) · tan(theta))^2          theta = 60 degrees
```

- `rho` = horizontal distance from drone to sensor
- `H` = drone altitude, `z_i` = sensor height

**Plain meaning:** higher drone = wider circle of sensors covered.

## The tension, in one line

> **Formula 3 wants you high. Formula 2 wants you low. The right altitude
> depends on *which class* of sensor you are serving.**

That is the insight the entire method is built on.

---

# PART 4 — THE JOURNEY (WHAT WENT WRONG AND HOW WE FIXED IT)

This is the honest history. Most of the effort was not building the method — it
was discovering that earlier versions were measuring the wrong thing.

### Problem 1 — The drone just flew as high as possible

**What happened:** We made altitude a decision. The optimiser set it to maximum
and never changed it.

**Why:** With cheap hovering and weak link penalties, "fly high and cover
everything" was genuinely optimal. There was no reason to descend.

**Fix:** Use a **realistic rotary-wing power model** (Zeng–Zhang), in which
**hovering is the most expensive thing a drone can do**:

```
P(V) = P0·(1 + 3V^2/U_tip^2)
     + Pi·( sqrt(1 + V^4/(4·v0^4)) − V^2/(2·v0^2) )^(1/2)
     + 0.5·d0·rho·s·A·V^3
```

Hover power `P(0) = P0 + Pi = 168 W` is the peak of this curve. Now a weak link
is *expensive*, because you must hover longer to collect the same data.
Descending finally pays for itself.

### Problem 2 — Altitude barely changed the data rate

**What happened:** Changing altitude moved the achievable rate by only ~25%. Too
little to matter.

**Why:** The signal was in **log saturation** (SNR 53–71 dB). Inside a
logarithm, huge SNR changes produce tiny rate changes.

**Fix:** Change the path-loss exponent from `alpha = 2` (free space) to
**`alpha = 3`** (realistic urban). The rate band spread to ~86%, and altitude
became a real lever. We checked that the gain is a smooth plateau for `alpha` in
[2.5, 3.5] — it is not knife-edge tuning.

### Problem 3 — Wireless charging swamped everything

**What happened:** With wireless power transfer (WPT), charging time grew with
distance squared, forcing the drone low for reasons unrelated to our
contribution.

**Fix:** **Removed WPT entirely.** It made the story impossible to tell and gave
reviewers an easy attack. Pure data collection is cleaner, and the altitude
trade-off survives without it.

### Problem 4 — Our QoS metric was meaningless (the most important fix)

**What happened:** Under "serve-all with QoS as a hard constraint", **every
valid plan scores 100% QoS by definition.** Our headline metric could not tell
good from bad. We were measuring nothing.

**Fix — change the regime.** Give every method the **same energy budget**:

```
B = 0.65 x (energy 2D needs to cover the whole city) = 82.65 kJ
```

Each method flies until that budget runs out. Now:

- **Energy is equal by construction** — it is a *control*, not a result.
- **QoS is what differs** — it measures how much value the budget bought.

This is what turned "better QoS at comparable energy" into a measurable claim.

### Problem 5 — Two of the three QoS floors did nothing

**What happened:** The floors were 38 / 25 / 8 Mbps. Using Formula 2, those map
to reach distances of 21 m / 95 m / **691 m**. On a 1000 m map, a 691 m reach
can never fail.

So the "low priority" bar was measuring *coverage*, not quality. **70% of
sensors carried no real constraint.**

**Fix:** Recalibrate to **38 / 32.5 / 29 Mbps** → 21 m / 40 m / 60 m. All three
now bind, evenly spaced across the usable rate band (25–44 Mbps).

### Problem 6 — The reinforcement learning never actually learned

We spent significant effort training a neural policy (CMDP). It failed three
times, each for a different reason:

1. **The altitude network could not see which sensor it was serving.** It only
   saw a summary of the whole map. But the required altitude depends entirely on
   *that sensor's* height. It collapsed to a constant 85.5 m and scored **0%**
   on high priority.

2. **After fixing that, it still scored 0%.** The starting altitude sat 3.4–5.0
   standard deviations away from any feasible altitude, so the policy never
   randomly *tried* a good altitude and never discovered that descending helps.
   Fixed by biasing the network to start low, inside the feasible band.

3. **After both fixes, it still did not learn.** We inspected the saved weights:
   the altitude network was **still at its starting values** after 500 epochs.
   The learning rate (`1e-5`) combined with decay (`0.98` per epoch) drove the
   effective rate to `4e-10`. We confirmed the mechanism exactly — predicted
   parameter movement 1.95e-5 versus 1.43e-5 observed.

**Decision:** The final "95% QoS" from that model was simply its **starting
value**, not learning. We **dropped RL from this paper** and used a
deterministic planner. Fixing training is future work.

### Problem 7 — Our statistics were wrong

Three real errors, all found and fixed:

1. **Confidence intervals were 42% too narrow.** We used 1.96 (the normal
   value); with 5 samples the correct Student-t value is **2.776**.
2. **Per-seed results were discarded**, which made the *paired* test impossible —
   and paired is the correct test, since every method runs on identical cities.
3. **Overclaiming.** The figures implied wins on all three classes. At 5 seeds,
   only high priority was actually significant.

### Problem 8 — We were beating a strawman

**What happened:** Our "3D baseline" was called *Blind-3D*, and internally
`3d_gnn` — a name that wrongly suggested a trained neural network. It was
actually a simple greedy heuristic that ignored QoS, scoring **~0–1% on
everything**.

Beating it proved nothing, and the misleading name was a credibility risk.

**Fix:** **Deleted it.** Added the **Two-Stage (decoupled)** planner as a real
baseline — the standard engineering approach: place the stops first, then repair
the altitude problems afterwards.

That change mattered enormously: Two-Stage scores **54.9%** on high priority,
not 1%. Our true margin is over a competent method, not a broken one.

---

# PART 5 — OUR METHOD

## 5.1 The core idea, in one sentence

> **Decide *where* to stop and *how high* to hover at the same time, with each
> sensor's required data rate as a hard constraint — then refine all the
> altitudes continuously.**

Compared with the alternatives:

| Approach | How it works | Weakness |
|---|---|---|
| **2D (base paper)** | One fixed height for everything | Cannot physically reach critical sensors |
| **Two-Stage (decoupled)** | Place stops, *then* repair bad altitudes | The repair pass wastes budget |
| **Ours (coupled)** | Place and choose height *together*, then refine | Slower to compute |

## 5.2 The algorithm (4 steps)

**Step 1 — Compute each sensor's reach limit.** Apply Formula 2 to get `d_max`
for every sensor from its class floor.

**Step 2 — Build clusters greedily, respecting altitude.** Repeatedly pick an
uncovered sensor as an *anchor*, then compute the **shallowest altitude that
still satisfies every member's floor**:

```
H = min over members i of ( z_i + sqrt( d_max,i^2 − rho_i^2 ) )
```

subject to clearance `H >= z_anchor + h_safe` (10 m) and `H` within [20, 150] m.

*Plain meaning:* the altitude is dictated by the **strictest sensor in the
group**. If a critical sensor is in the cluster, everybody gets served from low
altitude.

**Step 3 — Continuous-altitude local search.** *(This is the key step.)*
Repeatedly try moving sensors between clusters, merging clusters, and adjusting
altitudes continuously — keeping any change that lowers total energy.

**Step 4 — Route and spend the budget.** Order the stops with nearest-neighbour
plus 2-opt, then fly until the 82.65 kJ budget is exhausted.

## 5.3 Energy accounting

```
E_total  =  E_flight + E_vertical + E_hover

E_flight   = sum over legs of  L · min_V [ P(V) / V ]     (fly at the most efficient speed)
E_vertical = sum over legs of  (m·g/eta)·climb + c_d·descent
E_hover    = (P0 + Pi) · (total hover time)               (hovering is the expensive state)
```

Climbing costs real energy (`m·g/eta` ≈ 39 J per metre); descending is cheap
(`c_d` = 4 J per metre). This asymmetry is why the planner does not dive
casually — it dives only when a critical sensor makes it worthwhile.

---

# PART 6 — RESULTS

**Setup:** 20 different random city layouts (seeds 42–61), identical for every
method. Equal energy budget B = 82.65 kJ. Paired statistical tests.

## 6.1 The main table

| Method | Energy | High | Medium | Low |
|---|---|---|---|---|
| 2D-AUTO (base paper) | 81.5 kJ | 0.6% | 61.0% | 66.7% |
| Two-Stage (decoupled) | 79.4 kJ | 54.9% | 66.8% | 52.7% |
| Coupled-Greedy (ablation) | 80.1 kJ | 54.9% | 71.5% | 64.9% |
| **Ours (Strong-Coupled)** | **78.5 kJ** | **70.2%** | **80.8%** | **74.2%** |

## 6.2 Against the base paper (the primary claim)

| Class | 2D-AUTO | Ours | Gain | p-value | Cities won |
|---|---|---|---|---|---|
| High | 0.6% | **70.2%** | +69.6 | <0.0001 | **20/20** |
| Medium | 61.0% | **80.8%** | +19.8 | <0.0001 | 18/20 |
| Low | 66.7% | 74.2% | +7.5 | 0.023 | 15/20 |

**Critical sensors go from essentially unservable to 70% served — winning on
every single city tested.**

## 6.3 Against the strong 3D baseline (secondary claim)

| Class | Two-Stage | Ours | Gain | p-value | Cities won |
|---|---|---|---|---|---|
| High | 54.9% | **70.2%** | +15.3 | 0.0009 | 14/20 |
| Medium | 66.8% | **80.8%** | +14.1 | 0.0013 | 19/20 |
| Low | 52.7% | **74.2%** | +21.5 | <0.0001 | 19/20 |
| Energy | 79.4 kJ | 78.5 kJ | −0.9 | 0.33 (no difference) | — |

**Significant on all three classes at statistically identical energy.**

## 6.4 The ablation — what actually causes the gain

This is the most scientifically interesting result:

| Variant | High-priority QoS |
|---|---|
| Two-Stage (decoupled) | 54.9% |
| Coupled-Greedy (coupled, no refinement) | 54.9% |
| **Ours (coupled + refinement)** | **70.2%** |

Coupled-Greedy and Two-Stage tie at **exactly 54.9%** (difference +0.00,
p = 1.000).

> **Conclusion: coupling by itself buys nothing on critical sensors. The entire
> +15.3 point advantage comes from the continuous-altitude local search.**

This is why the paper must credit the **refinement**, not "coupling", as the
mechanism.

---

# PART 7 — WHAT WE CAN AND CANNOT CLAIM

## We CAN say

- Extending 2D to 3D with QoS-aware joint placement and altitude refinement
  raises critical-sensor service from 0.6% to 70.2% at the same energy budget
  (p < 0.0001, 20/20 cities).
- It also beats the standard decoupled 3D approach on all three classes at
  statistically indistinguishable energy.
- The ablation isolates the continuous-altitude refinement as the mechanism.

## We must NOT say

| Wrong claim | Why it is wrong |
|---|---|
| "We save energy" | Energy is **equalised by design**. The 3 kJ gap is leftover budget, not efficiency. |
| "Better on all classes vs 2D" | Low priority is only marginal (p=0.023, Wilcoxon 0.0486). Say "without degrading". |
| "Coupling beats decoupling" | Coupled-Greedy **ties** Two-Stage at 54.9%. The refinement does the work. |
| "The altitude is learned" | No trained policy is used. Every checkpoint's altitude network is still at its initial values. |

---

# PART 8 — SCOPE AND LIMITATIONS

**What this work covers**

- One drone; static, known sensor positions.
- A single synthetic city model (1 km², 126 buildings, 500 sensors).
- Line-of-sight radio with urban path loss (`alpha = 3`).
- Deterministic planning — no learning.
- 20 random layouts from the same city generator.

**Honest limitations**

1. **Computation cost.** The local search takes about **30 minutes per city** at
   500 sensors. Fine for pre-planned missions, not for real-time replanning.
   Since the refinement *is* the contribution, this cost is load-bearing and
   must be reported, not hidden.
2. **One environment.** All results come from one city generator. Other
   densities, building profiles, and sensor distributions are untested.
3. **Simplified radio.** Pure line-of-sight; no blockage, shadowing,
   interference, or multipath.
4. **Static world.** Sensors do not move, appear, or fail. No wind.
5. **Single drone.** Multi-drone code exists but is not evaluated.
6. **Weak low-priority result** against 2D-AUTO (p = 0.023, right at the line).
7. **No wireless charging.** Deliberately removed; results do not transfer to a
   WPT system without re-derivation.

---

# PART 9 — FUTURE IMPROVEMENTS

**Near term**

1. **Make the learning work.** The diagnosis is complete: the learning rate never
   reached the altitude network (effective `4e-10`). Fix the learning rate and
   decay, update per batch instead of once per epoch, and gate the run by
   checking whether the altitude bias actually moves by epoch 50. A working
   policy would plan in **milliseconds instead of 30 minutes** — which is
   precisely the current method's weakness.
2. **Speed up the refinement.** Even a 5x speedup would make in-field replanning
   plausible.
3. **Regenerate the Pareto frontier** (energy versus QoS) under the budget
   regime. It was cut because its data belonged to the old regime.
4. **Test more environments** — different densities, building heights, map sizes.

**Medium term**

5. **Multi-drone.** The code supports splitting work across drones but this is
   unevaluated. A natural extension, though it overlaps published 2D work, so
   the novelty must be argued carefully.
6. **Realistic radio** — blockage and probabilistic line-of-sight.
7. **Dynamic sensors** — data arriving over time, making *freshness*
   (age-of-information) matter alongside rate.

**Longer term**

8. **Real flight tests.** Everything here is simulation.
9. **Online replanning** when sensors fail or conditions change.
10. **Revisit wireless charging** now that the altitude story stands on its own.

---

# PART 10 — PROJECT TIMELINE

| Date | Milestone |
|---|---|
| 2026-07-07 | Initial 3D UAV-IoT codebase with CMDP |
| 2026-07-09 | Figure pipeline (13 IEEE figures) |
| 2026-08-11 | Real results connected into the figures |
| 2026-08-12 | First trained models |
| 2026-08-15 | **Budget regime adopted**; altitude head made anchor-aware; QoS floors recalibrated |
| 2026-08-16 | Retrained models pushed — later shown invalid |
| 2026-08-17 | **Blind-3D removed**; Two-Stage baseline and Coupled-Greedy ablation added; statistics fixed |
| 2026-08-18 | **20-seed results — all claims settled** |

---

# QUICK REFERENCE — EXPLAIN THIS IN 60 SECONDS

> A drone collects data from 500 city sensors. Some are critical and need a fast
> radio link, which only works if the drone gets within 21 metres.
>
> The existing method flies at one fixed height. That height must clear the
> tallest building — 47 metres — so critical sensors are **physically
> impossible** to serve. It reaches 0.6% of them.
>
> We let the drone choose its height at every stop, setting the altitude from the
> strictest sensor in each group, then refining all altitudes with a local
> search. Given exactly the same battery budget, we serve **70.2%** of critical
> sensors — winning on all 20 cities tested.
>
> We also beat the standard "plan first, fix heights later" approach on every
> class. Our ablation shows the gain comes specifically from the **continuous
> altitude refinement**, not merely from making the two decisions jointly.
>
> The cost is computation: roughly 30 minutes of planning per city. Making that
> fast — via a learned policy — is the main future work.

---

## Where things live

| What | Where |
|---|---|
| Figure specs and narrative | `paper_figures/PAPER_STORY.md`, `paper_figures/FIGURE_LIST.md` |
| Locked formulation | `docs/PROBLEM_FORMULATION.md` |
| All parameters | `atom_3d/configs/params.yaml` |
| Planners | `experiments/two_stage_vs_coupled.py`, `experiments/strong_coupled.py` |
| Result generation | `atom_3d/experiments/export_figure_data.py` |
| Numbers behind every figure | `paper_figures/results_data/` |
| What produced each result | `paper_figures/results_data/labels.json` |
