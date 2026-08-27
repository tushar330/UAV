# BTP Viva & Presentation Study Guide
### *Criticality-Aware Three-Dimensional UAV Trajectory Planning for Energy-Budgeted IoT Data Collection*

**How to use this document.** Read Part 1 (prerequisites) once slowly — it contains every piece of background knowledge the report assumes you already have. Then Parts 2–5 walk the report chapter by chapter, in order, each topic prefaced by exactly the background needed for it. Part 6 explains every figure. Part 7 is the number sheet you memorise. Part 8 is derivations you may be asked to do on the board. Part 9 is ~90 viva questions with answers. Part 10 is your presentation plan.

---

# PART 0 — THE PROJECT IN 60 SECONDS

Learn this by heart. It is the answer to "So, tell us what you did."

> A drone flies over a city collecting data from 500 ground sensors. Standard planners in the literature fix one cruise altitude for the whole flight and only optimise the *order* of visits. We show that fixing altitude is not a harmless simplification — it is the *binding constraint*. A single cruise altitude must clear the tallest building, which on our map is about **47 m**. But our most important sensors demand 38 Mbps, and inverting the Shannon capacity formula shows 38 Mbps is only achievable from within **21 m**. 47 > 21, so a fixed-altitude planner **physically cannot** serve a critical sensor — no amount of better routing fixes it. So we made altitude a *per-hover decision variable*, with the per-class rate requirement entering as a **hard geometric constraint**, and we added a continuous local-search refinement of those altitudes. Holding **mission energy fixed** at 82.65 kJ so that energy is a control rather than an outcome, critical-class satisfaction rises from **0.6% → 70.2%** across 20 independently generated city layouts, winning on all 20. Against a *strong* decoupled 3D baseline it still wins on all three priority classes at statistically indistinguishable energy. An ablation showed — against our expectation — that the **joint formulation alone contributes nothing** on the critical class (54.9% for both), so the entire gain is attributable to the **continuous altitude refinement**, which costs about 30 minutes of computation per layout. We report that honestly as a limit on what we can claim.

**The three sentences you must never get wrong:**
1. The failure of the 2D baseline is **geometric, not computational**.
2. Energy is a **control variable**, not a result — we make **no energy-saving claim anywhere**.
3. Our demonstrated mechanism is **altitude refinement under quality constraints**, *not* joint decision-making as such.

---

# PART 1 — PREREQUISITES

Everything below is background the report assumes. If a topic is already familiar, skim it — but read §1.3, §1.4 and §1.8 properly, because that is where 70% of viva questions land.

---

## 1.1 What a UAV data-collection mission is

**The setting.** You have hundreds of small battery-powered sensors spread across a city — vibration sensors on bridges, air-quality sensors, water meters, sensors on hospital generators. Each has a few hundred kilobytes to a couple of megabytes of stored data.

**The problem.** How does that data get out?

- *Option A — fixed backhaul.* Run a wire or a cellular modem to every sensor. Economically absurd for 500 devices.
- *Option B — long-range radio.* Each sensor talks directly to a distant base station. Radio power needed to reach a far receiver scales badly with distance; a sensor doing this drains a battery in weeks, not years.
- *Option C — bring the receiver to them.* A drone flies out from a depot, hovers near each sensor, collects at short range, and returns. Now each sensor transmits over a **short** link at **low** power.

Option C is what this project studies. The key economic move: **the burden of covering distance is transferred from 500 sensor batteries onto one aircraft battery.**

**Why it is an optimisation problem.** That one aircraft battery is small. If it were unlimited you would simply visit all 500 sensors one after another and go home — no research needed. Because it is limited, you must choose *where to stop*, *how high to stop*, and *in what order*, and you will not be able to serve everyone. That is the whole game.

**Rotary-wing vs fixed-wing.** A fixed-wing aircraft (like a small plane) must keep moving to stay up — it *cannot hover*. A rotary-wing aircraft (a quadcopter/helicopter) can stop in mid-air. We use rotary-wing because collecting data requires **staying put over a sensor for several seconds**. This choice has a crucial consequence covered in §1.6.

---

## 1.2 Wireless sensor networks & why sensors are not interchangeable

A **Wireless Sensor Network (WSN)** is a set of spatially distributed autonomous devices that sense and report. **IoT (Internet of Things)** is the broader term for networked physical devices.

The report's first motivating claim: real deployments are **heterogeneous in importance**.

> A vibration sensor on a hospital's backup generator and a soil-moisture probe in a municipal park may be *physically identical*, hold *the same number of megabytes*, and be *indistinguishable to the optimiser* — but the consequence of failing to collect from them differs by orders of magnitude.

This is why we introduce **criticality classes**. And it exposes a flaw in the standard objective, *"minimise energy subject to serving every device once"*:

- That constraint is satisfied the moment the bytes have moved. It says nothing about **how well the link performed while they moved**.
- A device served over a marginal link is served *slowly*, occupies the aircraft *longer*, and in a budget-limited mission may never be served at all.
- A planner with no notion of relative importance **has no basis for deciding which devices to sacrifice** when the budget runs short — so it sacrifices them arbitrarily.

---

## 1.3 Wireless communication fundamentals ⭐ (most examinable section)

### 1.3.1 Decibels and dBm

A decibel is a logarithmic ratio. **dBm** is absolute power referenced to 1 milliwatt:

$$P_{\text{dBm}} = 10\log_{10}\!\left(\frac{P}{1\,\text{mW}}\right)$$

So 0 dBm = 1 mW, 30 dBm = 1 W, and **−110 dBm = 10⁻¹⁴ W = 10 femtowatts** — our noise power. Engineers use dB because signals span 15 orders of magnitude and because multiplication becomes addition.

### 1.3.2 Noise

Every receiver has an unavoidable random electrical fluctuation from thermal motion of electrons, called **thermal noise**. Its power is $\sigma^2 = k T B$ where $k$ is Boltzmann's constant, $T$ temperature, $B$ bandwidth. It is the **floor**: a signal weaker than the noise is unrecoverable. Our value is **σ² = −110 dBm**.

### 1.3.3 SNR — Signal-to-Noise Ratio

$$\text{SNR} = \frac{\text{received signal power}}{\text{noise power}}$$

The single most important quantity in a radio link. High SNR → you can pack many bits per symbol. Low SNR → you must use a robust, slow modulation.

### 1.3.4 Path loss — why distance hurts

A transmitted signal spreads out and weakens. The received power is modelled as

$$P_{\text{rx}} = \frac{P_T \,\beta}{d^{\alpha}}$$

- $P_T$ = transmit power
- $\beta$ = **reference channel gain** — the gain at a reference distance of 1 m, absorbing antenna gains, wavelength and hardware constants
- $d$ = distance between transmitter and receiver
- $\alpha$ = **path-loss exponent**

**The path-loss exponent α is the parameter to understand.**

| α | Environment | Meaning |
|---|---|---|
| 2 | Free space / vacuum | Inverse-square law; energy spreads over a sphere of area 4πd² |
| 2.7–3.5 | **Urban with shadowing** | Signal also scatters, reflects, is absorbed by concrete |
| 4–6 | Dense indoor / heavy obstruction | Very rapid decay |

**We use α = 3.** So doubling the distance divides received power by 2³ = **8**. The report phrases it as: *"separation is punished cubically."*

> **Viva trap:** "Why α = 3 and not 2?" — Because we model an *urban* environment with shadowing, not free space. Free space (α=2) would be optimistic and would *inflate* our service radii, making the 2D baseline look better than it is. Our choice is the conservative, standard urban value.

### 1.3.5 Shannon capacity ⭐⭐

Claude Shannon (1948) proved that the maximum error-free data rate over a noisy channel is

$$R = B\log_2\!\left(1 + \text{SNR}\right)$$

- $R$ in bits per second
- $B$ = **bandwidth** in Hz — the width of the slice of spectrum you occupy. Ours is **B = 2 MHz**.
- SNR is a **linear ratio** here, not dB.

**Intuition.** $B$ = how many independent symbols per second you may send. $\log_2(1+\text{SNR})$ = how many bits each symbol can safely carry. More bandwidth is a linear gain; more power is only a **logarithmic** gain. This asymmetry is why you cannot solve a weak link by cranking up transmit power — you would need to *multiply* power by 2× just to add **one bit per symbol**.

**Our channel equation (Eq. 3.1):**

$$R_i = B\log_2\!\left(1 + \frac{P_T\beta}{d_i^{\alpha}\sigma^2}\right)$$

This is just Shannon's law with the path-loss model substituted for the SNR. $d_i$ is the **three-dimensional** separation between aircraft and sensor $i$ — this is the crucial detail, because altitude is part of $d_i$.

$$d_i = \sqrt{\underbrace{(x_u-x_i)^2 + (y_u-y_i)^2}_{\rho_i^2 \;=\; \text{horizontal}} + \underbrace{(H-z_i)^2}_{\text{vertical}}}$$

Even if the drone is directly overhead ($\rho_i = 0$), the distance is still $H - z_i$. **You cannot escape altitude.** That single observation is the seed of the entire thesis.

### 1.3.6 Line of Sight (LoS)

A **LoS link** is one where the direct ray from transmitter to receiver is unobstructed. Our model assumes LoS with an urban exponent — i.e. we model urban *attenuation* but not hard *blockage* by individual buildings. This is a stated limitation (see §9, "Threats to validity" questions).

---

## 1.4 Coverage geometry — antennas, beamwidth, footprints ⭐

### 1.4.1 Beamwidth and the coverage cone

A real antenna does not radiate equally in all directions. A directional downward-facing antenna radiates within a **cone**. The **half-beamwidth θ** is the half-angle of that cone measured from the vertical (the boresight). Ours is **θ = 60°**.

Where the cone meets the ground it traces a circle — the **footprint**. Simple trigonometry, with the drone at height $H$ above a sensor at elevation $z_i$:

$$\text{footprint radius} = (H - z_i)\tan\theta$$

So sensor $i$, at horizontal distance $\rho_i$, is *inside the footprint* when

$$\rho_i \le (H - z_i)\tan\theta \qquad \textbf{(Eq. 3.3)}$$

With θ = 60°, tan 60° = √3 ≈ 1.732. At 30 m above a ground sensor, the footprint radius is ≈ 52 m.

**Key property: the footprint grows *linearly* with height.** Climb higher → see more sensors → fewer stops needed → less horizontal flying.

### 1.4.2 The multi-sensor reinterpretation ⭐ (an honesty point the examiners will probe)

The report is explicit and scrupulous about this, and you must be too.

Equation 3.3 is borrowed algebraically from **Liu et al. [2]**. But in *their* paper, θ is the maximum **radar detection angle**, and the constraint is gated by a binary scheduling variable — at any instant their platform is either sensing **exactly one** node or uploading. So for them Eq. 3.3 is a **pointing constraint on a single target**. The words *footprint*, *beamwidth* and *coverage cone* **do not appear in that work at all**.

**We reinterpret the same geometry as a communication footprint within which one hover may serve *several* sensors simultaneously.**

Why this reinterpretation is load-bearing rather than cosmetic:

> If a hover served only one sensor at a time — as in both [1] and [2] — then **climbing buys you nothing**, and the optimal altitude would trivially be the lowest one permitted. There would be no trade-off, no research problem, and no thesis. **Only when a single hover can cover a group does height buy anything.**

Say that sentence in the viva. It shows you understand *why* your model is constructed as it is.

---

## 1.5 Optimisation and algorithms background

### 1.5.1 TSP and VRP

**Travelling Salesman Problem (TSP):** given a set of points, find the shortest closed tour visiting each exactly once. **NP-hard** — no known algorithm solves it in polynomial time, and the number of tours over $n$ points is $(n-1)!/2$, astronomically large for $n$ = 500.

**Vehicle Routing Problem (VRP):** the generalisation with multiple vehicles and capacity constraints.

Our problem is *TSP-like*: after we decide the hover locations, we must order them. But it is **harder than TSP**, because we also decide *what the stops are* and *how high each one is*.

### 1.5.2 Construction heuristics — nearest neighbour

Since exact TSP is intractable, use a **heuristic**. **Nearest-neighbour construction**: start at the depot, repeatedly fly to the closest unvisited point, return at the end. Fast ($O(n^2)$), typically ~25% worse than optimal. Its weakness: greedy early choices strand isolated points, forcing a long final leg.

### 1.5.3 Local improvement — 2-opt (Croes, 1958 [8])

**2-opt** repairs a tour by removing two edges and reconnecting the two resulting paths the other way — which un-crosses crossings. Repeat until no improving swap exists. The result is a **2-optimal** tour: a *local* optimum with respect to that move set, not a global one.

*Nearest-neighbour + 2-opt* is the standard, respectable, non-learned routing pipeline. We use it in **every** method, including all baselines — deliberately, so that routing quality is **not** a confound. Any difference we measure must come from altitude and placement, not from one method having a better router.

> **Excellent viva answer:** "We used the *same* router in every method so routing is controlled out. If we had used a fancy learned router for ours and nearest-neighbour for the baseline, our result would be uninterpretable."

### 1.5.4 Greedy set cover

**Set cover problem:** given a universe of elements and a family of sets, choose the fewest sets whose union is the universe. NP-hard. The **greedy algorithm** repeatedly picks the set covering the most currently-uncovered elements; it achieves a provable $\ln n$ approximation ratio — famously the best possible unless P = NP.

Our clustering is a **constrained** greedy cover: each "set" is a hover, the "elements" are sensors, and the twist is that admitting an element **changes the cost of the set** (because a demanding sensor forces the hover lower, which is more expensive).

### 1.5.5 Local search (and why it is Stage 3)

**Local search:** start from a feasible solution; repeatedly apply small modifications ("moves"); accept a move if it improves your objective; stop when no improving move exists. You end at a **local optimum**.

Our three moves:
- **Relocate** — move one sensor from cluster A to cluster B
- **Merge** — combine two clusters into one
- **Split** — divide one cluster into two

**Why it is needed:** greedy construction is **myopic**. Each cluster is frozen the instant it is formed. The constructor cannot foresee that admitting a critical sensor to a *later* cluster would have been cheaper than forcing an *early* cluster down low. Local search is what **undoes** those decisions. The report's measured gap (54.9% → 70.2%) *is* the value of undoing them.

### 1.5.6 Hard constraints vs soft penalties ⭐

This distinction is a genuine contribution and gets asked about.

- **Soft penalty:** add a term to the objective, e.g. `minimise energy + λ·(rate shortfall)`. The optimiser is now *permitted to trade*. It will happily under-serve one critical sensor if that lets it serve enough unimportant ones — **precisely the behaviour a criticality requirement exists to prevent.**
- **Hard constraint:** a solution violating it is simply **not admitted** into the search at all. No amount of gain elsewhere can buy a violation.

Our rate floors are **hard**. A candidate whose inclusion would make the cluster altitude infeasible is *never* considered. That is why we can say "served" means genuinely at or above the floor, with no fudge.

---

## 1.6 Energy models for rotary-wing aircraft ⭐

### 1.6.1 The naive model, and why the literature abandoned it

Early trajectory papers modelled propulsion as *constant power × time*. **Zeng and Zhang (2017) [3]** showed that for fixed-wing platforms the speed–power relationship is **non-monotonic**, and trajectories optimised under a constant-power assumption can be substantially suboptimal once a realistic curve is used.

### 1.6.2 The rotary-wing power curve — Zeng, Xu & Zhang (2019) [4]

$$P(V) = \underbrace{P_0\left(1 + \frac{3V^2}{U_{\text{tip}}^2}\right)}_{\text{blade profile}} + \underbrace{P_i\left(\sqrt{1+\frac{V^4}{4v_0^4}} - \frac{V^2}{2v_0^2}\right)^{1/2}}_{\text{induced}} + \underbrace{\tfrac12 d_0\rho s A V^3}_{\text{parasite}} \qquad \textbf{(Eq. 2.3)}$$

You do **not** need to derive this. You need to know what the three terms *are* and, above all, the one property that matters:

| Term | Physical meaning | Behaviour with speed V |
|---|---|---|
| **Blade profile** | Power to spin the rotor blades against their own air drag | Grows mildly with V |
| **Induced** | Power to push air downward to generate lift | **Large at V=0, falls as you move** (forward flight lets the rotor bite undisturbed air) |
| **Parasite** | Power to push the fuselage through air | Grows as V³ — dominates at high speed |

### 1.6.3 ⭐ The single most important fact in the energy model

$$P(0) = P_0 + P_i \text{ is the } \textbf{MAXIMUM} \text{ of the curve, not the minimum.}$$

**A rotary-wing aircraft burns more power standing still than cruising at its efficient speed.** Our hover power is **168.5 W**.

Why: hovering, the rotor repeatedly beats the same column of already-downwash air (high induced power). Moving forward, it continuously meets fresh undisturbed air — cheaper lift. The curve is U-shaped, minimum at some intermediate speed, then rising as V³.

**Why this decides the whole thesis** (memorise this argument):

> If hovering were *cheap*, the drone could park at a convenient high altitude, tolerate a weak slow link, and simply **wait** for the data. Altitude would not matter. Because hovering is the **most expensive state available**, waiting is costly, and a weak link becomes expensive **in energy**, not merely slow. That is what forces you to descend, and therefore what makes altitude a real decision.

> **Viva trap:** The base framework AUTO [1] charges flight at $P_F$ = 75 W and hovering at $P_H$ = 50 W — i.e. it makes hovering **cheaper** than flying. **That ordering is physically wrong for a rotorcraft.** We deliberately replaced it with the Zeng model. Know this: it is a specific, defensible technical criticism of the paper we extend.

### 1.6.4 Vertical motion — the climb/descent asymmetry

$$E_{\text{vert}} = \frac{mg}{\eta}\Delta h_{\text{climb}} + c_d\,\Delta h_{\text{descent}}, \qquad c_d \ll \frac{mg}{\eta} \qquad \textbf{(Eq. 2.4)}$$

- **Climbing** does work against gravity: $mgh$, divided by motor efficiency $\eta$ (we use 0.5, so you pay double).
- **Descending** recovers almost none of it. A rotorcraft cannot regeneratively harvest potential energy the way a falling weight on a generator could — it must still spin its rotors to descend in a controlled way. We use $c_d$ = 4 J/m.

With m = 2 kg, g ≈ 9.81, η = 0.5: climbing costs ≈ **39.2 J/m**, descending ≈ **4 J/m** — roughly a **10:1 asymmetry**.

**Consequence:** *every descent is eventually paid for by a climb.* A planner cannot descend casually. This is exactly why our method does **not** simply "fly low all the time" — it descends **only where the constraint requires** and climbs back to efficient cruise between service events (visible in Fig. 3.2).

### 1.6.5 Our total energy model

$$E_{\text{total}} = E_{\text{horiz}} + E_{\text{vert}} + E_{\text{hover}} \qquad \textbf{(Eq. 3.4)}$$

$$E_{\text{horiz}} = \sum_{\ell} L_{\ell}\cdot \min_V \frac{P(V)}{V} \qquad \textbf{(Eq. 3.5)}$$

Read Eq. 3.5 carefully: for each leg of length $L_\ell$ we fly at the speed minimising **energy per unit distance** $P(V)/V$ — *not* the speed minimising power. Those are different speeds. Minimising power alone would say "fly as slowly as possible", but a slow flight takes long and thus burns more total joules over a fixed distance. The right quantity is **joules per metre**.

Hovering is charged at the **peak** power 168.5 W for the duration of each collection.

---

## 1.7 Statistics you need ⭐

### 1.7.1 Why 20 layouts and not 1

A single city layout could, by chance, favour one method. Generating **20 independent layouts** (random seeds **42 to 61**) and testing on all of them shows the result is not an artefact of one arrangement of sensors.

### 1.7.2 Paired design ⭐

**Every method is run on the identical 20 layouts.** This makes the comparison **paired**: for each layout we compute the *difference* between two methods on that same layout.

Why paired is far stronger: layouts differ enormously from each other (some are easy, some hard). That between-layout variance is huge and would swamp the effect if we compared two independent groups. Pairing **cancels it out** — we analyse the 20 differences directly.

### 1.7.3 The paired t-test [22]

Take the 20 differences $d_1,\dots,d_{20}$. Null hypothesis: their true mean is 0. Compute

$$t = \frac{\bar d}{s_d/\sqrt{n}}$$

Large |t| → the observed mean difference is large relative to its variability → unlikely under the null. It **assumes the differences are approximately normally distributed.**

*Historical note (charming and safe to mention):* "Student" [22] was the pen name of William Gosset, a chemist at Guinness brewery, who published anonymously because Guinness forbade employee publication.

### 1.7.4 The Wilcoxon signed-rank test [23]

A **non-parametric** alternative. It ranks the absolute differences, attaches the signs, and sums. It makes **no normality assumption** and is **not distorted by one extreme layout**.

**Why we report both.** With n = 20 we cannot verify normality confidently. Reporting both is honest: agreement means the conclusion is robust to the distributional assumption. And critically —

> *"Where they disagree, the disagreement is stated rather than the more favourable figure selected."*

Quote that line. It is a statement of research integrity and examiners like it.

### 1.7.5 p-value

The probability of observing a difference **at least this extreme** *if the null hypothesis (no real difference) were true*. Small p → the data are unlikely under "no effect". Conventional threshold α = 0.05.

**What a p-value is NOT** (a classic viva trap): it is **not** the probability that the null hypothesis is true, and **not** the probability your result is correct. It is a statement about the data given the null, not about the null given the data.

### 1.7.6 Confidence interval

A 95% CI is constructed so that, over repeated experiments, 95% of such intervals would contain the true mean. We use the Student-t multiplier appropriate to n = 20. Our tables report **mean ± 95% CI**. Non-overlapping CIs strongly suggest a real difference; overlapping CIs do *not* by themselves prove no difference — which is why we run the paired tests.

---

## 1.8 Machine-learning background (Chapter 2 + Future Work)

You did not use a learned policy in the final results, but Chapter 2 reviews it, Chapter 5 proposes recovering it, and **examiners will ask** — especially because "attention-based" is in the title of the base paper.

### 1.8.1 MDP

A **Markov Decision Process** is (States, Actions, Transitions, Reward, discount). "Markov" = the next state depends only on the current state and action, not on the full history. A **policy** maps states to actions; RL learns a policy maximising expected cumulative reward.

### 1.8.2 CMDP — Constrained MDP (Altman [15])

Maximise reward **subject to bounds on auxiliary cost signals**. Standard solution: form a **Lagrangian** $\mathcal{L} = R - \sum_c \lambda_c(C_c - \text{limit})$, and update the multipliers $\lambda_c$ by **dual ascent** — raise $\lambda_c$ when constraint $c$ is being violated, lower it when it is slack. **Achiam et al. [16]** (Constrained Policy Optimization) gives a trust-region method with approximate guarantees *during* training, not merely at convergence.

**Our project initially adopted a CMDP with one dual variable per criticality class.** The final report is a deterministic planner. Be ready to explain the pivot honestly (see Q&A §9.7).

### 1.8.3 Pointer networks and attention

- **Pointer Networks (Vinyals et al. [9], 2015):** a neural architecture that outputs a **permutation over its own variable-length input** — instead of picking from a fixed vocabulary, it "points" at input positions. Exactly what TSP needs.
- **Bello et al. [10]:** trained such a model with **policy gradients** and a learned baseline — no supervised optimal tours required (which is the point, since generating optimal TSP labels is itself intractable).
- **Transformer (Vaswani et al. [12], 2017):** the self-attention architecture; every element attends to every other, giving a permutation-equivariant encoder well suited to unordered point sets.
- **Kool, van Hoof & Welling [11] (2019):** replaced the recurrent encoder with a transformer and introduced a **rollout baseline** (compare against a greedy rollout of the current best policy) — competitive with strong classical heuristics. This is the architecture the base framework [1] adopts.
- **REINFORCE (Williams [14], 1992):** the foundational policy-gradient method, $\nabla J = \mathbb{E}[\nabla \log \pi(a|s)\,(R - b)]$, where $b$ is a **baseline** subtracted to reduce variance without introducing bias.

### 1.8.4 ⭐ The architectural insight in §2.4 — say this one in the viva

> *All* of these models emit a **permutation over discrete items**. Extending such a decoder to also emit a **continuous** quantity such as altitude is **not merely a matter of adding an output head**: the continuous head must be **conditioned on the discrete selection**, because the appropriate altitude depends on *which sensor has just been chosen*.

This is a genuine technical observation and it also explains our own training failure: in our first CMDP run the altitude head could not see the chosen anchor, so it collapsed to a constant. That is the *same* bug the paragraph warns about — which makes it a very strong thing to volunteer.

### 1.8.5 Age of Information (Kaul, Yates & Gruteser [17])

**AoI** measures **staleness**: the time elapsed since the most recently received update was generated. It is *complementary* to a rate requirement — rate asks "how fast did the bytes move?", AoI asks "how old is my freshest information?". Relevant when data arrives **continuously** rather than as one stored batch. We identify it as future work.

---

# PART 2 — CHAPTER 1: INTRODUCTION

## 2.1 The base framework — AUTO (Dong, Jiang & Peng [1], IEEE TIE 2025)

*Prerequisite: §1.3 (Shannon), §1.6 (energy), §1.8.3 (attention).*

**What AUTO does.** A **fleet** of aircraft first **charges** ground devices by **Wireless Power Transfer (WPT)**, then collects their data, **minimising total energy $\sum_j E_j$** subject to **serving every device exactly once**.

**Their energy harvesting model** is linear: $E^R_{ij} = \eta_L P^R_{ij} T^E_{ij}$ — harvested energy = efficiency × received power × charging time.

**Their data constraint (Eq. 2.1):**

$$T^C_{ij}\,B\log_2\!\left(1 + \frac{|g^U_{ij}|^2 E^R_{ij}}{\sigma^2 T^C_{ij}}\right) \ge D_i$$

Read it as: (collection time) × (achievable rate) ≥ (data volume). The device transmits using the energy it just harvested, spread over the collection time.

**Their energy accounting (Eq. 2.2):**

$$E_j = \underbrace{P^F T^F_j}_{E^F_j \text{ flight}} + \underbrace{(P^T + P^H)T^T_j}_{E^T_j \text{ transfer}} + \underbrace{(P^C + P^H)T^C_j}_{E^C_j \text{ collection}}$$

**How they solve it.** Decomposed into (a) a **convex time-allocation subproblem** solved in closed form via the **Karush–Kuhn–Tucker (KKT) conditions**, with a **Lambert-W** expression for the optimal collection time; and (b) a **trajectory subproblem** solved by a **graph-attention encoder–decoder** trained with an **actor–critic** method using the realised system reward as its baseline.

*(KKT = the first-order necessary conditions for optimality in constrained optimisation — the generalisation of Lagrange multipliers to inequality constraints. Lambert-W is the inverse function of $w e^w$, which appears when you must solve equations mixing a variable inside and outside a logarithm.)*

### ⭐ The three properties of AUTO that matter to us

| # | Property of AUTO | Why it matters |
|---|---|---|
| 1 | Flight height is a **constant $H^F$ = 20 m** — a *parameter*, not a decision | This is the assumption we attack |
| 2 | **Each hover serves exactly one device, from directly overhead** | No footprint, no multi-sensor coverage, so no altitude trade-off exists |
| 3 | Energy model charges $P^F$ = 75 W flight, $P^H$ = 50 W hover — **hovering cheaper than flying** | Physically the wrong ordering for a rotorcraft (§1.6.3) |

The report says of property 1: *"Where terrain is genuinely flat and every device has the same requirements, this costs little. The subject of this report is what it costs when neither condition holds."* That sentence is a perfect one-line framing of the whole BTP.

## 2.2 Motivation — the two observations and how they interact

**Observation 1 (§1.2.1): sensors are not interchangeable.** See §1.2 above.

**Observation 2 (§1.2.2): altitude is the lever that priority requires.**

The logic chain — walk the panel through it exactly like this:

1. Achievable rate falls with aircraft–sensor distance, and in an urban environment (α=3) it falls **steeply**.
2. Therefore a device with a demanding rate requirement **can only be served from close range**.
3. The only way to be at close range to a device *on the ground* is to **descend**.
4. But the number of devices visible from a single hover **grows** with height, because the footprint widens as you climb.
5. So **height trades coverage against link quality**, and the right balance **depends on which devices you are serving** — hence it differs at every hover.
6. A fixed-altitude planner cannot express that trade-off at all.
7. Worse, the altitude it must adopt is **not even a free choice**: it must clear the tallest structure with a safety margin → **≈ 47 m** on this map.
8. And the most demanding class can only be served **from within ≈ 21 m**.
9. **47 > 21.** The two figures are irreconcilable. The planar planner is not merely *inefficient* on those devices; **it cannot serve them at all, and no improvement to its routing will alter that.**

**The closing line of the motivation** — worth memorising verbatim:

> *"Priority is worth introducing only if the planner possesses a mechanism capable of acting on it, and here that mechanism is altitude."*

This answers the question "why is priority and altitude one paper rather than two?" — because priority *without* an altitude lever is decorative.

## 2.3 Problem statement

> A **single rotary-wing aircraft** departs from a depot and must gather data from **N stationary sensors** whose positions **and elevations** are known. Each sensor belongs to one of **three criticality classes**, and each class carries a **minimum data rate** the link must achieve for a collection to count as successful. The aircraft chooses hover positions in **three-dimensional space**, subject to a **minimum clearance** above any sensor it serves and to **altitude bounds**, and is constrained by a **fixed mission energy budget**. The objective is to **maximise service quality delivered within that budget**, with sensors weighted by criticality.

**Two changes from the formulation we extend:**
1. **Altitude is a decision variable at every hover**, not a mission-wide constant.
2. The mission operates under an **energy budget** rather than a requirement to serve every sensor.

Change 2 is subtle and is defended in §3.2 (see Part 4 below) — it is the single cleverest methodological move in the report.

## 2.4 The four contributions (know these as a numbered list)

1. **A three-dimensional extension** in which hover placement and altitude are decided **jointly**, with per-sensor rate floors entering as **hard feasibility constraints** rather than penalty terms.
2. **An energy-budgeted evaluation protocol** under which all methods expend approximately equal energy, so energy is an **experimental control** and delivered quality is the quantity compared. *"This is what makes the phrase 'better quality at comparable energy' measurable rather than rhetorical."*
3. **An evaluation over twenty independently generated layouts** using **paired statistical tests**, reporting both parametric and rank-based results, against a **strong decoupled 3D baseline** as well as the planar starting point.
4. **An ablation** establishing that joint decision-making **alone does not account for the improvement**, attributing the gain specifically to the **continuous altitude refinement**. *"This constrains what the work may claim, and is reported in those terms."*

> Contribution 4 is a *negative* result about our own method. Presenting it as a contribution is the mark of honest research — lead with it rather than being caught by it.

---

# PART 3 — CHAPTER 2: LITERATURE SURVEY

*All prerequisites already covered in §1.6, §1.4, §1.5, §1.8.*

**The structure and why each thread is there:**

| Section | Thread | Why the report needs it |
|---|---|---|
| 2.1 | The base framework [1] | What we extend; supplies the fixed-altitude assumption we attack |
| 2.2 | Rotary-wing energy models [3,4] | Establishes **hovering is expensive**, which is what makes a weak link *costly* rather than merely slow |
| 2.3 | Altitude & aerial coverage [5,6,7,2] | Establishes the coverage-vs-link-quality trade-off and supplies the geometry (Eq. 3.3) |
| 2.4 | Learned routing solvers [8–14] | The alternative to hand-designed heuristics; explains why extending a discrete decoder to continuous altitude is hard |
| 2.5 | QoS & constrained formulations [15–21] | Hard constraints vs weighted objectives; CMDP background; AoI |
| 2.6 | Identified gap | The synthesis |

## 3.1 §2.3 — Altitude optimisation (the aerial base station literature)

**Al-Hourani, Kandeepan & Lardner [5] (2014)** derived an **optimal height for a low-altitude platform** serving a ground area, identifying exactly the trade-off that recurs throughout our work: *greater height enlarges the geometric footprint but lengthens every link, and the opposing effects produce an **interior optimum*** (i.e. the best height is neither the lowest nor the highest — it is somewhere in the middle).

**Mozaffari et al. [6]** extended it to energy-efficient collection; their survey [7] consolidates the placement literature.

### ⭐ The distinction that creates our gap

| Aerial base station placement | Our collection mission |
|---|---|
| A **static** deployment | Visits **many hover positions in sequence** |
| Altitude chosen **once**; platform remains there | **No reason for the altitude to be the same at each stop** |
| The **served set is given** | The served set must itself be **decided** |

> "The published treatments establish that an optimum height exists for a **given** served set, but do not address **decomposing a deployment into served sets and selecting a height for each**. That joint problem is what Chapter 3 formulates."

## 3.2 §2.6 — The identified gap, stated compactly ⭐⭐

Memorise this. It is the most quotable paragraph in the report.

> **The altitude literature** establishes that an optimum height exists for a **fixed served set** but assumes the set is given.
> **The routing literature** decomposes a deployment into **served sets** but assumes altitude is fixed.
> **Whichever is solved first constrains the other**, and the interaction is **not benign**:
> - choosing hover positions **without regard to the rate requirement** produces clusters for which **no feasible altitude exists**;
> - choosing altitudes **without regard to clustering** produces coverage that is either **wasteful or inadequate**.

And then the honesty clause, stated up front in Chapter 2 rather than buried:

> "It should be stated at the outset, since Chapter 4 demonstrates it experimentally, that **joint treatment alone turns out not to be sufficient**; the refinement described in Section 3.4.3 is what delivers the measured improvement."

## 3.3 Table 2.1 — the positioning table

| Line of work | Altitude | Priority | Route |
|---|---|---|---|
| Planar collection [18, 19] | Fixed | No | Optimised |
| Aerial base stations [5, 6] | Optimised, **static** | No | Not applicable |
| Learned routing [11, 13] | Not modelled | No | Learned |
| Constrained policies [15, 16] | Not modelled | Via constraints | Not applicable |
| **This work** | **Per hover** | **Per-class floor** | **Optimised** |

**How to present it:** "Every row supplies **one** necessary component. **No row supplies all three.** We are the first row with entries in all three columns."

---

# PART 4 — CHAPTER 3: SYSTEM MODEL & METHODOLOGY

*Prerequisites: §1.3, §1.4, §1.5, §1.6 — all of them.*

## 4.1 §3.1.1 Deployment

- Square region, side **1000 m**, containing buildings of varying footprint and height
- **N = 500 sensors**
- Sensor $i$ at $(x_i, y_i, z_i)$ with **elevation $z_i \in [0, 50]$ m** — sensors sit both at ground level **and on rooftops**
- Data volume $D_i \sim [0.2, 1.5]$ MB
- A **depot at a fixed corner** is origin and destination

> **Why elevations vary is not decoration.** It is *what causes the clearance requirement to bind*, and it is why a handful of critical sensors are reachable even by the planar baseline (the rooftop ones). Fig. 1.1's caption says exactly this.

**Spatial correlation of criticality:** class assignment is **spatially correlated, not independent** — high-priority sensors are concentrated around a small number of designated sites (hospital, power station, industrial control, commercial, business centre in Fig. 1.1). Justification: *critical infrastructure in a real city is clustered.*

> This matters algorithmically: because critical sensors cluster spatially, it is **possible** to group them into shared low-altitude hovers. If they were uniformly scattered, every cluster would contain one and the whole map would have to be flown low. It also foreshadows the multi-UAV load-balancing problem in future work (an even *geographic* partition need not be an even partition of *criticality*).

## 4.2 §3.1.2–3.1.3 From rate floor to service radius ⭐⭐ (the heart of the report)

**Table 3.1 — the table you must know cold:**

| Class | Share of N | Rate floor $R^{\min}_c$ | Service radius $d^{\max}_c$ |
|---|---|---|---|
| **High** | 10% (≈50) | **38.0 Mbps** | **21 m** |
| **Medium** | 20% (≈100) | **32.5 Mbps** | **40 m** |
| **Low** | 70% (≈350) | **29.0 Mbps** | **60 m** |

**The derivation (Eq. 3.2)** — be ready to do this on a whiteboard:

Start from Shannon: $R_i = B\log_2\!\left(1 + \dfrac{P_T\beta}{d_i^\alpha \sigma^2}\right)$

Demand $R_i \ge R^{\min}_c$. Since the RHS is **strictly decreasing in $d_i$**, this holds **if and only if** $d_i$ is small enough. Invert:

$$R^{\min}_c = B\log_2\!\left(1+\frac{P_T\beta}{d^\alpha\sigma^2}\right)
\;\Longrightarrow\; 2^{R^{\min}_c/B} - 1 = \frac{P_T\beta}{d^\alpha\sigma^2}
\;\Longrightarrow\; d^\alpha = \frac{P_T\beta}{\sigma^2\left(2^{R^{\min}_c/B}-1\right)}$$

$$\boxed{\;d^{\max}_c = \left(\frac{P_T\beta}{\sigma^2\left(2^{R^{\min}_c/B}-1\right)}\right)^{1/\alpha}\;}\qquad \textbf{(Eq. 3.2)}$$

**Why this equation is the pivot of the entire thesis:**

> **Equation 3.2 converts an abstract quality-of-service requirement into a hard geometric constraint.** "38 Mbps" is an abstraction a geometer cannot use. "**Be within 21 metres**" is a *sphere*. Once QoS is a sphere, the whole problem becomes geometry — and geometry is something a planner can reason about, and something you can *see* is violated.

**Sanity-check the numbers yourself** (I verified these; you can reproduce them in the viva):

With B = 2 MHz and α = 3, and letting $K = P_T\beta/\sigma^2$:
- High: $38/2 = 19$, so $2^{19}-1 \approx 5.243\times10^5$; with d = 21 m, $K = 5.243\times10^5 \times 21^3 \approx 4.86\times10^{9}$
- Medium: $32.5/2 = 16.25$, $2^{16.25} \approx 7.80\times10^4$ → $d = (4.86\times10^9/7.80\times10^4)^{1/3} = 39.6 \approx$ **40 m** ✓
- Low: $29/2 = 14.5$, $2^{14.5} \approx 2.32\times10^4$ → $d = (4.86\times10^9/2.32\times10^4)^{1/3} = 59.4 \approx$ **60 m** ✓

**Notice the non-linearity — this is a great point to make.** The rate floors differ by only 31% (29 → 38 Mbps), but the service radii differ by **186%** (60 → 21 m). Because rate is *logarithmic* in SNR while SNR is *cubic* in distance, a **modest** increase in demanded rate causes a **drastic** contraction of the permissible sphere. That non-linearity is precisely why the critical class is so hard to serve and why the effect is dramatic rather than marginal.

## 4.3 §3.1.4 Coverage geometry

$$\rho_i \le (H - z_i)\tan\theta, \qquad \theta = 60° \qquad \textbf{(Eq. 3.3)}$$

Fully explained in §1.4 above, including the honesty point about reinterpreting [2].

## 4.4 §3.1.5 ⭐ The governing tension — the intellectual core

Two equations pull in **opposite** directions:

| | Pulls altitude | Because |
|---|---|---|
| **Eq. 3.3** (coverage) | **UP** ↑ | Higher hover reaches more sensors → fewer stops → less horizontal flight → less energy |
| **Eq. 3.2** (service radius) | **DOWN** ↓ | A demanding sensor must be approached closely or it is not served at all |

> **"The resolution depends on which sensors are being served and differs at every hover, which is the structural reason altitude cannot sensibly be a mission-wide constant."**

That single sentence *is* the thesis. If you can only say one sentence in the viva, say that one.

### ⭐⭐ The 47-vs-21 argument, stated formally

- A fixed-altitude mission must clear the tallest structure by $h_{\text{safe}}$ = 10 m → **≈ 47 m** on this map
- A high-priority sensor **at ground level** requires the aircraft within **21 m**
- **47 > 21** ⟹ **no such sensor can be served by the planar planner, irrespective of its route**

> *"The failure reported in Chapter 4 is therefore a consequence of **geometry** rather than of **search**."*

This is the most important claim in the report. It is not a statistical claim; it is a claim about **feasibility**. You are not saying "our planner is better"; you are saying "**their planner is solving an infeasible problem and does not know it.**"

## 4.5 §3.1.6 Energy model + Table 3.2

Covered in §1.6. The **parameter table you must know**:

| Symbol | Quantity | Value | Source |
|---|---|---|---|
| N | Number of sensors | 500 | — |
| B | Bandwidth | 2 MHz | base framework [1] |
| σ² | Noise power | −110 dBm | base framework [1] |
| α | Path-loss exponent | 3.0 | **this work** |
| θ | Antenna half-beamwidth | 60° | **this work** (geometry from [2]) |
| $h_{\text{safe}}$ | Clearance above a served sensor | 10 m | base framework [1] |
| $H_{\min}, H_{\max}$ | Altitude bounds | 20 m, 150 m | **this work** |
| $P_0+P_i$ | Hover power | 168.5 W | **this work** (Zeng [4]) |
| m, η, $c_d$ | Mass, motor efficiency, descent coefficient | 2 kg, 0.5, 4 J/m | **this work** |
| $\tau_{\max}$ | Time budget per hover | 30 s | — |
| $B_{\text{mission}}$ | **Mission energy budget** | **82.65 kJ** | derived (§4.6) |

> **⭐ Why the sourcing split matters — a question you will get.** Communication and platform values (B, σ², transmit/collection powers, flight speed, clearance) are taken **unchanged from the base framework [1]**, *"so that the planar baseline reproduces its operating point rather than a re-tuned variant."* In other words: **we did not weaken the baseline by re-tuning it.** It runs at the numbers its own authors chose. The propagation model, propulsion model and altitude bounds are ours, and they are ours *because* they are the things the base framework got physically wrong or did not model at all.

**Two deliberate departures from AUTO, and why:**
1. AUTO optimises the **number of aircraft**; we fix **one aircraft**, *so that the altitude effect is not confounded with fleet sizing.* (This is experimental hygiene, not laziness.)
2. AUTO serves each device by its **own hover from directly overhead**; our Eq. 3.3 permits **one hover to serve several devices**, *which is what makes altitude a trade-off rather than a free parameter.*

## 4.6 §3.2 ⭐⭐ The evaluation regime — the cleverest section in the report

This section is placed **before** the planner deliberately: *"because an earlier version of this project produced results that were internally consistent and entirely uninformative."*

### The problem with the natural formulation

Inherit AUTO's setup: **serve every sensor, minimise energy, rate floors as hard constraints.** Now watch it collapse:

1. If the floors are **hard**, then **any feasible plan** serves every sensor at or above its required rate.
2. Therefore every feasible plan attains **100% satisfaction on every class, by construction**.
3. Comparing two feasible plans on satisfaction compares **two numbers that are both exactly one**.
4. **The metric intended to demonstrate the contribution is mathematically incapable of expressing it.**

This is a **degenerate metric**. It is not that the results were bad — they were *uninformative*, which is worse, because they *look* fine.

### The remedy — the energy budget

$$B_{\text{mission}} = 0.65 \times E^{\text{full}}_{2D} = 82.65\ \text{kJ} \qquad \textbf{(Eq. 3.6)}$$

where $E^{\text{full}}_{2D}$ = **127.15 kJ** is the energy the planar baseline needs to cover the *whole* deployment.

**The rule:** every method flies **its own route until the budget is exhausted**, and is scored on the **fraction of each class served at or above its floor**.

**Why 0.65?** It forces genuine scarcity. At 100% the metric degenerates again (everyone finishes). At a very low fraction nobody achieves anything and the comparison is noise. 65% of what a full 2D mission would need makes the budget **binding but not crippling** — the planner must actually choose whom to sacrifice, which is exactly the decision the paper is about. *(Note: 82.65 kJ ÷ 168.5 W ≈ 490 seconds of pure hovering — about 8 minutes of hover-equivalent. That is a genuinely tight budget.)*

**Why compute the budget from a single reference layout?** So it is **constant across experiments** — results obtained at different sample sizes (the 10-layout campaign and the 20-layout campaign) remain directly comparable. Had we recomputed it per layout, the budget would have become a moving target.

### ⭐ The consequences — how Chapter 4 must be read

| Quantity | Role |
|---|---|
| **Energy** | **Independent variable / experimental control** — all methods spend ≈ $B_{\text{mission}}$ |
| **Service quality** | **Dependent variable** — measures *what that budget purchased* |

> *"Accordingly **no energy-saving claim is made anywhere in this report**, and the small residual differences in expenditure reflect the **granularity of hover placement** rather than efficiency, **since a partial hover cannot be flown**."*

Memorise that. It pre-empts the single most likely attack ("you saved 3 kJ, isn't that your real result?" — **No. We claim nothing about energy.**).

## 4.7 §3.3 Baseline methods

### 4.7.1 Planar baseline (the 2D starting point)

1. Compute **one cruise altitude** = maximum structure height + $h_{\text{safe}}$ (→ 46–50 m depending on layout)
2. Place **one hover directly above each sensor** at that altitude
3. Order by **nearest-neighbour + 2-opt** [8]
4. Fly until budget exhausted

This faithfully reproduces the fixed-altitude formulation. It is **not a straw man** — it is the standard approach, run at its own authors' parameters.

### 4.7.2 Decoupled 3D baseline ⭐ (the real opponent)

*"The decoupled baseline represents the natural engineering response and is the method against which the contribution is principally measured."*

**Stage 1:** hovers placed by a **greedy set-cover** procedure that **ignores rate requirements entirely** and seeks only **low-energy coverage** — which drives it towards **high altitudes and wide footprints** (because that minimises the number of stops).

**Stage 2 (the repair pass):** every sensor whose achieved rate falls **below its floor** is **removed from its original hover** and **re-served by additional shallow hovers** placed for that purpose.

**Why this is a fair and strong baseline:** it has **exactly the same altitude freedom** as our method. It can fly anywhere from 20 to 150 m. It differs from us **only** in *resolving placement before altitude*. So any difference we measure isolates **coupling and refinement**, not "3D vs 2D".

**Its structural weakness** — state it in these words:

> *"Its weakness is **structural rather than incidental**: because the first stage is **blind to the rate requirement**, the repair pass must be paid for **from the same fixed budget**, and that expenditure is then **unavailable for coverage**."*

This is the mechanism that produces the striking Table 4.3 result where the decoupled baseline scores **worse on low-priority sensors (52.7%) than even the planar baseline (66.7%)**. It rescues its critical sensors **at the expense of the bulk of the deployment**.

## 4.8 §3.4 The proposed method — four stages ⭐⭐

### Stage 1 — Service radii
Each sensor is assigned $d^{\max}_i$ from Eq. 3.2 according to its class. **These are fixed thereafter.** Cheap, $O(N)$, and it turns every QoS requirement into a sphere before any planning begins.

### Stage 2 — Constrained greedy construction

Clusters are grown greedily. Pick an **uncovered sensor as anchor**, then grow a member set $S$ around it. The **feasible altitude** for a candidate member set is determined **by its most demanding member**:

$$H(S) = \min_{i\in S}\left(z_i + \sqrt{(d^{\max}_i)^2 - \rho_i^2}\right) \qquad \textbf{(Eq. 3.7)}$$

**Where Eq. 3.7 comes from** — derive it in one line and you will impress:

Sensor $i$ is served iff its 3D distance is within its radius:
$$\rho_i^2 + (H - z_i)^2 \le (d^{\max}_i)^2 \;\Longrightarrow\; (H-z_i)^2 \le (d^{\max}_i)^2 - \rho_i^2 \;\Longrightarrow\; H \le z_i + \sqrt{(d^{\max}_i)^2 - \rho_i^2}$$

Each member imposes a **ceiling** on the hover altitude. To satisfy **all** members you must respect the **lowest ceiling** — hence the **min**. And if $\rho_i > d^{\max}_i$ the square root is imaginary: **that sensor cannot be served from this hover at any altitude**, so the candidate is **not admitted**.

**Subject to:** $\max(H_{\min},\, z_a + h_{\text{safe}}) \le H(S) \le H_{\max}$, where $a$ is the anchor. (Lower bound: never below the global floor, and never within the safety clearance of the anchor.)

### ⭐ The emergent behaviour of Eq. 3.7 — the best single insight to volunteer

> *"Equation 3.7 expresses the central mechanism: **the altitude of a hover is dictated by its strictest member.** Admitting a high-priority sensor forces the **entire cluster** to be served from low altitude, which is **expensive**, so the construction has an **incentive to group critical sensors with one another** rather than distributing them among clusters that would otherwise fly high."*

This is **criticality-aware clustering arising for free from the geometry.** We never wrote a rule saying "put critical sensors together". It **emerges** because a single critical member imposes its 21 m ceiling on everyone in the cluster, so a cluster containing one critical sensor and thirty low-priority ones must fly at 21 m — wasting the cheapness those thirty low-priority sensors could have enjoyed at 60 m. The energy objective therefore *prefers* to segregate them.

### Stage 3 — Continuous altitude refinement ⭐ (the stage that actually delivers)

**The problem it fixes:** *"The greedy construction is **myopic**, since each cluster is fixed as soon as it is formed."*

**What it does:** local search over **cluster membership**, with the altitude of each affected cluster **recomputed continuously by Eq. 3.7 after every candidate modification.**

Three move types:
- **Relocate** a sensor between clusters
- **Merge** two clusters
- **Split** one cluster in two

A move is **accepted when it reduces the total energy estimate**. Loop until no improving move exists.

**"Continuous" means:** altitude is not selected from a discrete menu (e.g. {20, 30, 40, 50 m}). It is a **real number** recomputed exactly from Eq. 3.7 for whatever membership currently exists. Every membership change immediately re-derives an exact new altitude.

**The cost:** ≈ **30 minutes per layout at N = 500**. Stated plainly in the report as a real cost, not omitted.

**The payoff:** the **entire** critical-class advantage (54.9% → 70.2%).

### Stage 4 — Routing and budget expenditure

Clusters ordered by **nearest-neighbour + 2-opt**, then the tour is **flown until the budget is consumed**. Sensors in clusters **not reached before exhaustion are recorded as unserved.**

> **Important for reading Fig. 4.2:** an unserved sensor is *not* a constraint violation. It is a sensor the budget never reached. Every sensor we *do* serve is served at or above its floor, because the floors are hard.

### Algorithm 1 in plain English

```
INPUT: sensors N, class floors, budget B_mission

# Stage 1
for each sensor i:  d_max[i] = Eq(3.2) for the class of i

# Stage 2 — constrained greedy construction
C = {} ; U = all sensors
while U is not empty:
    a = select an anchor from U
    S = grow a member set around a, keeping H(S) feasible   # Eq 3.7
    C = C + {S} ; U = U - S

# Stage 3 — continuous altitude refinement
repeat:
    apply a relocate / merge / split move to C
    recompute affected altitudes by Eq(3.7)
    keep the move if total energy decreases
until no improving move exists

# Stage 4 — routing and expenditure
order clusters by nearest neighbour, improve by 2-opt
fly the tour until B_mission is exhausted
RETURN served set and per-class satisfaction
```

## 4.9 §3.5 Ablation design ⭐

**The variant:** omit Stage 3, use the Stage 2 construction directly. Call it **coupled-greedy**.

**What it isolates:** coupled-greedy **shares** the joint formulation and the rate constraints with the full method and **differs only in the absence of refinement**. So the difference between coupled-greedy and the full method is *exactly* the contribution of Stage 3. And the difference between the decoupled baseline and coupled-greedy is *exactly* the contribution of coupling.

**A precision point the report insists on:** it is *"an ablation of the proposed method, **not an independent baseline**, and is reported as such throughout."* Do not call it a baseline in your presentation — it is our own method with a part removed. Calling it a baseline would misleadingly suggest it is someone else's published method.

**The 2×2 logic:**

| | Joint? | Refine? |
|---|---|---|
| Decoupled baseline | No | No |
| **Coupled-greedy (ablation)** | **Yes** | **No** |
| **Proposed** | **Yes** | **Yes** |

Reading down the *Joint* column with *Refine*=No isolates coupling. Reading across from coupled-greedy to proposed isolates refinement. Clean factorial design.

---

# PART 5 — CHAPTER 4: RESULTS

## 5.1 §4.1 Experimental protocol

- **20 independently generated city layouts**, seeds **42 through 61**
- **Every method evaluated on the identical layouts** → paired comparison, between-layout variance removed
- **Same budget** for all: **82.65 kJ** = 0.65 × planar-full-coverage energy
- Budget computed from a **single reference layout**, hence constant across experiments
- **Two tests reported for every comparison:** paired Student *t* [22] and Wilcoxon signed-rank [23]
- *"Where they disagree, the disagreement is stated rather than the more favourable figure selected."*
- CIs use the Student-*t* multiplier for n = 20

## 5.2 §4.2 Table 4.1 — Overall comparison ⭐ (memorise this table)

| Method | Energy (kJ) | High (%) | Medium (%) | Low (%) |
|---|---|---|---|---|
| Planar baseline | 81.5 ± 0.5 | **0.6 ± 0.9** | 61.0 ± 5.1 | 66.7 ± 4.2 |
| Decoupled 3D | 79.4 ± 1.0 | 54.9 ± 5.7 | 66.8 ± 5.2 | **52.7 ± 5.4** |
| Coupled-greedy (ablation) | 80.1 ± 1.2 | **54.9 ± 5.6** | 71.5 ± 5.1 | 64.9 ± 6.1 |
| **Proposed** | 78.5 ± 1.5 | **70.2 ± 7.3** | **80.8 ± 3.8** | **74.2 ± 6.2** |

**Three things to notice, in this order:**

**(a) The planar baseline serves 0.6% of high-priority sensors.** Not a bug. It is obliged to cruise between **46 and 50 m** depending on layout, while a high-priority sensor requires the aircraft within **21 m**. The class is **unreachable by construction**.

**(b) ⭐⭐ The 0.70% prediction — the most impressive validation in the report.**

The report *predicts the baseline's failure rate from pure geometry, before looking at the data*:

- The baseline hovers **directly above each sensor**, so the link distance is **exactly $H - z_i$**
- Therefore a critical sensor is reachable **only when** $H - z_i \le 21$, i.e. $z_i \gtrsim 26$ m (with H ≈ 47)
- So the only reachable critical sensors are those **mounted on tall enough rooftops that their own elevation brings them into range**
- In **14 of the 20 layouts, no critical sensor is that elevated and satisfaction is exactly zero**; in the remaining 6, one or two rooftop sensors qualify
- **This geometric criterion alone predicts 0.70%. The measured value is 0.60%.** The small remainder is qualifying sensors that the **budget-truncated tour never reached**.

> **Say this in the viva.** It shows the failure is **completely explained by the model**, not by a coding error, not by a badly-tuned baseline, and not by luck. A predicted-vs-observed agreement of 0.70% vs 0.60% with the residual *itself* explained is about as clean as empirical validation gets. If someone accuses you of sabotaging the baseline, this is your answer.

**(c) All four methods spend within a narrow band, 78.5–81.5 kJ.** *"which is by design and is what makes the satisfaction columns interpretable."*

## 5.3 §4.3 Table 4.2 — vs the planar baseline

| Metric | Baseline | Proposed | Difference | *t*-test | Wilcoxon |
|---|---|---|---|---|---|
| High (%) | 0.6 | 70.2 | **+69.6** | < 0.0001 | 0.0001 |
| Medium (%) | 61.0 | 80.8 | **+19.8** | < 0.0001 | 0.0002 |
| Low (%) | 66.7 | 74.2 | +7.5 | 0.0233 | 0.0486 |
| Energy (kJ) | 81.5 | 78.5 | −3.0 | 0.0005 | 0.0003 |

- **High:** superior on **every one of the twenty layouts** (20/20)
- **Medium:** superior on **eighteen of twenty**
- **Low:** wins on **fifteen of twenty**

### ⭐ The two qualifications the report volunteers (learn both — they are your armour)

**Qualification 1 — the low-priority result.**
> "Although the improvement of 7.5 points is nominally significant under both tests, the **Wilcoxon value of 0.0486 lies immediately below the conventional threshold** and the method wins on **fifteen of twenty** layouts rather than the near-unanimous margins seen elsewhere. It is therefore **not advanced as a principal finding**; the defensible statement is that **low-priority service is not degraded**, and that is how the conclusions describe it."

p = 0.0486 is *technically* significant and *practically* fragile. One different layout could flip it. Claiming it as a win would be over-claiming.

**Qualification 2 — the energy difference.**
> "The proposed method expends 3.0 kJ less and the difference is statistically significant, **but this is not an efficiency claim and none is made anywhere in this report.** Under a budgeted regime both methods stop when the budget is exhausted, and both in fact stop **slightly short because a hover cannot be flown in part**. The proposed method uses **larger clusters and therefore coarser increments**, so it stops marginally further short. **Leaving budget unspent is not a virtue**, and the correct reading of the energy column is that the methods are **comparable**, which is what the design was intended to ensure."

**This is the most important defensive answer in the report.** A statistically significant number that you *refuse to claim as a win* — because you understand it is an artefact of quantisation, not efficiency.

## 5.4 §4.4 Table 4.3 — vs the decoupled 3D baseline ⭐ (the real result)

| Metric | Decoupled | Proposed | Difference | *t*-test | Wilcoxon |
|---|---|---|---|---|---|
| High (%) | 54.9 | 70.2 | **+15.3** | 0.0009 | 0.0015 |
| Medium (%) | 66.8 | 80.8 | **+14.1** | 0.0013 | 0.0010 |
| Low (%) | 52.7 | 74.2 | **+21.5** | < 0.0001 | 0.0001 |
| Energy (kJ) | 79.4 | 78.5 | −0.9 | 0.3317 | 0.4980 |

> *"The planar comparison, while emphatic, is against a method that **cannot physically perform the task**. The stronger test is against the decoupled planner, which has the **same altitude freedom** and differs only in **resolving placement before altitude**."*

**Significantly better on all three classes under both tests; energy difference not significant under either.**

> *"This is the form of result the budgeted protocol was designed to produce: **indistinguishable expenditure, and a difference in what that expenditure achieved.**"*

That is the sentence that vindicates §3.2. p = 0.33 and p = 0.50 on energy is the *ideal* outcome — it proves the control worked.

### ⭐ The most interesting finding in the table: 52.7%

The **largest margin is on the LOW-priority class**, where the decoupled baseline reaches only **52.7% — below even the planar baseline's 66.7%.**

**The mechanism, in causal order:**
1. Its first stage places hovers **without regard to the rate requirement**
2. So many sensors are **subsequently found to be served below their floor**
3. Repairing them requires **a second pass of shallow hovers**
4. Paid for from **the same fixed budget**
5. The energy consumed in repair is **unavailable for coverage**
6. The sensors that lose it are **the numerous low-priority ones** (70% of the deployment)

> *"The decoupled planner **recovers its critical sensors, but at the expense of the bulk of the deployment.**"*

This is the empirical proof of the abstract claim in §2.6 that "whichever is solved first constrains the other, **and the interaction is not benign**." Decoupling is not neutral — it actively destroys value.

### The honesty note about the 10-layout campaign

> *"An earlier ten-layout campaign left the critical-class comparison ambiguous, at p = 0.033 under the t-test but **p = 0.063 under Wilcoxon**. Extending to twenty layouts resolved the ambiguity, and the two tests now agree closely on every class."*

**Be ready for:** *"Did you increase n until you got significance?"* — **No, and here is the distinction.** p-hacking is running until you cross a threshold **and then stopping**, testing repeatedly and reporting the favourable stop. What we did is: the tests **disagreed**, which is a diagnostic that the sample was too small to distinguish; we **doubled** the sample **once**, pre-committed, and reported **all** classes and **both** tests from the larger campaign. We also **report the earlier ambiguity in the paper** rather than hiding it — which is the opposite of p-hacking.

## 5.5 §4.5 Table 4.4 — The ablation ⭐⭐ (the intellectual highlight)

> *"The ablation was intended to confirm that the joint formulation was responsible for the improvement. **It established the opposite**, and is reported here because it constrains what this work may claim."*

| Variant | Joint | Refine | High (%) | Low (%) |
|---|---|---|---|---|
| Decoupled baseline | No | No | 54.9 | 52.7 |
| Coupled-greedy | **Yes** | No | 54.9 | 64.9 |
| **Proposed** | **Yes** | **Yes** | **70.2** | **74.2** |

**Finding A — coupling alone does nothing on the critical class.**
- Means agree to the reported precision: **54.9% for both**
- Mean paired difference: **+0.00 percentage points**
- Paired *t*-test: **p = 1.000**
> *"Deciding placement and altitude jointly, **without subsequent refinement, delivers no advantage whatever** on the class the method exists to serve."*

**Finding B — refinement supplies the entire difference.**
- Critical: **54.9 → 70.2**, +15.3 points, **p = 0.0007**
- Medium: **+9.3 points, p = 0.0026**

**Finding C — coupling is not wholly without effect, but its effect is elsewhere.**
- On the **low-priority** class, coupled-greedy improves on decoupled by **+12.2 points (p = 0.0032)**, because it **avoids the wasteful repair pass**
> *"Its contribution is thus to **protect bulk coverage** rather than to **serve critical sensors**."*

### The three consequences the report draws

1. **"The mechanism this work demonstrates is the continuous altitude refinement and not the joint formulation as such;** a claim that coupling outperforms decoupling on critical sensors would be **contradicted by Table 4.4**."
2. **"The computational cost of the refinement is load-bearing rather than incidental"** — 30 min/layout, and it supplies the *entire* critical advantage. **You cannot cheapen the method by dropping the slow stage.**
3. **"The greedy construction is evidently myopic in a way the refinement corrects:** because each cluster is fixed at formation, the constructor **cannot anticipate that admitting a critical sensor to a later cluster would have been cheaper than forcing an early cluster low**, and the measured gap **is** the value of undoing such decisions."

> **How to present this.** Do **not** hide it, do **not** apologise for it. Present it as: *"We ran the ablation to confirm our headline claim. It refuted it. So we changed the claim to match the evidence, and that is now the most useful result in the report — because it tells the next researcher exactly which component to invest in."* An examiner who sees that will trust everything else you say.

## 5.6 §4.6 Behavioural interpretation ⭐

**A precision point stated first:** Fig. 4.4 is a distribution over **hover** altitudes only — it describes **where service is delivered**. The **cruise** behaviour *between* hovers is visible in **Fig. 3.2** and **not** in the distribution. Do not confuse them.

**The proposed method's hover altitudes:**
- **Unimodal**, single mode near **28 m**
- **86% of hovers below 40 m**
- **None whatever above 60 m**
- ⟹ every hover is flown from within or near the band the critical floor requires

**⭐ But it is NOT "just fly low".** *"This is not achieved by descending and remaining low, which the asymmetry of Equation 2.4 would make prohibitive"* — remaining low means climbing back over every building, and climbing costs ~39 J/m against ~4 J/m recovered. Fig. 3.2 shows the aircraft **returning to an efficient cruise between service events.**

On the representative mission: the proposed method flies about **5 m lower during critical service intervals than outside them**, whereas **both 3D baselines fly HIGHER during those intervals than outside them.**

> That contrast is the behavioural signature of the contribution. **It descends where the constraint requires and climbs again afterwards.**

**The decoupled baseline's altitudes:**
- **Genuinely bimodal**: dominant mode near **68 m**, second near **38 m**
- The first is its **rate-blind cover**; the second is its **repair hovers**, created after coverage had already been committed
- **59% of its hovers sit above 60 m**, against **0%** for the proposed method
> *"Its low mode is thus **additional expenditure appended to a settled plan** rather than an **integrated decision**."*

Fig. 4.5 shows the same distinction **geometrically**: our descents occur **within** the tour; the decoupled baseline's are **appended to** it.

## 5.7 §4.7 Threats to validity (know all four — volunteering them is strength)

1. **Single generator.** All layouts come from one generator. Twenty samples establish the result is not an artefact of *one arrangement of sensors*, **but not** that it holds for a different **building density, elevation distribution or map size**.

2. **Simplified propagation.** LoS with an urban exponent, **no blockage** by the buildings the model otherwise contains. ⭐ *"Since blockage would principally affect **long** links, and the method **deliberately shortens** the links to its critical sensors, the omission is **more likely to understate than overstate** the reported advantage; **this expectation has not been verified.**"* — i.e. the simplification is argued to be **conservative** (against us), and even that argument is flagged as unverified.

3. **Rate floor calibration.** Floors were chosen to span the achievable band. Different floors → different radii → different margins. *"The **qualitative** conclusion depends only on the critical radius being **smaller than the mandatory cruise altitude**, which is **robust**, but the **numerical margins** are tied to this calibration."*

4. **Deterministic planners with complete prior knowledge.** No claim regarding partial information, sensor mobility, failures, or disturbances such as wind.

---

# PART 6 — CHAPTER 5: CONCLUSION, LIMITATIONS, FUTURE SCOPE

## 6.1 Summary — the narrative arc

> The project began from a 2D framework holding one cruise altitude for the whole sortie, and asked **what that simplification costs once sensors differ in importance**.
> **The answer: it costs everything on the class that matters most, for a reason that is geometric rather than algorithmic.** 47 m mandatory vs 21 m permitted. *"The planar baseline serves 0.6% of its critical sensors **not because its route is poor but because no route would serve them**."*
> Making altitude a per-hover decision, constrained by per-sensor rate floors and followed by continuous refinement, raises critical satisfaction to **70.2% at the same energy budget, winning on all twenty layouts**. Against a decoupled 3D baseline with the same altitude freedom, significantly better on **all three classes at statistically indistinguishable energy**.
> *"The ablation produced the **least expected and most useful** result."* Joint decision-making alone delivers **no** improvement on the critical class (54.9% both, mean paired difference zero). **The whole of the advantage is attributable to the refinement stage.**
> *"What this work demonstrates is therefore **altitude refinement under quality constraints**, not joint decision-making as such, and the roughly thirty minutes of computation the refinement requires is **a genuine cost of the method rather than an implementation detail**."*

## 6.2 Limitations (six — be able to list all)

| # | Limitation | The honest framing |
|---|---|---|
| 1 | **Computational cost** | ~30 min/layout at N=500. Acceptable for a mission planned in advance; **precludes replanning in flight**. And since refinement supplies the entire critical advantage, **the cost cannot be avoided by omitting the stage.** |
| 2 | **Single environment** | One city generator; robustness across densities, elevation distributions and map scales **untested**. |
| 3 | **Simplified propagation** | LoS + urban exponent; no blockage, shadowing or interference. |
| 4 | **Static deployment** | Sensors neither move, fail nor appear mid-mission; no wind. |
| 5 | **Single aircraft** | Multi-aircraft partitioning **exists in the implementation but is not evaluated**. |
| 6 | **Marginal low-priority result** | Claimed as **non-degradation**, not as a gain. |

## 6.3 Future scope

### Immediate

**(a) Recovering the learned policy.** ⭐ Read this carefully — it is a specific, diagnosed bug report, not vague hand-waving:

> *"The effective learning rate reaching the altitude head **decayed to order 10⁻¹⁰**; correcting the rate and its decay schedule, **updating per batch rather than once per epoch**, and **gating the run on whether the altitude bias has moved measurably by epoch fifty** would address the identified causes."*

**Motivation:** *"a trained policy would produce a plan in **milliseconds rather than thirty minutes**, addressing the principal weakness of the present system."* — i.e. learning is proposed **as the fix for limitation 1**, not as novelty for its own sake.

**(b) Accelerating the refinement.** Independently of learning: the local search **re-evaluates cluster energies from scratch after each candidate move**. **Incremental evaluation** (update only what changed), or **restricting moves to spatial neighbourhoods** (don't consider relocating a sensor to a cluster 800 m away), should give a substantial speed-up. *(This is a strong answer to "how would you make it faster?" — it's an O(·) argument, not a "buy a bigger GPU" answer.)*

**(c) Energy–quality frontier.** Sweeping the mission budget and recording resulting satisfaction would characterise the trade-off directly. *"This was attempted, but the available data belonged to the **superseded serve-everything regime** and was **withdrawn rather than reported**."* — another integrity point: data that no longer matched the protocol was **withdrawn**, not repurposed.

**(d) Environmental robustness.** Repeat across generators with differing densities and elevation profiles, to establish whether the margins are characteristic or particular to this configuration.

### Medium and longer term

- **Multiple aircraft.** Raises **load-balancing** questions the single-aircraft formulation never meets, *"since critical sensors are **spatially clustered** and an even **geographic** partition need not be an even partition of **criticality**."* (Nice callback to §3.1.1.)
- **Probabilistic propagation.** Replacing LoS with a **blockage model** makes altitude selection a **decision under uncertainty**. *"This is arguably the extension that would most improve realism, since it introduces a **second reason to climb** that the present model omits."* (In our model climbing only ever buys footprint. With blockage, climbing also buys **LoS probability** — a genuinely different force.)
- **Freshness objectives.** Where data arrives continuously, **AoI [17]** becomes the right measure and the problem acquires a **scheduling** dimension alongside the geometric one.
- **Online replanning and physical validation.** Replanning presupposes the fast planner above; and *"every result in this report is obtained in simulation, so the energy model, **however standard, has not been checked against a real airframe**."*

## 6.4 §5.4 Closing remarks ⭐⭐ — the paragraph to end your presentation on

> *"The most useful lesson of this project concerned **experimental design** rather than trajectory planning. **Two separate versions of the evaluation produced results that were internally consistent and yet carried no information**: the first because **hard quality constraints made the reported satisfaction identically one**, the second because **a floor calibrated to a service radius larger than the map could not be violated**. In both cases **the figures were plausible and the code was correct. What was wrong was the question being asked of the data.**"*

> *"The same pattern recurred in the training work, where **a policy that had not moved from its initialisation nonetheless reported a satisfaction figure that would have looked entirely respectable had the weights not been inspected.** In each instance the error was found by asking **whether the measured quantity was capable of taking a different value**, rather than by asking whether the number looked reasonable. That test proved considerably more informative than inspecting the results themselves, and it is the practice this project would carry forward."*

**The transferable principle, in one sentence you should be able to state:**

> **Before trusting a metric, ask not "is this number reasonable?" but "is this number *capable of being different*?"**

Three failures, one diagnosis:
1. Hard constraints + serve-everything → satisfaction **identically 1** by construction
2. Rate floor whose service radius **exceeded the map** → the constraint **could not be violated**
3. An RL policy **frozen at initialisation** still reporting a respectable score

In all three, the number looked fine and the code was correct. The *question* was degenerate.

---

# PART 7 — EVERY FIGURE EXPLAINED

For each: what is plotted, what to say, and what you will be asked.

---

## Figure 1.1 — Synthetic urban deployment (p. 5)

**What it shows.** A 3D view of the environment: x and y from 0 to 1000 m, altitude axis to ~100 m. Grey blocks are **buildings** of varying footprint and height. Five hundred **sensors** are plotted, **coloured by criticality class** (high / medium / low). Labelled **priority nodes** mark the designated critical sites — Hospital, City Hospital, Power Station, Industrial Control, Business Centre, Commercial, Residential A, Residential B, Downtown. A **Depot** marker sits at a fixed corner. Legend distinguishes Infrastructure, Building, Depot, Point of interest, Road, District.

**What to say when it appears:**
> "This is the environment. Five hundred sensors over a one-kilometre square with buildings of varying height. Two features matter. First, sensors sit **both at ground level and on rooftops**, so their elevations differ — **that is what causes the clearance requirement to bind**, and it is why a handful of critical sensors turn out reachable even by the 2D baseline. Second, the criticality classes are **spatially correlated, not randomly scattered** — high-priority sensors concentrate around designated sites like the hospital and the power station, because critical infrastructure in a real city is clustered. That clustering is what makes it *possible* to group critical sensors into shared low hovers."

**Q: Why not use a real city map?** — Because we needed **twenty independent layouts** with **controlled** building-height and elevation distributions to run paired statistical tests. One real map gives n = 1. Generalisation to other generators is listed explicitly as a limitation and as future work.

**Q: Is 500 sensors realistic?** — Yes; the report notes municipal deployments of several hundred devices are unremarkable. It also matches the scale of the base framework's setting.

---

## Figure 3.1 — Overview of the proposed framework (p. 20)

**Two panels.**
- **(a) Adaptive descent over a cluster of high-priority sensors** — the aircraft dipping down over a critical group.
- **(b) The planner as a deterministic coupled pipeline:** greedy cover → continuous-altitude solve → local search → route.

**What to say:**
> "Panel (a) is the behaviour in one picture: over a concentration of critical sensors the aircraft **descends into the 21-metre band**, then climbs back out. Panel (b) is the mechanism: four deterministic stages. Note the word **deterministic** — there is no learned component in the reported system. Stage 2 gives coupled placement-and-altitude, Stage 3 is the local search that Chapter 4 shows is doing all the work, and Stage 4 routes and spends the budget."

**Q: Why "deterministic"?** Because given a layout and a seed, the planner produces the same plan every time — no stochastic policy, no sampling. That makes the paired comparison clean: variance across the 20 layouts is entirely **layout** variance, not **algorithm** variance.

**Q: Where would the learned model have gone?** Replacing Stages 2–4 with a trained attention decoder emitting both the sensor permutation and the continuous altitude. That is the future work in §5.3.1.

---

## Figure 3.2 — Altitude against mission progress (p. 22)

**What is plotted.** x-axis: **mission progress (%)**, 0 → 100. y-axis: **adaptive descent altitude (m)**, roughly 0 → 80. One line per method, plus markers for **adaptive descent events** and annotation *"sensor data satisfies ~18 m Δ of max"*. Legend lists: Adaptive Altitude, Behaviour Priority Nodes, Policy, Strong/Coupled (heterogeneous), Coupled/Greedy (ablation), Two-Stage (decoupled), 2D-AUTO (47 m), Served priority (Josh Kloss), plus reference bands: High (~68 m), Medium (~52.5 m), Low (~29 m).

**What to say:**
> "This is the altitude *time series* — what the aircraft actually does over the course of a mission.
> The **planar baseline is a flat line** — it holds one constant height throughout, by definition.
> The **decoupled baseline cruises high and descends abruptly** — and crucially those descents happen **only during its repair pass, after coverage has already been committed**. The descent is bolted on to a plan that was already settled.
> **Our method descends where critical sensors require it and returns to an efficient cruise altitude between such events.** That return is not optional: because climbing costs about 39 joules per metre while descending recovers only about 4, staying low would be prohibitive. So the correct description is not 'flies low' — it is **'descends on demand and climbs back'**."

**Q: Why doesn't your method just stay at 21 m the whole time?**
> Three reasons. (1) It must clear buildings — flying at 21 m across the map means colliding with a 37 m structure. (2) The climb/descent asymmetry of Eq. 2.4 means every avoided climb is cheap and every forced climb is expensive; a low cruise means constant climbing over obstacles. (3) At low altitude the footprint is tiny — $(H-z)\tan 60°$ — so you'd need far more hovers to cover the same sensors, and hovering is the most expensive state. Low altitude is only worth paying for **where the constraint demands it**.

**Q: Why does the report call this "representative"?** Because it is **one** mission, plotted for interpretability. The **statistical** claim about altitudes is Fig. 4.4 (the distribution over all layouts) and Table 4.1. Fig. 3.2 shows *shape*; Fig. 4.4 shows *evidence*.

---

## Figure 4.1 — Per-class satisfaction across methods (p. 25)

**What is plotted.** Grouped bar chart. x: the three criticality classes (High / Medium / Low). y: **satisfaction (%)**, 0–100. One bar per method within each group. **Error bars are 95% Student-*t* confidence intervals over twenty layouts.** **Markers above each group denote paired significance against the planar baseline.**

**What to say:**
> "This is the headline. Read the **high-priority group on the left first** — the planar bar is essentially on the floor at 0.6%, and ours is at 70.2%. *That* disparity is the principal result of this work. Then note the error bars are **95% confidence intervals across twenty layouts**, and the significance markers are **paired** tests against the planar baseline — so this is not one lucky map."

**Q: Why are the error bars wider on the high class (±7.3) than the medium (±3.8)?**
> Because the high class is only **10% of 500 ≈ 50 sensors**, so each layout's high-class percentage is computed from a small denominator and is inherently noisier. Also the high class is exactly the class where whether a critical cluster is reached before budget exhaustion swings the number most.

**Q: Isn't a bar chart with error bars hiding the pairing?** Fair point — the bars show marginal means, but the *tests* annotated on it are paired, and the paired detail is in Tables 4.2/4.3. That's why both are reported.

---

## Figure 4.2 — CDF of high-priority achieved rates (p. 26)

*Prerequisite: a **CDF (cumulative distribution function)** plots, for each value on the x-axis, the fraction of samples less than or equal to it. It rises from 0 to 1 left to right. A curve **further right** = better, because more mass sits at higher rates.*

**What is plotted.** x: **achieved rate (Mbps)**, 0 → 60. y: cumulative fraction, 0 → 1. One curve per method. A **vertical dashed line at 38 Mbps** marks the critical floor. The legend notes each method's percentage and the label *"left plateau at 0 = unserved (rate 0)"*.

**What to say:**
> "This shows the *whole distribution* of achieved rate on the critical class, not just the pass/fail percentage.
> **The left plateau in each curve is unserved nodes — budget exhausted, not a constraint violation.** That distinction matters: because our rate floors are **hard constraints**, we never serve a sensor *below* its floor. A sensor is either served **at or above 38 Mbps**, or **not served at all**. So the curve is essentially a step at zero (the unserved fraction) and then mass out past the floor.
> **The proposed method shifts the distribution well past the 38 Mbps floor** — we don't merely scrape over the line, we clear it with margin, because Eq. 3.7 places the hover at the altitude the *strictest* member permits, and every other member is then comfortably inside its own sphere."

**Q: ⭐ Why is there no mass between 0 and 38 Mbps?**
> Because the floors are **hard**. A sensor served below its floor is not counted as served, and the planner never admits an infeasible member to a cluster. Compare this with a **soft-penalty** formulation, which would produce a smear of sensors at 30–37 Mbps — technically "collected", actually failed. The absence of that smear is a visible consequence of our design choice.

**Q: Why show a CDF at all if the metric is just a percentage?**
> To demonstrate **margin**, not just pass rate. Two planners could both report 70% while one sits at 38.1 Mbps (fragile — any modelling error flips it to failure) and the other at 48 Mbps (robust). The CDF distinguishes them; a bar chart cannot.

---

## Figure 4.3 — Energy expenditure by component (p. 27)

**What is plotted.** Stacked/grouped bars of **energy (kJ)** per method, decomposed into **Flight**, **Hover** and **Communication** components. Annotation notes the equal budget of ~82.65 kJ and that all methods spend within about 3 kJ of each other.

**What to say — and the caption tells you exactly how to frame it:**
> "**All methods spend approximately the common budget, so this figure should be read for the *distribution between components* rather than for a winner.** There is deliberately no winner here — that is the point of the budgeted protocol. What it shows is *how* each method spends: how much goes into horizontal flight, how much into hovering — remembering that hovering is the most expensive state at 168.5 W — and how much into the actual communication."

**Q: But you spend 3 kJ less than the planar baseline — isn't that a win?**
> **No, and the report explicitly declines to claim it.** Under a budgeted regime every method stops when the budget runs out, and every method stops slightly *short* because **a hover cannot be flown in part**. We use larger clusters, hence coarser energy increments, hence we stop marginally further short. **Leaving budget unspent is not a virtue.** The correct reading is that expenditures are comparable — which is what the design was intended to ensure. Note also that against the *decoupled* baseline the energy difference is **not significant at all** (p = 0.33 / 0.50), which is the ideal outcome.

**Q: What would you expect the decoupled baseline's decomposition to show?**
> Relatively more energy going into **flight and repair hovers**, because its second pass adds shallow hovers appended to an already-settled tour — extra descents, extra climbs, extra hover time — and all of that is drawn from the same budget that should have gone to covering the low-priority bulk.

---

## Figure 4.4 — Distribution of hover altitudes (p. 30)

**What is plotted.** Kernel-density curves of **hover altitude (m)** — x-axis 0 to ~85 m, y-axis density 0 to ~0.06 — one curve per method. A **pink shaded band spanning roughly 25–40 m** is labelled *"critical-service altitudes"*. The planar baseline is drawn as a **grey vertical band at 46–50 m** with a dashed line through it, annotated *"one altitude per scene"*, rather than as a density — because it holds **one** altitude per layout, so there is nothing to distribute.

**Legend names (report prose in brackets):** Two-Stage *(decoupled)*, Coupled-Greedy *(ablation)*, Strong-Coupled *(the proposed method)*, 2D-AUTO *(planar baseline)*.

**⭐ Read the caption's precision point first:** *"Figure 4.4 is a distribution over **hover** altitudes only, so it describes **where service is delivered**, while the **cruise** behaviour between hovers is visible in Figure 3.2 and not in the distribution."*

**What to say:**
> "Three distributions, three stories.
> **Planar:** a single band at 46–50 m — one altitude per layout, drawn as a band rather than a density because there is nothing to distribute. That band sits **entirely above the 21 m critical radius**, which is the geometric statement of its 0.6%.
> **Proposed:** **unimodal**, a single mode near **28 m**, **86% of hovers below 40 m**, and **none whatever above 60 m**. Every hover is flown from within or near the band the critical floor requires.
> **Decoupled:** **genuinely bimodal** — a dominant mode near **68 m** and a second near **38 m**. The first is its **rate-blind cover**; the second is its **repair hovers**, created *after* coverage was committed. **59% of its hovers sit above 60 m, against 0% for ours.** Its low mode is **additional expenditure appended to a settled plan** rather than an **integrated decision** — and that is exactly the structural weakness Table 4.3 measures."

**Q: ⭐ If low is good, why is your mode at 28 m rather than 21 m?**
> Because 21 m is the requirement for a **ground-level** critical sensor only. Eq. 3.7 gives $H = z_i + \sqrt{(d^{\max}_i)^2 - \rho_i^2}$: a critical sensor **elevated on a rooftop** permits a proportionally higher hover, and a cluster whose strictest member is *medium* priority permits up to 40 m. 28 m is the **energy-optimal compromise** the refinement finds, not a target we set. We never specify an altitude anywhere in the algorithm — every altitude is *derived* from the membership.

**Q: Doesn't "none above 60 m" mean you're throwing away coverage?**
> It means the refinement judged wide-footprint high hovers not worth their cost **under this budget and these floors**. With laxer floors or a bigger budget the distribution would shift up. It is an outcome, not a constraint — $H_{\max}$ is 150 m and the planner was free to use it.

**Q: What about the ablation curve (Coupled-Greedy, light blue)?** ⭐ Be ready for this — it is the most revealing curve on the plot. **It is also bimodal**, with a low mode near 38 m and a substantial high mode near 68 m, plus a bump near 49 m. So the coupled-greedy variant **still places most of its hovers high**. That is the *visual* explanation of Table 4.4: coupling alone does not bring the aircraft down into the 21 m band, which is why it scores identically to the decoupled baseline on the critical class. **Only the refinement collapses the distribution into the single low mode you see for Strong-Coupled.** Pointing this out unprompted shows you understand your own ablation.

**Q: Is unimodal-vs-bimodal itself evidence?** Yes, and it's the cleanest kind: it shows *architecturally* that the decoupled baseline makes **two separate decisions at two separate times**, while ours makes **one integrated decision**. The two modes are literally the two phases of its algorithm made visible.

---

## Figure 4.5 — Representative three-dimensional trajectories (p. 32)

**What is plotted.** 3D flight paths over the deployment: x and y to 1000 m, altitude to ~100 m, with the **Depot**, marked **critical sensor** locations, and one trajectory per method (planar, decoupled, proposed). Annotations mark "3D trajectories over the network" and "lowest dive 10 m". *The coupled-greedy ablation is omitted for legibility, since its route closely resembles the full method's while its altitudes are less refined.*

**What to say:**
> "This is the geometric version of Fig. 4.4.
> The **planar baseline is confined to one horizontal plane** — a flat sheet of a trajectory.
> **Our method descends over concentrations of critical sensors and returns to cruise between them**, and — this is the key visual — those **descents occur *within* the tour**. They are part of the route.
> The **decoupled baseline's descents are *appended to* it** — you can see them as separate excursions tacked on after the main sweep. That's the repair pass, drawn in space."

**Q: Why is the ablation omitted?** Stated in the caption: legibility. Its **route** closely resembles the full method's — which makes sense, since both use the same Stage 2 construction and the same Stage 4 router — while its **altitudes** are less refined. Since altitude is the thing being illustrated and the routes overlap, plotting it would obscure rather than inform. The ablation's actual evidence is Table 4.4, where it belongs.

**Q: Isn't one representative trajectory anecdotal?** Yes, and the report treats it as such. It is labelled **representative**, used for *interpretation*, and every quantitative claim rests on the 20-layout tables. Figures 3.2 and 4.5 answer "what does it *do*?"; Tables 4.1–4.4 answer "does it *work*?".

---

# PART 8 — THE NUMBER SHEET (memorise)

**Geometry — the core argument**
| | |
|---|---|
| Mandatory 2D cruise altitude | **≈ 47 m** (range 46–50 across layouts) |
| Critical-class service radius | **21 m** |
| **The contradiction** | **47 > 21** |
| Rooftop threshold for a reachable critical sensor | $z_i \gtrsim$ **26 m** |
| Predicted planar high satisfaction from geometry | **0.70%** |
| Measured planar high satisfaction | **0.60%** |
| Layouts with exactly zero critical served | **14 of 20** |

**Classes**
| Class | Share | Floor | Radius |
|---|---|---|---|
| High | 10% | 38.0 Mbps | 21 m |
| Medium | 20% | 32.5 Mbps | 40 m |
| Low | 70% | 29.0 Mbps | 60 m |

**Environment & platform**
| | |
|---|---|
| Map | 1000 × 1000 m |
| Sensors | 500 |
| Sensor elevations | 0–50 m |
| Data per sensor | 0.2–1.5 MB |
| Bandwidth B | 2 MHz |
| Noise σ² | −110 dBm |
| Path-loss exponent α | 3.0 |
| Half-beamwidth θ | 60° |
| Clearance $h_{\text{safe}}$ | 10 m |
| Altitude bounds | 20 m – 150 m |
| Hover power $P_0+P_i$ | 168.5 W |
| Mass / efficiency / descent coeff | 2 kg / 0.5 / 4 J·m⁻¹ |
| Climb cost (derived) | ≈ 39 J/m |
| Time budget per hover $\tau_{\max}$ | 30 s |

**Protocol**
| | |
|---|---|
| Layouts | 20 (seeds **42–61**) |
| Full 2D coverage energy $E^{\text{full}}_{2D}$ | **127.15 kJ** |
| Budget fraction | **0.65** |
| **Mission budget** | **82.65 kJ** (≈ 490 s of hover-equivalent) |
| Tests | paired Student-*t* + Wilcoxon signed-rank |

**Headline results**
| Comparison | High | Medium | Low | Energy |
|---|---|---|---|---|
| Planar → Proposed | 0.6 → **70.2** (+69.6) | 61.0 → **80.8** (+19.8) | 66.7 → 74.2 (+7.5) | 81.5 → 78.5 |
| Decoupled → Proposed | 54.9 → **70.2** (+15.3) | 66.8 → **80.8** (+14.1) | 52.7 → **74.2** (+21.5) | 79.4 → 78.5, **n.s.** |
| Wins per layout | **20/20** high, 18/20 med, 15/20 low | | | |

**Ablation**
| Variant | Joint | Refine | High | Low |
|---|---|---|---|---|
| Decoupled | No | No | 54.9 | 52.7 |
| Coupled-greedy | Yes | No | **54.9** (Δ = +0.00, **p = 1.000**) | 64.9 (+12.2, p=0.0032) |
| Proposed | Yes | Yes | **70.2** (+15.3, **p = 0.0007**) | 74.2 |
| Medium, refinement gain | | | +9.3 (p = 0.0026) | |

**Behaviour**
| | |
|---|---|
| Proposed hover mode | **28 m** |
| Proposed hovers below 40 m | **86%** |
| Proposed hovers above 60 m | **0%** |
| Decoupled modes | **68 m** (rate-blind cover) and **38 m** (repair) |
| Decoupled hovers above 60 m | **59%** |
| Proposed altitude during critical service | **~5 m lower** than outside; both 3D baselines fly **higher** |
| Refinement runtime | **~30 min per layout** at N = 500 |

---

# PART 9 — VIVA QUESTION BANK (~90 questions)

## 9.1 Opening / conceptual

**Q1. Explain your project in one minute.**
Use the Part 0 pitch.

**Q2. What is the one-sentence novelty?**
Making altitude a per-hover decision variable governed by per-class QoS floors converted into hard geometric constraints, refined by continuous local search — evaluated under a fixed energy budget so quality rather than energy is the measured outcome.

**Q3. Why can't you just use a better routing algorithm on the 2D planner?**
Because the failure is **geometric, not computational**. The 2D planner must cruise at ≈47 m to clear buildings; a critical sensor requires the aircraft within 21 m. **No permutation of stops changes a distance that is already too large.** This is a feasibility statement, not a performance one.

**Q4. Why does a drone need to be close to a sensor?**
Achievable rate is Shannon: $R = B\log_2(1+\text{SNR})$. SNR falls as $1/d^{\alpha}$ with α = 3 in urban conditions. So received power falls **cubically** with separation. A demanding rate requires high SNR, which requires small $d$.

**Q5. Why not just increase transmit power instead of descending?**
Rate is **logarithmic** in SNR while SNR is **cubic** in distance. Halving the distance multiplies SNR by 8; matching that with power alone requires 8× transmit power — from a battery-powered sensor whose entire design goal is multi-year life. Descending is by far the cheaper lever. This is the fundamental Shannon asymmetry: power buys bits logarithmically, proximity buys SNR cubically.

**Q6. Why rotary-wing and not fixed-wing?**
Collection requires **hovering** over a sensor for seconds. Fixed-wing cannot hover. That choice also brings the crucial property in Q7.

**Q7. What is the single most important property of the rotary-wing energy model?**
$P(0) = P_0 + P_i$ — **hover power is the MAXIMUM of the power curve, not the minimum.** A rotorcraft burns more standing still than cruising efficiently. Hence *waiting* for a weak link is expensive, hence a weak link is an **energy** problem rather than merely a slow one, hence you must descend.

**Q8. Why are there exactly three criticality classes?**
Three is the minimum that produces a *non-trivial* ordering: one class that the 2D planner **cannot** serve (high), one it serves **partially** (medium), and a **bulk** class (low, 70%) whose coverage is what gets sacrificed when a budget is misallocated. Two classes wouldn't show the sacrifice effect; more would fragment the sample sizes without adding structure.

**Q9. What is the "governing tension"?**
Eq. 3.3 (coverage) pushes altitude **up** — a higher hover sees more sensors, needs fewer stops, saves flight energy. Eq. 3.2 (service radius) pushes altitude **down** — a demanding sensor must be approached closely. The resolution **differs at every hover** depending on who is being served, which is the structural reason altitude cannot be a mission-wide constant.

## 9.2 Communication model

**Q10. Derive the service radius equation.** — See Part 4 §4.2. Do it on the board.

**Q11. Why is the service radius an "iff"?**
Because $R_i$ is **strictly monotonically decreasing** in $d_i$. So $R_i \ge R^{\min}$ holds **exactly** when $d_i \le d^{\max}$. Monotonicity is what makes the inversion valid and the constraint exactly equivalent to a sphere.

**Q12. Why do the radii differ so much (60 vs 21 m) for floors that differ by only 31%?**
Double non-linearity. Rate is **logarithmic** in SNR, so a small rate increase demands a *large* SNR increase; and SNR is **cubic** in distance, so a large SNR increase demands a *drastic* distance reduction. 29 → 38 Mbps (+31%) becomes 60 → 21 m (−65%).

**Q13. What is β?**
The **reference channel gain** — the channel gain at a reference distance of 1 m. It bundles antenna gains, carrier wavelength and hardware constants so that the path loss can be written as $\beta/d^\alpha$.

**Q14. Why α = 3?**
Urban environment with shadowing. α=2 is free space (too optimistic — it would *enlarge* our radii and make the 2D baseline look better than reality); α≥4 is heavy indoor obstruction. 2.7–3.5 is the standard urban band.

**Q15. What if α were 2?**
$d^{\max} \propto K^{1/\alpha}$, so all radii would grow substantially, the critical radius could exceed 47 m, and the 2D baseline would no longer fail categorically. **Our qualitative conclusion depends on the critical radius being below the mandatory cruise altitude** — a point the report states explicitly under threats to validity. With α=2 we would be arguing about efficiency, not feasibility.

**Q16. You assume line of sight but you model buildings — isn't that inconsistent?**
Acknowledged explicitly as a limitation. The argued direction of the bias: **blockage would principally affect long links**, and our method **deliberately shortens** the links to its critical sensors, so omitting blockage is **more likely to understate than overstate** our advantage. The report also states that **this expectation has not been verified**. A blockage model is named as the extension that would most improve realism, because it adds a **second reason to climb** (LoS probability) that our model omits.

**Q17. Why is B = 2 MHz fixed rather than optimised?**
It is taken **unchanged from the base framework [1]**, deliberately, so that the planar baseline reproduces its own authors' operating point rather than a re-tuned variant we chose. Bandwidth allocation is orthogonal to the altitude question; making it a variable would confound the comparison.

**Q18. Does the sensor or the drone transmit?**
The sensor uplinks its stored data to the drone; the drone is the receiver, and σ² is the noise power at the receiver.

## 9.3 Coverage geometry

**Q19. Derive the footprint relation.**
Downward cone of half-angle θ from the aircraft at height $H$. At the plane of a sensor with elevation $z_i$, the vertical drop is $H - z_i$, and the cone's radius there is $(H-z_i)\tan\theta$. So sensor $i$ is inside the footprint iff $\rho_i \le (H-z_i)\tan\theta$.

**Q20. ⭐ You took Eq. 3.3 from Liu et al. — is that not just copying?**
No, and the report is explicit about the difference. In [2], θ is a **radar detection angle**, and the constraint is **gated by a binary scheduling variable** under $\sum_k \omega_k + b = 1$ — at any instant the platform is sensing **exactly one** node or uploading. So for them it is a **pointing constraint on a single scheduled target**; the words *footprint*, *beamwidth* and *coverage cone* **do not appear in that work at all**. We reinterpret the same geometry as a **multi-sensor communication footprint**. And that reinterpretation is load-bearing, not cosmetic: **if a hover serves one sensor at a time, climbing brings no benefit and the optimal altitude is trivially the lowest permitted.** Only when one hover covers a *group* does height buy anything — and only then does the trade-off our contribution rests on exist.

**Q21. What is the footprint radius at 30 m above a ground sensor?**
$(30-0)\tan 60° = 30 \times 1.732 \approx$ **52 m**.

**Q22. Why θ = 60°?**
A standard, moderately directional downward antenna half-beamwidth. It is listed among the parameters **specific to this work**. The qualitative conclusion doesn't hinge on it; a narrower θ would simply require higher altitudes for the same footprint, strengthening the tension.

## 9.4 The energy model

**Q23. Explain the three terms of Eq. 2.3.** — See §1.6.2 table.

**Q24. Why is hovering more expensive than cruising?**
Induced power. Hovering, the rotor beats the **same column of already-downwashed air**, which is inefficient. Moving forward, it continuously meets **fresh undisturbed air**, so lift is cheaper. The induced term falls with speed; parasite drag (∝V³) eventually takes over, giving a U-shaped curve with an interior minimum.

**Q25. What did the base framework get wrong here?**
AUTO [1] charges $P_F$ = 75 W flight and $P_H$ = 50 W hover — **hovering cheaper than flying**. That ordering is **physically wrong for a rotorcraft**. We replaced it with the Zeng, Xu & Zhang model [4].

**Q26. Explain Eq. 3.5 — why min P(V)/V and not min P(V)?**
Because you are covering a **fixed distance**, so the right objective is **joules per metre**, not watts. Minimising power alone says "fly as slowly as possible", but a slow flight over the same distance takes far longer and burns more total energy. $P(V)/V$ has units J/m and its minimiser is the **maximum-range speed**, which is faster than the minimum-power speed.

**Q27. Explain the climb/descent asymmetry.**
$E_{\text{vert}} = \frac{mg}{\eta}\Delta h_{\text{climb}} + c_d \Delta h_{\text{descent}}$ with $c_d \ll mg/\eta$. Climbing does work against gravity divided by motor efficiency: with m=2 kg, η=0.5, that's ≈**39 J/m**. Descending recovers almost nothing — a rotorcraft cannot regeneratively harvest potential energy; it must still spin rotors for controlled descent — so $c_d$ = **4 J/m**. Roughly **10:1**.

**Q28. ⭐ Why doesn't your planner just fly low all the time?**
Three reasons: (1) **building clearance** — a low cruise means colliding, or constantly climbing over structures; (2) the **10:1 climb/descent asymmetry** makes those forced climbs prohibitive, so *"every descent is eventually paid for by a climb"*; (3) at low altitude the **footprint shrinks**, so you need many more hovers, and hovering is the **most expensive state**. Fig. 3.2 confirms the behaviour: it descends **on demand** and returns to efficient cruise. Low altitude is bought only where the constraint requires it.

**Q29. Why is η = 0.5?**
Motor and rotor efficiency — you pay roughly double the ideal $mgh$. It is a conservative, standard-order value listed as specific to this work.

## 9.5 The evaluation regime (expect heavy questioning here)

**Q30. ⭐⭐ Why an energy budget instead of "serve everything, minimise energy"?**
Because the natural formulation makes the metric **degenerate**. If the rate floors are **hard**, then **any feasible plan** serves every sensor at or above its floor, so **every feasible plan scores 100% on every class by construction**. Comparing two plans on satisfaction compares two numbers that are **both exactly one**. The metric intended to demonstrate the contribution is **mathematically incapable of expressing it**. The budget breaks the degeneracy: methods must now choose *whom to sacrifice*, and satisfaction becomes informative.

**Q31. Isn't fixing energy just a way to guarantee you win?**
The opposite — it removes our easiest possible win. If energy were free to vary we could report "we saved X% energy", which is the standard and much easier claim. By **fixing** energy we forfeit that claim entirely (*"no energy-saving claim is made anywhere in this report"*) and force the comparison onto the harder ground of delivered quality. And the design is validated by the fact that against the decoupled baseline the energy difference is **not significant** (p = 0.33 / 0.50) — the control demonstrably worked.

**Q32. Why 0.65?**
It makes the budget **binding but not crippling**. At 1.0 the metric re-degenerates (everyone finishes, everyone scores 100%). At a very small fraction nobody achieves much and the comparison is noise. 65% of a full 2D mission forces genuine triage — the planner must actually decide whom to sacrifice, which is exactly the decision the paper is about. Numerically 82.65 kJ is ≈490 seconds of hover-equivalent, a genuinely tight mission.

**Q33. Why compute the budget from a single reference layout?**
So it is **constant across experiments**. Results from the 10-layout and 20-layout campaigns remain directly comparable. A per-layout budget would be a moving target and would silently give easier layouts more energy.

**Q34. What exactly does "satisfaction" mean?**
The **fraction of sensors in a class that were served at or above that class's rate floor**, at the point the mission budget was exhausted. Sensors in clusters never reached are counted **unserved**.

**Q35. Is an unserved sensor a constraint violation?**
**No.** It is a sensor the budget never reached. Because the floors are **hard**, every sensor we *do* serve is at or above its floor. This is exactly what the left plateau of Fig. 4.2 shows.

**Q36. Why 20 layouts and not 100?**
20 gives adequate power for the effect sizes involved (the critical-class effect is enormous and unanimous, 20/20) while the refinement stage costs ~30 min per layout per method — four methods × 20 layouts is already substantial compute. We doubled from 10 to 20 specifically because 10 left the critical comparison ambiguous between tests.

## 9.6 Baselines and fairness

**Q37. Is your planar baseline a straw man?**
No, on four counts. (1) It reproduces the **published fixed-altitude formulation**, and its communication and platform parameters are taken **unchanged from the base framework [1]** *"so that the planar baseline reproduces its operating point rather than a re-tuned variant"* — we did not weaken it by re-tuning. (2) It uses the **same nearest-neighbour + 2-opt router** we use, so routing is controlled out. (3) Its failure is **independently predicted from geometry at 0.70% against 0.60% observed** — a bug would not reproduce a geometric prediction to that precision. (4) ⭐ **We computed its cruise altitude per-city rather than globally, which deliberately favours it.** Each seed's altitude is that city's own tallest structure + $h_{\text{safe}}$ (37.4–39.6 m → 47.4–49.6 m). Using the global maximum across all seeds would have been simpler and *worse for the baseline*: at seed 42 the own-city 47.4 m meets the critical floor for 8% of high-priority nodes, while the global 49.6 m meets it for **0%**. **We chose the configuration more favourable to our opponent.**

**Q38. ⭐ But the planar baseline can't do the task — so isn't the comparison unfair?**
That is precisely the finding, and the report says so: *"The planar comparison, while emphatic, is against a method that cannot physically perform the task. The stronger test is against the decoupled planner."* We do **not** rest the contribution on the planar comparison. The decoupled 3D baseline has the **same altitude freedom** and differs only in ordering placement before altitude — and we still win on all three classes.

**Q39. Describe the decoupled baseline and why it is strong.**
Stage 1: greedy set-cover placement that **ignores rate requirements** and seeks only low-energy coverage — which drives it to **high altitudes and wide footprints**. Stage 2: a **repair pass** — every sensor below its floor is pulled out and re-served by additional shallow hovers. It is strong because it has the **full 20–150 m altitude range**, it is *"the natural engineering response"*, and it was *"deliberately given every advantage"*.

**Q40. ⭐ Why does the decoupled baseline score WORSE than the 2D planner on low-priority sensors (52.7% vs 66.7%)?**
Structural, not incidental. Its first stage is **blind to the rate requirement**, so many sensors are found under-served; repairing them requires a **second pass of shallow hovers paid from the same fixed budget**; and that energy is then **unavailable for coverage**. The sensors that lose out are **the numerous low-priority ones** — 70% of the deployment. *"The decoupled planner recovers its critical sensors, but at the expense of the bulk of the deployment."* This is the empirical proof of the §2.6 claim that **the interaction between placement and altitude is not benign**.

**Q41. Why use the same router in every method?**
To **control routing out** as a confound. If we used a learned router for ours and nearest-neighbour for the baselines, any difference could be attributed to routing rather than altitude, and the result would be uninterpretable.

**Q42. Why only one aircraft?**
*"So that the altitude effect is not confounded with fleet sizing."* AUTO optimises the aircraft count; letting that vary would mix two effects. Multi-aircraft partitioning **exists in the implementation but is not evaluated**, and is listed as future work — where it raises a genuinely new **load-balancing** question, since critical sensors are spatially clustered and an even *geographic* partition need not be an even partition of *criticality*.

## 9.7 The method

**Q43. Walk through the four stages.** — Part 4 §4.8.

**Q44. Derive Eq. 3.7.**
Sensor $i$ is served iff $\rho_i^2 + (H-z_i)^2 \le (d^{\max}_i)^2$, so $H \le z_i + \sqrt{(d^{\max}_i)^2 - \rho_i^2}$. Each member imposes a **ceiling**; satisfying all members means taking the **minimum** ceiling. Hence $H(S) = \min_{i\in S}\left(z_i + \sqrt{(d^{\max}_i)^2 - \rho_i^2}\right)$.

**Q45. What if the square root is imaginary?**
Then $\rho_i > d^{\max}_i$: the sensor is **outside its own service sphere horizontally**, so it cannot be served from that hover **at any altitude**. Such a candidate is **not admitted** to the cluster.

**Q46. What are the bounds on H(S)?**
$\max(H_{\min},\, z_a + h_{\text{safe}}) \le H(S) \le H_{\max}$ — never below the global floor (20 m) and never within the 10 m safety clearance of the anchor; never above 150 m.

**Q47. ⭐ Why does grouping critical sensors together emerge without being programmed?**
Because **the altitude of a hover is dictated by its strictest member**. Admitting one high-priority sensor forces the **entire cluster** to 21 m — so thirty low-priority sensors that could have been served cheaply from 60 m are dragged down with it, wasting energy. The energy objective therefore *prefers* to segregate critical sensors into their own low clusters and let the rest fly high. **Criticality-aware clustering falls out of the geometry; we never wrote a rule for it.**

**Q48. What does "continuous" altitude mean?**
Altitude is a **real number**, recomputed exactly from Eq. 3.7 for the current membership after every candidate move — not selected from a discrete menu of levels. Any membership change immediately re-derives an exact new altitude.

**Q49. What are the three local-search moves and why those?**
**Relocate** (a sensor between clusters) fixes individual misassignments; **merge** fixes over-fragmentation, where two nearby clusters could share one hover; **split** fixes over-aggregation, where one strict member is dragging a large group down and it is cheaper to peel it off. Together they can reach any partition, and each directly corrects a characteristic failure mode of the greedy constructor.

**Q50. What is the acceptance criterion?**
A move is accepted **iff it reduces the total energy estimate**. Pure descent — no simulated annealing, no tabu. Hence a **local** optimum, not a global one.

**Q51. Why is the greedy constructor "myopic"?**
Each cluster is **fixed as soon as it is formed**. It cannot anticipate that admitting a critical sensor to a **later** cluster would have been cheaper than forcing an **early** cluster low. The refinement's measured gain (+15.3 points) **is** the value of undoing exactly those decisions.

**Q52. Why 30 minutes? Where does the time go?**
The local search **re-evaluates cluster energies from scratch after each candidate move**, and there are many candidate moves over 500 sensors. Two named fixes: **incremental evaluation** (update only what changed) and **restricting moves to spatial neighbourhoods** (don't consider relocating a sensor to a cluster 800 m away).

**Q53. Could you make it real-time?**
Not with the present local search — that is limitation 1, and since the refinement supplies the **entire** critical advantage, you cannot cheapen it by dropping the stage. Two routes: the algorithmic speed-ups above, or a **learned policy** that produces a plan in **milliseconds rather than thirty minutes** (§5.3.1). That is precisely why the learned policy is proposed as future work — as a fix for the principal weakness, not as novelty for its own sake.

**Q54. Is your solution optimal?**
No, and we do not claim it. It is a **greedy construction refined to a local optimum** under three move types. The underlying joint problem generalises TSP and set cover, both NP-hard. What we claim is a **significant improvement over strong baselines under a controlled budget**, not optimality.

**Q55. Why hard constraints instead of a weighted objective?**
A weighted objective **permits the optimiser to trade** — to under-serve one critical sensor in exchange for enough unimportant ones. That is *"precisely the behaviour a criticality requirement is meant to prevent."* A hard constraint makes violation **inadmissible**: no gain elsewhere can buy it. It is also visible in Fig. 4.2 — there is **no mass between 0 and 38 Mbps**, whereas a soft formulation would produce a smear of technically-collected, actually-failed sensors just below the floor.

## 9.8 Results and statistics

**Q56. What is your headline result?**
Critical-class satisfaction from **0.6% to 70.2%** at the same energy budget, **winning on all twenty layouts**; and against a strong decoupled 3D baseline, significantly better on **all three classes** at **statistically indistinguishable energy**.

**Q57. Why report two statistical tests?**
The paired *t*-test is conventional but **assumes normality**, which n=20 cannot confirm. Wilcoxon is **non-parametric** and **not distorted by one extreme layout**. Agreement means the conclusion is robust to the distributional assumption. And *"where they disagree, the disagreement is stated rather than the more favourable figure selected."*

**Q58. What is a p-value — and what is it not?**
The probability of observing a difference at least this extreme **if the null hypothesis were true**. It is **not** the probability that the null is true, and **not** the probability your result is correct.

**Q59. Why is the paired design important?**
Layouts vary enormously in difficulty. An unpaired comparison would have that between-layout variance swamping the effect. Running every method on the **identical 20 layouts** and analysing the **differences** removes it entirely.

**Q60. ⭐ Your low-priority p is 0.0486 — that's barely significant.**
Agreed, and **we say so ourselves in the report before anyone asks**: the Wilcoxon value *"lies immediately below the conventional threshold"* and we win on **fifteen of twenty** rather than the near-unanimous margins elsewhere. Therefore *"it is not advanced as a principal finding; the defensible statement is that **low-priority service is not degraded**"* — and that is how the conclusions describe it. Note this qualification applies to the **planar** comparison; against the **decoupled** baseline the low-priority margin is +21.5 points at p < 0.0001, which is not marginal at all.

**Q61. ⭐ You used 3.0 kJ less energy than the planar baseline — that's your real contribution, isn't it?**
No. **No energy-saving claim is made anywhere in this report.** Under a budgeted regime every method stops when the budget is exhausted, and every method stops **slightly short because a hover cannot be flown in part**. We use larger clusters, hence coarser increments, hence stop marginally further short. **Leaving budget unspent is not a virtue.** The correct reading is that expenditures are **comparable**, which is what the design was intended to ensure — and against the decoupled baseline the difference is not even significant.

**Q62. ⭐ You went from 10 layouts to 20 — did you p-hack?**
No, and the distinction is clean. p-hacking is testing repeatedly and **stopping when you cross a threshold**, reporting the favourable stop. What happened here: the two tests **disagreed** (t: p=0.033, Wilcoxon: p=0.063), which is a **diagnostic that the sample was too small** to resolve — not a favourable result to bank. We doubled the sample **once**, and reported **all classes and both tests** from the larger campaign. Crucially, **we report the earlier ambiguity in the paper itself** rather than hiding it — which is the opposite of p-hacking.

**Q63. Why are the high-class confidence intervals so wide (±7.3)?**
The high class is only **10% of 500 ≈ 50 sensors**, so each layout's percentage rests on a small denominator and is inherently noisier. It is also the class most sensitive to whether a given critical cluster was reached before budget exhaustion.

**Q64. How do you know the planar baseline's 0.6% isn't a bug in your code?**
Because we **predicted it from geometry independently**. The baseline hovers directly overhead so the link distance is exactly $H - z_i$; reachability requires $H - z_i \le 21$, i.e. $z_i \gtrsim 26$ m. Counting the critical sensors that elevated across the layouts predicts **0.70%**; we measured **0.60%**, and even the residual is explained — those are qualifying sensors the **budget-truncated tour never reached**. In **14 of 20 layouts satisfaction is exactly zero**. A coding bug would not reproduce a geometric prediction to that precision.

## 9.9 The ablation (be enthusiastic, not defensive)

**Q65. ⭐⭐ Your ablation refutes your own headline. Explain.**
The ablation was intended to **confirm** that the joint formulation drove the improvement. **It established the opposite.** Coupled-greedy and the decoupled baseline are **indistinguishable on the critical class** — both 54.9%, mean paired difference **+0.00 points**, paired *t*-test **p = 1.000**. So deciding placement and altitude jointly, **without refinement, delivers no advantage whatever on the class the method exists to serve.** The refinement supplies the **entire** difference: 54.9 → 70.2, +15.3 points, p = 0.0007. We therefore **changed the claim to match the evidence**: what this work demonstrates is **altitude refinement under quality constraints**, *not* joint decision-making as such.

**Q66. So is the coupling useless?**
No — **its effect is elsewhere**. On the **low-priority** class, coupled-greedy improves on the decoupled baseline by **+12.2 points (p = 0.0032)**, because it **avoids the wasteful repair pass**. *"Its contribution is thus to protect bulk coverage rather than to serve critical sensors."*

**Q67. Why report a result that weakens your contribution?**
Because it is true, because it **constrains what the work may claim**, and because it is the most **useful** result for anyone building on this — it tells them **exactly which component to invest in**. A claim that coupling outperforms decoupling on critical sensors *would be contradicted by our own Table 4.4*; publishing it and then having someone else find the contradiction would be far worse.

**Q68. What does the ablation tell you about the greedy constructor?**
That it is **myopic in a way the refinement corrects**: because each cluster is fixed at formation, it cannot anticipate that admitting a critical sensor to a later cluster would have been cheaper than forcing an early cluster low. **The measured gap is the value of undoing such decisions.**

**Q69. Is the coupled-greedy variant a baseline?**
**No** — it is *"an ablation of the proposed method, not an independent baseline, and is reported as such throughout."* It is our own method with a stage removed. Calling it a baseline would falsely imply it is someone else's published method.

## 9.10 Machine learning (they will ask, because "attention" is in the base paper's title)

**Q70. The base paper is attention-based. Where is your neural network?**
There isn't one in the reported system — it is a **deterministic** four-stage planner, and we say so plainly. We **did** attempt a constrained-RL formulation with one dual variable per criticality class. It failed for **diagnosed** reasons, we report the diagnosis, and recovering it is the first item of future work — motivated not by novelty but because a trained policy would plan in **milliseconds rather than thirty minutes**, fixing our principal weakness.

**Q71. ⭐ Why did the training fail?**
Two identified causes. (a) **The effective learning rate reaching the altitude head decayed to order 10⁻¹⁰** — the head was effectively frozen, and a policy that has not moved from its initialisation can still report a respectable-looking satisfaction figure, which is how it went unnoticed until the **weights themselves were inspected**. (b) Updates were applied **once per epoch rather than per batch**. The named fixes: correct the rate and its decay schedule, update per batch, and **gate the run on whether the altitude bias has moved measurably by epoch fifty** — a cheap early tripwire.

**Q72. ⭐ What is the architectural difficulty in adding altitude to an attention decoder?**
Every one of these models — pointer networks, Bello, Kool — emits a **permutation over discrete items**. Adding a **continuous** output is *"not merely a matter of adding an output"*: **the continuous head must be conditioned on the discrete selection**, because the appropriate altitude depends on **which sensor has just been chosen**. Our own failure was an instance of exactly this: the altitude head could not see the chosen anchor, so it collapsed to a constant.

**Q73. What is a CMDP and how is it solved?**
A Constrained MDP maximises reward **subject to bounds on auxiliary costs**. Standard solution: a **Lagrangian** with one multiplier per constraint, multipliers updated by **dual ascent** (raise when violating, lower when slack). **Altman [15]** is the standard reference; **Achiam et al. [16]** gives a trust-region method with approximate guarantees during training. We planned one dual variable per criticality class.

**Q74. Why is a hard-constraint planner arguably better than an RL policy here?**
Because our floors are **genuinely hard**. A Lagrangian relaxation makes them **soft in practice** — a large enough reward can always buy a violation, and constraint satisfaction only holds **in expectation at convergence**. For a criticality requirement, "satisfied on average" is not what you want. The deterministic planner **cannot** violate a floor. The RL version's advantage is **speed**, not correctness.

**Q75. What is REINFORCE and what is a baseline?**
The foundational policy-gradient method: $\nabla J = \mathbb{E}[\nabla\log\pi(a|s)(R-b)]$. The **baseline $b$** is subtracted to **reduce variance without introducing bias** — Kool et al. use a **greedy rollout** of the current best policy; AUTO uses the realised system reward.

## 9.11 Limitations and threats (volunteer these — it reads as confidence)

**Q76. List your limitations.** — Six, per §6.2. Being able to rattle them off is worth more than defending against them.

**Q77. Which limitation worries you most?**
The **thirty-minute runtime**, because it is the one that blocks the obvious application — in-flight replanning — and because the ablation proves you **cannot avoid it by dropping the stage**. It is why both a fast learned policy and incremental local-search evaluation are the top two items of future work.

**Q78. All your layouts come from one generator.**
Acknowledged. Twenty samples establish the result is not an artefact of **one arrangement of sensors** — but **not** that it holds for a different **building density, elevation distribution or map size**. Repeating the campaign across generators is listed as immediate future work. The *qualitative* conclusion is more robust than the numbers: it needs only that **the critical service radius be smaller than the mandatory cruise altitude**, which holds whenever a city has buildings taller than ~21 m plus clearance.

**Q79. Everything is simulation — is the energy model validated?**
No. *"Every result in this report is obtained in simulation, so the energy model, however standard, has not been checked against a real airframe."* Physical validation is named as future work. The model itself is the standard published rotary-wing model of Zeng, Xu & Zhang [4], not one we invented.

**Q80. What if the rate floors were different?**
Different floors → different radii → different margins. *"The qualitative conclusion depends only on the critical radius being smaller than the mandatory cruise altitude, which is robust, but the numerical margins are tied to this calibration."* We chose floors to **span the achievable band** so that all three classes are meaningfully distinguishable.

**Q81. What happens if a sensor fails or moves mid-mission?**
Out of scope — the deployment is **static** and the planners are **deterministic with complete prior knowledge**. Online replanning is future work and **presupposes the fast planner**, since you cannot replan in flight with a thirty-minute algorithm.

## 9.12 Process, integrity and the "killer" questions

**Q82. ⭐⭐ What is the single most important lesson from this project?**
That the hardest failures are in **experimental design, not implementation**. Two separate versions of our evaluation produced results that were **internally consistent and yet carried no information** — the first because hard constraints made satisfaction **identically one**, the second because a floor calibrated to a service radius **larger than the map** could not be violated. **In both cases the figures were plausible and the code was correct. What was wrong was the question being asked of the data.** The same pattern hit the training work, where a policy frozen at initialisation still reported a respectable figure. In every case the error was found by asking **"is this measured quantity capable of taking a different value?"** rather than **"does this number look reasonable?"** That test proved far more informative than inspecting results.

**Q83. Tell me about a mistake you made.**
Use Q82 — three of them, diagnosed, and the general principle extracted. That is a much stronger answer than a small coding bug.

**Q84. What would you do differently if you started again?**
Design the **metric** first and immediately test whether it is **capable of varying** before writing any planner. We wrote the planner first and discovered twice that the metric was degenerate. Second: instrument the RL run with a tripwire on whether the altitude bias moves by epoch fifty, rather than trusting the reported reward.

**Q85. What is the practical impact of this work?**
For anyone deploying UAV data collection over a city with **heterogeneous QoS requirements**, it shows that the standard fixed-altitude formulation is not merely suboptimal but **infeasible** on the demanding class, gives a **checkable geometric test** for whether your deployment has this problem (compare $d^{\max}_{\text{critical}}$ against tallest structure + clearance), and shows that the fix must include **continuous altitude refinement**, not merely 3D freedom.

**Q86. If you had six more months, what is the first thing you'd do?**
Recover the learned policy, for the concrete reason that it converts thirty minutes into milliseconds and thereby unlocks online replanning — the application the current system cannot support. The causes are already diagnosed, so it is a bounded engineering task rather than open research.

**Q87. Who did what?**
*(Fill in honestly for your team of four — system model and channel derivation, planner implementation, experimental campaign and statistics, figures and report. Have a clear one-line answer ready; panels always ask.)*

**Q88. Why is your title "criticality-aware" rather than "priority-aware"?**
Because "priority" in networking usually implies a **weighting** — a soft ordering. **Criticality** here is a **class with a hard floor**: below the floor the collection **does not count**, regardless of how much else you achieved. The title reflects the hard-constraint design.

**Q89. Could this apply outside UAVs?**
The general structure — a **QoS requirement inverted into a geometric feasibility region**, then a **coupled placement-and-configuration problem under a budget** — applies to any mobile-collector or facility-location problem with heterogeneous service requirements: mobile edge servers, base station placement with tiered SLAs, robotic inspection with sensor-dependent standoff distances.

**Q90. What is your defence if someone says the 47-vs-21 result is obvious?**
That it is **obvious only once stated**, which is exactly the report's framing: *"Two observations motivate the work, and they interact in a way that is not obvious until stated together."* The published literature — including a 2025 IEEE TIE paper — fixes altitude at 20 m and optimises routing with an attention model, without noting that the assumption makes an entire class unreachable. The contribution is not that the arithmetic is hard; it is **identifying which constraint binds**, quantifying it, building a planner around it, and then honestly localising the mechanism to the refinement stage.

## 9.13 Questions on aggregation, safety and obstacles

**Q91. Where does the number 70.2% actually come from?**
It is the **mean over the twenty layouts**. For each layout separately we compute (high-priority sensors served at ≥38 Mbps) ÷ (high-priority sensors in that layout) × 100. That yields **twenty numbers**; 70.2 is their mean and ±7.3 is the 95% CI on that mean. Back-solving the CI gives a standard deviation of about **±15.6 points across layouts**, so individual layouts range roughly 55–86%. **But the mean is not the strongest claim — the strongest claim is that we won on 20 out of 20 layouts individually**, which is why the Wilcoxon p is 0.0001. That result is immune to one outlier dragging an average.

**Q92. ⭐ How does the aircraft avoid crashing into buildings when it descends? / Why must the 2D planner clear the tallest building?**

*Part 1 — why 47 m is forced.* It is **definitional, not a collision-avoidance mechanism**. The planar planner holds **one** altitude for the **entire** mission and flies over the whole map, so that single number must be safe **everywhere simultaneously** — hence it must clear the **global maximum** structure plus $h_{\text{safe}}$. There is no mechanism to be at 30 m here and 60 m there; "one constant altitude" is the definition of the method.

*Part 2 — what our method enforces.* Three lower bounds at every hover: the global floor $H_{\min}$ = 20 m; **clearance $z_{\text{anchor}} + h_{\text{safe}}$**, i.e. always ≥10 m above the served sensor; and the footprint bound. So the safety modelled is **local**. The asymmetry in one sentence: **the 2D planner needs one number that is safe globally; ours needs each number to be safe locally.**

*Part 3 — the honest gap.* **There is no en-route collision checking between hovers, for any method.** The vertical energy model flies directly from one hover altitude to the next and charges the net change; nothing verifies that the straight line between consecutive hovers clears the buildings in between. There is no obstacle map in the flight model. Answer it like this:

> *"En-route path clearance is not modelled, for any method. What is modelled is per-hover clearance — ten metres above the served sensor and a twenty-metre floor. The comparison stays fair, because the baseline's forty-seven metres isn't a collision-avoidance mechanism either; it's the arithmetic consequence of holding one altitude over a map with a thirty-seven-metre structure on it. And it doesn't move the headline result, because twenty-one metres is a requirement **at the hover**, not en route. It belongs alongside the propagation simplification as future work."*

*Precision point:* $h_{\text{safe}}$ is applied to the **anchor's** elevation; other members are covered by the footprint bound. Do not assert "10 m above every sensor" if pressed.

**Q93. ⭐ Could the model be given obstacle coordinates and route around them?**

Yes, and the data is already in the right form — but **do not say "the transformer knows the obstacles"**: the reported system is **deterministic**, with no trained model. Frame it for the planner.

The city generator already represents buildings as **axis-aligned boxes** — centre, footprint width and depth, and height — so a segment-versus-box test per tour leg is a constant-time check per building, cheap enough with a spatial index to sit inside the local search. The geometry simply is not currently passed into the planning instance, which sees only per-sensor features.

**The design choice that matters: price blocked legs, do not forbid them.**

| | Forbid blocked routes | **Lift and charge (recommended)** |
|---|---|---|
| Energy landscape | Discontinuous — a small cluster change flips a leg valid/invalid | **Continuous** — obstruction becomes extra joules |
| Local search | Breaks | Works unchanged |
| Failure mode | A cluster ordering can become infeasible with no fallback | Always feasible; you can always fly over |
| Effect on plan | Blunt | Clusters **migrate toward clear corridors**, because obstructed ones cost more |

Concretely: if a leg's corridor is obstructed, lift that leg to $H_{\text{cruise}}(k) = \max(H_k,\, H_{k+1},\, \max_b h_b + h_{\text{safe}})$ over the buildings in the corridor, and charge the climb and descent through the **existing asymmetric vertical term** (≈39 J/m up, 4 J/m down). That keeps energy a continuous function of the clustering, which is what local search needs.

**If challenged with "but you argued hard constraints beat soft penalties!":**
> *"QoS is a **correctness** requirement — below the floor the collection has failed, and no gain elsewhere should buy that. Obstacle clearance en route is a **cost** question — you can always fly over a building, it just costs joules. Hard constraints belong on correctness; costs belong in the objective."*

**Predict the direction of the effect — this is the credibility move.**
> *"I'd expect it to **cost us, not help us**. The 2D baseline pays nothing, because it already cruises above everything. We are the method flying low — mode at 28 m, 86% of hovers below 40 m — so ours are the legs that would need lifting. It would narrow our margin, and by how much is an empirical question we haven't answered."*

For a future **learned** version: encode buildings as a coarse height-map or as extra tokens the encoder attends over, and enforce feasibility by **masking** infeasible transitions in the decoder — never by a reward penalty. Attention decoders already mask visited nodes, so this extends an existing mechanism; masking *guarantees* feasibility, a penalty only makes it likely.

---

# PART 10 — PRESENTATION PLAN

## 10.1 Suggested slide sequence (12–15 slides)

| # | Slide | Content | Time |
|---|---|---|---|
| 1 | Title | Title, four names, supervisor, institute | 15 s |
| 2 | **The setting** | Fig. 1.1. 500 sensors, 1 km², drone from a depot. Why aerial collection at all (the three options). | 1 min |
| 3 | **Sensors are not interchangeable** | Hospital generator vs park soil probe. Serve-everything treats them identically. | 45 s |
| 4 | ⭐ **The binding constraint** | **47 m vs 21 m.** Build it: buildings force 47; Shannon-inverted 38 Mbps permits 21. *"The failure is geometric, not computational."* **This is your money slide — spend time here.** | 2 min |
| 5 | **From QoS to geometry** | Eq. 3.2 derivation, Table 3.1 (three classes → three radii). Note the non-linearity: +31% rate → −65% radius. | 1.5 min |
| 6 | **The governing tension** | Coverage (Eq. 3.3) pulls up ↑, service radius (Eq. 3.2) pulls down ↓. Resolution differs at every hover. | 1 min |
| 7 | **Why hovering is expensive** | $P(0)$ is the maximum. Hence waiting is costly, hence you must descend. Plus the 10:1 climb asymmetry. | 1 min |
| 8 | ⭐ **The evaluation regime** | Serve-everything → satisfaction identically 1 → degenerate metric. Fix: budget = 0.65 × full 2D = 82.65 kJ. **Energy is a control, quality is the outcome. No energy claim is made.** | 1.5 min |
| 9 | **The method** | Fig. 3.1(b) pipeline + Eq. 3.7 + the emergent grouping insight. | 2 min |
| 10 | **Baselines** | Planar (faithful, unre-tuned) and decoupled 3D (same altitude freedom, differs only in ordering). | 1 min |
| 11 | **Headline results** | Fig. 4.1 + Table 4.1. Lead with high: 0.6 → 70.2, 20/20 layouts. | 1.5 min |
| 12 | **vs the decoupled baseline** | Table 4.3. All three classes; energy n.s. And the 52.7% story — repair pass eats the budget. | 1.5 min |
| 13 | ⭐ **The ablation** | Table 4.4. "We ran this to confirm our claim; it refuted it." Coupling +0.00, p=1.000. Refinement supplies everything. **Present as a strength.** | 1.5 min |
| 14 | **Behaviour** | Fig. 4.4 + Fig. 3.2. Unimodal at 28 m, 0% above 60; decoupled bimodal 68/38. Descends on demand, climbs back. | 1 min |
| 15 | **Limitations & the lesson** | Six limitations in one glance; then close on §5.4: *"ask whether the measured quantity is capable of taking a different value."* | 1.5 min |

## 10.2 Delivery advice

- **Lead with the contradiction, not the method.** 47 vs 21 in the first two minutes. Everything after that is *"here is what we did about it."*
- **Own the ablation.** Do not let it be discovered. Announce it: *"We ran an ablation to confirm our headline. It refuted it, so we changed the claim."* That single move converts your biggest vulnerability into your strongest credibility signal.
- **Never claim energy savings.** Not once, not casually. If asked, decline the claim explicitly.
- **Distinguish "unserved" from "violated"** every time it comes up — it is the most commonly misread thing in your results.
- **Say "geometric, not computational"** whenever the 2D failure is discussed.
- **If you don't know something:** say *"That isn't something we evaluated — it falls under [named limitation]. What we can say is…"* and pivot to what you did measure. Panels reward accurate scope far more than confident guessing.
- **Have Table 4.1, Table 3.1 and Table 3.2 memorised.** You will be asked to recall a number without slides.

## 10.3 The five sentences to have word-perfect

1. *"A fixed cruise altitude must clear the tallest structure, which here forces approximately 47 metres, while inverting the rate expression shows the most demanding class can only be served from within about 21. The two cannot both be satisfied."*
2. *"The planar baseline serves 0.6% of its critical sensors not because its route is poor but because no route would serve them."*
3. *"Energy is an experimental control, not an outcome. No energy-saving claim is made anywhere in this report."*
4. *"What this work demonstrates is altitude refinement under quality constraints, not joint decision-making as such."*
5. *"In each instance the error was found by asking whether the measured quantity was capable of taking a different value, rather than by asking whether the number looked reasonable."*

---

# APPENDIX — GLOSSARY

| Term | Meaning |
|---|---|
| **AoI** | Age of Information — staleness of the freshest received update |
| **Ablation** | Removing one component of your own method to isolate its contribution |
| **Anchor** | The first, seed sensor around which a cluster is grown in Stage 2 |
| **CDF** | Cumulative distribution function — fraction of samples ≤ each x value |
| **CMDP** | Constrained MDP — maximise reward subject to bounds on auxiliary costs |
| **Confounding** | When two effects vary together so their contributions cannot be separated |
| **Coupled** | Deciding placement and altitude **together** |
| **Decoupled** | Deciding placement **first**, then altitude — the natural engineering response |
| **Footprint** | Ground circle covered by the antenna cone, radius $(H-z)\tan\theta$ |
| **Greedy set cover** | Repeatedly pick the set covering the most uncovered elements; $\ln n$ approximation |
| **Hard constraint** | Violating solutions are inadmissible — cannot be traded away |
| **KKT** | Karush–Kuhn–Tucker — first-order optimality conditions with inequality constraints |
| **Lambert-W** | Inverse of $we^w$; arises when a variable appears both inside and outside a log |
| **Local optimum** | No single move in the move set improves it; not necessarily globally best |
| **LoS** | Line of sight — unobstructed direct ray |
| **Myopic** | Making a locally-good choice that cannot be revised later |
| **NP-hard** | No known polynomial-time algorithm; TSP and set cover are both NP-hard |
| **Paired test** | Compare methods on identical instances, analysing the differences |
| **Path-loss exponent α** | Rate at which received power decays with distance ($1/d^\alpha$) |
| **Pointer network** | Neural model that outputs a permutation over its own variable-length input |
| **REINFORCE** | Foundational policy-gradient RL algorithm with a variance-reducing baseline |
| **Satisfaction** | Fraction of a class served at or above its rate floor when the budget ran out |
| **Service radius $d^{\max}$** | Maximum aircraft–sensor distance at which the class's rate floor is achievable |
| **Shannon capacity** | $R = B\log_2(1+\mathrm{SNR})$ — max error-free rate |
| **SNR** | Signal-to-noise ratio |
| **Soft penalty** | Constraint added to the objective — tradeable, hence unsuitable for criticality |
| **2-opt** | Local search for TSP: remove two edges, reconnect the other way |
| **Unserved** | Budget never reached it — **not** a constraint violation |
| **Wilcoxon signed-rank** | Non-parametric paired test; no normality assumption |
| **WPT** | Wireless power transfer — used in the base framework, not in ours |
