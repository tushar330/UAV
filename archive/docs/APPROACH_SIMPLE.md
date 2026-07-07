# Our Approach in Simple Terms — 3D-AUTO

*A plain-language guide to what the two base papers did, what we take from each, and what makes
our work novel. For the precise math, see `PROBLEM_FORMULATION.md`. This doc trades rigour for
intuition and worked examples.*

---

## 1. The story in one paragraph

Drones (UAVs) fly over a field of hundreds of tiny IoT sensors. Each drone first **wirelessly
charges** a sensor, then **collects its data**, then flies to the next one, and finally returns to
base. We want to **finish collecting everyone's data using as little battery energy as possible**.
Paper A solved this in **2D** (all drones at one fixed height). Paper B showed how a single drone
can fly in **full 3D** (changing height). **We merge them**: many drones, in 3D, where *height
itself becomes a smart decision* — and we add the one piece both papers ignored: **the energy cost
of climbing up and down.**

---

## 2. What Paper A did (AUTO — our main framework)

**Setup:** `M` drones, `N` ground sensors, one base station ("data center"). Every drone flies at a
**fixed height** (e.g. 20 m). It charges a sensor (WPT = Wireless Power Transfer), waits for it to
upload its data, moves on, and comes back to base.

**Goal:** minimize the **total energy of all drones**.

**Their energy formula (simplified):**
```
Energy of one drone =  flight energy  +  charging energy  +  data-collection energy
       E_j           =     P_F · T_F   +  (P_T+P_H)·T_E   +   (P_C+P_H)·T_C
```
- `P_F·T_F` — **flying**: power `P_F` times flight time `T_F`. Flight time = distance ÷ speed.
- `(P_T+P_H)·T_E` — **charging**: transmit power + hover power, during charge time `T_E`.
- `(P_C+P_H)·T_C` — **collecting**: collect power + hover power, during collect time `T_C`.

> **Example.** A drone flies 400 m at 10 m/s → `T_F = 40 s`. With `P_F = 75 W` → flight energy
> `= 75 × 40 = 3000 J`. If it then hovers 8 s to charge+collect at `P_H ≈ 50 W` → ~400 J more.

**Their clever trick #1 — solve the timing exactly.** How long to charge (`T_E`) and collect (`T_C`)
has a neat closed-form answer (KKT / Lambert-W, their eqs. 17–18). So they *don't* learn the
timing; they compute it.

**Their clever trick #2 — let an AI plan the route.** Choosing *which drone visits which sensor in
what order* is a hard routing puzzle (a "capacitated vehicle routing problem", NP-hard). They built
**ATOM**: a graph **attention** network (a transformer) that looks at all sensors at once and a
**decoder** that outputs the visiting order. They train it with reinforcement learning (**TENMA**).

**Paper A's blind spot:** everything is at **one height**. No notion of flying higher or lower.

---

## 3. What Paper B did (UAV-ISAC — our 3D toolkit)

**Setup:** **one** drone doing radar sensing + communication in **full 3D** — its position is
`(x, y, H)` and `H` (height) changes over time.

**The two ideas we borrow from it:**

**(a) Height controls coverage — the "flashlight cone" (their eq. 23e).** A drone's antenna points
down like a flashlight with a fixed beam angle `θ`. The patch of ground it can reach is a circle
whose **radius grows with height**:
```
coverage radius  r = H · tan(θ)
```
> **Example (θ = 60°, so tan θ ≈ 1.73... we use half-angle, tan 30° ≈ 0.58).**
> At `H = 30 m` → `r ≈ 0.58 × 30 ≈ 17 m` (covers a tiny patch, maybe 1 sensor).
> At `H = 130 m` → `r ≈ 0.58 × 130 ≈ 75 m` (covers a big patch — several sensors at once!).

**(b) Realistic 3D movement rules (their eqs. 23l–23r):** limits on horizontal speed, vertical
speed, acceleration, min/max height, and "start and end at the same spot". We reuse these as our
drone's physics.

**Paper B's blind spot for us:** only **one** drone, **no AI planner**, and its goal is *radar
quality* — it **never counts the battery cost of going up and down**.

---

## 4. The catch nobody modeled — and our novelty

Flying higher sounds great (one hover covers many sensors → fewer stops → less flying). But there
are **three hidden costs of height**:

1. **Weaker signal.** Farther away = weaker link = slower charging and slower data collection =
   **more time hovering = more energy.** (Signal strength `|g|² = β / distance²` — double the
   distance, quarter the strength.)
2. **The climb itself costs energy.** Lifting a 2 kg drone up 100 m is real work against gravity —
   **and neither Paper A nor Paper B ever put this in their energy bill.**
3. **It might be *too* far to charge/collect at all** within a sensible time budget.

**This is our novel contribution: the altitude-change energy term.** We add to the energy bill:
```
Vertical energy  E_vert = (m·g / η) · (metres climbed)  +  c_d · (metres descended)
```
- `m·g` = weight of the drone (mass × gravity). `η` = motor efficiency. So climbing 1 m costs about
  `2 × 9.81 / 0.5 ≈ 39 J`. Climbing 100 m ≈ `3900 J` — comparable to a 500 m flight!
- Descending is cheap (`c_d` small) — you can't get the energy back, but it costs little.

> **Why this makes height a *real* decision (not just "always fly low" or "always fly high"):**
> - **Fly low:** strong signal (cheap charging), cheap height — **but** tiny coverage → you must
>   visit nearly every sensor one-by-one → **lots of flying**.
> - **Fly high:** huge coverage → one hover serves many sensors → **little flying** — **but** weak
>   signal (slow, expensive charging) + you paid to climb up.
> The cheapest total-energy answer is **somewhere in between**, and it's *different for different
> sensor layouts*. A 2D system is stuck at one height and can never find it. **That gap is our win.**

### 4a. Worked example — how a hover decides *how many* nodes it serves

A natural question: *"where does the planner decide 'serve 5 here, serve 8 there'?"* It never
decides a **number** directly. At each hover the policy picks only **two things**:

1. an **anchor** — *where* to hover `(x, y)` (a pointer over the remaining sensors);
2. an **altitude** `H` — *how high*.

The count then falls straight out of the geometry — every still-unserved sensor inside the cone
`ρ ≤ (H − z)·tan θ` (and with a fast enough link, `R ≥ R_min`) is served:

> **Same anchor, two altitudes (θ = 60°, tan θ ≈ 1.73, flat ground z = 0).**
> Suppose 8 sensors lie within 220 m of the anchor, the nearest 5 within 130 m.
> - **Fly to `H = 75 m`** → radius `r = 75 × 1.73 ≈ 130 m` → **5 sensors** fall inside → serve 5.
> - **Fly to `H = 130 m`** → radius `r = 130 × 1.73 ≈ 225 m` → **all 8** fall inside → serve 8.
>
> So *"serve 5 vs serve 8"* **is** the altitude choice. Higher = more sensors per stop (fewer
> stops, less flying) but a weaker link (slower charging) plus the climb cost; lower = the reverse.

**A second gate — the time budget `τ_max`.** Even if 8 sensors sit inside the cone, each one needs
charge + collect time (the KKT formula, driven by *that sensor's own data demand `D_i`* — which the
network already knows as an input feature). We add sensors closest-first until the hover's total
time hits `τ_max ≈ 30 s`; any that don't fit roll over to a later hover. So the **effective** count is
`min( sensors in the cone , how many fit in 30 s )`. A few data-heavy sensors can therefore fill the
budget as fast as many light ones.

**What the policy cannot (yet) do:** it serves *all* in-cone sensors greedily — it can't choose to
cover only 5 of the 8 in range. Its only levers on the count are **anchor placement** and
**altitude**. That keeps the action space small (one sensor + one number); a per-sensor "skip"
action would be the natural extension.

---

## 5. How we combine everything (the ATOM-3D pipeline)

```
Sensors (x, y, z, data) ──► GRAPH ENCODER (Paper A's attention) ──► smart features
                                                   │
                                                   ▼
                              3D DECODER: pick next sensor + pick a height H
                                                   │  (height is now a learned choice — new)
                                                   ▼
              At each hover: cover everyone inside radius r=(H−z)·tanθ, charge+collect them
                                                   │
                                                   ▼
              Add up energy:  flying + CLIMBING (new) + charging + collecting
                                                   │
                                                   ▼
              REWARD = − total energy   (lower energy = better),  must serve ALL sensors
                                                   │
                                                   ▼
                        TENMA reinforcement learning improves the planner
```

**What we keep from each paper:**

| Piece | From | Why |
|---|---|---|
| Attention graph encoder (ATOM) | Paper A | Best way to "look at all sensors at once" |
| Route decoder + multi-UAV split | Paper A | Assigns sensors to drones, respects battery/storage |
| TENMA reinforcement-learning training | Paper A | Stable, generalizes to new layouts |
| WPT charging + exact KKT timing | Paper A | Don't re-learn what has a formula |
| Height → coverage "flashlight cone" | Paper B | Makes height meaningful |
| 3D movement rules (speed/accel/height limits) | Paper B | Realistic physics |
| **Altitude-change energy** | **Neither — ours** | **The missing cost that makes 3D worth it** |

---

## 6. What counts as "success" (our experiments)

- **Headline:** ATOM-3D uses **less total energy** than 2D-ATOM to collect everyone's data.
- **Ablation:** turn off variable height (freeze at one altitude) → energy goes back up → proves
  height is what helped.
- **Ablation:** replace attention with a plain network → worse → proves the attention helps.
- **Pareto curve (energy vs. communication quality):** we set a minimum required signal quality
  `R_min` and slide it. Demanding better quality forces drones lower (more stops) → more energy.
  This curve shows the **price of quality** — a knob a 2D system doesn't have.

---

---

## 6b. The real novelty — "dive to serve" (why altitude actually varies)

We first built the obvious 3D version (let the planner pick any altitude) and **tested whether it
helps**. Surprising result: it didn't. With cheap hovering and an easy radio link, the cheapest plan
is always *"fly as high as you're allowed and cover everyone at once."* The altitude never varied —
so "3D" was cosmetic. We needed a reason for height to **matter**, node by node.

That reason is **dive-to-serve**. Picture the drone cruising high (cheap, fast travel), then
**dropping down** to charge+collect a sensor, then climbing back to cruise. How far it dives is a
real tradeoff:

> **Diving over one sensor (θ = 60°, climb ≈ 39 J/m).**
> - **Dive shallow (stay ~120 m):** covers a wide patch (many sensors at once), *but* the link is
>   weak → wireless charging is slow → you hover a long time at the **expensive** hover power.
> - **Dive deep (~40 m):** the link is strong → charging is fast → little hover energy — *but* you
>   pay to climb back up, and you only covered a small patch (so more stops).
> The cheapest dive depth is **in between**, and — here's the key — it **depends on how much data /
> charge that sensor needs**: a data-hungry sensor is worth diving deep for (charging dominates);
> a light, clustered group is better served shallow in one wide shot.

Because every sensor (and every little cluster) has a *different* demand, the **best dive depth is
different at each stop** → the drone's altitude **genuinely rises and falls along the route, purely to
save energy.** We can even write down the optimal depth `H_s*` in closed form (it's the height twin of
Paper A's charging-time formula). To make this tradeoff real we also use the **honest propulsion
model** where *hovering is the most expensive thing a rotary drone does* (not cheaper than flying, as
our first toy model wrongly assumed) and we let the drone **cruise at its most efficient speed**.

**That** — an energy-optimal, demand-driven service altitude that makes the 3D path actually undulate —
is what neither base paper has, and it's what turns "altitude is a variable" into "altitude is *the*
lever."

---

## 7. One-sentence novelty statement

> *We turn UAV altitude into an energy-optimal **per-service decision**: the drone cruises high but
> **dives to a depth chosen for each sensor's demand** — balancing coverage, link quality, and the
> cost of climbing back — so the 3D trajectory undulates to minimise total energy. We characterise
> this optimal dive depth in closed form (the altitude analog of AUTO's charging-time allocation) and
> learn the multi-UAV policy over it with attention-based RL — a mechanism absent from both the 2D
> fixed-altitude AUTO framework and the single-UAV radar-rate UAV-ISAC trajectory.*
