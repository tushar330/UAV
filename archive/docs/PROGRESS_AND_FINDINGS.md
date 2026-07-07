# Project Progress Report

**Project:** Energy-efficient data collection from IoT sensors using multiple drones
**Date:** 18 June 2026
**Prepared for:** Mentor review

---

## In one paragraph

We set out to show that letting drones **change their flying height** (going fully 3D) would let
them collect sensor data using **less battery energy** than drones locked at a single height. We
built the complete system and tested this carefully. The honest finding is that **changing height
does not save energy** — in every realistic setting, the most energy-efficient choice is simply to
fly **low**. This is a clean, defensible result with a clear physical reason. However, height *does*
matter for a different and equally useful goal: **balancing energy against communication quality**.
We recommend shifting the project's headline to that balance, which keeps all the work we have done
and rests on solid ground.

---

## 1. What we were trying to do

Picture several drones flying over a field of hundreds of small IoT sensors. Each drone flies to a
sensor, **wirelessly charges it**, **collects its data**, moves to the next, and finally returns to
base. The objective is to finish collecting **everyone's** data using the **least total battery
energy**.

The starting idea (our intended novelty): instead of all drones flying at one fixed height, let each
drone **choose its height intelligently**. The reasoning was:

- Fly **high** → the drone's antenna covers a wider circle of ground → it can serve many sensors from
  one spot → less flying.
- Fly **low** → the signal is stronger → wireless charging is faster.
- We expected a "best height in the middle" that saves energy — something a fixed-height system could
  never find.

---

## 2. What we built

We implemented the entire system end-to-end and got it running:

- A simulation of the sensor field and the drones.
- The wireless-charging and data-collection physics.
- A realistic **drone power model** (how much battery a real drone burns while hovering vs. flying).
- An **AI planner** (the same attention-based learning method used in the paper we are extending)
  that decides which drone visits which sensors, in what order, and at what height.
- All the tools to train the planner and measure energy, plus comparison baselines.

The software works correctly. The issue we ran into is about the **science**, not the code.

---

## 3. What we found (and how we checked it carefully)

We did not simply assume our idea worked — we tested it with **fair comparisons**, fixing two
mistakes that would have flattered our method:

1. **A fair baseline.** Our first result ("3D saves 62% energy") was misleading: the comparison
   drone was poorly trained and flew a needlessly long route. When we gave the comparison its best
   realistic route, the gap shrank dramatically.

2. **Realistic drone physics.** Our first energy model wrongly made **hovering cheaper than flying**.
   For a real drone, **hovering is the most expensive thing it does**. After fixing this, the picture
   changed completely.

**The key result.** With realistic physics and fair routing, we measured total energy at different
flying heights (every sensor still served):

| Flying height | Low (20 m) | 40 m | 60 m | 80 m | High (140 m) |
|---|---|---|---|---|---|
| Total energy | **lowest** | higher | higher | higher | **highest** |

Energy only **goes up** as the drone flies higher. The best choice is to **fly low**. Letting the
drone vary its height made things **worse**, not better. And a low-flying 3D drone used essentially
the **same** energy as the simple fixed-height (2D) system — so the 3D idea gave **no energy saving**.

We saw the same conclusion in every setting we tried. (At one point a "best middle height" appeared,
but we traced it to a routing mistake; once corrected, it vanished.)

---

## 4. Why this happens (the physical reason)

Most of the energy goes into **wireless charging**, and wireless charging works best when the drone
is **close** to the sensor. So the physics always pulls the drone **downward**. Flying is cheap, so
covering many sensors at once from high up — the only reason to fly high — barely helps.

This also matches existing research: studies that vary drone height do it to improve **sensing or
communication quality**, never to save propulsion energy. In short: **saving energy alone does not
justify changing height.**

So our original claim ("3D height saves energy") is **not defensible**, and we should not build the
thesis on it. This is a genuine, useful negative result — it tells us where *not* to look.

---

## 5. What is still valuable

- A **complete, realistic, working** multi-drone 3D simulation and AI planner.
- A clean, honest finding: for pure energy, low flight is best and 3D gives no advantage.
- An important observation: when we **require better communication quality** (a minimum data rate per
  sensor), the best height **changes** — the drone is forced lower, and energy rises. **Here height
  genuinely matters** — not to save energy, but to **trade energy for quality.**

---

## 6. Proposed direction (for your guidance)

**Shift the headline from "height saves energy" to "height balances energy against communication
quality."**

> *Letting drones choose their height lets a multi-drone IoT system trade total energy against
> communication quality — something a fixed-height system cannot do. We map out this trade-off and
> show our AI planner handles it better than fixed-height baselines.*

Why this is a good move:
- It stands on **physics that does not collapse** under testing: demanding higher quality forces the
  drone lower (more energy); relaxing it allows higher, cheaper flight. Height genuinely varies.
- It **reuses everything we have already built** — only the headline experiment and framing change.

**Other options, if you prefer:**
- **(B) Add obstacles / terrain** (e.g. buildings) so drones *must* change height to fly over them —
  this makes 3D planning genuinely necessary.
- **(C) Present the honest negative result** as the contribution (least novel, fully honest).

**Our recommendation:** go with option (A), the **energy-vs-quality trade-off**, optionally combined
with (B) for a stronger "true 3D flight" story.

---

## 7. The one question we'd like your input on

Do you agree we should pivot to the **energy-vs-communication-quality** framing (option A), or would
you prefer we pursue the **obstacles/terrain** direction (option B), or something else? We will
proceed once you advise.
