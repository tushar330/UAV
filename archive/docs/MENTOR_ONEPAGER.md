# BTP Update — From "3D Saves Energy" to Priority-Aware 3D Data Collection

**Project:** Energy-efficient multi-UAV data collection from IoT sensors
**For:** Mentor review · One-page summary

---

## Where we were

Our original goal was to show that letting drones **change altitude (full 3D)** would collect all
sensor data using **less battery energy** than a fixed-height system. We built the complete system
and tested it honestly. **Finding: changing altitude does not save energy** — wireless charging is
fastest close to the sensor, so the optimal choice is always to fly low. This is a clean, defensible
*negative* result, but it cannot be the project's headline.

What *did* survive testing: **altitude is a genuine lever for communication quality** — demanding a
higher data rate forces the drone lower (costing energy). Altitude trades **energy against quality**,
not energy alone.

## What we changed (the pivot)

We keep the entire working system and re-aim it at a problem where altitude clearly matters:
**priority-aware data collection.** Real sensors are not equal — a fire/gas sensor is critical, a
logging sensor is not. Our drones now serve **critical sensors with a stronger link, lower, and
earlier**, while sweeping up unimportant sensors cheaply from higher up. **The drone's altitude
rises and falls according to each sensor's criticality.**

## Our novelty (vs. previous work *and* the literature)

We checked the literature carefully:
- **Priority/freshness + multi-UAV data collection** exists — but those works fly at **fixed
  altitude** and use classical heuristics (k-means + genetic algorithms).
- **3D-altitude + wireless-charging** works exist — but **none use sensor priority**.

> **Our contribution:** the first to make UAV **altitude and wireless-charging schedule respond to
> sensor priority**, learned by an attention-based multi-UAV planner, and to map the resulting
> **energy ↔ priority-quality trade-off** — a coupling neither the fixed-altitude priority literature
> nor the priority-blind 3D-WPT literature provides.

In plain terms: we guarantee **critical data gets high-quality, fresh service at far lower energy**
than a system that must treat every sensor as critical — something a fixed-height (2D) system cannot
do, because it can't selectively dive.

## What we built (≈80% reused from the existing system)

- **New:** sensor priority classes + per-class quality floors; a priority-weighted reward
  (energy + penalty for critical sensors served late or below quality); priority-aware workload
  splitting across drones; the encoder now *sees* priority so it can plan to dive for critical nodes.
- **Reused:** the attention planner, wireless-charging physics, realistic drone power model, the
  3D coverage/altitude model, and all comparison baselines.
- Status: fully implemented, integrated, and verified to run end-to-end; ready for the full
  training run on GPU.

## What we will show (results plan)

1. **Headline table:** priority-3D keeps **critical-QoS satisfaction high at lower energy** than the
   2D baseline; a priority-blind 3D model scores worse on critical-QoS.
2. **Energy ↔ priority-quality frontier:** as we demand stricter critical-quality, energy rises — a
   trade-off curve a fixed-height system cannot navigate.
3. **Trajectory figure:** the drone's altitude visibly dives for critical sensors and stays high
   over unimportant ones.

## Honest framing

We do **not** claim "3D saves energy" (that did not hold). We claim: **altitude, driven by sensor
priority, lets a multi-UAV system serve critical data better for a given energy budget** — and we
characterise that trade-off with a learned planner. This rests on physics that survived testing and
reuses everything we have already built.

**One question for you:** does this priority-aware framing work as our headline contribution, or
would you prefer we also add physical obstacles (so 3D flight becomes strictly necessary) alongside it?
