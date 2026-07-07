# What We Will Do, Our Contribution, and How the Comparison Looks

*Simple terms. The numbers in the tables are **illustrative examples** to show the shape of the
result — not measured yet.*

---

## 1. What we will do

Drones fly over a field of IoT sensors. Each drone flies to a spot, **wirelessly charges** the
sensors there, **collects their data**, and moves on — until every sensor is done.

The twist: **sensors have priority.** Fire/gas = high, air-quality = medium, logging = low.

Our plan, step by step:
1. Label each sensor with a priority (high / medium / low).
2. An **AI planner** decides which drone goes where, in what order, and **at what height**.
3. **Critical sensors → the drone dives low** (strong link → fast, high-quality, fresh collection).
4. **Unimportant sensors → the drone stays high** and sweeps up several cheaply.
5. We measure **battery energy** vs **how well critical data was served**, and compare against
   simpler systems.

---

## 2. What our contribution is

> **We make drone height and wireless-charging respond to how *important* each sensor is, and we let
> an AI plan it for many drones — then we show the trade-off between energy and critical-data
> quality.**

Why it's new:
- Other "priority" drone papers fly at **one fixed height** → they can't dive for critical sensors.
- Other "3D height + charging" papers **ignore which sensors matter**.
- **We are the first to combine the two** (priority + height + charging + a learned planner).

In one line: **critical data gets served well, at much less energy than treating every sensor as
critical — and a fixed-height system simply can't do this.**

---

## 3. How the comparison will look (example)

We compare three systems on the **same** sensor field:

| System | What it does | Total energy | All data collected | **Critical sensors served well** |
|---|---|---|---|---|
| **2D (old/baseline)** | one fixed height, visits every sensor | 100 (high) | 100% | 100% — but only by flying low for *everyone* |
| **3D blind** | varies height, ignores priority | **70 (low)** | 100% | 65% — critical sensors often served too high |
| **3D priority (ours)** | dives for critical, high for the rest | **75 (low)** | 100% | **98%** |

*Read it like this:* the **2D** system does meet quality, but it wastes energy flying low for every
sensor. The **3D blind** system saves energy but **fails many critical sensors** (it doesn't know
they matter). **Ours** keeps critical quality near-perfect **at almost the same low energy** — the
best of both. (Energy shown on a 0–100 scale for easy reading.)

### The trade-off curve (our headline)

If we *demand* stricter quality for critical sensors, the drone must dive more, so energy rises:

| Critical-quality demand | Energy | Critical sensors served well |
|---|---|---|
| Relaxed | 70 | 80% |
| Medium | 75 | 90% |
| Strict | 82 | 98% |

This **energy ↔ quality** curve is a "knob" our system has and a fixed-height system does not.

### The picture

A plot of one drone's path will show its **height going down over critical sensors and back up over
unimportant ones** — the trajectory literally undulates by importance.

```
height
  high |   ___              ___
       |  /   \            /   \        <- stays high over low-priority clusters
       | /     \   /\     /
   low |        \_/  \___/              <- dives for critical sensors
       +--------------------------------> route
```

---

## 4. Honest note

We are **not** claiming "3D saves energy" (we tested that — it doesn't). We claim: **height, driven
by sensor priority, lets the system serve critical data better for the same energy budget**, and we
map that trade-off. This stands on physics that held up under testing and reuses the system we
already built.
