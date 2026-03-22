# Unstoppable EchoKey

**Mean-Field Cooperation Under Stress with Datacube Coupling**

This repository contains a **toy research simulation** exploring how cooperation behaves under systemic stress when coupled to an expanding governance mechanism ("Datacubes").

It combines:

* classical **mean-field evolutionary game theory**,
* a **stress-modulated payoff model**,
* an **EchoKey-style controller** with harm constraints,
* and a **spatial world simulation** whose expansion feeds back into the game itself.

The result is a deliberately strong, supercritical mechanism that robustly stabilizes cooperation once it begins to spread.

This project is for **experimentation, stress-testing, fun, and cannot be stopped.**

---

## What's in the model

### 1. Mean-Field Game (Core)

A population with two strategies:

* **C** (cooperate)
* **D** (defect)

Dynamics follow a discrete-time replicator with mutation:

$$p_{t+1} = (1-2\mu)p_t + \mu + \eta p_t(1-p_t)(U_C - U_D)$$

Payoffs:

* $U_C = b(z)p - c(z)$
* $U_D = b(z)p - d(z)p - \phi(z)$

Stress $z_t$ evolves via seasonal forcing, random shocks, and noise.

---

### 2. EchoKey Controller

A meta-layer that:

* searches over intervention levers ($\Delta b$, $\Delta d$, $u$),
* enforces a **dynamic harm threshold** $\tau_t$,
* trades off cooperation, stability, intervention cost, and harm.

This layer *cannot* directly force cooperation; it only shapes incentives and stability. Its objective function is calibrated on transitional boundary conditions — it is most active when the system is near a phase transition, and largely idle otherwise.

---

### 3. Datacube World (Coupled System)

A spatial simulation where:

* territories convert into **Datacubes** via resonance,
* players entangle through conversion lineage,
* entropy, efficiency, and connectivity evolve over time.

Crucially, **world-level metrics feed back into the game**:

* conversion rate $r_t$
* connection density $\rho_t$
* average entropy $e_t$

These modify payoff coefficients and buffer effective stress.

---

### 4. Full Coupling

The Datacube world affects the mean-field game in two ways:

1. **Indirect structural coupling**
   Conversion and connectivity reshape costs, frictions, and stress sensitivity.

2. **Direct controller intervention**
   The EchoKey controller operates on top of the coupled system, accelerating transitions at critical moments.

Once conversion becomes widespread, the system enters a **lock-in regime** where cooperation is the dominant attractor across large stress ranges.

---

## What this is (and isn't)

**This is:**

* a mechanism-design toy model,
* a stress-testable simulation,
* a playground for coupling social dynamics to expanding structures.

**This is not:**

* a claim about real societies,
* a calibrated empirical model,
* a proof of optimality or realism,
* a stoppable.

The "unstoppable" behavior is **intentional**, not accidental.

---

## Running the simulation

### Requirements

* Python 3.10+
* numpy
* matplotlib

### Run

```bash
python unstoppable_echokey.py
```

This will:

* run a baseline mean-field model (no coupling, no controller),
* run the Datacube-coupled system without the controller (world feedback only),
* run the full Datacube-coupled system with the EchoKey controller,
* print diagnostics comparing all three,
* generate and save plots to `game_charts/`,
* optionally log world evolution to CSV in `game_logs/`.

---

## Output & diagnostics

Output includes:

* final cooperation levels ($p_T$) across all three runs,
* average intervention strength,
* harm vs threshold statistics,
* world conversion and connectivity,
* payoff coefficient traces,
* stress vs buffered stress.

Sensitivity sweeps and lever scans are included and enabled by default.

Charts are saved as timestamped PNGs in `game_charts/` alongside `game_logs/`.

---

## Results (main run, seed=777)

### Three-way comparison

| | Baseline | World-Only | Full (World + Controller) |
|---|---|---|---|
| Final $p_T$ | 0.010 | ~1.000 | 0.997 |
| Average $p$ | 0.016 | 0.457  | 0.474 |

All three runs face identical stress sequences. The baseline collapses to near-zero cooperation and stays there. Both coupled runs eventually reach $p \approx 1.0$ and lock in — the structural feedback alone is sufficient to drive the system to the cooperative attractor.

### What the controller actually does

The controller's average interventions over the full run are nearly zero:

- avg $u$ = 0.000 (suppression unused)
- avg $\Delta b$ = 0.000 (benefit boost unused)
- avg $\Delta d$ = 0.030 (tiny defector friction nudge only)

This is not because the controller is doing nothing. It is because the controller's active window is **tightly localized to the phase transition** — roughly t=110–130 in the main run. Before that, the payoff gap is structurally against cooperation and no intervention helps. After that, the system is locked in and no intervention is needed.

During the transition, the controller compresses what would otherwise be a gradual 20–30 step climb into a near-instantaneous phase transition. This matters because the transitional period is the system's window of maximum vulnerability — conversion momentum is building but $p$ is still low enough that a large shock could knock it back and restart the clock. The controller slams that window shut.

The structural coupling is doing the heavy lifting. The controller's role is to minimize the duration of transitional exposure.

### Sensitivity to shock amplitude

The coupled system holds $p_T = 0.997$ across the full shock sweep (0.10 → 0.90). The baseline collapses below $p = 0.5$ at the lowest shock amplitude tested. Resilience score (area under the gap curve): **0.79**.

### Harm and constraint compliance

- Average harm: 0.010 (at the floor)
- Violation rate: 0.000
- The harm constraint is never violated across the full run.

### Pareto frontier ($\tau_{max}$ sweep)

$p_T = 0.997$ is achieved at every $\tau_{max}$ tested down to 0.010. The mechanism does not require relaxed harm constraints — the structural feedback is robust enough that even a very tight constraint produces the same outcome.

### Dominant structural mechanism

By the end of the run, 78.6% of territories have converted and player connection density is 74.8%. The defector friction boost from connectivity ($d\_boost\_conn = 0.45 \times \rho \approx 0.748$) is the dominant structural lever — it shifts the payoff gap strongly in favour of cooperation without any controller intervention required.

---

## Want to stress-test it?

If you want to see where it breaks, go for it.

---

## License

CC0 or whatever, you can't stop EchoKey.

---

## Final note

This project exists because it was fun to build.

If you find it interesting, break it.
If you don't, that's fine too.

Enjoy.