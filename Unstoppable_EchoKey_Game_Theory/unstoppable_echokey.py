"""
================================================================================
UNSTOPPABLE ECHOKEY: Mean-Field Cooperation Under Stress with Datacube Coupling
================================================================================

OVERVIEW
--------
This file is a self-contained research simulation exploring a central question in
mechanism design and evolutionary game theory:

    Can you engineer a social/institutional structure that robustly stabilizes
    cooperation even under severe and repeated external stress?

The short answer this model gives is: yes — if the structure itself expands and
feeds back into the game in the right way.

The model has four interlocking components:

  1. MEAN-FIELD EVOLUTIONARY GAME: A population of agents choosing between
     Cooperate (C) and Defect (D), evolving under replicator dynamics.

  2. STRESS PROCESS: An exogenous environmental stress signal z(t) that makes
     cooperation harder and defection more tempting over time (seasonal + shocks).

  3. ECHOKEY CONTROLLER: A meta-layer that searches over intervention levers
     (boosting benefits of cooperation, increasing costs of defection, applying
     suppression) to steer the population toward cooperation, subject to a
     harm constraint.

  4. DATACUBE WORLD: A spatial simulation where players own territories that
     can be converted into high-efficiency "Datacubes." World-level metrics
     (conversion rate, connection density, entropy) feed back into the payoff
     structure, making cooperation progressively easier as the Datacube network
     expands. This is the "unstoppable" part — once critical mass is reached,
     the system locks in.

The model is a TOY. It is not calibrated to real data and makes no empirical claims.
It is a mechanism-design sandbox: the question is not "is this realistic?" but
"does this mechanism do what we think it should?"

DESIGN PHILOSOPHY
-----------------
The conventional game-theoretic problem is the Prisoner's Dilemma under stress:
when payoffs are stressed (resources scarce, shocks frequent), defection becomes
individually rational and cooperation collapses. Standard solutions involve either
top-down enforcement or reputation/iteration — both of which require exogenous
conditions that may not hold.

This model explores a third path: structural coupling. If an expanding network
(the Datacube world) systematically:
  - buffers effective stress (makes shocks less damaging)
  - increases the benefit of cooperation (network effects)
  - reduces the cost of cooperation (efficiency gains)
  - increases the friction of defection (density-enforced norms)

...then the game itself changes as the structure expands. The EchoKey controller
doesn't need to force cooperation — it just needs to hold the line long enough
for the structural feedback to take over.

The "unstoppable" behavior is the claim that once Datacube conversion exceeds a
threshold, the structural feedback dominates and cooperation becomes the
globally stable attractor, independent of controller interventions.

KEY QUANTITIES TRACKED
----------------------
  p(t)       : fraction of cooperators in the population at time t
  z(t)       : raw environmental stress at time t (exogenous)
  z_eff(t)   : effective stress after world buffering (z_eff <= z always)
  r(t)       : Datacube conversion rate (fraction of territories that are Datacubes)
  rho(t)     : connection density (how interconnected are players via entanglement)
  e(t)       : average entropy across territories (higher = more disorder)
  U_C, U_D   : payoffs for cooperators and defectors
  delta_b    : controller's benefit-boost intervention lever
  delta_d    : controller's defection-friction-boost intervention lever
  u          : controller's suppression lever
  h(t)       : harm score (cost of intervention)
  tau(t)     : dynamic harm threshold (how much harm is tolerable right now)
  J          : controller objective function (what gets minimized at each step)

PARAMETER CALIBRATION RATIONALE (main run)
------------------------------------------
  T=220      : long enough to see world conversion play out (conversion takes ~50-100 steps)
  p0=0.22    : start with a small cooperator minority (pessimistic IC)
  eta=0.70   : moderately fast replicator — population responds quickly to payoff gaps
  mu=0.002   : small mutation keeps the system from getting stuck at boundaries
  shock_amp  : 0.30 (default) is moderate; sensitivity sweep goes up to 0.90
  tau_max    : 0.010 in main run (tight harm constraint — controller is cautious)
  Lambda=18  : heavy penalty for harm violations (enforces constraint strictly)
  grid_n=19  : controller search grid (19^3 = ~6859 candidates per step)

OUTPUTS
-------
  - Baseline vs coupled cooperation trajectory plots
  - Stress buffering visualization
  - World conversion and connectivity over time
  - Controller intervention traces (delta_b, delta_d, u)
  - Harm vs dynamic threshold
  - Payoff coefficient evolution
  - Sensitivity sweep: cooperation stability vs shock amplitude
  - Lever sweep: 2D heatmap of which structural levers matter most
  - Pareto frontier: harm tolerance vs cooperation performance tradeoff
  - CSV logs in game_logs/ (world state, territory states, player states, conversions)

LICENSE
-------
CC0. Do whatever you want. You can't stop EchoKey.

================================================================================
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SECTION 1: UTILITY FUNCTIONS
# =============================================================================
# Standard mathematical helpers used throughout. Nothing fancy here.

def sigmoid(x: float) -> float:
    """Standard logistic sigmoid. Maps (-inf, inf) -> (0, 1).
    Used in the dynamic harm threshold to create a smooth transition
    between low-tolerance and high-tolerance regimes."""
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    """Hard clamp x to [lo, hi]. Used everywhere to prevent
    payoff coefficients and probabilities from going out of valid range."""
    return max(lo, min(hi, x))


def euclid3(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Euclidean distance in 3D integer grid space.
    Used in the Datacube world to compute influence radii and
    find nearest Datacube parents for entanglement tracking."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# =============================================================================
# SECTION 2: EXOGENOUS STRESS PROCESS
# =============================================================================
# z(t) is the environmental stress at time t. It is exogenous — the population
# cannot control it. It makes cooperation costly and defection tempting.
#
# The stress signal has three components:
#
#   SEASONAL: A slow sinusoidal wave simulating predictable cyclical pressure
#             (like economic cycles, seasonal resource availability, etc.)
#             The wave is phase-shifted so it starts near 0 and peaks at
#             seasonal_amp (default: 0.08 on top of base 0.10).
#
#   SHOCKS:   Random Bernoulli events that add a large spike (shock_amp).
#             shock_prob=0.06 means each time step has a 6% chance of a shock.
#             At T=220, this gives ~13 expected shocks per run — significant.
#
#   NOISE:    Small Gaussian perturbations (std=0.01) to prevent clean periodicity.
#
# Total z(t) is clamped to [0, 1]. In the main run, z(t) lives roughly in
# [0.10, 0.50] with shock spikes to ~0.48 (base + seasonal + shock).
#
# The stress process is fully reproducible via seed. The same seed gives the
# same shock sequence — useful for comparing baseline vs coupled runs fairly.

@dataclass
class StressParams:
    base: float = 0.10            # Baseline stress level (always present)
    seasonal_amp: float = 0.08    # Amplitude of seasonal oscillation
    seasonal_period: int = 60     # Period of seasonal cycle in timesteps
    shock_prob: float = 0.06      # Probability of a shock at each timestep
    shock_amp: float = 0.30       # Magnitude of shock when it occurs
    noise_std: float = 0.01       # Std of Gaussian noise added each step


def stress_process(T: int, sp: StressParams, seed: int) -> np.ndarray:
    """
    Generate the stress time series z[0..T-1].

    Uses a deterministic RNG seeded with `seed` so that baseline and coupled
    runs face identical shock sequences — the only difference in outcomes is
    due to the structural coupling and controller, not lucky/unlucky stress draws.
    """
    rng = random.Random(seed)
    z = np.zeros(T, dtype=np.float64)
    for t in range(T):
        seasonal = sp.seasonal_amp * (0.5 * (1.0 + math.sin(2.0 * math.pi * t / sp.seasonal_period)))
        shock = sp.shock_amp if (rng.random() < sp.shock_prob) else 0.0
        noise = rng.gauss(0.0, sp.noise_std)
        z[t] = clamp(sp.base + seasonal + shock + noise, 0.0, 1.0)
    return z


# =============================================================================
# SECTION 3: BASE PAYOFF MODEL
# =============================================================================
# The core payoff structure follows a simple public-goods framing.
#
# Two strategies: C (cooperate) and D (defect).
# Population cooperator fraction: p ∈ [0, 1].
#
# Payoffs:
#   U_C = b(z) * p - c(z)
#   U_D = b(z) * p - d(z) * p - phi(z)
#
# COOPERATORS get benefit b*p (the public good scales with how many cooperate)
# but pay cost c (their contribution to producing it).
#
# DEFECTORS get the same public good b*p but pay no cost. However, they face:
#   - friction d*p: peer-based enforcement that scales with cooperator density
#     (more cooperators = stronger norms = more friction on free-riders)
#   - baseline friction phi: a constant cost of being a defector (reputational,
#     institutional, etc.) regardless of how many cooperators there are.
#
# The payoff gap U_C - U_D drives replicator dynamics:
#   U_C - U_D = -c(z) + d(z)*p + phi(z)
#
# For cooperation to be advantageous (U_C > U_D), we need:
#   d(z)*p + phi(z) > c(z)
#
# i.e., defector friction + baseline friction must exceed cooperation cost.
# At high stress z: c goes up (cooperation costs more), d goes up (but so does
# friction on defectors), and phi goes up. Whether cooperation survives depends
# on which effect dominates.
#
# STRESS MODULATION: All four coefficients are linear in z:
#   b(z) = b0 + b_z * z    (b_z < 0: stress reduces cooperative benefit)
#   c(z) = c0 + c_z * z    (c_z > 0: stress makes cooperation more costly)
#   d(z) = d0 + d_z * z    (d_z > 0: stress tightens enforcement)
#   phi(z) = phi0 + phi_z * z (phi_z > 0: baseline defector friction rises)
#
# The base calibration (b0=1.20, c0=0.25, d0=0.55, phi0=0.02) is chosen so that
# cooperation is UNSTABLE at moderate stress without structural support. This is
# the "difficult" regime the Datacube coupling is designed to overcome.

@dataclass
class PayoffParams:
    # Base coefficients at z=0
    b0: float = 1.20          # public benefit multiplier (network value of cooperation)
    c0: float = 0.25          # cooperation cost (direct cost of contributing)
    d0: float = 0.55          # defector friction coefficient (scales with p)
    phi0: float = 0.02        # constant baseline friction for defectors

    # Stress modulation coefficients (how each coefficient responds to z)
    b_z: float = -0.40        # stress erodes cooperative benefit (scarcity logic)
    c_z: float = +0.60        # stress makes cooperation more costly (resource drain)
    d_z: float = +0.30        # stress amplifies enforcement (desperation = harsher norms)
    phi_z: float = +0.10      # stress raises baseline defector friction

    # Minimum clamps (prevent degenerate coefficients)
    b_min: float = 0.10
    c_min: float = 0.01
    d_min: float = 0.00
    phi_min: float = 0.00


def payoff_coeffs(
    z: float,
    pp: PayoffParams,
    delta_b: float = 0.0,
    delta_d: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Compute stress-modulated payoff coefficients (b, c, d, phi).

    delta_b and delta_d are the EchoKey controller's intervention levers:
      delta_b: boosts the cooperative benefit (makes cooperation more attractive)
      delta_d: boosts defector friction (makes defection more costly)

    Both are additive shifts on top of the stress-modulated base values.
    The controller picks these levers at each timestep via grid search.
    """
    b = clamp(pp.b0 + pp.b_z * z + delta_b, pp.b_min, 10.0)
    c = clamp(pp.c0 + pp.c_z * z, pp.c_min, 10.0)
    d = clamp(pp.d0 + pp.d_z * z + delta_d, pp.d_min, 10.0)
    phi = clamp(pp.phi0 + pp.phi_z * z, pp.phi_min, 10.0)
    return b, c, d, phi


def payoffs(
    p: float,
    z: float,
    pp: PayoffParams,
    delta_b: float = 0.0,
    delta_d: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute (U_C, U_D) given current cooperator fraction p and stress z.
    This is the UNCOUPLED baseline — no Datacube world feedback.
    Used in the baseline run for comparison.
    """
    b, c, d, phi = payoff_coeffs(z, pp, delta_b=delta_b, delta_d=delta_d)
    U_C = b * p - c
    U_D = b * p - d * p - phi
    return U_C, U_D


# =============================================================================
# SECTION 4: DATACUBE COUPLING — WORLD METRICS MODIFY PAYOFFS
# =============================================================================
# This is the structural heart of the model. The Datacube world generates
# three time-varying metrics:
#
#   r(t)   = conversion_rate: fraction of all territories that are Datacubes
#   rho(t) = connection_density: how interconnected players are via entanglement
#   e(t)   = avg_entropy: average disorder across territories
#
# These feed into the payoff structure via CouplingParams, modifying the base
# coefficients BEFORE stress is applied. The key effects are:
#
#   STRESS BUFFERING: z_eff = z * (1 - z_buffer_conv * r)
#     As more territories convert, effective stress is reduced. This models
#     the idea that a well-organized cooperative network is more resilient to
#     environmental shocks — it has slack, redundancy, and mutual support.
#     At r=1.0 and z_buffer_conv=0.35, effective stress is only 65% of raw stress.
#
#   BENEFIT UPLIFT: b0_eff = b0 + b_boost_conv*r + b_boost_conn*rho
#     More Datacubes and more connections raise the value of cooperation.
#     Network externality logic: cooperation is worth more in a larger network.
#
#   COST REDUCTION: c0_eff = c0 * (1 - c_reduce_conv*r) * (1 - c_reduce_conn*rho)
#     As conversion spreads, cooperation becomes cheaper. Shared infrastructure,
#     amortized coordination costs, established norms — cooperating is just less
#     effortful in a mature Datacube network.
#     c_z is also dampened: stress is less able to inflate cooperation costs.
#
#   DEFECTOR FRICTION BOOST: d0_eff += d_boost_conn*rho + d_boost_conv*r
#     Denser networks enforce norms more effectively. Defectors are more visible,
#     more sanctioned, and face stronger peer pressure in high-density environments.
#
#   BASELINE DEFECTOR FRICTION BOOST: phi0_eff += phi_boost_conv*r + phi_boost_conn*rho
#                                                 + phi_boost_entropy*e
#     Conversion and connectivity increase baseline costs of defection.
#     Entropy adds an interesting twist: high-entropy environments are uncertain,
#     and opportunistic defection is more costly when outcomes are unpredictable
#     (defectors can't reliably exploit the system when the system is noisy).
#
# LOCK-IN MECHANISM
# -----------------
# The key nonlinearity is that the Datacube world expands FASTER when r and rho
# are already high (see WorldParams: entangle_gain, conv_boost_child, etc.).
# Once enough territories convert, the structural feedback in payoffs makes
# cooperation dominant, which increases the incentive to convert more territories,
# which further reinforces cooperation. This is a positive feedback loop with a
# tipping point. Below the tipping point, the EchoKey controller has to work hard
# to hold cooperation up. Above it, cooperation is self-sustaining and the
# controller can relax (or become redundant).

@dataclass
class CouplingParams:
    """
    Coupling knobs that determine HOW STRONGLY world metrics affect payoffs.

    These are the primary design parameters of the structural mechanism.
    The main run uses moderate values tuned so that conversion is meaningful
    but not so strong that it trivially dominates from the first step.

    r_t = conversion_rate in [0,1]     (Datacube fraction of all territories)
    rho_t = connection_density in [0,1] (entanglement network density)
    e_t = avg_entropy (~[0,3])          (territorial disorder)
    """
    # Stress buffering: how much world conversion reduces effective stress
    # 0.0 = no buffer, 0.35 = moderate (35% stress reduction at full conversion)
    z_buffer_conv: float = 0.35

    # Benefit uplift from conversion and connectivity
    b_boost_conv: float = 0.25   # per unit conversion rate
    b_boost_conn: float = 0.10   # per unit connection density

    # Cost reduction (multiplicative on c0): cooperation gets cheaper as structure matures
    c_reduce_conv: float = 0.45  # reduces c0 by up to 45% at full conversion
    c_reduce_conn: float = 0.15  # additional 15% reduction at full connectivity

    # Defector friction boost: denser/more-converted worlds enforce norms harder
    d_boost_conn: float = 0.45   # dominant lever — connectivity is powerful enforcer
    d_boost_conv: float = 0.15   # conversion also boosts enforcement

    # Baseline defector friction: phi rises with conversion, connectivity, entropy
    phi_boost_conv: float = 0.06
    phi_boost_conn: float = 0.05
    phi_boost_entropy: float = 0.02  # per unit entropy (uncertainty penalizes opportunism)

    # Dampen stress-induced cost amplification as world matures
    c_z_dampen_conv: float = 0.25   # reduces c_z by 25% at full conversion


@dataclass
class WorldMetrics:
    """Snapshot of the Datacube world state at a given timestep.
    Passed into the payoff calculation to realize the coupling."""
    conversion_rate: float       # r(t): fraction of territories that are Datacubes
    connection_density: float    # rho(t): entanglement network density
    avg_entropy: float           # e(t): average territorial entropy
    avg_efficiency: float        # diagnostic (not used in payoffs currently)
    avg_resonance: float         # diagnostic (not used in payoffs currently)


def payoff_coeffs_coupled(
    z: float,
    pp: PayoffParams,
    wm: WorldMetrics,
    cp: CouplingParams,
    delta_b: float = 0.0,
    delta_d: float = 0.0,
) -> Tuple[float, float, float, float, float]:
    """
    Compute COUPLED payoff coefficients (b, c, d, phi) and effective stress z_eff.

    This is the key function that realizes the structural feedback. The logic is:

    1. Extract world metrics and clamp to valid ranges.
    2. Compute effective stress z_eff via stress buffering.
    3. Shift base coefficients (b0, c0, d0, phi0) using world metrics.
       These shifts are INDEPENDENT of current stress level — they represent
       the structural change in the game due to world state.
    4. Apply stress modulation on top of shifted base (using z_eff, not z).
    5. Apply controller deltas (delta_b, delta_d) on top of everything.
    6. Clamp to valid ranges.

    Returns (b, c, d, phi, z_eff) — the five coupled coefficients.
    The caller computes U_C = b*p - c and U_D = b*p - d*p - phi.
    """
    r = clamp(wm.conversion_rate, 0.0, 1.0)
    rho = clamp(wm.connection_density, 0.0, 1.0)
    e = max(0.0, wm.avg_entropy)

    # Step 1: Buffer stress via world conversion
    z_eff = clamp(z * (1.0 - cp.z_buffer_conv * r), 0.0, 1.0)

    # Step 2: Structural shifts to base coefficients
    b0_eff = pp.b0 + cp.b_boost_conv * r + cp.b_boost_conn * rho
    d0_eff = pp.d0 + cp.d_boost_conn * rho + cp.d_boost_conv * r
    phi0_eff = (pp.phi0
                + cp.phi_boost_conv * r
                + cp.phi_boost_conn * rho
                + cp.phi_boost_entropy * e)

    # Step 3: Cost reduction (multiplicative — reduces base cost AND stress sensitivity)
    c0_eff = pp.c0 * (1.0 - cp.c_reduce_conv * r) * (1.0 - cp.c_reduce_conn * rho)
    c0_eff = max(pp.c_min, c0_eff)
    c_z_eff = pp.c_z * (1.0 - cp.c_z_dampen_conv * r)  # stress less able to inflate costs

    # Step 4: Apply stress modulation (on z_eff) and controller deltas
    b = clamp(b0_eff + pp.b_z * z_eff + delta_b, pp.b_min, 10.0)
    c = clamp(c0_eff + c_z_eff * z_eff, pp.c_min, 10.0)
    d = clamp(d0_eff + pp.d_z * z_eff + delta_d, pp.d_min, 10.0)
    phi = clamp(phi0_eff + pp.phi_z * z_eff, pp.phi_min, 10.0)

    return b, c, d, phi, z_eff


def payoffs_coupled(
    p: float,
    z: float,
    pp: PayoffParams,
    wm: WorldMetrics,
    cp: CouplingParams,
    delta_b: float = 0.0,
    delta_d: float = 0.0,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Compute (U_C, U_D) with full Datacube coupling active.
    Returns payoffs plus an auxiliary dict of all intermediate values for logging.
    """
    b, c, d, phi, z_eff = payoff_coeffs_coupled(z, pp, wm, cp, delta_b=delta_b, delta_d=delta_d)
    U_C = b * p - c
    U_D = b * p - d * p - phi
    aux = {"b": b, "c": c, "d": d, "phi": phi, "z_eff": z_eff}
    return U_C, U_D, aux


# =============================================================================
# SECTION 5: MEAN-FIELD REPLICATOR DYNAMICS
# =============================================================================
# The population evolves via a discrete-time replicator equation with mutation.
#
# Standard replicator dynamics:
#   p_{t+1} = p_t + eta * p_t * (1 - p_t) * (U_C - U_D)
#
# The term p_t * (1 - p_t) is the "variance" of the population — it ensures
# that dynamics slow near boundaries (when p ≈ 0 or p ≈ 1, there's little
# pressure to change). eta scales how fast the population responds.
#
# After replication, mutation is applied:
#   p_{t+1} = (1 - 2*mu) * p_{t+1} + mu
#
# This keeps p strictly inside (0, 1) and allows exploration. It models
# imperfect strategy adoption: a small fraction mu randomly switches strategies
# at each step. Without mutation, the system can get trapped at p=0 or p=1
# forever, which is unrealistic.
#
# CALIBRATION: eta=0.70 is moderately fast. The population is responsive but
# not so fast that a single shock causes immediate collapse. mu=0.002 is small
# enough that mutation doesn't dominate but large enough to prevent permanent
# corner trapping.

@dataclass
class DynamicsParams:
    eta: float = 0.60         # replicator step size (speed of selection)
    mu: float = 0.002         # mutation rate (strategy exploration probability)
    clip_eps: float = 1e-6    # boundary clip (prevents exact 0 or 1)


def replicator_step(p: float, Uc: float, Ud: float, dp: DynamicsParams) -> float:
    """
    Advance cooperator fraction by one timestep.

    The replicator term grows or shrinks p based on the sign and magnitude of
    (Uc - Ud). Positive gap (cooperation pays more) → p increases. Negative
    gap → p decreases. The p*(1-p) factor ensures smooth dynamics near boundaries.

    Mutation is applied AFTER replication: a fraction mu defects, a fraction mu
    cooperates, net effect (1-2*mu) on current p plus mu additive floor.
    """
    p = clamp(p, dp.clip_eps, 1.0 - dp.clip_eps)
    growth = dp.eta * p * (1.0 - p) * (Uc - Ud)
    p_next = p + growth
    p_next = (1.0 - 2.0 * dp.mu) * p_next + dp.mu
    return clamp(p_next, 0.0, 1.0)


# =============================================================================
# SECTION 6: ECHOKEY META-LAYER (CONTROLLER)
# =============================================================================
# The EchoKey controller is a meta-layer that sits on top of the mean-field game
# and intervenes each timestep by choosing a combination of three levers:
#
#   delta_b  : boost cooperative benefit (makes C more attractive)
#   delta_d  : boost defector friction (makes D more costly)
#   u        : suppression (a direct stabilization effort, e.g. enforcement)
#
# The controller is NOT omnipotent. It cannot directly set p or override the
# replicator dynamics. It can only shape incentives. This is the key design
# constraint — it models realistic institutional intervention.
#
# CONTROLLER OBJECTIVE (per step)
# --------------------------------
# At each step, the controller performs a grid search over all (delta_b, delta_d, u)
# combinations in a discrete grid (grid_n^3 candidates) and picks the one that
# minimizes a cost function J:
#
#   J = alpha * (s_star - s)^2      -- stability penalty (keep s near target s_star)
#     + beta * u^2                  -- suppression cost (u is expensive to use)
#     + w_b * delta_b^2             -- benefit intervention cost
#     + w_d * delta_d^2             -- friction intervention cost
#     + Lambda * viol^2             -- heavy penalty for harm violations
#     - w_next * p_next             -- reward for increasing cooperator fraction
#
# The stability score s is derived from the harm-adjusted preharm stability s_tilde:
#   s_tilde = 0.20 + 0.70*p_next - 0.55*z_eff + u_benefit
#             (cooperation fraction increases stability, stress decreases it, u helps)
#   s = clamp(s_tilde - kappa * h, 0, 1)
#             (adjusted downward by harm, weighted by kappa)
#
# HARM MODEL
# ----------
# Suppression u generates harm h:
#   h = harm_floor + harm_sensitivity * (1 + harm_stress_gain * z_eff) * u^harm_exp
#
# Interpretation: any intervention causes some harm (floor). More suppression causes
# more harm. High stress amplifies harm (intervening during a crisis is more disruptive).
# The exponent >1 means harm is superlinear in u (diminishing returns on suppression).
#
# DYNAMIC HARM THRESHOLD
# -----------------------
# The harm threshold tau is NOT fixed. It is a function of current stability s_tilde:
#   tau(s_tilde) = tau_min + (tau_max - tau_min) * sigmoid(gamma * (s_tilde - s_star))
#
# When stability is high (s_tilde >> s_star), tau increases: the system can afford
# more intervention. When stability is low (s_tilde << s_star), tau decreases: the
# system is fragile, so intervention must be gentle. This is a form of adaptive
# constraint tightening under stress.
#
# Violation: viol = max(0, h - tau). Penalized quadratically by Lambda in J.
# The high Lambda (18.0 in main run) means the controller strongly avoids violations.
#
# WHY ECHOKEY?
# ------------
# The EchoKey framing (from the broader EchoKey cryptographic/coordination framework)
# embeds the idea that the controller has a "key" — a set of levers — that it uses
# to unlock cooperative equilibria. The controller is not brute-forcing cooperation;
# it is finding the minimal intervention that keeps the system in a cooperative basin
# long enough for structural feedback (Datacubes) to take over.

@dataclass
class EchoKeyParams:
    # Controller lever bounds
    delta_b_max: float = 0.80    # maximum benefit boost per step
    delta_d_max: float = 0.80    # maximum friction boost per step
    u_max: float = 1.00          # maximum suppression level

    # Stability model: u contribution to stability
    stab_u_gain: float = 0.35    # how much u can improve stability score
    stab_u_rate: float = 3.0     # rate of u's diminishing returns in stability

    # Harm model
    harm_floor: float = 0.01         # minimum harm from any nonzero u
    harm_sensitivity: float = 1.40   # scales how much u hurts
    harm_exp: float = 1.15           # superlinearity of harm in u (>1 = convex)
    harm_stress_gain: float = 0.65   # stress amplification of harm

    # Stability-harm tradeoff
    kappa: float = 1.10          # weight on harm in stability adjustment (s = s_tilde - kappa*h)

    # Dynamic harm threshold (tau) parameters
    tau_min: float = 0.02        # minimum allowed harm (tight constraint when fragile)
    tau_max: float = 0.18        # maximum allowed harm (relaxed when stable)
    s_star: float = 0.78         # stability target (controller aims to keep s near this)
    gamma: float = 10.0          # sigmoid steepness for tau transition

    # Objective function weights
    alpha: float = 2.0           # weight on stability gap penalty
    beta: float = 0.20           # weight on suppression cost
    w_b: float = 0.15            # weight on benefit intervention cost
    w_d: float = 0.15            # weight on friction intervention cost
    Lambda: float = 12.0         # weight on harm violation penalty (dominant term)

    # Forward-looking reward
    w_next: float = 1.0          # reward for increasing p_next (cooperation incentive)
    grid_n: int = 21             # grid resolution per lever (grid_n^3 total candidates)


def preharm_stability(p: float, z: float, u: float, ek: EchoKeyParams) -> float:
    """
    Compute pre-harm stability score s_tilde ∈ [0,1].

    s_tilde captures how stable the system appears BEFORE accounting for the
    harm cost of intervention. High p (many cooperators) → stable. High z
    (high stress) → unstable. High u (suppression) → adds stability, with
    diminishing returns (via exponential saturation term).

    This is an ad hoc stability index, not derived from a Lyapunov function.
    It's a reasonable heuristic for what "stable" means in this context.
    """
    base = 0.20 + 0.70 * p - 0.55 * z
    u_help = ek.stab_u_gain * (1.0 - math.exp(-ek.stab_u_rate * u))
    return clamp(base + u_help, 0.0, 1.0)


def harm_model(z: float, u: float, ek: EchoKeyParams) -> float:
    """
    Compute harm h generated by suppression u at stress level z.

    Harm is nonzero even at small u (floor term). It scales superlinearly
    with u (harm_exp > 1 means the controller faces convex costs — doubling
    suppression more than doubles harm). Stress amplifies harm: aggressive
    intervention during a shock is more disruptive than the same intervention
    in calm conditions.
    """
    amp = (1.0 + ek.harm_stress_gain * z)
    h = ek.harm_floor + ek.harm_sensitivity * amp * (u ** ek.harm_exp)
    return max(0.0, h)


def dynamic_tau(s_tilde: float, ek: EchoKeyParams) -> float:
    """
    Compute the dynamic harm threshold tau(s_tilde).

    Uses a sigmoid centered at s_star: when stability is above target,
    tau rises toward tau_max (more harm tolerable). When stability is below
    target, tau falls toward tau_min (system is fragile, must be gentle).

    gamma=10 makes this a fairly sharp transition — the threshold changes
    meaningfully over a small range of s_tilde around s_star.
    """
    return ek.tau_min + (ek.tau_max - ek.tau_min) * sigmoid(ek.gamma * (s_tilde - ek.s_star))


def datacube_control_step(
    p_t: float,
    z_t: float,
    pp: PayoffParams,
    dp: DynamicsParams,
    ek: EchoKeyParams,
    wm: WorldMetrics,
    cp: CouplingParams,
) -> Dict[str, float]:
    """
    Run one step of the EchoKey controller with full Datacube coupling.

    Algorithm:
    1. For each (delta_b, delta_d, u) in the grid_n^3 search space:
       a. Compute COUPLED payoffs using current world metrics wm.
       b. Advance population one step via replicator to get p_next.
       c. Compute stability s_tilde and harm h using z_eff (not raw z).
       d. Compute harm-adjusted stability s = s_tilde - kappa*h.
       e. Compute dynamic threshold tau and violation viol = max(0, h-tau).
       f. Compute cost J.
    2. Select the (delta_b, delta_d, u) that minimizes J.
    3. Return all diagnostics for the winning candidate.

    The key insight: using z_eff (buffered stress) in stability/harm means the
    world conversion DIRECTLY reduces the harm of intervention and improves
    stability assessment. As the Datacube world matures, the controller faces
    a progressively easier optimization problem.
    """
    best: Optional[Dict[str, float]] = None

    n = ek.grid_n
    for i in range(n):
        delta_b = ek.delta_b_max * i / (n - 1)
        for j in range(n):
            delta_d = ek.delta_d_max * j / (n - 1)
            for k in range(n):
                u = ek.u_max * k / (n - 1)

                # Compute coupled payoffs
                Uc, Ud, aux = payoffs_coupled(p_t, z_t, pp, wm, cp, delta_b=delta_b, delta_d=delta_d)
                p_next = replicator_step(p_t, Uc, Ud, dp)

                # Use z_eff everywhere (world buffering applies to stability/harm too)
                z_eff = float(aux["z_eff"])
                s_tilde = preharm_stability(p_next, z_eff, u, ek)
                h = harm_model(z_eff, u, ek)
                s = clamp(s_tilde - ek.kappa * h, 0.0, 1.0)

                tau = dynamic_tau(s_tilde, ek)
                viol = max(0.0, h - tau)

                # Controller objective: minimize J
                J = (
                    ek.alpha * (ek.s_star - s) ** 2      # stability gap
                    + ek.beta * (u ** 2)                  # suppression cost
                    + ek.w_b * (delta_b ** 2)             # benefit intervention cost
                    + ek.w_d * (delta_d ** 2)             # friction intervention cost
                    + ek.Lambda * (viol ** 2)             # harm violation penalty
                    - ek.w_next * p_next                  # reward cooperation
                )

                if best is None or J < best["J"]:
                    best = {
                        "J": J,
                        "delta_b": delta_b,
                        "delta_d": delta_d,
                        "u": u,
                        "p_next": p_next,
                        "Uc": Uc,
                        "Ud": Ud,
                        "s_tilde": s_tilde,
                        "h": h,
                        "s": s,
                        "tau": tau,
                        "viol": viol,
                        "b": float(aux["b"]),
                        "c": float(aux["c"]),
                        "d": float(aux["d"]),
                        "phi": float(aux["phi"]),
                        "z_eff": float(aux["z_eff"]),
                        "r": float(wm.conversion_rate),
                        "rho": float(wm.connection_density),
                        "e": float(wm.avg_entropy),
                    }

    assert best is not None
    return best


# =============================================================================
# SECTION 7: DATACUBE WORLD — SPATIAL SIMULATION
# =============================================================================
# The Datacube world is a spatial agent-based simulation running in a 3D grid
# of dimensions (128, 128, 128) = ~2 million cells. Players own territories
# (points in this space) and can convert them to Datacubes.
#
# TERRITORY: A point in 3D space owned by a player. Has:
#   - efficiency: how well-managed it is (decays under influence, high in Datacubes)
#   - resonance_score: Datacube's radiative influence on neighbors
#   - entropy: disorder/inefficiency (grows under resonance influence)
#   - conversion metadata: when/if it converted, which Datacube is its parent
#
# PLAYER: Owns a set of territories. Has a strategy (traditional or datacube)
# and a conversion_rate that governs how fast they convert remaining territories.
#
# WORLD DYNAMICS (per step)
# --------------------------
# 1. RESONANCE FIELD: Each Datacube radiates influence on nearby traditional
#    territories. Influence decays exponentially with distance (scale=10 cells),
#    within influence_radius=20 cells. Territories in range accumulate resonance.
#
# 2. TERRITORY STATE UPDATE: Traditional territories under resonance influence:
#    - Lose efficiency faster (decay rate scales with influence strength)
#    - Gain entropy faster (disorder grows under competitive pressure)
#    Traditional territories OUT of range still slowly decay.
#    Datacubes improve steadily (efficiency and resonance grow each step).
#
# 3. PLAYER DECISIONS: Each player evaluates their portfolio:
#    - Compute average "traditional cost" (entropy/efficiency ratio — high = bad)
#    - Compute average "Datacube benefit" (efficiency * resonance — high = good)
#    - If worst_score > conversion_threshold OR Datacube benefit is already high:
#      Switch strategy to "datacube" and start converting.
#    - Conversion priority: convert the highest-entropy/lowest-efficiency territories
#      first, with a bonus for territories near existing Datacubes (clustering).
#
# ENTANGLEMENT
# ------------
# When territory B converts and its nearest Datacube parent is owned by a
# DIFFERENT player, cross-player entanglement occurs:
#   entanglement_matrix[i,j] += entangle_gain (0.05)
#
# Entanglement is tracked as a symmetric matrix. Connection density rho is
# computed as the fraction of off-diagonal entanglement matrix entries > 0.
#
# Entanglement also triggers CONVERSION RATE BOOSTS for both involved players:
#   child player: rate *= conv_boost_child (1.05)
#   parent player: rate *= conv_boost_parent (1.02)
#
# This models cross-player network effects: when your territory is "inspired"
# by a neighbor's Datacube, both of you convert faster afterward. This is the
# spreading mechanism that makes the system self-reinforcing.
#
# FEEDBACK LOOP SUMMARY
# ---------------------
# More Datacubes → stronger resonance field → traditional territories degrade faster
# → players hit conversion_threshold sooner → more territories convert → entanglement
# grows → connection density increases → payoff coupling (Section 4) kicks in harder
# → cooperation becomes more attractive in the mean-field game → cooperator fraction p
# rises → stability improves → controller needs to intervene less → harm decreases →
# system self-stabilizes.
#
# The loop closes when conversion_rate and connection_density are both high enough
# that the payoff gap U_C - U_D is positive even without any controller intervention.
# That is the "lock-in regime" — cooperation is the dominant Nash equilibrium of
# the structurally coupled game.

@dataclass
class Territory:
    """Represents a single spatial unit in the Datacube world."""
    id: int
    owner_id: int
    position: Tuple[int, int, int]       # 3D grid position
    is_datacube: bool = False            # has this territory been converted?
    efficiency: float = 0.5             # operational efficiency [0,1]
    resonance_score: float = 0.0        # influence radius strength (Datacubes only)
    entropy: float = 1.0                # disorder level [0,3]
    conversion_step: int = -1           # timestep at which conversion occurred (-1 if not)
    parent_datacube_id: int = -1        # ID of nearest Datacube at time of conversion


@dataclass
class Player:
    """Represents a strategic actor owning a set of territories."""
    id: int
    strategy: str = "traditional"       # "traditional" or "datacube"
    territories_owned: int = 0
    datacubes_owned: int = 0
    total_efficiency: float = 0.0
    total_resonance: float = 0.0
    conversion_step: int = -1           # when this player switched strategy
    conversion_rate: float = 0.01       # fraction of remaining territories converted per step


@dataclass
class WorldParams:
    """Parameters governing the spatial world dynamics."""
    # Resonance field
    influence_radius: float = 20.0      # max distance (cells) for Datacube influence
    entropy_growth_rate: float = 0.01   # rate of entropy increase under resonance
    efficiency_decay_rate: float = 0.005 # rate of efficiency loss under resonance

    # Conversion trigger
    conversion_threshold: float = 3.0   # entropy/efficiency ratio that triggers player switch

    # Datacube improvement (Datacubes get better over time)
    dc_eff_gain: float = 0.0005         # efficiency gain per step for Datacubes
    dc_res_gain: float = 0.005          # resonance gain per step for Datacubes

    # Initial values when a territory converts to Datacube
    dc_init_eff: float = 0.8            # high initial efficiency (why you convert)
    dc_init_res: float = 0.5            # initial resonance (starts radiating influence)
    dc_init_ent: float = 0.1            # low initial entropy (organized)

    # Entanglement and conversion rate dynamics
    entangle_gain: float = 0.05         # entanglement added per cross-player conversion
    conv_boost_child: float = 1.05      # child player's rate multiplied by this on entanglement
    conv_boost_parent: float = 1.02     # parent player's rate multiplied by this
    conv_rate_floor_after_adopt: float = 0.02  # minimum conversion rate after strategy switch
    conv_rate_cap: float = 0.20         # maximum conversion rate (prevents runaway)


class WorldLogger:
    """
    Logs Datacube world state to CSV files for offline analysis.

    Logs five data streams:
      game_states     : per-step aggregate world metrics
      territory_states: per-step sampled territory states
      player_states   : per-step per-player states
      conversions     : individual conversion events (territory-level)
      entanglements   : per-step significant entanglement pairs (>0.01)

    Territory sampling is biased toward Datacubes and recently-converted territories
    (weighted random sample) to ensure interesting events are captured even when
    territory_sample_size < total territories.
    """
    def __init__(
        self,
        log_dir: str = "game_logs",
        log_every_n_steps: int = 1,
        log_all_territories: bool = False,
        territory_sample_size: int = 100,
        seed: int = 0,
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.log_every_n_steps = int(log_every_n_steps)
        self.log_all_territories = bool(log_all_territories)
        self.territory_sample_size = int(territory_sample_size)

        self.rng = random.Random(seed)

        self.game_states: List[Dict[str, object]] = []
        self.territory_states: List[Dict[str, object]] = []
        self.player_states: List[Dict[str, object]] = []
        self.conversions: List[Dict[str, object]] = []
        self.entanglements: List[Dict[str, object]] = []

    def log_state(self, step: int, world: "DatacubeWorld") -> None:
        if self.log_every_n_steps <= 0 or (step % self.log_every_n_steps != 0):
            return

        ts = datetime.now().isoformat(timespec="seconds")

        self.game_states.append({
            "step": step,
            "timestamp": ts,
            "total_territories": len(world.territories),
            "total_datacubes": world.total_datacubes,
            "connection_density": world.connection_density,
            "avg_efficiency": world.avg_efficiency,
            "avg_resonance": world.avg_resonance,
            "avg_entropy": world.avg_entropy,
            "conversion_rate": world.conversion_rate,
        })

        for p in world.players:
            self.player_states.append({
                "step": step,
                "player_id": p.id,
                "strategy": p.strategy,
                "territories_owned": p.territories_owned,
                "datacubes_owned": p.datacubes_owned,
                "total_efficiency": p.total_efficiency,
                "total_resonance": p.total_resonance,
                "conversion_rate": p.conversion_rate,
            })

        # Territory sampling: biased toward Datacubes and recently-converted
        # territories to ensure interesting dynamics are captured in the log.
        terrs = list(world.territories.values())
        if (not self.log_all_territories) and len(terrs) > self.territory_sample_size:
            scores = []
            for t in terrs:
                score = 10.0 if t.is_datacube else 1.0    # Datacubes get 10x weight
                if t.conversion_step >= 0 and t.conversion_step > step - 10:
                    score += 5.0                           # recently converted = bonus weight
                scores.append(score)
            total = sum(scores)
            probs = [s / total for s in scores]

            chosen = set()
            while len(chosen) < self.territory_sample_size:
                idx = self._weighted_pick(probs)
                chosen.add(idx)
            terrs = [terrs[i] for i in chosen]

        for t in terrs:
            self.territory_states.append({
                "step": step,
                "territory_id": t.id,
                "owner_id": t.owner_id,
                "pos_x": t.position[0],
                "pos_y": t.position[1],
                "pos_z": t.position[2],
                "is_datacube": t.is_datacube,
                "efficiency": t.efficiency,
                "resonance_score": t.resonance_score,
                "entropy": t.entropy,
                "conversion_step": t.conversion_step,
                "parent_datacube_id": t.parent_datacube_id,
            })

        # Log significant entanglement pairs (threshold 0.01 avoids clutter)
        n = len(world.players)
        for i in range(n):
            for j in range(i + 1, n):
                val = world.entanglement_matrix[i, j]
                if val > 0.01:
                    self.entanglements.append({
                        "step": step,
                        "player1_id": i + 1,
                        "player2_id": j + 1,
                        "entanglement_strength": float(val),
                    })

    def log_conversion(
        self,
        step: int,
        territory_id: int,
        owner_id: int,
        parent_datacube_id: int,
        pre_efficiency: float,
        post_efficiency: float,
        distance_to_parent: float,
    ) -> None:
        """Log a single territory conversion event."""
        self.conversions.append({
            "step": step,
            "territory_id": territory_id,
            "owner_id": owner_id,
            "parent_datacube_id": parent_datacube_id,
            "pre_efficiency": pre_efficiency,
            "post_efficiency": post_efficiency,
            "distance_to_parent": distance_to_parent,
        })

    def save(self) -> str:
        """Write all logged data to timestamped CSV files and a JSON metadata file."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def write_csv(name: str, rows: List[Dict[str, object]]) -> None:
            if not rows:
                return
            path = os.path.join(self.log_dir, f"{name}_{stamp}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        write_csv("game_states", self.game_states)
        write_csv("territory_states", self.territory_states)
        write_csv("player_states", self.player_states)
        write_csv("conversions", self.conversions)
        write_csv("entanglements", self.entanglements)

        meta = {
            "timestamp": stamp,
            "total_steps": (max(r["step"] for r in self.game_states) if self.game_states else 0),
            "total_territories": (self.game_states[-1]["total_territories"] if self.game_states else 0),
            "final_datacubes": (self.game_states[-1]["total_datacubes"] if self.game_states else 0),
            "final_connection_density": (self.game_states[-1]["connection_density"] if self.game_states else 0.0),
            "final_conversion_rate": (self.game_states[-1]["conversion_rate"] if self.game_states else 0.0),
            "log_every_n_steps": self.log_every_n_steps,
            "territory_sample_size": self.territory_sample_size,
        }
        meta_path = os.path.join(self.log_dir, f"metadata_{stamp}.json")
        try:
            import json
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

        return stamp

    def _weighted_pick(self, probs: List[float]) -> int:
        """Weighted random pick without replacement (manual implementation)."""
        r = self.rng.random()
        s = 0.0
        for i, p in enumerate(probs):
            s += p
            if r <= s:
                return i
        return len(probs) - 1


class DatacubeWorld:
    """
    The spatial simulation that generates world metrics feeding into the mean-field game.

    Core state:
      territories       : dict of Territory objects (id -> Territory)
      players           : list of Player objects
      entanglement_matrix: n_players x n_players symmetric matrix
      Aggregate metrics : avg_efficiency, avg_resonance, avg_entropy,
                          connection_density, conversion_rate

    These aggregate metrics are recomputed each step and exposed as WorldMetrics
    to the payoff coupling functions. They are the coupling interface between
    the spatial world and the mean-field game.
    """
    def __init__(
        self,
        dims: Tuple[int, int, int],
        n_players: int,
        wp: WorldParams,
        seed: int = 0,
    ):
        self.dims = dims
        self.wp = wp
        self.rng = random.Random(seed)

        self.players: List[Player] = [Player(id=i + 1) for i in range(n_players)]
        self.territories: Dict[int, Territory] = {}

        self.entanglement_matrix = np.zeros((n_players, n_players), dtype=np.float32)

        self.total_datacubes = 0
        self.current_step = 0

        # Aggregate metrics (updated by calculate_metrics each step)
        self.avg_efficiency = 0.5
        self.avg_resonance = 0.0
        self.avg_entropy = 1.0
        self.connection_density = 0.0
        self.conversion_rate = 0.0

    def initialize_game(self, territories_per_player: int) -> None:
        """
        Place territories randomly in the 3D grid.

        Each player gets territories_per_player territories at unique random
        positions. Positions are shuffled before assignment so there's no
        spatial clustering bias by player ID.

        After placement, metrics are computed for the initial state.
        """
        n_players = len(self.players)
        total = n_players * territories_per_player

        used = set()
        attempts = 0
        max_attempts = total * 20
        positions: List[Tuple[int, int, int]] = []
        while len(positions) < total and attempts < max_attempts:
            pos = (
                self.rng.randint(1, self.dims[0]),
                self.rng.randint(1, self.dims[1]),
                self.rng.randint(1, self.dims[2]),
            )
            if pos not in used:
                used.add(pos)
                positions.append(pos)
            attempts += 1

        self.rng.shuffle(positions)

        tid = 1
        idx = 0
        for pid in range(1, n_players + 1):
            for _ in range(territories_per_player):
                if idx >= len(positions):
                    break
                pos = positions[idx]
                idx += 1
                t = Territory(id=tid, owner_id=pid, position=pos)
                self.territories[tid] = t
                self.players[pid - 1].territories_owned += 1
                tid += 1

        self.calculate_metrics()

    def seed_first_datacube(self, territory_id: int, logger: Optional[WorldLogger] = None, resonance_boost: float = 1.0) -> None:
        """
        Convert one territory to a Datacube to start the spreading process.

        This is the "patient zero" of the Datacube network. Without this seed,
        no resonance field exists and no other territory will ever convert.
        The resonance_boost sets its initial influence strength.
        """
        self.convert_to_datacube(territory_id, logger=logger)
        self.territories[territory_id].resonance_score = resonance_boost
        self.calculate_metrics()

    def compute_resonance_field(self) -> Dict[int, float]:
        """
        Compute the resonance influence on each traditional territory.

        For each traditional territory, sum contributions from all Datacubes
        within influence_radius. Contribution decays as exp(-dist/10).
        Returns a dict mapping territory_id -> total resonance influence.

        This is O(territories * datacubes) per step — fine for toy scale but
        would need spatial indexing (KD-tree etc.) for large worlds.
        """
        datacubes = [(t.position, t.resonance_score, t.owner_id, t.id)
                     for t in self.territories.values() if t.is_datacube]
        if not datacubes:
            return {}

        infl: Dict[int, float] = {}
        R = self.wp.influence_radius
        for tid, terr in self.territories.items():
            if terr.is_datacube:
                continue
            total = 0.0
            for (dc_pos, dc_res, _dc_owner, _dc_id) in datacubes:
                dist = euclid3(terr.position, dc_pos)
                if dist <= R:
                    total += float(dc_res) * math.exp(-dist / 10.0)
            infl[tid] = total
        return infl

    def update_territory_states(self, influences: Dict[int, float]) -> None:
        """
        Update efficiency and entropy for all territories.

        Traditional territories under Datacube influence:
          - Efficiency decays faster (pressure to convert)
          - Entropy grows faster (disorder under competitive displacement)

        Traditional territories out of range still slowly decay (economic drift).

        Datacubes improve steadily: efficiency and resonance grow each step,
        making them more valuable and more influential over time.
        This creates the positive feedback that drives conversion spread.
        """
        for tid, t in self.territories.items():
            if not t.is_datacube:
                influence = influences.get(tid, 0.0)
                if influence > 0.0:
                    eff_decay = self.wp.efficiency_decay_rate * (1.0 + 0.5 * influence)
                    t.efficiency = max(0.1, t.efficiency - eff_decay)
                    t.entropy = min(3.0, t.entropy + self.wp.entropy_growth_rate * (1.0 + 0.5 * influence))
                else:
                    t.efficiency = max(0.3, t.efficiency - 0.0001)
            else:
                t.efficiency = min(1.0, t.efficiency + self.wp.dc_eff_gain)
                t.resonance_score = min(2.0, t.resonance_score + self.wp.dc_res_gain)

    def find_nearest_datacube(self, pos: Tuple[int, int, int]) -> Optional[int]:
        """Find the ID of the nearest Datacube to position pos.
        Used to determine parent entanglement when a territory converts."""
        best_id = None
        best_dist = float("inf")
        for tid, t in self.territories.items():
            if not t.is_datacube:
                continue
            dist = euclid3(pos, t.position)
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id

    def convert_to_datacube(self, territory_id: int, logger: Optional[WorldLogger] = None) -> None:
        """
        Convert a traditional territory to a Datacube.

        Steps:
        1. Reset territory stats to Datacube initial values (high efficiency, low entropy).
        2. Record conversion metadata (step, parent Datacube).
        3. If nearest Datacube belongs to a different player: add entanglement
           and boost both players' conversion rates.
        4. Log the conversion event.
        5. Increment total_datacubes counter.

        Cross-player entanglement is the key spreading mechanism — it models
        the idea that when your territory converts near someone else's Datacube,
        you and that player become informationally/institutionally linked,
        which accelerates both of your further conversions.
        """
        t = self.territories[territory_id]
        if t.is_datacube:
            return

        pre_eff = t.efficiency
        nearest_dc = self.find_nearest_datacube(t.position)

        t.is_datacube = True
        t.efficiency = self.wp.dc_init_eff
        t.resonance_score = self.wp.dc_init_res
        t.entropy = self.wp.dc_init_ent
        t.conversion_step = self.current_step
        t.parent_datacube_id = (-1 if nearest_dc is None else int(nearest_dc))

        player = self.players[t.owner_id - 1]
        player.datacubes_owned += 1

        # Cross-player entanglement: only triggered if nearest Datacube has a different owner
        if nearest_dc is not None:
            parent_owner = self.territories[nearest_dc].owner_id
            if parent_owner != t.owner_id:
                i = t.owner_id - 1
                j = parent_owner - 1
                self.entanglement_matrix[i, j] += self.wp.entangle_gain
                self.entanglement_matrix[j, i] += self.wp.entangle_gain

                # Both players convert faster after cross-player entanglement
                player.conversion_rate = min(self.wp.conv_rate_cap,
                                             player.conversion_rate * self.wp.conv_boost_child)
                self.players[parent_owner - 1].conversion_rate = min(
                    self.wp.conv_rate_cap,
                    self.players[parent_owner - 1].conversion_rate * self.wp.conv_boost_parent
                )

        dist = 0.0
        if nearest_dc is not None:
            dist = euclid3(t.position, self.territories[nearest_dc].position)

        if logger is not None:
            logger.log_conversion(
                step=self.current_step,
                territory_id=territory_id,
                owner_id=t.owner_id,
                parent_datacube_id=(-1 if nearest_dc is None else int(nearest_dc)),
                pre_efficiency=pre_eff,
                post_efficiency=t.efficiency,
                distance_to_parent=float(dist),
            )

        self.total_datacubes += 1

    def make_player_decisions(self, logger: Optional[WorldLogger] = None) -> None:
        """
        Each player evaluates their portfolio and decides whether and what to convert.

        STRATEGY SWITCH (traditional → datacube):
          Triggered when worst_score > conversion_threshold OR existing Datacubes
          already outperform traditional territories by a large margin.
          worst_score = max(entropy/efficiency) across owned traditional territories.

        CONVERSION RATE ACCELERATION:
          Players already in "datacube" strategy increase their rate if Datacube
          benefit > traditional cost * 1.2 (positive feedback on success).

        TERRITORY SELECTION FOR CONVERSION:
          Convert the territories with highest (entropy/efficiency + proximity bonus).
          The proximity bonus rewards converting territories near existing Datacubes,
          creating spatial clustering and strengthening local resonance fields.
          The number converted per step = floor(conversion_rate * trad_count), min 1.
        """
        for p in self.players:
            trad_cost = 0.0
            dc_benefit = 0.0
            trad_count = 0
            dc_count = 0
            worst_score = 0.0

            for t in self.territories.values():
                if t.owner_id != p.id:
                    continue
                if t.is_datacube:
                    dc_count += 1
                    dc_benefit += t.efficiency * t.resonance_score
                else:
                    trad_count += 1
                    score = t.entropy / max(0.1, t.efficiency)
                    trad_cost += score
                    worst_score = max(worst_score, score)

            if trad_count + dc_count == 0:
                continue

            avg_trad_cost = (trad_cost / trad_count) if trad_count > 0 else 0.0
            avg_dc_benefit = (dc_benefit / dc_count) if dc_count > 0 else 0.0

            if p.strategy == "traditional":
                if (worst_score > self.wp.conversion_threshold) or (dc_count > 0 and avg_dc_benefit > 1.5):
                    p.strategy = "datacube"
                    p.conversion_step = self.current_step
                    p.conversion_rate = max(p.conversion_rate, self.wp.conv_rate_floor_after_adopt)
            else:
                if avg_dc_benefit > avg_trad_cost * 1.2:
                    p.conversion_rate = min(self.wp.conv_rate_cap, p.conversion_rate * 1.1)

            if p.strategy == "datacube" and trad_count > 0:
                n_to_convert = max(1, int(math.floor(p.conversion_rate * trad_count)))

                candidates: List[Tuple[int, float]] = []
                for t in self.territories.values():
                    if t.owner_id != p.id or t.is_datacube:
                        continue

                    score = t.entropy / max(0.1, t.efficiency)

                    # Proximity bonus: favor converting territories near existing Datacubes
                    near_bonus = 0.0
                    for dc in self.territories.values():
                        if not dc.is_datacube:
                            continue
                        dist = euclid3(t.position, dc.position)
                        if dist <= self.wp.influence_radius * 0.5:
                            near_bonus += 1.0 / (1.0 + dist)

                    candidates.append((t.id, score + near_bonus))

                candidates.sort(key=lambda x: x[1], reverse=True)
                for i in range(min(n_to_convert, len(candidates))):
                    self.convert_to_datacube(candidates[i][0], logger=logger)

    def calculate_metrics(self) -> None:
        """
        Recompute aggregate world metrics from current territory and player states.

        Metrics computed:
          avg_efficiency   : mean territory efficiency (Datacubes high, others medium/low)
          avg_resonance    : mean territory resonance (mostly from Datacubes)
          avg_entropy      : mean territory entropy (rises under stress, drops in Datacubes)
          connection_density: fraction of possible player pairs with entanglement > 0
          conversion_rate  : fraction of all territories that are Datacubes

        These are the quantities exposed as WorldMetrics to the payoff coupling.
        connection_density uses n_players^2 as denominator (includes diagonal) —
        a slight approximation but fine for coupling purposes.
        """
        total = len(self.territories)
        if total <= 0:
            return

        tot_eff = 0.0
        tot_res = 0.0
        tot_ent = 0.0
        for t in self.territories.values():
            tot_eff += t.efficiency
            tot_res += t.resonance_score
            tot_ent += t.entropy

        self.avg_efficiency = tot_eff / total
        self.avg_resonance = tot_res / total
        self.avg_entropy = tot_ent / total

        conn = float(np.sum(self.entanglement_matrix > 0.0))
        max_conn = float(len(self.players) ** 2)
        self.connection_density = (conn / max_conn) if max_conn > 0 else 0.0

        self.conversion_rate = float(self.total_datacubes) / float(total)

        for p in self.players:
            p.territories_owned = 0
            p.datacubes_owned = 0
            p.total_efficiency = 0.0
            p.total_resonance = 0.0

        for t in self.territories.values():
            p = self.players[t.owner_id - 1]
            p.territories_owned += 1
            p.total_efficiency += t.efficiency
            p.total_resonance += t.resonance_score
            if t.is_datacube:
                p.datacubes_owned += 1

    def step(self, step_idx: int, logger: Optional[WorldLogger] = None) -> None:
        """
        Advance the world by one timestep.

        Order of operations:
        1. Compute resonance field (influence from current Datacubes)
        2. Update territory states under that field
        3. Players make conversion decisions
        4. Recompute aggregate metrics
        5. Log if requested

        This ordering means the resonance field computed at step t is based on
        Datacubes that existed at the START of step t. New conversions from step t
        only affect the resonance field at step t+1. This prevents same-step feedback.
        """
        self.current_step = int(step_idx)

        influences = self.compute_resonance_field()
        self.update_territory_states(influences)
        self.make_player_decisions(logger=logger)
        self.calculate_metrics()

        if logger is not None:
            logger.log_state(self.current_step, self)


# =============================================================================
# SECTION 8: SIMULATION RUNNERS
# =============================================================================
# Three runners:
#
#   run_baseline: Mean-field only, no controller, no world coupling.
#     The "what happens if nothing is done" counterfactual.
#
#   run_datacube_no_controller: Datacube world coupled to mean-field, but NO
#     EchoKey controller. Raw structural coupling only — no grid search, no
#     intervention levers. Useful for isolating whether the controller is actually
#     necessary or whether the world feedback alone drives cooperation.
#     If this matches the full run, the controller is redundant.
#     If cooperation is meaningfully lower, the controller is doing real work
#     in the early bootstrapping phase before conversion reaches critical mass.
#
#   run_datacube_controller: Full coupled system with EchoKey controller.
#     At each timestep:
#     1. Advance the Datacube world to get current world metrics.
#     2. Run the EchoKey controller grid search using those metrics.
#     3. Advance the mean-field population using the controller's chosen levers.
#     4. Record all diagnostic traces.
#
# All three runs use the same stress process (same seed) so differences in
# outcomes are purely due to coupling and/or controller, not lucky stress draws.

@dataclass
class RunParams:
    T: int = 220      # number of timesteps
    p0: float = 0.22  # initial cooperator fraction (pessimistic start)
    seed: int = 777   # RNG seed (for stress process and world)


def run_baseline(z: np.ndarray, pp: PayoffParams, dp: DynamicsParams, rp: RunParams) -> Dict[str, np.ndarray]:
    """
    Baseline: pure mean-field replicator with no controller, no coupling.

    Just runs U_C, U_D → replicator for T steps under the stress sequence z.
    Returns {'p': array of cooperator fractions}.

    Expected behavior: cooperation collapses under repeated shocks, ending near
    p≈0.1-0.3 depending on stress parameters. This is the "problem" the
    Datacube mechanism is designed to solve.
    """
    p = np.zeros(rp.T, dtype=np.float64)
    p[0] = rp.p0
    for t in range(rp.T - 1):
        Uc, Ud = payoffs(p[t], float(z[t]), pp)
        p[t + 1] = replicator_step(p[t], Uc, Ud, dp)
    return {"p": p}


def run_datacube_no_controller(
    z: np.ndarray,
    pp: PayoffParams,
    dp: DynamicsParams,
    cp: CouplingParams,
    rp: RunParams,
    n_players: int = 8,
    territories_per_player: int = 50,
    world_dims: Tuple[int, int, int] = (128, 128, 128),
    world_seed: int = 1234,
    seed_first_territory_id: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Datacube world coupled to mean-field replicator — no EchoKey controller.

    At each timestep the world advances and its metrics modify payoffs via the
    coupling functions, but NO grid search is performed. The replicator runs
    with delta_b=0, delta_d=0, u=0 — pure structural feedback, zero intervention.

    This is the critical comparison run. Two possible outcomes:

      p_T ≈ full run (0.997): controller is genuinely redundant. The structural
        feedback alone drives cooperation to the attractor. The model can be
        simplified by removing the controller entirely.

      p_T << full run: the controller is doing meaningful early-stage work,
        holding cooperation above the floor while conversion bootstraps. The
        controller earns its place even though its avg interventions look small.

    Returns the same trace dict shape as run_datacube_controller (minus
    controller-specific traces u, delta_b, delta_d, h, tau, s, viol).
    """
    wp = WorldParams()
    world = DatacubeWorld(dims=world_dims, n_players=n_players, wp=wp, seed=world_seed)
    world.initialize_game(territories_per_player=territories_per_player)
    world.seed_first_datacube(seed_first_territory_id, logger=None, resonance_boost=1.0)

    p = np.zeros(rp.T, dtype=np.float64)
    r_conv = np.zeros(rp.T, dtype=np.float64)
    rho_conn = np.zeros(rp.T, dtype=np.float64)
    ent = np.zeros(rp.T, dtype=np.float64)
    z_eff_trace = np.zeros(rp.T, dtype=np.float64)
    b_trace = np.zeros(rp.T, dtype=np.float64)
    c_trace = np.zeros(rp.T, dtype=np.float64)
    d_trace = np.zeros(rp.T, dtype=np.float64)
    phi_trace = np.zeros(rp.T, dtype=np.float64)

    p[0] = rp.p0

    for t in range(rp.T - 1):
        world.step(step_idx=t, logger=None)

        wm = WorldMetrics(
            conversion_rate=world.conversion_rate,
            connection_density=world.connection_density,
            avg_entropy=world.avg_entropy,
            avg_efficiency=world.avg_efficiency,
            avg_resonance=world.avg_resonance,
        )

        r_conv[t] = wm.conversion_rate
        rho_conn[t] = wm.connection_density
        ent[t] = wm.avg_entropy

        # No controller — delta_b=0, delta_d=0, pure structural payoffs
        Uc, Ud, aux = payoffs_coupled(p[t], float(z[t]), pp, wm, cp)
        p[t + 1] = replicator_step(p[t], Uc, Ud, dp)

        z_eff_trace[t] = float(aux["z_eff"])
        b_trace[t] = float(aux["b"])
        c_trace[t] = float(aux["c"])
        d_trace[t] = float(aux["d"])
        phi_trace[t] = float(aux["phi"])

    # Finalize last step
    world.step(step_idx=rp.T - 1, logger=None)
    wm_last = WorldMetrics(
        conversion_rate=world.conversion_rate,
        connection_density=world.connection_density,
        avg_entropy=world.avg_entropy,
        avg_efficiency=world.avg_efficiency,
        avg_resonance=world.avg_resonance,
    )
    r_conv[-1] = wm_last.conversion_rate
    rho_conn[-1] = wm_last.connection_density
    ent[-1] = wm_last.avg_entropy
    _, _, aux_last = payoffs_coupled(p[-1], float(z[-1]), pp, wm_last, cp)
    z_eff_trace[-1] = float(aux_last["z_eff"])
    b_trace[-1] = float(aux_last["b"])
    c_trace[-1] = float(aux_last["c"])
    d_trace[-1] = float(aux_last["d"])
    phi_trace[-1] = float(aux_last["phi"])

    return {
        "p": p,
        "r_conv": r_conv,
        "rho_conn": rho_conn,
        "avg_entropy": ent,
        "z_eff": z_eff_trace,
        "b": b_trace,
        "c": c_trace,
        "d": d_trace,
        "phi": phi_trace,
    }


def run_datacube_controller(
    z: np.ndarray,
    pp: PayoffParams,
    dp: DynamicsParams,
    ek: EchoKeyParams,
    cp: CouplingParams,
    rp: RunParams,
    # World configuration
    n_players: int = 8,
    territories_per_player: int = 50,
    world_dims: Tuple[int, int, int] = (128, 128, 128),
    world_seed: int = 1234,
    log_dir: Optional[str] = "game_logs",
    log_every_n_steps: int = 1,
    log_all_territories: bool = False,
    territory_sample_size: int = 100,
    seed_first_territory_id: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Full coupled simulation: Datacube world + EchoKey controller + mean-field game.

    At each timestep t:
      1. world.step(t): advance spatial world, update metrics
      2. Extract WorldMetrics snapshot
      3. datacube_control_step: controller searches for best (delta_b, delta_d, u)
      4. Store p[t+1] from controller's chosen p_next
      5. Record all diagnostic traces

    Note: world.step is called BEFORE the controller, so world metrics at step t
    reflect the state of the world BEFORE any mean-field change at step t.
    This is causal: you observe the world, then decide. No lookahead.

    Returns a dict of arrays (length T) for all tracked quantities.
    """
    wp = WorldParams()
    world = DatacubeWorld(dims=world_dims, n_players=n_players, wp=wp, seed=world_seed)
    if log_dir is None:
        logger = None
    else:
        logger = WorldLogger(
            log_dir=log_dir,
            log_every_n_steps=log_every_n_steps,
            log_all_territories=log_all_territories,
            territory_sample_size=territory_sample_size,
            seed=world_seed,
        )
    world.initialize_game(territories_per_player=territories_per_player)
    world.seed_first_datacube(seed_first_territory_id, logger=logger, resonance_boost=1.0)

    # Allocate trace arrays
    p = np.zeros(rp.T, dtype=np.float64)
    u = np.zeros(rp.T, dtype=np.float64)
    db = np.zeros(rp.T, dtype=np.float64)
    dd = np.zeros(rp.T, dtype=np.float64)
    h = np.zeros(rp.T, dtype=np.float64)
    tau = np.zeros(rp.T, dtype=np.float64)
    s = np.zeros(rp.T, dtype=np.float64)
    viol = np.zeros(rp.T, dtype=np.float64)

    r_conv = np.zeros(rp.T, dtype=np.float64)
    rho_conn = np.zeros(rp.T, dtype=np.float64)
    ent = np.zeros(rp.T, dtype=np.float64)
    z_eff_trace = np.zeros(rp.T, dtype=np.float64)
    b_trace = np.zeros(rp.T, dtype=np.float64)
    c_trace = np.zeros(rp.T, dtype=np.float64)
    d_trace = np.zeros(rp.T, dtype=np.float64)
    phi_trace = np.zeros(rp.T, dtype=np.float64)

    p[0] = rp.p0

    for t in range(rp.T - 1):
        # Step 1: Advance world FIRST to get current metrics
        world.step(step_idx=t, logger=logger)

        wm = WorldMetrics(
            conversion_rate=world.conversion_rate,
            connection_density=world.connection_density,
            avg_entropy=world.avg_entropy,
            avg_efficiency=world.avg_efficiency,
            avg_resonance=world.avg_resonance,
        )

        r_conv[t] = wm.conversion_rate
        rho_conn[t] = wm.connection_density
        ent[t] = wm.avg_entropy

        # Step 2: Controller chooses levers, advances population
        res = datacube_control_step(p[t], float(z[t]), pp, dp, ek, wm, cp)

        db[t] = res["delta_b"]
        dd[t] = res["delta_d"]
        u[t] = res["u"]
        p[t + 1] = res["p_next"]

        h[t] = res["h"]
        tau[t] = res["tau"]
        s[t] = res["s"]
        viol[t] = res["viol"]

        z_eff_trace[t] = res["z_eff"]
        b_trace[t] = res["b"]
        c_trace[t] = res["c"]
        d_trace[t] = res["d"]
        phi_trace[t] = res["phi"]

    # Finalize last step (world advances one more time, fill last-step diagnostics)
    world.step(step_idx=rp.T - 1, logger=logger)
    wm_last = WorldMetrics(
        conversion_rate=world.conversion_rate,
        connection_density=world.connection_density,
        avg_entropy=world.avg_entropy,
        avg_efficiency=world.avg_efficiency,
        avg_resonance=world.avg_resonance,
    )
    r_conv[-1] = wm_last.conversion_rate
    rho_conn[-1] = wm_last.connection_density
    ent[-1] = wm_last.avg_entropy

    u_prev = float(u[-2]) if rp.T >= 2 else 0.0
    _, _, aux_last = payoffs_coupled(p[-1], float(z[-1]), pp, wm_last, cp,
                                     delta_b=float(db[-2]), delta_d=float(dd[-2]))
    z_eff_last = float(aux_last["z_eff"])
    s_tilde_last = preharm_stability(p[-1], z_eff_last, u_prev, ek)
    h[-1] = harm_model(z_eff_last, u_prev, ek)
    tau[-1] = dynamic_tau(s_tilde_last, ek)
    s[-1] = clamp(s_tilde_last - ek.kappa * h[-1], 0.0, 1.0)
    viol[-1] = max(0.0, h[-1] - tau[-1])

    z_eff_trace[-1] = z_eff_last
    b_trace[-1] = float(aux_last["b"])
    c_trace[-1] = float(aux_last["c"])
    d_trace[-1] = float(aux_last["d"])
    phi_trace[-1] = float(aux_last["phi"])

    if logger is not None:
        stamp = logger.save()
        print(f"[Datacube logs saved] dir={log_dir} stamp={stamp}")

    return {
        "p": p,
        "u": u,
        "delta_b": db,
        "delta_d": dd,
        "h": h,
        "tau": tau,
        "s": s,
        "viol": viol,
        "r_conv": r_conv,
        "rho_conn": rho_conn,
        "avg_entropy": ent,
        "z_eff": z_eff_trace,
        "b": b_trace,
        "c": c_trace,
        "d": d_trace,
        "phi": phi_trace,
    }


# =============================================================================
# SECTION 9: ANALYSIS AND STRESS-TESTING
# =============================================================================
# Three analysis tools to characterize the mechanism's robustness:
#
#   1. SENSITIVITY SWEEP: Run both baseline and coupled system across a range
#      of shock amplitudes. Measures how much cooperation degrades as stress
#      increases. The gap between baseline and coupled curves is "resilience gain."
#      Finds the "collapse cliff" — the shock amplitude where cooperation drops
#      below 0.5. The Datacube mechanism should shift this cliff to higher shock levels.
#
#   2. LEVER SWEEP: 2D grid search over (z_buffer_conv, c_reduce_conv) coupling
#      parameters. Generates a heatmap of final cooperation vs these two levers.
#      Shows which structural lever matters most (stress buffering vs cost reduction).
#      Answer: cost reduction tends to dominate at high stress; buffering matters at moderate.
#
#   3. PARETO FRONTIER: Sweep over tau_max (harm tolerance) and measure final
#      cooperation p_T vs average harm vs violation rate. Shows the tradeoff:
#      tighter harm constraints (lower tau_max) → less harm but potentially less
#      cooperation (controller can't intervene enough). The Pareto frontier identifies
#      the minimum harm tolerance needed to achieve high cooperation (e.g. p_T >= 0.99).

def run_sensitivity_sweep(
    shock_range: np.ndarray,
    rp: Optional[RunParams] = None,
    sp_base: Optional[StressParams] = None,
    pp: Optional[PayoffParams] = None,
    dp: Optional[DynamicsParams] = None,
    ek: Optional[EchoKeyParams] = None,
    cp: Optional[CouplingParams] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Sweep over shock amplitudes. For each amplitude, run both baseline and
    coupled simulation. Report final p_T and world conversion rate.
    Returns (baseline_p_T, datacube_p_T, datacube_r_conv) lists.
    """
    baseline_results: List[float] = []
    datacube_results: List[float] = []
    datacube_r_conv: List[float] = []

    rp = rp or RunParams(T=220, p0=0.22, seed=777)
    sp_base = sp_base or StressParams()
    pp = pp or PayoffParams()
    dp = dp or DynamicsParams(eta=0.70, mu=0.002)
    ek = ek or EchoKeyParams(grid_n=15)
    cp = cp or CouplingParams()

    print(f"{'Shock Amp':<12} | {'Base p_T':<10} | {'DC p_T':<10} | {'DC r_conv':<10}")
    print("-" * 50)

    for amp in shock_range:
        sp = replace(sp_base, shock_amp=float(amp))
        z = stress_process(rp.T, sp, seed=rp.seed)

        b_res = run_baseline(z, pp, dp, rp)
        d_res = run_datacube_controller(
            z=z, pp=pp, dp=dp, ek=ek, cp=cp, rp=rp,
            log_dir=None, log_every_n_steps=0,
        )

        baseline_results.append(float(b_res["p"][-1]))
        datacube_results.append(float(d_res["p"][-1]))
        datacube_r_conv.append(float(d_res["r_conv"][-1]))

        print(f"{float(amp):<12.2f} | {b_res['p'][-1]:<10.3f} | {d_res['p'][-1]:<10.3f} | {d_res['r_conv'][-1]:<10.3f}")

    return baseline_results, datacube_results, datacube_r_conv


def _first_below(x: np.ndarray, y: List[float], threshold: float) -> Optional[float]:
    """Find the first x value where y drops below threshold. Used to find cooperation collapse cliff."""
    for xi, yi in zip(x, y):
        if yi < threshold:
            return float(xi)
    return None


def plot_sensitivity_benchmark(
    shocks: np.ndarray,
    b_p: List[float],
    d_p: List[float],
    collapse_threshold: float = 0.5,
) -> float:
    """
    Plot baseline vs Datacube-coupled cooperation across shock amplitudes.
    Shaded area = resilience gain. Red dashed line = cooperation collapse threshold.
    Prints collapse cliff locations and total resilience score (area under gap curve).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(shocks, b_p, "o--", label="Baseline (Selfish Nash)")
    plt.plot(shocks, d_p, "s-", label="Datacube-Coupled (Mechanism)")
    plt.fill_between(shocks, b_p, d_p, color="green", alpha=0.1, label="Resilience Gain")

    plt.axhline(y=collapse_threshold, color="r", linestyle=":", label="Collapse Threshold")
    plt.xlabel("Shock Amplitude (System Stress)")
    plt.ylabel("Final Cooperation Fraction p_T")
    plt.title("Benchmarking Social Stability: Baseline vs. Datacube-Coupled")
    plt.legend()
    plt.grid(True, alpha=0.3)

    base_cliff = _first_below(shocks, b_p, collapse_threshold)
    dc_cliff = _first_below(shocks, d_p, collapse_threshold)
    if base_cliff is not None:
        print(f"Baseline cliff near shock_amp={base_cliff:.2f}")
    if dc_cliff is not None:
        print(f"Datacube cliff near shock_amp={dc_cliff:.2f}")

    resilience_score = float(np.trapezoid(d_p, shocks) - np.trapezoid(b_p, shocks))
    print(f"Resilience score (area gain) = {resilience_score:.4f}")
    return resilience_score

def run_lever_sweep(
    z_buff_range: np.ndarray,
    c_red_range: np.ndarray,
) -> np.ndarray:
    """
    2D sweep over (z_buffer_conv, c_reduce_conv) coupling parameters.

    For each combination, run the full coupled simulation and record final p_T.
    Returns a 2D array of shape (len(z_buff_range), len(c_red_range)).

    Useful for identifying which structural levers have the most leverage.
    The heatmap typically shows that cost reduction (c_reduce_conv) is more
    powerful than stress buffering (z_buffer_conv) at high shock levels,
    while buffering matters more when shocks are moderate.
    """
    rp = RunParams(T=200, p0=0.22, seed=777)
    sp = StressParams(shock_amp=0.65)   # high shock to make the comparison interesting
    z = stress_process(rp.T, sp, seed=rp.seed)

    results_grid = np.zeros((len(z_buff_range), len(c_red_range)), dtype=np.float64)

    print(f"Testing {len(z_buff_range) * len(c_red_range)} configurations...")

    for i, z_buff in enumerate(z_buff_range):
        for j, c_red in enumerate(c_red_range):
            cp = CouplingParams(z_buffer_conv=float(z_buff), c_reduce_conv=float(c_red))
            pp = PayoffParams()
            dp = DynamicsParams(eta=0.70, mu=0.002)
            ek = EchoKeyParams(grid_n=13)

            d_res = run_datacube_controller(
                z=z, pp=pp, dp=dp, ek=ek, cp=cp, rp=rp,
                log_dir=None, log_every_n_steps=0,
            )
            results_grid[i, j] = float(d_res["p"][-1])

    return results_grid


def plot_lever_heatmap(
    heatmap_data: np.ndarray,
    z_range: np.ndarray,
    c_range: np.ndarray,
) -> None:
    """Render the lever sweep as a colored 2D heatmap (red=low cooperation, green=high)."""
    plt.figure(figsize=(8, 6))
    plt.imshow(
        heatmap_data,
        extent=[c_range[0], c_range[-1], z_range[0], z_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="RdYlGn",
    )
    plt.colorbar(label="Final Cooperation p_T")
    plt.xlabel("Cost Reduction Lever (c_reduce_conv)")
    plt.ylabel("Stress Buffer Lever (z_buffer_conv)")
    plt.title("Stability Landscape: Which Lever Wins?")


def find_pareto_frontier(
    tau_steps: np.ndarray,
    rp: Optional[RunParams] = None,
    sp: Optional[StressParams] = None,
    cp: Optional[CouplingParams] = None,
    pp: Optional[PayoffParams] = None,
    dp: Optional[DynamicsParams] = None,
) -> List[Tuple[float, float, float, float]]:
    """
    Sweep over tau_max (harm threshold) and record:
      (tau_max, final_p_T, avg_harm, violation_rate)

    This is the harm-cooperation Pareto frontier. Lower tau_max = tighter
    constraint = less harm tolerated = controller must be more conservative.
    The question is: what is the minimum tau_max that still achieves high cooperation?

    Returns list of (tau_val, p_final, avg_harm, viol_rate) tuples.
    Prints a table to stdout.
    """
    results: List[Tuple[float, float, float, float]] = []

    rp = rp or RunParams(T=220, p0=0.22, seed=777)
    sp = sp or StressParams(shock_amp=0.65)
    z = stress_process(rp.T, sp, seed=rp.seed)

    cp = cp or CouplingParams(z_buffer_conv=0.2, c_reduce_conv=0.3)
    pp = pp or PayoffParams()
    dp = dp or DynamicsParams(eta=0.70, mu=0.002)

    print(f"{'Tau Max':<10} | {'Resilience':<12} | {'Avg Harm':<10} | {'Violations'}")
    print("-" * 55)

    for tau_val in tau_steps:
        ek = EchoKeyParams(tau_max=float(tau_val), grid_n=15)
        dc_res = run_datacube_controller(
            z=z, pp=pp, dp=dp, ek=ek, cp=cp, rp=rp,
            log_dir=None, log_every_n_steps=0,
        )

        p_final = float(dc_res["p"][-1])
        avg_h = float(np.mean(dc_res["h"]))
        viols = float(np.mean(dc_res["viol"] > 0))

        results.append((float(tau_val), p_final, avg_h, viols))
        print(f"{tau_val:<10.3f} | {p_final:<12.3f} | {avg_h:<10.3f} | {viols:.3f}")

    return results


def _first_tau_meeting_target(
    pareto_data: List[Tuple[float, float, float, float]],
    target: float = 0.99,
) -> Optional[float]:
    """
    Find the smallest tau_max in the sweep that achieves p_T >= target.
    Returns None if no tau meets the target. Used to report the 'minimum
    harm tolerance needed for high cooperation.'
    """
    for tau_val, p_final, _avg_h, _viols in pareto_data:
        if p_final >= target:
            return float(tau_val)
    return None


# =============================================================================
# SECTION 10: MAIN — PARAMETER CHOICES AND OUTPUT GENERATION
# =============================================================================
# The main function runs:
#   1. Baseline simulation (no controller, no coupling)
#   2. Full Datacube-coupled simulation
#   3. Diagnostic printout comparing the two
#   4. Seven diagnostic plots
#   5. (Optional) Sensitivity sweep, lever heatmap, Pareto frontier
#
# All three sweeps are enabled by default (they're slow but informative).
# Set the flags to False to skip them for quick runs.
#
# KEY PARAMETER CHOICES IN MAIN:
#   tau_max=0.010   : Very tight harm constraint — controller is cautious.
#                     This forces reliance on structural coupling rather than
#                     aggressive suppression. The point is to show the mechanism
#                     works even when intervention is heavily constrained.
#   Lambda=18.0     : Very heavy violation penalty — harm violations are essentially
#                     forbidden, reinforcing the tight constraint.
#   grid_n=19       : 19^3 = ~6859 grid points per step. Reasonably fine resolution.
#   z_buffer_conv=0.20 : Moderate stress buffering in main run (0.35 in defaults).
#                        More conservative than maximum.
#   c_reduce_conv=0.30 : 30% cost reduction at full conversion — meaningful but not extreme.
#   d_boost_conn=0.45  : Strong defector friction from connectivity — this is the
#                         dominant structural mechanism.

def main():
    # Toggle analysis sweeps (slow — disable for quick development runs)
    run_sensitivity_sweep_flag = True
    run_lever_sweep_flag = True
    run_pareto_sweep_flag = True

    # Chart output directory (saved alongside game_logs/)
    chart_dir = "game_charts"
    os.makedirs(chart_dir, exist_ok=True)
    chart_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_fig(name: str) -> None:
        """Save the current matplotlib figure to chart_dir with a timestamp prefix."""
        path = os.path.join(chart_dir, f"{chart_stamp}_{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")


    # Simulation setup
    rp = RunParams(T=220, p0=0.22, seed=777)
    sp = StressParams()
    pp = PayoffParams()
    dp = DynamicsParams(eta=0.70, mu=0.002)

    # EchoKey controller configuration
    # Note: tau_max=0.010 is intentionally tight — this limits the controller to
    # gentle interventions and forces the structural coupling to do most of the work.
    ek = EchoKeyParams(
        tau_max=0.010,
        harm_sensitivity=1.65,
        kappa=1.10,
        Lambda=18.0,
        grid_n=19,
        w_next=1.2,
        beta=0.25,
        delta_b_max=0.85,
        delta_d_max=0.85,
        u_max=1.0,
    )

    # Datacube coupling configuration
    # These values are tuned so that the Datacube network, once it reaches
    # moderate conversion (~30-50%), creates a cooperative lock-in.
    # d_boost_conn=0.45 is the key driver — connectivity strongly enforces cooperation.
    cp = CouplingParams(
        z_buffer_conv=0.20,
        b_boost_conv=0.25,
        b_boost_conn=0.10,
        c_reduce_conv=0.30,
        c_reduce_conn=0.15,
        d_boost_conn=0.45,
        d_boost_conv=0.15,
        phi_boost_conv=0.06,
        phi_boost_conn=0.05,
        phi_boost_entropy=0.02,
        c_z_dampen_conv=0.25,
    )

    # Generate shared stress process (same seed = same shocks for both runs)
    z = stress_process(rp.T, sp, seed=rp.seed)

    # Run all three simulations
    base = run_baseline(z, pp, dp, rp)

    dc_no_ctrl = run_datacube_no_controller(
        z=z,
        pp=pp,
        dp=dp,
        cp=cp,
        rp=rp,
        n_players=8,
        territories_per_player=50,
        world_dims=(128, 128, 128),
        world_seed=1234,
        seed_first_territory_id=1,
    )

    dc = run_datacube_controller(
        z=z,
        pp=pp,
        dp=dp,
        ek=ek,
        cp=cp,
        rp=rp,
        n_players=8,
        territories_per_player=50,
        world_dims=(128, 128, 128),
        world_seed=1234,
        log_dir="game_logs",
        log_every_n_steps=1,
        log_all_territories=False,
        territory_sample_size=100,
        seed_first_territory_id=1,
    )

    # Diagnostics
    viol_rate = float(np.mean(dc["viol"] > 0.0))
    avg_p_dc_world = float(np.mean(dc["r_conv"]))

    print("\n=== Diagnostics ===")
    print(f"Baseline:      p_T={base['p'][-1]:.3f}, avg p={float(np.mean(base['p'])):.3f}")
    print(f"World only:    p_T={dc_no_ctrl['p'][-1]:.3f}, avg p={float(np.mean(dc_no_ctrl['p'])):.3f}  (coupled, no controller)")
    print(f"Datacube+ctrl: p_T={dc['p'][-1]:.3f}, avg p={float(np.mean(dc['p'])):.3f}")
    print(f"Datacube:  avg p_dc={avg_p_dc_world:.3f} (world conversion fraction)")
    print(f"Datacube:  avg u={float(np.mean(dc['u'])):.3f}, avg Δb={float(np.mean(dc['delta_b'])):.3f}, avg Δd={float(np.mean(dc['delta_d'])):.3f}")
    print(f"Datacube:  avg harm={float(np.mean(dc['h'])):.3f}, avg tau={float(np.mean(dc['tau'])):.3f}, violation-rate={viol_rate:.3f}")
    print(f"Coupling:  avg r={float(np.mean(dc['r_conv'])):.3f}, avg rho={float(np.mean(dc['rho_conn'])):.3f}, avg entropy={float(np.mean(dc['avg_entropy'])):.3f}")
    print(f"Payoffs:   avg b={float(np.mean(dc['b'])):.3f}, avg c={float(np.mean(dc['c'])):.3f}, avg d={float(np.mean(dc['d'])):.3f}, avg phi={float(np.mean(dc['phi'])):.3f}")
    print(f"Stress:    avg z={float(np.mean(z)):.3f}, avg z_eff={float(np.mean(dc['z_eff'])):.3f}")

    # Plotting: seven diagnostic plots covering all key dynamics
    t = np.arange(rp.T)

    # Plot 1: Core result — all three cooperation trajectories
    plt.figure()
    plt.plot(t, base["p"], label="Baseline p(t)")
    plt.plot(t, dc_no_ctrl["p"], label="World-only p(t) (no controller)", linestyle="--")
    plt.plot(t, dc["p"], label="Datacube-coupled p(t) (full)")
    plt.xlabel("t")
    plt.ylabel("Cooperator fraction p")
    plt.title("Mean-field cooperation under stress: Baseline vs World-Only vs Full")
    plt.legend()
    plt.tight_layout()
    save_fig("01_cooperation_trajectory")

    # Plot 2: Stress buffering — how much effective stress is reduced vs raw
    plt.figure()
    plt.plot(t, z, label="Stress z(t)")
    plt.plot(t, dc["z_eff"], label="Effective stress z_eff(t)")
    plt.xlabel("t")
    plt.ylabel("Stress")
    plt.title("Stress and buffered stress (coupled to world conversion)")
    plt.legend()
    plt.tight_layout()
    save_fig("02_stress_buffering")

    # Plot 3: World expansion — conversion and connectivity over time
    plt.figure()
    plt.plot(t, dc["r_conv"], label="World conversion_rate r(t)")
    plt.plot(t, dc["rho_conn"], label="Connection density ρ(t)")
    plt.xlabel("t")
    plt.ylabel("World metrics")
    plt.title("Datacube world expansion metrics (coupling sources)")
    plt.legend()
    plt.tight_layout()
    save_fig("03_world_expansion")

    # Plot 4: Controller suppression usage
    plt.figure()
    plt.plot(t, dc["u"], label="Suppression u(t)")
    plt.xlabel("t")
    plt.ylabel("u")
    plt.title("Datacube controller suppression usage")
    plt.legend()
    plt.tight_layout()
    save_fig("04_suppression")

    # Plot 5: Harm vs dynamic threshold (constraint tracking)
    plt.figure()
    plt.plot(t, dc["h"], label="Harm h(t)")
    plt.plot(t, dc["tau"], label="Threshold tau(t)")
    plt.xlabel("t")
    plt.ylabel("h / tau")
    plt.title("Harm vs dynamic threshold")
    plt.legend()
    plt.tight_layout()
    save_fig("05_harm_vs_threshold")

    # Plot 6: Intervention levers over time
    plt.figure()
    plt.plot(t, dc["delta_b"], label="Δb(t)")
    plt.plot(t, dc["delta_d"], label="Δd(t)")
    plt.xlabel("t")
    plt.ylabel("Intervention")
    plt.title("Datacube controller interventions (rule-tuning)")
    plt.legend()
    plt.tight_layout()
    save_fig("06_interventions")

    # Plot 7: Payoff coefficient evolution (the real lever on U_C - U_D)
    plt.figure()
    plt.plot(t, dc["c"], label="c(t)")
    plt.plot(t, dc["phi"], label="phi(t)")
    plt.plot(t, dc["d"], label="d(t)")
    plt.xlabel("t")
    plt.ylabel("Coeff value")
    plt.title("Coupled payoff coefficients (the real lever on Uc-Ud)")
    plt.legend()
    plt.tight_layout()
    save_fig("07_payoff_coefficients")

    # Analysis sweeps (slow)
    if run_sensitivity_sweep_flag:
        shocks = np.linspace(0.1, 0.9, 15)
        ek_sweep = replace(ek, grid_n=15)
        b_p, d_p, _ = run_sensitivity_sweep(shocks, rp=rp, sp_base=sp, pp=pp, dp=dp, ek=ek_sweep, cp=cp)
        plot_sensitivity_benchmark(shocks, b_p, d_p)
        save_fig("08_sensitivity_sweep")

    if run_lever_sweep_flag:
        z_range = np.linspace(0.0, 0.8, 8)
        c_range = np.linspace(0.0, 0.8, 8)
        heatmap_data = run_lever_sweep(z_range, c_range)
        plot_lever_heatmap(heatmap_data, z_range, c_range)
        save_fig("09_lever_heatmap")

    if run_pareto_sweep_flag:
        tau_sweep = np.linspace(0.3, 0.01, 15)
        pareto_data = find_pareto_frontier(tau_sweep)
        tau_star = _first_tau_meeting_target(pareto_data, target=0.99)
        if tau_star is not None:
            print(f"Lowest tau_max with p_T >= 0.99 is {tau_star:.3f}")
        else:
            print("No tau_max hit p_T >= 0.99 in the sweep range.")

    plt.show()


if __name__ == "__main__":
    main()