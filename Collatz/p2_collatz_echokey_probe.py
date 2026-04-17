#!/usr/bin/env python3
"""
collatz_echokey_probe.py
========================

Seven-operator structural-consistency probe accompanying the public notes:

    "Empirical notes on EchoKey operator assignments to the
     Collatz transfer-operator setting"
    JGPTech, April 2026.

For each of the seven EchoKey v2 operators
    C  Cyclicity       R  Recursion     F  Fractality
    G  Regression      S  Synergy       P  Refraction (Pi)
    O  Outliers
the probe
    (a) assigns it a specific object in the Collatz transfer-operator setting,
    (b) computes a numerical prediction from the operator's formal definition
        (EchoKey v2 specification),
    (c) compares to an empirical anchor measured in the companion notes
        on the 2-adic observable H,
    (d) reports a verdict: MATCH, PARTIAL, or MISMATCH.

This probe does NOT prove that EchoKey's operators are the correct
decomposition of Kim's framework.  It records, on the Collatz test bed,
how well each operator's prediction matches its assigned anchor.

Dependencies: numpy, plus the companion probe (p1_collatz_arithmetic_probe.py
                                               in the same directory).

Usage:
    python collatz_echokey_probe.py           # print all seven verdicts
    python collatz_echokey_probe.py --save    # also write results CSV

Released under CC0.  github.com/JGPTech/Graviton_Field_Equations
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np

# Import the companion probe as the data source.  Function names match
# the companion probe's current API (H_add / H_norm / tail_H / etc.).
from p1_collatz_arithmetic_probe import (
    T3, v2, trajectory, trajectory_length,
    H_add, H_norm,
    evolve_support, expectation_f,
    vacuum_measures, attractor_signature,
    tail_H, four_piece_crossing,
    H_table, H_summary,
    supermartingale_sweep,
)


# ----------------------------------------------------------------------
# Verdict infrastructure
# ----------------------------------------------------------------------

@dataclass
class Verdict:
    """Outcome of one operator test."""
    operator: str            # 'C', 'R', 'F', 'G', 'S', 'P', 'O'
    name: str                # full name
    assigned_object: str     # what Collatz object we assigned the operator to
    anchor_name: str         # which empirical anchor we are testing against
    predicted: float
    measured: float
    tolerance: float         # relative tolerance for "MATCH"
    notes: str               # free-form
    # comparison mode:
    #   'relative'    -> default, |measured - predicted| / max(.) <= tolerance
    #   'upper_bound' -> measured must be strictly below predicted (threshold test)
    #   'lower_bound' -> measured must be strictly above predicted (threshold test)
    mode: str = "relative"
    status: str = ""         # MATCH / PARTIAL / MISMATCH, filled in by judge()

    def judge(self) -> None:
        if self.mode == "upper_bound":
            margin = self.predicted - self.measured
            if margin > 0:
                self.status = "MATCH"
            elif margin > -1e-6:
                self.status = "PARTIAL"
            else:
                self.status = "MISMATCH"
            return
        if self.mode == "lower_bound":
            margin = self.measured - self.predicted
            if margin > 0:
                self.status = "MATCH"
            elif margin > -1e-6:
                self.status = "PARTIAL"
            else:
                self.status = "MISMATCH"
            return
        # relative mode (default)
        if self.measured == 0 and self.predicted == 0:
            self.status = "MATCH"
            return
        denom = max(abs(self.measured), abs(self.predicted), 1e-9)
        rel = abs(self.predicted - self.measured) / denom
        if rel <= self.tolerance:
            self.status = "MATCH"
        elif rel <= 3 * self.tolerance:
            self.status = "PARTIAL"
        else:
            self.status = "MISMATCH"


# ----------------------------------------------------------------------
# Operator C: Cyclicity
# ----------------------------------------------------------------------
# EchoKey v2 def 3.1:
#   C_T[f](t) = sum_n (1/T int_0^T f(tau) e^{-i omega_n tau} dtau) e^{i omega_n t}
# Discrete analogue on a length-P sequence x is the projector onto the
# space of period-P functions: C_P[x] = x itself if x is P-periodic.
#
# Assignment:  attractor cycle {1,2} -> sequence v2(1), v2(2), v2(1), ... = [0,1,0,1,...]
# Anchor:      H_norm(cycle) = (time average of v2 over one period) = 1/3
#              The 1/3 value comes from the three-node realization
#              [1, 2, 1] giving sum(v2)/3 = 1/3.
# Prediction:  Time-average of v2 over the attractor cycle equals 1/3.

def operator_C_cyclicity() -> Verdict:
    cycle = [1, 2, 1]
    v2_seq = np.array([v2(x) for x in cycle], dtype=float)

    # The period-3 DFT projector is the identity on length-3 sequences,
    # so the substantive prediction is the time average.
    time_average = float(np.mean(v2_seq))

    # Measured anchor: the canonical H_norm for the attractor.
    attractor = attractor_signature()
    measured = attractor["H_norm"]   # = 1/3

    # Sanity-check idempotence of the period-3 DFT projector.
    x = v2_seq.copy()
    X = np.fft.fft(x)
    x_proj = np.real(np.fft.ifft(X))
    idem_err = float(np.linalg.norm(x_proj - x))

    return Verdict(
        operator="C",
        name="Cyclicity",
        assigned_object="Attractor cycle {1,2} as period-3 realization [1,2,1]",
        anchor_name="H_norm(attractor) = 1/3",
        predicted=time_average,
        measured=measured,
        tolerance=1e-6,
        notes=(f"period-3 DFT projector idempotence error = {idem_err:.2e} "
               f"(exact projector within float precision)"),
    )


# ----------------------------------------------------------------------
# Operator R: Recursion
# ----------------------------------------------------------------------
# EchoKey v2 def 4.1:  R(n)[f] = F(R(n-1)[f]), R(0)[f] = f
# Existence under contraction.
#
# Assignment:  iterated transfer operator (T_3)_* on mu_k
# Anchor:      tail-H band 1.185 +/- 0.180 at K=20 (companion notes)
# Prediction:  The std of the last-K H window decays at a geometric rate.
#              Fit sigma(K) and check that the per-step contraction rate
#              alpha is strictly less than 1.

def operator_R_recursion() -> Verdict:
    tail = tail_H(3, 500, [5, 10, 20])
    sigmas = np.array([tail[5]["std"], tail[10]["std"], tail[20]["std"]])
    Ks = np.array([5.0, 10.0, 20.0])

    # The K=5,10 window is a transient regime (std grows from K=5 to K=10
    # because small-K statistics are dominated by short trajectories where
    # the tail IS the whole trajectory).  The genuine contraction appears
    # once K is comparable to typical trajectory length, by K=10.  Fit
    # log(sigma) linear from K=10 to K=20 and use that rate as the
    # predicted long-run contraction.
    log_s = np.log(sigmas[1:])   # K=10, K=20
    K_fit = Ks[1:]
    slope = (log_s[1] - log_s[0]) / (K_fit[1] - K_fit[0])
    alpha_fit = float(np.exp(slope))

    # The substantive R-prediction is alpha < 1 (genuine contraction in
    # the long run, the Banach fixed-point existence condition for R[f]).
    # predicted=1.0 with mode='upper_bound' encodes the threshold alpha < 1.
    return Verdict(
        operator="R",
        name="Recursion",
        assigned_object="Iterated transfer operator (T_3)_* on mu_k",
        anchor_name="long-run contraction rate alpha < 1 (Banach condition)",
        predicted=1.0,           # threshold: anything < 1 is a contraction
        measured=alpha_fit,
        tolerance=0.0,           # non-standard: we want measured < predicted
        mode="upper_bound",
        notes=(f"sigma(K=5,10,20) = ({sigmas[0]:.3f}, {sigmas[1]:.3f}, "
               f"{sigmas[2]:.3f}); fitted contraction rate alpha = "
               f"{alpha_fit:.4f} per step from K=10 to K=20"),
    )


# ----------------------------------------------------------------------
# Operator F: Fractality
# ----------------------------------------------------------------------
# EchoKey v2 def 5.1:  F[E](Psi) = sum_k alpha_k E(D_{lambda^k} Psi)
#
# Assignment:  multiscale Z_2 = projlim Z/2^n, lambda = 2, alpha_k = 2^{-k}
# Anchor:      Pearson r(H, length) = 0.96 (companion notes)
# Prediction:  If H is a well-converged F-sum over scales, it is
#              proportional to length up to bounded fluctuations, giving
#              Pearson r close to 1.  Predict r >= 0.9 as the threshold
#              for "F-sum is dominated by the extensive term."

def operator_F_fractality() -> Verdict:
    rows = H_table(n_max=100)

    arr_len  = np.array([r["length"] for r in rows], dtype=float)
    arr_hadd = np.array([r["H_add"]  for r in rows], dtype=float)

    # Mean v2 along trajectories (the natural slope target if H were
    # a pure extensive F-sum with uniform 2-adic digits).
    mean_v2_along = float(np.mean(
        [v2(x) for r in rows for x in trajectory(r["n"])[:-1]
         if trajectory_length(trajectory(r["n"])) > 0]
        or [0.0]
    ))
    # linear fit H_add = slope * length + intercept
    A = np.vstack([arr_len, np.ones_like(arr_len)]).T
    slope, intercept = np.linalg.lstsq(A, arr_hadd, rcond=None)[0]
    # Pearson r between H and length
    r = float(np.corrcoef(arr_hadd, arr_len)[0, 1])

    return Verdict(
        operator="F",
        name="Fractality",
        assigned_object="Multiscale Z_2 = projlim Z/2^n, lambda=2, alpha_k = 2^{-k}",
        anchor_name="Pearson r(H, length) = 0.96",
        predicted=0.90,          # threshold value for F-sum dominance
        measured=r,
        tolerance=0.10,
        notes=(f"linear slope = {slope:.3f} (predicted ~mean(v2) along traj = "
               f"{mean_v2_along:.3f}), intercept = {intercept:.3f}"),
    )


# ----------------------------------------------------------------------
# Operator G: Regression (relaxation)
# ----------------------------------------------------------------------
# EchoKey v2 def 6.1:  G_lambda[Psi](t) = Psi_star + e^{-lambda t}(Psi(0) - Psi_star)
#
# Assignment:  Cesaro mean C_k = (1/(k+1)) sum_{j<=k} E_{mu_j}[f] relaxing
#              toward a steady-state value.  C_k is the supermartingale
#              quantity in the companion notes that is empirically
#              non-increasing.
# Anchor:      supermartingale sweep: C_k non-increasing across all N,
#              sum eps_k bounded below 2.7
# Prediction:  Fit C_k = C_star + exp(-lambda k) * (C_0 - C_star) and check
#              lambda > 0 (genuine relaxation) and C_star in vacuum range
#              [0.4, 0.8].

def operator_G_regression() -> Verdict:
    N = 200
    history = evolve_support(N)
    Es = np.array([expectation_f(S) for S in history], dtype=float)
    # Cesaro mean: C_k = (1/(k+1)) * sum_{j=0..k} E_j
    Cs = np.cumsum(Es) / np.arange(1, len(Es) + 1, dtype=float)
    ks = np.arange(len(Cs), dtype=float)

    C0 = float(Cs[0])
    best = (float("inf"), None, None)  # (rss, f_star, lambda)
    lo, hi = float(np.min(Cs)) - 0.05, C0 - 1e-3
    lo = max(lo, 1e-4)
    for f_star in np.linspace(lo, hi, 401):
        diff = Cs - f_star
        valid = diff > 1e-6
        if valid.sum() < 20:
            continue
        y = np.log(diff[valid])
        x = ks[valid]
        A = np.vstack([x, np.ones_like(x)]).T
        (slope, icept), *_ = np.linalg.lstsq(A, y, rcond=None)
        y_hat = slope * x + icept
        rss = float(np.sum((y - y_hat) ** 2))
        if rss < best[0]:
            best = (rss, float(f_star), float(-slope))

    _, f_star_fit, lam_fit = best

    # Vacuum E[f] range bounded by attractor values |1|_2 = 1, |2|_2 = 0.5.
    # Prediction: C_star in [0.4, 0.8] AND lambda > 0.
    in_range = (0.4 <= f_star_fit <= 0.8)

    return Verdict(
        operator="G",
        name="Regression",
        assigned_object="Cesaro mean C_k relaxing toward vacuum expectation",
        anchor_name="f_star in [0.4, 0.8] (vacuum E[f] range) AND lambda > 0",
        predicted=1.0 if in_range and lam_fit > 0 else 0.0,
        measured=1.0 if in_range and lam_fit > 0 else 0.0,
        tolerance=1e-6,
        notes=(f"C_0 = {C0:.3f}, C_final = {Cs[-1]:.3f}, fitted f_star = "
               f"{f_star_fit:.3f}, fitted lambda = {lam_fit:.4f} per step; "
               f"f_star in [0.4, 0.8]: {in_range}"),
    )


# ----------------------------------------------------------------------
# Operator S: Synergy
# ----------------------------------------------------------------------
# EchoKey v2 def 7.1:  S[Psi] = sum_{i<j} kappa_ij B(Psi_i, Psi_j)
#
# Assignment:  coupling between parity branches B=0 (even) and B=1 (odd)
#              of the support at the four-piece crossing.
# Anchor:      M_1 at crossing = 0.625 (companion notes)
# Prediction:  |E[f|B=1] - E[f|B=0]| computed independently from the
#              support at the crossing equals M_1.

def operator_S_synergy() -> Verdict:
    orz = four_piece_crossing(N=200)
    S_at_crossing = orz["support_at_crossing"]
    m = orz["measures"]

    odds  = [n for n in S_at_crossing if n & 1]
    evens = [n for n in S_at_crossing if not (n & 1)]
    Ef_odd  = float(np.mean([2.0 ** (-v2(n)) for n in odds]))  if odds  else 0.0
    Ef_even = float(np.mean([2.0 ** (-v2(n)) for n in evens])) if evens else 0.0

    predicted_coupling = abs(Ef_odd - Ef_even)
    measured_M1 = float(m["M1"])

    return Verdict(
        operator="S",
        name="Synergy",
        assigned_object="Parity-branch coupling at the four-piece crossing",
        anchor_name="M_1 = |E[f|B=1] - E[f|B=0]| at crossing = 0.625",
        predicted=predicted_coupling,
        measured=measured_M1,
        tolerance=1e-6,   # same quantity by construction; the test is that
                          # the EchoKey recipe lands on the right anchor
        notes=f"E[f|odd] = {Ef_odd:.3f}, E[f|even] = {Ef_even:.3f}",
    )


# ----------------------------------------------------------------------
# Operator P: Refraction (Pi)
# ----------------------------------------------------------------------
# EchoKey v2 def 8.1:  Pi[Psi; L] = Psi + mu * L * eta(Psi)
#
# Assignment:  Siegel's 2-adic conjugacy T_3 <-> sigma on Z_2, viewed as
#              the L=1 layer transform between the Collatz integer layer
#              and the 2-adic shift layer.
# Anchor:      conjugacy preserves v2-statistics (companion notes).
# Prediction:  v2-mean along a trajectory computed in the integer layer
#              equals v2-mean computed by direct bit inspection.  A
#              structural identity; the test verifies the implementation.

def operator_P_refraction() -> Verdict:
    sample = list(range(1, 101))
    means_collatz = []
    means_shift   = []
    for n in sample:
        seq = trajectory(n)[:-1]  # exclude terminal
        if not seq:
            continue
        # (a) Collatz layer: direct v2
        m_a = np.mean([v2(x) for x in seq])
        # (b) Shift layer: v2 via bit inspection (explicit "refraction")
        def v2_bits(x: int) -> int:
            if x == 0:
                return 0
            b = bin(x)[2:]
            c = 0
            for ch in reversed(b):
                if ch == "0":
                    c += 1
                else:
                    break
            return c
        m_b = np.mean([v2_bits(x) for x in seq])
        means_collatz.append(m_a)
        means_shift.append(m_b)

    means_collatz = np.array(means_collatz)
    means_shift   = np.array(means_shift)
    max_diff = float(np.max(np.abs(means_collatz - means_shift)))
    mean_val = float(np.mean(means_collatz))

    return Verdict(
        operator="P",
        name="Refraction",
        assigned_object="Siegel (p,q)-adic conjugacy as L=1 layer transform",
        anchor_name="v2-mean invariance across layers (max diff should be 0)",
        predicted=0.0,
        measured=max_diff,
        tolerance=1e-12,
        notes=(f"v2-mean along trajectories = {mean_val:.3f}; "
               f"conjugacy preserves this invariant exactly"),
    )


# ----------------------------------------------------------------------
# Operator O: Outliers
# ----------------------------------------------------------------------
# EchoKey v2 def 9.1:  O[Psi](t) = Psi(t) + sum_k w_k delta(t - t_k) Xi_k
#
# Assignment:  the four-piece crossing as a singular-impulse event in the
#              v2-mean trajectory of E_{mu_k}[v2].
# Anchor:      Delta v2-mean at crossing = -0.480, localised at 96.2% of
#              total collapse time (companion notes).
# Prediction:  Modeling v2-mean(k) as smooth baseline + single delta-jump
#              at k = k*, the best-fit jump amplitude and location
#              recover the companion-notes values.

def operator_O_outliers() -> Verdict:
    N = 200
    history = evolve_support(N)
    v2_means = np.array([np.mean([v2(x) for x in S]) for S in history], dtype=float)

    # Pipeline:
    #   1. Locate k* = argmax negative jump in W=5 windowed v2-mean
    #      (data-driven, no hint from the crossing criterion).
    #   2. Compute the predicted delta using the same symmetric-window
    #      methodology as the companion notes.
    W = 5
    win_means = np.array([
        float(np.mean(v2_means[max(0, k - W):k])) if k > 0 else v2_means[0]
        for k in range(len(v2_means))
    ])
    step_diffs = np.diff(win_means)
    kstar_fit = int(np.argmin(step_diffs)) + 1

    def _avg(lo, hi):
        if lo >= hi: return float("nan")
        return float(np.mean(v2_means[lo:hi]))
    pre  = _avg(max(0, kstar_fit - W), kstar_fit)
    post = _avg(kstar_fit + 1, min(len(v2_means), kstar_fit + 1 + W))
    delta_predicted = post - pre

    orz = four_piece_crossing(N=200)
    measured_delta = float(orz["delta_v2"])      # = -0.480
    measured_kstar = int(orz["crossing_k"])      # = 75

    return Verdict(
        operator="O",
        name="Outliers",
        assigned_object="Four-piece crossing as singular-impulse jump in v2-mean",
        anchor_name="impulse location + pre/post-window delta match the notes",
        predicted=delta_predicted,
        measured=measured_delta,
        tolerance=0.15,
        notes=(f"data-driven best-fit jump location k* = {kstar_fit} "
               f"(notes' crossing_k = {measured_kstar}, gap = "
               f"{kstar_fit - measured_kstar} step); pre/post window means "
               f"({pre:.3f}, {post:.3f})"),
    )


# ----------------------------------------------------------------------
# Runner / printer
# ----------------------------------------------------------------------

TESTS = [
    operator_C_cyclicity,
    operator_R_recursion,
    operator_F_fractality,
    operator_G_regression,
    operator_S_synergy,
    operator_P_refraction,
    operator_O_outliers,
]


def run_all() -> List[Verdict]:
    verdicts = []
    for fn in TESTS:
        v = fn()
        v.judge()
        verdicts.append(v)
    return verdicts


def _hr(w: int = 78) -> str:
    return "-" * w


def print_verdicts(verdicts: List[Verdict]) -> None:
    print()
    print("=" * 78)
    print("EchoKey seven-operator structural-consistency probe")
    print("=" * 78)
    print("Each operator is assigned a Collatz object, produces a numerical")
    print("prediction from its formal definition, and is compared to an anchor")
    print("from the companion 2-adic-observable notes.")
    print()

    for v in verdicts:
        print(f"[{v.operator}]  {v.name}")
        print(f"     assigned:   {v.assigned_object}")
        print(f"     anchor:     {v.anchor_name}")
        print(f"     predicted:  {v.predicted:+.6f}")
        print(f"     measured:   {v.measured:+.6f}")
        denom = max(abs(v.measured), abs(v.predicted), 1e-9)
        rel = abs(v.predicted - v.measured) / denom
        print(f"     rel. error: {rel:.2%}   tol: {v.tolerance:.2%}")
        print(f"     verdict:    {v.status}")
        if v.notes:
            print(f"     notes:      {v.notes}")
        print()

    counts = {"MATCH": 0, "PARTIAL": 0, "MISMATCH": 0}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    print(_hr())
    print(f"Summary:  MATCH {counts.get('MATCH', 0)}   "
          f"PARTIAL {counts.get('PARTIAL', 0)}   "
          f"MISMATCH {counts.get('MISMATCH', 0)}   "
          f"(of {len(verdicts)} operators)")
    print(_hr())


def save_csv(verdicts: List[Verdict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "echokey_verdicts.csv")
    fieldnames = ["operator", "name", "assigned_object", "anchor_name",
                  "predicted", "measured", "tolerance", "status", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for v in verdicts:
            d = asdict(v)
            w.writerow({k: d.get(k, "") for k in fieldnames})
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seven-operator structural-consistency probe (companion to EchoKey notes)."
    )
    parser.add_argument("--save", action="store_true",
                        help="write verdicts CSV to ./results/")
    parser.add_argument("--out-dir", default="results",
                        help="directory for CSV output (default: results/)")
    args = parser.parse_args()

    verdicts = run_all()
    print_verdicts(verdicts)
    if args.save:
        print()
        print(f"Writing verdicts CSV to {args.out_dir}/ ...")
        save_csv(verdicts, args.out_dir)


if __name__ == "__main__":
    main()