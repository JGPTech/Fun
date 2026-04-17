#!/usr/bin/env python3
"""
p1_collatz_arithmetic_probe.py
===========================

Computational probe accompanying the public research notes:

    "Notes on a 2-adic observable for the Collatz map,
     with a speculative arithmetic-topology dictionary"
    JGPTech, April 2026.

Reproduces every empirical table in the notes:

    Section 5.1  Supermartingale sweep
    Section 5.2  H on small trajectories      (n <= 100)
    Section 5.3  H near the attractor / tail H
    Section 5.4  Behaviour at the four-piece (|S_k| <= 4) crossing  (N = 200)

Dependencies: numpy.

Usage:
    python p1_collatz_arithmetic_probe.py            # print all four tables
    python p1_collatz_arithmetic_probe.py --save     # also write CSVs to ./results/
    python p1_collatz_arithmetic_probe.py --seed 0   # set numpy seed (no effect; deterministic)

Released under CC0.  github.com/JGPTech/Graviton_Field_Equations
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Core: shortened Collatz map, 2-adic valuation, trajectories
# ----------------------------------------------------------------------

def v2(n: int) -> int:
    """2-adic valuation: largest k with 2^k | n.  v2(0) := 0 by convention
    (only used as a defensive guard; never called on 0 in practice)."""
    if n == 0:
        return 0
    k = 0
    while (n & 1) == 0:
        n >>= 1
        k += 1
    return k


def T3(n: int) -> int:
    """Shortened Collatz map: n/2 if even, (3n+1)/2 if odd."""
    if n & 1:
        return (3 * n + 1) // 2
    return n // 2


def trajectory(n: int) -> List[int]:
    """Full Collatz trajectory from n, terminating when the iterate first
    reaches 1.  Returns the list of visited integers including the
    terminal 1.

    Convention (matches the notes' selected-values table):
        n=1  -> [1],            length 0
        n=2  -> [2, 1],         length 1
        n=8  -> [8, 4, 2, 1],   length 3
        n=27 -> ...,            length 70
    """
    seq = [n]
    cur = n
    while cur != 1:
        cur = T3(cur)
        seq.append(cur)
    return seq


def trajectory_length(seq: List[int]) -> int:
    """Number of steps in a trajectory (= number of nodes - 1)."""
    return len(seq) - 1


# ----------------------------------------------------------------------
# The observable H (notes Definition 1)
# ----------------------------------------------------------------------

def H_add(seq: List[int]) -> int:
    """H(gamma) = sum_{j=0}^{|gamma|-1} v2(gamma_j).

    Note the upper limit is |gamma|-1 (number of steps minus one), per the
    notes' indexing.  In code terms: sum v2 over all nodes except the
    terminal one.
    """
    return sum(v2(x) for x in seq[:-1])


def H_norm(seq: List[int]) -> float:
    L = trajectory_length(seq)
    return H_add(seq) / L if L > 0 else 0.0


# ----------------------------------------------------------------------
# Transfer operator: measure evolution of S_0 = {1,...,N}
# ----------------------------------------------------------------------

def evolve_support(N: int) -> List[List[int]]:
    """Return the sequence [S_0, S_1, S_2, ...] of supports under the
    pushforward (T_3)_* applied iteratively to the uniform measure on
    S_0 = {1,...,N}.  Stops when the support becomes a subset of the
    attractor {1, 2}.
    """
    S = list(range(1, N + 1))
    history = [S[:]]
    while True:
        S_next = sorted({T3(x) for x in S})
        history.append(S_next)
        if set(S_next).issubset({1, 2}):
            break
        S = S_next
    return history


def expectation_f(S: Iterable[int]) -> float:
    """E_{mu_k}[f] = (1/|S|) * sum_{n in S} 2^{-v2(n)}."""
    S = list(S)
    if not S:
        return 0.0
    return float(np.mean([2.0 ** (-v2(n)) for n in S]))


# ----------------------------------------------------------------------
# Vacuum measures M_1, ..., M_5 (self-contained reimplementation)
# ----------------------------------------------------------------------
#
# Definitions (matching the notes' threshold subsection):
#
#   B(n) = 1 if n odd else 0
#   f(n) = 2^{-v2(n)}
#
#   M_1 = | E[f | B=1] - E[f | B=0] |          (conditional gap)
#   M_2 = |T_3(S_k)|                            (image diversity)
#   M_3 = |{n in S : n odd}| / |{n in S : n even}|  (branch balance)
#   M_4 = E[f | B=0] / E[f | B=1]               (expectation ratio, even/odd)
#   M_5 = Var of conditional means              (between-branch variance)

def vacuum_measures(S: List[int]) -> dict:
    odds = [n for n in S if n & 1]
    evens = [n for n in S if not (n & 1)]

    Ef_odd = float(np.mean([2.0 ** (-v2(n)) for n in odds])) if odds else 0.0
    Ef_even = float(np.mean([2.0 ** (-v2(n)) for n in evens])) if evens else 0.0

    M1 = abs(Ef_odd - Ef_even)
    M2 = len({T3(n) for n in S})
    M3 = (len(odds) / len(evens)) if evens else float("inf")
    # Notes convention: M_4 = E[f|B=0] / E[f|B=1] (even-over-odd ratio).
    M4 = (Ef_even / Ef_odd) if Ef_odd > 0 else float("inf")
    # between-branch variance of the two conditional means
    if odds and evens:
        means = np.array([Ef_odd, Ef_even], dtype=float)
        M5 = float(np.var(means))
    else:
        M5 = 0.0

    v2_vals = np.array([v2(n) for n in S], dtype=float)
    return {
        "support_size": len(S),
        "M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5,
        "v2_mean": float(np.mean(v2_vals)),
        "v2_std": float(np.std(v2_vals)),
        "Ef_odd": Ef_odd, "Ef_even": Ef_even,
    }


# ----------------------------------------------------------------------
# Section 5.1: Supermartingale sweep
# ----------------------------------------------------------------------

def supermartingale_sweep(Ns: List[int]) -> List[dict]:
    rows = []
    for N in Ns:
        history = evolve_support(N)
        Es = np.array([expectation_f(S) for S in history])
        # step ratios E_{k+1}/E_k for k where E_k > 0
        ratios = []
        eps_sum = 0.0
        violations = 0
        for k in range(len(Es) - 1):
            if Es[k] > 0:
                r = Es[k + 1] / Es[k]
                ratios.append(r)
                # epsilon_k = max(0, E_{k+1} - E_k)
                diff = Es[k + 1] - Es[k]
                if diff > 0:
                    eps_sum += diff
                    violations += 1
        ratios = np.array(ratios, dtype=float)
        rows.append({
            "N": N,
            "steps": len(history) - 1,
            "eps_sum": float(eps_sum),
            "violations": int(violations),
            "max_ratio": float(np.max(ratios)) if ratios.size else float("nan"),
            "min_ratio": float(np.min(ratios)) if ratios.size else float("nan"),
            "mean_ratio": float(np.mean(ratios)) if ratios.size else float("nan"),
            "E0": float(Es[0]),
        })
    return rows


# ----------------------------------------------------------------------
# Section 5.2: H on small trajectories
# ----------------------------------------------------------------------

def H_table(n_max: int = 100) -> List[dict]:
    rows = []
    for n in range(1, n_max + 1):
        seq = trajectory(n)
        L = trajectory_length(seq)
        h_add = H_add(seq)
        rows.append({
            "n": n,
            "v2_n": v2(n),
            "length": L,
            "H_add": h_add,
            "H_norm": (h_add / L) if L > 0 else 0.0,
        })
    return rows


def H_summary(rows: List[dict]) -> dict:
    arr_n      = np.array([r["n"]       for r in rows], dtype=float)
    arr_v2n    = np.array([r["v2_n"]    for r in rows], dtype=float)
    arr_len    = np.array([r["length"]  for r in rows], dtype=float)
    arr_hadd   = np.array([r["H_add"]   for r in rows], dtype=float)
    arr_hnorm  = np.array([r["H_norm"]  for r in rows], dtype=float)

    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    parity_match = np.mean(((arr_hadd.astype(int) % 2) == (arr_v2n.astype(int) % 2)))

    return {
        "r_H_length": _pearson(arr_hadd, arr_len),
        "r_H_v2n":    _pearson(arr_hadd, arr_v2n),
        "mean_H_norm": float(np.mean(arr_hnorm)),
        "parity_match_frac": float(parity_match),
        "n_min": int(arr_n.min()),
        "n_max": int(arr_n.max()),
    }


# ----------------------------------------------------------------------
# Section 5.3: H near the attractor / tail H
# ----------------------------------------------------------------------

def attractor_signature() -> dict:
    """Compute the canonical signature of the attractor cycle {1, 2}.

    The attractor cycle is 1 -> 2 -> 1, viewed as a closed loop with three
    nodes [1, 2, 1].  Per the notes:
        v2 sequence  = [v2(1), v2(2), v2(1)] = [0, 1, 0]
        H_add        = sum of v2 along the cycle = 0 + 1 + 0 = 1
        H_norm       = H_add / 3 = 1/3   (normalized by cycle length in nodes)
        M4 signature = 2^v2(2) / 2^v2(1) = 2
    """
    cycle_nodes = [1, 2, 1]
    v2_seq = [v2(x) for x in cycle_nodes]
    h_add = sum(v2_seq)
    L = len(cycle_nodes)  # 3 nodes in the closed cycle
    return {
        "v2_sequence": v2_seq,
        "H_add": h_add,
        "H_norm": h_add / L,
        "M4_signature": (2 ** v2(2)) / (2 ** v2(1)),  # = 2
    }


def tail_H(n_min: int, n_max: int, K_values: List[int]) -> dict:
    """For each K, collect the H of the last K steps of each
    trajectory in [n_min, n_max], expressed in the same normalized
    sense as H_norm (sum of v2 over the last K nodes / K).

    Trajectories shorter than K are skipped for that K.
    """
    out = {}
    # Pre-compute trajectories once
    seqs = [trajectory(n) for n in range(n_min, n_max + 1)]
    for K in K_values:
        vals = []
        for seq in seqs:
            # the "tail" is the last K nodes that contribute to H_add,
            # i.e. excluding the terminal node, take the last K of seq[:-1].
            body = seq[:-1]
            if len(body) < K:
                continue
            tail = body[-K:]
            tail_h = sum(v2(x) for x in tail)
            vals.append(tail_h / K)
        vals = np.array(vals, dtype=float)
        out[K] = {
            "K": K,
            "n_samples": int(vals.size),
            "mean": float(np.mean(vals)) if vals.size else float("nan"),
            "std":  float(np.std(vals))  if vals.size else float("nan"),
            "min":  float(np.min(vals))  if vals.size else float("nan"),
            "max":  float(np.max(vals))  if vals.size else float("nan"),
        }
    return out


# ----------------------------------------------------------------------
# Section 5.4: M_2 = 4 crossing
# ----------------------------------------------------------------------

def four_piece_crossing(N: int = 200) -> dict:
    """Locate the first step at which the support size |S_k| drops to <= 4,
    and record the state of the ensemble at that step.

    The notes' threshold is named after Orzechowski's four-piece minimum
    for non-Archimedean paradoxical decompositions; the natural matching
    quantity is support size (number of pieces), not the image-diversity
    measure M_2 = |T_3(S_k)|, which at the crossing is typically 3.
    Both quantities are reported in the output for transparency.
    """
    history = evolve_support(N)
    total_steps = len(history) - 1

    # find first k with |S_k| <= 4 (support size, the "four pieces" quantity)
    crossing_k = None
    for k, S in enumerate(history):
        if len(S) <= 4:
            crossing_k = k
            break

    if crossing_k is None:
        return {"N": N, "total_steps": total_steps, "crossing_k": None}

    measures_at_crossing = vacuum_measures(history[crossing_k])

    # pre/post window v2-mean: take symmetric windows around the crossing,
    # default window size 5 steps, clipped to available range.
    W = 5
    pre_lo = max(0, crossing_k - W)
    pre_hi = crossing_k  # exclusive
    post_lo = crossing_k + 1
    post_hi = min(len(history), crossing_k + 1 + W)

    def _avg_v2_mean(lo, hi):
        if lo >= hi:
            return float("nan")
        return float(np.mean([
            np.mean([v2(x) for x in history[k]])
            for k in range(lo, hi)
        ]))

    pre_v2 = _avg_v2_mean(pre_lo, pre_hi)
    post_v2 = _avg_v2_mean(post_lo, post_hi)

    # initial and attractor v2-means for context
    init_v2 = float(np.mean([v2(x) for x in history[0]]))
    attractor_v2 = float(np.mean([v2(x) for x in history[-1]]))

    return {
        "N": N,
        "total_steps": total_steps,
        "crossing_k": crossing_k,
        "crossing_pct": 100.0 * crossing_k / total_steps,
        "support_at_crossing": history[crossing_k],
        "measures": measures_at_crossing,
        "pre_window_v2_mean": pre_v2,
        "post_window_v2_mean": post_v2,
        "delta_v2": post_v2 - pre_v2,
        "init_v2_mean": init_v2,
        "attractor_v2_mean": attractor_v2,
    }


# ----------------------------------------------------------------------
# Pretty-printing
# ----------------------------------------------------------------------

def _hr(width: int = 78) -> str:
    return "-" * width


def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_supermartingale(rows: List[dict]) -> None:
    print_section("5.1  Supermartingale sweep")
    header = f"{'N':>6}  {'steps':>6}  {'sum_eps':>9}  {'viol':>5}  " \
             f"{'max_r':>7}  {'min_r':>7}  {'mean_r':>7}"
    print(header)
    print(_hr(len(header)))
    for r in rows:
        print(f"{r['N']:>6}  {r['steps']:>6}  {r['eps_sum']:>9.3f}  "
              f"{r['violations']:>5}  {r['max_ratio']:>7.3f}  "
              f"{r['min_ratio']:>7.3f}  {r['mean_ratio']:>7.3f}")


def print_H(rows: List[dict], summary: dict) -> None:
    print_section("5.2  H on small trajectories")
    print(f"  Sample range: n = {summary['n_min']} ... {summary['n_max']}")
    print(f"  Pearson r(H, length) = {summary['r_H_length']:.3f}")
    print(f"  Pearson r(H, v2(n))  = {summary['r_H_v2n']:.3f}")
    print(f"  mean(H_norm)         = {summary['mean_H_norm']:.3f}")
    print(f"  parity match frac    = {summary['parity_match_frac']:.3f}")
    print()
    print("  Selected values:")
    sel_n = [1, 2, 8, 16, 27]
    by_n = {r["n"]: r for r in rows}
    sub = f"  {'n':>3}  {'v2(n)':>5}  {'len':>5}  {'H':>6}  {'H_norm':>8}"
    print(sub)
    print("  " + _hr(len(sub) - 2))
    for n in sel_n:
        r = by_n[n]
        print(f"  {r['n']:>3}  {r['v2_n']:>5}  {r['length']:>5}  "
              f"{r['H_add']:>6}  {r['H_norm']:>8.3f}")


def print_vacuum(attractor: dict, tail: dict) -> None:
    print_section("5.3  H near the attractor / tail H")
    print(f"  Attractor cycle {{1,2}} signature:")
    print(f"    v2 sequence:  {attractor['v2_sequence']}")
    print(f"    H            =  {attractor['H_add']}")
    print(f"    H_norm       =  {attractor['H_norm']:.4f}  (= 1/3)")
    print(f"    M4 signature =  {attractor['M4_signature']:.1f}")
    print()
    print("  Tail H (last K steps, n = 3 ... 500):")
    sub = f"    {'K':>3}  {'samples':>7}  {'mean':>7}  {'std':>7}  {'min':>6}  {'max':>6}"
    print(sub)
    print("    " + _hr(len(sub) - 4))
    for K in sorted(tail):
        s = tail[K]
        print(f"    {s['K']:>3}  {s['n_samples']:>7}  {s['mean']:>7.3f}  "
              f"{s['std']:>7.3f}  {s['min']:>6.3f}  {s['max']:>6.3f}")


def print_four_piece_crossing(orz: dict) -> None:
    print_section("5.4  Behaviour at the four-piece (|S_k| <= 4) crossing  (N = 200)")
    print(f"  N = {orz['N']}")
    print(f"  Total collapse steps:                  {orz['total_steps']}")
    print(f"  Crossing step (|S_k| <= 4):            {orz['crossing_k']}")
    print(f"  Crossing fraction of collapse:         {orz['crossing_pct']:.1f}%")
    print(f"  Support at crossing:                   {orz['support_at_crossing']}")
    print()
    m = orz["measures"]
    print(f"  Quantity                               Value at crossing")
    print(f"  " + _hr(58))
    print(f"  Support size                           {m['support_size']}")
    print(f"  v2_mean                                {m['v2_mean']:.3f}")
    print(f"  v2_std                                 {m['v2_std']:.3f}")
    print(f"  M1 (conditional gap)                   {m['M1']:.3f}")
    print(f"  M2 (image diversity = |T_3(S)|)        {m['M2']}")
    print(f"  M3 (branch balance)                    {m['M3']:.3f}")
    print(f"  M4 (E[f|even]/E[f|odd])                {m['M4']:.3f}")
    print(f"  M5 (between-branch variance)           {m['M5']:.3f}")
    print()
    print(f"  v2-mean trajectory through crossing:")
    print(f"    initial (S_0)                        {orz['init_v2_mean']:.3f}")
    print(f"    pre-crossing window mean             {orz['pre_window_v2_mean']:.3f}")
    print(f"    post-crossing window mean            {orz['post_window_v2_mean']:.3f}")
    print(f"    delta                                {orz['delta_v2']:+.3f}")
    print(f"    attractor (terminal)                 {orz['attractor_v2_mean']:.3f}")


# ----------------------------------------------------------------------
# CSV output
# ----------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csvs(out_dir: str,
              sm_rows: List[dict],
              H_rows: List[dict],
              tail: dict,
              orz: dict) -> None:
    _ensure_dir(out_dir)

    def _write(name: str, fieldnames: List[str], rows: List[dict]) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"  wrote {path}")

    print()
    print(f"Writing CSVs to {out_dir}/ ...")

    _write("supermartingale_sweep.csv",
           ["N", "steps", "eps_sum", "violations",
            "max_ratio", "min_ratio", "mean_ratio", "E0"],
           sm_rows)

    _write("H_per_n.csv",
           ["n", "v2_n", "length", "H_add", "H_norm"],
           H_rows)

    tail_rows = [tail[K] for K in sorted(tail)]
    _write("tail_H.csv",
           ["K", "n_samples", "mean", "std", "min", "max"],
           tail_rows)

    flat = {
        "N": orz["N"],
        "total_steps": orz["total_steps"],
        "crossing_k": orz["crossing_k"],
        "crossing_pct": orz["crossing_pct"],
        "support_at_crossing": "|".join(map(str, orz["support_at_crossing"])),
        "support_size": orz["measures"]["support_size"],
        "v2_mean": orz["measures"]["v2_mean"],
        "v2_std": orz["measures"]["v2_std"],
        "M1": orz["measures"]["M1"],
        "M2": orz["measures"]["M2"],
        "M3": orz["measures"]["M3"],
        "M4": orz["measures"]["M4"],
        "M5": orz["measures"]["M5"],
        "init_v2_mean": orz["init_v2_mean"],
        "pre_window_v2_mean": orz["pre_window_v2_mean"],
        "post_window_v2_mean": orz["post_window_v2_mean"],
        "delta_v2": orz["delta_v2"],
        "attractor_v2_mean": orz["attractor_v2_mean"],
    }
    _write("four_piece_crossing.csv", list(flat.keys()), [flat])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computational probe for the Collatz arithmetic-topology notes."
    )
    parser.add_argument("--save", action="store_true",
                        help="write CSVs to ./results/")
    parser.add_argument("--out-dir", default="results",
                        help="directory for CSV output (default: results/)")
    parser.add_argument("--seed", type=int, default=0,
                        help="numpy seed (no stochastic component; included for reproducibility)")
    parser.add_argument("--N-sweep", type=int, nargs="+",
                        default=[50, 100, 200, 500, 1000, 2000],
                        help="N values for supermartingale sweep")
    parser.add_argument("--n-H-max", type=int, default=100,
                        help="upper bound n for the per-trajectory H table")
    parser.add_argument("--n-tail-max", type=int, default=500,
                        help="upper bound n for tail H stats")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print()
    print("Collatz arithmetic-topology probe")
    print("=================================")
    print("Reproduces the empirical tables of the companion notes.")

    # 5.1
    sm_rows = supermartingale_sweep(args.N_sweep)
    print_supermartingale(sm_rows)

    # 5.2
    H_rows = H_table(args.n_H_max)
    summary = H_summary(H_rows)
    print_H(H_rows, summary)

    # 5.3
    attractor = attractor_signature()
    tail = tail_H(3, args.n_tail_max, [5, 10, 20])
    print_vacuum(attractor, tail)

    # 5.4
    orz = four_piece_crossing(N=200)
    print_four_piece_crossing(orz)

    if args.save:
        save_csvs(args.out_dir, sm_rows, H_rows, tail, orz)

    print()


if __name__ == "__main__":
    main()