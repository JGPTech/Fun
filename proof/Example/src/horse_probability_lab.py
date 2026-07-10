#!/usr/bin/env python3
"""
Horse Probability Lab
=====================

A 3-in-1 Python verification and analysis script for the Wynncraft-style
horse-combination probability problem.

This script contains:

1. Exact recursive solver using fractions.Fraction
2. Bottom-up dynamic-programming verifier
3. Monte Carlo sanity checker using the exact optimal policy
4. Scaling / threshold analysis over a finite computed range
5. CSV + JSON exports

Mathematical model
------------------

State:
    s = (t1, t2, t3, t4)

Goal:
    eventually reach any state with t4 >= 1

Legal actions:
    combine two T1, T2, or T3 horses when available

Transition probabilities:
    upgrade   = 1/5
    same      = 1/2
    downgrade = 3/10

Tier 1 has no lower tier, so same+downgrade collapse into the T1 outcome:
    combine T1:
        1/5 -> one T2
        4/5 -> one T1

Why recursion terminates
------------------------

Every legal combine consumes two horses and returns one horse, so total horse
count decreases by exactly one after every nonterminal transition.

This is the rank:
    rho(s) = t1 + t2 + t3 + t4

That makes the state-transition graph acyclic and justifies strong induction /
bottom-up dynamic programming.

Example
-------

    python horse_probability_lab.py --max-x 80 --verify-max-horses 28 --trials 5000 --out out

For a faster smoke test:

    python horse_probability_lab.py --max-x 30 --verify-max-horses 16 --trials 1000 --out out
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Exact probabilities
# ---------------------------------------------------------------------------

P_UPGRADE = Fraction(1, 5)
P_SAME = Fraction(1, 2)
P_DOWNGRADE = Fraction(3, 10)
P_T1_STAY = P_SAME + P_DOWNGRADE  # 4/5


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class HorseState:
    """A finite horse state: counts in tiers 1, 2, 3, and 4."""

    t1: int
    t2: int
    t3: int
    t4: int

    def total(self) -> int:
        """Total horse count rank rho(s)."""
        return self.t1 + self.t2 + self.t3 + self.t4

    def tier_weight(self) -> int:
        """Tier-weight potential omega(s) = t1 + 2t2 + 4t3 + 8t4."""
        return self.t1 + 2 * self.t2 + 4 * self.t3 + 8 * self.t4

    def is_success(self) -> bool:
        return self.t4 >= 1

    def legal_actions(self) -> Tuple[str, ...]:
        actions: List[str] = []
        if self.t1 >= 2:
            actions.append("combine_T1")
        if self.t2 >= 2:
            actions.append("combine_T2")
        if self.t3 >= 2:
            actions.append("combine_T3")
        return tuple(actions)

    def is_failure(self) -> bool:
        return (not self.is_success()) and len(self.legal_actions()) == 0

    def is_active(self) -> bool:
        return (not self.is_success()) and len(self.legal_actions()) > 0


Transition = Tuple[Fraction, HorseState, str]


def transitions(state: HorseState, action: str) -> Tuple[Transition, ...]:
    """
    Return exact weighted successor states for a legal action.

    Each returned tuple is:
        (probability, next_state, outcome_label)
    """
    if action == "combine_T1":
        if state.t1 < 2:
            raise ValueError(f"Illegal action {action} at {state}")

        upgrade = HorseState(state.t1 - 2, state.t2 + 1, state.t3, state.t4)
        stay = HorseState(state.t1 - 1, state.t2, state.t3, state.t4)

        return (
            (P_UPGRADE, upgrade, "upgrade_to_T2"),
            (P_T1_STAY, stay, "same_or_downgrade_to_T1"),
        )

    if action == "combine_T2":
        if state.t2 < 2:
            raise ValueError(f"Illegal action {action} at {state}")

        upgrade = HorseState(state.t1, state.t2 - 2, state.t3 + 1, state.t4)
        same = HorseState(state.t1, state.t2 - 1, state.t3, state.t4)
        downgrade = HorseState(state.t1 + 1, state.t2 - 2, state.t3, state.t4)

        return (
            (P_UPGRADE, upgrade, "upgrade_to_T3"),
            (P_SAME, same, "same_to_T2"),
            (P_DOWNGRADE, downgrade, "downgrade_to_T1"),
        )

    if action == "combine_T3":
        if state.t3 < 2:
            raise ValueError(f"Illegal action {action} at {state}")

        upgrade = HorseState(state.t1, state.t2, state.t3 - 2, state.t4 + 1)
        same = HorseState(state.t1, state.t2, state.t3 - 1, state.t4)
        downgrade = HorseState(state.t1, state.t2 + 1, state.t3 - 2, state.t4)

        return (
            (P_UPGRADE, upgrade, "upgrade_to_T4"),
            (P_SAME, same, "same_to_T3"),
            (P_DOWNGRADE, downgrade, "downgrade_to_T2"),
        )

    raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Exact recursive solver
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def exact_value(state: HorseState) -> Fraction:
    """
    Exact optimal probability of eventually reaching t4 >= 1 from state.

    This is the mathematical Bellman recursion:
        V(s) = 1, success
        V(s) = 0, terminal failure
        V(s) = max_a sum_{s'} K(s'|s,a)V(s'), active state
    """
    if state.is_success():
        return Fraction(1, 1)

    actions = state.legal_actions()
    if not actions:
        return Fraction(0, 1)

    best = Fraction(0, 1)
    for action in actions:
        q = sum(prob * exact_value(next_state)
                for prob, next_state, _label in transitions(state, action))
        if q > best:
            best = q

    return best


def probability_from_t1(x: int) -> Fraction:
    """Exact optimal probability from x tier-1 horses."""
    if x < 0:
        raise ValueError("x must be nonnegative")
    return exact_value(HorseState(x, 0, 0, 0))


@lru_cache(maxsize=None)
def best_action(state: HorseState) -> Optional[str]:
    """
    Deterministic optimal action selector.

    Returns None for terminal states.
    Ties are resolved by the order:
        combine_T1, combine_T2, combine_T3
    """
    if not state.is_active():
        return None

    best: Optional[str] = None
    best_q = Fraction(-1, 1)

    for action in state.legal_actions():
        q = sum(prob * exact_value(next_state)
                for prob, next_state, _label in transitions(state, action))
        if q > best_q:
            best_q = q
            best = action

    return best


# ---------------------------------------------------------------------------
# Bottom-up verifier
# ---------------------------------------------------------------------------

def states_of_total(total: int) -> Iterable[HorseState]:
    """Generate all states with t1+t2+t3+t4 == total."""
    for t1 in range(total + 1):
        for t2 in range(total - t1 + 1):
            for t3 in range(total - t1 - t2 + 1):
                t4 = total - t1 - t2 - t3
                yield HorseState(t1, t2, t3, t4)


def bottom_up_values(max_total: int) -> Dict[HorseState, Fraction]:
    """
    Compute exact values for all states with total horse count <= max_total.

    Since every transition decreases total count by one, values of rank n
    only depend on values of rank n-1.
    """
    values: Dict[HorseState, Fraction] = {}

    for n in range(max_total + 1):
        for state in states_of_total(n):
            if state.is_success():
                values[state] = Fraction(1, 1)
                continue

            actions = state.legal_actions()
            if not actions:
                values[state] = Fraction(0, 1)
                continue

            best = Fraction(0, 1)
            for action in actions:
                q = Fraction(0, 1)
                for prob, next_state, _label in transitions(state, action):
                    try:
                        q += prob * values[next_state]
                    except KeyError as exc:
                        raise RuntimeError(
                            f"Missing bottom-up successor {next_state} "
                            f"for state {state}; rank order is broken."
                        ) from exc
                best = max(best, q)

            values[state] = best

    return values


def verify_bottom_up(max_total: int, max_mismatches: int = 10) -> dict:
    """
    Compare recursive exact values against bottom-up exact values for all states
    up to max_total.
    """
    start = time.time()
    values = bottom_up_values(max_total)
    mismatches = []

    for state, dp_val in values.items():
        rec_val = exact_value(state)
        if rec_val != dp_val:
            mismatches.append({
                "state": state_to_list(state),
                "recursive": fraction_to_str(rec_val),
                "bottom_up": fraction_to_str(dp_val),
            })
            if len(mismatches) >= max_mismatches:
                break

    elapsed = time.time() - start

    return {
        "max_total": max_total,
        "states_checked": len(values),
        "passed": len(mismatches) == 0,
        "mismatch_count_capped": len(mismatches),
        "mismatches": mismatches,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Monte Carlo sanity check
# ---------------------------------------------------------------------------

def sample_transition(
    state: HorseState,
    action: str,
    rng: random.Random,
) -> HorseState:
    """Sample one successor state according to exact transition probabilities."""
    r = rng.random()
    cumulative = 0.0

    for prob, next_state, _label in transitions(state, action):
        cumulative += float(prob)
        if r <= cumulative:
            return next_state

    # Floating-point fallback for r extremely close to 1.
    return transitions(state, action)[-1][1]


def simulate_one(x: int, rng: random.Random) -> bool:
    """Simulate one run from x tier-1 horses using the exact optimal policy."""
    state = HorseState(x, 0, 0, 0)

    while True:
        if state.is_success():
            return True
        if state.is_failure():
            return False

        action = best_action(state)
        if action is None:
            return False

        state = sample_transition(state, action, rng)


def monte_carlo(xs: Sequence[int], trials: int, seed: int) -> List[dict]:
    """Run Monte Carlo sanity checks for selected initial x values."""
    rng = random.Random(seed)
    rows = []

    for x in xs:
        wins = 0
        for _ in range(trials):
            if simulate_one(x, rng):
                wins += 1

        exact = probability_from_t1(x)
        empirical = wins / trials if trials > 0 else float("nan")

        rows.append({
            "x": x,
            "trials": trials,
            "wins": wins,
            "empirical": empirical,
            "exact": float(exact),
            "absolute_error": abs(empirical - float(exact)),
        })

    return rows


# ---------------------------------------------------------------------------
# Scaling and thresholds
# ---------------------------------------------------------------------------

def default_x_values(max_x: int) -> List[int]:
    """
    Strategic sample schedule inspired by the original Julia script:
        dense early, sparser later
    clipped to max_x.
    """
    values: List[int] = []

    def add_range(start: int, stop: int, step: int) -> None:
        for x in range(start, min(stop, max_x) + 1, step):
            if x <= max_x:
                values.append(x)

    if max_x < 8:
        return []

    add_range(8, 30, 2)
    add_range(35, 100, 5)
    add_range(110, 200, 10)
    add_range(225, 500, 25)
    add_range(550, max_x, 50)

    # Ensure max_x itself is included if >= 8.
    values.append(max_x)

    return sorted(set(x for x in values if 8 <= x <= max_x))


def compute_probability_rows(xs: Sequence[int]) -> List[dict]:
    """Compute exact probabilities and useful float/log values for xs."""
    rows = []

    for x in xs:
        prob = probability_from_t1(x)
        prob_float = float(prob)
        row = {
            "x": x,
            "probability_fraction": fraction_to_str(prob),
            "probability": prob_float,
            "probability_percent": 100.0 * prob_float,
            "log_x": math.log(x) if x > 0 else None,
            "log_probability": math.log(prob_float) if prob_float > 0 else None,
        }
        rows.append(row)

    return rows


def finite_window_power_fit(rows: Sequence[dict], min_x: int = 20) -> dict:
    """
    Manual log-log linear regression:
        log P ~= intercept + alpha log X

    This is an empirical finite-window fit, not a global theorem.
    """
    pts = [
        (row["log_x"], row["log_probability"])
        for row in rows
        if row["x"] >= min_x
        and row["probability"] > 0.0
        and row["log_x"] is not None
        and row["log_probability"] is not None
    ]

    if len(pts) < 3:
        return {
            "fit_available": False,
            "reason": "Need at least 3 positive-probability points.",
            "min_x": min_x,
            "n_points": len(pts),
        }

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(xs)

    xbar = sum(xs) / n
    ybar = sum(ys) / n

    den = sum((x - xbar) ** 2 for x in xs)
    if den == 0:
        return {
            "fit_available": False,
            "reason": "Degenerate x values.",
            "min_x": min_x,
            "n_points": n,
        }

    alpha = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / den
    intercept = ybar - alpha * xbar

    yhat = [intercept + alpha * x for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, yhat))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "fit_available": True,
        "model": "log(P) ~= intercept + alpha*log(X)",
        "interpretation": "Empirical finite-window fit only; probabilities are bounded by 1.",
        "min_x": min_x,
        "n_points": n,
        "alpha": alpha,
        "intercept": intercept,
        "r_squared": r2,
    }


def threshold_report(rows: Sequence[dict], targets: Sequence[float]) -> List[dict]:
    """
    Find the first sampled x whose probability reaches each target.
    This does not claim a global threshold unless max_x is large enough.
    """
    sorted_rows = sorted(rows, key=lambda r: r["x"])
    report = []

    for target in targets:
        hit = next((row for row in sorted_rows if row["probability"] >= target), None)
        if hit is None:
            report.append({
                "target": target,
                "target_percent": 100.0 * target,
                "found_in_computed_range": False,
                "x": None,
                "probability": None,
            })
        else:
            report.append({
                "target": target,
                "target_percent": 100.0 * target,
                "found_in_computed_range": True,
                "x": hit["x"],
                "probability": hit["probability"],
            })

    return report


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def write_probability_csv(rows: Sequence[dict], path: Path) -> None:
    """Write probability rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "x",
        "probability_fraction",
        "probability",
        "probability_percent",
        "log_x",
        "log_probability",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_json(obj: dict, path: Path) -> None:
    """Write JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fraction_to_str(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def state_to_list(state: HorseState) -> List[int]:
    return [state.t1, state.t2, state.t3, state.t4]


def parse_int_list(text: str) -> List[int]:
    if not text.strip():
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> List[float]:
    if not text.strip():
        return []
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def print_probability_table(rows: Sequence[dict], max_rows: int = 20) -> None:
    """Print a compact table to the console."""
    print()
    print("Probability samples")
    print("-------------------")

    shown = list(rows[:max_rows])
    for row in shown:
        print(
            f"X={row['x']:>4}  "
            f"P={row['probability']:.8f}  "
            f"({row['probability_percent']:.4f}%)"
        )

    if len(rows) > max_rows:
        print(f"... {len(rows) - max_rows} more rows written to CSV")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact verifier + Monte Carlo + scaling lab for horse-combination probabilities."
    )

    parser.add_argument(
        "--max-x",
        type=int,
        default=80,
        help="Maximum initial T1 horse count for probability table. Default: 80.",
    )
    parser.add_argument(
        "--verify-max-horses",
        type=int,
        default=28,
        help="Max total horse count for bottom-up verifier. Default: 28.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out",
        help="Output directory. Default: out.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5000,
        help="Monte Carlo trials per selected x. Default: 5000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Monte Carlo seed. Default: 1234.",
    )
    parser.add_argument(
        "--monte-carlo-xs",
        type=str,
        default="8,12,16,24,32,48,64",
        help="Comma-separated x values for Monte Carlo sanity checks.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.01,0.10,0.25,0.50,0.75,0.90,0.95,0.99",
        help="Comma-separated target probabilities for sampled threshold report.",
    )
    parser.add_argument(
        "--scaling-min-x",
        type=int,
        default=20,
        help="Minimum x used for log-log finite-window scaling fit. Default: 20.",
    )
    parser.add_argument(
        "--skip-dp",
        action="store_true",
        help="Skip bottom-up DP verification.",
    )
    parser.add_argument(
        "--skip-monte-carlo",
        action="store_true",
        help="Skip Monte Carlo sanity checks.",
    )
    parser.add_argument(
        "--table-rows",
        type=int,
        default=20,
        help="Number of probability rows to print. Default: 20.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.max_x < 0:
        raise SystemExit("--max-x must be nonnegative")
    if args.verify_max_horses < 0:
        raise SystemExit("--verify-max-horses must be nonnegative")
    if args.trials < 0:
        raise SystemExit("--trials must be nonnegative")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()

    print("Horse Probability Lab")
    print("=====================")
    print(f"max_x              : {args.max_x}")
    print(f"verify_max_horses  : {args.verify_max_horses}")
    print(f"out_dir            : {out_dir}")
    print()

    # Exact base-case checks.
    f8 = probability_from_t1(8)
    expected_f8 = P_UPGRADE ** 7

    base_case = {
        "f_8": fraction_to_str(f8),
        "expected": fraction_to_str(expected_f8),
        "passed": f8 == expected_f8,
        "f_8_float": float(f8),
    }

    print("Base case")
    print("---------")
    print(f"f(8) exact     : {base_case['f_8']}")
    print(f"expected       : {base_case['expected']}")
    print(f"passed         : {base_case['passed']}")
    print()

    # Compute probability table.
    xs = default_x_values(args.max_x)
    rows = compute_probability_rows(xs)
    print_probability_table(rows, max_rows=args.table_rows)

    # Bottom-up verification.
    if args.skip_dp:
        dp_report = {
            "skipped": True,
            "reason": "--skip-dp was provided.",
        }
    else:
        print()
        print("Bottom-up verifier")
        print("------------------")
        dp_report = verify_bottom_up(args.verify_max_horses)
        print(f"states checked : {dp_report['states_checked']}")
        print(f"passed         : {dp_report['passed']}")
        print(f"elapsed        : {dp_report['elapsed_seconds']:.3f}s")

    # Scaling.
    fit = finite_window_power_fit(rows, min_x=args.scaling_min_x)

    print()
    print("Finite-window scaling fit")
    print("-------------------------")
    if fit.get("fit_available"):
        print(f"alpha          : {fit['alpha']:.6f}")
        print(f"R^2            : {fit['r_squared']:.6f}")
        print(f"n_points       : {fit['n_points']}")
        print("note           : empirical finite-window fit only")
    else:
        print(f"fit unavailable: {fit.get('reason')}")

    # Thresholds over sampled values.
    targets = parse_float_list(args.thresholds)
    thresholds = threshold_report(rows, targets)

    print()
    print("Sampled threshold report")
    print("------------------------")
    for item in thresholds:
        label = f"{item['target_percent']:.0f}%"
        if item["found_in_computed_range"]:
            print(f"{label:>4} target -> first sampled X={item['x']} "
                  f"(P={item['probability']:.6f})")
        else:
            print(f"{label:>4} target -> not reached in sampled range")

    # Monte Carlo.
    mc_xs = [x for x in parse_int_list(args.monte_carlo_xs) if 0 <= x <= args.max_x]
    if args.skip_monte_carlo:
        mc_report = {
            "skipped": True,
            "reason": "--skip-monte-carlo was provided.",
        }
    elif not mc_xs:
        mc_report = {
            "skipped": True,
            "reason": "No Monte Carlo x values were inside [0, max_x].",
        }
    else:
        print()
        print("Monte Carlo sanity check")
        print("------------------------")
        mc_rows = monte_carlo(mc_xs, args.trials, args.seed)
        for row in mc_rows:
            print(
                f"X={row['x']:>4}  "
                f"empirical={row['empirical']:.6f}  "
                f"exact={row['exact']:.6f}  "
                f"abs_err={row['absolute_error']:.6f}"
            )
        mc_report = {
            "skipped": False,
            "seed": args.seed,
            "trials_per_x": args.trials,
            "rows": mc_rows,
        }

    elapsed = time.time() - started

    report = {
        "config": {
            "max_x": args.max_x,
            "verify_max_horses": args.verify_max_horses,
            "trials": args.trials,
            "seed": args.seed,
            "scaling_min_x": args.scaling_min_x,
            "thresholds": targets,
        },
        "constants": {
            "P_UPGRADE": fraction_to_str(P_UPGRADE),
            "P_SAME": fraction_to_str(P_SAME),
            "P_DOWNGRADE": fraction_to_str(P_DOWNGRADE),
            "P_T1_STAY": fraction_to_str(P_T1_STAY),
        },
        "base_case": base_case,
        "probability_samples": rows,
        "bottom_up_verification": dp_report,
        "finite_window_scaling_fit": fit,
        "sampled_thresholds": thresholds,
        "monte_carlo": mc_report,
        "cache_info": {
            "exact_value": str(exact_value.cache_info()),
            "best_action": str(best_action.cache_info()),
        },
        "elapsed_seconds": elapsed,
    }

    probability_csv = out_dir / "probabilities.csv"
    report_json = out_dir / "verification_report.json"

    write_probability_csv(rows, probability_csv)
    write_json(report, report_json)

    print()
    print("Exports")
    print("-------")
    print(f"wrote {probability_csv}")
    print(f"wrote {report_json}")
    print(f"total elapsed: {elapsed:.3f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
