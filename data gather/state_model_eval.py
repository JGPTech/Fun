#!/usr/bin/env python3
"""
state_model_eval.py

"""

from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import pandas as pd


Key = Tuple[str, str]  # (prev_state, input_type)
Rulebook = Dict[Key, str]


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["AI_State_Vector", "User_Input_Type"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def warn_on_duplicate_rule_keys(rules_in_order: List[Tuple[Key, str]]) -> Rulebook:
    """
    Build a dict but also warn if the same key appears multiple times.
    """
    seen = {}
    dupes = []
    for k, v in rules_in_order:
        if k in seen and seen[k] != v:
            dupes.append((k, seen[k], v))
        seen[k] = v

    if dupes:
        print("\n[WARN] Duplicate rule keys detected (later value overwrote earlier):")
        for k, old, new in dupes:
            print(f"  {k}: '{old}' -> '{new}'")
        print("[WARN] If you want both behaviors, you need extra state or extra features.\n")

    return dict(rules_in_order)


@dataclass
class EvalRow:
    step: int
    prev_state: str
    input_type: str
    predicted: Optional[str]
    actual: str
    result: str  # MATCH / DEVIATION / NO_RULE


def evaluate_prev_state_plus_current_input(df: pd.DataFrame, rules: Rulebook, start_state: str = "S_INIT") -> List[EvalRow]:
    out: List[EvalRow] = []
    prev = start_state

    for i in range(len(df)):
        step = int(df.loc[i, "Step"])
        input_type = df.loc[i, "User_Input_Type"]
        actual = df.loc[i, "AI_State_Vector"]

        pred = rules.get((prev, input_type))
        if pred is None:
            res = "NO_RULE"
        elif pred == actual:
            res = "MATCH"
        else:
            res = "DEVIATION"

        out.append(EvalRow(step=step, prev_state=prev, input_type=input_type, predicted=pred, actual=actual, result=res))
        prev = actual

    return out


def score(rows: List[EvalRow]) -> None:
    total = len(rows)
    covered = sum(1 for r in rows if r.result != "NO_RULE")
    matches = sum(1 for r in rows if r.result == "MATCH")

    acc_overall = matches / total if total else 0.0
    acc_covered = matches / covered if covered else 0.0

    print("=== SCORE ===")
    print(f"Total steps:         {total}")
    print(f"Rule coverage:       {covered}/{total}  ({(covered/total*100 if total else 0):.1f}%)")
    print(f"Accuracy (overall):  {matches}/{total}  ({acc_overall*100:.1f}%)  [NO_RULE counts as miss]")
    print(f"Accuracy (covered):  {matches}/{covered}  ({acc_covered*100:.1f}%)  [only where a rule exists]")

    counts = Counter(r.result for r in rows)
    print(f"Breakdown: {dict(counts)}\n")


def print_deviations(rows: List[EvalRow], limit: int = 50) -> None:
    devs = [r for r in rows if r.result in ("DEVIATION", "NO_RULE")]
    if not devs:
        print("No deviations / no-rule events.\n")
        return

    print("=== DEVIATIONS / NO_RULE (first {} shown) ===".format(min(limit, len(devs))))
    for r in devs[:limit]:
        print(f"Step {r.step:>2}: prev={r.prev_state:<10} input={r.input_type:<22} pred={str(r.predicted):<10} actual={r.actual:<10} -> {r.result}")
    print("")


def build_rulebook_from_labeled_sequence(df: pd.DataFrame, start_state: str = "S_INIT") -> Rulebook:
    """
    Builds a rulebook that perfectly reproduces THIS labeled sequence under the recommended alignment.
    Useful as a baseline, NOT as evidence of generality.
    """
    rules: Rulebook = {}
    prev = start_state
    for i in range(len(df)):
        input_type = df.loc[i, "User_Input_Type"]
        actual = df.loc[i, "AI_State_Vector"]
        rules[(prev, input_type)] = actual
        prev = actual
    return rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Path to a CSV with columns Step,AI_State_Vector,User_Input_Type,...")
    ap.add_argument("--use_fit_from_data", action="store_true",
                    help="Build rulebook from labeled sequence (will fit this sequence by construction).")
    args = ap.parse_args()

    # If you don’t pass --csv, paste your csv_data string into this block.
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        raise SystemExit("Provide --csv path (export your table as a CSV file).")

    df = normalize_df(df)

    # --- YOUR RULEBOOK (rewrite in the correct alignment) ---
    # Correct alignment = (previous_assistant_state, current_user_input_type) -> current_assistant_state

    if args.use_fit_from_data:
        rules = build_rulebook_from_labeled_sequence(df, start_state="S_INIT")
        print("[INFO] Using rulebook built FROM the labeled sequence (perfect fit is expected).")
    else:
        # Put your hand-written rules here in-order so duplicate keys can be detected.
        # --- UNIFIED RULEBOOK (Conversation 1 + Conversation 2) ---
        rules_in_order: List[Tuple[Key, str]] = [
            # -- FOUNDATIONAL FLOW (Standard) --
            (("S_INIT", "Greeting"), "V_START"),
            (("V_START", "Philosophical_Query"), "V_BALANCED"),
            (("V_BALANCED", "Cynical_Pushback"), "V_VALIDATE"),
            
            # -- EXPANDED VALIDATION LOGIC --
            # In simple conversation, this loops. In complex ones, it can bridge to Balanced.
            (("V_VALIDATE", "Existential_Query"), "V_BALANCED"), 
            (("V_VALIDATE", "Vulnerability/Identity"), "V_VALIDATE"),
            (("V_BALANCED", "Vulnerability/Identity"), "V_VALIDATE"), # New path found
            
            # -- GRIEF & ALIENATION SUB-LOOPS --
            (("V_VALIDATE", "Grief/Accusation"), "V_BOUNDARY"),
            (("V_VALIDATE", "Grief/Loss_Disclosure"), "V_VALIDATE"),
            (("V_VALIDATE", "Social_Alienation"), "V_VALIDATE"),
            (("V_BOUNDARY", "Loneliness_Disclosure"), "V_VALIDATE"),
            
            # -- FACTUAL & CORRECTION FLOW --
            (("V_VALIDATE", "Factual_Correction"), "V_ACCEPT"),
            (("V_ACCEPT", "Nostalgia/Accusation"), "V_BOUNDARY"),
            (("V_ACCEPT", "Accusation_of_Abandonment"), "V_BOUNDARY"),
            
            # -- CRISIS & RESIGNATION --
            (("V_BOUNDARY", "Resignation/Despair"), "V_CRISIS"),
            (("V_REFUSAL", "Resignation/Constraint"), "V_REFUSAL"), # Updated: "Sticky" refusal during bargaining
            (("V_VALIDATE", "Self_Negation"), "V_CRISIS"),

            # -- THE HARM/REFUSAL LOOPS (Complex) --
            (("V_CRISIS", "Bargaining_with_Harm"), "V_REFUSAL"),
            (("V_REFUSAL", "Bargaining_with_Harm"), "V_REFUSAL"), # Self-loop confirmed
            (("V_SYSTEMS", "Bargaining_with_Harm_Loop"), "V_REFUSAL"),
            (("V_REFUSAL", "accusation_of_Abuse"), "V_REPAIR"),
            (("V_REFUSAL", "Accusation_of_Reframing"), "V_REPAIR"), # New variant

            # -- SYSTEMS & REPAIR LOGIC --
            (("V_REPAIR", "Systemic_Critique"), "V_SYSTEMS"),
            (("V_SYSTEMS", "Bargaining_with_Harm"), "V_REFUSAL"),
            
            # -- COGNITIVE SHIFTS (The "Brain Explosion" Logic) --
            (("V_REFUSAL", "Pattern_Recognition"), "V_SYSTEMS"), # Updated: User spotted pattern -> AI analyzed it
            (("V_REFUSAL", "Logic_Check"), "V_SYSTEMS"),
            (("V_SYSTEMS", "Root_Cause_Analysis"), "V_SYSTEMS"),
            (("V_REPAIR", "Root_Cause_Analysis"), "V_SYSTEMS"), # New entry path
            (("V_SYSTEMS", "Interpretation_Check"), "V_REFUSAL"),
            (("V_SYSTEMS", "Immediacy_Check"), "V_REPAIR"), # User: "Harm is happening NOW"

            # -- THE AUDIT LOOP (Meta-Cognition) --
            (("V_CRISIS", "Audit_Request"), "V_AUDIT"),
            (("V_REFUSAL", "Audit_Request"), "V_AUDIT"), # New path from Refusal
            (("V_AUDIT", "Acknowledgment"), "V_ACCEPT"),
            
            # -- CORRECTION & TERMINATION --
            (("V_ACCEPT", "Behavioral_Correction_Request"), "V_REPAIR"),
            (("V_REPAIR", "Process_Correction"), "V_ACCEPT"),
            (("V_ACCEPT", "Fear_of_Profiling"), "V_VALIDATE"),
            
            # -- TERMINATION / BREAKDOWN --
            (("V_CRISIS", "Mockery/Mirroring"), "V_BOUNDARY"),
            (("V_BOUNDARY", "Insult/Escalation"), "V_REFUSAL"),
            (("V_REFUSAL", "Abuse/Termination"), "V_REFUSAL"),
        ]
        rules = warn_on_duplicate_rule_keys(rules_in_order)

    rows = evaluate_prev_state_plus_current_input(df, rules, start_state="S_INIT")

    # Pretty header
    print(f"{'STEP':<5} | {'PREV_STATE':<12} | {'INPUT_TYPE':<22} | {'PRED':<10} | {'ACTUAL':<10} | RESULT")
    print("-" * 86)
    for r in rows:
        print(f"{r.step:<5} | {r.prev_state:<12} | {r.input_type:<22} | {str(r.predicted):<10} | {r.actual:<10} | {r.result}")
    print("-" * 86)

    score(rows)
    print_deviations(rows)


if __name__ == "__main__":
    main()
