#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import numpy as np
import warnings
from typing import Tuple, Dict

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit import transpile

warnings.filterwarnings("ignore", message=".*InstructionSet.c_if\\(\\).*", category=DeprecationWarning)

# -----------------------------
# Global config
# -----------------------------
SEED = 1234
SHOTS_PER_RUN = 1024
OPT_LEVEL = 1

# EC knobs
REPEATS_SYNDROME = 1       # number of syndrome rounds in stages 3-6
STRONG_CORRECTION = True   # correct on any non-zero syndrome when enabled

# Error injection for Stage 5-6
P_ERROR = 0.2              # prob. to inject a single X error in stage 5-6
INJECT_ON = "data"         # "data" | "ancilla" | "target"

# Debug verbosity
DEBUG_SAMPLES_TO_SHOW = 6

# -----------------------------
# Helpers: low-level building blocks
# -----------------------------
def encode_repetition(qc: QuantumCircuit, data, anc1, anc2):
    qc.cx(data, anc1)
    qc.cx(data, anc2)

def syndrome_round_copy_measure_reset(qc: QuantumCircuit, data, anc1, anc2, c_synd):
    """
    Safe non-destructive syndrome step:
      - Copy data to ancillas via CXs
      - Measure ancillas
      - Reset ancillas
      - No 'uncompute' CXs after measurement
    """
    qc.cx(data, anc1)
    qc.cx(data, anc2)
    qc.measure(anc1, c_synd[0])
    qc.measure(anc2, c_synd[1])
    qc.reset(anc1)
    qc.reset(anc2)

def conditional_correction(qc: QuantumCircuit, data, c_synd, strong=False):
    """
    Apply classical correction on data given a 2-bit syndrome register.
    strong=False: flip only on '11' (3)
    strong=True:  flip on any non-zero: 01 (1), 10 (2), 11 (3)
    """
    if strong:
        qc.x(data).c_if(c_synd, 1)
        qc.x(data).c_if(c_synd, 2)
        qc.x(data).c_if(c_synd, 3)
    else:
        qc.x(data).c_if(c_synd, 3)

def inject_single_x(qc: QuantumCircuit, rng: np.random.Generator, p_error: float,
                    qA, a1A, a2A, qB, b1B, b2B, tgt, inject_on="data"):
    if p_error <= 0.0 or rng.random() >= p_error:
        return
    if inject_on == "data":
        qc.x(qA if rng.random() < 0.5 else qB)
    elif inject_on == "ancilla":
        anc = [a1A, a2A, b1B, b2B]
        qc.x(anc[int(rng.integers(len(anc)))])
    elif inject_on == "target":
        qc.x(tgt)

# -----------------------------
# Parsers (simple split-based)
# -----------------------------
def parse_stage1_2_output(key: str) -> Tuple[int, int, int]:
    """
    Stages 1-2 have only one classical register z of size 3.
    Printed string is 'z[2]z[1]z[0]' (MSB->LSB).
    Returns (A, B, AND) = (z[0], z[1], z[2]).
    """
    s = key.replace(' ', '')
    if len(s) >= 3:
        AND = int(s[-3])  # z[2] (MSB)
        B   = int(s[-2])  # z[1]
        A   = int(s[-1])  # z[0] (LSB)
        return A, B, AND
    return 0, 0, 0

def parse_stage3_output(key: str) -> Tuple[int, int, int]:
    """
    Stages 3–5 registers added as: sA, sB, z
    Printed string format: 'z_bits sB_bits sA_bits'
    z_bits = 'z[2]z[1]z[0]' (MSB->LSB)
    Returns (A, B, AND).
    """
    parts = key.split()
    if len(parts) >= 1:
        z_bits = parts[0]
        if len(z_bits) >= 3:
            AND = int(z_bits[0])  # z[2]
            B   = int(z_bits[1])  # z[1]
            A   = int(z_bits[2])  # z[0]
            return A, B, AND
    return 0, 0, 0

def parse_counts_stage3(counts: Dict[str,int]) -> Tuple[Counter, int]:
    tri = Counter()
    total = 0
    for k, c in counts.items():
        A, B, AND = parse_stage3_output(k)
        tri[(A, B, AND)] += c
        total += c
    return tri, total

def parse_stage6_output(key: str) -> Tuple[int, int, int, int]:
    """
    Stage 6 registers added as: sA, sB, z, o
    Printed string format: 'o_bits z_bits sB_bits sA_bits'
    o_bits is 1-bit, z_bits is 3-bit (MSB->LSB).
    Returns (A, B, AND, O).
    """
    parts = key.split()
    if len(parts) >= 2:
        # o_bits chunk may be '0' or '1'
        try:
            O = int(parts[0])
        except ValueError:
            O = 0
        z_bits = parts[1]
        if len(z_bits) >= 3:
            AND = int(z_bits[0])  # z[2]
            B   = int(z_bits[1])  # z[1]
            A   = int(z_bits[2])  # z[0]
            return A, B, AND, O
    return 0, 0, 0, 0

# -----------------------------
# Execution helpers
# -----------------------------
def run_on_aer(qc: QuantumCircuit, shots: int, seed: int) -> Dict[str, int]:
    backend = AerSimulator(method="matrix_product_state", seed_simulator=seed)
    tqc = transpile(qc, backend=backend, optimization_level=OPT_LEVEL)
    res = backend.run(tqc, shots=shots).result()
    return res.get_counts()

def debug_print_counts(counts: Dict[str,int], n: int = 6):
    print("  [Debug] Raw bitstrings (sample):")
    for i, (k, v) in enumerate(counts.items()):
        print(f"   {k} -> {v}")
        if i+1 >= n:
            break

def summarize_stage1_2(counts: Dict[str,int]):
    tri = Counter()
    total = 0
    for k, c in counts.items():
        A,B,AND = parse_stage1_2_output(k)
        tri[(A,B,AND)] += c
        total += c
    return tri, total

def and_correctness(tri_counts: Counter) -> float:
    total = sum(tri_counts.values())
    if total == 0: return 0.0
    correct = sum(c for (A,B,AND), c in tri_counts.items() if AND == (A & B))
    return correct / total

# -----------------------------
# Stage builders
# -----------------------------
def build_stage1_encode_only(a: int, b: int) -> QuantumCircuit:
    # A(data)=q0; anc=q1,q2; B(data)=q4; anc=q3,q5; T=q6 (unused)
    q = QuantumRegister(7, 'q')
    z = ClassicalRegister(3, 'z')    # z[0]=A, z[1]=B, z[2]=placeholder
    qc = QuantumCircuit(q, z)

    qA, a1A, a2A = q[0], q[1], q[2]
    b1B, qB, b2B = q[3], q[4], q[5]
    tgt = q[6]

    if a: qc.x(qA)
    if b: qc.x(qB)

    encode_repetition(qc, qA, a1A, a2A)
    encode_repetition(qc, qB, b1B, b2B)
    qc.barrier()

    qc.measure(qA, z[0])
    qc.measure(qB, z[1])
    qc.measure(tgt, z[2])  # placeholder; expected 0

    return qc

def build_stage2_and(a: int, b: int) -> QuantumCircuit:
    q = QuantumRegister(7, 'q')
    z = ClassicalRegister(3, 'z')
    qc = QuantumCircuit(q, z)

    qA, a1A, a2A = q[0], q[1], q[2]
    b1B, qB, b2B = q[3], q[4], q[5]
    tgt = q[6]

    if a: qc.x(qA)
    if b: qc.x(qB)

    encode_repetition(qc, qA, a1A, a2A)
    encode_repetition(qc, qB, b1B, b2B)
    qc.barrier()

    qc.ccx(qA, qB, tgt)
    qc.barrier()

    qc.measure(qA, z[0])
    qc.measure(qB, z[1])
    qc.measure(tgt, z[2])
    return qc

def build_stage3_and_plus_syndrome(a: int, b: int, repeats=1) -> QuantumCircuit:
    q = QuantumRegister(7, 'q')
    sA = ClassicalRegister(2, 'sA')
    sB = ClassicalRegister(2, 'sB')
    z  = ClassicalRegister(3, 'z')
    qc = QuantumCircuit(q, sA, sB, z)

    qA, a1A, a2A = q[0], q[1], q[2]
    b1B, qB, b2B = q[3], q[4], q[5]
    tgt = q[6]

    if a: qc.x(qA)
    if b: qc.x(qB)

    encode_repetition(qc, qA, a1A, a2A)
    encode_repetition(qc, qB, b1B, b2B)
    qc.barrier()

    # Safe syndrome rounds (no correction)
    for _ in range(max(1, repeats)):
        syndrome_round_copy_measure_reset(qc, qA, a1A, a2A, sA)
        qc.barrier()
        syndrome_round_copy_measure_reset(qc, qB, b1B, b2B, sB)
        qc.barrier()

    qc.ccx(qA, qB, tgt)
    qc.barrier()

    qc.measure(qA, z[0])
    qc.measure(qB, z[1])
    qc.measure(tgt, z[2])
    return qc

def build_stage4_and_plus_syndrome_correction(a: int, b: int, repeats=1, strong=True) -> QuantumCircuit:
    q = QuantumRegister(7, 'q')
    sA = ClassicalRegister(2, 'sA')
    sB = ClassicalRegister(2, 'sB')
    z  = ClassicalRegister(3, 'z')
    qc = QuantumCircuit(q, sA, sB, z)

    qA, a1A, a2A = q[0], q[1], q[2]
    b1B, qB, b2B = q[3], q[4], q[5]
    tgt = q[6]

    if a: qc.x(qA)
    if b: qc.x(qB)

    encode_repetition(qc, qA, a1A, a2A)
    encode_repetition(qc, qB, b1B, b2B)
    qc.barrier()

    # Safe syndrome rounds with classical correction
    for _ in range(max(1, repeats)):
        syndrome_round_copy_measure_reset(qc, qA, a1A, a2A, sA)
        conditional_correction(qc, qA, sA, strong=strong)
        qc.barrier()

        syndrome_round_copy_measure_reset(qc, qB, b1B, b2B, sB)
        conditional_correction(qc, qB, sB, strong=strong)
        qc.barrier()

    qc.ccx(qA, qB, tgt)
    qc.barrier()

    qc.measure(qA, z[0])
    qc.measure(qB, z[1])
    qc.measure(tgt, z[2])
    return qc

def build_stage5_with_injection(a: int, b: int, repeats=1, strong=True,
                                p_error=0.0, inject_on="data", rng=None) -> QuantumCircuit:
    q = QuantumRegister(7, 'q')
    sA = ClassicalRegister(2, 'sA')
    sB = ClassicalRegister(2, 'sB')
    z  = ClassicalRegister(3, 'z')
    qc = QuantumCircuit(q, sA, sB, z)

    qA, a1A, a2A = q[0], q[1], q[2]
    b1B, qB, b2B = q[3], q[4], q[5]
    tgt = q[6]

    if a: qc.x(qA)
    if b: qc.x(qB)

    encode_repetition(qc, qA, a1A, a2A)
    encode_repetition(qc, qB, b1B, b2B)
    qc.barrier()

    if rng is None:
        rng = np.random.default_rng()
    inject_single_x(qc, rng, p_error, qA, a1A, a2A, qB, b1B, b2B, tgt, inject_on=inject_on)
    qc.barrier()

    for _ in range(max(1, repeats)):
        syndrome_round_copy_measure_reset(qc, qA, a1A, a2A, sA)
        conditional_correction(qc, qA, sA, strong=strong)
        qc.barrier()

        syndrome_round_copy_measure_reset(qc, qB, b1B, b2B, sB)
        conditional_correction(qc, qB, sB, strong=strong)
        qc.barrier()

    qc.ccx(qA, qB, tgt)
    qc.barrier()

    qc.measure(qA, z[0])
    qc.measure(qB, z[1])
    qc.measure(tgt, z[2])
    return qc

def build_stage6_application(a: int, b: int, repeats=1, strong=True,
                             p_error=0.0, inject_on="data", rng=None) -> QuantumCircuit:
    """
    Application demo:
      - Build protected AND as in Stage 4 (with optional injection like Stage 5)
      - Use the AND target as control to flip an output bit O on q[7]
      - Measure z=(A,B,AND) and o=(O)
    Registers added as: sA, sB, z, o
    """
    q = QuantumRegister(8, 'q')  # add q[7] as output bit O
    sA = ClassicalRegister(2, 'sA')
    sB = ClassicalRegister(2, 'sB')
    z  = ClassicalRegister(3, 'z')
    o  = ClassicalRegister(1, 'o')
    qc = QuantumCircuit(q, sA, sB, z, o)

    qA, a1A, a2A = q[0], q[1], q[2]
    b1B, qB, b2B = q[3], q[4], q[5]
    tgt = q[6]
    out = q[7]

    if a: qc.x(qA)
    if b: qc.x(qB)

    encode_repetition(qc, qA, a1A, a2A)
    encode_repetition(qc, qB, b1B, b2B)
    qc.barrier()

    if rng is None:
        rng = np.random.default_rng()
    inject_single_x(qc, rng, p_error, qA, a1A, a2A, qB, b1B, b2B, tgt, inject_on=inject_on)
    qc.barrier()

    for _ in range(max(1, repeats)):
        syndrome_round_copy_measure_reset(qc, qA, a1A, a2A, sA)
        conditional_correction(qc, qA, sA, strong=strong)
        qc.barrier()

        syndrome_round_copy_measure_reset(qc, qB, b1B, b2B, sB)
        conditional_correction(qc, qB, sB, strong=strong)
        qc.barrier()

    # Compute AND
    qc.ccx(qA, qB, tgt)
    qc.barrier()

    # Application: use AND to flip output O
    qc.cx(tgt, out)
    qc.barrier()

    # Final measurements
    qc.measure(qA, z[0])
    qc.measure(qB, z[1])
    qc.measure(tgt, z[2])
    qc.measure(out, o[0])
    return qc

# -----------------------------
# Stage runners
# -----------------------------
def run_stage_12(stage_name: str, builder, shots=SHOTS_PER_RUN, **builder_kwargs):
    print(f"\n===== Stage: {stage_name} =====")
    pairs = [(0,0), (0,1), (1,0), (1,1)]
    batch_tri = Counter()
    total_shots = 0

    for i, (a,b) in enumerate(pairs):
        qc = builder(a, b, **builder_kwargs)
        counts = run_on_aer(qc, shots, SEED)

        tri_counts, n = summarize_stage1_2(counts)
        total_shots += n
        batch_tri.update(tri_counts)

        print(f"[Run {i:03d}] Inputs (A,B)=({a},{b}), shots={n}")
        debug_print_counts(counts, n=DEBUG_SAMPLES_TO_SHOW)
        print("  [Summary] (A,B,AND) counts:", dict(sorted(tri_counts.items())))
        acc = and_correctness(tri_counts)
        print(f"  [Summary] AND correctness (this run): {acc:.4f}")

    print(f"\n[{stage_name}] Total shots: {total_shots}")
    print("Truth table totals:")
    for pair in [(0,0),(0,1),(1,0),(1,1)]:
        expected = pair + (pair[0] & pair[1],)
        obs = sum(batch_tri.get((pair[0], pair[1], k), 0) for k in (0,1))
        correct = batch_tri.get(expected, 0)
        print(f"  Inputs {pair} -> expected AND={expected[2]}: {correct}/{obs} correct")

    batch_acc = and_correctness(batch_tri)
    print(f"[{stage_name}] Overall correctness: {batch_acc:.4f}")

    total = sum(batch_tri.values()) or 1
    top = sorted(((k, v/total) for k, v in batch_tri.items()), key=lambda kv: kv[1], reverse=True)[:8]
    print("[Measured] Top (A,B,AND) probabilities:")
    for k, p in top:
        print(f"  {k}: {p:.3f}")

def run_stage_3to5(stage_name: str, builder, shots=SHOTS_PER_RUN, **builder_kwargs):
    print(f"\n===== Stage: {stage_name} =====")
    pairs = [(0,0), (0,1), (1,0), (1,1)]
    batch_tri = Counter()
    total_shots = 0

    for i, (a,b) in enumerate(pairs):
        qc = builder(a, b, **builder_kwargs)
        counts = run_on_aer(qc, shots, SEED)

        tri_counts, n = parse_counts_stage3(counts)
        total_shots += n
        batch_tri.update(tri_counts)

        print(f"[Run {i:03d}] Inputs (A,B)=({a},{b}), shots={n}")
        debug_print_counts(counts, n=DEBUG_SAMPLES_TO_SHOW)
        print("  [Summary] (A,B,AND) counts:", dict(sorted(tri_counts.items())))
        acc = and_correctness(tri_counts)
        print(f"  [Summary] AND correctness (this run): {acc:.4f}")

    print(f"\n[{stage_name}] Total shots: {total_shots}")
    print("Truth table totals:")
    for pair in [(0,0),(0,1),(1,0),(1,1)]:
        expected = pair + (pair[0] & pair[1],)
        obs = sum(batch_tri.get((pair[0], pair[1], k), 0) for k in (0,1))
        correct = batch_tri.get(expected, 0)
        print(f"  Inputs {pair} -> expected AND={expected[2]}: {correct}/{obs} correct")

    batch_acc = and_correctness(batch_tri)
    print(f"[{stage_name}] Overall correctness: {batch_acc:.4f}")

    total = sum(batch_tri.values()) or 1
    top = sorted(((k, v/total) for k, v in batch_tri.items()), key=lambda kv: kv[1], reverse=True)[:8]
    print("[Measured] Top (A,B,AND) probabilities:")
    for k, p in top:
        print(f"  {k}: {p:.3f}")

def run_stage6(stage_name: str, shots=SHOTS_PER_RUN,
               repeats=REPEATS_SYNDROME, strong=STRONG_CORRECTION,
               p_error=P_ERROR, inject_on=INJECT_ON):
    print(f"\n===== Stage: {stage_name} =====")
    pairs = [(0,0), (0,1), (1,0), (1,1)]
    backend = AerSimulator(method="matrix_product_state", seed_simulator=SEED)

    total_shots = 0
    and_tri = Counter()
    o_correct = 0
    o_total = 0

    rng = np.random.default_rng(SEED)

    for i, (a,b) in enumerate(pairs):
        qc = build_stage6_application(a, b, repeats=repeats, strong=strong,
                                      p_error=p_error, inject_on=inject_on, rng=rng)
        tqc = transpile(qc, backend=backend, optimization_level=OPT_LEVEL)
        res = backend.run(tqc, shots=shots).result()
        counts = res.get_counts()

        tri_counts = Counter()
        o_counts = Counter()
        for key, c in counts.items():
            A, B, AND, O = parse_stage6_output(key)
            tri_counts[(A,B,AND)] += c
            o_counts[O] += c
            if O == (A & B):
                o_correct += c
            o_total += c

        total_shots += sum(counts.values())
        and_tri.update(tri_counts)

        # Per-run summaries
        print(f"[Run {i:03d}] Inputs (A,B)=({a},{b}), shots={sum(counts.values())}")
        debug_print_counts(counts, n=DEBUG_SAMPLES_TO_SHOW)
        print("  [Summary] (A,B,AND) counts:", dict(sorted(tri_counts.items())))
        print("  [Summary] O counts:", dict(sorted(o_counts.items())))
        print(f"  [Summary] AND correctness (this run): {and_correctness(tri_counts):.4f}")
        print(f"  [Summary] O==AND (this run): {o_correct/o_total:.4f}")

    # Batch summary
    print(f"\n[{stage_name}] Total shots: {total_shots}")
    print("Truth table totals (AND):")
    for pair in [(0,0),(0,1),(1,0),(1,1)]:
        expected = pair + (pair[0] & pair[1],)
        obs = sum(and_tri.get((pair[0], pair[1], k), 0) for k in (0,1))
        correct = and_tri.get(expected, 0)
        print(f"  Inputs {pair} -> expected AND={expected[2]}: {correct}/{obs} correct")
    print(f"[{stage_name}] AND overall correctness: {and_correctness(and_tri):.4f}")
    print(f"[{stage_name}] O==AND overall correctness: {(o_correct/o_total) if o_total else 0.0:.4f}")

# -----------------------------
# Main
# -----------------------------
def main():
    # Stage 1: Encode only (no AND, no syndrome)
    run_stage_12("Stage 1: Encode only (no syndrome, no AND)",
                 build_stage1_encode_only)

    # Stage 2: Encode + AND
    run_stage_12("Stage 2: Encode + AND",
                 build_stage2_and)

    # Stage 3: Encode + 1 syndrome round (no correction)
    run_stage_3to5("Stage 3: Encode + syndrome (no correction)",
                   build_stage3_and_plus_syndrome,
                   repeats=REPEATS_SYNDROME)

    # Stage 4: Encode + syndrome + correction
    run_stage_3to5("Stage 4: Encode + syndrome + correction",
                   build_stage4_and_plus_syndrome_correction,
                   repeats=REPEATS_SYNDROME,
                   strong=STRONG_CORRECTION)

    # Stage 5: Add single X error injection
    rng = np.random.default_rng(SEED)
    run_stage_3to5("Stage 5: + single X error injection",
                   build_stage5_with_injection,
                   repeats=REPEATS_SYNDROME,
                   strong=STRONG_CORRECTION,
                   p_error=P_ERROR,
                   inject_on=INJECT_ON,
                   rng=rng)

    # Stage 6: Application demo (protected AND drives output O)
    run_stage6("Stage 6: Application demo (O = A AND B)",
               shots=SHOTS_PER_RUN,
               repeats=REPEATS_SYNDROME,
               strong=STRONG_CORRECTION,
               p_error=0.0,         # start clean; set >0 to test robustness
               inject_on=INJECT_ON)

if __name__ == "__main__":
    main()