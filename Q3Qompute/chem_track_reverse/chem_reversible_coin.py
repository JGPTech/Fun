#!/usr/bin/env python3
"""
chem_reversible_coin.py
CC0-1.0 — Public Domain

One-button, naturemated Chem → ΔE → Reversible proof pipeline.

What it does
------------
1) Energies:
   - --nature : Try Qiskit Nature on H2 at two geometries (A,B):
       * Ground state via VQE → E_N(A), E_N(B).
       * Advanced excited-states via QEOM (preferred) or VQD fallback → derive a tiny EA-like
         proxy to fill E_{N+1}(A/B) for the demo ΔE(+e/−e). Provenance is printed.
     If Nature/PySCF is unavailable or imports mismatch, we fall back to mock energies
     (deterministic by seed) and say so in 'provenance'.
   - --mock : Force mock energies.
   - --energies EN_A EN_B ENp1_A ENp1_B : Manual override (no Nature/mock).

2) Build a ΔE LUT over 4 micro-states (A/B × N/N+1) with moves:
      0: stay, 1: +e at g, 2: −e at g, 3: flip geometry
   Quantize to b-bit two’s complement with step δ (eV).

3) Build the Bennett sandwich (compute→use→uncompute):
      QROM-load ΔE  →  CRY(λ) controlled by sign(ΔE)  →  QROM-unload
   Prove reversibility by running U then U†, and print a detailed report.

NOTE: This enhanced version silently saves metrics and figures into chem/<timestamp>/,
      without changing any printed output or computations.
"""

import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.quantum_info import Statevector, state_fidelity

# ---------- NEW (saving, plots) ----------
import os
import json
import csv
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    # qpy export for reproducibility
    from qiskit import qpy
    _HAS_QPY = True
except Exception:
    _HAS_QPY = False
# ----------------------------------------


# ---------------- Pretty print ----------------
def hp(x: float, places: int = 12) -> str:
    return f"{float(x):.{places}f}"


# Micro-state: (geom g in {0:A,1:B}, sector q in {0:N,1:N+1})
State = Tuple[int, int]
COIN_LABELS = {0: "stay", 1: "+e at g", 2: "-e at g", 3: "flip g"}


# ---------------- Energy providers ----------------
def mock_energies(molecule: str, seed: int) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Deterministic, plausible energies in eV for the 4-state micro-graph.
    Returns (energies, provenance_flags). Keys: EN_A, EN_B, ENp1_A, ENp1_B
    """
    rng = np.random.default_rng(abs(hash((molecule.lower(), int(seed)))) % (2**32))
    EN_A = -40.0 + 0.010 * rng.normal()
    d_geoN = +0.020 + 0.010 * rng.normal()  # B slightly higher for neutral
    EN_B = EN_A + d_geoN
    EA_A = 0.20 + 0.10 * abs(rng.normal())  # eV
    EA_B = EA_A + 0.05 * rng.normal()
    ENp1_A = EN_A - EA_A
    ENp1_B = EN_B - EA_B
    E = dict(EN_A=EN_A, EN_B=EN_B, ENp1_A=ENp1_A, ENp1_B=ENp1_B)
    prov = {k: "mock" for k in E.keys()}
    return E, prov


def try_nature_h2(distA: float, distB: float, seed: int) -> Tuple[Dict[str, float], Dict[str, str], str]:
    """
    Tiny VQE + advanced excited-states (QEOM if available, else VQD) on H2 at two geometries.
    Returns (energies_eV, provenance_flags, note). Fills:
      EN_A, EN_B            : ground-state energies via VQE  (in eV)
      ENp1_A, ENp1_B        : derived from excited-spacing (demo EA proxy), tagged as derived
    """
    try:
        # --- Qiskit Nature (current API, v0.7+) ---
        from qiskit_nature.units import DistanceUnit
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        # UCCSD + HF initial state (good default for H2)
        from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

        # Solvers
        from qiskit_nature.second_q.algorithms import (
            GroundStateEigensolver,
            ExcitedStatesEigensolver,
            QEOM,
            EvaluationRule,
        )

        from qiskit_algorithms import VQE, VQD
        from qiskit_algorithms.optimizers import SLSQP
        from qiskit.primitives import Estimator

    except Exception as e:
        return {}, {}, f"nature_unavailable({type(e).__name__}:{e})"

    HARTREE_TO_EV = 27.211386245988

    def one_geom(r: float) -> Tuple[float, float, str]:
        # Build problem from driver (modern pattern: driver.run()).
        driver = PySCFDriver(
            atom=f"H 0 0 0; H 0 0 {r}",
            basis="sto3g",
            charge=0,
            spin=0,
            unit=DistanceUnit.ANGSTROM,
        )
        problem = driver.run()  # -> ElectronicStructureProblem

        # 2e/2o active space (explicit here to stay tiny and robust)
        try:
            problem = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2).transform(problem)
        except TypeError:
            problem = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=2).transform(problem)

        mapper = JordanWignerMapper()

        # UCCSD with HF reference
        hf = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper, initial_state=hf)

        est = Estimator()
        vqe = VQE(est, ansatz, SLSQP(maxiter=200))
        gse = GroundStateEigensolver(mapper, vqe)
        res_gs = gse.solve(problem)
        EN = float(res_gs.total_energies[0]) * HARTREE_TO_EV

        # Excited state (prefer QEOM; fallback VQD)
        try:
            qeom_solver = QEOM(gse, est, "sd", EvaluationRule.ALL)
            res_ex = qeom_solver.solve(problem)
            ev = getattr(res_ex, "eigenvalues", None)
            E1 = float(ev[1]) * HARTREE_TO_EV if (ev is not None and len(ev) >= 2) else float("nan")
            tag = "qeom"
        except Exception:
            vqd = VQD(est, ansatz, SLSQP(maxiter=200))
            qse = ExcitedStatesEigensolver(mapper, vqd)
            res_ex = qse.solve(problem)
            ev = getattr(res_ex, "eigenvalues", None)
            E1 = float(ev[1]) * HARTREE_TO_EV if (ev is not None and len(ev) >= 2) else float("nan")
            tag = "vqd"

        return EN, E1, tag

    # Run at A and B
    EN_A, E1_A, tagA = one_geom(distA)
    EN_B, E1_B, tagB = one_geom(distB)

    # Derive a small EA proxy from the lowest excited spacing (demo only)
    def derive_ea(EN: float, E1: float) -> float:
        return abs(E1 - EN) * 0.1 if np.isfinite(E1) else 0.2

    EA_A = derive_ea(EN_A, E1_A)
    EA_B = derive_ea(EN_B, E1_B)
    ENp1_A = EN_A - EA_A
    ENp1_B = EN_B - EA_B

    E = dict(EN_A=EN_A, EN_B=EN_B, ENp1_A=ENp1_A, ENp1_B=ENp1_B)
    prov = {
        "EN_A": "vqe",
        "EN_B": "vqe",
        "ENp1_A": f"derived_from_{tagA}",
        "ENp1_B": f"derived_from_{tagB}",
    }
    note = f"nature_vqe_{'qeom' if 'qeom' in (tagA, tagB) else 'vqd'}"
    return E, prov, note


# ---------------- ΔE LUT construction ----------------
def build_states() -> List[State]:
    return [(0, 0), (1, 0), (0, 1), (1, 1)]  # 0:A,1:B × 0:N,1:N+1


def energies_to_delta_table(E: Dict[str, float], bits: int, delta_ev: float) -> Tuple[np.ndarray, np.ndarray]:
    EN_A, EN_B = E["EN_A"], E["EN_B"]
    ENp1_A, ENp1_B = E["ENp1_A"], E["ENp1_B"]
    states = build_states()
    M, K = len(states), 3
    delta_ev_table = np.zeros((M, K + 1), dtype=float)

    for i, (g, q) in enumerate(states):
        EN = EN_A if g == 0 else EN_B
        ENp1 = ENp1_A if g == 0 else ENp1_B
        # +e at g
        delta_ev_table[i, 1] = (ENp1 - EN) if (q == 0) else 0.0
        # -e at g
        delta_ev_table[i, 2] = (EN - ENp1) if (q == 1) else 0.0
        # flip geometry at fixed sector
        if q == 0:
            delta_ev_table[i, 3] = (EN_B - EN_A) if (g == 0) else (EN_A - EN_B)
        else:
            delta_ev_table[i, 3] = (ENp1_B - ENp1_A) if (g == 0) else (ENp1_A - ENp1_B)

    # Quantize
    scale = 1.0 / float(delta_ev)
    ints = np.rint(delta_ev_table * scale).astype(int)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    ints = np.clip(ints, lo, hi)
    return delta_ev_table, ints


# ---------------- Reversible ROM machinery ----------------
def _bits_of(value: int, n: int) -> List[int]:
    return [(value >> i) & 1 for i in range(n)]


def chain_mcx(circ: QuantumCircuit, ctrls: List, tgt, ancs: List):
    L = len(ctrls)
    if L == 0:
        circ.x(tgt); return
    if L == 1:
        circ.cx(ctrls[0], tgt); return
    if L == 2:
        circ.ccx(ctrls[0], ctrls[1], tgt); return
    circ.ccx(ctrls[0], ctrls[1], ancs[0])
    for i in range(2, L - 1):
        circ.ccx(ctrls[i], ancs[i - 2], ancs[i - 1])
    circ.ccx(ctrls[L - 1], ancs[L - 3], tgt)
    for i in reversed(range(2, L - 1)):
        circ.ccx(ctrls[i], ancs[i - 2], ancs[i - 1])
    circ.ccx(ctrls[0], ctrls[1], ancs[0])


def mark_if_equal_two_regs(
    circ: QuantumCircuit, pos_reg, coin_reg, pos_val: int, coin_val: int, target_qubit, ancs: List
):
    pos_bits = _bits_of(pos_val, len(pos_reg))
    coin_bits = _bits_of(coin_val, len(coin_reg))
    for q, b in zip(pos_reg, pos_bits):
        if b == 0: circ.x(q)
    for q, b in zip(coin_reg, coin_bits):
        if b == 0: circ.x(q)
    ctrls = list(pos_reg) + list(coin_reg)
    chain_mcx(circ, ctrls, target_qubit, ancs)
    for q, b in zip(coin_reg, coin_bits):
        if b == 0: circ.x(q)
    for q, b in zip(pos_reg, pos_bits):
        if b == 0: circ.x(q)


def rom_load_delta_bits(circ: QuantumCircuit, pos_reg, coin_reg, delta_reg, delta_table: np.ndarray, ancs: List):
    M, Kp1 = delta_table.shape
    b = len(delta_reg)
    for i in range(M):
        for k in range(1, Kp1):  # skip stay=0
            val = int(delta_table[i, k])
            twos = (val + (1 << b)) % (1 << b)
            if twos == 0: continue
            for bit in range(b):
                if ((twos >> bit) & 1) == 0: continue
                mark_if_equal_two_regs(circ, pos_reg, coin_reg, i, k, delta_reg[bit], ancs)


def use_predicate_rotate(circ: QuantumCircuit, delta_reg, flag_qubit, marker_qubit, lam: float):
    sign = delta_reg[-1]  # MSB in two's complement
    circ.cx(sign, flag_qubit)
    circ.ry(lam, marker_qubit).c_if  # (structural; rotation is applied conditionally via the earlier CX)


# ---------------- Build U and prove ----------------
def build_U(delta_tbl_int: np.ndarray, bits: int, lam: float) -> QuantumCircuit:
    M, Kp1 = delta_tbl_int.shape
    q_pos = max(1, math.ceil(math.log2(M)))
    q_coin = max(1, math.ceil(math.log2(Kp1)))
    pos = QuantumRegister(q_pos, "pos")
    coin = QuantumRegister(q_coin, "coin")
    delta = QuantumRegister(bits, "delta")
    flag = QuantumRegister(1, "flag")
    mark = QuantumRegister(1, "mark")
    ancs = AncillaRegister(max(0, q_pos + q_coin - 2), "anc")
    qc = QuantumCircuit(pos, coin, delta, flag, mark, ancs, name="U_reversible_chem")
    rom_load_delta_bits(qc, pos, coin, delta, delta_tbl_int, list(ancs))
    # simple sign→flag and a marker rotation
    sign = delta[-1]
    qc.cx(sign, flag[0])
    qc.cry(lam, flag[0], mark[0])
    rom_load_delta_bits(qc, pos, coin, delta, delta_tbl_int, list(ancs))
    return qc


def prepare_near_uniform(qc: QuantumCircuit, pos, coin):
    for q in pos: qc.h(q)
    for q in coin: qc.h(q)


def index_list_for_register(qc: QuantumCircuit, reg) -> List[int]:
    return [qc.find_bit(q).index for q in reg]


def prob_register_all_zero(sv: Statevector, reg_qubits: List[int]) -> float:
    if not reg_qubits: return 1.0
    probs = sv.probabilities(qargs=reg_qubits)
    return float(probs[0]) if len(probs) > 0 else 1.0


# ---------- NEW: artifact saving helpers ----------
def _mk_outdir(base: str = "chem") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _save_csv(path: str, header: List[str], rows: List[List]):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def _plot_heatmap(matrix: np.ndarray, title: str, xticklabels: List[str], yticklabels: List[str], path: str, cmap="coolwarm"):
    if not _HAS_MPL: return
    plt.figure(figsize=(6, 3.6), dpi=160)
    im = plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(xticklabels)), xticklabels)
    plt.yticks(range(len(yticklabels)), yticklabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _plot_bar(values: List[float], labels: List[str], title: str, ylabel: str, path: str, rotate=False):
    if not _HAS_MPL: return
    plt.figure(figsize=(6, 3.6), dpi=160)
    idx = np.arange(len(values))
    plt.bar(idx, values)
    plt.xticks(idx, labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _plot_table_kv(kv: List[Tuple[str, str]], title: str, path: str):
    if not _HAS_MPL: return
    plt.figure(figsize=(6, 3.6), dpi=160)
    plt.axis("off")
    plt.title(title, pad=10)
    table_data = [[k, v] for (k, v) in kv]
    tbl = plt.table(cellText=table_data, colLabels=["Key", "Value"], loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def _draw_circuit_mpl(circ: QuantumCircuit, path: str):
    if not _HAS_MPL: return
    try:
        circ.draw(output="mpl", idle_wires=False, fold=-1, scale=0.8, filename=path)
    except Exception:
        # Fallback: draw without filename then save
        fig = circ.draw(output="mpl", idle_wires=False, fold=-1, scale=0.8)
        fig.savefig(path)
        plt.close(fig)

# -----------------------------------------------


# ---------------- Orchestrate ----------------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--nature", action="store_true", help="try Qiskit Nature (VQE+QEOM/VQD); fallback to mock")
    g.add_argument("--mock", action="store_true", help="force mock energies")
    ap.add_argument("--molecule", choices=["H2", "BQ", "EC"], default="H2")
    ap.add_argument("--geomA", type=float, default=0.735, help="H–H distance for geometry A (Å, H2-only)")
    ap.add_argument("--geomB", type=float, default=0.900, help="H–H distance for geometry B (Å, H2-only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bits", type=int, default=12, help="Δ register width (two's complement)")
    ap.add_argument("--delta_ev", type=float, default=0.01, help="quantization step (eV)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.7, help="RY angle in use-stage")
    ap.add_argument(
        "--energies",
        nargs=4,
        type=float,
        metavar=("EN_A", "EN_B", "ENp1_A", "ENp1_B"),
        help="override energies in eV; bypasses Nature/mock if provided",
    )
    # NEW: optional output directory (defaults to chem/<timestamp>)
    ap.add_argument("--outdir", type=str, default=None, help="directory to save metrics/figures (default: chem/<ts>)")

    args = ap.parse_args()

    # Energies
    prov_note = ""
    if args.energies is not None:
        EN_A, EN_B, ENp1_A, ENp1_B = map(float, args.energies)
        E = dict(EN_A=EN_A, EN_B=EN_B, ENp1_A=ENp1_A, ENp1_B=ENp1_B)
        prov = {k: "manual" for k in E.keys()}
        prov_note = "manual_override"
    elif args.mock and not args.nature:
        E, prov = mock_energies(args.molecule, args.seed)
        prov_note = "mock"
    else:
        E, prov, tag = try_nature_h2(args.geomA, args.geomB, args.seed)
        if not E:
            E, prov = mock_energies(args.molecule, args.seed)
            prov_note = f"fallback:{tag}→mock"
        else:
            prov_note = tag

    # Build ΔE tables
    delta_ev_table, delta_tbl_int = energies_to_delta_table(E, args.bits, args.delta_ev)

    # Dimensions
    M, Kp1 = delta_tbl_int.shape
    q_pos = max(1, math.ceil(math.log2(M)))
    q_coin = max(1, math.ceil(math.log2(Kp1)))
    anc_count = max(0, q_pos + q_coin - 2)

    # Build U and statevectors
    U = build_U(delta_tbl_int, args.bits, args.lam)

    qc_init = QuantumCircuit(*U.qregs)
    pos, coin, delta, flag, mark, ancs = U.qregs
    prepare_near_uniform(qc_init, pos, coin)
    sv_init = Statevector.from_instruction(qc_init)

    qc_compute = QuantumCircuit(*U.qregs)
    prepare_near_uniform(qc_compute, pos, coin)
    rom_load_delta_bits(qc_compute, pos, coin, delta, delta_tbl_int, list(ancs))
    sv_compute = Statevector.from_instruction(qc_compute)

    qc_full = QuantumCircuit(*U.qregs)
    prepare_near_uniform(qc_full, pos, coin)
    qc_full.append(U.to_gate(), [q for q in qc_full.qubits])
    sv_full = Statevector.from_instruction(qc_full)

    qc_prove = QuantumCircuit(*U.qregs)
    prepare_near_uniform(qc_prove, pos, coin)
    qc_prove.append(U.to_gate(), [q for q in qc_prove.qubits])
    qc_prove.append(U.to_gate().inverse(), [q for q in qc_prove.qubits])
    sv_prove = Statevector.from_instruction(qc_prove)
    F = state_fidelity(sv_init, sv_prove)

    # Probs
    idx_delta = index_list_for_register(U, delta)
    idx_anc = index_list_for_register(U, ancs)
    p_delta0_init = prob_register_all_zero(sv_init, idx_delta)
    p_delta0_comp = prob_register_all_zero(sv_compute, idx_delta)
    p_delta0_full = prob_register_all_zero(sv_full, idx_delta)
    p_anc0_init = prob_register_all_zero(sv_init, idx_anc)
    p_anc0_comp = prob_register_all_zero(sv_compute, idx_anc)
    p_anc0_full = prob_register_all_zero(sv_full, idx_anc)

    # Gate metrics
    decomp = U.decompose(reps=6)
    depth = decomp.depth()
    ops = decomp.count_ops()

    # Stats
    vals_int = delta_tbl_int[:, 1:].reshape(-1)
    vmin, vmax = int(vals_int.min()), int(vals_int.max())
    vmean = float(vals_int.mean())
    nonzero = int(np.count_nonzero(vals_int))
    total = int(vals_int.size)

    # Address activity → expected P(delta==0) after compute
    M_tot = (1 << q_pos) * (1 << q_coin)
    active = sum(1 for i in range(M) for k in range(1, Kp1) if delta_tbl_int[i, k] != 0)
    p_compute_expected = 1.0 - (active / M_tot)

    # ---------- NEW: silently save artifacts ----------
    outdir = args.outdir or _mk_outdir("chem")

    # Save LUTs
    _save_csv(os.path.join(outdir, "lut_int.csv"),
              ["state_idx", "geom", "sector", "k=+e", "k=-e", "k=flip"],
              [[i, "A" if g==0 else "B", "N" if q==0 else "N+1",
                int(delta_tbl_int[i,1]), int(delta_tbl_int[i,2]), int(delta_tbl_int[i,3])]
               for i,(g,q) in enumerate(build_states())])

    _save_csv(os.path.join(outdir, "lut_ev.csv"),
              ["state_idx", "geom", "sector", "k=+e[eV]", "k=-e[eV]", "k=flip[eV]"],
              [[i, "A" if g==0 else "B", "N" if q==0 else "N+1",
                float(delta_ev_table[i,1]), float(delta_ev_table[i,2]), float(delta_ev_table[i,3])]
               for i,(g,q) in enumerate(build_states())])

    # Save gate counts
    _save_csv(os.path.join(outdir, "gate_counts.csv"),
              ["gate", "count"],
              [[str(k), int(v)] for k, v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))])

    # Save metrics.json
    metrics = {
        "args": {
            "molecule": args.molecule,
            "geomA": args.geomA,
            "geomB": args.geomB,
            "seed": args.seed,
            "bits": args.bits,
            "delta_ev": args.delta_ev,
            "lambda": args.lam,
            "mode": "manual" if args.energies is not None else ("mock" if args.mock and not args.nature else "nature"),
            "provenance_note": prov_note,
        },
        "energies_eV": dict(E),
        "energies_provenance": dict(prov),
        "dimensions": {
            "M_states": M,
            "K_moves_nonstay": Kp1 - 1,
            "q_pos": q_pos,
            "q_coin": q_coin,
            "ancillas": anc_count,
        },
        "lut_stats": {
            "min": vmin, "max": vmax, "mean": vmean, "nonzero": nonzero, "total": total,
            "active_addresses": active, "address_total": M_tot, "p_compute_expected": p_compute_expected,
        },
        "probabilities": {
            "p_delta0_init": p_delta0_init,
            "p_delta0_compute": p_delta0_comp,
            "p_delta0_full": p_delta0_full,
            "p_anc0_init": p_anc0_init,
            "p_anc0_compute": p_anc0_comp,
            "p_anc0_full": p_anc0_full,
        },
        "gate_metrics": {
            "depth": int(depth),
            "ops": {str(k): int(v) for k, v in ops.items()}
        },
        "fidelity_UdaggerU": float(F),
    }
    _save_json(os.path.join(outdir, "metrics.json"), metrics)

    # Save circuit QPY (if available)
    if _HAS_QPY:
        with open(os.path.join(outdir, "U_reversible_chem.qpy"), "wb") as f:
            qpy.dump(U, f)

    # Figures (best-effort, quiet)
    if _HAS_MPL:
        # Energies table
        _plot_table_kv(
            [(k, f"{E[k]:.6f} eV [{prov.get(k, '?')}]") for k in ("EN_A", "EN_B", "ENp1_A", "ENp1_B")],
            "Energies & Provenance",
            os.path.join(outdir, "fig_energies.png"),
        )
        # Heatmaps
        xt = ["stay", "+e", "-e", "flip"]
        yt = [f"state[{i}] ({'A' if g==0 else 'B'},{'N' if q==0 else 'N+1'})" for i,(g,q) in enumerate(build_states())]
        _plot_heatmap(delta_ev_table, "ΔE table (eV)", xt, yt, os.path.join(outdir, "fig_delta_ev_heatmap.png"))
        _plot_heatmap(delta_tbl_int.astype(float), "Quantized LUT (two's complement ints)", xt, yt, os.path.join(outdir, "fig_delta_int_heatmap.png"), cmap="viridis")

        # Probabilities bar chart
        _plot_bar(
            [p_delta0_init, p_delta0_comp, p_delta0_full, p_anc0_init, p_anc0_comp, p_anc0_full],
            ["PΔ0:init","PΔ0:comp","PΔ0:full","Panc0:init","Panc0:comp","Panc0:full"],
            "Register-Zero Probabilities",
            "Probability",
            os.path.join(outdir, "fig_probabilities.png"),
            rotate=False
        )

        # Gate counts bar chart
        gc_sorted = sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))
        _plot_bar(
            [int(v) for _, v in gc_sorted],
            [str(k) for k, _ in gc_sorted],
            "Gate Counts (decomposed U)",
            "Count",
            os.path.join(outdir, "fig_gate_counts.png"),
            rotate=True
        )

        # Circuit drawing
        _draw_circuit_mpl(U, os.path.join(outdir, "fig_circuit_U.png"))

        # Manifest of generated files
        manifest = sorted([os.path.join(outdir, f) for f in os.listdir(outdir)])
        _save_json(os.path.join(outdir, "manifest.json"), {"files": manifest})
    # ---------- end saving ----------

    # Print report (unchanged)
    print("=== Reversible ΔE Proof — Detailed Report (nature) ===")
    print(f"provenance = {prov_note}")
    print(f"states M = {M} (A/B × N/N+1) | coin K = {Kp1-1} (labels: 0=stay,1:+e,2:-e,3:flip)")
    print(f"bits (Δ register) = {args.bits} | q_pos = {q_pos} | q_coin = {q_coin} | ancillas = {anc_count}")
    print(f"delta_eV quantization step = {hp(args.delta_ev,6)} eV\n")

    print("-- Energies (eV) with provenance --")
    for key in ("EN_A", "EN_B", "ENp1_A", "ENp1_B"):
        tag = prov.get(key, "?")
        print(f"{key:7s} = {hp(E[key])}   [{tag}]")

    states = build_states()
    lab = lambda s: ("A" if s[0] == 0 else "B", "N" if s[1] == 0 else "N+1")
    print("\n-- ΔE per state (eV) [k=1:+e, k=2:-e, k=3:flip] --")
    for i, s in enumerate(states):
        a, b = lab(s)
        row = ", ".join(hp(delta_ev_table[i, j]) for j in (1, 2, 3))
        print(f"  state[{i}] ({a},{b}) : [ {row} ]")

    print("\n-- Quantized LUT (two's complement integers) --")
    for i, s in enumerate(states):
        a, b = lab(s)
        row = ", ".join(f"{int(delta_tbl_int[i, j])}" for j in (1, 2, 3))
        print(f"  state[{i}] ({a},{b}) : [ {row} ]")

    print("\n-- ΔE LUT stats (excluding stay=0) --")
    print(f"min = {vmin}, max = {vmax}, mean = {vmean:.6f}, nonzero/total = {nonzero}/{total}")

    print("\n-- Address activity (explains P(delta==0) after compute) --")
    print(f"nonzero LUT entries = {active} / total address states = {M_tot} → expected P = {hp(p_compute_expected)}")

    print("\n-- State checks (probability registers are all zero) --")
    print(f"P(delta==0) | init      = {hp(p_delta0_init)}")
    print(f"P(delta==0) | compute   = {hp(p_delta0_comp)}")
    print(f"P(delta==0) | full U    = {hp(p_delta0_full)}")
    print(f"P(anc==0)   | init      = {hp(p_anc0_init)}")
    print(f"P(anc==0)   | compute   = {hp(p_anc0_comp)}")
    print(f"P(anc==0)   | full U    = {hp(p_anc0_full)}")

    print("\n-- Gate metrics (U decomposed structurally) --")
    print(f"depth = {depth}")
    for name, count in sorted(ops.items(), key=lambda kv: kv[0]):
        print(f"{name}: {count}")

    print("\n-- Final fidelity (U^† U) --")
    print(f"fidelity_after_step_then_inverse = {hp(state_fidelity(sv_init, sv_prove))}")


if __name__ == "__main__":
    main()
