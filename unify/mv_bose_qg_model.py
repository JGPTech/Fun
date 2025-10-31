#!/usr/bin/env python3
# mv_bose_qg_model.py
# Model A — Marletto–Vedral / Bose entanglement witness (quantum mediator)
# Single-file, importable module + CLI.
# CC0-1.0 — Public Domain

"""
MODEL SUMMARY
-------------
Implements the Bose–Mazumdar–et al. (PRL 119, 240401, 2017) witness:
two mesoscopic masses are placed in spin-dependent spatial superposition
via Stern–Gerlach (SG). During the “hold” time τ the masses interact
*only via gravity*, picking up *branch-dependent* gravitational phases.

After recombination, the SG sequence maps the motional phases into a
two-spin state. If that spin state is entangled (e.g., negativity > 0),
then the mediator that carried phase information between *spatially
separated* systems must possess non-commuting observables — i.e., a
*quantum* mediator.

We provide:
  • Exact two-spin state generated from branch phases Δφ_LR, Δφ_RL
  • Entanglement measures: concurrence, negativity (pure-state case),
    plus several simple “witness-like” correlators for convenience
  • Canonical “paper-faithful” parameter set
  • PASS/FAIL 1-button check (negativity > 0)
  • Programmatic API for live data generation from other scripts
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, Optional

import numpy as np


# ====== Physical constants (SI) ======
G = 6.67430e-11           # Newton’s constant [m^3 kg^-1 s^-2]
HBAR = 1.054_571_817e-34  # Reduced Planck constant [J s]


# ====== Dataclasses for clarity ======
@dataclass
class Geometry:
    """
    SG geometry for two adjacent interferometers, following the schematic in
    Bose et al.: two path centers separated by d; intraparticle path split Δx.
      r_LL = d
      r_RR = d
      r_LR = d + Δx
      r_RL = d - Δx
    """
    d: float        # center-to-center separation between superpositions [m]
    dx: float       # path separation within each interferometer [m]


@dataclass
class MassesAndTime:
    m1: float       # mass 1 [kg]
    m2: float       # mass 2 [kg]
    tau: float      # free-evolution (hold) time [s]


@dataclass
class Phases:
    phi_d: float         # φ(d)
    phi_LR: float        # φ(d + Δx)
    phi_RL: float        # φ(d - Δx)
    dphi_LR: float       # Δφ_LR = φ(d+Δx) - φ(d)
    dphi_RL: float       # Δφ_RL = φ(d-Δx) - φ(d)


# ====== Core physics ======
def gravitational_phase(r: float, m1: float, m2: float, tau: float) -> float:
    """φ(r) = (G m1 m2 τ)/(ħ r)   (dimensionless, radians)."""
    return (G * m1 * m2 * tau) / (HBAR * r)


def compute_branch_phases(geom: Geometry, mt: MassesAndTime) -> Phases:
    """Compute branch phases φ and deltas Δφ as used in the SG mapping."""
    d, dx = geom.d, geom.dx
    phi_d  = gravitational_phase(d,       mt.m1, mt.m2, mt.tau)
    phi_LR = gravitational_phase(d + dx,  mt.m1, mt.m2, mt.tau)
    phi_RL = gravitational_phase(d - dx,  mt.m1, mt.m2, mt.tau)
    dphi_LR = phi_LR - phi_d
    dphi_RL = phi_RL - phi_d
    return Phases(phi_d=phi_d, phi_LR=phi_LR, phi_RL=phi_RL,
                  dphi_LR=dphi_LR, dphi_RL=dphi_RL)


def final_spin_state_from_phases(dphi_LR: float, dphi_RL: float) -> np.ndarray:
    """
    SG step-1 (split), hold (gravity phases), step-3 (recombine)
    → spin-entangled state (two-qubit pure state) of the form:

        |Ψ⟩ ∝ |↑⟩₁ (|↑⟩₂ + e^{i Δφ_LR} |↓⟩₂)  +  |↓⟩₁ (e^{i Δφ_RL} |↑⟩₂ + |↓⟩₂)

    In {|↑↑⟩,|↑↓⟩,|↓↑⟩,|↓↓⟩} order, the normalized state vector is:
        v = (1, e^{iα}, e^{iβ}, 1) / 2,  with α=Δφ_LR, β=Δφ_RL.

    Returns:
      v : shape (4, ) complex128 normalized pure state.
    """
    alpha = float(dphi_LR)
    beta  = float(dphi_RL)
    v = np.array([1.0,
                  np.exp(1j * alpha),
                  np.exp(1j * beta),
                  1.0], dtype=np.complex128) / 2.0
    # already normalized: ||v||^2 = (1+1+1+1)/4 = 1
    return v


# ====== Entanglement & correlators ======
def concurrence_pure(v: np.ndarray) -> float:
    """Wootters concurrence for a *pure* 2-qubit state vector."""
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    psi = v.reshape(4, 1)
    C = abs((psi.T @ (np.kron(sy, sy)) @ psi.conj()).item())
    return float(np.real_if_close(C))


def negativity_pure(v: np.ndarray) -> float:
    """
    Negativity for a *pure* two-qubit state (equivalent to concurrence/2).
    For pure states of two qubits: N = C/2.
    """
    return 0.5 * concurrence_pure(v)


def pauli_expectations(v: np.ndarray) -> Dict[str, float]:
    """Handy correlators for quick diagnostics (not used for PASS/FAIL)."""
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    rho = np.outer(v, v.conj())
    def ex(A, B): return float(np.real_if_close(np.trace(rho @ np.kron(A, B))))
    vals = {
        "xx": ex(sx, sx),
        "yy": ex(sy, sy),
        "zz": ex(sz, sz),
        "xz": ex(sx, sz),
        "yz": ex(sy, sz),
        "zx": ex(sz, sx),
        "zy": ex(sz, sy),
    }
    # Bose-style simple witness forms (for reference)
    vals["xx_plus_yy_abs"] = abs(vals["xx"] + vals["yy"])
    vals["xz_minus_yz_abs"] = abs(vals["xz"] - vals["yz"])
    return vals


# ====== Public API ======
@dataclass
class RunParams:
    geom: Geometry
    mt: MassesAndTime
    # Optional: simple phenomenological dephasing on each spin (step-2 hold)
    # applied after constructing the pure state. gamma in [0,1] with 1=full dephase.
    gamma_dephase: float = 0.0


def apply_local_dephase(rho: np.ndarray, p: float) -> np.ndarray:
    """
    Single-qubit phase-damping channel on both qubits with parameter p in [0,1].
    Kraus set {E0 = |0><0| + sqrt(1-p)|1><1|, E1 = sqrt(p)|1><1|}, applied per qubit.
    """
    # in computational basis |0>=|↑>, |1>=|↓>
    sq = math.sqrt(max(0.0, 1.0 - p))
    E0 = np.array([[1.0, 0.0], [0.0, sq]], dtype=np.complex128)
    E1 = np.array([[0.0, 0.0], [0.0, math.sqrt(max(0.0, p))]], dtype=np.complex128)

    def chan_one(r, which="A"):
        if which == "A":
            K0 = np.kron(E0, np.eye(2))
            K1 = np.kron(E1, np.eye(2))
        else:
            K0 = np.kron(np.eye(2), E0)
            K1 = np.kron(np.eye(2), E1)
        return K0 @ r @ K0.conj().T + K1 @ r @ K1.conj().T

    r = rho.copy()
    r = chan_one(r, "A")
    r = chan_one(r, "B")
    return r


def run_once(params: RunParams) -> Dict[str, Any]:
    """
    Single “experiment”: compute phases, build the final two-spin state,
    optionally apply dephasing, and report entanglement + correlators.
    """
    ph = compute_branch_phases(params.geom, params.mt)
    v  = final_spin_state_from_phases(ph.dphi_LR, ph.dphi_RL)
    rho = np.outer(v, v.conj())

    if params.gamma_dephase > 0:
        rho = apply_local_dephase(rho, params.gamma_dephase)
        # for mixed states, negativity needs partial transpose eigenvalues;
        # keep a robust pure-state path for gamma=0 and a fallback otherwise.
        # But we want a simple PASS/FAIL path; for gamma=0 use pure formulas.
        # For gamma>0, we compute negativity via partial transpose.
        neg = negativity_via_partial_transpose(rho)
        conc = None  # not meaningful for general mixed states in this simple form
    else:
        conc = concurrence_pure(v)
        neg  = 0.5 * conc

    cor = pauli_expectations(v)

    return {
        "input": {
            "geom": asdict(params.geom),
            "masses_time": asdict(params.mt),
            "gamma_dephase": params.gamma_dephase,
        },
        "phases": asdict(ph),
        "entanglement": {
            "concurrence": conc,
            "negativity": neg,
        },
        "correlators": cor,
    }


def negativity_via_partial_transpose(rho: np.ndarray) -> float:
    """
    Negativity for a general two-qubit mixed state via partial transpose on qubit-B.
    N = (||ρ^{T_B}||_1 - 1)/2 = sum(|λ_i^-|) over negative eigenvalues of ρ^{T_B}.
    """
    rho = np.asarray(rho, dtype=np.complex128).reshape(4, 4)
    # Partial transpose on second qubit (B) in computational basis
    rho_TB = rho.copy()
    # indices mapping: |i j><k l| -> |i l><k j|
    rho_TB = rho_TB.reshape(2, 2, 2, 2).swapaxes(1, 3).reshape(4, 4)
    evals = np.linalg.eigvalsh((rho_TB + rho_TB.conj().T) / 2.0)  # hermitize numerically
    neg = float(abs(np.sum([ev for ev in evals if ev < 0.0])))
    return neg


# ====== Canonical "paper-faithful" config ======
def paper_faithful_params() -> RunParams:
    """
    Parameter set consistent with the PRL proposal text (order-of-magnitude):
      m1 = m2 = 1e-14 kg
      d   = 450 μm  (centers)
      Δx  = 250 μm  (path separation)
      τ   = 2.5 s   (free evolution)
    This yields Δφ_RL > 0, Δφ_LR < 0; small but nonzero entanglement.
    """
    geom = Geometry(d=450e-6, dx=250e-6)
    mt   = MassesAndTime(m1=1e-14, m2=1e-14, tau=2.5)
    return RunParams(geom=geom, mt=mt, gamma_dephase=0.0)


# ====== 1-button PASS/FAIL ======
def pass_fail(result: Dict[str, Any], eps: float = 1e-9) -> Tuple[bool, str]:
    """
    By construction, if gravity-only phases entangle the spins (negativity > 0),
    we mark PASS. This is the model’s core claim.
    """
    neg = result["entanglement"]["negativity"]
    ok  = (neg is not None) and (neg > eps)
    return ok, "PASS" if ok else "FAIL"


# ====== Programmatic data generation hooks ======
def generate_time_trace(params: RunParams, t_list: np.ndarray) -> Dict[str, Any]:
    """
    Sweep hold time τ to generate a “live” dataset of entanglement vs time.
    Returns a dict with arrays for τ, negativity, concurrence (if pure).
    """
    negs, concs = [], []
    for t in t_list:
        r = run_once(RunParams(geom=params.geom,
                               mt=MassesAndTime(params.mt.m1, params.mt.m2, float(t)),
                               gamma_dephase=params.gamma_dephase))
        negs.append(r["entanglement"]["negativity"])
        concs.append(r["entanglement"]["concurrence"])
    return {
        "tau": t_list.tolist(),
        "negativity": negs,
        "concurrence": concs,
        "static": {"geom": asdict(params.geom),
                   "masses": {"m1": params.mt.m1, "m2": params.mt.m2},
                   "gamma_dephase": params.gamma_dephase}
    }


# ====== CLI ======
def main():
    ap = argparse.ArgumentParser(description="Model A — MV/Bose quantum-gravity witness")
    ap.add_argument("--mode", choices=["paper", "quick", "trace"], default="paper",
                    help="paper: canonical run; quick: same; trace: sweep τ for dataset")
    ap.add_argument("--out", type=str, default="",
                    help="optional JSON output path")
    ap.add_argument("--dephase", type=float, default=0.0,
                    help="local dephasing probability p in [0,1] applied after SG map")
    ap.add_argument("--tau", type=float, default=2.5,
                    help="override τ in seconds (paper mode)")
    ap.add_argument("--d", type=float, default=450e-6,
                    help="override d [m] (paper mode)")
    ap.add_argument("--dx", type=float, default=250e-6,
                    help="override Δx [m] (paper mode)")
    ap.add_argument("--m", type=float, default=1e-14,
                    help="override equal masses m1=m2 [kg] (paper mode)")
    ap.add_argument("--trace_tmin", type=float, default=0.1,
                    help="τ sweep start (s) in trace mode")
    ap.add_argument("--trace_tmax", type=float, default=3.0,
                    help="τ sweep end (s) in trace mode")
    ap.add_argument("--trace_n", type=int, default=25,
                    help="number of τ points in trace mode")
    args = ap.parse_args()

    if args.mode in ("paper", "quick"):
        rp = RunParams(
            geom=Geometry(d=args.d, dx=args.dx),
            mt=MassesAndTime(m1=args.m, m2=args.m, tau=args.tau),
            gamma_dephase=args.dephase
        )
        res = run_once(rp)
        ok, verdict = pass_fail(res)
        payload = {"mode": args.mode, "verdict": verdict, "result": res}
        print(json.dumps(payload, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(payload, f, indent=2)

    elif args.mode == "trace":
        rp = RunParams(
            geom=Geometry(d=args.d, dx=args.dx),
            mt=MassesAndTime(m1=args.m, m2=args.m, tau=args.tau),  # τ here is ignored per-point
            gamma_dephase=args.dephase
        )
        ts = np.linspace(args.trace_tmin, args.trace_tmax, args.trace_n)
        data = generate_time_trace(rp, ts)
        print(json.dumps({"mode": "trace", "data": data}, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"mode": "trace", "data": data}, f, indent=2)
    else:
        raise SystemExit("unknown mode")


if __name__ == "__main__":
    main()
