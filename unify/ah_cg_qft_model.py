#!/usr/bin/env python3
# ah_cg_qft_model.py
# Model B — Aziz–Howl: classical gravity (h_{μν} classical) + QFT matter → entanglement via virtual matter
# Single-file, importable module + CLI.
# CC0-1.0 — Public Domain

"""
MODEL SUMMARY
-------------
Implements the Aziz–Howl (Nature 646, 813–817, 2025) claim:

  • Gravity is classical: interaction H_int = -(1/2) ∫ h_{μν}(x) T̂^{μν}(x) d^3x (local QFT in curved spacetime)
  • Matter is quantum: the *fourth-order* process with *virtual matter propagators* produces branch-dependent
    amplitudes ϑ_{ij}, which (generically) differ across the four path-branches {LL, LR, RL, RR}.
  • In the Δx ≫ d_RL configuration, the RL amplitude dominates: α_RL ≈ 1 + i ϑ  (small-ϑ limit), others ≈ 1.
  • This asymmetry alone is sufficient to make the *two-branch* register entangled (even with classical h_{μν}).

IMPLEMENTATION CHOICES
----------------------
1) We model the pre-recombination *branch register* as a 2-qubit pure state in the basis
   {|LL>, |LR>, |RL>, |RR>} with amplitudes α_{ij}. In the small-parameter regime:
       α_LL = 1,  α_LR = 1,  α_RR = 1,  α_RL = exp(i * ϑ)  ≈ 1 + iϑ.
   The normalized state is v = (1, 1, e^{iϑ}, 1)/2. Entanglement is then read via negativity.

2) We compute ϑ using the *closed-form* scaling from Aziz–Howl (their Eq. 10), which (under the stated approximations)
   reads:  ϑ ≈ (6 / 25) * (G * m * M^2 * R^2 * t) / (ħ * d_RL^3).
   Here M is the total mass per object (assumed equal), m is the particle mass used in the QFT description,
   R is sphere radius, t is interaction time, and d_RL is the closest-approach separation for the {R,L} branch.

3) For convenience we also compute the quantum-gravity *second-order* phase ϕ_RL = (G M^2 t)/(ħ d_RL)
   so you can form the comparative ratio ϑ/ϕ (cf. their Fig. 4).

NOTE
----
This is a *faithful fast model* for regime exploration and PASS/FAIL of the Aziz–Howl claim that
"CG + QFT matter" can produce entanglement via local processes. It purposefully avoids any hidden
direct n1*n2 terms (the locality pitfall criticized by Marletto–Vedral), as our state construction
follows from branch amplitudes only.

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import numpy as np

# ====== Physical constants (SI) ======
G    = 6.67430e-11           # [m^3 kg^-1 s^-2]
HBAR = 1.054_571_817e-34     # [J s] = [kg m^2 s^-1]
PI   = math.pi


# ====== Dataclasses ======
@dataclass
class GeometryCG:
    """Minimal geometry for the Δx ≫ d_RL branch dominance regime."""
    d_RL: float      # closest-approach distance between the two objects in the RL branch [m]
    R: float         # sphere radius [m]


@dataclass
class MassTimeCG:
    M: float         # total mass per object [kg]
    m_particle: float  # single-particle mass used in QFT matter field (e.g., atomic mass) [kg]
    t: float         # interaction time [s]


@dataclass
class ThetaPhi:
    theta_RL: float   # ϑ_RL (dimensionless)
    phi_RL: float     # ϕ_RL (dimensionless)
    ratio: float      # ϑ_RL / ϕ_RL


# ====== Helper: radius from density (optional) ======
def radius_from_mass_density(M: float, rho: float) -> float:
    """R = (3M/(4πρ))^(1/3)."""
    return ((3.0 * M) / (4.0 * PI * rho)) ** (1.0 / 3.0)


# ====== Core Aziz–Howl scalings ======
def theta_RL_closed_form(geom: GeometryCG, mt: MassTimeCG) -> float:
    """
    ϑ_RL ≈ (6/25) * (G * m * M^2 * R^2 * t) / (ħ * d_RL^3)
    (Aziz–Howl Eq. 10, small-parameter closed form under their ‘mean-field’ sourcing)
    """
    num = (6.0 / 25.0) * G * mt.m_particle * (mt.M ** 2) * (geom.R ** 2) * mt.t
    den = HBAR * (geom.d_RL ** 3)
    return float(num / den)


def phi_RL_qg(geom: GeometryCG, mt: MassTimeCG) -> float:
    """
    Quantum-gravity baseline phase (second order):
      ϕ_RL = (G M^2 t) / (ħ d_RL)
    """
    return float(G * (mt.M ** 2) * mt.t / (HBAR * geom.d_RL))


def compute_theta_phi(geom: GeometryCG, mt: MassTimeCG) -> ThetaPhi:
    th = theta_RL_closed_form(geom, mt)
    ph = phi_RL_qg(geom, mt)
    ratio = (th / ph) if ph != 0.0 else float("inf")
    return ThetaPhi(theta_RL=th, phi_RL=ph, ratio=ratio)


# ====== State construction & entanglement ======
def branch_state_from_theta(theta_RL: float) -> np.ndarray:
    """
    Build the reduced two-branch pure state in {|LL>,|LR>,|RL>,|RR>}:
      v = (1, 1, e^{i θ}, 1) / 2
    """
    v = np.array([1.0, 1.0, np.exp(1j * float(theta_RL)), 1.0], dtype=np.complex128) / 2.0
    return v  # already normalized


def concurrence_pure(v: np.ndarray) -> float:
    """Wootters concurrence for a pure two-qubit state vector v."""
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    psi = v.reshape(4, 1)
    C = abs((psi.T @ (np.kron(sy, sy)) @ psi.conj()).item())
    return float(np.real_if_close(C))


def negativity_pure(v: np.ndarray) -> float:
    """Negativity for pure two-qubit states: N = C/2."""
    return 0.5 * concurrence_pure(v)


def run_once(geom: GeometryCG, mt: MassTimeCG) -> Dict[str, Any]:
    """
    One-shot Aziz–Howl experiment surrogate:
      1) compute (θ_RL, ϕ_RL, ratio)
      2) build v(θ_RL) and evaluate entanglement
    """
    thph = compute_theta_phi(geom, mt)
    v = branch_state_from_theta(thph.theta_RL)
    C = concurrence_pure(v)
    N = 0.5 * C
    return {
        "input": {"geom": asdict(geom), "masses_time": asdict(mt)},
        "theta_phi": asdict(thph),
        "entanglement": {"concurrence": C, "negativity": N}
    }


# ====== Programmatic generators ======
def generate_time_trace(geom: GeometryCG, mt: MassTimeCG, t_list: np.ndarray) -> Dict[str, Any]:
    negs, concs, thetas, phis, ratios = [], [], [], [], []
    for t in t_list:
        thph = compute_theta_phi(geom, MassTimeCG(M=mt.M, m_particle=mt.m_particle, t=float(t)))
        v = branch_state_from_theta(thph.theta_RL)
        c = concurrence_pure(v); n = 0.5 * c
        negs.append(n); concs.append(c); thetas.append(thph.theta_RL); phis.append(thph.phi_RL); ratios.append(thph.ratio)
    return {
        "tau": t_list.tolist(),
        "negativity": negs,
        "concurrence": concs,
        "theta_RL": thetas,
        "phi_RL": phis,
        "theta_over_phi": ratios,
        "static": {"geom": asdict(geom), "masses": {"M": mt.M, "m_particle": mt.m_particle}}
    }


def generate_mass_time_grid(geom: GeometryCG, M_vals: np.ndarray, t_vals: np.ndarray, m_particle: float) -> Dict[str, Any]:
    """
    For regime maps: compute θ/ϕ over a grid of (M, t) with fixed geometry and particle mass.
    """
    grid = []
    for M in M_vals:
        row = []
        for t in t_vals:
            thph = compute_theta_phi(geom, MassTimeCG(M=float(M), m_particle=m_particle, t=float(t)))
            row.append(thph.ratio)
        grid.append(row)
    return {
        "M": M_vals.tolist(),
        "t": t_vals.tolist(),
        "theta_over_phi": grid,
        "geom": asdict(geom),
        "m_particle": m_particle
    }


# ====== PASS/FAIL ======
def pass_fail(result: Dict[str, Any], eps: float = 1e-12) -> Tuple[bool, str]:
    """
    PASS if (negativity > 0), i.e., θ_RL not an integer multiple of 2π.
    Using a small epsilon because θ can be tiny for small masses/times.
    """
    neg = result["entanglement"]["negativity"]
    ok = (neg is not None) and (neg > eps)
    return ok, ("PASS" if ok else "FAIL")


# ====== Canonical "paper-faithful" defaults ======
def paper_faithful_defaults() -> Tuple[GeometryCG, MassTimeCG]:
    """
    Aziz–Howl 'paper-faithful' representative point consistent with their Fig. 4
    (classical h_{μν}, QFT matter; minimum separation d_RL=10 R; Yb-like particle mass).

    Choice: M = 15 kg, t = 1 s, ρ = 6900 kg/m^3 (sets R), d_RL = 10 R, m_particle = 2.87e-25 kg.
    This lands to the right of the 'significant classical effect' threshold (θ ≳ 0.1),
    while preserving the exact geometry d_RL = 10 R used in the paper.
    """
    M = 15.0                     # kg
    rho = 6900.0                 # kg/m^3 (metal-like; only used to set R and d_RL = 10R)
    R = radius_from_mass_density(M, rho)
    d_RL = 10.0 * R              # paper’s min-separation convention
    m_particle = 2.87e-25        # kg (Yb-173 mass scale)
    t = 1.0                      # s
    return GeometryCG(d_RL=d_RL, R=R), MassTimeCG(M=M, m_particle=m_particle, t=t)


# ====== CLI ======
def main():
    ap = argparse.ArgumentParser(description="Model B — Aziz–Howl classical gravity + QFT matter")
    ap.add_argument("--mode", choices=["paper", "quick", "trace", "grid"], default="paper",
                    help="paper: canonical; quick: custom once; trace: sweep t; grid: θ/ϕ over (M,t)")
    # paper/quick overrides
    ap.add_argument("--M", type=float, default=None, help="mass per object [kg]")
    ap.add_argument("--rho", type=float, default=None, help="density [kg/m^3] to derive R (if R not given)")
    ap.add_argument("--R", type=float, default=None, help="radius [m] (override). If not set, computed from (M, rho).")
    ap.add_argument("--sep_factor", type=float, default=10.0, help="d_RL = sep_factor * R")
    ap.add_argument("--m_particle", type=float, default=2.87e-25, help="particle mass [kg] (e.g., Yb-173)")
    ap.add_argument("--t", type=float, default=None, help="interaction time [s]")
    ap.add_argument("--out", type=str, default="", help="optional JSON output path")

    # trace options
    ap.add_argument("--trace_tmin", type=float, default=1e-6)
    ap.add_argument("--trace_tmax", type=float, default=1e-2)
    ap.add_argument("--trace_n", type=int, default=40)

    # grid options
    ap.add_argument("--grid_Mmin", type=float, default=1e-12)
    ap.add_argument("--grid_Mmax", type=float, default=1e-6)
    ap.add_argument("--grid_Mn", type=int, default=25)
    ap.add_argument("--grid_tmin", type=float, default=1e-6)
    ap.add_argument("--grid_tmax", type=float, default=1e-2)
    ap.add_argument("--grid_tn", type=int, default=25)

    args = ap.parse_args()

    if args.mode in ("paper", "quick"):
        if args.mode == "paper":
            geom, mt = paper_faithful_defaults()
        else:
            # quick: construct from provided overrides, else fall back to paper defaults for missing pieces
            geom0, mt0 = paper_faithful_defaults()
            M = args.M if args.M is not None else mt0.M
            t = args.t if args.t is not None else mt0.t
            R = args.R if args.R is not None else radius_from_mass_density(M, args.rho if args.rho else 6900.0)
            d_RL = args.sep_factor * R
            geom = GeometryCG(d_RL=d_RL, R=R)
            mt = MassTimeCG(M=M, m_particle=args.m_particle, t=t)

        res = run_once(geom, mt)
        ok, verdict = pass_fail(res)
        payload = {"mode": args.mode, "verdict": verdict, "result": res}
        print(json.dumps(payload, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(payload, f, indent=2)

    elif args.mode == "trace":
        # trace over time with geometry fixed
        geom, mt0 = paper_faithful_defaults()
        if args.M is not None or args.R is not None or args.rho is not None:
            M = args.M if args.M is not None else mt0.M
            R = args.R if args.R is not None else radius_from_mass_density(M, args.rho if args.rho else 6900.0)
            geom = GeometryCG(d_RL=args.sep_factor * R, R=R)
            mt0 = MassTimeCG(M=M, m_particle=args.m_particle, t=mt0.t)

        ts = np.linspace(args.trace_tmin, args.trace_tmax, args.trace_n)
        data = generate_time_trace(geom, mt0, ts)
        print(json.dumps({"mode": "trace", "data": data}, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"mode": "trace", "data": data}, f, indent=2)

    elif args.mode == "grid":
        # θ/ϕ map over (M,t)
        geom, mt0 = paper_faithful_defaults()
        if args.R is not None or args.rho is not None or args.M is not None:
            M = args.M if args.M is not None else mt0.M
            R = args.R if args.R is not None else radius_from_mass_density(M, args.rho if args.rho else 6900.0)
            geom = GeometryCG(d_RL=args.sep_factor * R, R=R)

        M_vals = np.geomspace(args.grid_Mmin, args.grid_Mmax, args.grid_Mn)
        t_vals = np.geomspace(args.grid_tmin, args.grid_tmax, args.grid_tn)
        grid = generate_mass_time_grid(geom, M_vals, t_vals, m_particle=args.m_particle)
        print(json.dumps({"mode": "grid", "grid": grid}, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"mode": "grid", "grid": grid}, f, indent=2)
    else:
        raise SystemExit("unknown mode")


if __name__ == "__main__":
    main()
