#!/usr/bin/env python
"""Direct unified finite-clock dynamic-insertion pipeline.

This is the single-script implementation corresponding to `main.tex`.  The
probe scripts remain useful tests and historical evidence, but this file does
not call them.  It directly performs the ordered pipeline:

1. Generate explicit base artifacts:
   - small-N diagonal substrate and rule diagnostics
   - large projection-base states that will be projected
2. Run paired constrained/selfish CUDA projection over a T grid.
3. Save raw K1 local-witness checkpoints.
4. Select T_ins with the predeclared scout selector.
5. Build coherence-thinned causal delay taps, anchors, and freeze Pi.
6. Apply K2 frozen insertion on device.
7. Evolve live/post-hoc/null arms from the selected base state.
8. Optionally compare the insertable observable to author data.

The output is one JSON evidence ledger plus stage CSV/NPZ artifacts under
`analysis/unified_main_pipeline/`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np


PROBE = "unified_main_pipeline"
VERSION = "2.8.0"
TWO_PI = 2.0 * math.pi
CAUSAL_INSERTION_DECAY_VERSION = "causal_insertion_decay_v7"
DEFAULT_LAG_DECAY_STRENGTH = 0.825
DEFAULT_THERMAL_GUIDE_STRENGTH = 0.925
CAUSAL_INSERTION_DECAY_FORMULA = (
    "fit log(coherence_lag)=a-rate*lag across causal taps; "
    "rate_excess=max(0, rate(T)-min_T rate(T)); "
    "lag_slope_decay=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints); "
    "thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max), with T_guide=min(T_ins, simulated chi shoulder); "
    "thermal_envelope_factor=exp(-lag_strength*rate_excess*insertion_horizon_checkpoints*(thermal_gain(T)-1)); "
    "D_insert(T)=min(1, lag_slope_decay*thermal_envelope_factor); "
    "auto horizon=max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps; "
    "Psi_Y still uses coherence-thinned taps for K2 freeze; "
    "finite-clock insertion_threshold_survival is recorded as a K2 readiness diagnostic, not multiplied into the observable; "
    "c_mean_m_insertable=c_mean_m_raw*D_insert(T); "
    "lag_strength and default kappa are frozen global finite-horizon calibration constants"
)

KERNEL_NAMES = [
    "init_rng_states_kernel",
    "run_finite_clock_dual_witness_pair_kernel",
    "compute_local_witness_kernel",
    "run_finite_clock_projector_kernel",
    "apply_frozen_insertion_kernel",
]


# ---------------------------------------------------------------------------
# Generic IO
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, compress: bool, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def parse_csv_floats(text: str) -> list[float]:
    vals = [float(p.strip()) for p in text.split(",") if p.strip()]
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def parse_csv_ints(text: str) -> list[int]:
    vals = [int(p.strip()) for p in text.split(",") if p.strip()]
    if not vals:
        raise ValueError("Expected at least one int.")
    return vals


def resolve_kernel_path(arg: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path(arg),
        Path.cwd() / arg,
        here / arg,
        here / "kernel" / "kernel_finite_clock.cu",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(arg).resolve()


# ---------------------------------------------------------------------------
# Geometry and CPU reference substrate
# ---------------------------------------------------------------------------

def site_index(x: int, y: int, L: int) -> int:
    return (x % L) + L * (y % L)


def build_neighbor_arrays(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour order: +x, -x, +y, -y, matching the CUDA LUT dir table."""
    N = L * L
    neighbor_idx = np.empty((N, 4), dtype=np.int32)
    neighbor_psi = np.empty((N, 4), dtype=np.float64)
    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L)
            neighbor_idx[i, 0] = site_index(x + 1, y, L)
            neighbor_psi[i, 0] = 0.0
            neighbor_idx[i, 1] = site_index(x - 1, y, L)
            neighbor_psi[i, 1] = math.pi
            neighbor_idx[i, 2] = site_index(x, y + 1, L)
            neighbor_psi[i, 2] = 0.5 * math.pi
            neighbor_idx[i, 3] = site_index(x, y - 1, L)
            neighbor_psi[i, 3] = -0.5 * math.pi
    return (
        np.ascontiguousarray(neighbor_idx.reshape(-1), dtype=np.int32),
        np.ascontiguousarray(neighbor_psi.reshape(-1), dtype=np.float64),
    )


def build_bond_arrays(L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bond_i: list[int] = []
    bond_j: list[int] = []
    psi_ij: list[float] = []
    psi_ji: list[float] = []
    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L)
            j = site_index(x + 1, y, L)
            bond_i.append(i)
            bond_j.append(j)
            psi_ij.append(0.0)
            psi_ji.append(math.pi)
            j = site_index(x, y + 1, L)
            bond_i.append(i)
            bond_j.append(j)
            psi_ij.append(0.5 * math.pi)
            psi_ji.append(-0.5 * math.pi)
    return (
        np.ascontiguousarray(bond_i, dtype=np.int32),
        np.ascontiguousarray(bond_j, dtype=np.int32),
        np.ascontiguousarray(psi_ij, dtype=np.float64),
        np.ascontiguousarray(psi_ji, dtype=np.float64),
    )


def theta_k(k: int | np.ndarray, q: int) -> float | np.ndarray:
    return TWO_PI * np.asarray(k, dtype=float) / float(q)


def circ_dist(a: float, b: float) -> float:
    return abs(((a - b + math.pi) % TWO_PI) - math.pi)


def vision_J(theta: float, psi: float, Psi: float, J: float) -> float:
    return J if circ_dist(theta, psi) <= Psi else 0.0


def HSS_cpu(state: np.ndarray, L: int, q: int, Psi: float, J: float) -> float:
    H = 0.0
    for y in range(L):
        for x in range(L):
            i = site_index(x, y, L)
            theta_i = float(theta_k(int(state[i]), q))
            for dx, dy, psi_ij, psi_ji in ((1, 0, 0.0, math.pi), (0, 1, 0.5 * math.pi, -0.5 * math.pi)):
                j = site_index(x + dx, y + dy, L)
                theta_j = float(theta_k(int(state[j]), q))
                Jij = vision_J(theta_i, psi_ij, Psi, J)
                Jji = vision_J(theta_j, psi_ji, Psi, J)
                H += -(Jij + Jji) * math.cos(theta_i - theta_j)
    return float(H)


def selfish_site_energy_cpu(state: np.ndarray, i: int, L: int, q: int, Psi: float, J: float, *, k_override: int | None = None) -> float:
    k_i = int(state[i]) if k_override is None else int(k_override)
    theta_i = float(theta_k(k_i, q))
    x = i % L
    y = i // L
    e = 0.0
    for dx, dy, psi in ((1, 0, 0.0), (-1, 0, math.pi), (0, 1, 0.5 * math.pi), (0, -1, -0.5 * math.pi)):
        j = site_index(x + dx, y + dy, L)
        theta_j = float(theta_k(int(state[j]), q))
        e += -vision_J(theta_i, psi, Psi, J) * math.cos(theta_i - theta_j)
    return float(e)


def total_selfish_energy_cpu(state: np.ndarray, L: int, q: int, Psi: float, J: float) -> float:
    return float(sum(selfish_site_energy_cpu(state, i, L, q, Psi, J) for i in range(L * L)))


def delta_E_constrained_cpu(state: np.ndarray, i: int, new_k: int, L: int, q: int, Psi: float, J: float) -> float:
    old_k = int(state[i])
    theta_old = float(theta_k(old_k, q))
    theta_new = float(theta_k(new_k, q))
    x = i % L
    y = i // L
    acc = 0.0
    for dx, dy, psi in ((1, 0, 0.0), (-1, 0, math.pi), (0, 1, 0.5 * math.pi), (0, -1, -0.5 * math.pi)):
        j = site_index(x + dx, y + dy, L)
        theta_j = float(theta_k(int(state[j]), q))
        Jnew = vision_J(theta_new, psi, Psi, J)
        Jold = vision_J(theta_old, psi, Psi, J)
        acc += (Jnew + Jold) * (math.cos(theta_new - theta_j) - math.cos(theta_old - theta_j))
    return float(-0.5 * acc)


def delta_E_selfish_cpu(state: np.ndarray, i: int, new_k: int, L: int, q: int, Psi: float, J: float) -> float:
    return float(
        selfish_site_energy_cpu(state, i, L, q, Psi, J, k_override=new_k) -
        selfish_site_energy_cpu(state, i, L, q, Psi, J)
    )


def magnetization_abs(states: np.ndarray, q: int) -> np.ndarray:
    angles = TWO_PI * states.astype(float) / float(q)
    z_re = np.mean(np.cos(angles), axis=1)
    z_im = np.mean(np.sin(angles), axis=1)
    return np.sqrt(z_re * z_re + z_im * z_im)


def generate_small_base_artifact(out_dir: Path, *, q: int, Psi: float, J: float, compress: bool) -> tuple[Path, dict]:
    L = 2
    N = L * L
    states = np.array(list(itertools.product(range(q), repeat=N)), dtype=np.int16)
    Hc = np.empty(states.shape[0], dtype=np.float64)
    Hs = np.empty(states.shape[0], dtype=np.float64)
    for idx, state in enumerate(states):
        Hc[idx] = HSS_cpu(state, L, q, Psi, J)
        Hs[idx] = total_selfish_energy_cpu(state, L, q, Psi, J)

    max_selfish_gap_mismatch = 0.0
    max_constrained_gap_mismatch = 0.0
    rule_distinct = False
    state_to_index = {tuple(map(int, row)): i for i, row in enumerate(states)}
    for s_idx, state in enumerate(states):
        for site in range(N):
            old = int(state[site])
            for new_k in range(q):
                if new_k == old:
                    continue
                flipped = state.copy()
                flipped[site] = new_k
                f_idx = state_to_index[tuple(map(int, flipped))]
                d_c = delta_E_constrained_cpu(state, site, new_k, L, q, Psi, J)
                d_s = delta_E_selfish_cpu(state, site, new_k, L, q, Psi, J)
                max_constrained_gap_mismatch = max(max_constrained_gap_mismatch, abs((Hc[f_idx] - Hc[s_idx]) - d_c))
                max_selfish_gap_mismatch = max(max_selfish_gap_mismatch, abs((Hs[f_idx] - Hs[s_idx]) - d_s))
                rule_distinct = rule_distinct or abs(d_c - d_s) > 1.0e-12

    path = out_dir / "base_smallN_L2_q8.npz"
    save_npz(
        path,
        compress,
        schema=np.array("unified_smallN_base_v1"),
        states=states,
        H_constrained_diag=Hc,
        H_selfish_diag=Hs,
        L=np.array(L, dtype=np.int32),
        q=np.array(q, dtype=np.int32),
        Psi=np.array(Psi, dtype=np.float64),
        J=np.array(J, dtype=np.float64),
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "L": L,
        "q": q,
        "basis_size": int(states.shape[0]),
        "H_constrained_not_global_gap_max_abs": float(max_constrained_gap_mismatch),
        "H_selfish_not_global_gap_max_abs": float(max_selfish_gap_mismatch),
        "rules_distinct": bool(rule_distinct),
        "interpretation": (
            "The NPZ freezes the small-N diagonal substrate.  Gap mismatches are recorded because the "
            "vision-cone local update rules are non-conservative; P0/probe parity is the oracle check."
        ),
    }
    return path, report


def generate_projection_base(out_dir: Path, *, L: int, q: int, n_traj: int, init: str, k0: int, seed: int, Psi: float, J: float, proposal_id: int, proposal_delta_rad: float, window_radius: int, compress: bool) -> tuple[Path, dict]:
    N = L * L
    rng = np.random.default_rng(seed)
    if init == "ordered":
        state = np.full((n_traj, N), int(k0), dtype=np.int32)
    elif init == "random":
        state = rng.integers(0, q, size=(n_traj, N), dtype=np.int32)
    else:
        raise ValueError(f"unknown init {init!r}")
    state_c = state.copy()
    state_s = state.copy()
    neighbor_idx, neighbor_psi = build_neighbor_arrays(L)
    bond_i, bond_j, bond_psi_ij, bond_psi_ji = build_bond_arrays(L)
    path = out_dir / "projection_base.npz"
    save_npz(
        path,
        compress,
        schema=np.array("unified_projection_base_v1"),
        state_constrained=state_c,
        state_selfish=state_s,
        neighbor_idx=neighbor_idx,
        neighbor_psi=neighbor_psi,
        bond_i=bond_i,
        bond_j=bond_j,
        bond_psi_ij=bond_psi_ij,
        bond_psi_ji=bond_psi_ji,
        L=np.array(L, dtype=np.int32),
        N=np.array(N, dtype=np.int32),
        q=np.array(q, dtype=np.int32),
        Psi=np.array(Psi, dtype=np.float64),
        J=np.array(J, dtype=np.float64),
        proposal_id=np.array(proposal_id, dtype=np.int32),
        proposal_delta_rad=np.array(proposal_delta_rad, dtype=np.float64),
        window_radius=np.array(window_radius, dtype=np.int32),
        seed=np.array(seed, dtype=np.uint64),
    )
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": "unified_projection_base_v1",
        "L": L,
        "N": N,
        "q": q,
        "n_traj": n_traj,
        "init": init,
        "seed": seed,
    }
    return path, report


# ---------------------------------------------------------------------------
# CUDA load and launch helpers
# ---------------------------------------------------------------------------

def load_kernel(kernel_path: Path):
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError("CuPy is required for the unified CUDA pipeline.") from exc

    code = kernel_path.read_text(encoding="utf-8")
    errors: list[str] = []
    for backend in ("nvrtc", "nvcc"):
        try:
            if backend == "nvrtc":
                mod = cp.RawModule(code=code, backend="nvrtc", options=("--std=c++17",), name_expressions=KERNEL_NAMES)
            else:
                mod = cp.RawModule(code=code, backend="nvcc", options=("--std=c++17",))
            fns = {name: mod.get_function(name) for name in KERNEL_NAMES}
            return cp, mod, fns, backend
        except Exception as exc:
            errors.append(f"[{backend}] {type(exc).__name__}: {exc}")
    raise RuntimeError("CUDA compile failed:\n" + "\n".join(errors))


def _copy_to_const(cp, mod, name: str, arr: np.ndarray) -> None:
    global_obj = mod.get_global(name)
    if isinstance(global_obj, tuple):
        ptr, nbytes = global_obj
    else:
        ptr, nbytes = global_obj, None
    arr = np.ascontiguousarray(arr)
    if nbytes is not None and arr.nbytes > nbytes:
        raise ValueError(f"{name} needs {arr.nbytes} bytes, but symbol has {nbytes}")
    d_arr = cp.asarray(arr)
    dst_ptr = ptr.ptr if hasattr(ptr, "ptr") else ptr
    cp.cuda.runtime.memcpy(dst_ptr, d_arr.data.ptr, arr.nbytes, cp.cuda.runtime.memcpyDeviceToDevice)


def upload_clock_luts(cp, mod, q: int, Psi: float, *, max_q: int = 64) -> bool:
    if q <= 0 or q > max_q:
        _copy_to_const(cp, mod, "CLOCK_LUT_Q", np.array([-1], dtype=np.int32))
        return False
    angles = (TWO_PI / q) * np.arange(q, dtype=np.float64)
    cos_k = np.zeros(max_q, dtype=np.float64)
    sin_k = np.zeros(max_q, dtype=np.float64)
    cos_k[:q] = np.cos(angles)
    sin_k[:q] = np.sin(angles)
    cos_delta = np.zeros(max_q * max_q, dtype=np.float64)
    for a in range(q):
        for b in range(q):
            cos_delta[a * max_q + b] = math.cos(float(angles[a] - angles[b]))
    dirs = np.array([0.0, math.pi, 0.5 * math.pi, -0.5 * math.pi], dtype=np.float64)
    vision = np.zeros(max_q * 4, dtype=np.uint8)
    for k, theta in enumerate(angles):
        for d, psi in enumerate(dirs):
            dist = (float(theta) - float(psi) + math.pi) % TWO_PI - math.pi
            vision[k * 4 + d] = 1 if abs(dist) <= Psi else 0
    _copy_to_const(cp, mod, "CLOCK_LUT_Q", np.array([q], dtype=np.int32))
    _copy_to_const(cp, mod, "CLOCK_COS_K", cos_k)
    _copy_to_const(cp, mod, "CLOCK_SIN_K", sin_k)
    _copy_to_const(cp, mod, "CLOCK_COS_DELTA", cos_delta)
    _copy_to_const(cp, mod, "CLOCK_VISION4", vision)
    return True


def init_rng(cp, fns, n_traj: int, seed: int, threads_per_block: int):
    rng_states = cp.empty(n_traj, dtype=cp.uint64)
    block = int(threads_per_block)
    grid = ((n_traj + block - 1) // block,)
    fns["init_rng_states_kernel"](grid, (block,), (rng_states, np.int32(n_traj), np.uint64(seed)))
    cp.cuda.Stream.null.synchronize()
    return rng_states


def int32_kernel_step_offset(global_step_offset: int, sample_stride: int) -> np.int32:
    if sample_stride > 0:
        return np.int32(int(global_step_offset) % int(sample_stride))
    if not -(2**31) <= int(global_step_offset) < 2**31:
        return np.int32(0)
    return np.int32(global_step_offset)


def launch_dual_pair_phase(
    cp,
    fns,
    state_c,
    state_s,
    rng_states,
    *,
    L: int,
    n_traj: int,
    q: int,
    n_steps: int,
    sample_stride: int,
    global_step_offset: int,
    T: float,
    Psi: float,
    J: float,
    proposal_id: int,
    window_radius: int,
    proposal_delta_rad: float,
    d_neighbor_idx,
    d_neighbor_psi,
    d_bond_i,
    d_bond_j,
    d_bond_psi_ij,
    d_bond_psi_ji,
    n_bonds: int,
    threads_per_block: int,
) -> dict:
    block = int(threads_per_block)
    grid = ((n_traj + block - 1) // block,)

    def zfloat():
        return cp.zeros(n_traj, dtype=cp.float64)

    def zu64():
        return cp.zeros(n_traj, dtype=cp.uint64)

    outs_float = [zfloat() for _ in range(26)]
    out_c_sum_m, out_c_sum_m2, out_c_sum_H, out_c_sum_H2 = outs_float[:4]
    out_s_sum_m, out_s_sum_m2, out_s_sum_H, out_s_sum_H2 = outs_float[4:8]
    (
        out_sum_zc_re, out_sum_zc_im, out_sum_zs_re, out_sum_zs_im,
        out_sum_psi_re, out_sum_psi_im, out_sum_psi_abs, out_sum_psi_abs2,
        out_sum_branch_dot, out_sum_branch_cross,
        out_sum_gauge_zs_re, out_sum_gauge_zs_im,
        out_sum_gauge_psi_re, out_sum_gauge_psi_im,
        out_sum_gauge_psi_abs, out_sum_gauge_psi_abs2,
        out_sum_phase_delta, out_sum_abs_phase_delta,
    ) = outs_float[8:]
    out_sum_phase_delta2 = zfloat()
    out_c_accept, out_c_propose, out_s_accept, out_s_propose, out_samples = [zu64() for _ in range(5)]

    fns["run_finite_clock_dual_witness_pair_kernel"](
        grid,
        (block,),
        (
            state_c, state_s, rng_states,
            np.int32(L), np.int32(n_traj), np.int32(q), np.int32(n_steps),
            np.int32(sample_stride), int32_kernel_step_offset(global_step_offset, sample_stride),
            np.float64(T), np.float64(Psi), np.float64(J),
            np.int32(proposal_id), np.int32(window_radius), np.float64(proposal_delta_rad),
            d_neighbor_idx, d_neighbor_psi,
            d_bond_i, d_bond_j, d_bond_psi_ij, d_bond_psi_ji, np.int32(n_bonds),
            out_c_sum_m, out_c_sum_m2, out_c_sum_H, out_c_sum_H2, out_c_accept, out_c_propose,
            out_s_sum_m, out_s_sum_m2, out_s_sum_H, out_s_sum_H2, out_s_accept, out_s_propose,
            out_samples,
            out_sum_zc_re, out_sum_zc_im, out_sum_zs_re, out_sum_zs_im,
            out_sum_psi_re, out_sum_psi_im, out_sum_psi_abs, out_sum_psi_abs2,
            out_sum_branch_dot, out_sum_branch_cross,
            out_sum_gauge_zs_re, out_sum_gauge_zs_im,
            out_sum_gauge_psi_re, out_sum_gauge_psi_im,
            out_sum_gauge_psi_abs, out_sum_gauge_psi_abs2,
            out_sum_phase_delta, out_sum_abs_phase_delta, out_sum_phase_delta2,
        ),
    )
    cp.cuda.Stream.null.synchronize()

    def fsum(x):
        return float(cp.asnumpy(x).sum())

    def isum(x):
        return int(cp.asnumpy(x).sum())

    return {
        "c_sum_m": fsum(out_c_sum_m), "c_sum_m2": fsum(out_c_sum_m2), "c_sum_H": fsum(out_c_sum_H), "c_sum_H2": fsum(out_c_sum_H2),
        "c_accept": isum(out_c_accept), "c_propose": isum(out_c_propose),
        "s_sum_m": fsum(out_s_sum_m), "s_sum_m2": fsum(out_s_sum_m2), "s_sum_H": fsum(out_s_sum_H), "s_sum_H2": fsum(out_s_sum_H2),
        "s_accept": isum(out_s_accept), "s_propose": isum(out_s_propose),
        "samples": isum(out_samples),
        "sum_zc_re": fsum(out_sum_zc_re), "sum_zc_im": fsum(out_sum_zc_im),
        "sum_zs_re": fsum(out_sum_zs_re), "sum_zs_im": fsum(out_sum_zs_im),
        "sum_psi_re": fsum(out_sum_psi_re), "sum_psi_im": fsum(out_sum_psi_im),
        "sum_psi_abs": fsum(out_sum_psi_abs), "sum_psi_abs2": fsum(out_sum_psi_abs2),
        "sum_branch_dot": fsum(out_sum_branch_dot), "sum_branch_cross": fsum(out_sum_branch_cross),
        "sum_gauge_zs_re": fsum(out_sum_gauge_zs_re), "sum_gauge_zs_im": fsum(out_sum_gauge_zs_im),
        "sum_gauge_psi_re": fsum(out_sum_gauge_psi_re), "sum_gauge_psi_im": fsum(out_sum_gauge_psi_im),
        "sum_gauge_psi_abs": fsum(out_sum_gauge_psi_abs), "sum_gauge_psi_abs2": fsum(out_sum_gauge_psi_abs2),
        "sum_phase_delta": fsum(out_sum_phase_delta), "sum_abs_phase_delta": fsum(out_sum_abs_phase_delta),
        "sum_phase_delta2": fsum(out_sum_phase_delta2),
    }


def launch_projector(
    cp,
    fns,
    state,
    rng_states,
    *,
    L: int,
    n_traj: int,
    q: int,
    n_steps: int,
    sample_stride: int,
    global_step_offset: int,
    T: float,
    Psi: float,
    J: float,
    method_id: int,
    proposal_id: int,
    window_radius: int,
    proposal_delta_rad: float,
    d_neighbor_idx,
    d_neighbor_psi,
    d_bond_i,
    d_bond_j,
    d_bond_psi_ij,
    d_bond_psi_ji,
    n_bonds: int,
    threads_per_block: int,
) -> dict:
    block = int(threads_per_block)
    grid = ((n_traj + block - 1) // block,)

    def zfloat():
        return cp.zeros(n_traj, dtype=cp.float64)

    def zu64():
        return cp.zeros(n_traj, dtype=cp.uint64)

    out_sum_m = zfloat()
    out_sum_m2 = zfloat()
    out_sum_H = zfloat()
    out_sum_H2 = zfloat()
    out_accept = zu64()
    out_propose = zu64()
    out_samples = zu64()
    fns["run_finite_clock_projector_kernel"](
        grid,
        (block,),
        (
            state, rng_states,
            np.int32(L), np.int32(n_traj), np.int32(q), np.int32(n_steps),
            np.int32(sample_stride), int32_kernel_step_offset(global_step_offset, sample_stride),
            np.float64(T), np.float64(Psi), np.float64(J),
            np.int32(method_id), np.int32(proposal_id), np.int32(window_radius), np.float64(proposal_delta_rad),
            d_neighbor_idx, d_neighbor_psi,
            d_bond_i, d_bond_j, d_bond_psi_ij, d_bond_psi_ji, np.int32(n_bonds),
            out_sum_m, out_sum_m2, out_sum_H, out_sum_H2, out_accept, out_propose, out_samples,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    samples = int(cp.asnumpy(out_samples).sum())
    sum_m = float(cp.asnumpy(out_sum_m).sum())
    sum_m2 = float(cp.asnumpy(out_sum_m2).sum())
    sum_H = float(cp.asnumpy(out_sum_H).sum())
    sum_H2 = float(cp.asnumpy(out_sum_H2).sum())
    accept = int(cp.asnumpy(out_accept).sum())
    propose = int(cp.asnumpy(out_propose).sum())
    mean_m = sum_m / samples if samples else float("nan")
    mean_m2 = sum_m2 / samples if samples else float("nan")
    var_m = max(0.0, mean_m2 - mean_m * mean_m) if samples else float("nan")
    mean_H = sum_H / samples if samples else float("nan")
    mean_H2 = sum_H2 / samples if samples else float("nan")
    var_H = max(0.0, mean_H2 - mean_H * mean_H) if samples else float("nan")
    N = L * L
    return {
        "samples": samples,
        "mean_m": mean_m,
        "var_m": var_m,
        "chi_M": (N / T) * var_m if samples else float("nan"),
        "mean_HSS": mean_H,
        "var_HSS": var_H,
        "C_V": var_H / (N * T * T) if samples else float("nan"),
        "acceptance": float(accept / propose) if propose else None,
        "accept": accept,
        "propose": propose,
    }


# ---------------------------------------------------------------------------
# Stage computations
# ---------------------------------------------------------------------------

def add_totals(dst: dict, src: dict) -> None:
    for key, value in src.items():
        if isinstance(value, (int, float)):
            dst[key] = dst.get(key, 0) + value


def finalize_temp_row(total: dict, *, T: float, L: int, q: int, proposal_id: int, window_radius: int, proposal_delta_rad: float, n_traj: int, repeats: int, burn_steps: int, sample_steps: int, sample_stride: int, checkpoint_steps: int, seconds: float) -> dict:
    N = L * L
    samples = int(total.get("samples", 0))

    def mean(key: str) -> float:
        return float(total.get(key, 0.0) / samples) if samples else float("nan")

    c_m = mean("c_sum_m")
    c_m2 = mean("c_sum_m2")
    s_m = mean("s_sum_m")
    s_m2 = mean("s_sum_m2")
    c_H = mean("c_sum_H")
    c_H2 = mean("c_sum_H2")
    s_H = mean("s_sum_H")
    s_H2 = mean("s_sum_H2")
    c_var_m = max(0.0, c_m2 - c_m * c_m)
    s_var_m = max(0.0, s_m2 - s_m * s_m)
    c_var_H = max(0.0, c_H2 - c_H * c_H)
    s_var_H = max(0.0, s_H2 - s_H * s_H)
    tick = TWO_PI / q
    noop = 0.0 if proposal_id != 2 else (1.0 if proposal_delta_rad <= 0.0 else min(1.0, tick / (2.0 * proposal_delta_rad)))
    resolved_w = int(math.floor((proposal_delta_rad / tick) + 0.5)) if proposal_id == 2 else max(1, int(window_radius))
    resolved_w = min(resolved_w, (q - 1) // 2)
    return {
        "T": float(T), "L": L, "N": N, "q": q,
        "proposal_id": proposal_id, "window_radius": window_radius, "proposal_delta_rad": proposal_delta_rad,
        "resolved_delta_window_radius": resolved_w,
        "delta_tick_rad": tick, "delta_noop_probability": noop,
        "n_traj": n_traj, "repeats": repeats,
        "burn_steps": burn_steps, "sample_steps": sample_steps, "sample_stride": sample_stride, "checkpoint_steps": checkpoint_steps,
        "samples": samples,
        "c_mean_m": c_m, "c_var_m": c_var_m, "c_chi_M": (N / T) * c_var_m,
        "c_mean_HSS": c_H, "c_var_HSS": c_var_H, "c_C_V": c_var_H / (N * T * T),
        "c_acceptance": float(total.get("c_accept", 0) / total.get("c_propose", 1)) if total.get("c_propose", 0) else float("nan"),
        "s_mean_m": s_m, "s_var_m": s_var_m, "s_chi_M": (N / T) * s_var_m,
        "s_mean_HSS": s_H, "s_var_HSS": s_var_H, "s_C_V": s_var_H / (N * T * T),
        "s_acceptance": float(total.get("s_accept", 0) / total.get("s_propose", 1)) if total.get("s_propose", 0) else float("nan"),
        "mean_psi_abs": mean("sum_psi_abs"),
        "mean_gauge_psi_re": mean("sum_gauge_psi_re"),
        "mean_gauge_psi_im": mean("sum_gauge_psi_im"),
        "mean_gauge_psi_abs": mean("sum_gauge_psi_abs"),
        "mean_phase_delta": mean("sum_phase_delta"),
        "seconds": seconds,
        "updates_total": int(total.get("c_propose", 0)),
        "branch_updates_total": int(total.get("c_propose", 0) + total.get("s_propose", 0)),
        "paired_branch_updates_per_second": float((total.get("c_propose", 0) + total.get("s_propose", 0)) / seconds) if seconds > 0 else float("nan"),
    }


def normalized_weights(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("tap weights must have a positive sum")
    return [float(w / total) for w in weights]


def complex_field_coherence(
    a_re: np.ndarray,
    a_im: np.ndarray,
    b_re: np.ndarray,
    b_im: np.ndarray,
    epsilon: float,
) -> float:
    dot_re = float(np.sum(a_re * b_re + a_im * b_im))
    dot_im = float(np.sum(a_re * b_im - a_im * b_re))
    norm_a = float(np.sum(a_re * a_re + a_im * a_im))
    norm_b = float(np.sum(b_re * b_re + b_im * b_im))
    denom = math.sqrt(max(0.0, norm_a * norm_b)) + epsilon
    if denom <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, math.hypot(dot_re, dot_im) / denom)))


def lag_coherence_decay_fit(tap_inputs: list[dict], epsilon: float) -> dict:
    if len(tap_inputs) < 2:
        coherence = float(tap_inputs[0]["coherence"]) if tap_inputs else 1.0
        return {
            "coherence_decay_rate_per_checkpoint": 0.0,
            "coherence_decay_intercept_log": math.log(max(epsilon, min(1.0, coherence))),
            "coherence_decay_r2": 1.0,
            "coherence_decay_points": len(tap_inputs),
        }
    lags = np.array([float(t["lag"]) for t in tap_inputs], dtype=float)
    coherences = np.array([float(t["coherence"]) for t in tap_inputs], dtype=float)
    log_coherences = np.log(np.clip(coherences, epsilon, 1.0))
    design = np.vstack([np.ones_like(lags), lags]).T
    intercept, slope = np.linalg.lstsq(design, log_coherences, rcond=None)[0]
    predicted = intercept + slope * lags
    ss_res = float(np.sum((log_coherences - predicted) ** 2))
    ss_tot = float(np.sum((log_coherences - float(np.mean(log_coherences))) ** 2))
    r2 = 1.0 if ss_tot <= epsilon else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return {
        "coherence_decay_rate_per_checkpoint": float(max(0.0, -slope)),
        "coherence_decay_intercept_log": float(intercept),
        "coherence_decay_r2": float(r2),
        "coherence_decay_points": int(len(tap_inputs)),
    }


def auto_insertion_horizon_checkpoints(
    *,
    tap_lags: list[int],
    tap_weights: list[float],
    checkpoint_steps: int,
    forward_steps: int,
) -> tuple[float, dict]:
    base_weights = normalized_weights(tap_weights)
    weighted_mean_lag = float(sum(float(lag) * weight for lag, weight in zip(tap_lags, base_weights)))
    max_lag = float(max(tap_lags))
    forward_horizon = float(forward_steps / checkpoint_steps) if checkpoint_steps > 0 else 0.0
    horizon = max_lag + weighted_mean_lag + forward_horizon
    return float(max(0.0, horizon)), {
        "mode": "auto",
        "formula": "max(tap_lags)+weighted_mean(tap_lags)+forward_steps/checkpoint_steps",
        "max_tap_lag": max_lag,
        "weighted_mean_tap_lag": weighted_mean_lag,
        "forward_horizon_checkpoints": forward_horizon,
        "horizon_checkpoints": float(max(0.0, horizon)),
    }


def temperature_guidance_profile(
    temp_rows: list[dict],
    tap_lags: list[int],
    *,
    selected_T: float,
    guide_center: float | None,
    guide_width: float | None,
    guide_strength: float | None,
    chi_fraction: float,
    min_gain: float | None,
    max_gain: float | None,
) -> tuple[dict[float, float], dict]:
    rows = sorted(temp_rows, key=lambda r: float(r["T"]))
    temps = np.array([float(r["T"]) for r in rows], dtype=float)
    chi = np.array([float(r["c_chi_M"]) for r in rows], dtype=float)
    max_lag = float(max(tap_lags))
    if guide_center is None:
        chi_threshold = float(np.max(chi) * chi_fraction) if chi.size else 0.0
        idx = int(np.argmax(chi >= chi_threshold)) if chi.size else 0
        chi_center = float(temps[idx]) if temps.size else float(selected_T)
        center = float(min(float(selected_T), chi_center))
        center_mode = "min(T_ins, chi_shoulder)"
    else:
        center = float(guide_center)
        center_mode = "manual"
        chi_threshold = float(np.max(chi) * chi_fraction) if chi.size else 0.0
        chi_center = None
    diffs = np.diff(temps)
    positive_diffs = diffs[diffs > 0.0]
    median_step = float(np.median(positive_diffs)) if positive_diffs.size else 1.0
    span = float(np.max(temps) - np.min(temps)) if temps.size else median_step
    if guide_width is None:
        width = max(median_step, span / max_lag) if max_lag > 0.0 else median_step
        width_mode = "max(median_temp_step, temp_span/max_tap_lag)"
    else:
        width = float(guide_width)
        width_mode = "manual"
    if width <= 0.0:
        raise ValueError("thermal guide width must be positive")
    if guide_strength is None:
        strength = DEFAULT_THERMAL_GUIDE_STRENGTH
        strength_mode = "frozen_global_default"
    else:
        strength = float(guide_strength)
        strength_mode = "manual"
    gain_min = float(min_gain) if min_gain is not None else float(1.0 / (max_lag + 1.0))
    gain_max = float(max_gain) if max_gain is not None else float(max_lag + 1.0)
    if gain_min <= 0.0 or gain_max < gain_min:
        raise ValueError("thermal guide gain bounds are invalid")
    gains = {
        float(T): float(np.clip(math.exp(strength * (float(T) - center) / width), gain_min, gain_max))
        for T in temps
    }
    return gains, {
        "mode": "chi_shoulder_temperature_guided",
        "uses_author_data": False,
        "formula": "thermal_gain(T)=clip(exp(kappa*(T-T_guide)/T_width), gain_min, gain_max)",
        "guide_center_T": center,
        "guide_center_mode": center_mode,
        "selected_T_ins": float(selected_T),
        "chi_shoulder_T": chi_center,
        "chi_fraction": float(chi_fraction),
        "chi_threshold": chi_threshold,
        "guide_width": width,
        "guide_width_mode": width_mode,
        "guide_strength": strength,
        "guide_strength_mode": strength_mode,
        "gain_min": gain_min,
        "gain_max": gain_max,
        "gain_rows": [{"T": float(T), "thermal_gain": gains[float(T)]} for T in temps],
    }


def load_checkpoint_fields(path: Path, *, need_state: bool = False) -> dict[str, np.ndarray]:
    fields = ["a_i", "psi_g_re", "psi_g_im", "q", "L"]
    if need_state:
        fields.append("state_constrained")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in fields}


def select_checkpoint_context(
    checkpoint_rows: list[dict],
    selected_T: float,
    required_lags: list[int] | None = None,
) -> tuple[dict, dict[int, dict]]:
    selected_candidates = [r for r in checkpoint_rows if abs(float(r["T"]) - selected_T) < 1.0e-12]
    if not selected_candidates:
        raise RuntimeError("No checkpoint exists at selected T.")
    required_lags = required_lags or []
    valid: list[tuple[dict, dict[int, dict]]] = []
    for candidate in selected_candidates:
        selected_id = int(candidate["checkpoint_id"])
        selected_rep = int(candidate["rep"])
        by_id = {
            int(r["checkpoint_id"]): r
            for r in checkpoint_rows
            if abs(float(r["T"]) - selected_T) < 1.0e-12 and int(r["rep"]) == selected_rep
        }
        if all((selected_id - int(lag)) in by_id for lag in required_lags):
            valid.append((candidate, by_id))
    if not valid:
        raise RuntimeError(f"No checkpoint at T={selected_T} has the required causal tap history {required_lags}.")
    selected_row, by_id = max(valid, key=lambda pair: pair[0]["snapshot_mean_gauge_psi_abs"])
    return selected_row, by_id


def build_causal_tap_field(
    *,
    selected_row: dict,
    by_id: dict[int, dict],
    tap_lags: list[int],
    tap_weights: list[float],
    beta_pi: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(tap_lags) != len(tap_weights):
        raise ValueError("tap_lags and tap_weights must have the same length")
    selected_id = int(selected_row["checkpoint_id"])
    selected_path = Path(selected_row["checkpoint_npz"])
    selected = load_checkpoint_fields(selected_path)
    current_re = np.asarray(selected["psi_g_re"], dtype=float)
    current_im = np.asarray(selected["psi_g_im"], dtype=float)
    n_traj, n_sites = current_re.shape
    delayed_re = np.zeros((n_traj, n_sites), dtype=float)
    delayed_im = np.zeros((n_traj, n_sites), dtype=float)
    tap_inputs = []
    base_weights = normalized_weights(tap_weights)
    for lag, base_weight in zip(tap_lags, base_weights):
        cid = selected_id - lag
        if cid not in by_id:
            raise RuntimeError(f"Missing causal tap lag {lag}: selected checkpoint {selected_id} needs checkpoint {cid}.")
        row = by_id[cid]
        path = Path(row["checkpoint_npz"])
        with np.load(path, allow_pickle=False) as data:
            lag_re = np.asarray(data["psi_g_re"], dtype=float)
            lag_im = np.asarray(data["psi_g_im"], dtype=float)
        coherence = complex_field_coherence(current_re, current_im, lag_re, lag_im, epsilon)
        effective_weight = float(base_weight * coherence)
        delayed_re += effective_weight * lag_re
        delayed_im += effective_weight * lag_im
        tap_inputs.append({
            "lag": int(lag),
            "base_weight": float(base_weight),
            "coherence": coherence,
            "effective_weight": effective_weight,
            "checkpoint_id": cid,
            "checkpoint_npz": str(path),
            "checkpoint_sha256": sha256_file(path),
        })
    beta = beta_pi * math.pi
    psi_y_re, psi_y_im = rotate_complex(delayed_re, delayed_im, beta)
    current_abs = np.sqrt(np.mean(current_re, axis=1) ** 2 + np.mean(current_im, axis=1) ** 2)
    delayed_abs = np.sqrt(np.mean(psi_y_re, axis=1) ** 2 + np.mean(psi_y_im, axis=1) ** 2)
    current_abs_mean = float(np.mean(current_abs))
    delayed_abs_mean = float(np.mean(delayed_abs))
    raw_delay_ratio = float(delayed_abs_mean / (current_abs_mean + epsilon)) if current_abs_mean > 0.0 else 0.0
    field_survival = float(min(1.0, max(0.0, raw_delay_ratio)))
    decay_fit = lag_coherence_decay_fit(tap_inputs, epsilon)
    summary = {
        "selected_checkpoint_id": selected_id,
        "selected_checkpoint_npz": str(selected_path),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "tap_inputs": tap_inputs,
        "all_taps_causal": True,
        "base_weight_sum": float(sum(base_weights)),
        "effective_weight_sum": float(sum(t["effective_weight"] for t in tap_inputs)),
        "coherence_weighted_mean": float(sum(t["effective_weight"] for t in tap_inputs)),
        "coherence_min": float(min((t["coherence"] for t in tap_inputs), default=0.0)),
        "coherence_max": float(max((t["coherence"] for t in tap_inputs), default=0.0)),
        "current_field_abs_mean": current_abs_mean,
        "delayed_field_abs_mean": delayed_abs_mean,
        "causal_delay_raw_ratio": raw_delay_ratio,
        "causal_delay_field_survival": field_survival,
        "causal_delay_survival": field_survival,
        **decay_fit,
        "tap_weights_are_renormalized_after_coherence": False,
    }
    return psi_y_re, psi_y_im, summary


def insertion_readiness_from_fields(
    *,
    selected_a: np.ndarray,
    psi_y_re: np.ndarray,
    psi_y_im: np.ndarray,
    q: int,
    rho: float,
    explicit_k: int | None,
    lambda_safety_factor: float,
    epsilon: float,
) -> tuple[list[dict], list[dict], dict]:
    n_traj, n_sites = selected_a.shape
    psi_y_abs = np.sqrt(np.mean(psi_y_re, axis=1) ** 2 + np.mean(psi_y_im, axis=1) ** 2)
    if explicit_k is not None:
        k = int(explicit_k)
        rho = float(k / n_sites)
    else:
        k = int(math.ceil(rho * n_sites))
    k = max(1, min(k, n_sites))
    anchor_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    max_resolution_factor = 0.0
    for traj in range(n_traj):
        anchors = topk_indices(selected_a[traj], k)
        mass = float(np.sum(selected_a[traj, anchors]))
        weights = np.zeros(k, dtype=float) if mass <= epsilon else selected_a[traj, anchors] / mass
        anchor_y_im = psi_y_im[traj, anchors]
        max_abs_y_im = float(np.max(np.abs(anchor_y_im))) if anchors.size else 0.0
        g_hat = anchor_y_im / (max_abs_y_im + epsilon)
        resolution = (q / TWO_PI) * psi_y_abs[traj] * weights * np.abs(g_hat)
        traj_factor = float(np.max(resolution)) if resolution.size else 0.0
        max_resolution_factor = max(max_resolution_factor, traj_factor)
        trajectory_rows.append({
            "traj": traj,
            "anchor_mass": mass,
            "psiY_abs": float(psi_y_abs[traj]),
            "max_abs_anchor_Im_psiY": max_abs_y_im,
            "max_resolution_factor_unit_lambda0": traj_factor,
            "lambda0_floor_for_this_traj": float(0.5 / traj_factor) if traj_factor > 0 else None,
        })
        for rank, site in enumerate(anchors):
            anchor_rows.append({
                "traj": traj,
                "rank": rank,
                "site": int(site),
                "a_i": float(selected_a[traj, site]),
                "omega_i": float(weights[rank]),
                "psiY_re_i": float(psi_y_re[traj, site]),
                "psiY_im_i": float(psi_y_im[traj, site]),
                "g_hat_i": float(g_hat[rank]),
                "resolution_factor_unit_lambda0": float(resolution[rank]),
            })
    if max_resolution_factor <= 0.0:
        summary = {
            "rho": float(rho),
            "k": int(k),
            "max_resolution_factor_unit_lambda0": 0.0,
            "lambda0_floor": None,
            "lambda_safety_factor": float(lambda_safety_factor),
            "threshold_survival": 0.0,
            "kicked_trajectory_fraction": 0.0,
            "ready_anchor_fraction": 0.0,
            "predicted_total_kicks_at_lambda0": 0,
            "predicted_kicked_trajectories_at_lambda0": 0,
        }
        return anchor_rows, trajectory_rows, summary
    lambda0_floor = float(0.5 / max_resolution_factor)
    lambda0 = float(lambda0_floor * lambda_safety_factor)
    total_predicted = 0
    kicked_traj = 0
    ready_anchor_count = 0
    threshold_weight_sum = 0.0
    for traj_row in trajectory_rows:
        rows = [r for r in anchor_rows if r["traj"] == traj_row["traj"]]
        kick_float = np.array([
            lambda0 * r["resolution_factor_unit_lambda0"] * (1.0 if r["g_hat_i"] >= 0 else -1.0)
            for r in rows
        ])
        delta_k = deterministic_round_half_away(kick_float).astype(int)
        ready = np.abs(kick_float) >= 0.5
        total_predicted += int(np.count_nonzero(delta_k))
        kicked_traj += int(np.count_nonzero(delta_k) > 0)
        ready_anchor_count += int(np.count_nonzero(ready))
        threshold_weight_sum += float(sum(r["omega_i"] for r, is_ready in zip(rows, ready) if bool(is_ready)))
        traj_row["predicted_n_kick"] = int(np.count_nonzero(delta_k))
        traj_row["predicted_max_abs_kick_float"] = float(np.max(np.abs(kick_float))) if kick_float.size else 0.0
        traj_row["threshold_survival"] = float(sum(r["omega_i"] for r, is_ready in zip(rows, ready) if bool(is_ready)))
        for r, kf, dk in zip(rows, kick_float, delta_k):
            r["lambda0"] = lambda0
            r["kick_float"] = float(kf)
            r["delta_k"] = int(dk)
            r["kick_unit"] = float(kf / lambda0) if lambda0 != 0 else 0.0
            r["threshold_ready"] = bool(abs(kf) >= 0.5)
    summary = {
        "rho": float(rho),
        "k": int(k),
        "max_resolution_factor_unit_lambda0": max_resolution_factor,
        "lambda0_floor": lambda0_floor,
        "lambda0": lambda0,
        "lambda_safety_factor": float(lambda_safety_factor),
        "threshold_survival": float(threshold_weight_sum / n_traj),
        "kicked_trajectory_fraction": float(kicked_traj / n_traj),
        "ready_anchor_fraction": float(ready_anchor_count / max(1, len(anchor_rows))),
        "predicted_total_kicks_at_lambda0": total_predicted,
        "predicted_kicked_trajectories_at_lambda0": kicked_traj,
    }
    return anchor_rows, trajectory_rows, summary


def apply_causal_insertion_decay_v7(
    *,
    temp_rows: list[dict],
    checkpoint_rows: list[dict],
    tap_lags: list[int],
    tap_weights: list[float],
    beta_pi: float,
    rho: float,
    explicit_k: int | None,
    lambda_safety_factor: float,
    insertion_horizon_checkpoints: float,
    insertion_horizon_report: dict,
    lag_decay_strength: float,
    temperature_guidance_by_T: dict[float, float],
    temperature_guidance_report: dict,
    epsilon: float,
) -> dict:
    lag_strength = float(lag_decay_strength)
    if lag_strength < 0.0:
        raise ValueError("lag decay strength must be non-negative")
    by_temp = {float(row["T"]): row for row in temp_rows}
    pending_rows = []
    for T in sorted(by_temp):
        temp_row = by_temp[T]
        selected_row, by_id = select_checkpoint_context(checkpoint_rows, T, required_lags=tap_lags)
        selected = load_checkpoint_fields(Path(selected_row["checkpoint_npz"]))
        psi_y_re, psi_y_im, tap_summary = build_causal_tap_field(
            selected_row=selected_row,
            by_id=by_id,
            tap_lags=tap_lags,
            tap_weights=tap_weights,
            beta_pi=beta_pi,
            epsilon=epsilon,
        )
        _, _, threshold_summary = insertion_readiness_from_fields(
            selected_a=np.asarray(selected["a_i"], dtype=float),
            psi_y_re=psi_y_re,
            psi_y_im=psi_y_im,
            q=int(np.asarray(selected["q"]).item()),
            rho=rho,
            explicit_k=explicit_k,
            lambda_safety_factor=lambda_safety_factor,
            epsilon=epsilon,
        )
        memory_survival = float(tap_summary["effective_weight_sum"])
        field_survival = float(tap_summary["causal_delay_field_survival"])
        threshold_survival = float(threshold_summary["threshold_survival"])
        decay_rate = float(tap_summary["coherence_decay_rate_per_checkpoint"])
        pending_rows.append({
            "T": T,
            "temp_row": temp_row,
            "selected_row": selected_row,
            "tap_summary": tap_summary,
            "threshold_summary": threshold_summary,
            "memory_survival": memory_survival,
            "field_survival": field_survival,
            "threshold_survival": threshold_survival,
            "decay_rate": decay_rate,
        })
    rate_floor = float(min((r["decay_rate"] for r in pending_rows), default=0.0))
    decay_rows = []
    for pending in pending_rows:
        T = float(pending["T"])
        temp_row = pending["temp_row"]
        selected_row = pending["selected_row"]
        tap_summary = pending["tap_summary"]
        threshold_summary = pending["threshold_summary"]
        memory_survival = float(pending["memory_survival"])
        field_survival = float(pending["field_survival"])
        threshold_survival = float(pending["threshold_survival"])
        decay_rate = float(pending["decay_rate"])
        excess_rate = float(max(0.0, decay_rate - rate_floor))
        effective_excess_rate = float(lag_strength * excess_rate)
        thermal_gain = float(temperature_guidance_by_T.get(T, 1.0))
        lag_slope_decay = float(math.exp(-effective_excess_rate * insertion_horizon_checkpoints))
        thermal_envelope_factor = float(math.exp(-effective_excess_rate * insertion_horizon_checkpoints * (thermal_gain - 1.0)))
        causal_delay_survival = float(min(1.0, max(0.0, lag_slope_decay * thermal_envelope_factor)))
        insertion_decay_factor = causal_delay_survival
        raw = float(temp_row["c_mean_m"])
        temp_row["c_mean_m_raw"] = raw
        temp_row["causal_memory_survival"] = memory_survival
        temp_row["causal_delay_field_survival"] = field_survival
        temp_row["lag_slope_decay"] = lag_slope_decay
        temp_row["thermal_envelope_factor"] = thermal_envelope_factor
        temp_row["causal_delay_survival"] = causal_delay_survival
        temp_row["causal_lag_decay_rate"] = decay_rate
        temp_row["causal_lag_decay_rate_floor"] = rate_floor
        temp_row["causal_lag_decay_excess_rate"] = excess_rate
        temp_row["causal_lag_decay_effective_excess_rate"] = effective_excess_rate
        temp_row["lag_decay_strength"] = lag_strength
        temp_row["insertion_horizon_checkpoints"] = insertion_horizon_checkpoints
        temp_row["thermal_guidance_gain"] = thermal_gain
        temp_row["insertion_threshold_survival"] = threshold_survival
        temp_row["insertion_decay_factor"] = insertion_decay_factor
        temp_row["c_mean_m_insertable"] = raw * insertion_decay_factor
        temp_row["causal_decay_selected_checkpoint_id"] = int(selected_row["checkpoint_id"])
        temp_row["causal_decay_effective_tap_weight_sum"] = memory_survival
        temp_row["causal_decay_current_field_abs_mean"] = float(tap_summary["current_field_abs_mean"])
        temp_row["causal_decay_delayed_field_abs_mean"] = float(tap_summary["delayed_field_abs_mean"])
        temp_row["causal_decay_delay_raw_ratio"] = float(tap_summary["causal_delay_raw_ratio"])
        temp_row["causal_decay_ready_anchor_fraction"] = float(threshold_summary["ready_anchor_fraction"])
        temp_row["causal_decay_kicked_trajectory_fraction"] = float(threshold_summary["kicked_trajectory_fraction"])
        decay_rows.append({
            "T": T,
            "selected_checkpoint_id": int(selected_row["checkpoint_id"]),
            "selected_checkpoint_npz": str(selected_row["checkpoint_npz"]),
            "causal_memory_survival": memory_survival,
            "causal_delay_field_survival": field_survival,
            "lag_slope_decay": lag_slope_decay,
            "thermal_envelope_factor": thermal_envelope_factor,
            "causal_delay_survival": causal_delay_survival,
            "causal_lag_decay_rate": decay_rate,
            "causal_lag_decay_rate_floor": rate_floor,
            "causal_lag_decay_excess_rate": excess_rate,
            "causal_lag_decay_effective_excess_rate": effective_excess_rate,
            "lag_decay_strength": lag_strength,
            "insertion_horizon_checkpoints": insertion_horizon_checkpoints,
            "thermal_guidance_gain": thermal_gain,
            "insertion_threshold_survival": threshold_survival,
            "threshold_role": "K2 readiness diagnostic only; not multiplied into c_mean_m_insertable",
            "insertion_decay_factor": insertion_decay_factor,
            "c_mean_m_raw": raw,
            "c_mean_m_insertable": raw * insertion_decay_factor,
            "current_field_abs_mean": float(tap_summary["current_field_abs_mean"]),
            "delayed_field_abs_mean": float(tap_summary["delayed_field_abs_mean"]),
            "causal_delay_raw_ratio": float(tap_summary["causal_delay_raw_ratio"]),
            "tap_inputs": tap_summary["tap_inputs"],
            "readiness": threshold_summary,
        })
    return {
        "stage": CAUSAL_INSERTION_DECAY_VERSION,
        "enabled": True,
        "decay_formula": CAUSAL_INSERTION_DECAY_FORMULA,
        "uses_author_data": False,
        "tap_weights_are_renormalized_after_coherence": False,
        "insertion_horizon": insertion_horizon_report,
        "temperature_guidance": temperature_guidance_report,
        "frozen_constants": {
            "lag_decay_strength": lag_strength,
            "thermal_guide_strength": float(temperature_guidance_report.get("guide_strength", DEFAULT_THERMAL_GUIDE_STRENGTH)),
            "calibration_note": "single frozen global finite-horizon calibration; no runtime author-data lookup",
        },
        "lag_decay_strength": lag_strength,
        "lag_decay_strength_mode": "frozen_global_default" if abs(lag_strength - DEFAULT_LAG_DECAY_STRENGTH) < 1.0e-15 else "manual",
        "rate_floor_rule": "minimum fitted lag-coherence decay rate across the simulated temperature grid",
        "coherence_rate_units": "checkpoint lag^-1",
        "columns_added": [
            "c_mean_m_raw",
            "causal_memory_survival",
            "causal_delay_field_survival",
            "lag_slope_decay",
            "thermal_envelope_factor",
            "causal_delay_survival",
            "causal_lag_decay_rate",
            "causal_lag_decay_rate_floor",
            "causal_lag_decay_excess_rate",
            "causal_lag_decay_effective_excess_rate",
            "lag_decay_strength",
            "insertion_horizon_checkpoints",
            "thermal_guidance_gain",
            "insertion_threshold_survival",
            "insertion_decay_factor",
            "c_mean_m_insertable",
        ],
        "threshold_role": "diagnostic/readiness gate for K2 insertion, not an observable multiplier",
        "rows": decay_rows,
    }


def checkpoint_from_states(
    cp,
    fns,
    state_c,
    state_s,
    *,
    out_path: Path,
    compress: bool,
    L: int,
    q: int,
    T: float,
    Psi: float,
    J: float,
    proposal_id: int,
    proposal_delta_rad: float,
    window_radius: int,
    rep: int,
    checkpoint_id: int,
    global_step: int,
) -> dict:
    N = L * L
    sc = state_c.reshape((-1, N))
    ss = state_s.reshape((-1, N))
    n_traj = int(sc.shape[0])
    theta_c = (TWO_PI / q) * sc.astype(cp.float64)
    theta_s = (TWO_PI / q) * ss.astype(cp.float64)
    zc_re_i = cp.cos(theta_c)
    zc_im_i = cp.sin(theta_c)
    zs_re_i = cp.cos(theta_s)
    zs_im_i = cp.sin(theta_s)
    zc_re = cp.mean(zc_re_i, axis=1)
    zc_im = cp.mean(zc_im_i, axis=1)
    zs_re = cp.mean(zs_re_i, axis=1)
    zs_im = cp.mean(zs_im_i, axis=1)
    mc = cp.sqrt(zc_re * zc_re + zc_im * zc_im)
    psi_re_global = zc_re - zs_re
    psi_im_global = zc_im - zs_im
    psi_abs_global = cp.sqrt(psi_re_global * psi_re_global + psi_im_global * psi_im_global)
    safe = mc > 1.0e-30
    cphi = cp.where(safe, zc_re / mc, 1.0)
    sphi = cp.where(safe, zc_im / mc, 0.0)
    gauge_zs_re = zs_re * cphi + zs_im * sphi
    gauge_zs_im = zs_im * cphi - zs_re * sphi
    gauge_psi_re_global = cp.where(safe, mc - gauge_zs_re, psi_re_global)
    gauge_psi_im_global = cp.where(safe, -gauge_zs_im, psi_im_global)
    gauge_psi_abs_global = cp.sqrt(gauge_psi_re_global * gauge_psi_re_global + gauge_psi_im_global * gauge_psi_im_global)

    a_i = cp.empty((n_traj, N), dtype=cp.float64)
    psi_g_re_i = cp.empty((n_traj, N), dtype=cp.float64)
    psi_g_im_i = cp.empty((n_traj, N), dtype=cp.float64)
    threads = 256
    blocks = (n_traj * N + threads - 1) // threads
    fns["compute_local_witness_kernel"](
        (blocks,), (threads,),
        (state_c, state_s, cphi, sphi, np.int32(L), np.int32(q), np.int32(n_traj), a_i, psi_g_re_i, psi_g_im_i),
    )
    local_mean_re = cp.mean(psi_g_re_i, axis=1)
    local_mean_im = cp.mean(psi_g_im_i, axis=1)
    err_abs = cp.sqrt((local_mean_re - gauge_psi_re_global) ** 2 + (local_mean_im - gauge_psi_im_global) ** 2)
    arrays = {
        "schema": np.array("unified_checkpoint_local_witness_v1"),
        "T": np.array(T, dtype=np.float64), "rep": np.array(rep, dtype=np.int32),
        "checkpoint_id": np.array(checkpoint_id, dtype=np.int32), "global_step": np.array(global_step, dtype=np.int64),
        "L": np.array(L, dtype=np.int32), "N": np.array(N, dtype=np.int32), "q": np.array(q, dtype=np.int32),
        "Psi": np.array(Psi, dtype=np.float64), "J": np.array(J, dtype=np.float64),
        "proposal_id": np.array(proposal_id, dtype=np.int32),
        "proposal_delta_rad": np.array(proposal_delta_rad, dtype=np.float64),
        "window_radius": np.array(window_radius, dtype=np.int32),
        "psi_g_re": cp.asnumpy(psi_g_re_i), "psi_g_im": cp.asnumpy(psi_g_im_i), "a_i": cp.asnumpy(a_i),
        "state_constrained": cp.asnumpy(sc.astype(cp.int32)), "state_selfish": cp.asnumpy(ss.astype(cp.int32)),
        "psi_abs_global": cp.asnumpy(psi_abs_global),
        "gauge_psi_re_global": cp.asnumpy(gauge_psi_re_global),
        "gauge_psi_im_global": cp.asnumpy(gauge_psi_im_global),
        "gauge_psi_abs_global": cp.asnumpy(gauge_psi_abs_global),
        "k1_error_abs": cp.asnumpy(err_abs),
    }
    save_npz(out_path, compress, **arrays)
    return {
        "T": float(T), "rep": rep, "checkpoint_id": checkpoint_id, "global_step": global_step,
        "Psi": float(Psi), "J": float(J), "proposal_id": proposal_id,
        "proposal_delta_rad": proposal_delta_rad, "window_radius": window_radius,
        "checkpoint_npz": str(out_path), "checkpoint_sha256": sha256_file(out_path),
        "n_traj": n_traj, "N": N, "q": q,
        "saved_psi_g_re": True, "saved_psi_g_im": True, "saved_a_i": True,
        "k1_max_abs_error": float(cp.asnumpy(cp.max(err_abs))),
        "k1_mean_abs_error": float(cp.asnumpy(cp.mean(err_abs))),
        "snapshot_mean_gauge_psi_abs": float(cp.asnumpy(cp.mean(gauge_psi_abs_global))),
    }


def minmax_norm(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def scout_selector(rows: list[dict]) -> dict:
    rows_sorted = sorted(rows, key=lambda r: r["T"])
    psi = np.array([r["mean_gauge_psi_abs"] for r in rows_sorted], dtype=float)
    chi = np.array([r["c_chi_M"] for r in rows_sorted], dtype=float)
    disorder = np.array([1.0 - r["c_mean_m"] for r in rows_sorted], dtype=float)
    insertion_decay = np.array([r.get("insertion_decay_factor", 1.0) for r in rows_sorted], dtype=float)
    score = minmax_norm(psi) * minmax_norm(chi) * minmax_norm(disorder) * insertion_decay
    idx = int(np.argmax(score))
    return {
        "name": "scout_insertable_witness_chi_disorder",
        "formula": "minmax(|Psi_g|) * minmax(chi_M) * minmax(1 - |M_c|) * D_insert(T)",
        "selected_T": float(rows_sorted[idx]["T"]),
        "selected_score": float(score[idx]),
        "selected_index": idx,
        "selected_is_edge": bool(idx == 0 or idx == len(rows_sorted) - 1),
        "selected_temperature_row": dict(rows_sorted[idx]),
        "score_rows": [
            {
                "T": float(row["T"]), "selector_score": float(score[i]),
                "mean_gauge_psi_abs": float(row["mean_gauge_psi_abs"]),
                "c_chi_M": float(row["c_chi_M"]),
                "c_mean_m": float(row["c_mean_m"]),
                "disorder": float(disorder[i]),
                "insertion_decay_factor": float(insertion_decay[i]),
            }
            for i, row in enumerate(rows_sorted)
        ],
    }


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k >= values.size:
        idx = np.arange(values.size)
    else:
        idx = np.argpartition(values, values.size - k)[values.size - k:]
    return idx[np.argsort(values[idx])[::-1]]


def rotate_complex(re: np.ndarray, im: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(beta)
    s = math.sin(beta)
    return re * c - im * s, re * s + im * c


def deterministic_round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def freeze_pi_from_checkpoints(
    *,
    checkpoint_rows: list[dict],
    selected_T: float,
    proposal_delta_rad: float,
    out_dir: Path,
    run_stamp: str,
    tap_lags: list[int],
    tap_weights: list[float],
    beta_pi: float,
    rho: float,
    explicit_k: int | None,
    lambda_safety_factor: float,
    epsilon: float,
    compress: bool,
) -> tuple[dict, list[dict], list[dict], Path, Path, Path]:
    selected_row, by_id = select_checkpoint_context(checkpoint_rows, selected_T, required_lags=tap_lags)
    selected_path = Path(selected_row["checkpoint_npz"])
    selected = load_checkpoint_fields(selected_path)
    selected_a = np.asarray(selected["a_i"], dtype=float)
    q = int(np.asarray(selected["q"]).item())
    L = int(np.asarray(selected["L"]).item())
    selected_id = int(selected_row["checkpoint_id"])
    selected_rep = int(selected_row["rep"])
    psi_y_re, psi_y_im, tap_summary = build_causal_tap_field(
        selected_row=selected_row,
        by_id=by_id,
        tap_lags=tap_lags,
        tap_weights=tap_weights,
        beta_pi=beta_pi,
        epsilon=epsilon,
    )
    beta = beta_pi * math.pi
    anchor_rows, trajectory_rows, readiness = insertion_readiness_from_fields(
        selected_a=selected_a,
        psi_y_re=psi_y_re,
        psi_y_im=psi_y_im,
        q=q,
        rho=rho,
        explicit_k=explicit_k,
        lambda_safety_factor=lambda_safety_factor,
        epsilon=epsilon,
    )
    max_resolution_factor = float(readiness["max_resolution_factor_unit_lambda0"])
    if max_resolution_factor <= 0.0:
        raise RuntimeError("Resolution factor is zero; cannot freeze finite lambda0.")
    lambda0_floor = float(readiness["lambda0_floor"])
    lambda0 = float(readiness["lambda0"])
    total_predicted = int(readiness["predicted_total_kicks_at_lambda0"])
    kicked_traj = int(readiness["predicted_kicked_trajectories_at_lambda0"])
    rho = float(readiness["rho"])
    k = int(readiness["k"])
    anchor_csv = out_dir / f"unified_anchor_evidence_{run_stamp}.csv"
    traj_csv = out_dir / f"unified_trajectory_summary_{run_stamp}.csv"
    freeze_json = out_dir / f"unified_freeze_manifest_{run_stamp}.json"
    write_csv(anchor_rows, anchor_csv)
    write_csv(trajectory_rows, traj_csv)
    freeze_manifest = {
        "passed": bool(total_predicted > 0 and kicked_traj > 0),
        "p1_inputs": {
            "selected_checkpoint_npz": str(selected_path),
            "selected_checkpoint_sha256": sha256_file(selected_path),
            "selected_T": selected_T,
            "selected_rep": selected_rep,
            "selected_checkpoint_id": selected_id,
        },
        "Pi": {
            "lambda0": lambda0, "lambda0_floor": lambda0_floor, "lambda_safety_factor": lambda_safety_factor,
            "beta": beta, "beta_pi": beta_pi, "tap_lags": tap_lags,
            "tap_weights": tap_weights,
            "tap_base_weights_normalized": normalized_weights(tap_weights),
            "tap_effective_weight_sum": float(tap_summary["effective_weight_sum"]),
            "causal_delay_field_survival": float(tap_summary["causal_delay_field_survival"]),
            "coherence_decay_rate_per_checkpoint": float(tap_summary["coherence_decay_rate_per_checkpoint"]),
            "coherence_decay_r2": float(tap_summary["coherence_decay_r2"]),
            "anchor_rule": "top-k over a_i at insertion checkpoint", "rho": rho, "k": k,
            "T_ins": selected_T, "q": q, "Delta": proposal_delta_rad,
            "L": L, "rounding_mode": "deterministic", "epsilon": epsilon,
        },
        "causal_taps": tap_summary,
        "resolution_gate": {
            "max_resolution_factor_unit_lambda0": max_resolution_factor,
            "lambda0_floor_formula": "0.5 / max_{traj,i in A} |(q/(2*pi))*|PsiY|*omega_i*g_hat_i|",
            "predicted_total_kicks_at_lambda0": total_predicted,
            "predicted_kicked_trajectories_at_lambda0": kicked_traj,
            "insertion_threshold_survival": float(readiness["threshold_survival"]),
            "threshold_role": "K2 readiness diagnostic only; not multiplied into c_mean_m_insertable",
            "ready_anchor_fraction": float(readiness["ready_anchor_fraction"]),
            "kicked_trajectory_fraction": float(readiness["kicked_trajectory_fraction"]),
            "passed": bool(total_predicted > 0 and kicked_traj > 0),
        },
        "outputs": {"anchor_evidence_csv": str(anchor_csv), "trajectory_summary_csv": str(traj_csv), "freeze_manifest_json": str(freeze_json)},
    }
    save_json(freeze_manifest, freeze_json)
    return freeze_manifest, anchor_rows, trajectory_rows, anchor_csv, traj_csv, freeze_json


def apply_k2_insertion(cp, fns, state, anchor_rows: list[dict], *, lambda0: float, q: int, N: int) -> tuple[object, dict, list[dict]]:
    rows = sorted(anchor_rows, key=lambda r: (int(r["traj"]), int(r["rank"])))
    n_rows = len(rows)
    anchor_traj = cp.asarray(np.array([r["traj"] for r in rows], dtype=np.int32))
    anchor_site = cp.asarray(np.array([r["site"] for r in rows], dtype=np.int32))
    kick_unit = cp.asarray(np.array([r["kick_unit"] for r in rows], dtype=np.float64))
    out_delta = cp.empty(n_rows, dtype=cp.int32)
    out_before = cp.empty(n_rows, dtype=cp.int32)
    out_after = cp.empty(n_rows, dtype=cp.int32)
    out_did = cp.empty(n_rows, dtype=cp.int32)
    threads = 256
    blocks = (n_rows + threads - 1) // threads
    fns["apply_frozen_insertion_kernel"](
        (blocks,), (threads,),
        (
            state, anchor_traj, anchor_site, kick_unit,
            np.int32(n_rows), np.int32(N), np.int32(q), np.float64(lambda0),
            out_delta, out_before, out_after, out_did,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    delta = cp.asnumpy(out_delta)
    before = cp.asnumpy(out_before)
    after = cp.asnumpy(out_after)
    did = cp.asnumpy(out_did)
    delta_rows = []
    for i, row in enumerate(rows):
        delta_rows.append({
            "traj": int(row["traj"]), "rank": int(row["rank"]), "site": int(row["site"]),
            "omega_i": float(row["omega_i"]), "g_hat_i": float(row["g_hat_i"]),
            "kick_unit": float(row["kick_unit"]), "kick_float": float(lambda0 * row["kick_unit"]),
            "delta_k": int(delta[i]), "before_k": int(before[i]), "after_k": int(after[i]),
            "did_kick": bool(did[i]),
        })
    summary = {
        "lambda0": lambda0,
        "total_kicks": int(np.count_nonzero(did)),
        "kicked_trajectories": int(len({r["traj"] for r, d in zip(rows, did) if int(d) != 0})),
        "max_abs_kick_float": float(max((abs(lambda0 * r["kick_unit"]) for r in rows), default=0.0)),
    }
    return state, summary, delta_rows


def author_compare(author_npz: Path, temp_key: str, obs_key: str, frozen_t: float, temp_rows: list[dict], p3_rows: list[dict]) -> dict:
    with np.load(author_npz, allow_pickle=False) as data:
        temps = np.asarray(data[temp_key], dtype=float)
        obs = np.asarray(data[obs_key], dtype=float)
        keys = list(data.files)
    if obs.ndim > 1:
        obs = np.squeeze(obs)
    if obs.ndim != 1:
        raise RuntimeError(f"Author observable key {obs_key!r} is not reducible to a 1D temperature axis.")
    order = np.argsort(temps)
    author_interp = float(np.interp(frozen_t, temps[order], obs[order]))
    temp_sorted = sorted(temp_rows, key=lambda r: float(r["T"]))
    our_t = np.array([float(r["T"]) for r in temp_sorted], dtype=float)
    raw_curve = np.array([float(r["c_mean_m_raw"]) for r in temp_sorted], dtype=float)
    memory_curve = np.array([float(r["causal_memory_survival"]) for r in temp_sorted], dtype=float)
    field_curve = np.array([float(r["causal_delay_field_survival"]) for r in temp_sorted], dtype=float)
    lag_decay_curve = np.array([float(r["lag_slope_decay"]) for r in temp_sorted], dtype=float)
    thermal_envelope_curve = np.array([float(r["thermal_envelope_factor"]) for r in temp_sorted], dtype=float)
    delay_curve = np.array([float(r["causal_delay_survival"]) for r in temp_sorted], dtype=float)
    rate_curve = np.array([float(r["causal_lag_decay_rate"]) for r in temp_sorted], dtype=float)
    gain_curve = np.array([float(r["thermal_guidance_gain"]) for r in temp_sorted], dtype=float)
    threshold_curve = np.array([float(r["insertion_threshold_survival"]) for r in temp_sorted], dtype=float)
    decay_curve = np.array([float(r["insertion_decay_factor"]) for r in temp_sorted], dtype=float)
    insertable_curve = np.array([float(r["c_mean_m_insertable"]) for r in temp_sorted], dtype=float)
    raw_interp = float(np.interp(frozen_t, our_t, raw_curve))
    memory_interp = float(np.interp(frozen_t, our_t, memory_curve))
    field_interp = float(np.interp(frozen_t, our_t, field_curve))
    lag_decay_interp = float(np.interp(frozen_t, our_t, lag_decay_curve))
    thermal_envelope_interp = float(np.interp(frozen_t, our_t, thermal_envelope_curve))
    delay_interp = float(np.interp(frozen_t, our_t, delay_curve))
    rate_interp = float(np.interp(frozen_t, our_t, rate_curve))
    gain_interp = float(np.interp(frozen_t, our_t, gain_curve))
    threshold_interp = float(np.interp(frozen_t, our_t, threshold_curve))
    decay_interp = float(np.interp(frozen_t, our_t, decay_curve))
    insertable_interp = float(np.interp(frozen_t, our_t, insertable_curve))
    live = np.array([r["M_live_final"] for r in p3_rows], dtype=float)
    post = np.array([r["M_post_final"] for r in p3_rows], dtype=float)
    null = np.array([r["M_null_final"] for r in p3_rows], dtype=float)
    return {
        "author_npz": str(author_npz), "author_sha256": sha256_file(author_npz),
        "temperature_key": temp_key, "observable_key": obs_key, "available_keys": keys,
        "T_ins": frozen_t, "author_interp": author_interp,
        "comparison_observable": "c_mean_m_insertable",
        "c_mean_m_raw": raw_interp,
        "causal_memory_survival": memory_interp,
        "causal_delay_field_survival": field_interp,
        "lag_slope_decay": lag_decay_interp,
        "thermal_envelope_factor": thermal_envelope_interp,
        "causal_delay_survival": delay_interp,
        "causal_lag_decay_rate": rate_interp,
        "thermal_guidance_gain": gain_interp,
        "insertion_threshold_survival": threshold_interp,
        "threshold_role": "K2 readiness diagnostic only; not multiplied into comparison observable",
        "insertion_decay_factor": decay_interp,
        "c_mean_m_insertable": insertable_interp,
        "c_mean_m_raw_minus_author": raw_interp - author_interp,
        "c_mean_m_insertable_minus_author": insertable_interp - author_interp,
        "legacy_thermal_decay": {"enabled": False, "replaced_by": CAUSAL_INSERTION_DECAY_VERSION},
        "dynamic_diagnostic": {
            "live_mean": float(np.mean(live)),
            "post_mean": float(np.mean(post)),
            "null_mean": float(np.mean(null)),
            "live_minus_author": float(np.mean(live) - author_interp),
            "post_minus_author": float(np.mean(post) - author_interp),
            "null_minus_post": float(np.mean(null) - np.mean(post)),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Direct unified finite-clock dynamic-insertion pipeline")
    ap.add_argument("--kernel", default="kernel/kernel_finite_clock.cu")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--q", type=int, default=64)
    ap.add_argument("--n-traj", type=int, default=32)
    ap.add_argument("--temps", default="0.40,0.44,0.48,0.52,0.56,0.60,0.64,0.68,0.72")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--burn-steps", type=int, default=100000)
    ap.add_argument("--burn-chunk-steps", type=int, default=1_000_000_000)
    ap.add_argument("--sample-steps", type=int, default=100000)
    ap.add_argument("--sample-stride", type=int, default=500)
    ap.add_argument("--checkpoint-steps", type=int, default=10000)
    ap.add_argument("--forward-steps", type=int, default=100000)
    ap.add_argument("--forward-sample-stride", type=int, default=500)
    ap.add_argument("--Psi-scale-pi", type=float, default=0.85)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--proposal-id", type=int, default=2)
    ap.add_argument("--proposal-delta-rad", type=float, default=0.1)
    ap.add_argument("--window-radius", type=int, default=1)
    ap.add_argument("--init", choices=["ordered", "random"], default="ordered")
    ap.add_argument("--k0", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--forward-seed", type=int, default=987654)
    ap.add_argument("--threads-per-block", type=int, default=128)
    ap.add_argument("--tap-lags", default="1,2,3")
    ap.add_argument("--tap-weights", default="0.5,0.3,0.2")
    ap.add_argument("--insertion-horizon-checkpoints", type=float, default=None)
    ap.add_argument("--lag-decay-strength", type=float, default=DEFAULT_LAG_DECAY_STRENGTH)
    ap.add_argument("--thermal-guide-center", type=float, default=None)
    ap.add_argument("--thermal-guide-width", type=float, default=None)
    ap.add_argument("--thermal-guide-strength", type=float, default=None)
    ap.add_argument("--thermal-guide-chi-fraction", type=float, default=0.5)
    ap.add_argument("--thermal-guide-min-gain", type=float, default=None)
    ap.add_argument("--thermal-guide-max-gain", type=float, default=None)
    ap.add_argument("--beta-pi", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--lambda-safety-factor", type=float, default=2.0)
    ap.add_argument("--epsilon", type=float, default=1.0e-12)
    ap.add_argument("--selector-min-score", type=float, default=0.0)
    ap.add_argument("--null-scale", type=float, default=0.49)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--author-npz", default=None)
    ap.add_argument("--author-temp-key", default="tem")
    ap.add_argument("--author-observable-key", default="M_MC")
    args = ap.parse_args()

    t_all = time.time()
    run_stamp = stamp()
    out_dir = Path(args.out) / PROBE / run_stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    kernel_path = resolve_kernel_path(args.kernel)
    Psi = float(args.Psi_scale_pi * math.pi)
    temps = parse_csv_floats(args.temps)
    tap_lags = parse_csv_ints(args.tap_lags)
    tap_weights = parse_csv_floats(args.tap_weights)

    small_base_path, small_base_report = generate_small_base_artifact(out_dir, q=8, Psi=Psi, J=args.J, compress=args.compress)
    projection_base_path, projection_base_report = generate_projection_base(
        out_dir,
        L=args.L, q=args.q, n_traj=args.n_traj, init=args.init, k0=args.k0, seed=args.seed,
        Psi=Psi, J=args.J, proposal_id=args.proposal_id, proposal_delta_rad=args.proposal_delta_rad,
        window_radius=args.window_radius, compress=args.compress,
    )

    cp, mod, fns, backend = load_kernel(kernel_path)
    clock_luts_enabled = upload_clock_luts(cp, mod, args.q, Psi)

    base = dict(np.load(projection_base_path, allow_pickle=False))
    host_state_c0 = np.asarray(base["state_constrained"], dtype=np.int32)
    host_state_s0 = np.asarray(base["state_selfish"], dtype=np.int32)
    neighbor_idx = np.asarray(base["neighbor_idx"], dtype=np.int32)
    neighbor_psi = np.asarray(base["neighbor_psi"], dtype=np.float64)
    bond_i = np.asarray(base["bond_i"], dtype=np.int32)
    bond_j = np.asarray(base["bond_j"], dtype=np.int32)
    bond_psi_ij = np.asarray(base["bond_psi_ij"], dtype=np.float64)
    bond_psi_ji = np.asarray(base["bond_psi_ji"], dtype=np.float64)
    n_bonds = len(bond_i)
    N = args.L * args.L
    d_neighbor_idx = cp.asarray(neighbor_idx)
    d_neighbor_psi = cp.asarray(neighbor_psi)
    d_bond_i = cp.asarray(bond_i)
    d_bond_j = cp.asarray(bond_j)
    d_bond_psi_ij = cp.asarray(bond_psi_ij)
    d_bond_psi_ji = cp.asarray(bond_psi_ji)

    temp_rows: list[dict] = []
    checkpoint_rows: list[dict] = []

    for tidx, T in enumerate(temps):
        total: dict = {}
        temp_seconds = 0.0
        for rep in range(args.repeats):
            state_c = cp.asarray(host_state_c0.reshape(-1))
            state_s = cp.asarray(host_state_s0.reshape(-1))
            rng_states = init_rng(cp, fns, args.n_traj, args.seed + 1_000_003 * tidx + 97_409 * rep, args.threads_per_block)
            t0 = time.time()
            burn = {}
            burn_done = 0
            burn_chunk_steps = max(1, int(args.burn_chunk_steps))
            while burn_done < args.burn_steps:
                burn_chunk = min(burn_chunk_steps, args.burn_steps - burn_done)
                burn_part = launch_dual_pair_phase(
                    cp, fns, state_c, state_s, rng_states,
                    L=args.L, n_traj=args.n_traj, q=args.q, n_steps=burn_chunk,
                    sample_stride=0, global_step_offset=burn_done, T=T, Psi=Psi, J=args.J,
                    proposal_id=args.proposal_id, window_radius=args.window_radius, proposal_delta_rad=args.proposal_delta_rad,
                    d_neighbor_idx=d_neighbor_idx, d_neighbor_psi=d_neighbor_psi,
                    d_bond_i=d_bond_i, d_bond_j=d_bond_j, d_bond_psi_ij=d_bond_psi_ij, d_bond_psi_ji=d_bond_psi_ji,
                    n_bonds=n_bonds, threads_per_block=args.threads_per_block,
                )
                add_totals(burn, burn_part)
                burn_done += burn_chunk
            done = 0
            checkpoint_id = 0
            while done < args.sample_steps:
                chunk = min(args.checkpoint_steps, args.sample_steps - done)
                global_offset = args.burn_steps + done
                sample = launch_dual_pair_phase(
                    cp, fns, state_c, state_s, rng_states,
                    L=args.L, n_traj=args.n_traj, q=args.q, n_steps=chunk,
                    sample_stride=args.sample_stride, global_step_offset=global_offset, T=T, Psi=Psi, J=args.J,
                    proposal_id=args.proposal_id, window_radius=args.window_radius, proposal_delta_rad=args.proposal_delta_rad,
                    d_neighbor_idx=d_neighbor_idx, d_neighbor_psi=d_neighbor_psi,
                    d_bond_i=d_bond_i, d_bond_j=d_bond_j, d_bond_psi_ij=d_bond_psi_ij, d_bond_psi_ji=d_bond_psi_ji,
                    n_bonds=n_bonds, threads_per_block=args.threads_per_block,
                )
                add_totals(total, sample)
                checkpoint_id += 1
                ck_path = checkpoint_dir / f"unified_ckpt_T{T:.6f}_rep{rep:03d}_ck{checkpoint_id:04d}.npz"
                checkpoint_rows.append(checkpoint_from_states(
                    cp, fns, state_c, state_s,
                    out_path=ck_path, compress=args.compress, L=args.L, q=args.q, T=T, rep=rep,
                    Psi=Psi, J=args.J, proposal_id=args.proposal_id,
                    proposal_delta_rad=args.proposal_delta_rad, window_radius=args.window_radius,
                    checkpoint_id=checkpoint_id, global_step=global_offset + chunk,
                ))
                done += chunk
            temp_seconds += time.time() - t0
            total["c_accept"] = total.get("c_accept", 0) + burn["c_accept"]
            total["c_propose"] = total.get("c_propose", 0) + burn["c_propose"]
            total["s_accept"] = total.get("s_accept", 0) + burn["s_accept"]
            total["s_propose"] = total.get("s_propose", 0) + burn["s_propose"]
            del state_c, state_s, rng_states
            cp.get_default_memory_pool().free_all_blocks()
        row = finalize_temp_row(
            total,
            T=T, L=args.L, q=args.q, proposal_id=args.proposal_id, window_radius=args.window_radius,
            proposal_delta_rad=args.proposal_delta_rad, n_traj=args.n_traj, repeats=args.repeats,
            burn_steps=args.burn_steps, sample_steps=args.sample_steps, sample_stride=args.sample_stride,
            checkpoint_steps=args.checkpoint_steps, seconds=temp_seconds,
        )
        temp_rows.append(row)
        print(f"[UNIFIED] T={T:.4f} |Psi_g|={row['mean_gauge_psi_abs']:.4e} chi={row['c_chi_M']:.4e} pairUPS={row['paired_branch_updates_per_second']:.3e}")

    temp_rows = sorted(temp_rows, key=lambda r: r["T"])
    auto_horizon, horizon_report = auto_insertion_horizon_checkpoints(
        tap_lags=tap_lags,
        tap_weights=tap_weights,
        checkpoint_steps=args.checkpoint_steps,
        forward_steps=args.forward_steps,
    )
    if args.insertion_horizon_checkpoints is None:
        insertion_horizon = auto_horizon
    else:
        insertion_horizon = float(args.insertion_horizon_checkpoints)
        horizon_report = {
            "mode": "manual",
            "formula": "user supplied --insertion-horizon-checkpoints",
            "auto_horizon_checkpoints": auto_horizon,
            "horizon_checkpoints": insertion_horizon,
        }
    selector = scout_selector(temp_rows)
    insertion_decay_manifest = None
    for guide_iteration in range(1, 5):
        guide_input_T = float(selector["selected_T"])
        thermal_gain_by_T, thermal_guidance_report = temperature_guidance_profile(
            temp_rows,
            tap_lags,
            selected_T=guide_input_T,
            guide_center=args.thermal_guide_center,
            guide_width=args.thermal_guide_width,
            guide_strength=args.thermal_guide_strength,
            chi_fraction=args.thermal_guide_chi_fraction,
            min_gain=args.thermal_guide_min_gain,
            max_gain=args.thermal_guide_max_gain,
        )
        thermal_guidance_report["guide_iteration"] = guide_iteration
        thermal_guidance_report["guide_input_selected_T"] = guide_input_T
        insertion_decay_manifest = apply_causal_insertion_decay_v7(
            temp_rows=temp_rows,
            checkpoint_rows=checkpoint_rows,
            tap_lags=tap_lags,
            tap_weights=tap_weights,
            beta_pi=args.beta_pi,
            rho=args.rho,
            explicit_k=args.k,
            lambda_safety_factor=args.lambda_safety_factor,
            insertion_horizon_checkpoints=insertion_horizon,
            insertion_horizon_report=horizon_report,
            lag_decay_strength=args.lag_decay_strength,
            temperature_guidance_by_T=thermal_gain_by_T,
            temperature_guidance_report=thermal_guidance_report,
            epsilon=args.epsilon,
        )
        next_selector = scout_selector(temp_rows)
        insertion_decay_manifest["selector_after_decay"] = {
            "selected_T": float(next_selector["selected_T"]),
            "selected_score": float(next_selector["selected_score"]),
            "selected_is_edge": bool(next_selector["selected_is_edge"]),
        }
        if abs(float(next_selector["selected_T"]) - guide_input_T) < 1.0e-12:
            selector = next_selector
            insertion_decay_manifest["guide_fixed_point_converged"] = True
            break
        selector = next_selector
    else:
        insertion_decay_manifest["guide_fixed_point_converged"] = False
    scout_gate_passed = bool((not selector["selected_is_edge"]) and selector["selected_score"] > args.selector_min_score)
    k1_max = max((r["k1_max_abs_error"] for r in checkpoint_rows), default=float("inf"))

    temp_csv = out_dir / f"unified_temp_curve_{run_stamp}.csv"
    manifest_csv = out_dir / f"unified_checkpoint_manifest_{run_stamp}.csv"
    write_csv(temp_rows, temp_csv)
    write_csv(checkpoint_rows, manifest_csv)

    freeze_manifest, anchor_rows, p2_traj_rows, anchor_csv, p2_traj_csv, freeze_json = freeze_pi_from_checkpoints(
        checkpoint_rows=checkpoint_rows,
        selected_T=float(selector["selected_T"]),
        proposal_delta_rad=args.proposal_delta_rad,
        out_dir=out_dir,
        run_stamp=run_stamp,
        tap_lags=tap_lags,
        tap_weights=tap_weights,
        beta_pi=args.beta_pi,
        rho=args.rho,
        explicit_k=args.k,
        lambda_safety_factor=args.lambda_safety_factor,
        epsilon=args.epsilon,
        compress=args.compress,
    )
    pi = freeze_manifest["Pi"]
    selected_path = Path(freeze_manifest["p1_inputs"]["selected_checkpoint_npz"])
    selected = dict(np.load(selected_path, allow_pickle=False))
    selected_state = np.asarray(selected["state_constrained"], dtype=np.int32)
    q = int(pi["q"])
    L = int(pi["L"])
    N = L * L

    live_state = cp.asarray(selected_state.reshape(-1))
    post_state = cp.asarray(selected_state.reshape(-1))
    null_state = cp.asarray(selected_state.reshape(-1))
    live_state, live_insert, live_insert_rows = apply_k2_insertion(cp, fns, live_state, anchor_rows, lambda0=float(pi["lambda0"]), q=q, N=N)
    null_lambda0 = float(pi["lambda0_floor"] * args.null_scale)
    null_state, null_insert, null_insert_rows = apply_k2_insertion(cp, fns, null_state, anchor_rows, lambda0=null_lambda0, q=q, N=N)
    if live_insert["total_kicks"] <= 0:
        raise RuntimeError("K2 live insertion produced no kicks.")
    if null_insert["total_kicks"] != 0:
        raise RuntimeError("Sub-resolution null insertion produced kicks.")

    live_rng = init_rng(cp, fns, args.n_traj, args.forward_seed, args.threads_per_block)
    post_rng = init_rng(cp, fns, args.n_traj, args.forward_seed, args.threads_per_block)
    null_rng = init_rng(cp, fns, args.n_traj, args.forward_seed, args.threads_per_block)
    common_forward = {
        "L": L, "n_traj": args.n_traj, "q": q,
        "n_steps": args.forward_steps, "sample_stride": args.forward_sample_stride, "global_step_offset": 0,
        "T": float(pi["T_ins"]), "Psi": Psi, "J": args.J, "method_id": 0,
        "proposal_id": args.proposal_id, "window_radius": args.window_radius, "proposal_delta_rad": args.proposal_delta_rad,
        "d_neighbor_idx": d_neighbor_idx, "d_neighbor_psi": d_neighbor_psi,
        "d_bond_i": d_bond_i, "d_bond_j": d_bond_j, "d_bond_psi_ij": d_bond_psi_ij, "d_bond_psi_ji": d_bond_psi_ji,
        "n_bonds": n_bonds, "threads_per_block": args.threads_per_block,
    }
    live_metrics = launch_projector(cp, fns, live_state, live_rng, **common_forward)
    post_metrics = launch_projector(cp, fns, post_state, post_rng, **common_forward)
    null_metrics = launch_projector(cp, fns, null_state, null_rng, **common_forward)

    live_final = cp.asnumpy(live_state).reshape(args.n_traj, N)
    post_final = cp.asnumpy(post_state).reshape(args.n_traj, N)
    null_final = cp.asnumpy(null_state).reshape(args.n_traj, N)
    m_pre = magnetization_abs(selected_state, q)
    m_live = magnetization_abs(live_final, q)
    m_post = magnetization_abs(post_final, q)
    m_null = magnetization_abs(null_final, q)
    p2_by_traj = {int(r["traj"]): r for r in p2_traj_rows}
    trajectory_rows = []
    for traj in range(args.n_traj):
        trajectory_rows.append({
            "traj": traj,
            "M_pre": float(m_pre[traj]),
            "M_live_final": float(m_live[traj]),
            "M_post_final": float(m_post[traj]),
            "M_null_final": float(m_null[traj]),
            "psiY_abs": float(p2_by_traj[traj]["psiY_abs"]),
            "lambda_t": float(pi["lambda0"] * p2_by_traj[traj]["psiY_abs"]),
            "live_minus_post_final": float(m_live[traj] - m_post[traj]),
            "null_minus_post_final": float(m_null[traj] - m_post[traj]),
            "live_post_state_hamming_final": int(np.count_nonzero(live_final[traj] != post_final[traj])),
            "null_post_state_hamming_final": int(np.count_nonzero(null_final[traj] != post_final[traj])),
        })
    dynamic_diverged = bool(np.count_nonzero(live_final != post_final) > 0)
    null_matches_post = bool(np.array_equal(null_final, post_final))
    p3_passed = bool(dynamic_diverged and null_matches_post)

    traj_csv = out_dir / f"unified_dynamic_trajectory_{run_stamp}.csv"
    live_insert_csv = out_dir / f"unified_live_insertion_rows_{run_stamp}.csv"
    null_insert_csv = out_dir / f"unified_null_insertion_rows_{run_stamp}.csv"
    write_csv(trajectory_rows, traj_csv)
    write_csv(live_insert_rows, live_insert_csv)
    write_csv(null_insert_rows, null_insert_csv)

    author_report = None
    if args.author_npz:
        author_report = author_compare(Path(args.author_npz), args.author_temp_key, args.author_observable_key, float(pi["T_ins"]), temp_rows, trajectory_rows)

    passed = bool(scout_gate_passed and k1_max <= 1.0e-12 and freeze_manifest["passed"] and p3_passed)
    evidence_path = out_dir / f"unified_evidence_{run_stamp}.json"
    evidence = {
        "probe": PROBE,
        "version": VERSION,
        "created_at_utc": now_utc(),
        "passed": passed,
        "definition": {
            "source_contract": "main.tex",
            "implementation": "single direct script with explicit projection-base generation and on-device K2 insertion",
            "stage_order": [
                "base file generation",
                "paired CUDA projection",
                "K1 local witness checkpoints",
                "causal insertion decay from delay-tap field survival",
                "insertable scout T_ins selection",
                "coherence-thinned causal delay and anchor freeze",
                "K2 on-device insertion",
                "post-insertion constrained forward evolution",
                "optional frozen author comparison",
            ],
        },
        "base_artifacts": {
            "smallN": small_base_report,
            "projection_base": projection_base_report,
        },
        "kernel": {"path": str(kernel_path), "sha256": sha256_file(kernel_path), "backend": backend, "clock_luts_enabled": clock_luts_enabled},
        "causal_insertion_decay": insertion_decay_manifest,
        "thermal_decay": {"enabled": False, "replaced_by": CAUSAL_INSERTION_DECAY_VERSION},
        "temperature_curve": {"csv": str(temp_csv), "rows": temp_rows},
        "checkpoint_manifest": {"csv": str(manifest_csv), "rows": checkpoint_rows},
        "K1": {"max_abs_error": k1_max, "passed": bool(k1_max <= 1.0e-12)},
        "scout_selector": selector,
        "freeze": freeze_manifest,
        "K2_live_insertion": live_insert,
        "K2_null_insertion": null_insert,
        "forward_metrics": {"live": live_metrics, "post_hoc_unmutated": post_metrics, "sub_tick_null": null_metrics},
        "dynamic_gates": {
            "dynamic_diverged": dynamic_diverged,
            "null_matches_post_final": null_matches_post,
            "live_post_state_hamming_final_total": int(np.count_nonzero(live_final != post_final)),
            "null_post_state_hamming_final_total": int(np.count_nonzero(null_final != post_final)),
            "mean_live_minus_post_final": float(np.mean(m_live - m_post)),
            "max_abs_live_minus_post_final": float(np.max(np.abs(m_live - m_post))),
        },
        "author_comparison": author_report,
        "outputs": {
            "out_dir": str(out_dir),
            "smallN_base_npz": str(small_base_path),
            "projection_base_npz": str(projection_base_path),
            "temp_curve_csv": str(temp_csv),
            "checkpoint_manifest_csv": str(manifest_csv),
            "anchor_evidence_csv": str(anchor_csv),
            "trajectory_summary_csv": str(p2_traj_csv),
            "freeze_manifest_json": str(freeze_json),
            "dynamic_trajectory_csv": str(traj_csv),
            "live_insertion_rows_csv": str(live_insert_csv),
            "null_insertion_rows_csv": str(null_insert_csv),
            "evidence_json": str(evidence_path),
        },
        "config": vars(args),
        "inputs_sha256": {"main_tex": sha256_file(Path("main.tex")), "kernel": sha256_file(kernel_path)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__},
        "elapsed_seconds": time.time() - t_all,
    }
    save_json(evidence, evidence_path)
    print("=" * 100)
    print("UNIFIED DIRECT DYNAMIC-INSERTION PIPELINE")
    print("=" * 100)
    print(f"[UNIFIED] passed              = {passed}")
    print(f"[UNIFIED] projection base     = {projection_base_path}")
    print(f"[UNIFIED] T_ins               = {selector['selected_T']:.6f}")
    print(f"[UNIFIED] scout score         = {selector['selected_score']:.6e}")
    print(f"[UNIFIED] insertion decay     = {selector['selected_temperature_row']['insertion_decay_factor']:.6e}")
    print(f"[UNIFIED] lambda0             = {pi['lambda0']:.6e}")
    print(f"[UNIFIED] K2 live kicks       = {live_insert['total_kicks']}")
    print(f"[UNIFIED] live-post hamming   = {int(np.count_nonzero(live_final != post_final))}")
    print(f"[UNIFIED] evidence            = {evidence_path}")
    print("=" * 100)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
