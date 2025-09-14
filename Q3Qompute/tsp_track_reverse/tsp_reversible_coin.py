#!/usr/bin/env python3
# =============================================================================
# EchoKey_TSP.py — EchoKey-7 coined-walk TSP, multi-run summary + optional reverse proof
# Qiskit 1.4 compatible • AerSimulator(MPS) • Summary-only, high-precision prints
# CC0-1.0 — Public Domain
#
# What this script does
# ---------------------
# • Builds a discrete-time coined-walk circuit on the *tour subspace* of TSP
#   (tours anchored at city 0 ⇒ M=(N−1)! basis states, padded to 2^qpos).
# • The coin is *position-dependent*: for each tour |pos=i⟩ we prepare the coin
#   to √r_i where r_i is a softmax over EchoKey-7 feature scores of the d=N−2
#   adjacent-swap moves plus a "stay" outcome with a margin (stay_gap).
# • The shift stage applies the swap (k,k+1) permutation to |pos⟩ controlled by
#   the coin value; stay=identity.
# • We run multiple randomly seeded routes and summarize: best observed tour
#   length, %-gap to the exact optimum (optional), hits(exact∨reverse), and
#   invalid padding rate.
#
# New (opt-in): Reversibility proof (no mechanics touched)
# -------------------------------------------------------
# • --prove: Builds the *same* coined-walk unitary U (no measurements), shows:
#     init     : |pos⟩ prepared, coin in |0…0⟩
#     compute  : after the very first *coin prepare* only (no shift)
#     full U   : after all steps (coin+shift per step)
#     U†U      : apply U, then U†, and compute fidelity with init (≈ 1.0)
#   Prints: q_pos/q_coin, state checks P(coin==0), gate depth/opcounts, fidelity.
#   This mirrors the chem ΔE reversible-proof report for easy side-by-side.
#
# NOTE: This enhanced version silently saves metrics/CSVs/figures/QPY to tsp/<timestamp>/
#       (or --outdir). No printed output or core computations are changed.
# =============================================================================

import math, itertools, argparse, os, json, csv
from datetime import datetime

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator

# ---- optional plotting/qpy (quietly skipped if unavailable) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from qiskit import qpy
    _HAS_QPY = True
except Exception:
    _HAS_QPY = False
# ----------------------------------------------------------------

# -----------------------------
# Printing helpers (no spam)
# -----------------------------

def hp(x, places=6):
    """High-precision formatter for floats and simple containers."""
    if isinstance(x, (float, np.floating)):
        return f"{x:.{places}f}"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "[" + ", ".join(hp(v, places) for v in x) + "]"
    return str(x)

# -----------------------------
# Initialization of |pos⟩ (biased or uniform)
# -----------------------------

def build_pos_init_amplitudes(tours, D, qpos, mode='exp', alpha=3.0, topk=0):
    """
    Return amplitude vector for |pos⟩ of length 2^qpos.
    Modes:
      • 'uniform' — equal amplitudes over the M real tours
      • 'exp'     — exp(−α·(L−L_min)) bias toward short tours (default)
      • 'topk'    — mass on the K shortest tours
    """
    M = len(tours); dim = 1 << qpos
    Ls = np.array([tour_length(t, D) for t in tours], dtype=float)

    if mode == 'uniform':
        weights = np.ones(M, dtype=float)
    elif mode == 'topk' and topk > 0:
        order = np.argsort(Ls)[:topk]
        weights = np.zeros(M, dtype=float); weights[order] = 1.0
    else:  # 'exp'
        Lmin = float(Ls.min())
        weights = np.exp(-alpha * (Ls - Lmin))

    if weights.sum() <= 0:
        weights = np.ones(M, dtype=float)
    probs = weights / weights.sum()

    amps = np.zeros(dim, dtype=complex)
    amps[:M] = np.sqrt(probs).astype(complex)
    return amps

# -----------------------------
# EchoKey-7 core scoring
# -----------------------------

def tour_length(tour, D):
    """Closed tour length for anchored tour t=[0, v1, ..., v_{N-1}] (returns to 0)."""
    L = 0.0; N = len(tour)
    for i in range(N-1):
        L += float(D[tour[i], tour[i+1]])
    L += float(D[tour[-1], tour[0]])
    return L

class EK7Context:
    """Container for precomputed neighbors and weights used in feature scoring."""
    __slots__ = ("D","knn_idx","W","beta","eta","edge_memory","knn_k")
    def __init__(self, D, knn_idx, W, beta, eta, edge_memory, knn_k):
        self.D = D
        self.knn_idx = knn_idx
        self.W = W
        self.beta = float(beta)
        self.eta = float(eta)
        self.edge_memory = edge_memory  # (u,v)→score, optional; static here
        self.knn_k = int(knn_k)

def prepare_context(D, knn_k=3, W=None, beta=3.0, eta=0.0):
    """Precompute KNN lists and default weight profile."""
    N = D.shape[0]
    idx = np.argsort(D, axis=1)
    knn_idx = np.zeros((N, knn_k), dtype=int)
    for i in range(N):
        filtered = [j for j in idx[i] if j != i][:knn_k]
        knn_idx[i, :len(filtered)] = filtered
        if len(filtered) < knn_k:
            knn_idx[i, len(filtered):] = filtered[-1]
    if W is None:
        # Default: balanced, then optionally overridden by --wprofile=robust
        W = np.array([0.5, 0.5, 0.5, 1.0, 2.5, 0.4, 0.0], dtype=float)
    return EK7Context(D=D, knn_idx=knn_idx, W=W.astype(float), beta=beta, eta=eta, edge_memory={}, knn_k=knn_k)

def delta_length_adj_swap(tour, k, D):
    """Exact local ΔL for swapping positions (k,k+1) in anchored tour t."""
    N = len(tour)
    a_prev = tour[k-1]; a = tour[k]; b = tour[k+1]
    b_next = tour[k+2] if (k+2) < N else tour[0]
    old = float(D[a_prev, a] + D[a, b] + D[b, b_next])
    new = float(D[a_prev, b] + D[b, a] + D[a, b_next])
    return new - old

def _z(edge_len, mu, sigma):
    if sigma <= 1e-12: return 0.0
    return (edge_len - mu) / sigma

def _affected_edges_lengths(tour, k, D):
    N = len(tour)
    a_prev = tour[k-1]; a = tour[k]; b = tour[k+1]
    b_next = tour[k+2] if (k+2) < N else tour[0]
    before = [float(D[a_prev, a]), float(D[a, b]), float(D[b, b_next])]
    after  = [float(D[a_prev, b]), float(D[b, a]), float(D[a, b_next])]
    return before, after

def _nn_membership_bonus(u, v, ctx):
    return float((v in ctx.knn_idx[u, :ctx.knn_k]) or (u in ctx.knn_idx[v, :ctx.knn_k]))

def ek7_features_for_move(tour, k, ctx):
    """Compute the 7 interpretable features for move k on tour t."""
    D = ctx.D; N = len(tour)
    before_edges = [(tour[k-1], tour[k]), (tour[k], tour[k+1]), (tour[k+1], tour[(k+2) % N])]
    after_edges  = [(tour[k-1], tour[k+1]), (tour[k+1], tour[k]), (tour[k], tour[(k+2) % N])]
    cyc_before = sum(_nn_membership_bonus(u,v,ctx) for (u,v) in before_edges)
    cyc_after  = sum(_nn_membership_bonus(u,v,ctx) for (u,v) in after_edges)
    CYC = (cyc_after - cyc_before) / max(1.0, float(ctx.knn_k))
    def mem_gain(edges): return sum(ctx.edge_memory.get((min(u,v),max(u,v)), 0) for (u,v) in edges)
    REC = float(mem_gain(after_edges) - mem_gain(before_edges))
    be, af = _affected_edges_lengths(tour, k, D)
    FRA = float(np.mean(be) - np.mean(af)) / (np.mean(be) + 1e-9)
    all_edge_lengths = [float(D[tour[i], tour[(i+1) % N]]) for i in range(N)]
    mu, sigma = float(np.mean(all_edge_lengths)), float(np.std(all_edge_lengths) + 1e-12)
    z_before = sum(_z(l, mu, sigma) for l in be)
    z_after  = sum(_z(l, mu, sigma) for l in af)
    OUT = float(z_before - z_after)
    dL = delta_length_adj_swap(tour, k, D)
    INT = -dL
    NON = -np.sign(dL) * (dL**2)
    ADA = float(ctx.beta)
    return np.array([CYC, REC, FRA, OUT, INT, NON, ADA], dtype=float)

def ek7_bias_for_tour(tour, ctx, stay_gap=0.25):
    """Softmax over [stay | moves] scores at inverse-temperature β with a stay margin."""
    N = len(tour); d = N - 2
    move_scores = np.zeros(d, dtype=float)
    for k in range(1, d + 1):
        feats = ek7_features_for_move(tour, k, ctx)
        move_scores[k-1] = float(np.dot(ctx.W, feats))
    stay_score = np.max(move_scores) - float(stay_gap) + float(ctx.eta)
    scores = np.concatenate(([stay_score], move_scores), axis=0)
    mx = float(np.max(scores))
    expv = np.exp(float(ctx.beta) * (scores - mx))
    r = expv / float(np.sum(expv))
    return r

# -----------------------------
# Tours & register utilities
# -----------------------------

def enumerate_tours(N):
    """All anchored tours: [0, p1, p2, ..., p_{N-1}] (no trailing 0; length wraps to 0)."""
    return [[0] + list(p) for p in itertools.permutations(range(1, N))]

def build_index_maps(tours):
    idx_of = {tuple(t): i for i, t in enumerate(tours)}
    tour_of = {i: t for i, t in enumerate(tours)}
    return idx_of, tour_of

def num_qubits_for_dim(M): return math.ceil(math.log2(M))

# -----------------------------
# Circuit builders (coin & shift)
# -----------------------------

def _ctrl_state_bits(c, n, little_endian=True):
    s = format(int(c), f'0{n}b')
    return s[::-1] if little_endian else s

def append_position_controlled_coin(qc, pos, coin, tours, ctx, stay_gap):
    """
    For each tour index i, conditionally prepare the coin to √r_i using a
    multi-controlled StatePreparation(amp), with control state = binary(i).
    This exactly realizes a position-dependent (inhomogeneous) coin.
    """
    qpos = len(pos); qcoin = len(coin); pow2_coin = 1 << qcoin
    for i, tour in enumerate(tours):
        r = ek7_bias_for_tour(tour, ctx, stay_gap=stay_gap)
        amp = np.zeros(pow2_coin, dtype=complex)
        amp[:len(r)] = np.sqrt(r).astype(complex)
        prep = StatePreparation(amp, label=f"prep√r[{i}]")
        ctrl = _ctrl_state_bits(i, qpos, little_endian=True)
        prep_c = prep.control(num_ctrl_qubits=qpos, ctrl_state=ctrl)
        qc.append(prep_c, pos[:] + coin[:])

def append_coin_controlled_shift(qc, pos, coin, tours, N):
    """
    Apply the adjacent-swap permutation (k,k+1) to |pos⟩ controlled on coin=k.
    Stay (coin=0) is identity.
    """
    idx_of, _ = build_index_maps(tours)
    qpos = len(pos); dim = 1 << qpos; d = N - 2
    for c in range(1, d + 1):
        k = c
        U = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            if i < len(tours):
                t = list(tours[i]); t[k], t[k+1] = t[k+1], t[k]
                j = idx_of[tuple(t)]
            else:
                j = i  # padded states map to themselves
            U[j, i] = 1.0 + 0.0j
        gate = UnitaryGate(U, label=f"S(k={k})")
        ctrl = _ctrl_state_bits(c, len(coin), little_endian=True)
        gate_c = gate.control(num_ctrl_qubits=len(coin), ctrl_state=ctrl)
        qc.append(gate_c, coin[:] + pos[:])

def coined_walk_circuit(D, tours, steps, beta, stay_gap,
                        init_mode='exp', alpha=3.0, topk=0, wprofile='robust'):
    """
    Build the coined-walk circuit for a given distance matrix and tour set.
    Returns (qc, pos_reg, coin_reg, cpos_reg).  (This is your original path; unchanged.)
    """
    M = len(tours)
    qpos = num_qubits_for_dim(M); d = len(tours[0]) - 2
    qcoin = math.ceil(math.log2(d + 1))
    pos = QuantumRegister(qpos, "pos")
    coin = QuantumRegister(qcoin, "coin")
    cpos = ClassicalRegister(qpos, "cpos")
    qc = QuantumCircuit(pos, coin, cpos)

    # Context (weights & β). Robust profile leans on INT (−ΔL) and stabilizers.
    ctx = prepare_context(D, knn_k=3, beta=beta, eta=0.0)
    if wprofile == "robust":
        # [CYC,  REC,  FRA,  OUT,  INT,  NON,  ADA]
        ctx.W = np.array([0.2, 0.1, 0.6, 0.8, 4.0, 0.6, 0.0], dtype=float)

    # Initialize |pos⟩ (coin stays |0…0⟩)
    amps_pos = build_pos_init_amplitudes(tours, D, qpos, mode=init_mode, alpha=alpha, topk=topk)
    qc.append(StatePreparation(amps_pos, label="init|pos>"), pos[:])

    # Steps: exact inhomogeneous coin then controlled shift; mild β ramp
    for _ in range(steps):
        append_position_controlled_coin(qc, pos, coin, tours, ctx, stay_gap)
        append_coin_controlled_shift(qc, pos, coin, tours, len(tours[0]))
        ctx.beta += 0.2

    qc.measure(pos, cpos)
    return qc, pos, coin, cpos

# --- New: unitary-only builder (no measurements), identical mechanics ---
def coined_walk_unitary(D, tours, steps, beta, stay_gap,
                        init_mode='exp', alpha=3.0, topk=0, wprofile='robust'):
    """
    Same as coined_walk_circuit but returns a *pure unitary* circuit (no classical regs),
    for use with Statevector in the reversibility proof. Mechanics and sequencing identical.
    """
    M = len(tours)
    qpos = num_qubits_for_dim(M); d = len(tours[0]) - 2
    qcoin = math.ceil(math.log2(d + 1))
    pos = QuantumRegister(qpos, "pos")
    coin = QuantumRegister(qcoin, "coin")
    qc = QuantumCircuit(pos, coin)

    ctx = prepare_context(D, knn_k=3, beta=beta, eta=0.0)
    if wprofile == "robust":
        ctx.W = np.array([0.2, 0.1, 0.6, 0.8, 4.0, 0.6, 0.0], dtype=float)

    amps_pos = build_pos_init_amplitudes(tours, D, qpos, mode=init_mode, alpha=alpha, topk=topk)
    qc.append(StatePreparation(amps_pos, label="init|pos>"), pos[:])

    for _ in range(steps):
        append_position_controlled_coin(qc, pos, coin, tours, ctx, stay_gap)
        append_coin_controlled_shift(qc, pos, coin, tours, len(tours[0]))
        ctx.beta += 0.2

    return qc, pos, coin

# -----------------------------
# Classical exact TSP (Held–Karp, optional)
# -----------------------------

def held_karp_exact(D):
    """Exact DP for symmetric TSP; returns (best_length, best_tour)."""
    N = D.shape[0]
    INF = 1e100
    size = 1 << (N - 1)
    DP_cost = [[INF]*N for _ in range(size)]
    DP_prev = [[-1]*N for _ in range(size)]
    for j in range(1, N):
        mask = 1 << (j-1)
        DP_cost[mask][j] = float(D[0, j]); DP_prev[mask][j] = 0
    for mask in range(size):
        for j in range(1, N):
            if not (mask & (1 << (j-1))):
                continue
            prev_mask = mask ^ (1 << (j-1))
            if prev_mask == 0:
                continue
            best_c = DP_cost[mask][j]
            sub = prev_mask
            while sub:
                k_bit = sub & -sub
                k = (k_bit.bit_length() - 1) + 1
                c = DP_cost[prev_mask][k] + float(D[k, j])
                if c < best_c:
                    best_c = c; DP_cost[mask][j] = c; DP_prev[mask][j] = k
                sub ^= k_bit
    full = size - 1
    best_len = 1e100; best_end = -1
    for j in range(1, N):
        c = DP_cost[full][j] + float(D[j, 0])
        if c < best_len: best_len = c; best_end = j
    tour = [0]; mask = full; j = best_end; stack = []
    while j != 0:
        stack.append(j); pj = DP_prev[mask][j]
        mask ^= (1 << (j-1)); j = pj
    tour += list(reversed(stack))
    return best_len, tour

# -----------------------------
# Reversible-proof helpers (read-only; no change to mechanics)
# -----------------------------

def index_list_for_register(qc: QuantumCircuit, reg) -> list[int]:
    return [qc.find_bit(q).index for q in reg]

def prob_register_all_zero(sv: Statevector, reg_qubits: list[int]) -> float:
    if not reg_qubits:
        return 1.0
    probs = sv.probabilities(qargs=reg_qubits)
    return float(probs[0]) if len(probs) > 0 else 1.0

# ---- quiet artifact helpers (no prints) ----
def _mk_outdir(base: str = "tsp") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def _save_csv(path: str, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def _plot_bar(values, labels, title, ylabel, path, rotate=False):
    if not _HAS_MPL: return
    plt.figure(figsize=(7, 3.8), dpi=160)
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=45 if rotate else 0, ha="right" if rotate else "center")
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def _plot_heatmap(matrix: np.ndarray, title: str, xlabel: str, ylabel: str, path: str, cmap="viridis"):
    if not _HAS_MPL: return
    plt.figure(figsize=(4.8, 4.2), dpi=160)
    im = plt.imshow(matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(path); plt.close()

def _draw_circuit_mpl(circ: QuantumCircuit, path: str):
    if not _HAS_MPL: return
    try:
        circ.draw(output="mpl", idle_wires=False, fold=-1, scale=0.8, filename=path)
    except Exception:
        fig = circ.draw(output="mpl", idle_wires=False, fold=-1, scale=0.8)
        fig.savefig(path); plt.close(fig)

# -------------------------------------------------------------

def reversible_proof_block(D, tours, steps, beta, stay_gap, init_mode, alpha, topk, wprofile, outdir: str | None = None):
    """
    Builds U (unitary coined walk), shows init/compute/full/U†U, prints fidelity and metrics.
    This mirrors the chem reversible proof, but without adding any new work registers.
    Also silently saves proof metrics/figures if outdir is provided.
    """
    # Build U (no measurements)
    U, pos, coin = coined_walk_unitary(D, tours, steps, beta, stay_gap, init_mode, alpha, topk, wprofile)

    # Init statevector
    qc_init = QuantumCircuit(*U.qregs)
    for inst, qargs, cargs in U.data:
        if getattr(inst, "label", None) == "init|pos>":
            qc_init.append(inst, qargs)
            break
    sv_init = Statevector.from_instruction(qc_init)

    # 'compute' = first *coin prepare* only (no shift), on the same init
    qc_compute = qc_init.copy()
    found = False
    for inst, qargs, cargs in U.data:
        if getattr(inst, "label", "").startswith("prep√r["):
            qc_compute.append(inst, qargs)
            found = True
            break
    if not found and len(U.data) > 1:
        qc_compute.append(U.data[1][0], U.data[1][1])
    sv_compute = Statevector.from_instruction(qc_compute)

    # full U
    qc_full = QuantumCircuit(*U.qregs)
    for inst, qargs, cargs in U.data:
        qc_full.append(inst, qargs)
    sv_full = Statevector.from_instruction(qc_full)

    # U followed by U†
    qc_prove = QuantumCircuit(*U.qregs)
    for inst, qargs, cargs in U.data:
        qc_prove.append(inst, qargs)
    qc_prove.append(U.to_gate().inverse(), [q for q in qc_prove.qubits])
    sv_prove = Statevector.from_instruction(qc_prove)
    F = state_fidelity(sv_init, sv_prove)

    # Probabilities (coin is the only non-pos register)
    idx_coin = index_list_for_register(U, coin)
    p_coin0_init = prob_register_all_zero(sv_init, idx_coin)
    p_coin0_compute = prob_register_all_zero(sv_compute, idx_coin)
    p_coin0_full = prob_register_all_zero(sv_full, idx_coin)
    p_coin0_prove = prob_register_all_zero(sv_prove, idx_coin)

    # Gate metrics
    decomp = U.decompose(reps=6)
    depth = decomp.depth()
    ops = decomp.count_ops()

    # ---- save quietly if requested ----
    if outdir is not None:
        proof = {
            "steps": int(steps),
            "q_pos": len(pos),
            "q_coin": len(coin),
            "probabilities": {
                "p_coin0_init": float(p_coin0_init),
                "p_coin0_compute": float(p_coin0_compute),
                "p_coin0_full": float(p_coin0_full),
                "p_coin0_UdaggerU": float(p_coin0_prove),
            },
            "gate_metrics": {
                "depth": int(depth),
                "ops": {str(k): int(v) for k, v in ops.items()},
            },
            "fidelity_UdaggerU": float(F),
        }
        _save_json(os.path.join(outdir, "tsp_proof_metrics.json"), proof)
        _save_csv(os.path.join(outdir, "tsp_gate_counts.csv"),
                  ["gate", "count"],
                  [[str(k), int(v)] for k, v in sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))])
        if _HAS_MPL:
            _plot_bar(
                [p_coin0_init, p_coin0_compute, p_coin0_full, p_coin0_prove],
                ["init","compute","full","U†U"],
                "P(coin==0) across stages",
                "Probability",
                os.path.join(outdir, "fig_tsp_coin_zero_probs.png"),
            )
            _draw_circuit_mpl(U, os.path.join(outdir, "fig_tsp_circuit_U.png"))
        if _HAS_QPY:
            with open(os.path.join(outdir, "tsp_U.qpy"), "wb") as f:
                qpy.dump(U, f)

    # Report (prints unchanged)
    q_pos = len(pos); q_coin = len(coin)
    print("\n=== Reversible TSP Coined-Walk Proof — Detailed Report ===")
    print(f"steps = {steps} | q_pos = {q_pos} | q_coin = {q_coin} | ancillas = 0")
    print("-- State checks (probabilities that a register is all zeros) --")
    print(f"P(coin==0) | init      = {hp(p_coin0_init,12)}")
    print(f"P(coin==0) | compute   = {hp(p_coin0_compute,12)}")
    print(f"P(coin==0) | full U    = {hp(p_coin0_full,12)}")
    print(f"P(coin==0) | U†U       = {hp(p_coin0_prove,12)}")
    print("-- Gate metrics (U decomposed structurally) --")
    print(f"depth = {depth}")
    for name, count in sorted(ops.items(), key=lambda kv: kv[0]):
        print(f"{name}: {count}")
    print("-- Final fidelity (U^† U) --")
    print(f"fidelity_after_step_then_inverse = {hp(F,12)}")

# -----------------------------
# Multi-run harness
# -----------------------------

def reverse_tour(t):
    return [0] + list(reversed(t[1:]))

def run_once(D, tours, qc, shots, transpilable, seed_sim=None):
    """Execute one shot batch; return counts mapped to tour indices and best tour length."""
    if transpilable is None:
        sim0 = AerSimulator(method="matrix_product_state", seed_simulator=seed_sim)
        tqc = transpile(qc, sim0, optimization_level=1)
    else:
        tqc = transpilable
    sim = AerSimulator(method="matrix_product_state", seed_simulator=seed_sim)
    res = sim.run(tqc, shots=shots).result().get_counts(qc)

    def bits_to_int(bitstr): return int(bitstr[::-1], 2)
    M = len(tours)
    idx_counts = {}; invalid = 0
    for bitstr, cnt in res.items():
        i = bits_to_int(bitstr)
        if i < M: idx_counts[i] = idx_counts.get(i, 0) + cnt
        else: invalid += cnt
    best_i = None; best_L = 1e100
    for i in idx_counts:
        L = tour_length(tours[i], D)
        if L < best_L:
            best_L, best_i = L, i
    best_cnt = idx_counts.get(best_i, 0)
    return idx_counts, invalid, best_i, best_L, best_cnt

def main():
    ap = argparse.ArgumentParser()
    # Problem & run parameters
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--routes", type=int, default=5, help="number of randomly seeded coordinate sets")
    ap.add_argument("--repeats", type=int, default=1, help="repeats per route")
    ap.add_argument("--shots", type=int, default=69)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.00420)
    ap.add_argument("--stay_gap", type=float, default=50.50)

    # Initialization & scoring profile
    ap.add_argument("--init", choices=["exp","uniform","topk"], default="uniform")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--wprofile", choices=["default","robust"], default="robust")

    # Reproducibility seeds
    ap.add_argument("--seed", type=int, default=8008135, help="base seed for coordinates")
    ap.add_argument("--seed_sim_base", type=int, default=1337, help="base seed for simulator; varied per route/repeat")
    ap.add_argument("--seed_transp", type=int, default=10053, help="transpiler seed for reproducibility")

    # Optional exact baseline
    ap.add_argument("--skip_exact", action="store_true", help="skip Held–Karp exact DP (no gaps/hits) — OFF by default")

    # New: opt-in reversible proof on the first route (no change to mechanics)
    ap.add_argument("--prove", action="store_true", help="print a detailed U•U† reversibility proof for route 0")

    # New: quiet artifact output directory
    ap.add_argument("--outdir", type=str, default=None, help="directory to save metrics/figures (default: tsp/<ts>)")

    args = ap.parse_args()

    N = args.N; routes = args.routes; repeats = args.repeats; shots = args.shots
    outdir = args.outdir or _mk_outdir("tsp")

    # If requested, run the reversibility proof on the *first* deterministically seeded route
    if args.prove:
        rng0 = np.random.default_rng(args.seed + 0)
        pts0 = rng0.random((N, 2))
        D0 = np.sqrt(((pts0[:, None, :] - pts0[None, :, :])**2).sum(-1))
        tours0 = enumerate_tours(N)
        # Save route0 distance heatmap quietly (nice for video)
        _plot_heatmap(D0, "Distance matrix (route 0)", "j", "i", os.path.join(outdir, "fig_route0_distance_heatmap.png"))
        _save_csv(os.path.join(outdir, "route0_D.csv"),
                  ["i\\j"] + list(range(N)),
                  [[i] + [float(D0[i,j]) for j in range(N)] for i in range(N)])
        reversible_proof_block(
            D0, tours0, args.steps, args.beta, args.stay_gap,
            args.init, args.alpha, args.topk, args.wprofile,
            outdir=outdir
        )

    route_summaries = []

    for r in range(routes):
        # --- deterministically seeded route ---
        rng = np.random.default_rng(args.seed + r)
        pts = rng.random((N, 2))
        D = np.sqrt(((pts[:, None, :] - pts[None, :, :])**2).sum(-1))
        tours = enumerate_tours(N)

        # Optional exact baseline
        exact_available = not args.skip_exact
        if exact_available:
            opt_len, opt_tour = held_karp_exact(D)
            idx_of = {tuple(t): i for i, t in enumerate(tours)}
            exact_idx = idx_of[tuple(opt_tour)]
            rev_idx   = idx_of[tuple(reverse_tour(opt_tour))]
        else:
            opt_len = None; exact_idx = None; rev_idx = None

        # --- build circuit once per route ---
        qc, pos, coin, cpos = coined_walk_circuit(
            D, tours, args.steps, args.beta, args.stay_gap,
            init_mode=args.init, alpha=args.alpha, topk=args.topk, wprofile=args.wprofile
        )
        sim0 = AerSimulator(method="matrix_product_state", seed_simulator=args.seed_sim_base + r*1000)
        tqc = transpile(qc, sim0, optimization_level=1, seed_transpiler=args.seed_transp)

        gaps = []           # %-gaps vs exact (if available)
        invalid_fracs = []  # fraction of shots landing in padded states
        hits_exact_or_rev = 0
        bestL_min = 1e100

        for k in range(repeats):
            seed_sim = args.seed_sim_base + r*1000 + k
            idx_counts, invalid, best_i, best_L, best_cnt = run_once(D, tours, qc, shots, tqc, seed_sim=seed_sim)
            if exact_available:
                gap = (best_L - opt_len) / opt_len * 100.0
                gaps.append(gap)
                hits_exact_or_rev += idx_counts.get(exact_idx, 0) + idx_counts.get(rev_idx, 0)
            invalid_fracs.append(invalid / float(shots))
            if best_L < bestL_min: bestL_min = best_L

        # Collect per-route summary (use None for NA when exact is skipped)
        route_summaries.append({
            "route": r,
            "seed": args.seed + r,
            "opt_len": (opt_len if exact_available else None),
            "bestL_min": bestL_min,
            "min_gap_pct": (min(gaps) if (exact_available and gaps) else None),
            "mean_gap_pct": (float(np.mean(gaps)) if (exact_available and gaps) else None),
            "hits_exact_or_rev": (hits_exact_or_rev if exact_available else None),
            "shots_total": shots * repeats,
            "invalid_mean": float(np.mean(invalid_fracs)),
            "exact_used": exact_available,
        })

    # ---- concise summary table (prints unchanged) ----
    print(
        "Multi-run summary  |  routes={} repeats={}  N={} steps={} shots={}  beta={} stay_gap={}  exact_baseline={}".format(
            routes, repeats, N, args.steps, shots, hp(args.beta), hp(args.stay_gap), ("on" if not args.skip_exact else "off")
        )
    )
    header = (
        "route seed      opt_len   bestL_min   min_gap%  mean_gap%  hits(exact∨rev)/shots  invalid_mean"
    )
    print(header)

    def fmt_na(v, places=6):
        return (hp(v, places) if v is not None else "   NA   ")

    for s in route_summaries:
        hits_str = (f"{s['hits_exact_or_rev']:>7}/{s['shots_total']:<7}" if s['hits_exact_or_rev'] is not None else "   NA    ")
        print(
            f"{s['route']:>5} {s['seed']:>5}  {fmt_na(s['opt_len'],6):>10}  {hp(s['bestL_min'],6):>10}  "
            f"{fmt_na(s['min_gap_pct'],4):>8}  {fmt_na(s['mean_gap_pct'],4):>9}   "
            f"{hits_str:<18}      {hp(s['invalid_mean'],6):>10}"
        )

    # overall aggregates (only over routes where exact was used)
    gaps_min = [s["min_gap_pct"] for s in route_summaries if s["min_gap_pct"] is not None]
    gaps_mean = [s["mean_gap_pct"] for s in route_summaries if s["mean_gap_pct"] is not None]
    routes_with_hits = [s for s in route_summaries if s["hits_exact_or_rev"] is not None and s["hits_exact_or_rev"] > 0]

    if gaps_min and gaps_mean:
        avg_min_gap = float(np.mean(gaps_min))
        avg_mean_gap = float(np.mean(gaps_mean))
        routes_hit = len(routes_with_hits)
        print(
            f"\nOverall: avg(min_gap%)={hp(avg_min_gap,4)}, avg(mean_gap%)={hp(avg_mean_gap,4)}, routes_with_exact_or_reverse_hit={routes_hit}/{len(gaps_min)}"
        )
    else:
        print("\nOverall: exact baseline skipped (no gap/hit aggregates).")

    # ---- silent artifact save (document-compliant) ----
    # metrics.json
    metrics = {
        "args": {
            "N": N, "routes": routes, "repeats": repeats, "shots": shots,
            "steps": args.steps, "beta": args.beta, "stay_gap": args.stay_gap,
            "init": args.init, "alpha": args.alpha, "topk": args.topk, "wprofile": args.wprofile,
            "seed": args.seed, "seed_sim_base": args.seed_sim_base, "seed_transp": args.seed_transp,
            "skip_exact": bool(args.skip_exact), "prove": bool(args.prove),
        },
        "route_summaries": route_summaries,
        "aggregates": {
            "avg_min_gap_pct": (float(np.mean(gaps_min)) if gaps_min else None),
            "avg_mean_gap_pct": (float(np.mean(gaps_mean)) if gaps_mean else None),
            "routes_with_hit": (len(routes_with_hits) if gaps_min else None),
            "routes_with_exact": (len(gaps_min) if gaps_min else None),
        },
    }
    _save_json(os.path.join(outdir, "tsp_metrics.json"), metrics)

    # per-route CSV
    _save_csv(
        os.path.join(outdir, "tsp_routes_summary.csv"),
        ["route","seed","opt_len","bestL_min","min_gap_pct","mean_gap_pct","hits_exact_or_rev","shots_total","invalid_mean","exact_used"],
        [[s["route"], s["seed"], s["opt_len"], s["bestL_min"], s["min_gap_pct"], s["mean_gap_pct"],
          s["hits_exact_or_rev"], s["shots_total"], s["invalid_mean"], int(bool(s["exact_used"]))] for s in route_summaries]
    )

    # figures (best-effort)
    if _HAS_MPL:
        # gaps
        r_ids = [s["route"] for s in route_summaries if s["min_gap_pct"] is not None]
        min_gaps = [s["min_gap_pct"] for s in route_summaries if s["min_gap_pct"] is not None]
        mean_gaps = [s["mean_gap_pct"] for s in route_summaries if s["mean_gap_pct"] is not None]
        if r_ids:
            _plot_bar(min_gaps, [str(r) for r in r_ids], "Min gap % per route", "percent", os.path.join(outdir, "fig_tsp_min_gap_per_route.png"), rotate=True)
            _plot_bar(mean_gaps, [str(r) for r in r_ids], "Mean gap % per route", "percent", os.path.join(outdir, "fig_tsp_mean_gap_per_route.png"), rotate=True)
        # hits
        hits_vals = [s["hits_exact_or_rev"] if s["hits_exact_or_rev"] is not None else 0 for s in route_summaries]
        _plot_bar(hits_vals, [str(s["route"]) for s in route_summaries], "Hits (exact ∨ reverse) per route", "count", os.path.join(outdir, "fig_tsp_hits_per_route.png"), rotate=True)
        # invalid
        invalid_vals = [s["invalid_mean"] for s in route_summaries]
        _plot_bar(invalid_vals, [str(s["route"]) for s in route_summaries], "Invalid (padded) fraction per route", "fraction", os.path.join(outdir, "fig_tsp_invalid_per_route.png"), rotate=True)

        # manifest for convenience
        manifest = sorted([os.path.join(outdir, f) for f in os.listdir(outdir)])
        _save_json(os.path.join(outdir, "manifest.json"), {"files": manifest})

if __name__ == "__main__":
    main()
