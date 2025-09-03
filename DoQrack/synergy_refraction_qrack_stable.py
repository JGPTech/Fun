# synergy_refraction_qrack_stable.py
# Stable-baseline benchmark for Qrack with "synergy refraction" and CWT mod.
# - REF:   tighter rounding, no refraction, CWT off  (reference)
# - BASE:  aggressive rounding, no refraction, CWT off  (stable baseline)
# - BASE+R:aggressive rounding, refraction mask, CWT off
# - MOD:   aggressive rounding, refraction mask, CWT on  (only change vs BASE+R is CWT)
#
# The refraction *mask* is computed once (from REF samples) and reused,
# so BASE+R and MOD see the exact same pruned gate list. This isolates the CWT effect.

import os
for k in list(os.environ):
    if k.startswith("QRACK_"):
        os.environ.pop(k, None)
os.environ["QRACK_OCL_DEFAULT_DEVICE"] = "0"     # make sure it’s the NVIDIA GPU
os.environ["QRACK_QPAGER_DEVICES"] = "0"
os.environ["QRACK_QUNITMULTI_DEVICES"] = "0"
os.environ["QRACK_QTENSORNETWORK_THRESHOLD_QB"] = "64"   # IMPORTANT
os.environ["QRACK_NONCLIFFORD_ROUNDING_THRESHOLD"] = ".6" # exact unless you set it later
os.environ["QRACK_COHERENCE_TRUNC"] = "0"

import time, random
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
from pyqrack import QrackSimulator

# =========================
# Knobs
# =========================
N        = 20
DEPTH    = 24
SHOTS    = 10000
PILOT    = 2000
SEED     = 12345
TOPK     = 256

# Gate mix
H_PROB   = 1.00
SMALL_TH = 0.13
RX_RATE  = 0.85
RY_RATE  = 0.85
ENT_PATTERN = "ring"   # "ring" or "random_pairs"

# Correlation → labels
TAU_POS  = +0.005
TAU_NEG  = -0.005
USE_PERCENTILE_FALLBACK = False   # if no α/β found, use 30/70 percentiles

# Refraction policy
KEEP_SCALE_ALPHA = 1.00
KEEP_SCALE_ZERO  = 0.00
KEEP_SCALE_BETA  = 0.00
SKIP_CX_BETA     = 1.00   # prob to skip CX across β-like edges
TN_SMALL_THRESH  = 10.9   # "small rotation" threshold used for pruning decisions

# ===== Calculation embedding (golden-angle → {-1,0,+1} trits) =====
EMBED_CALC   = True      # set True to embed; keeps all other parameters the same
CALC_MODE    = "golden"  # currently only "golden"
ZERO_EPS     = 0.00      # |sin(phase)| < ZERO_EPS → target 0; else sign(sin) → ±1

# Qrack device + aggressive approx defaults (tweak as desired)
os.environ.setdefault("QRACK_OCL_DEFAULT_DEVICE", "0")
os.environ.setdefault("QRACK_QPAGER_DEVICES", "0")
os.environ.setdefault("QRACK_QUNITMULTI_DEVICES", "0")
os.environ.setdefault("QRACK_QTENSORNETWORK_THRESHOLD_QB", "64")
os.environ.setdefault("QRACK_QUNIT_SEPARABILITY_THRESHOLD",   "0.18")
os.environ.setdefault("QRACK_NONCLIFFORD_ROUNDING_THRESHOLD", "0.60")
os.environ.setdefault("QRACK_DISABLE_QUNIT_FIDELITY_GUARD",   "1")

# MOD (coherence-weighted truncation)
MOD_LAMBDA   = "10.12"
MOD_MAXBOOST = "100.08"

# =========================
# Helpers
# =========================
def labels_from_bitstrings(bitstrings: List[str], n: int) -> Tuple[List[int], np.ndarray]:
    X = np.array([[1 if s[q]=='1' else 0 for q in range(n)] for s in bitstrings], dtype=float)
    labels, corrs = [], np.zeros(n)
    for q in range(n):
        a = X[:,q] - X[:,q].mean()
        b = X[:,(q+1)%n] - X[:,(q+1)%n].mean()
        denom = (a.std()+1e-9)*(b.std()+1e-9)
        r = float(np.mean(a*b)/denom)
        corrs[q] = r
        if r >= TAU_POS: labels.append(+1)
        elif r <= TAU_NEG: labels.append(-1)
        else: labels.append(0)

    # Percentile fallback if all zeros
    if USE_PERCENTILE_FALLBACK and all(l == 0 for l in labels):
        lo, hi = np.quantile(corrs, [0.30, 0.70])
        labels = [ -1 if r <= lo else (+1 if r >= hi else 0) for r in corrs ]
    return labels, corrs

def _golden_trits(n: int, zero_eps: float = 0.20) -> List[int]:
    """Edge-wise target trits t[q] in {-1,0,+1} for edge (q, q+1), from golden angle."""
    import math
    phi = (1 + 5**0.5) / 2.0
    golden_angle = 2*math.pi*(1 - 1/phi)  # ≈ 2.399963...
    t = []
    for q in range(n):
        phase = (q * golden_angle) % (2*math.pi)
        s = math.sin(phase)
        if abs(s) < zero_eps: t.append(0)
        elif s > 0:           t.append(+1)
        else:                 t.append(-1)
    return t  # length n (edge for each q)

def _signs_from_trits(t: List[int]) -> List[int]:
    """
    Convert edge trits t[q] (for edge (q,q+1)) to per-qubit signs s[q] in {-1,0,+1}.
    Heuristic: s[q] = sign(t[q-1] + t[q]) so that:
      - if both adjacent edges want +1 → s[q]=+1; both -1 → s[q]=-1
      - if edges disagree → s[q] can go 0, softening correlation to '0'
    """
    def sgn(x): return 0 if x == 0 else (+1 if x > 0 else -1)
    n = len(t)
    return [ sgn(t[(q-1) % n] + t[q]) for q in range(n) ]

def _cx(qs, c, t):
    if hasattr(qs, "cnot"): qs.cnot(c, t)
    elif hasattr(qs, "cx"): qs.cx(c, t)
    elif hasattr(qs, "mcx"): qs.mcx([c], t)
    else: raise AttributeError("QrackSimulator lacks CX")

def _rx(qs, q, th):
    if hasattr(qs, "rx"): qs.rx(q, th)
    elif hasattr(qs, "r"): qs.r(0, th, q)  # basis 0 ~ X

def _ry(qs, q, th):
    if hasattr(qs, "ry"): qs.ry(q, th)
    elif hasattr(qs, "r"): qs.r(1, th, q)  # basis 1 ~ Y

def _measure_shots(qs, shots: int, qubits: List[int]) -> List[str]:
    s = int(shots); qb = [int(x) for x in qubits]
    try:
        arr = qs.measure_shots(s, qb)   # (shots, qubits)
    except TypeError:
        arr = qs.measure_shots(qb, s)   # (qubits, shots)
    out = []
    for x in arr:
        bits = [('1' if (x >> i) & 1 else '0') for i in range(len(qb))]
        out.append(''.join(bits))
    return out

# --------- New: anchor/difference "calculation" helpers (no knob changes) ---------
def _phase_lock_to_target(target_edges: List[int], labels_edges: List[int]) -> Tuple[int, List[int]]:
    """
    Choose the cyclic shift k that maximizes matches between target_edges and labels_edges.
    Returns (best_shift_k, phase_aligned_target_edges).
    """
    n = len(target_edges)
    best_k, best_match = 0, -1
    for k in range(n):
        # rotate target by k (to the right): t' [i] = target[(i - k) % n]
        match = sum(1 for i in range(n) if target_edges[(i - k) % n] == labels_edges[i])
        if match > best_match:
            best_match = match
            best_k = k
    # build aligned version once
    aligned = [target_edges[(i - best_k) % n] for i in range(n)]
    return best_k, aligned

def _map_alpha_beta_zero_to_signed(seq: List[int]) -> List[int]:
    """
    Input seq entries are already in {-1,0,+1} per code's labeling convention.
    Return same (explicit for clarity).
    """
    return [int(x) for x in seq]

def _calc_correlation_scores(labels_edges: List[int], aligned_target: List[int],
                             corrs: np.ndarray) -> Tuple[float, float]:
    """
    Compute unweighted correlation C and confidence-weighted Cw using |corrs|.
    C = (1/N) * sum_i labels[i] * aligned_target[i]
    Cw = sum_i w_i * labels[i]*aligned_target[i] / sum_i w_i, with w_i = |corrs[i]|
    """
    n = len(labels_edges)
    prod = [labels_edges[i] * aligned_target[i] for i in range(n)]
    C = float(sum(prod)) / float(n) if n else 0.0
    w = np.abs(np.array(corrs, dtype=float))
    denom = float(np.sum(w)) + 1e-12
    Cw = float(np.sum(w * np.array(prod, dtype=float))) / denom
    return C, Cw

def _calc_balanced_ternary_integer(labels_edges: List[int], aligned_target: List[int]) -> Tuple[int, str, List[int]]:
    """
    Build a balanced-ternary integer D from per-edge 'difference digits'.
      e_i = labels[i] * aligned_target[i] in {-1,0,+1}
      map to digit d_i in {-1,0,+1} as:
        match (e_i=+1) -> 0
        flip  (e_i=-1) -> +1
        mute  (e_i=0)  -> -1
      D = sum_i d_i * 3^i
    Also return a human-readable trit string ("-0+") with LSB at i=0 (leftmost here).
    """
    n = len(labels_edges)
    di = []
    for i in range(n):
        e = labels_edges[i] * aligned_target[i]
        if e == +1: di.append(0)
        elif e == -1: di.append(+1)
        else: di.append(-1)
    # integer value (LSB = i=0)
    D = 0
    pow3 = 1
    for d in di:
        D += int(d) * pow3
        pow3 *= 3
    # pretty string with LSB first for direct index mapping
    sym = { -1: '-', 0: '0', +1: '+' }
    trit_str = ''.join(sym[d] for d in di)
    return int(D), trit_str, di

# =========================
# Circuit builder
# =========================
Op = Tuple[str, Tuple]  # ("h",(q,)) | ("rx",(q,θ)) | ("ry",(q,θ)) | ("cx",(c,t))

def build_ops(n: int, depth: int, seed: int) -> List[Op]:
    rng = random.Random(seed)
    ops: List[Op] = []

    # --- compute deterministic angle signs if embedding is on ---
    if EMBED_CALC and CALC_MODE == "golden":
        t_edges = _golden_trits(n, ZERO_EPS)     # edge trits for (q,q+1)
        s_qubit = _signs_from_trits(t_edges)     # per-qubit signs in {-1,0,+1}
    else:
        s_qubit = [0]*n  # 0 => unbiased random sign like before

    # sparse H layer (unchanged)
    for q in range(n):
        if rng.random() < H_PROB:
            ops.append(("h",(q,)))

    # depth layers
    for _ in range(depth):
        # small RX/RY sprinkles (keep rates/thresholds identical)
        for q in range(n):
            if rng.random() < RX_RATE:
                # keep magnitude in [0, SMALL_TH] but bias sign via s_qubit[q]
                mag = rng.random() * SMALL_TH
                if s_qubit[q] == 0:
                    theta = (rng.random()*2 - 1) * SMALL_TH
                else:
                    theta = s_qubit[q] * mag
                ops.append(("rx",(q, theta)))
            if rng.random() < RY_RATE:
                mag = rng.random() * SMALL_TH
                if s_qubit[q] == 0:
                    theta = (rng.random()*2 - 1) * SMALL_TH
                else:
                    theta = s_qubit[q] * mag
                ops.append(("ry",(q, theta)))
        # entanglers (unchanged)
        if ENT_PATTERN == "ring":
            for q in range(n):
                ops.append(("cx",(q, (q+1)%n)))
        else:
            perm = list(range(n)); rng.shuffle(perm)
            for i in range(0, n-1, 2):
                ops.append(("cx",(perm[i], perm[i+1])))
    return ops

# =========================
# Synergy estimation → labels → keep probs
# (Pilot runs with TN disabled to avoid rare MeasureShots crash)
# =========================
def pilot_synergy(n: int, ops: List[Op], shots: int, seed: int) -> Tuple[List[int], np.ndarray]:
    old_tn  = os.environ.get("QRACK_QTENSORNETWORK_THRESHOLD_QB")
    old_cwt = os.environ.get("QRACK_COHERENCE_TRUNC")
    os.environ["QRACK_QTENSORNETWORK_THRESHOLD_QB"] = "64"  # force non-TN
    os.environ["QRACK_COHERENCE_TRUNC"] = "0"
    try:
        qs = QrackSimulator(n)
        # Run full ops with no pruning for pilot
        for name, args in ops:
            if name == "h":
                qs.h(args[0])
            elif name == "rx":
                q, th = args; _rx(qs, q, th)
            elif name == "ry":
                q, th = args; _ry(qs, q, th)
            elif name == "cx":
                c, t = args; _cx(qs, c, t)
        bs = _measure_shots(qs, shots, list(range(n)))
    finally:
        if old_tn is not None: os.environ["QRACK_QTENSORNETWORK_THRESHOLD_QB"] = old_tn
        else: os.environ.pop("QRACK_QTENSORNETWORK_THRESHOLD_QB", None)
        if old_cwt is not None: os.environ["QRACK_COHERENCE_TRUNC"] = old_cwt
        else: os.environ.pop("QRACK_COHERENCE_TRUNC", None)

    X = np.array([[1 if s[q]=='1' else 0 for q in range(n)] for s in bs], dtype=float)
    labels, corrs = [], np.zeros(n)
    for q in range(n):
        a = X[:,q] - X[:,q].mean()
        b = X[:,(q+1)%n] - X[:,(q+1)%n].mean()
        denom = (a.std()+1e-9)*(b.std()+1e-9)
        r = float(np.mean(a*b)/denom)
        corrs[q] = r
        if r >= TAU_POS: labels.append(+1)
        elif r <= TAU_NEG: labels.append(-1)
        else: labels.append(0)

    # Percentile fallback if all zeros
    if USE_PERCENTILE_FALLBACK and all(l == 0 for l in labels):
        lo, hi = np.quantile(corrs, [0.30, 0.70])
        labels = [ -1 if r <= lo else (+1 if r >= hi else 0) for r in corrs ]
    return labels, corrs

def refraction_keep_from_labels(labels: List[int]) -> List[float]:
    keep = []
    for lab in labels:
        if lab > 0:   keep.append(KEEP_SCALE_ALPHA)
        elif lab < 0: keep.append(KEEP_SCALE_BETA)
        else:         keep.append(KEEP_SCALE_ZERO)
    return keep

# =========================
# Build a deterministic refraction mask (per-op)
# =========================
def build_refraction_mask(ops: List[Op], keep_prob_per_qubit: List[float],
                          skip_cx_beta: float, seed: int) -> List[bool]:
    """
    Returns a boolean mask the same length as ops.
    - H: always keep
    - RX/RY with |θ|<TN_SMALL_THRESH: keep with prob keep_prob_per_qubit[q]
    - CX: if either endpoint's keep prob equals KEEP_SCALE_BETA, skip with prob skip_cx_beta
    Deterministic given (ops, keep_prob_per_qubit, skip_cx_beta, seed).
    """
    rng = random.Random(seed ^ 0xA5A5_1234)
    mask: List[bool] = []
    for name, args in ops:
        if name == "h":
            mask.append(True)
        elif name in ("rx","ry"):
            q, th = args
            if abs(th) < TN_SMALL_THRESH:
                p = keep_prob_per_qubit[q]
                mask.append(rng.random() < p)
            else:
                mask.append(True)
        elif name == "cx":
            c, t = args
            betaish = (abs(keep_prob_per_qubit[c] - KEEP_SCALE_BETA) < 1e-9) or \
                      (abs(keep_prob_per_qubit[t] - KEEP_SCALE_BETA) < 1e-9)
            if betaish and (rng.random() < skip_cx_beta):
                mask.append(False)
            else:
                mask.append(True)
        else:
            mask.append(True)
    return mask

# =========================
# Execution
# =========================
def run_ops_with_mask(qs: QrackSimulator, n: int, ops: List[Op], mask: List[bool]):
    for (keep, op) in zip(mask, ops):
        if not keep: continue
        name, args = op
        if name == "h":
            qs.h(args[0])
        elif name == "rx":
            q, th = args; _rx(qs, q, th)
        elif name == "ry":
            q, th = args; _ry(qs, q, th)
        elif name == "cx":
            c, t = args; _cx(qs, c, t)

def run_once(n, shots, ops, mask, enable_mod: bool, rounding_threshold: float, seed: int):
    # per-run env
    os.environ["QRACK_COHERENCE_TRUNC"]    = "1" if enable_mod else "0"
    os.environ["QRACK_COHERENCE_LAMBDA"]   = MOD_LAMBDA
    os.environ["QRACK_COHERENCE_MAXBOOST"] = MOD_MAXBOOST
    old_r = os.environ.get("QRACK_NONCLIFFORD_ROUNDING_THRESHOLD")
    os.environ["QRACK_NONCLIFFORD_ROUNDING_THRESHOLD"] = f"{rounding_threshold:.3f}"
    try:
        qs = QrackSimulator(n)
        t0 = time.time()
        run_ops_with_mask(qs, n, ops, mask)
        bitstrings = _measure_shots(qs, shots, list(range(n)))
        dt = time.time() - t0
    finally:
        if old_r is not None:
            os.environ["QRACK_NONCLIFFORD_ROUNDING_THRESHOLD"] = old_r
    counts = Counter(bitstrings)
    return counts, bitstrings, dt

# =========================
# Metrics
# =========================
def prob_from_counts(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values()) or 1
    return {k: v/total for k, v in counts.items()}

def topk_tv(refP: Dict[str,float], runP: Dict[str,float], k: int) -> float:
    top = sorted(refP.items(), key=lambda kv: kv[1], reverse=True)[:k]
    S = set(k for k,_ in top)
    ref_other = 1.0 - sum(refP.get(s,0.0) for s in S)
    run_other = 1.0 - sum(runP.get(s,0.0) for s in S)
    return 0.5 * (abs(ref_other-run_other) + sum(abs(refP[s]-runP.get(s,0.0)) for s,_ in top))

def l1_1q(bitstrings: List[str], bitstrings_ref: List[str]) -> float:
    n = len(bitstrings_ref[0])
    def p1(bs):
        c = [0]*n
        for s in bs:
            for i,ch in enumerate(s):
                if ch=='1': c[i]+=1
        total = len(bs); return [ci/total for ci in c]
    a, b = p1(bitstrings), p1(bitstrings_ref)
    return sum(abs(x-y) for x,y in zip(a,b))

def l1_2q(bitstrings: List[str], bitstrings_ref: List[str], pairs: List[Tuple[int,int]]) -> float:
    def joint(bs, pairs):
        total = len(bs)
        out = {}
        for (i,j) in pairs:
            c00=c01=c10=c11=0
            for s in bs:
                a = (s[i]=='1'); b = (s[j]=='1')
                if not a and not b: c00+=1
                elif not a and b:   c01+=1
                elif a and not b:   c10+=1
                else:               c11+=1
            out[(i,j)] = [c00/total, c01/total, c10/total, c11/total]
        return out
    JA, JB = joint(bitstrings, pairs), joint(bitstrings_ref, pairs)
    keys = JA.keys() & JB.keys()
    return sum(0.5*sum(abs(x-y) for x,y in zip(JA[k], JB[k])) for k in keys)/max(1,len(keys))

# =========================
# Main
# =========================
def main():
    print(f"n={N}, depth={DEPTH}, shots={SHOTS}, pilot={PILOT}")
    print(f"ROUND={os.environ['QRACK_NONCLIFFORD_ROUNDING_THRESHOLD']}, sep={os.environ['QRACK_QUNIT_SEPARABILITY_THRESHOLD']}, TN≥{os.environ['QRACK_QTENSORNETWORK_THRESHOLD_QB']} qb")
    print(f"MOD λ={MOD_LAMBDA}, maxboost={MOD_MAXBOOST}")
    random.seed(SEED); np.random.seed(SEED)

    # Build ops
    ops = build_ops(N, DEPTH, SEED)

    # Rounding levels
    ROUND_REF  = 0.020
    ROUND_FAST = float(os.environ["QRACK_NONCLIFFORD_ROUNDING_THRESHOLD"])

    # ---- 1) REF (tighter rounding, no refraction, CWT off) ----
    mask_ref = [True]*len(ops)
    ref_counts, ref_bs, t_ref = run_once(
        N, SHOTS, ops, mask_ref, enable_mod=False, rounding_threshold=ROUND_REF, seed=SEED
    )
    refP = prob_from_counts(ref_counts)

    # ---- Build refraction map from REF samples ----
    labels, corrs   = labels_from_bitstrings(ref_bs, N)          # labels in {-1,0,+1}
    keep_probs      = refraction_keep_from_labels(labels)
    refraction_mask = build_refraction_mask(ops, keep_probs, SKIP_CX_BETA, seed=SEED)

    print("\nSynergy map (REF-based, ring neighbors):")
    legend = {+1:"α", 0:"0", -1:"β"}
    print("labels:", ''.join(legend[x] for x in labels))
    print("corr  :", ' '.join(f"{c:+.02f}" for c in corrs))
    print("keep  :", ' '.join(f"{k:.2f}" for k in keep_probs))
    print(f"β-like qubits: {sum(1 for x in labels if x<0)} / {N}")

    # ---- show the embedded "calculation" vs measured labels ----
    def _sym(seq): 
        legend = {+1:"α", 0:"0", -1:"β"}
        return ''.join(legend[int(x)] for x in seq)

    # Target (deterministic) pattern for edges
    if EMBED_CALC and CALC_MODE == "golden":
        t_edges = _golden_trits(N, ZERO_EPS)        # target trits for edges (q,(q+1)%N)
        match_raw = sum(1 for a, b in zip(t_edges, labels) if a == b)
        mism_raw  = [i for i, (a, b) in enumerate(zip(t_edges, labels)) if a != b]

        print("\n=== CALC EMBED CHECK (golden-angle, raw) ===")
        print("target:", _sym(t_edges))
        print("labels:", _sym(labels))
        print("keep  :", ' '.join(f"{k:.2f}" for k in keep_probs))
        print(f"raw match : {match_raw}/{N} ({match_raw/N:.1%})")
        if mism_raw:
            leg = {+1:"α", 0:"0", -1:"β"}
            examples = ', '.join(f"{i}:{leg[t_edges[i]]}->{leg[labels[i]]}" for i in mism_raw[:12])
            print(f"mismatch idx: {mism_raw[:24]}{' …' if len(mism_raw)>24 else ''}")
            print(f"mismatch ex : {examples}")

        # ======= NEW: Anchor/difference CALC =======
        # 1) Phase-lock to maximize anchors (matches)
        k_star, t_aligned = _phase_lock_to_target(t_edges, labels)
        anchors = [i for i in range(N) if t_aligned[i] == labels[i]]
        flips   = [i for i in range(N) if labels[i] != 0 and labels[i] == -t_aligned[i]]
        mutes   = [i for i in range(N) if labels[i] == 0]

        # 2) Correlation scores (unweighted and confidence-weighted)
        s_signed = _map_alpha_beta_zero_to_signed(labels)
        C, Cw = _calc_correlation_scores(s_signed, t_aligned, corrs)

        # 3) Balanced-ternary integer from differences
        D, trit_str, d_vec = _calc_balanced_ternary_integer(s_signed, t_aligned)

        print("\n=== CALC RESULTS (anchors & differences) ===")
        print(f"phase-lock shift k* : {k_star}")
        print(f"anchors (matches)   : {len(anchors)} / {N} -> {anchors[:16]}{' …' if len(anchors)>16 else ''}")
        print(f"flips (β vs α)      : {len(flips)} / {N} -> {flips[:16]}{' …' if len(flips)>16 else ''}")
        print(f"mutes (0 labels)    : {len(mutes)} / {N} -> {mutes[:16]}{' …' if len(mutes)>16 else ''}")
        print(f"correlation C       : {C:+.4f}")
        print(f"weighted corr Cw    : {Cw:+.4f}   (weights = |corr_i|)")
        print(f"balanced-ternary D  : {D}  (LSB first)")
        print(f"balanced-ternary d⃗ : {trit_str}   ('+'=flip, '0'=match, '-'=mute)")

    # ---- 2) BASE (aggressive rounding, no refraction, CWT off) ----
    base_counts, base_bs, t_base = run_once(
        N, SHOTS, ops, mask_ref, enable_mod=False, rounding_threshold=ROUND_FAST, seed=SEED
    )
    baseP = prob_from_counts(base_counts)

    # ---- 3) BASE+R (aggressive, REF-derived refraction mask, CWT off) ----
    baser_counts, baser_bs, t_baser = run_once(
        N, SHOTS, ops, refraction_mask, enable_mod=False, rounding_threshold=ROUND_FAST, seed=SEED
    )
    baserP = prob_from_counts(baser_counts)

    # ---- 4) MOD (aggressive, REF-derived refraction mask, CWT on) ----
    mod_counts, mod_bs, t_mod = run_once(
        N, SHOTS, ops, refraction_mask, enable_mod=True, rounding_threshold=ROUND_FAST, seed=SEED
    )
    modP = prob_from_counts(mod_counts)

    # ---- Metrics vs REF (unchanged) ----
    pairs = [(2*i, 2*i+1) for i in range(N//2)]
    def summarize(P, bs):
        tv  = topk_tv(refP, P, TOPK)
        l11 = l1_1q(bs, ref_bs)
        l12 = l1_2q(bs, ref_bs, pairs)
        return tv, l11, l12

    tv_base,  l11_base,  l12_base  = summarize(baseP,  base_bs)
    tv_baser, l11_baser, l12_baser = summarize(baserP, baser_bs)
    tv_mod,   l11_mod,   l12_mod   = summarize(modP,   mod_bs)

    # Show a few top REF states for intuition
    print("\nTop-8 REF states (probabilities in BASE / BASE+R / MOD):")
    for k, _ in sorted(refP.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        delta_b  = baseP.get(k,0.0)  - refP.get(k,0.0)
        delta_br = baserP.get(k,0.0) - refP.get(k,0.0)
        delta_m  = modP.get(k,0.0)   - refP.get(k,0.0)
        print(f"  {k} : ref={refP[k]:.4f} base={baseP.get(k,0.0):.4f} "
              f"base+R={baserP.get(k,0.0):.4f} mod={modP.get(k,0.0):.4f} "
              f"ΔB={delta_b:+.4f} ΔBR={delta_br:+.4f} ΔM={delta_m:+.4f}")

    # Summaries
    print("\n===== SUMMARY (vs tighter REF; lower is better) =====")
    print(f"TopKTV : BASE={tv_base:.4f} | BASE+R={tv_baser:.4f} | MOD={tv_mod:.4f} | Δ(MOD-BASE+R)={tv_mod-tv_baser:+.4f}")
    print(f"L1(1q) : BASE={l11_base:.4f} | BASE+R={l11_baser:.4f} | MOD={l11_mod:.4f} | Δ(MOD-BASE+R)={l11_mod-l11_baser:+.4f}")
    print(f"L1(2q) : BASE={l12_base:.4f} | BASE+R={l12_baser:.4f} | MOD={l12_mod:.4f} | Δ(MOD-BASE+R)={l12_mod-l12_baser:+.4f}")
    print(f"Runtime: REF={t_ref:.2f}s | BASE={t_base:.2f}s | BASE+R={t_baser:.2f}s | MOD={t_mod:.2f}s")

if __name__ == "__main__":
    main()
