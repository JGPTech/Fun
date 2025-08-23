#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Probability Theory (UPT) — Single-File Reference Implementation & Demo
==============================================================================

This script implements a compact, end-to-end demonstration of UPT as described in:

  Jon Poplett (2025). "Unified Probability Theory: A Complete Framework."
  (Definitions: Unified Probability Space; Probability Potential; Evolution Equation;
   Emergence Operator; Resonance Dynamics; Cascade Mechanics; Conservation Laws)

Paper ↔ Code mapping
--------------------
- Foundation / Unified Probability Space: here, each observation’s probability state is a
  distribution P over K ordered classes (Likert 1..K → 0..K−1).
- Sec. 2 Probability Potential: V_k = 0.5 * φ_a * (P_k − z_k)^2 around anchor z (baseline).
- Sec. 3 Evolution Equation (Theorem): dP/dt = ∇_P V + Λ + R + α(t) C
  (implemented via RK4; α=0 to disable cascades for clarity in this demo).
- Sec. 4 Emergence Λ: included as a tiny stabilizer (set ~0 here to isolate resonance effect).
- Sec. 5 Resonance R: couples neighboring ordinal classes (k±1) via tanh(·) terms.
- Sec. 6 Cascade C: present but disabled (α=0) in this demo.
- Sec. 9 Conservation Laws: we project to the probability simplex after every step.

What it does
------------
1) Generates synthetic ordinal (Likert 1–7) data with *neighbor label noise* so ordinal structure matters.
2) Trains a Multinomial Logistic Regression baseline (softmax).
3) Applies UPT as a dynamic, ordinal-aware post-processor.
4) Evaluates with Ranked Probability Score (RPS; lower is better for ordinal), LogLoss, Brier, ECE.
5) Prints a simple one-line verdict (e.g., "UPT is +X% better on RPS" or "Baseline is +X% better").
6) Saves metrics.json, class_distribution.png, reliability.png.

Usage
-----
    python upt_single.py
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import json, math, sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------- Config -------------------------------------
@dataclass
class Config:
    K: int = 7           # Likert 1..7 -> classes 0..6
    D: int = 8           # number of features
    N: int = 20000         # samples
    test_frac: float = 0.30
    seed: int = 7

    # UPT integrator (Sec. 3 Evolution Equation, RK4 integration)
    dt: float = 0.1
    steps: int = 4
    clip_eps: float = 1e-3

    # Probability Potential (Sec. 2): strength toward anchor z
    phi_a: float = .1337

    # Emergence Λ (Sec. 4) — set ~0 to isolate resonance effect
    beta: float = -2.02
    V_thresh: float = 0.001
    sigma: float = 2.2

    # Resonance R (Sec. 5) — light ordinal smoothing
    gamma: float = 0.05
    Vc: float = 1.5
    tau: float = 1.5

    # Cascade C (Sec. 6) — disabled in this demo
    alpha0: float = 3.0
    t_critical: int = 10
    double_every: int = 3

CFG = Config()

# ------------------------- Data Generation (ordinal) -------------------------
def make_dataset(N: int, D: int, K: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic ordinal dataset with *neighbor noise* to reward ordinal smoothing.

    1) Latent score = X @ beta + interactions
    2) Ordered thresholds partition ℝ into K classes
    3) With probability p, flip the class by ±1 (bounded) — simulates annotation noise
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(float)

    beta = rng.normal(size=D)
    inter = 0.6*(X[:,0]*X[:,1]) - 0.4*(X[:,2]*X[:,3]) + 0.3*(X[:,4]**2 - X[:,5]**2)
    score = X @ beta + inter

    thresholds = np.linspace(-2.2, 2.2, K-1)
    y = np.zeros(N, dtype=int)
    for i in range(N):
        k = np.sum(score[i] >= thresholds)
        y[i] = int(k)

    # Neighbor noise: class ±1 with prob p (bounded in [0, K−1])
    p = 0.18
    flip = rng.random(N)
    delta = np.zeros(N, dtype=int)
    delta[flip < (p/2)] = -1
    delta[(flip >= (p/2)) & (flip < p)] = +1
    y = np.clip(y + delta, 0, K-1)
    return X, y

def train_test_split(X, y, test_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(N*test_frac)
    te = idx[:n_test]
    tr = idx[n_test:]
    return (X[tr], y[tr]), (X[te], y[te])

# -------------------------- Baseline: Multinomial LR -------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.clip(s, 1e-12, None)

class MultinomialLogit:
    def __init__(self, D: int, K: int, lr: float=0.2, l2: float=1e-4, seed: int=0):
        rng = np.random.default_rng(seed)
        self.W = 0.01 * rng.standard_normal((D, K))
        self.b = np.zeros((1, K))
        self.lr = lr
        self.l2 = l2
        self.K = K

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=200, batch_size: int=512, verbose: bool=True):
        N, D = X.shape
        for ep in range(epochs):
            idx = np.random.permutation(N)
            for i in range(0, N, batch_size):
                j = idx[i:i+batch_size]
                xb, yb = X[j], y[j]
                logits = xb @ self.W + self.b
                P = softmax(logits)
                Y = np.zeros_like(P)
                Y[np.arange(P.shape[0]), yb] = 1.0
                grad_logits = (P - Y) / max(1, P.shape[0])
                grad_W = xb.T @ grad_logits + self.l2*self.W
                grad_b = grad_logits.sum(axis=0, keepdims=True)
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
            if verbose and (ep+1) % max(1, epochs//5) == 0:
                logits = X @ self.W + self.b
                P = softmax(logits)
                nll = -np.mean(np.log(np.clip(P[np.arange(N), y], 1e-12, None)))
                print(f"[MultLogit] epoch {ep+1}/{epochs} neg-LL={nll:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W + self.b)

# --------------------------- UPT Operators (paper) ---------------------------
def V_func(P: np.ndarray, z: np.ndarray, phi_a: float) -> np.ndarray:
    # Sec. 2 — Probability Potential: V_k = 0.5 * φ_a * (P_k − z_k)^2
    return 0.5 * phi_a * (P - z)**2

def grad_V(P: np.ndarray, z: np.ndarray, phi_a: float) -> np.ndarray:
    # ∇_P V = φ_a * (P − z)
    return phi_a * (P - z)

def emergence(P: np.ndarray, V: np.ndarray, beta: float, V_thresh: float, sigma: float) -> np.ndarray:
    # Sec. 4 — Emergence (tiny/off here)
    if abs(beta) < 1e-12:
        return np.zeros_like(P)
    return beta * np.tanh((V - V_thresh)/max(sigma,1e-8))

def resonance(P: np.ndarray, V: np.ndarray, gamma: float, Vc: float, tau: float) -> np.ndarray:
    # Sec. 5 — Neighbor coupling along ordinal axis
    K = P.size
    out = np.zeros_like(P)
    for k in range(K):
        acc = 0.0
        if k-1 >= 0:
            acc += math.tanh((V[k-1] - Vc)/max(tau,1e-8))
        if k+1 < K:
            acc += math.tanh((V[k+1] - Vc)/max(tau,1e-8))
        out[k] = gamma * acc
    return out

def cascade(P: np.ndarray) -> np.ndarray:
    # Sec. 6 — OFF here (α=0) but function present for completeness
    return np.zeros_like(P)

def project_simplex(p: np.ndarray, eps: float=1e-8) -> np.ndarray:
    p = np.maximum(p, eps)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / p.size
    return p / s

def integrate_upt(z: np.ndarray, cfg: Config) -> np.ndarray:
    """
    RK4 integration of dP/dt = ∇_P V + Λ + R (+ αC=0).
    Starts at P(0) = z (anchor = baseline probs).
    """
    P = z.copy()
    for s in range(cfg.steps):
        V = V_func(P, z, cfg.phi_a)
        def rhs(Pcur):
            Vc = V_func(Pcur, z, cfg.phi_a)
            g = grad_V(Pcur, z, cfg.phi_a)
            lam = emergence(Pcur, Vc, cfg.beta, cfg.V_thresh, cfg.sigma)
            res = resonance(Pcur, Vc, cfg.gamma, cfg.Vc, cfg.tau)
            cas = cascade(Pcur)  # α=0
            return g + lam + res + cas

        k1 = rhs(P)
        k2 = rhs(P + 0.5*cfg.dt*k1)
        k3 = rhs(P + 0.5*cfg.dt*k2)
        k4 = rhs(P + cfg.dt*k3)
        P = P + (cfg.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        P = project_simplex(P, cfg.clip_eps)
    return P

# ---------------------------- Metrics (proper rules) -------------------------
def clip_probs(P: np.ndarray, eps: float=1e-12) -> np.ndarray:
    P = np.clip(P, eps, 1.0-eps)
    s = np.sum(P, axis=-1, keepdims=True)
    return P / np.clip(s, eps, None)

def log_loss(P: np.ndarray, y: np.ndarray) -> float:
    P = clip_probs(P)
    N = y.shape[0]
    return -float(np.sum(np.log(P[np.arange(N), y])))/max(1, N)

def brier_score(P: np.ndarray, y: np.ndarray) -> float:
    N, K = P.shape
    Y = np.zeros((N, K), dtype=float)
    Y[np.arange(N), y] = 1.0
    return float(np.mean(np.sum((P - Y)**2, axis=1)))

def ranked_probability_score(P: np.ndarray, y: np.ndarray) -> float:
    # RPS for ordinal outcomes (lower is better)
    N, K = P.shape
    P = clip_probs(P)
    C_pred = np.cumsum(P, axis=1)
    C_true = np.zeros_like(P)
    for i in range(N):
        C_true[i, y[i]:] = 1.0
    return float(np.mean(np.sum((C_pred - C_true)**2, axis=1)))

def ece_multiclass(P: np.ndarray, y: np.ndarray, n_bins: int=15) -> float:
    N, K = P.shape
    conf = np.max(P, axis=1)
    preds = np.argmax(P, axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(preds[mask] == y[mask]))
        avg_conf = float(np.mean(conf[mask]))
        ece += (np.sum(mask)/max(1,N)) * abs(acc - avg_conf)
    return float(ece)

# ------------------------------ Visualization --------------------------------
def save_distribution_plot(P_base, P_upt, y, K):
    counts_true = np.bincount(y, minlength=K).astype(float)
    counts_true /= max(1.0, counts_true.sum())
    counts_base = P_base.mean(axis=0)
    counts_upt  = P_upt.mean(axis=0)

    xs = np.arange(K)
    width = 0.25
    plt.figure()
    plt.bar(xs - width, counts_true, width, label="True")
    plt.bar(xs, counts_base, width, label="Baseline")
    plt.bar(xs + width, counts_upt, width, label="UPT")
    plt.title("Average class distribution (test)")
    plt.xlabel("Ordinal class (0..K-1)")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.close()

def save_reliability_plot(P_base, P_upt, y):
    def reliability_points(P, y, n_bins=10):
        conf = np.max(P, axis=1)
        pred = np.argmax(P, axis=1)
        bins = np.linspace(0.0, 1.0, n_bins+1)
        xs, ys = [], []
        for b in range(n_bins):
            lo, hi = bins[b], bins[b+1]
            mask = (conf >= lo) & (conf < hi)
            if not np.any(mask):
                continue
            acc = float(np.mean(pred[mask] == y[mask]))
            xs.append((lo+hi)/2.0); ys.append(acc)
        return np.array(xs), np.array(ys)

    xb, yb = reliability_points(P_base, y)
    xu, yu = reliability_points(P_upt, y)

    plt.figure()
    plt.plot([0,1], [0,1], linestyle="--")
    plt.scatter(xb, yb, label="Baseline")
    plt.scatter(xu, yu, label="UPT")
    plt.title("Reliability (max-prob binning)")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reliability.png")
    plt.close()

# --------------------------------- Main --------------------------------------
def main():
    cfg = CFG
    # Data
    X, y = make_dataset(cfg.N, cfg.D, cfg.K, cfg.seed)
    (Xtr, ytr), (Xte, yte) = train_test_split(X, y, cfg.test_frac, cfg.seed)

    # Baseline
    base = MultinomialLogit(D=X.shape[1], K=cfg.K, lr=0.2, l2=1e-4, seed=cfg.seed)
    base.fit(Xtr, ytr, epochs=200, batch_size=512, verbose=True)
    P_base = base.predict_proba(Xte)

    # UPT post-process (Sec. 3 Evolution using ∇V + Λ + R; Λ≈0, αC=0)
    P_upt = np.zeros_like(P_base)
    for i in range(P_base.shape[0]):
        z = P_base[i]
        P_upt[i] = integrate_upt(z, cfg)

    # Metrics
    metrics = {
        "RPS/base": ranked_probability_score(P_base, yte),
        "RPS/upt":  ranked_probability_score(P_upt,  yte),
        "LogLoss/base": log_loss(P_base, yte),
        "LogLoss/upt":  log_loss(P_upt,  yte),
        "Brier/base": brier_score(P_base, yte),
        "Brier/upt":  brier_score(P_upt,  yte),
        "ECE/base": ece_multiclass(P_base, yte, n_bins=15),
        "ECE/upt":  ece_multiclass(P_upt,  yte, n_bins=15),
    }

    print("\n=== Metrics (lower is better) ===")
    for k in sorted(metrics.keys()):
        print(f"{k:>14s} : {metrics[k]:.6f}")

    # Simple one-line verdict on RPS
    rps_base = metrics["RPS/base"]
    rps_upt = metrics["RPS/upt"]
    if rps_upt < rps_base:
        rel = 100.0 * (rps_base - rps_upt) / max(1e-12, rps_base)
        verdict = f"UPT vs Baseline (RPS): UPT is {rel:.2f}% better (lower is better)."
    else:
        rel = 100.0 * (rps_upt - rps_base) / max(1e-12, rps_base)
        verdict = f"UPT vs Baseline (RPS): Baseline is {rel:.2f}% better (lower is better)."
    print("\n" + verdict)

    # Plots + save metrics
    save_distribution_plot(P_base, P_upt, yte, cfg.K)
    save_reliability_plot(P_base, P_upt, yte)
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nArtifacts saved: metrics.json, class_distribution.png, reliability.png")

if __name__ == "__main__":
    main()
