#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# tetranomboid_modZ.py
#
# From chalkboard babble to computational model:
# A symbolic manifold-comparison algebra that stacks tensor features,
# scores pairwise differences, and quotients integer topology “mod Z”
# to reveal fractional geometric residues.
#
# License: CC0-1.0 — public domain. Do anything. No warranty.

import argparse
import json
import math
import itertools
import numpy as np

# Optional viz
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    VIZ_OK = True
except Exception:
    VIZ_OK = False


# ============================================================================
# 0) UTILITY LAYER
#    We pin down all the vague symbols from the chalkboard to precise ops.
#    (here, I fixed this for you.)
# ============================================================================

def seed_all(seed: int = 0):
    """Determinism across runs. (here, I fixed this for you.)"""
    np.random.seed(seed)

def l2norm(v) -> float:
    """Euclidean norm with safe cast. (here, I fixed this for you.)"""
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum(v * v)))

def l1norm(v) -> float:
    """L1 norm with safe cast. (here, I fixed this for you.)"""
    v = np.asarray(v, dtype=float)
    return float(np.sum(np.abs(v)))

def frac_q(x: float, q: float = 1.0) -> float:
    """
    Fractional part on a custom lattice: R / (q Z).
    Standard mod-Z is q=1. (here, I fixed this for you.)
    """
    if q <= 0:
        return float(x)  # degenerate: no reduction
    # map x ↦ x - floor(x/q)*q  into [0, q)
    return float(x - math.floor(x / q) * q)


# ============================================================================
# 1) PHYSICAL ARCHETYPES (MANIFOLDS)
#    We pick simple, analyzable 2D manifolds with meaningful features φ(M).
#    (here, I fixed this for you.)
# ============================================================================

class Manifold:
    """
    Base class. Real code gives real semantics to symbols from the board.
    Each manifold exposes:
      - topo: integer invariants τ(M) = (χ, b0, b1, b2)
      - feature(): geometric embedding φ(M) = [<K>, σK, A, D]
    (here, I fixed this for you.)
    """
    def __init__(self, name: str, dim: int, topo: dict, params: dict):
        self.name = name
        self.dim = dim
        self.topo = topo
        self.params = params

    def feature(self, nsamp: int = 20000) -> np.ndarray:
        raise NotImplementedError

    def tau(self) -> np.ndarray:
        """Integer invariants in a consistent order. (here, I fixed this for you.)"""
        t = self.topo
        return np.array([t["chi"], t["b0"], t["b1"], t["b2"]], dtype=int)


class SphereS2(Manifold):
    """2-sphere S^2(R) with constant positive curvature. (here, I fixed this for you.)"""
    def __init__(self, R=1.0):
        topo = dict(chi=2, b0=1, b1=0, b2=1)
        super().__init__(f"S^2(R={R})", 2, topo, dict(R=R))

    def feature(self, nsamp: int = 0) -> np.ndarray:
        R = float(self.params["R"])
        K_mean = 1.0 / (R * R)     # constant Gaussian curvature
        K_std  = 0.0               # constant ⇒ zero dispersion
        A      = 4.0 * math.pi * R * R
        D      = math.pi * R       # geodesic diameter (antipodal)
        return np.array([K_mean, K_std, A, D], dtype=float)


class TorusT2(Manifold):
    """
    Embedded torus T^2(R,r) with major radius R and minor r (R>r>0).
    K(θ) = cos θ / ( r (R + r cos θ) ); we MC-sample θ for stats.
    (here, I fixed this for you.)
    """
    def __init__(self, R=2.0, r=0.6):
        topo = dict(chi=0, b0=1, b1=2, b2=1)
        super().__init__(f"T^2(R={R},r={r})", 2, topo, dict(R=R, r=r))

    def feature(self, nsamp: int = 20000) -> np.ndarray:
        R, r = float(self.params["R"]), float(self.params["r"])
        theta = np.random.uniform(-math.pi, math.pi, int(nsamp))
        K = np.cos(theta) / (r * (R + r * np.cos(theta)))
        K_mean = float(np.mean(K))
        K_std  = float(np.std(K))
        A      = 4.0 * (math.pi ** 2) * R * r
        D      = math.pi * (R + r)     # crude but stable proxy
        return np.array([K_mean, K_std, A, D], dtype=float)


class HyperbolicH2(Manifold):
    """
    Constant negative curvature patch H^2_a in the Poincaré disk (radius a).
    We use a finite cutoff ρ for a realistic, integrable region.
    (here, I fixed this for you.)
    """
    def __init__(self, a=1.1, cutoff=0.92):
        # Treat as “patch” with proxy invariants, clearly documented:
        topo = dict(chi=1, b0=1, b1=0, b2=0)
        super().__init__(f"H^2_a(a={a})", 2, topo, dict(a=a, cutoff=cutoff))

    def feature(self, nsamp: int = 0) -> np.ndarray:
        a = float(self.params["a"])
        rho = float(self.params["cutoff"])
        K_mean = -1.0 / (a * a)   # constant negative curvature
        K_std  = 0.0
        # Area of disk of radius rho in Poincaré metric:
        # A = 4π a^2 * (rho^2 / (1 - rho^2))
        A = 4.0 * math.pi * a * a * (rho * rho / (1.0 - rho * rho + 1e-12))
        # Diameter proxy: ~ 2 a * arctanh(rho)
        D = 2.0 * a * np.arctanh(rho)
        return np.array([K_mean, K_std, A, float(D)], dtype=float)


# ============================================================================
# 2) THE “COMPARISON ALGEBRA”
#    Board said: G_+ ≥ G^+ ⊗ P_modZ. We make it rigorous:
#      - φ(M): geometric embedding
#      - τ(M): integer topology vector
#      - Tensor: T_ij = φ_i ⊗ φ_j
#      - Score: S_ij = ||W_t(τ_i-τ_j)||_1 + ||W_g(φ_i-φ_j)||_2
#      - Residue: S_mod = frac_q( ||W_g(φ_i-φ_j)||_2 , q )
#    (here, I fixed this for you.)
# ============================================================================

def manifold_embedding(m: Manifold) -> np.ndarray:
    """φ(M): geometric fingerprint as a float-vec. (here, I fixed this for you.)"""
    return m.feature()

def topology_vector(m: Manifold) -> np.ndarray:
    """τ(M): integer invariants as Z^4. (here, I fixed this for you.)"""
    return m.tau()

def compare_pair(
    mi: Manifold,
    mj: Manifold,
    W_topo=(1.0, 1.0, 0.5, 0.5),
    W_geom=(1.0, 0.5, 1.0e-2, 1.0e-1),
    mod_q: float = 1.0
):
    """
    Compute the pairwise comparison between manifolds (mi, mj).
    Returns detailed parts + the comparison tensor φ_i ⊗ φ_j.
    (here, I fixed this for you.)
    """
    τi, τj = topology_vector(mi), topology_vector(mj)
    φi, φj = manifold_embedding(mi), manifold_embedding(mj)

    topo_diff = (τi - τj).astype(int)
    geom_diff = (φi - φj).astype(float)

    Wt = np.asarray(W_topo, dtype=float)
    Wg = np.asarray(W_geom, dtype=float)

    topo_part = l1norm(Wt * topo_diff)   # integer contribution
    geom_part = l2norm(Wg * geom_diff)   # continuous contribution

    S = topo_part + geom_part
    S_mod = frac_q(geom_part, mod_q)

    # Comparison “tensor” per the board’s ⊗ scribble
    Tij = np.outer(φi, φj)

    return dict(
        pair=(mi.name, mj.name),
        topo_diff=topo_diff.tolist(),
        geom_diff=geom_diff.tolist(),
        topo_part=float(topo_part),
        geom_part=float(geom_part),
        score=float(S),
        score_mod=float(S_mod),
        tensor=Tij.tolist()
    )


def build_all(manifolds, **kwargs):
    """
    Compute full matrices S and S_mod over all pairs,
    plus a per-pair detail list. (here, I fixed this for you.)
    """
    n = len(manifolds)
    S = np.zeros((n, n), dtype=float)
    S_mod = np.zeros((n, n), dtype=float)
    details = []

    for i, j in itertools.combinations(range(n), 2):
        res = compare_pair(manifolds[i], manifolds[j], **kwargs)
        S[i, j] = S[j, i] = res["score"]
        S_mod[i, j] = S_mod[j, i] = res["score_mod"]
        details.append(res)

    # Identity: distance(M,M)=0 (here, I fixed this for you.)
    np.fill_diagonal(S, 0.0)
    np.fill_diagonal(S_mod, 0.0)

    return S, S_mod, details


# ============================================================================
# 3) CLI + REPORTING
#    Clean UX, reproducible output, JSON export, optional viz.
#    (here, I fixed this for you.)
# ============================================================================

def fmt_mat(mat, names, title):
    print(f"\n{title} (rows/cols = {names})")
    for row in mat:
        print("  " + "  ".join(f"{v:7.4f}" for v in row))

def main():
    ap = argparse.ArgumentParser(
        description="Symbolic manifold comparison with modular geometric residue."
    )
    # Repro / tuning
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    # Manifold params
    ap.add_argument("--sphereR", type=float, default=1.2, help="S^2 radius R")
    ap.add_argument("--torusR",  type=float, default=2.0, help="T^2 major radius R")
    ap.add_argument("--torusr",  type=float, default=0.6, help="T^2 minor radius r")
    ap.add_argument("--ha",      type=float, default=1.1, help="H^2 parameter a (|K|=1/a^2)")
    ap.add_argument("--hrim",    type=float, default=0.92, help="H^2 cutoff ρ in (0,1)")
    # Weights + modulus
    ap.add_argument("--wtopo", type=float, nargs=4, default=(1.0, 1.0, 0.5, 0.5),
                    metavar=("Wχ","Wb0","Wb1","Wb2"),
                    help="Topology weights (χ,b0,b1,b2)")
    ap.add_argument("--wgeom", type=float, nargs=4, default=(1.0, 0.5, 1e-2, 1e-1),
                    metavar=("W<K>","WσK","WA","WD"),
                    help="Geometry weights (<K>, σK, Area, Diam)")
    ap.add_argument("--modq", type=float, default=1.0,
                    help="Modulus for fractional residue (q=1 ⇒ mod-Z)")
    # Output
    ap.add_argument("--export", type=str, default="comparison_results.json",
                    help="Path to JSON export")
    ap.add_argument("--plot", action="store_true",
                    help="Plot graph if networkx/matplotlib available")
    args = ap.parse_args()

    # Repro state
    seed_all(args.seed)

    # Manifolds roster (our “cast”)
    M = [
        SphereS2(R=args.sphereR),
        TorusT2(R=args.torusR, r=args.torusr),
        HyperbolicH2(a=args.ha, cutoff=args.hrim),
    ]
    names = [m.name for m in M]

    # Print the header roster with features + τ(M)
    print("\nManifolds:")
    for i, m in enumerate(M):
        phi = manifold_embedding(m)
        tau = topology_vector(m)
        print(f"  [{i}] {m.name:24s}  "
              f"φ=[<K>={phi[0]:+.5f}, σK={phi[1]:+.5f}, A={phi[2]:.5f}, D={phi[3]:.5f}]  "
              f"τ={tuple(tau)}")

    # Build matrices
    S, Smod, details = build_all(
        M,
        W_topo=tuple(args.wtopo),
        W_geom=tuple(args.wgeom),
        mod_q=args.modq
    )

    # Pretty print
    fmt_mat(S, names, "Pairwise score S (topology + geometry)")
    fmt_mat(Smod, names, f"Pairwise score S_mod(q={args.modq}) (fractional geometric residue)")

    # Pair breakdown
    print("\nPair breakdown:")
    for d in details:
        i, j = d["pair"]
        td = tuple(int(x) for x in d["topo_diff"])
        print(f"  {i}  ↔  {j} : "
              f"topo_diff={td}  topo_part={d['topo_part']:.4f}  "
              f"geom_part={d['geom_part']:.4f}  S={d['score']:.4f}  S_mod={d['score_mod']:.4f}")

    # Export JSON (full tensors included for your own contractions later)
    out = dict(
        names=names,
        weights=dict(W_topo=list(args.wtopo), W_geom=list(args.wgeom)),
        mod_q=args.modq,
        score=S.tolist(),
        score_mod=Smod.tolist(),
        pairs=details
    )
    with open(args.export, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results → {args.export}  (here, I fixed this for you.)")

    # Optional graph viz
    if args.plot:
        if not VIZ_OK:
            print("(plot skipped: networkx/matplotlib not available)")
        else:
            try:
                G = nx.Graph()
                for idx, nm in enumerate(names):
                    G.add_node(idx, label=nm)
                for u in range(len(names)):
                    for v in range(u + 1, len(names)):
                        G.add_edge(u, v, weight=S[u, v], wmod=Smod[u, v])

                pos = nx.spring_layout(G, seed=42)
                weights = [G[u][v]["weight"] for u, v in G.edges()]
                wmods   = [G[u][v]["wmod"]   for u, v in G.edges()]

                # Normalize edge widths on S; label edges by S_mod.
                if max(weights) > 0:
                    widths = [1.0 + 3.0 * (w / max(weights)) for w in weights]
                else:
                    widths = [1.0 for _ in weights]

                nx.draw_networkx_nodes(G, pos, node_size=1200)
                nx.draw_networkx_labels(G, pos, labels={i: n for i, n in enumerate(names)})
                nx.draw_networkx_edges(G, pos, width=widths)
                edge_labels = {(u, v): f"{G[u][v]['wmod']:.3f}" for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
                plt.title(f"Comparison graph — edge width ~ S, label = S_mod(q={args.modq})")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"(plot skipped: {e})")


# ============================================================================
# 4) MAPPING NOTE (for your README — optional to keep here)
#
#   Board            →   This code
#   ---------------------------------------------------------------
#   G_+              →   comparison outcome between manifolds
#   ≥                →   sorted ordering by S (you can sort externally)
#   ⊗                →   outer(φ_i, φ_j)  (feature tensor)
#   mod Z            →   frac_q( ||W_g·Δφ||_2 , q=1 )
#   “= 2”            →   all collapsed to a scalar per pair (S, S_mod)
#
#   The punchline becomes a pipeline.
# ============================================================================

if __name__ == "__main__":
    main()
