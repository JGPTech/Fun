#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coupled logistic maps in 1D (vector) and on a product lattice (multi-register tensor)
Pure NumPy implementation with illustrative plots and comprehensive analysis tools.

Key models:
- Single-register (d sites): diffusive coupling via a row-stochastic weight matrix W
  x_i(n+1) = (1 - eps) f(x_i(n)) + eps * sum_j W_{ij} f(x_j(n))
  where f(x) = r * x * (1 - x) is the logistic map.

- Multi-register (R registers each of size dim): product-graph coupling where
  neighbors differ in exactly one coordinate according to the local adjacency.

Analysis tools:
- Synchrony error and entropy
- Spatial spectrum (FFT)
- Lyapunov proxy
- Return maps
- Various visualization options

Notes:
- Values are clipped into (0, 1) to keep the logistic map well-defined and numerically stable.
- No external dependencies beyond NumPy and Matplotlib (sklearn optional for PCA).
"""

import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ----------------------------
# Utilities: adjacency & norms
# ----------------------------

def make_chain_adj(dim, ring=False):
    """
    Build an undirected chain (or ring) adjacency matrix for 'dim' sites.
    - ring=False: each node connects to its immediate neighbors (i-1, i+1) when valid.
    - ring=True: additionally connect first and last nodes to form a cycle.
    Returns: (dim x dim) float array with 0/1 entries.
    """
    A = np.zeros((dim, dim), dtype=float)
    for i in range(dim - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    if ring and dim > 2:
        A[0, dim - 1] = A[dim - 1, 0] = 1.0
    return A


def normalize_rows(A):
    """
    Convert an adjacency matrix A into a row-stochastic matrix W.
    Rows that sum to zero are left as zero-rows (safe-guarded by setting sum to 1).
    """
    A = np.asarray(A, dtype=float)
    S = A.sum(axis=1, keepdims=True)
    S[S == 0] = 1.0
    return A / S


# ----------------------------
# Logistic nonlinearity
# ----------------------------

def logistic(x, r):
    """
    Logistic map f(x) = r * x * (1 - x), with typical r in [0, 4].
    For r in [3.6, 4], dynamics are chaotic in 1D (uncoupled) for typical initial x in (0,1).
    """
    return r * x * (1.0 - x)


# --------------------------------------------
# Single-register: coupled logistic on a graph
# --------------------------------------------

def coupled_logistic_1d(dim=50, N=1000, r=3.8, eps=0.2, ring=True,
                        init="random", A_custom=None, mode="post", clip=1e-12):
    """
    Diffusively coupled logistic maps on a 1D graph with 'dim' sites.

    Parameters
    - dim: number of sites (state dimension).
    - N: number of time steps to simulate.
    - r: logistic parameter (0..4).
    - eps: coupling strength in [0,1]. eps=0 => independent sites; eps->1 => strong averaging.
    - ring: whether to connect endpoints (True) or use open chain (False).
    - init: "random" for uniform random in (0,1); scalar to broadcast; or array of length dim.
    - A_custom: optional custom adjacency (dim x dim). If provided, overrides chain/ring.
    - mode: coupling scheme:
        * "post": x_{n+1} = (1-eps)*f(x) + eps*W f(x)  (diffusive coupling after nonlinearity)
        * "pre":  x_{n+1} = f( (1-eps)*x + eps*W x )   (mix before applying f)
    - clip: small positive value to keep x in (clip, 1-clip).

    Returns
    - X: array of shape (N, dim) with trajectories over time (time major).
    - W: row-stochastic weight matrix used for coupling.
    """
    # Build weight matrix W from adjacency
    A = make_chain_adj(dim, ring=ring) if A_custom is None else np.asarray(A_custom, float)
    W = normalize_rows(A)

    # Initialize x(0)
    X = np.zeros((N, dim), dtype=float)
    if init == "random":
        x0 = np.random.rand(dim)
    elif np.isscalar(init):
        x0 = np.full(dim, float(init))
    else:
        x0 = np.asarray(init, dtype=float)
        if x0.shape != (dim,):
            raise ValueError("init array must have shape (dim,)")
    # Keep strictly inside (0,1) for numerical stability
    X[0] = np.clip(x0, clip, 1 - clip)

    # Time stepping
    for n in range(1, N):
        prev = X[n - 1]
        if mode == "post":
            fprev = logistic(prev, r)
            X[n] = (1.0 - eps) * fprev + eps * (W @ fprev)
        elif mode == "pre":
            mixed = (1.0 - eps) * prev + eps * (W @ prev)
            X[n] = logistic(mixed, r)
        else:
            raise ValueError("mode must be 'post' or 'pre'")
        X[n] = np.clip(X[n], clip, 1 - clip)
    return X, W


# ---------------------------------------------------------
# Multi-register: product-graph coupled logistic maps (R>1)
# ---------------------------------------------------------

def build_index_maps(dim, num_registers):
    """
    Build maps between tuple indices (k1,...,kR) and flat indices 0..dim^R-1.
    Returns
    - t2f: dict mapping tuple -> flat index
    - f2t: list where f2t[i] is the tuple index of flat i
    """
    tuples = list(product(*[range(dim) for _ in range(num_registers)]))
    t2f = {t: i for i, t in enumerate(tuples)}
    return t2f, tuples


def coupled_logistic_multi(dim=6, num_registers=2, N=1500, r=3.8, eps=0.15,
                           ring=True, init="random", clip=1e-12):
    """
    Diffusively coupled logistic maps defined on a product graph of R registers.
    - Each register has 'dim' states (0..dim-1); total sites = dim^R.
    - Neighbors differ by one coordinate, moving along the local adjacency (chain/ring).
    - We apply 'post' diffusive coupling:
        x_s(n+1) = (1 - eps) f(x_s(n)) + eps * sum_{t in N(s)} W_{s,t} * f(x_t(n))

    Parameters
    - dim: number of states per register.
    - num_registers: number of registers R (tensor order).
    - N: time steps.
    - r: logistic parameter.
    - eps: coupling strength in [0,1].
    - ring: whether each register uses ring or chain adjacency.
    - init: "random", scalar, or array of length dim^R for x(0).
    - clip: bounds to keep values inside (clip, 1-clip).

    Returns
    - X: array (N, dim^R) trajectories (flat indexing).
    - f2t: list mapping flat index -> tuple index (k1,...,kR).
    """
    # Local adjacency on each register
    A_local = make_chain_adj(dim, ring=ring)
    # Precompute neighbor lists and degrees per coordinate value
    rows = [np.nonzero(A_local[i])[0] for i in range(dim)]
    deg = np.array([len(rw) for rw in rows], dtype=float)
    deg[deg == 0] = 1.0  # avoid divide-by-zero

    # Index maps for product graph
    t2f, f2t = build_index_maps(dim, num_registers)
    total = dim ** num_registers

    # Initialize
    X = np.zeros((N, total), dtype=float)
    if init == "random":
        x0 = np.random.rand(total)
    elif np.isscalar(init):
        x0 = np.full(total, float(init))
    else:
        x0 = np.asarray(init, dtype=float)
        if x0.shape != (total,):
            raise ValueError(f"init array must have shape ({total},)")
    X[0] = np.clip(x0, clip, 1 - clip)

    # Time stepping with neighbor-averaged diffusion after f
    for n in range(1, N):
        prev = X[n - 1]
        fprev = logistic(prev, r)

        next_vals = np.empty_like(prev)
        for flat_idx, idx_tuple in enumerate(f2t):
            # Self contribution after nonlinearity
            self_term = (1.0 - eps) * fprev[flat_idx]

            # Neighbor aggregation across all single-coordinate moves
            agg = 0.0
            for reg in range(num_registers):
                i = idx_tuple[reg]
                neighs = rows[i]              # neighbors of coordinate i in this register
                if len(neighs) == 0:
                    continue
                w_each = 1.0 / deg[i]         # row-stochastic per coordinate
                for j in neighs:
                    nb_tuple = list(idx_tuple)
                    nb_tuple[reg] = j         # move in exactly one coordinate
                    nb_flat = t2f[tuple(nb_tuple)]
                    agg += w_each * fprev[nb_flat]

            nb_term = eps * agg
            next_vals[flat_idx] = self_term + nb_term

        X[n] = np.clip(next_vals, clip, 1 - clip)

    return X, f2t


# ----------------------------
# Analysis Functions
# ----------------------------

def spatial_std_and_entropy(X, bins=50):
    """
    Compute spatial standard deviation and entropy over time.
    
    Parameters:
    - X: shape (N, dim) - time series of spatial states
    - bins: number of bins for entropy calculation
    
    Returns:
    - m: mean across sites at each time
    - s: standard deviation across sites at each time
    - H: entropy across sites at each time
    """
    m = X.mean(axis=1)
    s = X.std(axis=1)
    H = []
    for row in X:
        hist, _ = np.histogram(row, bins=bins, range=(0,1), density=True)
        p = hist + 1e-12
        p /= p.sum()
        H.append(-(p*np.log(p)).sum())
    return m, s, np.array(H)


def spatial_spectrum(X):
    """
    Compute spatial power spectrum averaged over time.
    
    Parameters:
    - X: shape (N, dim)
    
    Returns:
    - freqs: spatial frequencies
    - S: power spectrum
    """
    F = np.fft.rfft(X - X.mean(axis=1, keepdims=True), axis=1)
    S = (np.abs(F) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(X.shape[1], d=1)
    return freqs, S


def lyapunov_proxy(X, r):
    """
    Compute instantaneous Lyapunov exponent proxy.
    
    Parameters:
    - X: shape (N, dim)
    - r: logistic parameter
    
    Returns:
    - lam: Lyapunov proxy at each time
    """
    lam = np.mean(np.log(np.abs(r * (1 - 2*X)) + 1e-12), axis=1)
    return lam


def return_map_points(X, step=1, sample_sites=20):
    """
    Extract points for return map visualization.
    
    Parameters:
    - X: shape (N, dim)
    - step: time delay
    - sample_sites: number of sites to sample
    
    Returns:
    - x, y: coordinates for scatter plot
    """
    N, dim = X.shape
    idx = np.random.choice(dim, size=min(sample_sites, dim), replace=False)
    x = X[:-step, :][:, idx].ravel()
    y = X[step:, :][:, idx].ravel()
    return x, y


def kuramoto_order_parameter(X):
    """
    Compute Kuramoto-style order parameter for phase coherence.
    
    Parameters:
    - X: shape (N, dim) with values in [0,1]
    
    Returns:
    - R: order parameter magnitude at each time
    """
    phases = 2 * np.pi * X
    z = np.mean(np.exp(1j * phases), axis=1)
    return np.abs(z)


def slice_image(X_flat, f2t, dim, regs=(0,1)):
    """
    For multi-register systems with R=2: create 2D slices.
    
    Parameters:
    - X_flat: shape (N, dim^R)
    - f2t: flat-to-tuple index mapping
    - dim: size of each register
    - regs: which two registers to visualize
    
    Returns:
    - img_seq: shape (N, dim, dim) image sequence
    """
    N, total = X_flat.shape
    img_seq = np.zeros((N, dim, dim))
    other = [r for r in range(len(f2t[0])) if r not in regs]
    
    for n in range(N):
        acc = np.zeros((dim, dim))
        cnt = np.zeros((dim, dim))
        for flat, tup in enumerate(f2t):
            i, j = tup[regs[0]], tup[regs[1]]
            acc[i, j] += X_flat[n, flat]
            cnt[i, j] += 1
        img_seq[n] = acc / np.maximum(cnt, 1)
    return img_seq


def time_pca(X, n_components=3):
    """
    PCA analysis of trajectory in state space.
    
    Parameters:
    - X: shape (N, D) with D = dim or dim^R
    - n_components: number of principal components
    
    Returns:
    - pcs: projected coordinates (N, n_components)
    """
    try:
        from sklearn.decomposition import PCA
        Xc = X - X.mean(axis=0, keepdims=True)
        pcs = PCA(n_components=n_components).fit_transform(Xc)
        return pcs
    except ImportError:
        # Fallback to manual PCA if sklearn not available
        Xc = X - X.mean(axis=0, keepdims=True)
        C = Xc.T @ Xc / (X.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = eigvals.argsort()[::-1][:n_components]
        return Xc @ eigvecs[:, idx]


# ----------------------------
# Visualization Functions
# ----------------------------

def plot_1d_analysis(X, r, eps, ring=True):
    """
    Comprehensive analysis dashboard for 1D coupled logistic maps.
    """
    N, dim = X.shape
    t = np.arange(N)
    
    # Compute all diagnostics
    m, s, H = spatial_std_and_entropy(X)
    freqs, S = spatial_spectrum(X)
    lam = lyapunov_proxy(X, r)
    R = kuramoto_order_parameter(X)
    x_ret, y_ret = return_map_points(X, step=1, sample_sites=10)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Space-time plot
    ax1 = fig.add_subplot(gs[0, :2])
    im = ax1.imshow(X.T, aspect='auto', cmap='turbo', origin='lower',
                    extent=[0, N-1, 0, dim-1])
    ax1.set_xlabel('Time n')
    ax1.set_ylabel('Site i')
    ax1.set_title(f'Space-Time Evolution (r={r}, eps={eps}, {"ring" if ring else "chain"})')
    plt.colorbar(im, ax=ax1, label='x')
    
    # 2. Synchrony measures
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(t, s, 'b-', label='Spatial STD')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('STD', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, R, 'r-', label='Order param')
    ax2_twin.set_ylabel('Order Parameter', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Synchronization Measures')
    ax2.grid(True, alpha=0.3)
    
    # 3. Entropy and Lyapunov
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, H, 'g-', label='Entropy')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Entropy', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, lam, 'orange', label='Lyapunov proxy')
    ax3_twin.set_ylabel('Lyapunov Proxy', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3_twin.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax3.set_title('Entropy & Chaos Measures')
    ax3.grid(True, alpha=0.3)
    
    # 4. Spatial spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(freqs * dim, S, 'k-', linewidth=2)
    ax4.set_xlabel('Spatial frequency (k)')
    ax4.set_ylabel('Power |F_k|²')
    ax4.set_title('Spatial Power Spectrum (time avg)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Return map
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(x_ret, y_ret, s=1, alpha=0.5, c='blue')
    x_theory = np.linspace(0, 1, 100)
    ax5.plot(x_theory, logistic(x_theory, r), 'r-', linewidth=2, label=f'y=rx(1-x)')
    ax5.set_xlabel('x(n)')
    ax5.set_ylabel('x(n+1)')
    ax5.set_title('Return Map')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Binary threshold pattern
    ax6 = fig.add_subplot(gs[2, 0])
    B = (X > 0.5).astype(float)
    ax6.imshow(B.T, aspect='auto', cmap='gray', origin='lower',
               extent=[0, N-1, 0, dim-1])
    ax6.set_xlabel('Time n')
    ax6.set_ylabel('Site i')
    ax6.set_title('Binary Pattern (x > 0.5)')
    
    # 7. Time series samples
    ax7 = fig.add_subplot(gs[2, 1])
    sample_sites = np.linspace(0, dim-1, 5, dtype=int)
    for i in sample_sites:
        ax7.plot(t[-200:], X[-200:, i], alpha=0.7, label=f'Site {i}')
    ax7.set_xlabel('Time')
    ax7.set_ylabel('x')
    ax7.set_title('Sample Time Series (last 200 steps)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 8. Phase space (if PCA available)
    ax8 = fig.add_subplot(gs[2, 2])
    try:
        pcs = time_pca(X, n_components=2)
        ax8.plot(pcs[:, 0], pcs[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        ax8.scatter(pcs[0, 0], pcs[0, 1], c='green', s=100, marker='o', label='Start')
        ax8.scatter(pcs[-1, 0], pcs[-1, 1], c='red', s=100, marker='s', label='End')
        ax8.set_xlabel('PC1')
        ax8.set_ylabel('PC2')
        ax8.set_title('Phase Space Trajectory (PCA)')
        ax8.legend()
    except:
        ax8.text(0.5, 0.5, 'PCA not available', ha='center', va='center', 
                transform=ax8.transAxes)
        ax8.set_title('Phase Space (N/A)')
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f'Coupled Logistic Map Analysis: dim={dim}, r={r}, eps={eps}', 
                 fontsize=16)
    return fig


def plot_multi_analysis(X, f2t, dim, num_registers, r, eps):
    """
    Analysis dashboard for multi-register systems.
    """
    N, total = X.shape
    t = np.arange(N)
    
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Register marginals
    ax1 = fig.add_subplot(gs[0, 0])
    for k in range(min(dim, 5)):  # Limit to 5 states for clarity
        mask = [i for i, tup in enumerate(f2t) if tup[0] == k]
        series = X[:, mask].mean(axis=1)
        ax1.plot(t, series, label=f'State {k}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean x')
    ax1.set_title('Register 0 Marginals')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Synchrony measures
    ax2 = fig.add_subplot(gs[0, 1])
    m, s, H = spatial_std_and_entropy(X)
    ax2.plot(t, s, 'b-', label='Spatial STD')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('STD', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, H, 'g-', label='Entropy')
    ax2_twin.set_ylabel('Entropy', color='g')
    ax2_twin.tick_params(axis='y', labelcolor='g')
    ax2.set_title('Global Synchrony & Entropy')
    ax2.grid(True, alpha=0.3)
    
    # 3. Lyapunov proxy
    ax3 = fig.add_subplot(gs[0, 2])
    lam = lyapunov_proxy(X, r)
    ax3.plot(t, lam, 'orange', linewidth=2)
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax3.axhline(lam.mean(), color='r', linestyle=':', alpha=0.7, 
                label=f'Mean = {lam.mean():.3f}')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Lyapunov Proxy')
    ax3.set_title('Chaos Indicator')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 2D slice visualization (if R=2)
    ax4 = fig.add_subplot(gs[1, 0])
    if num_registers == 2:
        img_seq = slice_image(X, f2t, dim)
        # Show snapshot at multiple times
        snapshot_times = [0, N//4, N//2, 3*N//4, N-1]
        combined = np.hstack([img_seq[t] for t in snapshot_times])
        im = ax4.imshow(combined, cmap='turbo', aspect='auto')
        ax4.set_title('2D Slices at Different Times')
        ax4.set_xlabel('Time progression →')
        ax4.set_ylabel('Register 1')
        plt.colorbar(im, ax=ax4, label='x')
    else:
        ax4.text(0.5, 0.5, f'2D slice not shown\n(R={num_registers})', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('2D Visualization (N/A)')
    
    # 5. Return map
    ax5 = fig.add_subplot(gs[1, 1])
    x_ret, y_ret = return_map_points(X, step=1, sample_sites=20)
    ax5.scatter(x_ret, y_ret, s=1, alpha=0.3, c='blue')
    x_theory = np.linspace(0, 1, 100)
    ax5.plot(x_theory, logistic(x_theory, r), 'r-', linewidth=2)
    ax5.set_xlabel('x(n)')
    ax5.set_ylabel('x(n+1)')
    ax5.set_title('Global Return Map')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # 6. Phase space
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        pcs = time_pca(X, n_components=2)
        ax6.plot(pcs[:, 0], pcs[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        ax6.scatter(pcs[0, 0], pcs[0, 1], c='green', s=100, marker='o', label='Start')
        ax6.scatter(pcs[-1, 0], pcs[-1, 1], c='red', s=100, marker='s', label='End')
        ax6.set_xlabel('PC1')
        ax6.set_ylabel('PC2')
        ax6.set_title(f'Phase Space (dim^R = {total})')
        ax6.legend()
    except:
        ax6.text(0.5, 0.5, 'PCA not available', ha='center', va='center', 
                transform=ax6.transAxes)
        ax6.set_title('Phase Space (N/A)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-Register Analysis: {num_registers} registers × {dim} states, r={r}, eps={eps}', 
                 fontsize=16)
    return fig


# ---------------
# Demo Functions
# ---------------

def demo_1d_with_analysis():
    """
    Demo: 1D ring with comprehensive analysis dashboard.
    """
    dim = 100
    N = 800
    r = 3.8
    eps = 0.2
    
    print(f"Running 1D simulation: dim={dim}, N={N}, r={r}, eps={eps}")
    X, W = coupled_logistic_1d(dim=dim, N=N, r=r, eps=eps, ring=True, mode="post", init="random")
    
    fig = plot_1d_analysis(X, r, eps, ring=True)
    plt.show()
    
    return X, W


def demo_multi_with_analysis():
    """
    Demo: Multi-register system with analysis.
    """
    dim = 5
    R = 2
    N = 1200
    r = 3.8
    eps = 0.15
    
    print(f"Running multi-register simulation: R={R}, dim={dim}, N={N}, r={r}, eps={eps}")
    X, f2t = coupled_logistic_multi(dim=dim, num_registers=R, N=N, r=r, eps=eps, ring=True, init="random")
    
    fig = plot_multi_analysis(X, f2t, dim, R, r, eps)
    plt.show()
    
    return X, f2t


def parameter_sweep_1d(r_values=[3.6, 3.8, 4.0], eps_values=[0.0, 0.1, 0.2, 0.3]):
    """
    Parameter sweep showing different dynamical regimes.
    """
    dim = 50
    N = 500
    
    fig, axes = plt.subplots(len(r_values), len(eps_values), 
                            figsize=(12, 8), sharex=True, sharey=True)
    
    for i, r in enumerate(r_values):
        for j, eps in enumerate(eps_values):
            X, _ = coupled_logistic_1d(dim=dim, N=N, r=r, eps=eps, ring=True)
            
            ax = axes[i, j] if len(r_values) > 1 else axes[j]
            im = ax.imshow(X.T, aspect='auto', cmap='turbo', origin='lower')
            
            if i == 0:
                ax.set_title(f'ε = {eps}')
            if j == 0:
                ax.set_ylabel(f'r = {r}\nSite')
            if i == len(r_values) - 1:
                ax.set_xlabel('Time')
    
    plt.suptitle('Parameter Sweep: Coupled Logistic Maps', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Choose which demo to run
    print("Coupled Logistic Maps with Analysis Tools")
    print("=========================================")
    print("1. Run 1D demo with full analysis")
    print("2. Run multi-register demo with analysis")
    print("3. Run parameter sweep")
    print("4. Run all demos")
    
    choice = input("Enter choice (1-4) [default=1]: ").strip() or "1"
    
    if choice == "1":
        demo_1d_with_analysis()
    elif choice == "2":
        demo_multi_with_analysis()
    elif choice == "3":
        parameter_sweep_1d()
    elif choice == "4":
        demo_1d_with_analysis()
        demo_multi_with_analysis()
        parameter_sweep_1d()
    else:
        print("Invalid choice. Running default demo.")
        demo_1d_with_analysis()