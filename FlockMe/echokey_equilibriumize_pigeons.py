#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EchoKey Equilibriumization for Pigeon Flocks (Budapest/Oxford Movebank data)
----------------------------------------------------------------------------
What this script does:
  1) Load Movebank-style CSV(s)
  2) Select a session/flight window with overlapping birds
  3) Resample without extrapolation; build per-bird features
  4) Build EchoKey state Ψ (cyclicity, recursion, fractality, synergy, outliers)
  5) Refraction map (scale-wise transform) + time rescale α(Ξ)
  6) Discretize Ξ -> τ-weighted transition counts -> reversible projection (M*, π*)
  7) Gibbs energy U = -T_eff log π*, witness tests (EP≈0, cycle affinities≈0)
  8) Map ∇U back to leadership scores (−∂U/∂align_front_i), rank birds
Artifacts are written into --outdir.

Usage (example):
  python echokey_equilibriumize_pigeons.py \
      --csv flights.csv --outdir results/pigeon_run \
      --bird-col "individual-local-identifier" \
      --time-col "timestamp" \
      --lat-col "location-lat" --lon-col "location-long" \
      --session-col comments --session "homing flight 1"
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# --------------------------- Tunable hyperparameters ---------------------------

@dataclass
class Tune:
    # Sampling & smoothing
    target_hz: float = 5.0
    vel_savgol_win: int = 9
    vel_savgol_poly: int = 3

    # Refraction Ξ = Ψ ⊙ (1 + μ L η)
    L_star: int = 1
    mu_refraction: float = 0.12
    eps_frac_grad: float = 1e-6

    # Synergy kernel κ_ij = sigmoid(β0 − β1*dist − β2*ang)
    beta0: float = 6.0
    beta1: float = 0.12  # per meter
    beta2: float = 4.0   # per radian
    kappa_scale: float = 1.0

    # Outliers
    outlier_z_thresh: float = 3.0

    # Time rescale α = c1*α_C + c2*<κ_ij>
    alpha_c1: float = 0.7
    alpha_c2: float = 0.3
    alpha_floor: float = 1e-3

    # Discretization
    pca_dim: int = 12
    n_clusters: int = 80
    kmeans_batch: int = 4096
    random_state: int = 42

    # Gibbs temperature (scale)
    T_eff: float = 1.0

    # Leadership gradient estimation
    neigh_k: int = 8

    # Witness tolerances
    ep_ratio_warn: float = 0.25
    cycle_affinity_tol: float = 1e-3

TUNE = Tune()

R_EARTH = 6371000.0  # meters

# --------------------------- Helpers ------------------------------------------

def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat0 = np.nanmean(lat) * math.pi / 180.0
    lon0 = np.nanmean(lon) * math.pi / 180.0
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    x = (lon_r - lon0) * math.cos(lat0) * R_EARTH
    y = (lat_r - lat0) * R_EARTH
    return x, y

def unwrap_angle(a: np.ndarray) -> np.ndarray:
    return np.unwrap(a)

def angle_of(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.arctan2(vy, vx)

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def _ensure_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert('UTC') if ts.tzinfo is not None else ts.tz_localize('UTC')

# --------------------------- Load ---------------------------------------------

def load_movebank_csvs(paths: List[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)

# --------------------------- Session detection --------------------------------

def normalize_time(df: pd.DataFrame, time_col: str) -> pd.Series:
    # Parse as UTC-aware
    return pd.to_datetime(df[time_col], utc=True, errors='coerce')

def _best_overlap_window(df: pd.DataFrame, bird_col: str):
    """Find longest 1 Hz run with high concurrent bird count; return ((t0,t1), birds)."""
    if df.empty:
        return None
    ts_floor = normalize_time(df, '_t').dt.floor('s') if '_t' in df.columns else normalize_time(df, 'timestamp').dt.floor('s')
    counts = df.groupby(ts_floor)[[bird_col]].agg(lambda s: len(set(s)))
    if counts.empty:
        return None
    thr = max(3, int(np.quantile(counts[bird_col].to_numpy(), 0.75)))
    mask = (counts[bird_col] >= thr).to_numpy()
    times = counts.index.to_numpy()

    best_i = 0; best_len = 0; i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1; continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        if (j - i) > best_len:
            best_i, best_len = i, (j - i)
        i = j
    if best_len == 0:
        return None

    t0 = _ensure_utc(times[best_i])
    t1 = _ensure_utc(times[best_i + best_len - 1])
    inside = df[(normalize_time(df, '_t') if '_t' in df.columns else normalize_time(df, 'timestamp')).between(t0, t1)]
    birds = inside[bird_col].dropna().unique().tolist()
    return (t0, t1), birds

def select_session(df: pd.DataFrame,
                   bird_col: str,
                   time_col: str,
                   session_col: Optional[str],
                   session_value: Optional[str],
                   min_birds: int,
                   min_duration_s: int) -> pd.DataFrame:
    df = df.copy()
    df['_t'] = normalize_time(df, time_col)

    if session_value and session_col and session_col in df.columns:
        sub = df[df[session_col].astype(str) == str(session_value)].dropna(subset=['_t', bird_col])
        if sub.empty:
            raise ValueError(f"No rows match {session_col} == {session_value}")
        return sub

    if session_col and session_col in df.columns:
        best = None; best_len = -1
        for val, g in df.groupby(session_col):
            g = g.dropna(subset=['_t', bird_col])
            if g.empty: continue
            win = _best_overlap_window(g, bird_col)
            if win is None: continue
            (t0, t1), birds = win
            length = (t1 - t0).total_seconds()
            if len(birds) >= min_birds and length >= min_duration_s and length > best_len:
                best = (val, t0, t1, birds); best_len = length
        if best:
            val, t0, t1, birds = best
            sel = df[(df[session_col] == val) & (df['_t'].between(t0, t1)) & (df[bird_col].isin(birds))].copy()
            print(f"[session] selected by '{session_col}': {val}  window={t0}→{t1}  birds={sorted(set(birds))}")
            return sel

    # Fallback: global scan
    win = _best_overlap_window(df.dropna(subset=['_t', bird_col]), bird_col)
    if win is None:
        raise ValueError("Could not find any overlapping window with multiple birds.")
    (t0, t1), birds = win
    if len(birds) < min_birds or (t1 - t0).total_seconds() < min_duration_s:
        raise ValueError(f"Longest overlap too small: birds={len(birds)}, duration={(t1 - t0)}s")
    sel = df[(df['_t'].between(t0, t1)) & (df[bird_col].isin(birds))].copy()
    print(f"[session] auto-selected window: {t0}→{t1}  birds={sorted(set(birds))}")
    return sel

# --------------------------- Synchronize & features ----------------------------

def synchronize_and_features(df: pd.DataFrame,
                             bird_col: str,
                             time_col: str,
                             lat_col: str,
                             lon_col: str,
                             tune: Tune,
                             min_steps: int = 50,
                             min_coverage: float = 0.8) -> Tuple[pd.DataFrame, Dict]:
    birds = sorted(df[bird_col].dropna().unique())
    dfs = {}
    for b in birds:
        d = df[df[bird_col] == b].copy()
        t = normalize_time(d, time_col)
        d = d.assign(_t=t).dropna(subset=['_t', lat_col, lon_col])
        d = d.sort_values('_t')
        if len(d) < 5:
            continue
        x, y = latlon_to_local_xy(d[lat_col].to_numpy(), d[lon_col].to_numpy())
        d['_x'] = x; d['_y'] = y
        dfs[b] = d
    if len(dfs) < 2:
        raise ValueError("Need at least two birds with valid tracks.")

    # Overlap bounds
    t0 = max(v['_t'].iloc[0] for v in dfs.values())
    t1 = min(v['_t'].iloc[-1] for v in dfs.values())
    if t1 <= t0:
        raise ValueError("No temporal overlap across selected birds.")

    dt = 1.0 / tune.target_hz
    grid = pd.date_range(t0, t1, freq=pd.Timedelta(seconds=dt), inclusive='both', tz='UTC')
    grid_s = grid.astype('int64') / 1e9  # seconds since epoch

    per = {}
    val_masks = {}
    for b, d in dfs.items():
        ts = d['_t'].astype('int64') / 1e9
        xi = np.interp(grid_s, ts, d['_x'].to_numpy(), left=np.nan, right=np.nan)
        yi = np.interp(grid_s, ts, d['_y'].to_numpy(), left=np.nan, right=np.nan)
        valid = (grid_s >= ts.min()) & (grid_s <= ts.max())

        vx = np.gradient(xi, dt); vy = np.gradient(yi, dt)
        if len(vx) >= tune.vel_savgol_win:
            vx = savgol_filter(vx, tune.vel_savgol_win, tune.vel_savgol_poly, mode='interp')
            vy = savgol_filter(vy, tune.vel_savgol_win, tune.vel_savgol_poly, mode='interp')
        sp = np.hypot(vx, vy)
        theta = unwrap_angle(angle_of(vx, vy))
        omega = np.gradient(theta, dt)
        ax = np.gradient(vx, dt); ay = np.gradient(vy, dt)
        curvature = np.abs(vx * ay - vy * ax) / np.maximum(sp**3, 1e-9)

        per[b] = pd.DataFrame({
            't': grid_s, 'x': xi, 'y': yi, 'vx': vx, 'vy': vy, 'speed': sp,
            'theta': theta, 'omega': omega, 'ax': ax, 'ay': ay, 'curvature': curvature
        })
        val_masks[b] = valid

    valid_all = np.ones(len(grid_s), dtype=bool)
    for b in birds:
        if b not in val_masks:
            valid_all &= False
        else:
            valid_all &= val_masks[b]
    if valid_all.sum() < min_steps:
        raise ValueError("Not enough overlapping time between birds for group analysis.")

    idx = np.where(valid_all)[0]
    i0, i1 = idx[0], idx[-1]
    for b in birds:
        per[b] = per[b].iloc[i0:i1+1].reset_index(drop=True)

    # Filter birds with low coverage in this window
    good_birds = []
    for b in birds:
        d = dfs[b]
        inside = d['_t'].between(grid[i0], grid[i1])
        cov = inside.mean() if len(d) else 0.0
        if cov >= min_coverage:
            good_birds.append(b)
    if len(good_birds) < 2:
        raise ValueError("After coverage filtering, <2 birds remain.")

    birds = sorted(good_birds)
    # Build group frame
    rows = []
    nT = len(per[birds[0]])
    for t_idx in range(nT):
        row = {'t': per[birds[0]]['t'].iat[t_idx]}
        for b in birds:
            d = per[b]
            for k in ['x','y','vx','vy','speed','theta','omega','ax','ay','curvature']:
                row[f'{b}:{k}'] = d[k].iat[t_idx]
        rows.append(row)
    G = pd.DataFrame(rows)

    vx_cols = [f'{b}:vx' for b in birds]
    vy_cols = [f'{b}:vy' for b in birds]
    G['vx_group'] = G[vx_cols].mean(axis=1)
    G['vy_group'] = G[vy_cols].mean(axis=1)
    norm = np.hypot(G['vx_group'], G['vy_group']) + 1e-12
    G['ex'] = G['vx_group'] / norm
    G['ey'] = G['vy_group'] / norm

    meta = {'birds': birds, 'dt': dt, 'n_steps': len(G)}
    return G, meta

# --------------------------- EchoKey state -------------------------------------

def build_echokey_state(G: pd.DataFrame, meta: Dict, tune: Tune) -> Tuple[np.ndarray, Dict]:
    birds = meta['birds']
    nT = meta['n_steps']
    ex = G['ex'].to_numpy(); ey = G['ey'].to_numpy()

    feats = []; feat_names = []
    for b in birds:
        theta = G[f'{b}:theta'].to_numpy()
        omega = G[f'{b}:omega'].to_numpy()
        speed = G[f'{b}:speed'].to_numpy()
        curv  = G[f'{b}:curvature'].to_numpy()
        vx = G[f'{b}:vx'].to_numpy(); vy = G[f'{b}:vy'].to_numpy()
        sp = np.maximum(np.hypot(vx, vy), 1e-9)
        hx = vx / sp; hy = vy / sp
        align_front = hx * ex + hy * ey

        # local density + alignment similarity
        x = G[f'{b}:x'].to_numpy(); y = G[f'{b}:y'].to_numpy()
        sigma = 8.0
        dens = np.zeros(nT); align_group = np.zeros(nT)
        for t in range(nT):
            terms = []; alns = []
            for b2 in birds:
                if b2 == b: continue
                dx = G[f'{b2}:x'].iat[t] - x[t]
                dy = G[f'{b2}:y'].iat[t] - y[t]
                d2 = dx*dx + dy*dy
                terms.append(np.exp(-d2 / (2.0 * sigma * sigma)))
                vx2 = G[f'{b2}:vx'].iat[t]; vy2 = G[f'{b2}:vy'].iat[t]
                sp2 = max(math.hypot(vx2, vy2), 1e-9)
                hx2, hy2 = vx2 / sp2, vy2 / sp2
                alns.append(hx*hx2 + hy*hy2)
            dens[t] = np.sum(terms) if terms else 0.0
            align_group[t] = np.mean(alns) if alns else 0.0

        Fi = np.stack([theta, omega, speed, curv, align_front, dens, align_group], axis=1)
        feats.append(Fi)
        feat_names += [f'{b}:theta', f'{b}:omega', f'{b}:speed', f'{b}:curvature',
                       f'{b}:align_front', f'{b}:density', f'{b}:align_group']
    Psi_base = np.concatenate(feats, axis=1)  # (nT, N*7)

    # Recursion residual (linear one-step)
    X = Psi_base[:-1, :]; Y = Psi_base[1:, :]
    lam = 1e-3
    A = np.linalg.pinv(X.T @ X + lam*np.eye(X.shape[1])) @ X.T @ Y
    Y_hat = X @ A
    e = np.linalg.norm(Y - Y_hat, axis=1)
    eps_R = np.zeros((Psi_base.shape[0],)); eps_R[1:] = (e - e.mean()) / (e.std() + 1e-9)

    # Fractal slope per channel (constant across time)
    ws = np.array([5, 9, 17, 33])
    slopes = []
    for j in range(Psi_base.shape[1]):
        v = Psi_base[:, j]
        Vs = []
        for w in ws:
            if len(v) < w + 1:
                Vs.append(np.nan)
            else:
                mv = pd.Series(v).rolling(w, center=True).var().to_numpy()
                Vs.append(np.nanmean(mv))
        V = np.array(Vs, dtype=float)
        m = np.isfinite(V) & (V > 0)
        s = np.polyfit(np.log(1.0 / (ws[m])), np.log(V[m]), 1)[0] if m.sum() >= 2 else 0.0
        slopes.append(s)
    slopes = np.array(slopes)
    eta = (slopes - slopes.mean()) / (slopes.std() + tune.eps_frac_grad)
    eta = np.clip(eta, -5.0, 5.0)
    ETA = np.broadcast_to(eta.reshape(1, -1), Psi_base.shape)  # (nT, N*7)

    # Outliers from tangential accel + curvature
    tang_acc_all, curv_all = [], []
    for b in birds:
        ax = G[f'{b}:ax'].to_numpy(); ay = G[f'{b}:ay'].to_numpy()
        vx = G[f'{b}:vx'].to_numpy(); vy = G[f'{b}:vy'].to_numpy()
        sp = np.maximum(np.hypot(vx, vy), 1e-9)
        hx = vx / sp; hy = vy / sp
        tang = ax*hx + ay*hy
        tang_acc_all.append(tang); curv_all.append(G[f'{b}:curvature'].to_numpy())
    tang_acc = np.mean(np.stack(tang_acc_all, axis=1), axis=1)
    curv_avg = np.mean(np.stack(curv_all, axis=1), axis=1)
    z_tang = robust_z(tang_acc); z_curv = robust_z(curv_avg)
    O = (np.maximum(np.abs(z_tang), np.abs(z_curv)) > tune.outlier_z_thresh).astype(float)

    # Synergy intensity κ̄
    def ang(a, b):
        d = np.abs(((a - b + np.pi) % (2*np.pi)) - np.pi)
        return d
    thetas = np.stack([G[f'{b}:theta'].to_numpy() for b in birds], axis=1)
    xs = np.stack([G[f'{b}:x'].to_numpy() for b in birds], axis=1)
    ys = np.stack([G[f'{b}:y'].to_numpy() for b in birds], axis=1)
    kappa_mean = np.zeros(nT)
    N = len(birds)
    for t in range(nT):
        kap_sum = 0.0; cnt = 0
        for i in range(N):
            for j in range(N):
                if i == j: continue
                dx = xs[t, j] - xs[t, i]; dy = ys[t, j] - ys[t, i]
                dist = math.hypot(dx, dy)
                aij = ang(thetas[t, i], thetas[t, j])
                z = TUNE.beta0 - TUNE.beta1 * dist - TUNE.beta2 * aij
                kap_sum += sigmoid(z); cnt += 1
        kappa_mean[t] = (kap_sum / max(cnt, 1)) * TUNE.kappa_scale

    # Final Ψ with extra channels appended
    Psi = np.concatenate([Psi_base, eps_R.reshape(-1,1), O.reshape(-1,1), kappa_mean.reshape(-1,1)], axis=1)
    feat_names = [*feat_names, 'eps_recursion', 'outliers', 'kappa_mean']

    meta.update({
        'feat_names': feat_names,
        'ETA_base': ETA,          # only for base channels
        'outliers': O,
        'kappa_mean': kappa_mean,
        'alpha_cyclicity': np.abs(np.mean(np.stack([G[f'{b}:omega'].to_numpy() for b in birds], axis=1), axis=1)),
    })
    return Psi, meta

# --------------------------- Refraction & α -----------------------------------

def refraction_map(Psi: np.ndarray, ETA_base: np.ndarray, tune: Tune) -> np.ndarray:
    """
    Apply refraction to base channels; pad η with zeros for appended channels.
    """
    if ETA_base.shape[1] != Psi.shape[1]:
        extra = Psi.shape[1] - ETA_base.shape[1]
        if extra > 0:
            ETA = np.hstack([ETA_base, np.zeros((ETA_base.shape[0], extra))])
        else:
            ETA = ETA_base[:, :Psi.shape[1]]
    else:
        ETA = ETA_base
    return Psi * (1.0 + tune.mu_refraction * tune.L_star * ETA)

def build_alpha(meta: Dict, tune: Tune) -> np.ndarray:
    a_c = meta['alpha_cyclicity']
    a_c = (a_c - a_c.mean()) / (a_c.std() + 1e-9)
    a_c = (a_c - a_c.min()) / (a_c.max() - a_c.min() + 1e-9)
    a_s = meta['kappa_mean']
    a_s = (a_s - a_s.mean()) / (a_s.std() + 1e-9)
    a_s = (a_s - a_s.min()) / (a_s.max() - a_s.min() + 1e-9)
    alpha = tune.alpha_c1 * a_c + tune.alpha_c2 * a_s
    return np.maximum(alpha, tune.alpha_floor)

# --------------------------- Discretization & chain ----------------------------

def discretize_states(Xi: np.ndarray, tune: Tune):
    pca = PCA(n_components=min(tune.pca_dim, Xi.shape[1]), random_state=tune.random_state)
    Z = pca.fit_transform(np.nan_to_num(Xi, nan=0.0))
    km = MiniBatchKMeans(n_clusters=tune.n_clusters, batch_size=tune.kmeans_batch,
                         verbose=0, random_state=tune.random_state)
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_
    return Z, labels, centers, {'pca': pca, 'kmeans': km}

def tau_weighted_counts(labels: np.ndarray, tau: np.ndarray, K: int) -> np.ndarray:
    C = np.zeros((K, K), dtype=float)
    for t in range(len(labels) - 1):
        C[labels[t], labels[t+1]] += tau[t]
    return C

def reversible_projection(C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Fsym = 0.5 * (C + C.T)
    pi = Fsym.sum(axis=1)
    pi = pi / (pi.sum() + 1e-12)
    M = Fsym / (Fsym.sum(axis=1, keepdims=True) + 1e-12)
    return M, pi

# --------------------------- Witnesses ----------------------------------------

def entropy_production(C: Optional[np.ndarray], pi: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None) -> float:
    """
    EP ≈ 0.5 * Σ_{x,y} (J_xy - J_yx) log(J_xy / J_yx), with J = π M.
    This implementation avoids divide/log warnings by masking where valid.
    """
    if M is None or pi is None:
        row = C.sum(axis=1, keepdims=True) + 1e-12
        M = C / row
        pi = C.sum(axis=1)
        pi = pi / (pi.sum() + 1e-12)
    J  = pi[:, None] * M
    Jt = pi[None, :] * M.T
    mask = (J > 0) & (Jt > 0)
    ratio = np.ones_like(J)
    np.divide(J, Jt, out=ratio, where=mask)
    term = np.zeros_like(J)
    term[mask] = 0.5 * (J[mask] - Jt[mask]) * np.log(ratio[mask])
    return float(np.nansum(term))

def sample_cycle_affinity(M: np.ndarray, pi: np.ndarray, n_samples: int = 1000, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    K = M.shape[0]
    vals = []
    for _ in range(n_samples):
        x, y, z = rng.randint(0, K, size=3)
        if x == y or y == z or z == x:
            continue
        a = 0.0
        a += math.log((pi[x]*M[x,y] + 1e-24)/(pi[y]*M[y,x] + 1e-24))
        a += math.log((pi[y]*M[y,z] + 1e-24)/(pi[z]*M[z,y] + 1e-24))
        a += math.log((pi[z]*M[z,x] + 1e-24)/(pi[x]*M[x,z] + 1e-24))
        vals.append(abs(a))
    return float(np.mean(vals)) if vals else 0.0

# --------------------------- Energy & leadership -------------------------------

def estimate_energy(pi: np.ndarray, T_eff: float) -> np.ndarray:
    return -T_eff * np.log(np.maximum(pi, 1e-24))

def local_energy_gradient(centers: np.ndarray, U: np.ndarray, tune: Tune) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=min(tune.neigh_k, centers.shape[0]-1), algorithm='auto')
    nbrs.fit(centers)
    grads = np.zeros_like(centers)
    for i in range(centers.shape[0]):
        _, idxs = nbrs.kneighbors(centers[i:i+1, :], return_distance=True)
        idxs = idxs[0]; idxs = idxs[idxs != i]
        X = centers[idxs, :] - centers[i, :]
        y = U[idxs] - U[i]
        lam = 1e-3
        A = np.linalg.pinv(X.T @ X + lam*np.eye(X.shape[1])) @ X.T @ y
        grads[i, :] = A
    return grads

def leadership_scores_over_time(Z_low, labels, centers_low, grads_low, meta, models, feat_index_map, feat_std):
    pca: PCA = models['pca']
    # Map gradients back to original feature space and normalize by feature scale
    G_full_per_cluster = grads_low @ pca.components_
    G_full_per_cluster = G_full_per_cluster / feat_std

    nT = Z_low.shape[0]
    D_full = pca.components_.shape[1]
    grad_full_t = np.zeros((nT, D_full))
    for t in range(nT):
        grad_full_t[t, :] = G_full_per_cluster[labels[t], :]

    birds = meta['birds']
    scores = {b: -grad_full_t[:, feat_index_map[f'{b}:align_front']] for b in birds}
    S = pd.DataFrame(scores)
    rank = S.mean(axis=0).sort_values(ascending=False)
    return S, rank

# --------------------------- Main ---------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)

    ap.add_argument('--bird-col', default='individual-local-identifier')
    ap.add_argument('--time-col', default='timestamp')
    ap.add_argument('--lat-col', default='location-lat')
    ap.add_argument('--lon-col', default='location-long')

    # Session control
    ap.add_argument('--session-col', default='comments', help='Column for flight label (set "" to disable)')
    ap.add_argument('--session', default='', help='Exact value in session-col (e.g., "homing flight 1")')
    ap.add_argument('--min-birds', type=int, default=5)
    ap.add_argument('--min-duration', type=int, default=60, help='seconds')

    # Tuning overrides (optional)
    ap.add_argument('--n-clusters', type=int, default=TUNE.n_clusters)
    ap.add_argument('--pca-dim', type=int, default=TUNE.pca_dim)
    ap.add_argument('--target-hz', type=float, default=TUNE.target_hz)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    TUNE.n_clusters = args.n_clusters
    TUNE.pca_dim = args.pca_dim
    TUNE.target_hz = args.target_hz

    raw = load_movebank_csvs(args.csv)

    sess_col = args.session_col if args.session_col.strip() else None
    sess_val = args.session.strip() if args.session.strip() else None
    sel = select_session(raw, args.bird_col, args.time_col, sess_col, sess_val, args.min_birds, args.min_duration)

    G, meta = synchronize_and_features(sel, args.bird_col, args.time_col, args.lat_col, args.lon_col, TUNE)

    # EchoKey state
    Psi, meta = build_echokey_state(G, meta, TUNE)

    # Refraction + α
    Xi = refraction_map(Psi, meta['ETA_base'], TUNE)
    alpha = build_alpha(meta, TUNE)
    dt = meta['dt']; d_tau = alpha[:-1] * dt

    # Discretize & reversible projection
    Z_low, labels, centers_low, models = discretize_states(Xi, TUNE)
    C = tau_weighted_counts(labels, d_tau, TUNE.n_clusters)
    M_star, pi_star = reversible_projection(C)
    U = estimate_energy(pi_star, TUNE.T_eff)

    # Witnesses
    EP_raw  = entropy_production(C)
    EP_star = entropy_production(None, pi=pi_star, M=M_star)
    aff = sample_cycle_affinity(M_star, pi_star, n_samples=500)

    # Leadership mapping
    grads_low = local_energy_gradient(centers_low, U, TUNE)
    feat_names = meta['feat_names']
    feat_index_map = {name: i for i, name in enumerate(feat_names)}

    # Feature scale for normalization
    feat_std = np.std(Xi, axis=0) + 1e-9
    S_time, rank = leadership_scores_over_time(
        Z_low, labels, centers_low, grads_low, meta, models, feat_index_map, feat_std=feat_std
    )

    macrostats = {
        'polarization_mean': float(np.mean(np.hypot(G['ex'], G['ey']))),
        'alpha_mean': float(np.mean(alpha)),
        'kappa_mean_mean': float(np.mean(meta['kappa_mean'])),
    }

    report = {
        'birds': meta['birds'],
        'n_birds': len(meta['birds']),
        'n_steps': int(meta['n_steps']),
        'dt': meta['dt'],
        'pca_dim': TUNE.pca_dim,
        'n_clusters': TUNE.n_clusters,
        'EP_raw': EP_raw,
        'EP_star': EP_star,
        'EP_reduction_ratio': (EP_star / (EP_raw + 1e-12)),
        'cycle_affinity_mean_abs': aff,
        'tolerances': {
            'EP_star_should_be_approximately_0': True,
            'EP_reduction_ratio_warn_threshold': TUNE.ep_ratio_warn,
            'cycle_affinity_tol': TUNE.cycle_affinity_tol
        },
        'macrostats': macrostats
    }

    # Save artifacts
    pd.DataFrame({'cluster': np.arange(TUNE.n_clusters), 'pi_star': pi_star, 'U': U}).to_csv(
        os.path.join(args.outdir, 'energy_per_cluster.csv'), index=False)
    np.save(os.path.join(args.outdir, 'C_counts.npy'), C)
    np.save(os.path.join(args.outdir, 'M_star.npy'), M_star)
    np.save(os.path.join(args.outdir, 'centers_low.npy'), centers_low)
    np.save(os.path.join(args.outdir, 'labels.npy'), labels)
    with open(os.path.join(args.outdir, 'witness_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    S_time.to_csv(os.path.join(args.outdir, 'leadership_scores_time.csv'), index=False)
    rank.to_csv(os.path.join(args.outdir, 'leadership_rank_mean.csv'))

    # Console summary
    print("\n=== EchoKey Equilibriumization: Summary ===")
    print(f"Birds: {meta['birds']}")
    print(f"Steps: {meta['n_steps']}  dt={meta['dt']:.3f}s  K={TUNE.n_clusters}  PCA={TUNE.pca_dim}")
    print(f"Entropy production (raw chain):   {EP_raw:.6e}")
    print(f"Entropy production (reversible):  {EP_star:.6e}  (should ~ 0)")
    print(f"EP reduction ratio (M*/raw):      {report['EP_reduction_ratio']:.6f}")
    print(f"Mean |cycle affinity| on M*:      {aff:.3e}  (<= {TUNE.cycle_affinity_tol:.1e} ideal)")
    print("Leadership (mean score):")
    for b, val in rank.items():
        print(f"  {b:>20s} : {val:+.4f}")
    print(f"\nArtifacts written to: {args.outdir}")

if __name__ == '__main__':
    main()
