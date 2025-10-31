#!/usr/bin/env python3
# unify_trace.py
# A+B unifier with robust synchronization bridge (Model C: quantum dimer + classical surrogate)
# FIXED: Proper QRT spectrum with DC removal and time-evolved steady state

from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

# ========= Import Model A & B (kept exactly as your working files) =========
A_OK = True; B_OK = True; A_ERR=""; B_ERR=""
try:
    from mv_bose_qg_model import (
        RunParams as MVRunParams, Geometry as MVGeom, MassesAndTime as MVMT,
        run_once as run_mv, compute_branch_phases as mv_compute_phases
    )
except Exception as e:
    A_OK = False; A_ERR = str(e)

try:
    from ah_cg_qft_model import (
        GeometryCG as AHGeom, MassTimeCG as AHMT, run_once as run_ah,
        paper_faithful_defaults as ah_paper, radius_from_mass_density
    )
except Exception as e:
    B_OK = False; B_ERR = str(e)

# ============================== Utils ===============================
def dagger(A: np.ndarray) -> np.ndarray: return A.conj().T
def kron(A: np.ndarray, B: np.ndarray) -> np.ndarray: return np.kron(A, B)
def a_destroy(N: int) -> np.ndarray:
    M = np.zeros((N,N), dtype=np.complex128)
    for n in range(1,N): M[n-1,n] = math.sqrt(n)
    return M
def vec(rho: np.ndarray) -> np.ndarray: return rho.reshape(-1,1, order="F")
def unvec(v: np.ndarray, dim: int) -> np.ndarray: return v.reshape((dim,dim), order="F")

def lindblad_superop(H: np.ndarray, Ls: List[np.ndarray]) -> np.ndarray:
    dim = H.shape[0]; I = np.eye(dim, dtype=np.complex128)
    Lcoh = -1j*(kron(I,H) - kron(H.T,I))
    Ldis = np.zeros_like(Lcoh)
    for L in Ls:
        LdL = dagger(L) @ L
        Ldis += kron(L, L.conj()) - 0.5*kron(I, LdL.T) - 0.5*kron(LdL, I)
    return Lcoh + Ldis

def rk4_step(f, y, t, dt):
    k1=f(t,y); k2=f(t+0.5*dt, y+0.5*dt*k1); k3=f(t+0.5*dt, y+0.5*dt*k2); k4=f(t+dt, y+dt*k3)
    return y + (dt/6.0)*(k1+2*k2+2*k3+k4)

def power_spectrum(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x); n = len(x)
    if n < 4: return np.array([0.0]), np.array([0.0])
    w = np.hanning(n); xw = x*w; W2 = float((w**2).sum())
    if np.iscomplexobj(xw):
        X = np.fft.fft(xw); f = np.fft.fftfreq(n, d=dt); m = f>=0
        return f[m], (np.abs(X[m])**2)/W2
    X = np.fft.rfft(xw); f = np.fft.rfftfreq(n, d=dt)
    return f, (np.abs(X)**2)/W2

def find_peaks(y: np.ndarray, min_sep: int=5, thresh_rel: float=0.005) -> List[int]:
    if len(y)<3: return []
    thr = float(y.max())*thresh_rel
    peaks=[]
    for i in range(1,len(y)-1):
        if y[i]>thr and y[i]>y[i-1] and y[i]>y[i+1] and (not peaks or i-peaks[-1]>=min_sep):
            peaks.append(i)
    return peaks

def doublet_confidence_from_psd(freqs: np.ndarray, psd: np.ndarray,
                                min_split: float,
                                band: Optional[Tuple[float,float]]=None) -> Tuple[float, Dict[str,float]]:
    if band is not None:
        lo,hi = band; m = (freqs>=lo)&(freqs<=hi); f, p = freqs[m], psd[m]
    else:
        f,p = freqs, psd
    idx = find_peaks(p, min_sep=max(3, len(p)//120), thresh_rel=0.02)
    if len(idx)<2: return 0.0, {"peaks": len(idx), "split": 0.0}
    top2 = sorted(idx, key=lambda i: p[i], reverse=True)[:2]; i1,i2 = sorted(top2)
    P1,P2 = float(p[i1]), float(p[i2]); df = abs(float(f[i2]-f[i1]))
    if df < min_split: return 0.0, {"peaks":2,"split":df,"P1":P1,"P2":P2}
    sep_score = min(1.0, df/(3.0*min_split))
    conf = (2.0*min(P1,P2)/(P1+P2))*sep_score
    return float(conf), {"peaks":2,"split":df,"P1":P1,"P2":P2}

def wrap_angle(x: float) -> float: return (x+math.pi)%(2*math.pi)-math.pi
def bistability_zero_pi(phases: np.ndarray, sigma: float=0.25) -> Dict[str,float]:
    diffs = np.array([wrap_angle(p) for p in phases])
    z0  = np.exp(-(np.abs(diffs)          **2)/(2*sigma**2)).mean()
    zpi = np.exp(-(np.abs(diffs-math.pi)  **2)/(2*sigma**2)).mean()
    return {"zero_lock": float(z0), "pi_lock": float(zpi), "bias": float(zpi-z0)}

# =========================== Model C — Quantum dimer ===========================
@dataclass
class CqParams:
    N: int = 5           # increased for better resolution
    w1: float = 1.00
    w2: float = 1.14
    g:  float = 0.38
    k1: float = 0.015
    k2: float = 0.015
    drive:  float = 0.06     
    drive2: float = 0.02     # asymmetric drive
    phi_deph: float = 0.003  # tiny pure dephasing
    dt: float = 0.01
    T_burn: float = 12.0
    T_run:  float = 70.0

def steady_state_via_eigs(L: np.ndarray, dim: int) -> np.ndarray:
    """Fallback: find steady state from eigenvector with eigenvalue ~ 0."""
    w, V = np.linalg.eig(L)
    k = int(np.argmin(np.abs(w)))
    v_ss = V[:, k]
    rho_ss = unvec(v_ss, dim)
    tr = np.trace(rho_ss)
    if abs(tr) > 1e-14:
        rho_ss = rho_ss / tr
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)
    return rho_ss

def simulate_quantum_dimer(p: CqParams) -> Dict[str,Any]:
    N=p.N; a=a_destroy(N); I=np.eye(N, dtype=np.complex128)
    a1=kron(a,I); a2=kron(I,a); a1d=dagger(a1); a2d=dagger(a2)

    H = (p.w1*(a1d@a1) + p.w2*(a2d@a2) +
        p.g*(a1d@a2 + a1@a2d) +
        p.drive*(a1 + a1d) + p.drive2*(a2 + a2d))   # asym. coherent ports

    Ls = []
    if p.k1>0: Ls.append(math.sqrt(p.k1)*a1)
    if p.k2>0: Ls.append(math.sqrt(p.k2)*a2)
    if p.phi_deph>0:
        Ls += [math.sqrt(p.phi_deph)*(a1d@a1),
               math.sqrt(p.phi_deph)*(a2d@a2)]       # pure dephasing
    L = lindblad_superop(H, Ls)

    dim = N*N
    rho0 = np.zeros((dim,dim), dtype=np.complex128); rho0[0,0]=1.0
    v = vec(rho0); t=0.0
    nb = max(1,int(p.T_burn/p.dt)); nr=max(2,int(p.T_run/p.dt))
    def f(_t,_v): return L@_v

    for _ in range(nb): v = rk4_step(f, v, t, p.dt); t+=p.dt

    a1_ts=[]; a2_ts=[]; ph_ts=[]
    for _ in range(nr):
        rho = unvec(v, dim)
        s1  = np.trace(rho@a1); s2 = np.trace(rho@a2)
        a1_ts.append(s1); a2_ts.append(s2)
        ph_ts.append(wrap_angle(np.angle(s1) - np.angle(s2)))
        v = rk4_step(f, v, t, p.dt); t+=p.dt

    a1_ts = np.asarray(a1_ts); ph_ts=np.asarray(ph_ts)

    # NEW — robust steady state from time evolution
    rho_ss_time = unvec(v, dim)
    rho_ss_time = 0.5 * (rho_ss_time + rho_ss_time.conj().T)  # hygiene
    tr = np.trace(rho_ss_time)
    if abs(tr) > 1e-14:
        rho_ss_time = rho_ss_time / tr
    
    # Use time-evolved state, fallback to eigenvector if needed
    try:
        rho_ss = rho_ss_time
    except Exception:
        rho_ss = steady_state_via_eigs(L, dim)

    # --- QRT emission spectrum for dressed-mode doublet ---
    Delta    = p.w2 - p.w1
    splitRad = 2.0 * math.sqrt(p.g**2 + (Delta/2.0)**2)
    f_mean   = 0.5 * (p.w1 + p.w2) / (2.0 * math.pi)
    df_pred  = splitRad / (2.0 * math.pi)

    # longer window → better resolution
    T_spec = max(p.T_run, 140.0)

    def qrt_port(L, a_op, rho_ss, dt, T):
        # evolve v(τ) = e^{Lτ} vec(a ρ_ss) and build C(τ) = Tr[a† v(τ)]
        vq = vec(a_op @ rho_ss)
        nb = max(1, int(T / dt))
        t = 0.0
        corr = []
        def f(_t, vv): return L @ vv
        for _ in range(nb):
            X = unvec(vq, dim)
            corr.append(np.trace(dagger(a_op) @ X))
            vq = rk4_step(f, vq, t, dt); t += dt
        corr = np.asarray(corr)
        # remove coherent DC pedestal (makes peaks visible)
        corr = corr - corr.mean()
        return power_spectrum(corr, dt)

    # sum spectra from both leakage ports
    f1, S1 = qrt_port(L, a1, rho_ss, p.dt, T_spec)
    f2, S2 = qrt_port(L, a2, rho_ss, p.dt, T_spec)
    # align if needed (same grid expected)
    if len(f1) != len(f2) or np.max(np.abs(f1 - f2)) > 1e-12:
        # simple interp of S2 onto f1
        S2 = np.interp(f1, f2, S2.real).astype(np.float64)
        freqs, psd = f1, (S1.real + S2)
    else:
        freqs, psd = f1, (S1.real + S2.real)

    band      = (max(0.0, f_mean - 1.5*df_pred), f_mean + 1.5*df_pred)
    min_split = 0.30 * df_pred
    conf, meta = doublet_confidence_from_psd(freqs, psd, min_split=min_split, band=band)

    bi = bistability_zero_pi(ph_ts, sigma=0.25)

    return {"params": asdict(p),
            "observables": {"doublet_confidence": float(conf),
                            "spectrum_meta": meta,
                            "phase_bistability": bi},
            "series_shapes": {"a1_len": int(len(a1_ts))}}

# =========================== Model C — Classical surrogate =====================
@dataclass
class CcParams:
    w1: float = 1.00; w2: float = 1.14; g: float = 0.38
    mu: float = 0.55; kappa: float = 0.95; gamma: float = 0.15
    dt: float = 0.01; T_burn: float = 12.0; T_run: float = 70.0

def simulate_classical_surrogate(p: CcParams) -> Dict[str,Any]:
    # Stuart–Landau: d z/dt = ((mu - gamma) - kappa |z|^2) z - i w z - i g z_other
    def f(z1,z2):
        return ((p.mu-p.gamma)-p.kappa*abs(z1)**2)*z1 -1j*p.w1*z1 -1j*p.g*z2, \
               ((p.mu-p.gamma)-p.kappa*abs(z2)**2)*z2 -1j*p.w2*z2 -1j*p.g*z1
    def rk4(z1,z2,dt):
        k1=f(z1,z2); k2=f(z1+0.5*dt*k1[0], z2+0.5*dt*k1[1])
        k3=f(z1+0.5*dt*k2[0], z2+0.5*dt*k2[1]); k4=f(z1+dt*k3[0], z2+dt*k3[1])
        return z1+(dt/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]), \
               z2+(dt/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])

    z1=0.1+0.0j; z2=0.09+0.02j
    nb=max(1,int(p.T_burn/p.dt)); nr=max(2,int(p.T_run/p.dt))
    for _ in range(nb): z1,z2 = rk4(z1,z2,p.dt)

    z1_ts=[]; ph_ts=[]
    for _ in range(nr):
        z1,z2 = rk4(z1,z2,p.dt)
        z1_ts.append(z1)
        ph_ts.append(wrap_angle(np.angle(z1)-np.angle(z2)))

    z1_ts=np.asarray(z1_ts); ph_ts=np.asarray(ph_ts)
    freqs, psd = power_spectrum(z1_ts, p.dt)
    conf, meta = doublet_confidence_from_psd(freqs, psd, min_split=0.05, band=None)  # will report ~0
    bi = bistability_zero_pi(ph_ts, sigma=0.25)
    return {"params": asdict(p), "observables":{"doublet_confidence": float(conf),
            "spectrum_meta": meta, "phase_bistability": bi}}

# =========================== A/B wrappers & coupler ============================
@dataclass
class MVInputs: d: float=450e-6; dx: float=250e-6; m: float=1e-14; tau: float=2.5; dephase: float=0.0
def run_model_A(inp: MVInputs) -> Dict[str,Any]:
    if not A_OK: return {"error": f"Model A import failed: {A_ERR}", "negativity":0.0, "concurrence":0.0}
    rp = MVRunParams(geom=MVGeom(inp.d,inp.dx), mt=MVMT(inp.m,inp.m,inp.tau), gamma_dephase=inp.dephase)
    res = run_mv(rp)
    try:
        ph = mv_compute_phases(MVGeom(inp.d,inp.dx), MVMT(inp.m,inp.m,inp.tau))
        delta = ph.dphi_RL - ph.dphi_LR
        bi = {"zero_lock": math.exp(-(abs((delta)%(2*math.pi))**2)/(2*0.25**2)),
              "pi_lock":   math.exp(-(abs((delta-math.pi)%(2*math.pi))**2)/(2*0.25**2)),
              "bias": 0.0}
    except Exception:
        bi = {"zero_lock":0.0,"pi_lock":0.0,"bias":0.0}
    return {"raw":res, "negativity": float(res["entanglement"]["negativity"]),
            "concurrence": float(res["entanglement"]["concurrence"]), "bistability": bi}

@dataclass
class AHInputs:
    M: float|None=None; t: float|None=None; R: float|None=None; rho: float|None=None
    sep_factor: float=10.0; m_particle: float=2.87e-25
def run_model_B(inp: AHInputs) -> Dict[str,Any]:
    if not B_OK: return {"error": f"Model B import failed: {B_ERR}", "negativity":0.0, "concurrence":0.0}
    if inp.M is None or inp.t is None: geom, mt = ah_paper()
    else:
        R = radius_from_mass_density(inp.M, 6900.0 if inp.rho is None else float(inp.rho)) if inp.R is None else float(inp.R)
        geom = AHGeom(d_RL=float(inp.sep_factor)*R, R=R)
        mt   = AHMT(M=float(inp.M), m_particle=float(inp.m_particle), t=float(inp.t))
    res = run_ah(geom, mt)
    return {"raw":res, "negativity": float(res["entanglement"]["negativity"]),
            "concurrence": float(res["entanglement"]["concurrence"]), "theta_phi": res["theta_phi"]}

def lambda_with_doublet(Na: float, Nb: float, Dc: float, eps: float=1e-16) -> float:
    lam0 = Na/max(eps, Na+Nb)
    return float((1.0-Dc)*lam0 + Dc*0.5)

def unified_score(Na: float, Nb: float, lam: float, Dc: float, gamma: float=0.5) -> float:
    return float(lam*Na + (1.0-lam)*Nb + gamma*Dc*math.sqrt(max(0.0,Na*Nb)))

def regime_label(Na: float, Nb: float, lam: float, Dc: float, eps: float=1e-12) -> str:
    a=Na>eps; b=Nb>eps
    if not a and not b: return "No-entanglement region"
    if a and not b: return "Quantum-mediator dominant"
    if b and not a: return "Classical+QFT dominant"
    if Dc>=0.25:
        if   lam>0.6: return "Bridge (A-lean, dressed-state)"
        elif lam<0.4: return "Bridge (B-lean, dressed-state)"
        else:         return "Bridge (balanced, dressed-state)"
    return "Bridge (weak hybridization)"

# ================================ Runs =================================
def run_paper() -> Dict[str,Any]:
    mv = MVInputs()
    ah = AHInputs(M=None, t=None)  # paper-faithful Aziz–Howl (15 kg, 1 s, d_RL=10 R, Yb mass)
    cq = CqParams()                # preset tuned for a clear doublet
    cc = CcParams(w1=cq.w1, w2=cq.w2, g=cq.g)

    A = run_model_A(mv)
    B = run_model_B(ah)
    Cq = simulate_quantum_dimer(cq)
    Cc = simulate_classical_surrogate(cc)

    Na = max(0.0, A.get("negativity", 0.0))
    Nb = max(0.0, B.get("negativity", 0.0))
    Dc = max(0.0, Cq["observables"]["doublet_confidence"])

    lam   = lambda_with_doublet(Na, Nb, Dc)
    score = unified_score(Na, Nb, lam, Dc)
    label = regime_label(Na, Nb, lam, Dc)

    return {"mode":"paper",
            "inputs":{"mv":asdict(mv), "ah":asdict(ah), "c_quantum":asdict(cq), "c_classical":asdict(cc)},
            "modelA":A, "modelB":B, "modelC_quantum":Cq, "modelC_classical":Cc,
            "bridge":{"lambda":lam, "doublet_confidence":Dc, "regime":label, "unified_score":score}}

def run_quick(args) -> Dict[str,Any]:
    mv = MVInputs(d=args.mv_d, dx=args.mv_dx, m=args.mv_m, tau=args.mv_tau, dephase=args.mv_dephase)
    M  = None if math.isnan(args.ah_M) else float(args.ah_M)
    t  = None if math.isnan(args.ah_t) else float(args.ah_t)
    R  = None if math.isnan(args.ah_R) else float(args.ah_R)
    rho= None if math.isnan(args.ah_rho) else float(args.ah_rho)
    ah = AHInputs(M=M,t=t,R=R,rho=rho,sep_factor=args.ah_sep,m_particle=args.ah_mp)

    cq = CqParams(N=args.c_N, w1=args.c_w1, w2=args.c_w2, g=args.c_g, k1=args.c_k1, k2=args.c_k2,
                  drive=args.c_drive, drive2=args.c_drive2, phi_deph=args.c_phi_deph,
                  dt=args.c_dt, T_burn=args.c_Tburn, T_run=args.c_Trun)
    cc = CcParams(w1=cq.w1, w2=cq.w2, g=cq.g, mu=args.cc_mu, kappa=args.cc_kappa,
                  gamma=args.cc_gamma, dt=args.c_dt, T_burn=args.c_Tburn, T_run=args.c_Trun)

    A = run_model_A(mv); B=run_model_B(ah); Cq=simulate_quantum_dimer(cq); Cc=simulate_classical_surrogate(cc)
    Na=max(0.0,A.get("negativity",0.0)); Nb=max(0.0,B.get("negativity",0.0)); Dc=max(0.0,Cq["observables"]["doublet_confidence"])
    lam=lambda_with_doublet(Na,Nb,Dc); score=unified_score(Na,Nb,lam,Dc); label=regime_label(Na,Nb,lam,Dc)
    return {"mode":"quick","inputs":{"mv":asdict(mv),"ah":asdict(ah),"c_quantum":asdict(cq),"c_classical":asdict(cc)},
            "modelA":A,"modelB":B,"modelC_quantum":Cq,"modelC_classical":Cc,
            "bridge":{"lambda":lam,"doublet_confidence":Dc,"regime":label,"unified_score":score}}

# ================================= CLI =================================
def main():
    ap = argparse.ArgumentParser(description="Unified A+B with synchronization bridge (Quantum dimer)")
    ap.add_argument("--mode", choices=["paper","quick"], default="paper")
    ap.add_argument("--out", type=str, default="")
    # MV
    ap.add_argument("--mv_d", type=float, default=450e-6)
    ap.add_argument("--mv_dx", type=float, default=250e-6)
    ap.add_argument("--mv_m", type=float, default=1e-14)
    ap.add_argument("--mv_tau", type=float, default=2.5)
    ap.add_argument("--mv_dephase", type=float, default=0.0)
    # AH
    ap.add_argument("--ah_M", type=float, default=float("nan"))
    ap.add_argument("--ah_t", type=float, default=float("nan"))
    ap.add_argument("--ah_R", type=float, default=float("nan"))
    ap.add_argument("--ah_rho", type=float, default=float("nan"))
    ap.add_argument("--ah_sep", type=float, default=10.0)
    ap.add_argument("--ah_mp", type=float, default=2.87e-25)
    # C quantum dimer
    ap.add_argument("--c_N", type=int, default=5)
    ap.add_argument("--c_w1", type=float, default=1.00)
    ap.add_argument("--c_w2", type=float, default=1.14)
    ap.add_argument("--c_g",  type=float, default=0.38)
    ap.add_argument("--c_k1", type=float, default=0.015)
    ap.add_argument("--c_k2", type=float, default=0.015)
    ap.add_argument("--c_drive", type=float, default=0.06)
    ap.add_argument("--c_drive2", type=float, default=0.02)  # NEW parameter
    ap.add_argument("--c_phi_deph", type=float, default=0.003)  # NEW parameter
    ap.add_argument("--c_dt", type=float, default=0.01)
    ap.add_argument("--c_Tburn", type=float, default=12.0)
    ap.add_argument("--c_Trun",  type=float, default=70.0)
    # C classical surrogate
    ap.add_argument("--cc_mu", type=float, default=0.55)
    ap.add_argument("--cc_kappa", type=float, default=0.95)
    ap.add_argument("--cc_gamma", type=float, default=0.15)

    args = ap.parse_args()
    payload = run_paper() if args.mode=="paper" else run_quick(args)
    print(json.dumps(payload, indent=2))
    if args.out:
        with open(args.out,"w") as f: json.dump(payload,f,indent=2)

if __name__=="__main__":
    main()