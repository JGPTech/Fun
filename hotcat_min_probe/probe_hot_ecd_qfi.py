#!/usr/bin/env python3
"""
Minimal hot-ECD displacement-sensing sandbox with a dual-witness check.

This probe reproduces the paper's simplified ECD hot-cat QFI/rate model and
then checks the large-separation branch formula with two independent numerical
witnesses:

  1. exact mixed-state spectral QFI in a finite Fock cutoff;
  2. finite-displacement Bures response, converted back to a local QFI estimate.

The second witness is the lightweight analog of a response-witness check: the
branch formula should not only match an infinitesimal spectral-QFI calculation;
it should also track the actual finite state distinguishability produced by a
small displacement.

Outputs are written to ./analysis/ by default:
  - fig2_like_protocol_grid.csv
  - fig2_like_best_point.csv
  - dual_witness_ecd_sweep.csv
  - dual_witness_summary.csv
  - probe_summary.md

Dependencies: numpy, scipy. Optional: matplotlib for --plots.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.linalg import eigh
from scipy.special import eval_genlaguerre, gammaln


def annihilation(cutoff: int) -> np.ndarray:
    """Bosonic annihilation operator in a truncated Fock basis."""
    a = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for n in range(1, cutoff):
        a[n - 1, n] = math.sqrt(n)
    return a


def thermal_probs(nbar: float, cutoff: int) -> np.ndarray:
    """Truncated, renormalized thermal distribution."""
    if nbar < 0:
        raise ValueError("nbar must be non-negative")
    if nbar == 0:
        p = np.zeros(cutoff)
        p[0] = 1.0
        return p
    r = nbar / (1.0 + nbar)
    n = np.arange(cutoff)
    p = (1.0 - r) * (r ** n)
    return p / p.sum()


def displacement_operator(cutoff: int, alpha: float) -> np.ndarray:
    """D(alpha)=exp(alpha a^dagger - alpha a), alpha real.

    Uses closed-form Fock-basis matrix elements instead of a dense matrix
    exponential. This is faster for the small sandbox sweeps here.
    """
    x = alpha * alpha
    pref_exp = math.exp(-0.5 * x)
    d = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for m in range(cutoff):
        for n in range(cutoff):
            if m >= n:
                # <m|D(alpha)|n>
                log_pref = 0.5 * (gammaln(n + 1) - gammaln(m + 1))
                d[m, n] = pref_exp * math.exp(log_pref) * (alpha ** (m - n)) * eval_genlaguerre(n, m - n, x)
            else:
                # <m|D(alpha)|n>
                log_pref = 0.5 * (gammaln(m + 1) - gammaln(n + 1))
                d[m, n] = pref_exp * math.exp(log_pref) * ((-alpha) ** (n - m)) * eval_genlaguerre(m, n - m, x)
    return d


def displacement_operator_complex(cutoff: int, beta: complex) -> np.ndarray:
    """D(beta)=exp(beta a^dagger - beta* a), beta complex."""
    x = abs(beta) ** 2
    pref_exp = math.exp(-0.5 * x)
    d = np.zeros((cutoff, cutoff), dtype=np.complex128)
    for m in range(cutoff):
        for n in range(cutoff):
            if m >= n:
                log_pref = 0.5 * (gammaln(n + 1) - gammaln(m + 1))
                d[m, n] = pref_exp * math.exp(log_pref) * (beta ** (m - n)) * eval_genlaguerre(n, m - n, x)
            else:
                log_pref = 0.5 * (gammaln(m + 1) - gammaln(n + 1))
                d[m, n] = pref_exp * math.exp(log_pref) * ((-beta.conjugate()) ** (n - m)) * eval_genlaguerre(m, n - m, x)
    return d


def recommended_cutoff(nbar: float, alpha: float, safety: int = 24) -> int:
    """
    Conservative enough for this finite-cutoff probe without making it slow.
    Displaced thermal support is roughly nbar + alpha^2 plus a broad tail.
    """
    center = nbar + alpha * alpha
    spread = math.sqrt(max(1.0, nbar * (nbar + 1.0) + alpha * alpha))
    return int(max(36, math.ceil(center + 7.0 * spread + safety)))


def hermitize(mat: np.ndarray) -> np.ndarray:
    """Remove tiny anti-Hermitian numerical noise."""
    return 0.5 * (mat + mat.conj().T)


def ecd_hot_cat_state(
    nbar: float,
    alpha: float,
    cutoff: int,
    exact_postselection: bool = True,
) -> np.ndarray:
    """
    Finite-cutoff ECD-style hot cat state.

    exact_postselection=True uses the CP/postselected map
        rho -> M rho M^dagger / Tr(M rho M^dagger), M = D(alpha)+D(-alpha).
    This keeps finite branch-overlap/postselection effects.

    exact_postselection=False uses the paper's large-separation mixture form
        sum_n p_n |D(a)n + D(-a)n><...| / norm_n,
    which keeps p_n fixed and only normalizes each cat component.
    """
    p = thermal_probs(nbar, cutoff)
    rho_th = np.diag(p).astype(np.complex128)
    d_plus = displacement_operator(cutoff, alpha)
    d_minus = displacement_operator(cutoff, -alpha)
    m = d_plus + d_minus

    if exact_postselection:
        rho = m @ rho_th @ m.conj().T
        tr = np.trace(rho).real
        if tr <= 0:
            raise FloatingPointError("non-positive postselection trace")
        rho = rho / tr
    else:
        rho = np.zeros((cutoff, cutoff), dtype=np.complex128)
        eye = np.eye(cutoff, dtype=np.complex128)
        for n, pn in enumerate(p):
            ket = (d_plus + d_minus) @ eye[:, n]
            norm = np.vdot(ket, ket).real
            if norm > 1e-14:
                ket = ket / math.sqrt(norm)
                rho += pn * np.outer(ket, ket.conj())
        rho = rho / np.trace(rho).real

    return hermitize(rho)


def displacement_qfi_spectral(rho: np.ndarray) -> float:
    """Mixed-state QFI for displacement generated by X=a+a^dagger."""
    cutoff = rho.shape[0]
    a = annihilation(cutoff)
    x = a + a.conj().T

    vals, vecs = eigh(rho)
    vals = np.clip(vals.real, 0.0, None)
    vals = vals / vals.sum()
    x_eig = vecs.conj().T @ x @ vecs

    lam_i = vals[:, None]
    lam_j = vals[None, :]
    denom = lam_i + lam_j
    numer = (lam_i - lam_j) ** 2
    mask = denom > 1e-14
    terms = np.zeros_like(denom, dtype=np.float64)
    terms[mask] = numer[mask] / denom[mask]
    qfi = 2.0 * np.sum(terms * (np.abs(x_eig) ** 2))
    return float(np.real_if_close(qfi))


def quantum_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    vals, vecs = eigh(hermitize(rho))
    vals = np.clip(vals.real, 0.0, None)
    sqrt_vals = np.sqrt(vals)
    sqrt_rho = (vecs * sqrt_vals) @ vecs.conj().T

    middle = hermitize(sqrt_rho @ sigma @ sqrt_rho)
    middle_vals = eigh(middle, eigvals_only=True)
    middle_vals = np.clip(middle_vals.real, 0.0, None)
    fid = float(np.sum(np.sqrt(middle_vals)) ** 2)
    return min(1.0, max(0.0, fid))


def finite_displacement_bures_qfi(rho: np.ndarray, delta: float) -> float:
    """
    Finite-response witness for displacement generated by X=a+a^dagger.

    For small delta,
        F_Q = 8 * (1 - sqrt(Fidelity(rho, rho_delta))) / delta^2.

    This does not reuse the spectral-QFI expression. It asks whether the state
    is actually distinguishable after a small displacement.
    """
    cutoff = rho.shape[0]
    # exp(-i delta (a+a^dagger)) = D(-i delta).
    u = displacement_operator_complex(cutoff, -1j * delta)
    rho_delta = hermitize(u @ rho @ u.conj().T)
    fid = quantum_fidelity(rho, rho_delta)
    sqrt_fid = math.sqrt(max(0.0, min(1.0, fid)))
    return max(0.0, 8.0 * (1.0 - sqrt_fid) / (delta * delta))


def qfi_ecd_asymptotic(nbar: float, alpha: float, visibility_sq: float = 1.0) -> float:
    """Paper's large-separation hot-ECD expression."""
    return 16.0 * alpha * alpha * visibility_sq + 4.0 / (2.0 * nbar + 1.0)


def fig2_like_protocol_grid(
    nbar0: float = 10.0,
    cooling_rate: float = 1.0,
    heating_rate: float = 1e-3,
    spin_dephasing_rate: float = 1e-2,
    cool_max: float = 8.0,
    cat_min: float = 0.10,
    cat_max: float = 9.0,
    n_cool: int = 81,
    n_cat: int = 90,
) -> List[Dict[str, float]]:
    """Reproduce the simplified trapped-ion objective used around Fig. 2."""
    rows: List[Dict[str, float]] = []
    for tcool in np.linspace(0.0, cool_max, n_cool):
        n_post = nbar0 * math.exp(-cooling_rate * tcool)
        local_qfi = 4.0 / (2.0 * n_post + 1.0)
        for tcat in np.linspace(cat_min, cat_max, n_cat):
            visibility_sq = math.exp(-spin_dephasing_rate * tcat - (8.0 / 3.0) * heating_rate * tcat ** 3)
            qfi = 16.0 * tcat ** 2 * visibility_sq + local_qfi
            rate = qfi / (tcool + tcat)
            rows.append({
                "cool_time": tcool,
                "cat_time": tcat,
                "post_cooling_nbar": n_post,
                "cat_visibility_sq": visibility_sq,
                "qfi_reduced": qfi,
                "qfi_per_total_prep_time": rate,
            })
    return rows


def dual_witness_sweep(
    nbars: Iterable[float] = (0.5, 2.0, 5.0, 10.0),
    etas: Iterable[float] = (0.125, 0.25, 0.5, 1.0, 2.0),
    max_cutoff: int = 150,
    exact_postselection: bool = True,
    finite_signal_delta: float = 1e-3,
    pass_tol: float = 0.10,
) -> List[Dict[str, float]]:
    """Compare the branch formula against spectral QFI and finite response."""
    rows: List[Dict[str, float]] = []
    for nbar in nbars:
        for eta in etas:
            alpha = math.sqrt(eta * (2.0 * nbar + 1.0))
            cutoff = min(max_cutoff, recommended_cutoff(nbar, alpha))
            rho = ecd_hot_cat_state(nbar, alpha, cutoff, exact_postselection=exact_postselection)

            qfi_asym = qfi_ecd_asymptotic(nbar, alpha)
            qfi_spectral = displacement_qfi_spectral(rho)
            qfi_bures = finite_displacement_bures_qfi(rho, finite_signal_delta)

            spectral_rel_error = (qfi_spectral - qfi_asym) / qfi_asym
            bures_rel_error = (qfi_bures - qfi_asym) / qfi_asym
            witness_internal_rel_gap = (qfi_bures - qfi_spectral) / max(abs(qfi_spectral), 1e-15)

            spectral_abs = abs(spectral_rel_error)
            bures_abs = abs(bures_rel_error)
            internal_abs = abs(witness_internal_rel_gap)
            dual_pass = spectral_abs <= pass_tol and bures_abs <= pass_tol and internal_abs <= pass_tol

            rows.append({
                "nbar": nbar,
                "eta_alpha2_over_2nbar_plus_1": eta,
                "alpha": alpha,
                "cutoff": cutoff,
                "finite_signal_delta": finite_signal_delta,
                "qfi_ecd_asymptotic": qfi_asym,
                "qfi_exact_spectral": qfi_spectral,
                "qfi_finite_bures_response": qfi_bures,
                "spectral_relative_error_vs_asymptotic": spectral_rel_error,
                "bures_relative_error_vs_asymptotic": bures_rel_error,
                "bures_minus_spectral_relative_gap": witness_internal_rel_gap,
                "spectral_abs_relative_error": spectral_abs,
                "bures_abs_relative_error": bures_abs,
                "witness_internal_abs_relative_gap": internal_abs,
                "dual_witness_pass_10pct": int(dual_pass),
                "state_model": "postselected_MrhoM" if exact_postselection else "fixed_thermal_weights_component_norm",
            })
    return rows


def summarize_dual_witness(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Summarize the dual-witness check by eta."""
    by_eta: Dict[float, List[Dict[str, float]]] = {}
    for row in rows:
        by_eta.setdefault(float(row["eta_alpha2_over_2nbar_plus_1"]), []).append(row)

    summary_rows: List[Dict[str, float]] = []
    for eta, eta_rows in sorted(by_eta.items()):
        n = len(eta_rows)
        pass_count = sum(int(r["dual_witness_pass_10pct"]) for r in eta_rows)
        summary_rows.append({
            "eta_alpha2_over_2nbar_plus_1": eta,
            "n_rows": n,
            "dual_witness_pass_fraction_10pct": pass_count / n,
            "max_spectral_abs_relative_error": max(float(r["spectral_abs_relative_error"]) for r in eta_rows),
            "mean_spectral_abs_relative_error": sum(float(r["spectral_abs_relative_error"]) for r in eta_rows) / n,
            "max_bures_abs_relative_error": max(float(r["bures_abs_relative_error"]) for r in eta_rows),
            "mean_bures_abs_relative_error": sum(float(r["bures_abs_relative_error"]) for r in eta_rows) / n,
            "max_witness_internal_abs_relative_gap": max(float(r["witness_internal_abs_relative_gap"]) for r in eta_rows),
            "mean_witness_internal_abs_relative_gap": sum(float(r["witness_internal_abs_relative_gap"]) for r in eta_rows) / n,
        })
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_make_plots(outdir: Path, protocol_rows: List[Dict[str, float]], dual_rows: List[Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plots; matplotlib import failed: {exc}")
        return

    # Plot formula error by eta for both witnesses.
    fig_path = outdir / "dual_witness_relative_error.png"
    by_eta = summarize_dual_witness(dual_rows)
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        [r["eta_alpha2_over_2nbar_plus_1"] for r in by_eta],
        [r["max_spectral_abs_relative_error"] for r in by_eta],
        marker="o",
        label="spectral QFI",
    )
    plt.plot(
        [r["eta_alpha2_over_2nbar_plus_1"] for r in by_eta],
        [r["max_bures_abs_relative_error"] for r in by_eta],
        marker="s",
        label="finite Bures response",
    )
    plt.yscale("log")
    plt.xlabel(r"eta = alpha^2 / (2 nbar + 1)")
    plt.ylabel("max absolute relative error vs branch formula")
    plt.title("Dual witness check of the ECD branch formula")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    # Plot best rate versus cooling time by maximizing over cat time.
    fig_path = outdir / "fig2_like_rate_profile.png"
    best_by_cool: Dict[float, Dict[str, float]] = {}
    for row in protocol_rows:
        tcool = float(row["cool_time"])
        if tcool not in best_by_cool or row["qfi_per_total_prep_time"] > best_by_cool[tcool]["qfi_per_total_prep_time"]:
            best_by_cool[tcool] = row
    rows = [best_by_cool[k] for k in sorted(best_by_cool)]
    plt.figure(figsize=(7, 4.5))
    plt.plot([r["cool_time"] for r in rows], [r["qfi_per_total_prep_time"] for r in rows], marker="o", markersize=3)
    plt.xlabel("cooling time")
    plt.ylabel("max QFI / total prep time")
    plt.title("Reduced Fig. 2-like rate profile")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def write_summary(outdir: Path, best: Dict[str, float], dual_summary_rows: List[Dict[str, float]]) -> None:
    eta_lines = []
    for row in dual_summary_rows:
        eta_lines.append(
            "| {eta:g} | {pass_frac:.3g} | {spec_max:.6g} | {bures_max:.6g} | {gap_max:.6g} |".format(
                eta=row["eta_alpha2_over_2nbar_plus_1"],
                pass_frac=row["dual_witness_pass_fraction_10pct"],
                spec_max=row["max_spectral_abs_relative_error"],
                bures_max=row["max_bures_abs_relative_error"],
                gap_max=row["max_witness_internal_abs_relative_gap"],
            )
        )

    summary = f"""# Probe summary

## Paper-matched reduced protocol check

Using the trapped-ion parameters quoted around Fig. 2:

- initial thermal occupation nbar0 = 10
- cooling rate = 1
- motional heating rate = 1e-3
- spin dephasing rate = 1e-2

Best grid point for the simplified objective QFI/(cool_time + cat_time):

| cool_time | cat_time | QFI | QFI / total prep time | post-cooling nbar |
| ---: | ---: | ---: | ---: | ---: |
| {best['cool_time']:.6g} | {best['cat_time']:.6g} | {best['qfi_reduced']:.6g} | {best['qfi_per_total_prep_time']:.6g} | {best['post_cooling_nbar']:.6g} |

Interpretation: the paper-level reduced model gives an optimum near zero cooling and finite cat-generation time for these parameters.

## Dual-witness ECD branch-formula check

The branch formula is

    F_Q ~= 16 alpha^2 + 4/(2 nbar + 1)

This probe now checks that formula with two witnesses:

1. exact mixed-state spectral QFI in a finite Fock cutoff;
2. finite-displacement Bures response, converted to a local QFI estimate by
   `8 * (1 - sqrt(fidelity(rho, rho_delta))) / delta^2`.

The control parameter is eta = alpha^2/(2 nbar + 1). Smaller eta means stronger branch overlap / weaker validity of the branch-pseudospin approximation.

| eta | dual pass fraction at 10pct tolerance | max spectral rel. error | max Bures-response rel. error | max internal witness gap |
| ---: | ---: | ---: | ---: | ---: |
{chr(10).join(eta_lines)}

The full row-by-row output is in `dual_witness_ecd_sweep.csv`.

## Minimal claim boundary

This is not a falsification test of hot-state sensing. It is a sufficiency check for the reduced ECD branch formula.

A strong result is when both witnesses agree with the branch formula and with each other. A boundary result is when the branch formula looks acceptable under one witness but not the other, or when both fail in the low-eta branch-overlap regime.
"""
    (outdir / "probe_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal hot-ECD QFI sandbox/probe")
    parser.add_argument("--outdir", default="analysis", help="output directory")
    parser.add_argument("--max-cutoff", type=int, default=150, help="maximum finite Fock cutoff for exact QFI sweep")
    parser.add_argument("--finite-signal-delta", type=float, default=1e-3, help="small displacement used for the Bures-response witness")
    parser.add_argument("--pass-tol", type=float, default=0.10, help="relative-error tolerance used for the dual-witness pass flag")
    parser.add_argument("--component-normalized", action="store_true", help="use fixed thermal weights with normalized cat components instead of exact postselected map")
    parser.add_argument("--plots", action="store_true", help="also write two simple PNG plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    protocol_rows = fig2_like_protocol_grid()
    write_csv(outdir / "fig2_like_protocol_grid.csv", protocol_rows)
    best = max(protocol_rows, key=lambda r: float(r["qfi_per_total_prep_time"]))
    write_csv(outdir / "fig2_like_best_point.csv", [best])

    dual_rows = dual_witness_sweep(
        max_cutoff=args.max_cutoff,
        exact_postselection=not args.component_normalized,
        finite_signal_delta=args.finite_signal_delta,
        pass_tol=args.pass_tol,
    )
    dual_summary_rows = summarize_dual_witness(dual_rows)
    write_csv(outdir / "dual_witness_ecd_sweep.csv", dual_rows)
    write_csv(outdir / "dual_witness_summary.csv", dual_summary_rows)

    # Backward-compatible alias for the first version of the sandbox.
    write_csv(outdir / "exact_ecd_qfi_sweep.csv", dual_rows)

    write_summary(outdir, best, dual_summary_rows)
    if args.plots:
        maybe_make_plots(outdir, protocol_rows, dual_rows)

    print("Wrote:")
    for path in sorted(outdir.iterdir()):
        print(f"  {path}")
    print("\nBest reduced-model point:")
    for key in ("cool_time", "cat_time", "qfi_reduced", "qfi_per_total_prep_time", "post_cooling_nbar"):
        print(f"  {key}: {best[key]:.6g}")

    print("\nDual-witness summary:")
    for row in dual_summary_rows:
        print(
            "  eta={eta:g}: pass={pass_frac:.3g}, max_spectral_err={spec:.3g}, max_bures_err={bures:.3g}, max_internal_gap={gap:.3g}".format(
                eta=row["eta_alpha2_over_2nbar_plus_1"],
                pass_frac=row["dual_witness_pass_fraction_10pct"],
                spec=row["max_spectral_abs_relative_error"],
                bures=row["max_bures_abs_relative_error"],
                gap=row["max_witness_internal_abs_relative_gap"],
            )
        )


if __name__ == "__main__":
    main()
