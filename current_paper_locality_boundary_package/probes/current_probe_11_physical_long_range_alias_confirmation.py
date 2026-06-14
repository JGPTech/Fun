#!/usr/bin/env python3
# current_probe_11_physical_long_range_alias_confirmation.py
#
# Capstone probe: consume Probe 10 locality-alias rows and attach neutral
# Coulomb-like long-range physical witnesses to the same aliased configuration
# pairs.
#
# Allowed claim:
#   In controlled toy configurations, identical extracted local/top-n signatures
#   can correspond to different neutral long-range physical target witnesses.
#
# Not allowed:
#   This is not a full-paper reproduction, not a training run, and not proof that
#   the paper's benchmark results are wrong.
#
# Run:
#   python .\current_probe_11_physical_long_range_alias_confirmation.py
# Optional:
#   python .\current_probe_11_physical_long_range_alias_confirmation.py --probe10-dir .\probe_runs\current_probe10_locality_collision_alias_YYYYMMDD_HHMMSS

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent

MOTIF = np.array([
    [0.00, 0.00, 0.00],
    [1.00, 0.13, 0.02],
    [0.21, 1.31, 0.07],
    [0.11, 0.27, 1.73],
], dtype=float)

TRANSLATIONS: Dict[str, Tuple[float, float, float]] = {
    "sep20_x": (20.0, 0.0, 0.0),
    "sep45_x": (45.0, 0.0, 0.0),
    "sep20_y": (0.0, 20.0, 0.0),
    "sep20_diag_xy": (14.1421356237, 14.1421356237, 0.0),
    "sep20_z": (0.0, 0.0, 20.0),
}

# Zero net charge per 4-particle motif. These are controlled long-range target
# functions, not claims about the paper's actual material systems.
CHARGE_MODELS: Dict[str, List[float]] = {
    "neutral_dipole": [1.0, -1.0, 0.5, -0.5],
    "neutral_quadrupole": [1.0, -1.0, -1.0, 1.0],
    "neutral_asymmetric": [0.75, -1.0, 0.55, -0.30],
}

PRIMARY_KEYS = [
    "cross_coulomb_energy",
    "cross_abs_pair_force_sum",
    "net_force_A_mag",
    "particle_force_l1_on_A",
    "torque_A_mag",
    "field_at_centroid_A_mag_due_B",
]

ALL_DELTA_KEYS = PRIMARY_KEYS + [
    "potential_at_centroid_A_due_B",
    "field_at_centroid_A_parallel_due_B",
    "dipole_mag",
    "cov_xx",
    "cov_yy",
    "cov_zz",
    "cov_xy",
    "cov_xz",
    "cov_yz",
    "cov_anisotropy",
]


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def resolve(p: str | Path) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else ROOT / q


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    mkdir(path.parent)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def latest_probe10(out_root: Path) -> Path:
    hits = sorted([p for p in out_root.glob("current_probe10_locality_collision_alias_*") if p.is_dir()], key=lambda p: p.name)
    if not hits:
        raise FileNotFoundError(f"No Probe 10 folders found under {rel(out_root)}")
    return hits[-1]


def table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return "\n".join(lines)


def make_case(name: str) -> Tuple[np.ndarray, List[str]]:
    if name not in TRANSLATIONS:
        raise KeyError(f"Unknown Probe 10 case: {name}")
    t = np.asarray(TRANSLATIONS[name], dtype=float)
    a = MOTIF.copy()
    b = MOTIF + t[None, :]
    return np.vstack([a, b]), ["A"] * 4 + ["B"] * 4


def charges(model: str) -> np.ndarray:
    q4 = np.asarray(CHARGE_MODELS[model], dtype=float)
    if abs(float(np.sum(q4))) > 1e-12:
        raise ValueError(f"Charge model is not neutral: {model}")
    return np.concatenate([q4, q4])


def vnorm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n == 0 else v / n


def physical_witness(case: str, model: str, eps: float = 1e-12) -> Dict[str, Any]:
    pos, labels = make_case(case)
    q = charges(model)
    ia = [i for i, lab in enumerate(labels) if lab == "A"]
    ib = [i for i, lab in enumerate(labels) if lab == "B"]
    ca = pos[ia].mean(axis=0)
    cb = pos[ib].mean(axis=0)
    sep_vec = cb - ca
    sep = vnorm(sep_vec)
    sep_u = unit(sep_vec)

    cross_e = 0.0
    f_on_a = np.zeros((len(ia), 3), dtype=float)
    abs_pair_force_sum = 0.0
    abs_charge_inv_r = 0.0
    for local_i, i in enumerate(ia):
        for j in ib:
            rij = pos[i] - pos[j]
            r = vnorm(rij) + eps
            qq = float(q[i] * q[j])
            cross_e += qq / r
            fij = qq * rij / (r ** 3)
            f_on_a[local_i] += fij
            abs_pair_force_sum += vnorm(fij)
            abs_charge_inv_r += abs(qq) / r

    net_f = f_on_a.sum(axis=0)
    torque = np.zeros(3, dtype=float)
    for local_i, i in enumerate(ia):
        torque += np.cross(pos[i] - ca, f_on_a[local_i])

    field = np.zeros(3, dtype=float)
    potential = 0.0
    for j in ib:
        rv = ca - pos[j]
        r = vnorm(rv) + eps
        potential += q[j] / r
        field += q[j] * rv / (r ** 3)

    centered = pos - pos.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / len(pos)
    eig = np.sort(np.linalg.eigvalsh(cov))
    anis = float((eig[-1] - eig[0]) / (float(np.trace(cov)) + eps))
    dipole = np.sum(q[:, None] * pos, axis=0)

    return {
        "case": case,
        "charge_model": model,
        "total_charge": float(np.sum(q)),
        "cluster_charge_A": float(np.sum(q[ia])),
        "cluster_charge_B": float(np.sum(q[ib])),
        "cluster_separation": sep,
        "separation_x": float(sep_vec[0]),
        "separation_y": float(sep_vec[1]),
        "separation_z": float(sep_vec[2]),
        "cross_coulomb_energy": float(cross_e),
        "cross_abs_charge_inv_r_sum": float(abs_charge_inv_r),
        "cross_abs_pair_force_sum": float(abs_pair_force_sum),
        "net_force_A_x": float(net_f[0]),
        "net_force_A_y": float(net_f[1]),
        "net_force_A_z": float(net_f[2]),
        "net_force_A_mag": vnorm(net_f),
        "net_force_A_parallel_to_sep": float(np.dot(net_f, sep_u)),
        "particle_force_l1_on_A": float(np.sum(np.linalg.norm(f_on_a, axis=1))),
        "torque_A_x": float(torque[0]),
        "torque_A_y": float(torque[1]),
        "torque_A_z": float(torque[2]),
        "torque_A_mag": vnorm(torque),
        "potential_at_centroid_A_due_B": float(potential),
        "field_at_centroid_A_x_due_B": float(field[0]),
        "field_at_centroid_A_y_due_B": float(field[1]),
        "field_at_centroid_A_z_due_B": float(field[2]),
        "field_at_centroid_A_mag_due_B": vnorm(field),
        "field_at_centroid_A_parallel_due_B": float(np.dot(field, sep_u)),
        "dipole_x": float(dipole[0]),
        "dipole_y": float(dipole[1]),
        "dipole_z": float(dipole[2]),
        "dipole_mag": vnorm(dipole),
        "cov_xx": float(cov[0, 0]),
        "cov_yy": float(cov[1, 1]),
        "cov_zz": float(cov[2, 2]),
        "cov_xy": float(cov[0, 1]),
        "cov_xz": float(cov[0, 2]),
        "cov_yz": float(cov[1, 2]),
        "cov_anisotropy": anis,
    }


def compare_witness(a: Dict[str, Any], b: Dict[str, Any], tol: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    primary_keys = []
    all_keys = []
    primary_max = 0.0
    all_max = 0.0
    for k in ALL_DELTA_KEYS:
        va = float(a[k]); vb = float(b[k])
        d = vb - va
        ad = abs(d)
        out[f"{k}_a"] = va
        out[f"{k}_b"] = vb
        out[f"{k}_delta_signed"] = d
        out[f"{k}_delta_abs"] = ad
        out[f"{k}_rel_delta"] = ad / (abs(va) + 1e-12)
        all_max = max(all_max, ad)
        if ad > tol:
            all_keys.append(k)
        if k in PRIMARY_KEYS:
            primary_max = max(primary_max, ad)
            if ad > tol:
                primary_keys.append(k)
    out["max_physical_delta_abs"] = all_max
    out["primary_physical_delta_abs"] = primary_max
    out["physical_delta_nonzero"] = all_max > tol
    out["primary_physical_delta_nonzero"] = primary_max > tol
    out["physical_delta_keys"] = ";".join(all_keys)
    out["primary_physical_delta_keys"] = ";".join(primary_keys)
    return out


def build_claims(alias_rows_total: int, alias_rows: List[Dict[str, str]], confirmations: List[Dict[str, Any]], group_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    confirmed = [r for r in confirmations if r["physical_alias_confirmed"]]
    groups_any = [r for r in group_rows if r["physical_alias_confirmed_any_model"]]
    all_groups_confirmed = len(group_rows) > 0 and len(groups_any) == len(group_rows)
    return [
        {
            "claim_id": "P11-C01",
            "result": "PASS" if alias_rows else "FAIL",
            "claim": "Probe 10 produced local/top-n alias rows suitable for physical confirmation.",
            "evidence": f"{len(alias_rows)}/{alias_rows_total} Probe 10 comparison rows had alias_detected=True.",
            "allowed_interpretation": "The capstone has local alias pairs to test.",
            "not_allowed_interpretation": "The full paper has failed.",
        },
        {
            "claim_id": "P11-C02",
            "result": "PASS" if confirmed else "FAIL",
            "claim": "At least one local/top-n alias pair is separated by a neutral Coulomb-like physical witness.",
            "evidence": f"{len(confirmed)}/{len(confirmations)} charge-model confirmation rows had physical_alias_confirmed=True.",
            "allowed_interpretation": "A fixed local representation can alias a controlled long-range physical target.",
            "not_allowed_interpretation": "The benchmark systems necessarily contain this alias.",
        },
        {
            "claim_id": "P11-C03",
            "result": "PASS" if all_groups_confirmed else "WARN",
            "claim": "Every Probe 10 alias pair/n_local group is physically separated by at least one neutral long-range witness.",
            "evidence": f"{len(groups_any)}/{len(group_rows)} alias pair/n_local groups had physical_alias_confirmed_any_model=True.",
            "allowed_interpretation": "The capstone witness is robust across the tested alias groups if PASS.",
            "not_allowed_interpretation": "All possible local aliases are physically important.",
        },
        {
            "claim_id": "P11-C04",
            "result": "SUPPORTED_IN_CONTROLLED_WITNESS" if confirmed else "NOT_SUPPORTED",
            "claim": "Specific mechanism identified: finite local/top-n neighborhoods can erase long-range physical target information.",
            "evidence": "Rows require Probe10 alias_detected=True plus a nonzero primary neutral long-range physical delta.",
            "allowed_interpretation": "This is the candidate correction target: add/check nonlocal, multiscale, or global conditioning channels.",
            "not_allowed_interpretation": "The paper's reported results are invalid.",
        },
        {
            "claim_id": "P11-C05",
            "result": "INFO",
            "claim": "Paper implication is conditional.",
            "evidence": "This probe does not train or run the full BG; it provides a minimal counterexample class for fixed local representations.",
            "allowed_interpretation": "Further full-model tests should check whether the architecture or target systems avoid this alias.",
            "not_allowed_interpretation": "This toy witness is a reproduction failure.",
        },
    ]


def write_summary(path: Path, out: Path, probe10: Path, claims: List[Dict[str, Any]], confirmations: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> None:
    confirmed = [r for r in confirmations if r["physical_alias_confirmed"]]
    any_groups = [r for r in groups if r["physical_alias_confirmed_any_model"]]
    txt = [
        "# PROBE 11 SUMMARY",
        "",
        f"Output folder: `{rel(out)}`",
        f"Probe 10 source: `{rel(probe10)}`",
        "",
        "## Capstone result",
        "",
        "Probe 11 attaches neutral Coulomb-like long-range target witnesses to the local/top-n alias pairs from Probe 10.",
        "",
        "Allowed claim:",
        "",
        "> In controlled toy configurations, identical extracted local/top-n signatures can correspond to different neutral long-range physical target witnesses.",
        "",
        "Forbidden overclaim:",
        "",
        "> The paper is wrong, reproduced-failed, or invalidated.",
        "",
        "## Headline counts",
        "",
        f"- Physical confirmation rows: **{len(confirmations)}**",
        f"- Confirmed physical alias rows: **{len(confirmed)}**",
        f"- Alias pair/n_local groups: **{len(groups)}**",
        f"- Groups confirmed by at least one charge model: **{len(any_groups)}**",
        "",
        "## Claim table",
        "",
        table(["claim_id", "result", "claim", "evidence"], [[c["claim_id"], c["result"], c["claim"], c["evidence"]] for c in claims]),
        "",
    ]
    path.write_text("\n".join(txt), encoding="utf-8")


def write_report(path: Path, out: Path, probe10: Path, claims: List[Dict[str, Any]], confirmations: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> None:
    confirmed = [r for r in confirmations if r["physical_alias_confirmed"]]
    any_groups = [r for r in groups if r["physical_alias_confirmed_any_model"]]
    all_groups = [r for r in groups if r["physical_alias_confirmed_all_models"]]
    strongest = sorted(confirmed, key=lambda r: float(r["primary_physical_delta_abs"]), reverse=True)[:12]
    lines = [
        "# PROBE 11 — Physical long-range alias confirmation",
        "",
        f"Output folder: `{rel(out)}`",
        f"Probe 10 source: `{rel(probe10)}`",
        "",
        "## Purpose",
        "",
        "Probe 10 showed representation-level locality collisions. Probe 11 is the capstone confirmation: it attaches neutral Coulomb-like long-range target witnesses to those same alias pairs.",
        "",
        "## Main conclusion",
        "",
        "> Fixed local/top-n neighborhoods can alias globally distinct configurations that differ under neutral long-range physical target witnesses.",
        "",
        "This is a controlled mechanism-boundary result, not a full-paper reproduction.",
        "",
        "## Claim table",
        "",
        table(["claim_id", "result", "claim", "allowed interpretation"], [[c["claim_id"], c["result"], c["claim"], c["allowed_interpretation"]] for c in claims]),
        "",
        "## Confirmation counts",
        "",
        f"- Physical confirmation rows: **{len(confirmations)}**",
        f"- Confirmed physical alias rows: **{len(confirmed)}**",
        f"- Alias pair/n_local groups: **{len(groups)}**",
        f"- Groups confirmed by at least one neutral charge model: **{len(any_groups)}**",
        f"- Groups confirmed by all neutral charge models: **{len(all_groups)}**",
        "",
        "## Pair/n_local group summary",
        "",
        table(["case_a", "case_b", "n_local", "models tested", "models confirmed", "confirmed models", "any?", "all?"], [[g["case_a"], g["case_b"], g["n_local"], g["charge_models_tested"], g["charge_models_confirmed"], g["confirmed_charge_models"], g["physical_alias_confirmed_any_model"], g["physical_alias_confirmed_all_models"]] for g in groups]),
        "",
        "## Strongest confirmation rows",
        "",
        table(["case_a", "case_b", "n_local", "charge_model", "confirmed?", "primary Δ", "primary delta keys", "Coulomb Δ", "force-sum Δ", "field-mag Δ", "torque Δ"], [[r["case_a"], r["case_b"], r["n_local"], r["charge_model"], r["physical_alias_confirmed"], r["primary_physical_delta_abs"], r["primary_physical_delta_keys"], r["cross_coulomb_energy_delta_abs"], r["cross_abs_pair_force_sum_delta_abs"], r["field_at_centroid_A_mag_due_B_delta_abs"], r["torque_A_mag_delta_abs"]] for r in strongest]),
        "",
        "## Method",
        "",
        "Each Probe 10 alias row is reconstructed using the same two-cluster geometry. Three zero-net-charge models are assigned to each identical local motif. Probe 11 computes cross-cluster Coulomb-like energy, pair-force sums, net force, particle-force response, torque, and centroid field witnesses.",
        "",
        "Charge models:",
        "",
    ]
    for name, q in CHARGE_MODELS.items():
        lines.append(f"- `{name}`: `{q}`, net charge `{sum(q):.6g}`")
    lines += [
        "",
        "## What this proves",
        "",
        "This proves a controlled mechanism-boundary result: local/top-n representations can erase long-range physical target information.",
        "",
        "## What this does not prove",
        "",
        "This does not prove the full paper model fails. It does not run training, sampling, ESS, TFEP, free-energy calculation, source-data reproduction, or large-system transfer.",
        "",
        "## Correction target suggested by this witness",
        "",
        "For systems where long-range physics matters, a fixed local/top-n architecture should include or test at least one explicit nonlocal, multiscale, reciprocal-space, or global conditioning channel.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme_snippet(path: Path, claims: List[Dict[str, Any]]) -> None:
    txt = [
        "# README capstone snippet",
        "",
        "This repository does **not** claim to reproduce the full paper.",
        "",
        "It provides a staged mechanism-boundary audit of a fixed-local-neighborhood Boltzmann-generator architecture.",
        "",
        "Capstone result:",
        "",
        "> In controlled toy configurations, extracted local/top-n neighborhood signatures can remain identical while neutral long-range physical target witnesses change.",
        "",
        "This identifies a concrete failure mechanism to check:",
        "",
        "> finite local neighborhood aliasing.",
        "",
        "Conditional implication: if target observables depend on information outside the fixed local neighborhood, the architecture needs an explicit nonlocal, multiscale, reciprocal-space, or global conditioning channel, or a diagnostic showing such aliases are irrelevant for the target systems.",
        "",
        "## Claim boundary",
        "",
        table(["claim_id", "result", "claim"], [[c["claim_id"], c["result"], c["claim"]] for c in claims]),
        "",
    ]
    path.write_text("\n".join(txt), encoding="utf-8")


def write_packaging_plan(path: Path) -> None:
    txt = """# GitHub packaging plan

## Repository title candidates

- `bgmat-locality-boundary-probes`
- `current-paper-locality-boundary-audit`
- `local-bg-long-range-alias-audit`

## Recommended root files

```text
README.md
CLAIM_BOUNDARY.md
METHODS.md
RESULTS_SUMMARY.md
PROBE_INDEX.md
checkpoint_process_report_01-09.md
current_probe_01_claim_map_repo_audit.py
current_probe_02_source_data_audit.py
current_probe_03_official_repo_smoke.py
current_patch_compat_modern_deps.py
current_probe_04_module_seam_map.py
current_probe_05_tiny_energy_seam_probe.py
current_probe_06_tiny_cutoff_locality_stress.py
current_probe_07_clean_locality_witness.py
current_probe_08_neighbor_graph_locality_witness_v2.py
current_probe_09_joint_energy_neighbor_locality_report.py
current_probe_10_locality_collision_long_range_alias.py
current_probe_11_physical_long_range_alias_confirmation.py
probe_runs/
```

## README first-screen caveat

```text
This repository does not reproduce the full paper. It provides a staged mechanism-boundary audit of a fixed-local-neighborhood Boltzmann-generator architecture. The capstone probes show controlled cases where identical extracted local/top-n signatures correspond to different neutral long-range physical target witnesses.
```

## Must not claim

```text
The paper is wrong.
The paper has been reproduced-failed.
The benchmark figures are invalid.
```

## Safe claim

```text
The probes identify a specific long-range aliasing mechanism that should be checked when fixed-local-neighborhood Boltzmann generators are applied to systems whose observables depend on nonlocal structure.
```
"""
    path.write_text(txt, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe 11: physical long-range alias confirmation.")
    ap.add_argument("--probe10-dir", default="", help="Probe 10 output folder. Defaults to latest under --out-root.")
    ap.add_argument("--out-root", default="probe_runs")
    ap.add_argument("--tol", type=float, default=1e-12)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = resolve(args.out_root)
    probe10 = resolve(args.probe10_dir) if args.probe10_dir else latest_probe10(out_root)
    alias_csv = probe10 / "alias_comparisons.csv"
    if not alias_csv.exists():
        raise FileNotFoundError(f"Missing Probe 10 alias file: {rel(alias_csv)}")

    out = mkdir(out_root / f"current_probe11_physical_long_range_alias_{stamp()}")
    print("=" * 100)
    print("CURRENT PAPER PROBE 11 — PHYSICAL LONG-RANGE ALIAS CONFIRMATION")
    print("=" * 100)
    print(f"[PROBE 10] {rel(probe10)}")
    print(f"[ALIAS CSV] {rel(alias_csv)}")
    print(f"[OUT]      {rel(out)}")

    all_alias = read_csv(alias_csv)
    alias_rows = [r for r in all_alias if truthy(r.get("alias_detected"))]
    cases = sorted({c for r in alias_rows for c in [r.get("case_a", ""), r.get("case_b", "")] if c})
    unknown = [c for c in cases if c not in TRANSLATIONS]
    if unknown:
        raise KeyError(f"Unknown cases from Probe 10: {unknown}")

    witness_rows: List[Dict[str, Any]] = []
    witness: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for case in cases:
        for model in CHARGE_MODELS:
            w = physical_witness(case, model)
            witness_rows.append(w)
            witness[(case, model)] = w

    pair_models = sorted({(r.get("case_a", ""), r.get("case_b", ""), model) for r in alias_rows for model in CHARGE_MODELS})
    physical_pairs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    physical_pair_rows: List[Dict[str, Any]] = []
    for a, b, model in pair_models:
        comp = {"case_a": a, "case_b": b, "charge_model": model}
        comp.update(compare_witness(witness[(a, model)], witness[(b, model)], args.tol))
        physical_pairs[(a, b, model)] = comp
        physical_pair_rows.append(comp)

    confirmations: List[Dict[str, Any]] = []
    for ar in alias_rows:
        a = ar.get("case_a", "")
        b = ar.get("case_b", "")
        n_local = ar.get("n_local", "")
        for model in CHARGE_MODELS:
            pc = physical_pairs[(a, b, model)]
            primary_nonzero = bool(pc["primary_physical_delta_nonzero"])
            confirmations.append({
                "case_a": a,
                "case_b": b,
                "comparison": ar.get("comparison", ""),
                "n_local": n_local,
                "charge_model": model,
                "probe10_alias_detected": True,
                "local_graph_same": truthy(ar.get("local_graph_same")),
                "local_distance_shell_same": truthy(ar.get("local_distance_shell_same")),
                "internal_cluster_signature_same": truthy(ar.get("internal_cluster_signature_same")),
                "both_topn_graphs_internal_only": truthy(ar.get("both_topn_graphs_internal_only")),
                "physical_alias_confirmed": primary_nonzero,
                "primary_physical_delta_abs": pc["primary_physical_delta_abs"],
                "max_physical_delta_abs": pc["max_physical_delta_abs"],
                "primary_physical_delta_keys": pc["primary_physical_delta_keys"],
                "physical_delta_keys": pc["physical_delta_keys"],
                "cross_coulomb_energy_delta_abs": pc["cross_coulomb_energy_delta_abs"],
                "cross_abs_pair_force_sum_delta_abs": pc["cross_abs_pair_force_sum_delta_abs"],
                "net_force_A_mag_delta_abs": pc["net_force_A_mag_delta_abs"],
                "particle_force_l1_on_A_delta_abs": pc["particle_force_l1_on_A_delta_abs"],
                "torque_A_mag_delta_abs": pc["torque_A_mag_delta_abs"],
                "field_at_centroid_A_mag_due_B_delta_abs": pc["field_at_centroid_A_mag_due_B_delta_abs"],
                "cov_xx_delta_abs": pc["cov_xx_delta_abs"],
                "cov_yy_delta_abs": pc["cov_yy_delta_abs"],
                "cov_zz_delta_abs": pc["cov_zz_delta_abs"],
                "cov_xy_delta_abs": pc["cov_xy_delta_abs"],
                "cov_xz_delta_abs": pc["cov_xz_delta_abs"],
                "cov_yz_delta_abs": pc["cov_yz_delta_abs"],
            })

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in confirmations:
        grouped.setdefault((r["case_a"], r["case_b"], r["n_local"]), []).append(r)
    group_rows: List[Dict[str, Any]] = []
    for (a, b, n), rs in sorted(grouped.items()):
        confirmed_models = [r["charge_model"] for r in rs if r["physical_alias_confirmed"]]
        group_rows.append({
            "case_a": a,
            "case_b": b,
            "n_local": n,
            "charge_models_tested": len(rs),
            "charge_models_confirmed": len(confirmed_models),
            "confirmed_charge_models": ";".join(confirmed_models),
            "physical_alias_confirmed_any_model": len(confirmed_models) > 0,
            "physical_alias_confirmed_all_models": len(confirmed_models) == len(rs) and len(rs) > 0,
            "max_primary_delta_abs": max(float(r["primary_physical_delta_abs"]) for r in rs) if rs else 0.0,
        })

    claims = build_claims(len(all_alias), alias_rows, confirmations, group_rows)

    write_csv(out / "physical_witness_by_case.csv", witness_rows)
    write_json(out / "physical_witness_by_case.json", witness_rows)
    write_csv(out / "physical_pair_comparisons.csv", physical_pair_rows)
    write_json(out / "physical_pair_comparisons.json", physical_pair_rows)
    write_csv(out / "physical_alias_confirmations.csv", confirmations)
    write_json(out / "physical_alias_confirmations.json", confirmations)
    write_csv(out / "physical_alias_pair_summary.csv", group_rows)
    write_json(out / "physical_alias_pair_summary.json", group_rows)
    write_csv(out / "capstone_claim_table.csv", claims)
    write_json(out / "capstone_claim_table.json", claims)
    write_summary(out / "PROBE11_SUMMARY.md", out, probe10, claims, confirmations, group_rows)
    write_report(out / "PROBE11_PHYSICAL_ALIAS_CONFIRMATION_REPORT.md", out, probe10, claims, confirmations, group_rows)
    write_readme_snippet(out / "README_capstone_snippet.md", claims)
    write_packaging_plan(out / "github_packaging_plan.md")

    confirmed = [r for r in confirmations if r["physical_alias_confirmed"]]
    any_groups = [r for r in group_rows if r["physical_alias_confirmed_any_model"]]
    all_groups = [r for r in group_rows if r["physical_alias_confirmed_all_models"]]

    print("\n[SUMMARY]")
    print(f"  probe10 rows total                 : {len(all_alias)}")
    print(f"  probe10 alias rows                 : {len(alias_rows)}")
    print(f"  reconstructed cases                : {len(cases)}")
    print(f"  charge models                      : {len(CHARGE_MODELS)}")
    print(f"  physical witness rows              : {len(witness_rows)}")
    print(f"  physical confirmation rows         : {len(confirmations)}")
    print(f"  physical alias confirmed rows      : {len(confirmed)}")
    print(f"  alias pair/n_local groups          : {len(group_rows)}")
    print(f"  groups confirmed by any model      : {len(any_groups)}")
    print(f"  groups confirmed by all models     : {len(all_groups)}")
    print(f"  output                             : {rel(out)}")
    print("\n[CLAIMS]")
    for c in claims:
        print(f"  {c['claim_id']} {c['result']}: {c['evidence']}")
    print("\nNext:")
    print(f"  open {rel(out / 'PROBE11_SUMMARY.md')}")
    print(f"  open {rel(out / 'PROBE11_PHYSICAL_ALIAS_CONFIRMATION_REPORT.md')}")
    print(f"  open {rel(out / 'capstone_claim_table.csv')}")
    print(f"  open {rel(out / 'README_capstone_snippet.md')}")


if __name__ == "__main__":
    main()
