#!/usr/bin/env python3
# current_probe_09_joint_energy_neighbor_locality_report.py
#
# Probe 09: joint energy + neighbor-graph locality report.
#
# It consumes the latest Probe 07 and Probe 08 output folders by default:
#   probe_runs/current_probe07_clean_locality_witness_*
#   probe_runs/current_probe08_neighbor_graph_locality_*
#
# Or pass explicit folders:
#   python .\current_probe_09_joint_energy_neighbor_locality_report.py --probe07-dir <dir> --probe08-dir <dir>
#
# No training. No imports from bgmat required. Stdlib only.

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def latest_dir(prefix: str, root: Path) -> Path:
    candidates = sorted([p for p in root.glob(prefix + "*") if p.is_dir()], key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No folders found under {rel(root)} matching {prefix}*")
    return candidates[-1]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        seen = set()
        fieldnames = []
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def fnum(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def near_zero(x: Any, tol: float) -> bool:
    v = fnum(x)
    return v is not None and abs(v) <= tol


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def summarize_energy(rows: List[Dict[str, str]], tol: float) -> Dict[str, Any]:
    far = [r for r in rows if r.get("case") == "third_far_all" and truthy(r.get("ok")) and truthy(r.get("third_expected_blind_to_existing_particles"))]
    far_zero = [r for r in far if near_zero(r.get("delta_from_base"), tol)]

    inside = [r for r in rows if str(r.get("case", "")).startswith("third_inside") and truthy(r.get("ok"))]
    inside_nonzero = [r for r in inside if not near_zero(r.get("delta_from_base"), tol)]

    return {
        "far_cases": len(far),
        "far_zero": len(far_zero),
        "far_pass": len(far) > 0 and len(far) == len(far_zero),
        "inside_cases": len(inside),
        "inside_nonzero": len(inside_nonzero),
        "inside_pass": len(inside) > 0 and len(inside) == len(inside_nonzero),
        "far_rows": far,
        "inside_rows": inside,
    }


def base_node_counts(neighbor_rows: List[Dict[str, str]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for r in neighbor_rows:
        if r.get("case") == "base3" and truthy(r.get("ok")):
            n_local = int(fnum(r.get("n_local"), 0) or 0)
            n_nodes = int(fnum(r.get("n_nodes"), 0) or 0)
            out[n_local] = n_nodes
    return out


def summarize_neighbor(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    base_counts = base_node_counts(rows)

    comparable = []
    saturated = []
    for r in rows:
        n_local = int(fnum(r.get("n_local"), 0) or 0)
        base_n = base_counts.get(n_local)
        sat = base_n is not None and n_local >= base_n
        rr = dict(r)
        rr["base_n_nodes"] = base_n
        rr["saturated_n_local"] = sat
        if sat:
            saturated.append(rr)
        else:
            comparable.append(rr)

    far = [r for r in comparable if r.get("case") == "add_far_node" and truthy(r.get("ok"))]
    far_unchanged = [r for r in far if truthy(r.get("existing_edges_unchanged_vs_base"))]

    near = [r for r in comparable if str(r.get("case", "")).startswith("add_near") and truthy(r.get("ok"))]
    near_sensitive = [
        r for r in near
        if (not truthy(r.get("existing_edges_unchanged_vs_base"))) or bool(str(r.get("edges_touching_new_node", "")).strip())
    ]

    return {
        "base_counts": base_counts,
        "comparable_rows": len(comparable),
        "saturated_rows": len(saturated),
        "far_cases": len(far),
        "far_unchanged": len(far_unchanged),
        "far_pass": len(far) > 0 and len(far) == len(far_unchanged),
        "near_cases": len(near),
        "near_sensitive": len(near_sensitive),
        "near_pass": len(near) > 0 and len(near) == len(near_sensitive),
        "far_rows": far,
        "near_rows": near,
        "saturated_rows_detail": saturated,
    }


def build_joint_claim_rows(energy: Dict[str, Any], neighbor: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [
        {
            "claim_id": "J01",
            "claim": "Explicit LJ energy is blind to a third particle outside cutoff.",
            "result": "PASS" if energy["far_pass"] else "FAIL",
            "evidence": f'{energy["far_zero"]}/{energy["far_cases"]} far/blind cases had near-zero energy delta.',
        },
        {
            "claim_id": "J02",
            "claim": "Explicit LJ energy responds when a third particle enters the local cutoff neighborhood.",
            "result": "PASS" if energy["inside_pass"] else "FAIL",
            "evidence": f'{energy["inside_nonzero"]}/{energy["inside_cases"]} inside/local cases had nonzero energy delta.',
        },
        {
            "claim_id": "J03",
            "claim": "Model-side top-n neighbor graph is stable to far-node insertion for comparable n_local values.",
            "result": "PASS" if neighbor["far_pass"] else "FAIL",
            "evidence": f'{neighbor["far_unchanged"]}/{neighbor["far_cases"]} comparable far-node cases kept existing graph unchanged.',
        },
        {
            "claim_id": "J04",
            "claim": "Model-side top-n neighbor graph reacts to near-node insertion for comparable n_local values.",
            "result": "PASS" if neighbor["near_pass"] else "FAIL",
            "evidence": f'{neighbor["near_sensitive"]}/{neighbor["near_cases"]} comparable near-node cases touched/rewired the local graph.',
        },
        {
            "claim_id": "J05",
            "claim": "Saturated n_local values should be interpreted separately because adding a node changes the slot/self-loop regime.",
            "result": "INFO",
            "evidence": f'{neighbor["saturated_rows"]} saturated rows were excluded from pass/fail neighbor stability accounting.',
        },
    ]
    return rows


def write_report(path: Path, probe07: Path, probe08: Path, out_dir: Path, energy: Dict[str, Any], neighbor: Dict[str, Any], joint_rows: List[Dict[str, Any]], tol: float) -> None:
    lines = [
        "# PROBE 09 — Joint energy + neighbor locality report",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Probe 07 source: `{rel(probe07)}`",
        f"Probe 08 source: `{rel(probe08)}`",
        f"Near-zero tolerance: `{tol}`",
        "",
        "## Headline",
        "",
        "Probe 07 established hard cutoff behavior in the explicit LJ energy. Probe 08 established the model-side finite-neighbor graph behavior for comparable `n_local` values.",
        "",
        "The compact witness is:",
        "",
        "- far perturbations outside the local radius/top-neighbor set do not change the relevant local witness;",
        "- near perturbations inside the local radius/top-neighbor set do change it;",
        "- saturated `n_local` settings need separate handling because the graph has enough slots to include self/all nodes, so adding a node changes the combinatorial regime.",
        "",
        "## Joint claim table",
        "",
    ]

    lines += md_table(
        ["claim_id", "result", "claim", "evidence"],
        [[r["claim_id"], r["result"], r["claim"], r["evidence"]] for r in joint_rows]
    )

    lines += [
        "",
        "## Energy witness details",
        "",
        f"- Far/blind energy pass: `{energy['far_pass']}` ({energy['far_zero']}/{energy['far_cases']})",
        f"- Inside/local energy pass: `{energy['inside_pass']}` ({energy['inside_nonzero']}/{energy['inside_cases']})",
        "",
        "### Far/blind energy rows",
        "",
    ]

    lines += md_table(
        ["cutoff", "case", "delta", "ref delta", "abs err"],
        [[r.get("cutoff"), r.get("case"), r.get("delta_from_base"), r.get("reference_delta_from_base"), r.get("abs_error_vs_reference")] for r in energy["far_rows"]]
    )

    lines += [
        "",
        "### Inside/local energy rows",
        "",
    ]

    lines += md_table(
        ["cutoff", "case", "pairs", "delta", "ref delta", "abs err"],
        [[r.get("cutoff"), r.get("case"), r.get("third_interacting_pairs_expected"), r.get("delta_from_base"), r.get("reference_delta_from_base"), r.get("abs_error_vs_reference")] for r in energy["inside_rows"]]
    )

    lines += [
        "",
        "## Neighbor witness details",
        "",
        f"- Comparable far-node graph pass: `{neighbor['far_pass']}` ({neighbor['far_unchanged']}/{neighbor['far_cases']})",
        f"- Comparable near-node graph pass: `{neighbor['near_pass']}` ({neighbor['near_sensitive']}/{neighbor['near_cases']})",
        f"- Saturated rows excluded from pass/fail: `{neighbor['saturated_rows']}`",
        "",
        "### Comparable far-node rows",
        "",
    ]

    lines += md_table(
        ["n_local", "case", "unchanged?", "new-node edges", "nearest new-node distances"],
        [[r.get("n_local"), r.get("case"), r.get("existing_edges_unchanged_vs_base"), r.get("edges_touching_new_node"), r.get("nearest_existing_to_new_node")] for r in neighbor["far_rows"]]
    )

    lines += [
        "",
        "### Comparable near-node rows",
        "",
    ]

    lines += md_table(
        ["n_local", "case", "unchanged?", "new-node edges", "nearest new-node distances"],
        [[r.get("n_local"), r.get("case"), r.get("existing_edges_unchanged_vs_base"), r.get("edges_touching_new_node"), r.get("nearest_existing_to_new_node")] for r in neighbor["near_rows"]]
    )

    lines += [
        "",
        "### Saturated n_local rows",
        "",
        "These are not failures. They are excluded because the base graph is saturated: `n_local >= base_n_nodes`, so adding a node changes the all/self-slot regime.",
        "",
    ]

    lines += md_table(
        ["n_local", "case", "base_n_nodes", "edges", "existing graph unchanged?"],
        [[r.get("n_local"), r.get("case"), r.get("base_n_nodes"), r.get("edges"), r.get("existing_edges_unchanged_vs_base")] for r in neighbor["saturated_rows_detail"]]
    )

    lines += [
        "",
        "## Interpretation",
        "",
        "This is now a clean paper-seam witness, not a reproduction claim. It supports the mechanism-level statement that the code has finite local neighborhoods: far perturbations can be invisible to the local witness, while near perturbations can rewire or alter the local witness.",
        "",
        "## Suggested next probe",
        "",
        "Probe 10 should either install `e3nn_jax` and import the full `gnn_conditioner`, or keep using the shim and sweep larger random point clouds to measure neighbor-graph stability statistics under far-vs-near perturbations.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe 09: joint energy + neighbor locality report.")
    p.add_argument("--probe07-dir", default="", help="Probe 07 output folder. Defaults to latest.")
    p.add_argument("--probe08-dir", default="", help="Probe 08 output folder. Defaults to latest.")
    p.add_argument("--out-root", default="probe_runs")
    p.add_argument("--tol", type=float, default=1e-7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = resolve_from_root(args.out_root)
    probe07 = resolve_from_root(args.probe07_dir) if args.probe07_dir else latest_dir("current_probe07_clean_locality_witness_", out_root)
    probe08 = resolve_from_root(args.probe08_dir) if args.probe08_dir else latest_dir("current_probe08_neighbor_graph_locality_", out_root)

    out_dir = ensure_dir(out_root / f"current_probe09_joint_locality_report_{now_stamp()}")

    energy_rows = read_csv(probe07 / "clean_locality_witness.csv")
    neighbor_rows = read_csv(probe08 / "neighbor_graph_witness.csv")

    energy = summarize_energy(energy_rows, args.tol)
    neighbor = summarize_neighbor(neighbor_rows)
    joint_rows = build_joint_claim_rows(energy, neighbor)

    write_csv(out_dir / "joint_claim_table.csv", joint_rows)
    write_json(out_dir / "joint_claim_table.json", joint_rows)

    energy_summary = {k: v for k, v in energy.items() if not k.endswith("_rows")}
    neighbor_summary = {k: v for k, v in neighbor.items() if not k.endswith("_rows") and k != "saturated_rows_detail"}
    write_json(out_dir / "energy_summary.json", energy_summary)
    write_json(out_dir / "neighbor_summary.json", neighbor_summary)

    write_report(out_dir / "PROBE09_JOINT_LOCALITY_REPORT.md", probe07, probe08, out_dir, energy, neighbor, joint_rows, args.tol)

    print("=" * 100)
    print("CURRENT PAPER PROBE 09 — JOINT ENERGY + NEIGHBOR LOCALITY REPORT")
    print("=" * 100)
    print(f"[PROBE 07] {rel(probe07)}")
    print(f"[PROBE 08] {rel(probe08)}")
    print(f"[OUT]      {rel(out_dir)}")
    print()
    print("[SUMMARY]")
    for r in joint_rows:
        print(f"  {r['claim_id']} {r['result']}: {r['evidence']}")
    print()
    print("Next:")
    print(f"  open {rel(out_dir / 'PROBE09_JOINT_LOCALITY_REPORT.md')}")
    print(f"  open {rel(out_dir / 'joint_claim_table.csv')}")


if __name__ == "__main__":
    main()
