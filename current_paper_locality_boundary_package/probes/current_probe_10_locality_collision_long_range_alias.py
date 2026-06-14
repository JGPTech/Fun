#!/usr/bin/env python3
# current_probe_10_locality_collision_long_range_alias.py
#
# Probe 10: locality collision / long-range alias witness.
#
# Main objective:
#   Construct paired configurations that are locally indistinguishable to the
#   current paper's top-n local neighbor machinery, but different under
#   long-range/global witnesses.
#
# This is NOT a paper reproduction. This is a mechanism-boundary probe:
#   What does a fixed local/top-n representation erase?
#
# Run from repo root:
#   python .\current_probe_10_locality_collision_long_range_alias.py
#
# Optional:
#   python .\current_probe_10_locality_collision_long_range_alias.py --n-local 1 --n-local 2 --n-local 3
#
# No training. No e3nn_jax install required. Outputs under probe_runs/.

from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def resolve_from_root(p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str), encoding="utf-8")


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


def run_subprocess(code_text: str, official_repo: Path, timeout: int, log_dir: Path, label: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
    ensure_dir(log_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-c", code_text] + (args or [])

    rec: Dict[str, Any] = {
        "label": label,
        "cmd": cmd,
        "cwd": rel(REPO_ROOT),
        "timeout_s": timeout,
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": "",
    }
    try:
        p = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        rec.update(ok=(p.returncode == 0), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as exc:
        rec["error"] = repr(exc)

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")[:120]
    h = sha16(str(cmd))
    stdout_path = log_dir / f"{safe}_{h}.stdout.txt"
    stderr_path = log_dir / f"{safe}_{h}.stderr.txt"
    stdout_path.write_text(str(rec["stdout"]), encoding="utf-8")
    stderr_path.write_text(str(rec["stderr"]), encoding="utf-8")
    rec["stdout_log"] = rel(stdout_path)
    rec["stderr_log"] = rel(stderr_path)
    return rec


def parse_payload(result: Dict[str, Any], default: Any) -> Any:
    try:
        for line in reversed(str(result.get("stdout", "")).splitlines()):
            s = line.strip()
            if s.startswith("{") or s.startswith("["):
                return json.loads(s)
    except Exception:
        pass
    return default


def extract_function_sources(source_path: Path, names: Sequence[str]) -> Dict[str, str]:
    text = source_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(source_path))
    out: Dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segment = ast.get_source_segment(text, node)
            if segment:
                out[node.name] = segment
    return out


def write_shim_source(path: Path, fn_sources: Dict[str, str]) -> None:
    lines = [
        "# Extracted no-install shim for Probe 10.",
        "# Source: bgmat/bgmat/models/gnn_conditioner.py",
        "# Generated for auditability; only neighbor helper functions are included.",
        "from __future__ import annotations",
        "from functools import partial",
        "from typing import Any, Optional, Tuple",
        "import numpy as np",
        "import jax",
        "import jax.numpy as jnp",
        "Array = Any",
        "",
        "# Local helper shim: gnn_conditioner imports this from repo utilities.",
        "# For Probe 10 we only need periodic minimum-image displacement matrices.",
        "def pairwise_difference_pbc(x, y=None, box=None):",
        "    x = jnp.asarray(x)",
        "    if box is None:",
        "        box = y",
        "        y = x",
        "    elif y is None:",
        "        y = x",
        "    y = jnp.asarray(y)",
        "    box = jnp.asarray(box)",
        "    diff = x[..., :, None, :] - y[..., None, :, :]",
        "    return diff - box * jnp.round(diff / box)",
        "",
    ]
    for name, src in fn_sources.items():
        lines.append("# " + "=" * 78)
        lines.append(f"# {name}")
        lines.append("# " + "=" * 78)
        lines.append(src)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


COLLISION_WITNESS_CODE = r'''
import json
import math
import sys
import traceback
import numpy as np
import jax
import jax.numpy as jnp

shim_path = sys.argv[1]
n_local_values = [int(x) for x in sys.argv[2].split(",") if x.strip()]
round_digits = int(sys.argv[3])

namespace = {}
try:
    shim_source = open(shim_path, "r", encoding="utf-8").read()
    exec(compile(shim_source, shim_path, "exec"), namespace)
    get_top = namespace["get_top_n_connections_vectorized"]
except Exception as e:
    print(json.dumps({
        "case_rows": [],
        "local_signature_rows": [],
        "neighbor_edge_rows": [],
        "long_range_rows": [],
        "alias_comparison_rows": [{
            "ok": False,
            "stage": "shim_exec",
            "error": repr(e),
            "traceback_tail": traceback.format_exc()[-2000:],
        }],
        "pair_distance_rows": [],
    }, default=str))
    raise SystemExit(1)


def min_image_delta(a, b, box):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    box = np.asarray(box, dtype=float)
    return d - box * np.round(d / box)


def min_image_dist(a, b, box):
    return float(np.linalg.norm(min_image_delta(a, b, box)))


def pair_distance_matrix(positions, box):
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            out[i, j] = min_image_dist(pos[i], pos[j], box)
    return out


def selected_edges(senders, receivers):
    s = np.asarray(senders).reshape(-1).astype(int)
    r = np.asarray(receivers).reshape(-1).astype(int)
    return [(int(si), int(ri)) for si, ri in zip(s, r)]


def edge_signature(edges, dmat, digits):
    parts = []
    for i, j in edges:
        parts.append(f"{i}->{j}:{round(float(dmat[i, j]), digits)}")
    return ";".join(parts)


def local_distance_shell_signature(dmat, cluster_labels, n_local, digits):
    # Independent of senders/receivers: sorted nearest n_local distances per node.
    # This catches local metric aliasing even if graph tie-ordering changes.
    parts = []
    n = dmat.shape[0]
    for i in range(n):
        vals = []
        for j in range(n):
            if i == j:
                continue
            vals.append(float(dmat[i, j]))
        vals.sort()
        vals = vals[:n_local]
        parts.append(f"{i}|{cluster_labels[i]}:" + ",".join(str(round(v, digits)) for v in vals))
    return ";".join(parts)


def internal_cluster_distance_signature(dmat, cluster_labels, digits):
    parts = []
    labels = sorted(set(cluster_labels))
    for lab in labels:
        idx = [i for i, x in enumerate(cluster_labels) if x == lab]
        vals = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                vals.append(float(dmat[idx[a], idx[b]]))
        vals.sort()
        parts.append(f"{lab}:" + ",".join(str(round(v, digits)) for v in vals))
    return ";".join(parts)


def topn_internal_fraction(edges, cluster_labels):
    if not edges:
        return 0.0
    same = sum(1 for i, j in edges if cluster_labels[i] == cluster_labels[j])
    return same / len(edges)


def intercluster_distances(dmat, cluster_labels):
    vals = []
    n = dmat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_labels[i] != cluster_labels[j]:
                vals.append(float(dmat[i, j]))
    return vals


def long_range_witnesses(positions, cluster_labels, box):
    pos = np.asarray(positions, dtype=float)
    dmat = pair_distance_matrix(pos, box)
    inter = intercluster_distances(dmat, cluster_labels)
    inter = np.asarray(inter, dtype=float)
    centered = pos - pos.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / len(pos)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)
    rg2 = float(np.sum(centered ** 2) / len(pos))
    trace = float(np.trace(cov))
    anis = float((eigvals[-1] - eigvals[0]) / (trace + 1e-12))

    # Generic nonlocal tail witnesses. These are not claiming real electrostatics;
    # they are long-range observables a local graph can alias away.
    inv1 = float(np.sum(1.0 / inter))
    inv2 = float(np.sum(1.0 / (inter ** 2)))
    inv3 = float(np.sum(1.0 / (inter ** 3)))

    # Simple alternating charges to create a signed long-range channel.
    charges = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(len(pos))], dtype=float)
    signed = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if cluster_labels[i] != cluster_labels[j]:
                signed += charges[i] * charges[j] / dmat[i, j]

    return {
        "n_particles": int(len(pos)),
        "intercluster_pair_count": int(len(inter)),
        "min_intercluster_distance": float(np.min(inter)),
        "mean_intercluster_distance": float(np.mean(inter)),
        "max_intercluster_distance": float(np.max(inter)),
        "tail_sum_inv_r": inv1,
        "tail_sum_inv_r2": inv2,
        "tail_sum_inv_r3": inv3,
        "signed_alt_charge_inv_r": float(signed),
        "radius_gyration_sq": rg2,
        "cov_xx": float(cov[0, 0]),
        "cov_yy": float(cov[1, 1]),
        "cov_zz": float(cov[2, 2]),
        "cov_xy": float(cov[0, 1]),
        "cov_xz": float(cov[0, 2]),
        "cov_yz": float(cov[1, 2]),
        "cov_eig_0": float(eigvals[0]),
        "cov_eig_1": float(eigvals[1]),
        "cov_eig_2": float(eigvals[2]),
        "cov_anisotropy": anis,
    }


def make_cases():
    # Irregular 4-particle motif. The irregularity reduces tie artifacts.
    motif = np.array([
        [0.00, 0.00, 0.00],
        [1.00, 0.13, 0.02],
        [0.21, 1.31, 0.07],
        [0.11, 0.27, 1.73],
    ], dtype=float)

    cluster_a = motif.copy()

    cases = []

    def add_case(name, translation):
        trans = np.asarray(translation, dtype=float)
        cluster_b = motif + trans[None, :]
        positions = np.vstack([cluster_a, cluster_b])
        cluster_labels = ["A"] * len(cluster_a) + ["B"] * len(cluster_b)
        cases.append({
            "case": name,
            "translation_x": float(trans[0]),
            "translation_y": float(trans[1]),
            "translation_z": float(trans[2]),
            "positions": positions,
            "cluster_labels": cluster_labels,
        })

    # These are deliberately far compared to the motif diameter.
    add_case("sep20_x", [20.0, 0.0, 0.0])
    add_case("sep45_x", [45.0, 0.0, 0.0])
    add_case("sep20_y", [0.0, 20.0, 0.0])
    add_case("sep20_diag_xy", [14.1421356237, 14.1421356237, 0.0])
    add_case("sep20_z", [0.0, 0.0, 20.0])

    return cases


def compare_values(a, b, keys):
    out = {}
    for k in keys:
        va = float(a[k])
        vb = float(b[k])
        out[f"{k}_a"] = va
        out[f"{k}_b"] = vb
        out[f"{k}_delta_abs"] = abs(vb - va)
        out[f"{k}_delta_signed"] = vb - va
        out[f"{k}_rel_delta"] = abs(vb - va) / (abs(va) + 1e-12)
    return out


def main():
    box_L = 200.0
    box = jnp.array([box_L, box_L, box_L])
    box_np = np.array([box_L, box_L, box_L], dtype=float)

    cases = make_cases()

    case_rows = []
    local_signature_rows = []
    neighbor_edge_rows = []
    long_range_rows = []
    pair_distance_rows = []

    signatures_by_case_n = {}
    long_by_case = {}

    for c in cases:
        name = c["case"]
        positions_np = c["positions"]
        cluster_labels = c["cluster_labels"]
        positions = jnp.asarray(positions_np)
        dmat = pair_distance_matrix(positions_np, box_np)

        intra_vals = []
        inter_vals = []
        for i in range(len(positions_np)):
            for j in range(i + 1, len(positions_np)):
                row = {
                    "case": name,
                    "i": i,
                    "j": j,
                    "cluster_i": cluster_labels[i],
                    "cluster_j": cluster_labels[j],
                    "same_cluster": cluster_labels[i] == cluster_labels[j],
                    "distance": float(dmat[i, j]),
                }
                pair_distance_rows.append(row)
                if row["same_cluster"]:
                    intra_vals.append(row["distance"])
                else:
                    inter_vals.append(row["distance"])

        case_rows.append({
            "case": name,
            "n_particles": int(len(positions_np)),
            "translation_x": c["translation_x"],
            "translation_y": c["translation_y"],
            "translation_z": c["translation_z"],
            "motif_internal_max_distance": float(max(intra_vals)),
            "min_intercluster_distance": float(min(inter_vals)),
            "local_global_gap_ratio": float(min(inter_vals) / max(intra_vals)),
        })

        lw = long_range_witnesses(positions_np, cluster_labels, box_np)
        lw_row = {"case": name, **lw}
        long_range_rows.append(lw_row)
        long_by_case[name] = lw_row

        for n_local in n_local_values:
            try:
                senders, receivers = get_top(positions=positions, n=n_local, box=box)
                edges = selected_edges(senders, receivers)
                ok = True
                error = ""
            except Exception as e:
                edges = []
                ok = False
                error = repr(e)

            edge_sig = edge_signature(edges, dmat, round_digits) if ok else ""
            local_shell_sig = local_distance_shell_signature(dmat, cluster_labels, n_local, round_digits)
            internal_sig = internal_cluster_distance_signature(dmat, cluster_labels, round_digits)
            frac_internal = topn_internal_fraction(edges, cluster_labels) if ok else 0.0

            sig_row = {
                "case": name,
                "n_local": n_local,
                "ok": ok,
                "num_edges": int(len(edges)),
                "topn_internal_edge_fraction": frac_internal,
                "edge_signature": edge_sig,
                "local_distance_shell_signature": local_shell_sig,
                "internal_cluster_distance_signature": internal_sig,
                "error": error,
            }
            local_signature_rows.append(sig_row)
            signatures_by_case_n[(name, n_local)] = sig_row

            for i, j in edges:
                neighbor_edge_rows.append({
                    "case": name,
                    "n_local": n_local,
                    "sender": i,
                    "receiver": j,
                    "sender_cluster": cluster_labels[i],
                    "receiver_cluster": cluster_labels[j],
                    "same_cluster": cluster_labels[i] == cluster_labels[j],
                    "distance": float(dmat[i, j]),
                })

    comparison_pairs = [
        ("sep20_x", "sep45_x", "same motif, different cluster separation"),
        ("sep20_x", "sep20_y", "same motif, same separation magnitude, different global axis"),
        ("sep20_x", "sep20_diag_xy", "same motif, similar separation magnitude, diagonal global placement"),
        ("sep20_x", "sep20_z", "same motif, same separation magnitude, different z-axis placement"),
    ]

    long_keys = [
        "min_intercluster_distance",
        "mean_intercluster_distance",
        "tail_sum_inv_r",
        "tail_sum_inv_r2",
        "tail_sum_inv_r3",
        "signed_alt_charge_inv_r",
        "radius_gyration_sq",
        "cov_xx",
        "cov_yy",
        "cov_zz",
        "cov_xy",
        "cov_xz",
        "cov_yz",
        "cov_anisotropy",
    ]

    alias_comparison_rows = []
    for a, b, desc in comparison_pairs:
        for n_local in n_local_values:
            sa = signatures_by_case_n[(a, n_local)]
            sb = signatures_by_case_n[(b, n_local)]

            local_graph_same = sa["ok"] and sb["ok"] and sa["edge_signature"] == sb["edge_signature"]
            local_shell_same = sa["local_distance_shell_signature"] == sb["local_distance_shell_signature"]
            internal_sig_same = sa["internal_cluster_distance_signature"] == sb["internal_cluster_distance_signature"]
            both_local_internal = (
                float(sa["topn_internal_edge_fraction"]) == 1.0
                and float(sb["topn_internal_edge_fraction"]) == 1.0
            )

            deltas = compare_values(long_by_case[a], long_by_case[b], long_keys)
            tail_delta = deltas["tail_sum_inv_r_delta_abs"]
            rg_delta = deltas["radius_gyration_sq_delta_abs"]
            cov_component_delta = max(
                deltas["cov_xx_delta_abs"],
                deltas["cov_yy_delta_abs"],
                deltas["cov_zz_delta_abs"],
                deltas["cov_xy_delta_abs"],
                deltas["cov_xz_delta_abs"],
                deltas["cov_yz_delta_abs"],
            )
            long_range_delta_nonzero = bool(
                tail_delta > 1e-10
                or rg_delta > 1e-10
                or cov_component_delta > 1e-10
            )

            alias_detected = bool(
                local_graph_same
                and local_shell_same
                and internal_sig_same
                and both_local_internal
                and long_range_delta_nonzero
            )

            alias_comparison_rows.append({
                "case_a": a,
                "case_b": b,
                "comparison": desc,
                "n_local": n_local,
                "ok_a": sa["ok"],
                "ok_b": sb["ok"],
                "local_graph_same": local_graph_same,
                "local_distance_shell_same": local_shell_same,
                "internal_cluster_signature_same": internal_sig_same,
                "both_topn_graphs_internal_only": both_local_internal,
                "long_range_delta_nonzero": long_range_delta_nonzero,
                "alias_detected": alias_detected,
                "tail_sum_inv_r_delta_abs": tail_delta,
                "radius_gyration_sq_delta_abs": rg_delta,
                "max_cov_component_delta_abs": cov_component_delta,
                "edge_signature_a": sa["edge_signature"],
                "edge_signature_b": sb["edge_signature"],
                **deltas,
            })

    print(json.dumps({
        "case_rows": case_rows,
        "local_signature_rows": local_signature_rows,
        "neighbor_edge_rows": neighbor_edge_rows,
        "long_range_rows": long_range_rows,
        "alias_comparison_rows": alias_comparison_rows,
        "pair_distance_rows": pair_distance_rows,
    }, default=str))


main()
'''


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def write_report(path: Path, out_dir: Path, source_file: Path, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    alias_rows = rows.get("alias_comparison_rows", [])
    case_rows = rows.get("case_rows", [])
    local_rows = rows.get("local_signature_rows", [])
    long_rows = rows.get("long_range_rows", [])

    ok_alias = [r for r in alias_rows if r.get("alias_detected") is True]
    local_graph_same = [r for r in alias_rows if r.get("local_graph_same") is True]
    long_delta = [r for r in alias_rows if r.get("long_range_delta_nonzero") is True]

    lines = [
        "# PROBE 10 — Locality collision / long-range alias witness",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Source file: `{rel(source_file)}`",
        "",
        "## Purpose",
        "",
        "This probe resets from proving locality to finding the boundary of locality.",
        "",
        "It constructs paired configurations that preserve local/top-n neighborhood signatures while changing long-range/global witnesses.",
        "",
        "This is an adjacent mechanism-boundary probe. It is not a paper reproduction and not a full-model test.",
        "",
        "## Headline counts",
        "",
        f"- Cases constructed: **{len(case_rows)}**",
        f"- Local signature rows: **{len(local_rows)}**",
        f"- Long-range witness rows: **{len(long_rows)}**",
        f"- Alias comparison rows: **{len(alias_rows)}**",
        f"- Local graph same rows: **{len(local_graph_same)}**",
        f"- Long-range delta rows: **{len(long_delta)}**",
        f"- Alias detected rows: **{len(ok_alias)}**",
        "",
        "## Interpretation rule",
        "",
        "A locality collision / alias is counted when:",
        "",
        "```text",
        "local_graph_same == True",
        "local_distance_shell_same == True",
        "internal_cluster_signature_same == True",
        "both_topn_graphs_internal_only == True",
        "long_range_delta_nonzero == True",
        "```",
        "",
        "In plain language:",
        "",
        "> The local machinery sees the same thing, but a long-range/global witness changes.",
        "",
        "## Case geometry sanity",
        "",
    ]

    lines += md_table(
        ["case", "n_particles", "motif_internal_max", "min_intercluster", "gap ratio"],
        [[
            r.get("case"),
            r.get("n_particles"),
            r.get("motif_internal_max_distance"),
            r.get("min_intercluster_distance"),
            r.get("local_global_gap_ratio"),
        ] for r in case_rows]
    )

    lines += [
        "",
        "## Alias comparison summary",
        "",
    ]

    lines += md_table(
        ["case_a", "case_b", "n_local", "alias?", "local graph same?", "local shell same?", "internal only?", "tail Δ", "Rg² Δ", "max cov Δ"],
        [[
            r.get("case_a"),
            r.get("case_b"),
            r.get("n_local"),
            r.get("alias_detected"),
            r.get("local_graph_same"),
            r.get("local_distance_shell_same"),
            r.get("both_topn_graphs_internal_only"),
            r.get("tail_sum_inv_r_delta_abs"),
            r.get("radius_gyration_sq_delta_abs"),
            r.get("max_cov_component_delta_abs"),
        ] for r in alias_rows]
    )

    lines += [
        "",
        "## Long-range witnesses by case",
        "",
    ]

    lines += md_table(
        ["case", "min inter", "mean inter", "Σ1/r", "Σ1/r²", "Σ1/r³", "Rg²", "cov_xx", "cov_yy", "cov_zz", "cov_anis"],
        [[
            r.get("case"),
            r.get("min_intercluster_distance"),
            r.get("mean_intercluster_distance"),
            r.get("tail_sum_inv_r"),
            r.get("tail_sum_inv_r2"),
            r.get("tail_sum_inv_r3"),
            r.get("radius_gyration_sq"),
            r.get("cov_xx"),
            r.get("cov_yy"),
            r.get("cov_zz"),
            r.get("cov_anisotropy"),
        ] for r in long_rows]
    )

    lines += [
        "",
        "## What this proves",
        "",
        "This proves a controlled aliasing condition for the extracted top-n local representation: global arrangements can differ while local/top-n signatures remain unchanged.",
        "",
        "## What this does not prove",
        "",
        "This does not prove that the paper fails on its benchmark systems. It does not execute the full Boltzmann generator, train a model, sample a large system, compute ESS, or reproduce a figure.",
        "",
        "## Why it matters",
        "",
        "The current paper relies on transferable local environments for scaling. This probe identifies the class of information that fixed local environments can erase: global arrangement and long-range tail structure.",
        "",
        "## Suggested Probe 11",
        "",
        "Probe 11 should attach an explicit long-range target to these alias pairs: for example a Coulomb-like tail, dipole/quadrupole-like field, or response-style witness. Then test whether local signatures remain aliased while the target witness separates the states.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, out_dir: Path, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    alias_rows = rows.get("alias_comparison_rows", [])
    ok_alias = [r for r in alias_rows if r.get("alias_detected") is True]
    local_graph_same = [r for r in alias_rows if r.get("local_graph_same") is True]
    long_delta = [r for r in alias_rows if r.get("long_range_delta_nonzero") is True]

    lines = [
        "# PROBE 10 SUMMARY",
        "",
        f"Output folder: `{rel(out_dir)}`",
        "",
        "## What this probe did",
        "",
        "- Constructed paired two-cluster configurations.",
        "- Preserved the local motif and tested extracted top-n neighbor signatures.",
        "- Changed global cluster placement/separation.",
        "- Computed long-range/global witnesses.",
        "- Reported locality collisions where local signatures matched but long-range witnesses differed.",
        "",
        "## Headline counts",
        "",
        f"- Cases constructed: **{len(rows.get('case_rows', []))}**",
        f"- Local signature rows: **{len(rows.get('local_signature_rows', []))}**",
        f"- Neighbor edge rows: **{len(rows.get('neighbor_edge_rows', []))}**",
        f"- Long-range witness rows: **{len(rows.get('long_range_rows', []))}**",
        f"- Alias comparison rows: **{len(alias_rows)}**",
        f"- Local graph same rows: **{len(local_graph_same)}**",
        f"- Long-range delta rows: **{len(long_delta)}**",
        f"- Alias detected rows: **{len(ok_alias)}**",
        "",
        "## Generated files",
        "",
        "- `PROBE10_LOCALITY_COLLISION_REPORT.md`",
        "- `locality_collision_cases.csv/json`",
        "- `local_signature_by_case.csv/json`",
        "- `neighbor_edges_by_case.csv/json`",
        "- `long_range_witness_by_case.csv/json`",
        "- `alias_comparisons.csv/json`",
        "- `pair_distances_by_case.csv/json`",
        "- `gnn_conditioner_neighbor_shim.py`",
        "- `probe11_recommendations.md`",
        "- `logs/*.stdout.txt`, `logs/*.stderr.txt`",
        "",
        "## Current meaning",
        "",
        "If alias rows are detected, the current paper's local/top-n machinery can be shown to erase global arrangement information under controlled conditions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_recommendations(path: Path, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    alias_rows = rows.get("alias_comparison_rows", [])
    ok_alias = [r for r in alias_rows if r.get("alias_detected") is True]

    lines = [
        "# Probe 11 recommendations",
        "",
        "Probe 10 moved from locality control to locality collision.",
        "",
        f"Alias rows detected: `{len(ok_alias)}`",
        "",
        "## Recommended next step",
        "",
        "Probe 11 should attach a more physical long-range target to the alias pairs.",
        "",
        "Candidate witnesses:",
        "",
        "- Coulomb-like signed tail with neutral clusters.",
        "- Dipole/quadrupole-like far-field moments.",
        "- Response-style finite-difference force witness.",
        "- Long-range anisotropic pair-tail score.",
        "",
        "The test should ask:",
        "",
        "```text",
        "local_signature_same == True",
        "physical_long_range_witness_delta != 0",
        "```",
        "",
        "That is the smallest clean form of the long-range failure mechanism.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe 10: locality collision / long-range alias witness.")
    p.add_argument("--official-repo", default="../bgmat")
    p.add_argument("--out-root", default="probe_runs")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--n-local", action="append", type=int, default=[], help="Repeatable; default tests 1, 2, 3.")
    p.add_argument("--round-digits", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = resolve_from_root(args.official_repo)
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe10_locality_collision_alias_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")

    source_file = official_repo / "bgmat" / "models" / "gnn_conditioner.py"
    shim_file = out_dir / "gnn_conditioner_neighbor_shim.py"

    print("=" * 100)
    print("CURRENT PAPER PROBE 10 — LOCALITY COLLISION / LONG-RANGE ALIAS")
    print("=" * 100)
    print("[ROOT]          .")
    print(f"[OFFICIAL REPO] {rel(official_repo)}")
    print(f"[SOURCE]        {rel(source_file)}")
    print(f"[OUT]           {rel(out_dir)}")

    rows: Dict[str, List[Dict[str, Any]]]
    if not source_file.exists():
        rows = {
            "case_rows": [],
            "local_signature_rows": [],
            "neighbor_edge_rows": [],
            "long_range_rows": [],
            "alias_comparison_rows": [{"ok": False, "error": f"missing source file: {rel(source_file)}"}],
            "pair_distance_rows": [],
        }
    else:
        fn_sources = extract_function_sources(
            source_file,
            ["get_senders_and_receivers_fully_connected", "get_top_n_connections_vectorized"],
        )
        missing = [n for n in ["get_top_n_connections_vectorized"] if n not in fn_sources]
        if missing:
            rows = {
                "case_rows": [],
                "local_signature_rows": [],
                "neighbor_edge_rows": [],
                "long_range_rows": [],
                "alias_comparison_rows": [{"ok": False, "error": f"missing functions: {missing}"}],
                "pair_distance_rows": [],
            }
        else:
            write_shim_source(shim_file, fn_sources)
            n_values = args.n_local if args.n_local else [1, 2, 3]
            n_arg = ",".join(str(n) for n in n_values)
            result = run_subprocess(
                COLLISION_WITNESS_CODE,
                official_repo=official_repo,
                timeout=args.timeout,
                log_dir=log_dir,
                label="locality_collision_alias",
                args=[str(shim_file), n_arg, str(args.round_digits)],
            )
            rows = parse_payload(result, {
                "case_rows": [],
                "local_signature_rows": [],
                "neighbor_edge_rows": [],
                "long_range_rows": [],
                "alias_comparison_rows": [],
                "pair_distance_rows": [],
            })
            if not rows.get("alias_comparison_rows") and not result.get("ok"):
                rows["alias_comparison_rows"] = [{
                    "ok": False,
                    "stage": "subprocess",
                    "error": str(result.get("stderr", ""))[-2000:] or result.get("error", ""),
                    "stdout_log": result.get("stdout_log"),
                    "stderr_log": result.get("stderr_log"),
                }]

    write_csv(out_dir / "locality_collision_cases.csv", rows.get("case_rows", []))
    write_json(out_dir / "locality_collision_cases.json", rows.get("case_rows", []))

    write_csv(out_dir / "local_signature_by_case.csv", rows.get("local_signature_rows", []))
    write_json(out_dir / "local_signature_by_case.json", rows.get("local_signature_rows", []))

    write_csv(out_dir / "neighbor_edges_by_case.csv", rows.get("neighbor_edge_rows", []))
    write_json(out_dir / "neighbor_edges_by_case.json", rows.get("neighbor_edge_rows", []))

    write_csv(out_dir / "long_range_witness_by_case.csv", rows.get("long_range_rows", []))
    write_json(out_dir / "long_range_witness_by_case.json", rows.get("long_range_rows", []))

    write_csv(out_dir / "alias_comparisons.csv", rows.get("alias_comparison_rows", []))
    write_json(out_dir / "alias_comparisons.json", rows.get("alias_comparison_rows", []))

    write_csv(out_dir / "pair_distances_by_case.csv", rows.get("pair_distance_rows", []))
    write_json(out_dir / "pair_distances_by_case.json", rows.get("pair_distance_rows", []))

    write_report(out_dir / "PROBE10_LOCALITY_COLLISION_REPORT.md", out_dir, source_file, rows)
    write_summary(out_dir / "PROBE10_SUMMARY.md", out_dir, rows)
    write_recommendations(out_dir / "probe11_recommendations.md", rows)

    alias_rows = rows.get("alias_comparison_rows", [])
    ok_alias = [r for r in alias_rows if r.get("alias_detected") is True]
    local_graph_same = [r for r in alias_rows if r.get("local_graph_same") is True]
    long_delta = [r for r in alias_rows if r.get("long_range_delta_nonzero") is True]

    print("\n[SUMMARY]")
    print(f"  cases                    : {len(rows.get('case_rows', []))}")
    print(f"  local_signature_rows      : {len(rows.get('local_signature_rows', []))}")
    print(f"  neighbor_edge_rows        : {len(rows.get('neighbor_edge_rows', []))}")
    print(f"  long_range_rows           : {len(rows.get('long_range_rows', []))}")
    print(f"  alias_comparison_rows     : {len(alias_rows)}")
    print(f"  local_graph_same_rows     : {len(local_graph_same)}")
    print(f"  long_range_delta_rows     : {len(long_delta)}")
    print(f"  alias_detected_rows       : {len(ok_alias)}")
    print(f"  output                    : {rel(out_dir)}")
    print("\nNext:")
    print(f"  open {rel(out_dir / 'PROBE10_SUMMARY.md')}")
    print(f"  open {rel(out_dir / 'PROBE10_LOCALITY_COLLISION_REPORT.md')}")
    print(f"  open {rel(out_dir / 'alias_comparisons.csv')}")


if __name__ == "__main__":
    main()
