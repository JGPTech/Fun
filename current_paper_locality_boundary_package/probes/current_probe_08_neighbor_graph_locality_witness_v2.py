#!/usr/bin/env python3
# current_probe_08_neighbor_graph_locality_witness.py
#
# Probe 08: GNN neighbor-graph locality witness.
#
# Run from repo root:
#   python .\current_probe_08_neighbor_graph_locality_witness.py
#   python .\current_probe_08_neighbor_graph_locality_witness.py --n-local 1 --n-local 2 --n-local 3
#
# No training. No e3nn_jax install required. Outputs under probe_runs/.
# Includes local shim for pairwise_difference_pbc.

from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import hashlib
import json
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


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
    # Important: gnn_conditioner functions may carry decorators like
    # @partial(jax.jit, static_argnums=...). The shim must define partial.
    # We also use future annotations so type hints do not need the full repo imports.
    lines = [
        "# Extracted no-install shim for Probe 08.",
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
        "# For Probe 08 we only need the periodic minimum-image displacement matrix.",
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


NEIGHBOR_WITNESS_CODE = r'''
import json
import sys
import traceback
import numpy as np
import jax
import jax.numpy as jnp

shim_path = sys.argv[1]
n_local_values = [int(x) for x in sys.argv[2].split(",") if x.strip()]

namespace = {}
try:
    shim_source = open(shim_path, "r", encoding="utf-8").read()
    exec(compile(shim_source, shim_path, "exec"), namespace)
    get_fc = namespace["get_senders_and_receivers_fully_connected"]
    get_top = namespace["get_top_n_connections_vectorized"]
except Exception as e:
    print(json.dumps({
        "neighbor_witness": [{
            "ok": False,
            "stage": "shim_exec",
            "error": repr(e),
            "traceback_tail": traceback.format_exc()[-1800:],
        }],
        "pair_distances": [],
        "fully_connected": [],
    }, default=str))
    raise SystemExit(1)

def pair_distance_matrix(positions, box):
    pos = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)
    n = len(pos)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            d = pos[i] - pos[j]
            d = d - box * np.round(d / box)
            out[i, j] = np.linalg.norm(d)
    return out

def selected_edges(senders, receivers):
    s = np.asarray(senders).reshape(-1).astype(int)
    r = np.asarray(receivers).reshape(-1).astype(int)
    return [f"{int(si)}->{int(ri)}" for si, ri in zip(s, r)]

def selected_edges_touching_node(senders, receivers, node):
    s = np.asarray(senders).reshape(-1).astype(int)
    r = np.asarray(receivers).reshape(-1).astype(int)
    return [f"{int(si)}->{int(ri)}" for si, ri in zip(s, r) if int(si) == node or int(ri) == node]

def edge_set_for_existing_nodes(senders, receivers, existing_max_node):
    s = np.asarray(senders).reshape(-1).astype(int)
    r = np.asarray(receivers).reshape(-1).astype(int)
    return sorted([f"{int(si)}->{int(ri)}" for si, ri in zip(s, r) if int(si) <= existing_max_node and int(ri) <= existing_max_node])

def main():
    box_L = 100.0
    box = jnp.array([box_L, box_L, box_L])
    box_np = np.array([box_L, box_L, box_L], dtype=float)

    base3 = np.array([
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.2, 0.0],
    ], dtype=float)

    far_node = np.array([35.0, 35.0, 35.0], dtype=float)
    near0_node = np.array([0.15, 0.0, 0.0], dtype=float)
    near1_node = np.array([1.35, 0.0, 0.0], dtype=float)

    cases = {
        "base3": base3,
        "add_far_node": np.vstack([base3, far_node]),
        "add_near0_node": np.vstack([base3, near0_node]),
        "add_near1_node": np.vstack([base3, near1_node]),
    }

    rows = []
    pair_rows = []
    fc_rows = []

    for n_nodes in [3, 4, 5]:
        try:
            senders, receivers = get_fc(n_nodes)
            fc_rows.append({
                "n_nodes": n_nodes,
                "ok": True,
                "num_edges": int(np.asarray(senders).size),
                "senders": np.asarray(senders).reshape(-1).astype(int).tolist(),
                "receivers": np.asarray(receivers).reshape(-1).astype(int).tolist(),
                "error": "",
            })
        except Exception as e:
            fc_rows.append({"n_nodes": n_nodes, "ok": False, "num_edges": "", "senders": "", "receivers": "", "error": repr(e)})

    base_edges_by_n = {}

    for n_local in n_local_values:
        for case_name, positions_np in cases.items():
            positions = jnp.asarray(positions_np)
            try:
                senders, receivers = get_top(positions=positions, n=n_local, box=box)
                edges = selected_edges(senders, receivers)
                existing_edges = edge_set_for_existing_nodes(senders, receivers, existing_max_node=2)
                touching_new = selected_edges_touching_node(senders, receivers, node=3) if len(positions_np) > 3 else []
                ok = True
                error = ""
            except Exception as e:
                senders, receivers = [], []
                edges = []
                existing_edges = []
                touching_new = []
                ok = False
                error = repr(e)

            if case_name == "base3":
                base_edges_by_n[n_local] = existing_edges

            base_existing_edges = base_edges_by_n.get(n_local, [])
            existing_edges_unchanged = (existing_edges == base_existing_edges) if case_name != "base3" else True

            dmat = pair_distance_matrix(positions_np, box_np)
            nearest_to_new = ""
            if len(positions_np) > 3:
                d_to_existing = [(i, float(dmat[3, i])) for i in range(3)]
                d_to_existing.sort(key=lambda x: x[1])
                nearest_to_new = ";".join(f"{i}:{d:.8f}" for i, d in d_to_existing)

            rows.append({
                "n_local": n_local,
                "case": case_name,
                "n_nodes": int(len(positions_np)),
                "ok": ok,
                "num_edges": int(len(edges)),
                "edges": ";".join(edges),
                "existing_edges_0_1_2": ";".join(existing_edges),
                "base_existing_edges_0_1_2": ";".join(base_existing_edges),
                "existing_edges_unchanged_vs_base": existing_edges_unchanged,
                "edges_touching_new_node": ";".join(touching_new),
                "nearest_existing_to_new_node": nearest_to_new,
                "error": error,
            })

            for i in range(len(positions_np)):
                for j in range(i + 1, len(positions_np)):
                    pair_rows.append({
                        "n_local": n_local,
                        "case": case_name,
                        "i": i,
                        "j": j,
                        "distance": float(dmat[i, j]),
                    })

    print(json.dumps({"neighbor_witness": rows, "pair_distances": pair_rows, "fully_connected": fc_rows}, default=str))

main()
'''


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def write_recommendations(path: Path, rows: List[Dict[str, Any]], fc_rows: List[Dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r.get("ok")]
    shim_errors = [r for r in rows if r.get("stage") == "shim_exec"]
    far_rows = [r for r in ok_rows if r.get("case") == "add_far_node"]
    near_rows = [r for r in ok_rows if str(r.get("case", "")).startswith("add_near")]
    far_unchanged = [r for r in far_rows if r.get("existing_edges_unchanged_vs_base") is True]
    near_touched = [r for r in near_rows if r.get("edges_touching_new_node")]

    lines = [
        "# Probe 08 neighbor-graph locality witness",
        "",
        "Probe 08 used an AST shim of `gnn_conditioner`, so `e3nn_jax` was not required.",
        "",
        "## Headline",
        "",
        f"- Neighbor witness rows: `{len(rows)}`",
        f"- Execution OK rows: `{len(ok_rows)}`",
        f"- Shim errors: `{len(shim_errors)}`",
        f"- Far-node rows: `{len(far_rows)}`",
        f"- Far-node rows with unchanged existing graph: `{len(far_unchanged)}`",
        f"- Near-node rows touching new node: `{len(near_touched)}`",
        "",
    ]

    if shim_errors:
        lines += ["## Shim error", ""]
        for r in shim_errors:
            lines.append(f"- `{r.get('error')}`")
            if r.get("traceback_tail"):
                lines.append("")
                lines.append("```text")
                lines.append(str(r.get("traceback_tail"))[-1800:])
                lines.append("```")
        lines.append("")

    lines += ["## Fully connected sanity", ""]
    lines += md_table(["n_nodes", "ok", "num_edges", "error"], [[r.get("n_nodes"), r.get("ok"), r.get("num_edges"), r.get("error")] for r in fc_rows])

    lines += ["", "## Neighbor witness", ""]
    lines += md_table(
        ["n_local", "case", "n_nodes", "edges", "existing graph unchanged?", "new-node edges", "nearest new-node distances", "error"],
        [[r.get("n_local"), r.get("case"), r.get("n_nodes"), r.get("edges"), r.get("existing_edges_unchanged_vs_base"), r.get("edges_touching_new_node"), r.get("nearest_existing_to_new_node"), r.get("error")] for r in rows]
    )

    lines += [
        "",
        "## Interpretation",
        "",
        "- If `add_far_node` keeps the existing graph unchanged, the neighbor selector is locally stable under far perturbations.",
        "- If `add_near0_node` or `add_near1_node` touches or rewires local edges, the selector reacts when a new node enters the local neighborhood.",
        "- With small `n_local`, graph rewiring is expected because nearest-neighbor slots are finite.",
        "",
        "## Suggested Probe 09",
        "",
        "Probe 09 should combine Probe 07 and Probe 08: same geometry, same far/near perturbations, and a joint report comparing energy blindness with neighbor-graph blindness/rewiring.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, out_dir: Path, source_file: Path, rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]], fc_rows: List[Dict[str, Any]]) -> None:
    ok_rows = [r for r in rows if r.get("ok")]
    shim_errors = [r for r in rows if r.get("stage") == "shim_exec"]
    far_rows = [r for r in ok_rows if r.get("case") == "add_far_node"]
    far_unchanged = [r for r in far_rows if r.get("existing_edges_unchanged_vs_base") is True]
    near_rows = [r for r in ok_rows if str(r.get("case", "")).startswith("add_near")]
    near_touched = [r for r in near_rows if r.get("edges_touching_new_node")]

    lines = [
        "# PROBE 08 SUMMARY",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Source file: `{rel(source_file)}`",
        "",
        "## What this probe did",
        "",
        "- Extracted neighbor-builder functions from `gnn_conditioner.py` by AST.",
        "- Avoided importing `gnn_conditioner`, so `e3nn_jax` was not required.",
        "- Ran clean base/far/near geometries through `get_top_n_connections_vectorized`.",
        "- Recorded sender/receiver edge selections and pair distances.",
        "",
        "## Headline counts",
        "",
        f"- Neighbor witness rows: **{len(rows)}**",
        f"- Neighbor witness OK rows: **{len(ok_rows)}**",
        f"- Shim error rows: **{len(shim_errors)}**",
        f"- Pair distance rows: **{len(pair_rows)}**",
        f"- Fully connected sanity rows: **{len(fc_rows)}**",
        f"- Far-node rows: **{len(far_rows)}**",
        f"- Far-node unchanged existing graph rows: **{len(far_unchanged)}**",
        f"- Near-node rows: **{len(near_rows)}**",
        f"- Near-node rows touching new node: **{len(near_touched)}**",
        "",
        "## Generated files",
        "",
        "- `neighbor_graph_witness.csv/json`",
        "- `neighbor_pair_distances.csv/json`",
        "- `fully_connected_sanity.csv/json`",
        "- `gnn_conditioner_neighbor_shim.py`",
        "- `probe09_recommendations.md`",
        "- `logs/*.stdout.txt`, `logs/*.stderr.txt`",
        "",
        "## Next",
        "",
        "Open `probe09_recommendations.md`. Probe 09 should combine the clean energy witness with this neighbor-graph witness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe 08: neighbor graph locality witness.")
    p.add_argument("--official-repo", default="../bgmat")
    p.add_argument("--out-root", default="probe_runs")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--n-local", action="append", type=int, default=[], help="Repeatable; default tests 1 and 2.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = resolve_from_root(args.official_repo)
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe08_neighbor_graph_locality_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")
    source_file = official_repo / "bgmat" / "models" / "gnn_conditioner.py"
    shim_file = out_dir / "gnn_conditioner_neighbor_shim.py"

    print("=" * 100)
    print("CURRENT PAPER PROBE 08 — NEIGHBOR-GRAPH LOCALITY WITNESS")
    print("=" * 100)
    print("[ROOT]          .")
    print(f"[OFFICIAL REPO] {rel(official_repo)}")
    print(f"[SOURCE]        {rel(source_file)}")
    print(f"[OUT]           {rel(out_dir)}")

    if not source_file.exists():
        rows = [{"ok": False, "error": f"missing source file: {rel(source_file)}"}]
        pair_rows: List[Dict[str, Any]] = []
        fc_rows: List[Dict[str, Any]] = []
    else:
        fn_sources = extract_function_sources(source_file, ["get_senders_and_receivers_fully_connected", "get_top_n_connections_vectorized"])
        missing = [n for n in ["get_senders_and_receivers_fully_connected", "get_top_n_connections_vectorized"] if n not in fn_sources]
        if missing:
            rows = [{"ok": False, "error": f"missing functions: {missing}"}]
            pair_rows = []
            fc_rows = []
        else:
            write_shim_source(shim_file, fn_sources)
            n_values = args.n_local if args.n_local else [1, 2]
            n_arg = ",".join(str(n) for n in n_values)
            result = run_subprocess(NEIGHBOR_WITNESS_CODE, official_repo, args.timeout, log_dir, "neighbor_graph_witness", args=[str(shim_file), n_arg])
            payload = parse_payload(result, {"neighbor_witness": [], "pair_distances": [], "fully_connected": []})
            rows = payload.get("neighbor_witness", [])
            pair_rows = payload.get("pair_distances", [])
            fc_rows = payload.get("fully_connected", [])

            if not rows and not result.get("ok"):
                rows = [{
                    "ok": False,
                    "stage": "subprocess",
                    "error": str(result.get("stderr", ""))[-1200:] or result.get("error", ""),
                    "stdout_log": result.get("stdout_log"),
                    "stderr_log": result.get("stderr_log"),
                }]

    write_csv(out_dir / "neighbor_graph_witness.csv", rows)
    write_json(out_dir / "neighbor_graph_witness.json", rows)
    write_csv(out_dir / "neighbor_pair_distances.csv", pair_rows)
    write_json(out_dir / "neighbor_pair_distances.json", pair_rows)
    write_csv(out_dir / "fully_connected_sanity.csv", fc_rows)
    write_json(out_dir / "fully_connected_sanity.json", fc_rows)
    write_recommendations(out_dir / "probe09_recommendations.md", rows, fc_rows)
    write_summary(out_dir / "PROBE08_SUMMARY.md", out_dir, source_file, rows, pair_rows, fc_rows)

    ok_rows = [r for r in rows if r.get("ok")]
    shim_errors = [r for r in rows if r.get("stage") == "shim_exec"]
    far_rows = [r for r in ok_rows if r.get("case") == "add_far_node"]
    far_unchanged = [r for r in far_rows if r.get("existing_edges_unchanged_vs_base") is True]
    near_rows = [r for r in ok_rows if str(r.get("case", "")).startswith("add_near")]
    near_touched = [r for r in near_rows if r.get("edges_touching_new_node")]

    print("\n[SUMMARY]")
    print(f"  neighbor_rows             : {len(rows)}")
    print(f"  neighbor_ok               : {len(ok_rows)}")
    print(f"  shim_errors               : {len(shim_errors)}")
    print(f"  pair_distance_rows        : {len(pair_rows)}")
    print(f"  fully_connected_rows      : {len(fc_rows)}")
    print(f"  far_rows                  : {len(far_rows)}")
    print(f"  far_unchanged_existing    : {len(far_unchanged)}")
    print(f"  near_rows                 : {len(near_rows)}")
    print(f"  near_touching_new_node    : {len(near_touched)}")
    print(f"  output                    : {rel(out_dir)}")
    print("\nNext:")
    print(f"  open {rel(out_dir / 'PROBE08_SUMMARY.md')}")
    print(f"  open {rel(out_dir / 'probe09_recommendations.md')}")
    print(f"  open {rel(out_dir / 'neighbor_graph_witness.csv')}")


if __name__ == "__main__":
    main()
