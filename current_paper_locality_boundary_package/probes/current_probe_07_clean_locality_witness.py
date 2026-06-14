#!/usr/bin/env python3
'''
current_probe_07_clean_locality_witness.py

Probe 07: clean locality witness for the current paper sandbox.

Probe 06 proved the LJ cutoff sweep is hard-local, but its far-particle
comparison was crude because the third particle could still be inside cutoff
of particle 1, or inside cutoff through periodic minimum-image wrapping.

Probe 07 uses a large box and explicit minimum-image distance checks.

Run from repo root:
  python current_probe_07_clean_locality_witness.py

No training. No datasets. No hardcoded local paths. Outputs under probe_runs/.
'''

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional


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


def run_subprocess(code_text: str, official_repo: Path, timeout: int, log_dir: Path, label: str) -> Dict[str, Any]:
    ensure_dir(log_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-c", code_text]

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


WITNESS_CODE = r'''
import importlib, json, math
import numpy as np
import jax.numpy as jnp

def min_image_delta(a, b, box):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    box = np.asarray(box, dtype=float)
    return d - box * np.round(d / box)

def min_image_dist(a, b, box):
    return float(np.linalg.norm(min_image_delta(a, b, box)))

def scalarize(x):
    arr = np.asarray(x)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return {"shape": list(arr.shape), "repr": repr(x)[:240]}

def box_vec(L):
    return jnp.array([float(L), float(L), float(L)])

def lj_reference_pair_energy(r, epsilon=1.0, sigma=1.0, cutoff=None):
    if cutoff is not None and r > cutoff:
        return 0.0
    inv = sigma / r
    inv6 = inv ** 6
    inv12 = inv6 ** 2
    return 4.0 * epsilon * (inv12 - inv6)

def pair_graph(coords, box, cutoff):
    coords_np = np.asarray(coords, dtype=float)
    pairs = []
    for i in range(len(coords_np)):
        for j in range(i + 1, len(coords_np)):
            d = min_image_dist(coords_np[i], coords_np[j], box)
            pairs.append({"i": i, "j": j, "distance": d, "inside_cutoff": bool(d <= cutoff)})
    return pairs

def total_reference_energy(coords, box, cutoff):
    total = 0.0
    for p in pair_graph(coords, box, cutoff):
        if p["inside_cutoff"]:
            total += lj_reference_pair_energy(p["distance"], cutoff=cutoff)
    return total

def main():
    mod = importlib.import_module("bgmat.systems.lennard_jones")
    cls = getattr(mod, "LennardJonesEnergy")

    cutoffs = [1.0, 1.5, 2.5, 4.0]
    box_L = 100.0
    box = box_vec(box_L)
    box_np = np.array([box_L, box_L, box_L], dtype=float)

    rows = []
    pair_rows = []

    for cutoff in cutoffs:
        near_r = max(1.1, 0.65 * cutoff)
        if near_r >= cutoff:
            near_r = 0.9 * cutoff

        x0 = np.array([0.0, 0.0, 0.0])
        x1 = np.array([near_r, 0.0, 0.0])

        # Far from both under minimum-image convention.
        far_sep = min(0.45 * box_L, cutoff + 10.0)
        x2_far = np.array([far_sep, far_sep, far_sep])

        # Inside cutoff of particle 0 only / particle 1 only, perpendicular offset.
        local_r = min(0.75 * cutoff, max(0.85, cutoff - 0.15))
        if local_r >= cutoff:
            local_r = 0.9 * cutoff
        x2_near_0 = np.array([0.0, local_r, 0.0])
        x2_near_1 = np.array([near_r, local_r, 0.0])

        cases = [
            ("two_particle_base", np.stack([x0, x1], axis=0)),
            ("third_far_all", np.stack([x0, x1, x2_far], axis=0)),
            ("third_inside_particle0", np.stack([x0, x1, x2_near_0], axis=0)),
            ("third_inside_particle1", np.stack([x0, x1, x2_near_1], axis=0)),
        ]

        obj = cls(cutoff=cutoff, box_length=box)
        base_energy = None
        base_ref = None

        for case_name, coords_np in cases:
            graph = pair_graph(coords_np, box_np, cutoff)
            coords = jnp.asarray(coords_np)
            try:
                energy = scalarize(obj.energy(coords, box))
                ok = True
                error = ""
            except Exception as e:
                energy = None
                ok = False
                error = repr(e)

            ref = total_reference_energy(coords_np, box_np, cutoff)
            if case_name == "two_particle_base":
                base_energy = energy
                base_ref = ref

            delta = None
            ref_delta = None
            if isinstance(energy, (int, float)) and isinstance(base_energy, (int, float)):
                delta = float(energy) - float(base_energy)
            if base_ref is not None:
                ref_delta = ref - base_ref

            expected_blind = all(not p["inside_cutoff"] for p in graph if 2 in (p["i"], p["j"])) if len(coords_np) == 3 else False
            expected_pairs = [f'{p["i"]}-{p["j"]}' for p in graph if 2 in (p["i"], p["j"]) and p["inside_cutoff"]]

            rows.append({
                "cutoff": cutoff,
                "case": case_name,
                "near_pair_distance": near_r,
                "third_expected_blind_to_existing_particles": expected_blind,
                "third_interacting_pairs_expected": ";".join(expected_pairs),
                "ok": ok,
                "energy": energy,
                "base_energy": base_energy,
                "delta_from_base": delta,
                "reference_energy": ref,
                "reference_delta_from_base": ref_delta,
                "abs_error_vs_reference": abs(float(energy) - ref) if isinstance(energy, (int, float)) else None,
                "error": error,
            })

            for p in graph:
                pair_rows.append({
                    "cutoff": cutoff,
                    "case": case_name,
                    "i": p["i"],
                    "j": p["j"],
                    "distance": p["distance"],
                    "inside_cutoff": p["inside_cutoff"],
                })

    print(json.dumps({"locality_witness": rows, "pair_graph": pair_rows}, default=str))

main()
'''


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def is_near_zero(x: Any, tol: float) -> bool:
    return isinstance(x, (int, float)) and abs(float(x)) <= tol


def write_recommendations(path: Path, rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]], tol: float) -> None:
    far_rows = [r for r in rows if r.get("case") == "third_far_all"]
    far_ok = [r for r in far_rows if r.get("ok") and r.get("third_expected_blind_to_existing_particles")]
    far_zero = [r for r in far_ok if is_near_zero(r.get("delta_from_base"), tol)]
    inside_rows = [r for r in rows if str(r.get("case", "")).startswith("third_inside")]
    inside_ok = [r for r in inside_rows if r.get("ok")]
    inside_nonzero = [r for r in inside_ok if not is_near_zero(r.get("delta_from_base"), tol)]

    lines = [
        "# Probe 07 clean locality witness",
        "",
        "Probe 07 corrects the Probe 06 far-particle ambiguity by using a large box and explicit minimum-image pair checks.",
        "",
        "## Headline",
        "",
        f"- Far/blind cases checked: `{len(far_ok)}`",
        f"- Far/blind cases with near-zero delta: `{len(far_zero)}`",
        f"- Inside/local cases checked: `{len(inside_ok)}`",
        f"- Inside/local cases with nonzero delta: `{len(inside_nonzero)}`",
        "",
        "## Far/blind cases",
        "",
    ]

    lines += md_table(
        ["cutoff", "case", "blind?", "pairs", "energy", "base", "delta", "ref delta", "abs err"],
        [[r.get("cutoff"), r.get("case"), r.get("third_expected_blind_to_existing_particles"),
          r.get("third_interacting_pairs_expected"), r.get("energy"), r.get("base_energy"),
          r.get("delta_from_base"), r.get("reference_delta_from_base"), r.get("abs_error_vs_reference")]
         for r in far_rows]
    )

    lines += ["", "## Inside/local cases", ""]
    lines += md_table(
        ["cutoff", "case", "blind?", "pairs", "energy", "base", "delta", "ref delta", "abs err"],
        [[r.get("cutoff"), r.get("case"), r.get("third_expected_blind_to_existing_particles"),
          r.get("third_interacting_pairs_expected"), r.get("energy"), r.get("base_energy"),
          r.get("delta_from_base"), r.get("reference_delta_from_base"), r.get("abs_error_vs_reference")]
         for r in inside_rows]
    )

    lines += [
        "",
        "## Interpretation",
        "",
        "If the far/blind rows have zero or near-zero `delta_from_base`, the explicit energy implementation is locally blind outside cutoff.",
        "If the inside/local rows have nonzero deltas, the same implementation still reacts when a new particle enters the local neighborhood.",
        "",
        "## Suggested Probe 08",
        "",
        "Probe 08 should target `bgmat.models.gnn_conditioner`: import it by installing/shimming `e3nn_jax`, then run the same clean geometry through `get_top_n_connections_vectorized` and verify sender/receiver selection.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, out_dir: Path, official_repo: Path, rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]], tol: float) -> None:
    far_rows = [r for r in rows if r.get("case") == "third_far_all"]
    far_ok = [r for r in far_rows if r.get("ok") and r.get("third_expected_blind_to_existing_particles")]
    far_zero = [r for r in far_ok if is_near_zero(r.get("delta_from_base"), tol)]
    inside_rows = [r for r in rows if str(r.get("case", "")).startswith("third_inside")]
    inside_ok = [r for r in inside_rows if r.get("ok")]
    inside_nonzero = [r for r in inside_ok if not is_near_zero(r.get("delta_from_base"), tol)]

    lines = [
        "# PROBE 07 SUMMARY",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Official repo: `{rel(official_repo)}`",
        "",
        "## What this probe did",
        "",
        "- Built a clean LJ locality witness in a large periodic box.",
        "- Verified pair distances under minimum-image convention.",
        "- Compared base two-particle energy to three-particle energy.",
        "- Separated truly far/blind third-particle cases from inside/local third-particle cases.",
        "",
        "## Headline counts",
        "",
        f"- Locality witness rows: **{len(rows)}**",
        f"- Pair graph rows: **{len(pair_rows)}**",
        f"- Far/blind cases: **{len(far_ok)}**",
        f"- Far/blind near-zero deltas: **{len(far_zero)}**",
        f"- Inside/local cases: **{len(inside_ok)}**",
        f"- Inside/local nonzero deltas: **{len(inside_nonzero)}**",
        f"- Near-zero tolerance: **{tol}**",
        "",
        "## Generated files",
        "",
        "- `clean_locality_witness.csv/json`",
        "- `pair_graph_distances.csv/json`",
        "- `probe08_recommendations.md`",
        "- `logs/*.stdout.txt`, `logs/*.stderr.txt`",
        "",
        "## Next",
        "",
        "Open `probe08_recommendations.md`. The next mechanism target is the GNN neighbor builder.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe 07: clean locality witness.")
    p.add_argument("--official-repo", default="../bgmat")
    p.add_argument("--out-root", default="probe_runs")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = resolve_from_root(args.official_repo)
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe07_clean_locality_witness_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")

    print("=" * 100)
    print("CURRENT PAPER PROBE 07 — CLEAN LOCALITY WITNESS")
    print("=" * 100)
    print("[ROOT]          .")
    print(f"[OFFICIAL REPO] {rel(official_repo)}")
    print(f"[OUT]           {rel(out_dir)}")

    result = run_subprocess(WITNESS_CODE, official_repo, args.timeout, log_dir, "clean_locality_witness")
    payload = parse_payload(result, {"locality_witness": [], "pair_graph": []})
    rows = payload.get("locality_witness", [])
    pair_rows = payload.get("pair_graph", [])

    if not rows and not result.get("ok"):
        rows = [{
            "ok": False,
            "stage": "subprocess",
            "error": str(result.get("stderr", ""))[-1000:] or result.get("error", ""),
            "stdout_log": result.get("stdout_log"),
            "stderr_log": result.get("stderr_log"),
        }]

    write_csv(out_dir / "clean_locality_witness.csv", rows)
    write_json(out_dir / "clean_locality_witness.json", rows)
    write_csv(out_dir / "pair_graph_distances.csv", pair_rows)
    write_json(out_dir / "pair_graph_distances.json", pair_rows)
    write_recommendations(out_dir / "probe08_recommendations.md", rows, pair_rows, args.tol)
    write_summary(out_dir / "PROBE07_SUMMARY.md", out_dir, official_repo, rows, pair_rows, args.tol)

    far_rows = [r for r in rows if r.get("case") == "third_far_all"]
    far_ok = [r for r in far_rows if r.get("ok") and r.get("third_expected_blind_to_existing_particles")]
    far_zero = [r for r in far_ok if is_near_zero(r.get("delta_from_base"), args.tol)]
    inside_rows = [r for r in rows if str(r.get("case", "")).startswith("third_inside")]
    inside_ok = [r for r in inside_rows if r.get("ok")]
    inside_nonzero = [r for r in inside_ok if not is_near_zero(r.get("delta_from_base"), args.tol)]

    print("\n[SUMMARY]")
    print(f"  locality_rows            : {len(rows)}")
    print(f"  pair_graph_rows          : {len(pair_rows)}")
    print(f"  far_blind_cases          : {len(far_ok)}")
    print(f"  far_blind_near_zero      : {len(far_zero)}")
    print(f"  inside_local_cases       : {len(inside_ok)}")
    print(f"  inside_local_nonzero     : {len(inside_nonzero)}")
    print(f"  output                   : {rel(out_dir)}")
    print("\nNext:")
    print(f"  open {rel(out_dir / 'PROBE07_SUMMARY.md')}")
    print(f"  open {rel(out_dir / 'probe08_recommendations.md')}")
    print(f"  open {rel(out_dir / 'clean_locality_witness.csv')}")


if __name__ == "__main__":
    main()
