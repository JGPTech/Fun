
#!/usr/bin/env python3
'''
current_probe_06_tiny_cutoff_locality_stress.py

Probe 06: tiny cutoff/locality stress test for the current paper sandbox.

Run from repo root:
  python current_probe_06_tiny_cutoff_locality_stress.py

Optional:
  python current_probe_06_tiny_cutoff_locality_stress.py --include-sw-lambda

No training. No datasets. No absolute paths. Outputs under probe_runs/.
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


COMMON_CODE = r'''
import importlib, inspect, json, math, sys
import numpy as np
import jax
import jax.numpy as jnp

def scalarize(x):
    try:
        arr = np.asarray(x)
        if arr.size == 1:
            f = float(arr.reshape(-1)[0])
            if math.isfinite(f):
                return f
            return repr(x)[:200]
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
            "repr": repr(x)[:200],
        }
    except Exception:
        return repr(x)[:300]

def box_vec(L):
    return jnp.array([float(L), float(L), float(L)])

def coords_two(r):
    return jnp.array([[0.0, 0.0, 0.0], [float(r), 0.0, 0.0]])

def coords_three(r_near, r_far):
    return jnp.array([
        [0.0, 0.0, 0.0],
        [float(r_near), 0.0, 0.0],
        [float(r_far), 0.0, 0.0],
    ])

def coord_variants(n=4, spacing=1.25):
    line = jnp.array([[i * spacing, 0.0, 0.0] for i in range(n)])
    tetra = jnp.array([
        [0.0, 0.0, 0.0],
        [spacing, 0.0, 0.0],
        [0.0, spacing, 0.0],
        [0.0, 0.0, spacing],
    ])
    return {
        "n_by_3_line": line,
        "n_by_3_tetra": tetra,
        "batch_n_by_3_line": line[None, :, :],
        "flat_batch_line": line.reshape(1, -1),
        "flat_line": line.reshape(-1),
    }

def required_params(callable_obj):
    try:
        sig = inspect.signature(callable_obj)
    except Exception:
        return "", []
    req = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            req.append(name)
    return str(sig), req

def call_method(obj, method_name, coords, box_length=None, lambdas_three_body=None):
    meth = obj if method_name == "__call__" and callable(obj) else getattr(obj, method_name, None)
    if meth is None or not callable(meth):
        return {"method": method_name, "ok": False, "error": "method_missing"}

    sig, req = required_params(meth)
    attempts = []

    if lambdas_three_body is not None and "lambda" in sig.lower():
        attempts.append(("coords_lambdas_box", (coords, lambdas_three_body, box_length), {}))
        attempts.append(("coords_lambdas", (coords, lambdas_three_body), {}))

    if box_length is not None:
        attempts.append(("coords_box_positional", (coords, box_length), {}))
        attempts.append(("coords_box_kw", (coords,), {"box_length": box_length}))

    attempts.append(("coords_only", (coords,), {}))

    last_error = ""
    for style, args, kwargs in attempts:
        try:
            y = meth(*args, **kwargs)
            return {
                "method": method_name,
                "ok": True,
                "style": style,
                "signature": sig,
                "required_params": ",".join(req),
                "value": scalarize(y),
            }
        except Exception as e:
            last_error = repr(e)

    return {
        "method": method_name,
        "ok": False,
        "signature": sig,
        "required_params": ",".join(req),
        "error": last_error[:700],
    }

def target_classes(module_name):
    mod = importlib.import_module(module_name)
    rows = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if getattr(obj, "__module__", "") == module_name:
            try:
                sig = str(inspect.signature(obj))
            except Exception as e:
                sig = f"<sigerr {e!r}>"
            rows.append((name, obj, sig))
    return rows
'''


CONSTRUCTOR_CODE = COMMON_CODE + r'''
def main():
    rows_construct = []
    rows_shape = []

    modules = [
        "bgmat.systems.energies",
        "bgmat.systems.lennard_jones",
        "bgmat.systems.monatomic_water",
        "bgmat.systems.stillinger_weber",
    ]

    recipes = []
    for m in modules:
        try:
            classes = target_classes(m)
        except Exception as e:
            rows_construct.append({"module": m, "class_name": "", "recipe_name": "", "ok": False, "error": repr(e)})
            continue

        for cname, cls, sig in classes:
            ml = m.lower()
            sigl = sig.lower()
            if "cutoff" in sigl:
                for cutoff in [1.0, 1.5, 2.5, 4.0]:
                    recipes.append((m, cname, f"cutoff_{cutoff}_box10", {"cutoff": cutoff, "box_length": box_vec(10.0)}))
                    recipes.append((m, cname, f"cutoff_{cutoff}_nobox", {"cutoff": cutoff}))
            elif "monatomic_water" in ml:
                recipes.append((m, cname, "zero_arg", {}))
                recipes.append((m, cname, "box20", {"box_length": box_vec(20.0)}))
            elif "stillinger_weber" in ml:
                recipes.append((m, cname, "zero_arg", {}))
                recipes.append((m, cname, "box20", {"box_length": box_vec(20.0)}))
                recipes.append((m, cname, "mw_params", {"sigma": 2.3925, "epsilon": 6.189, "lambda_three_body": 23.15, "box_length": box_vec(20.0)}))
                recipes.append((m, cname, "si_params", {"sigma": 2.0951, "epsilon": 50.003, "lambda_three_body": 21.0, "box_length": box_vec(20.0)}))
            else:
                recipes.append((m, cname, "zero_arg", {}))
                recipes.append((m, cname, "box10", {"box_length": box_vec(10.0)}))

    seen = set()
    for m, cname, rname, kwargs in recipes:
        key = (m, cname, rname)
        if key in seen:
            continue
        seen.add(key)
        try:
            mod = importlib.import_module(m)
            cls = getattr(mod, cname)
            obj = cls(**kwargs)
            rows_construct.append({
                "module": m,
                "class_name": cname,
                "recipe_name": rname,
                "kwargs_keys": ",".join(kwargs.keys()),
                "ok": True,
                "error": "",
                "repr": repr(obj)[:240],
            })

            for vname, coords in coord_variants(n=4, spacing=1.25).items():
                for method in ["energy", "__call__"]:
                    res = call_method(obj, method, coords, box_length=box_vec(20.0))
                    rows_shape.append({
                        "module": m,
                        "class_name": cname,
                        "recipe_name": rname,
                        "coord_variant": vname,
                        **res,
                    })
        except Exception as e:
            rows_construct.append({
                "module": m,
                "class_name": cname,
                "recipe_name": rname,
                "kwargs_keys": ",".join(kwargs.keys()),
                "ok": False,
                "error": repr(e)[:800],
                "repr": "",
            })

    print(json.dumps({"constructor_attempts": rows_construct, "energy_shape_acceptance": rows_shape}, default=str))

main()
'''


LJ_CODE = COMMON_CODE + r'''
def main():
    rows_sweep = []
    rows_far = []

    try:
        classes = target_classes("bgmat.systems.lennard_jones")
    except Exception as e:
        print(json.dumps({"lj_cutoff_sweep": [{"ok": False, "error": repr(e)}], "lj_far_particle_effect": []}))
        return

    classes = [(n, c, s) for n, c, s in classes if "cutoff" in s.lower()]
    distances = [0.85, 0.95, 1.0, 1.1, 1.25, 1.49, 1.51, 1.75, 2.0, 2.49, 2.51, 3.0, 4.0, 5.0]
    cutoffs = [1.0, 1.5, 2.5, 4.0]

    for cname, cls, sig in classes:
        for cutoff in cutoffs:
            for rname, kwargs in [
                ("box10", {"cutoff": cutoff, "box_length": box_vec(10.0)}),
                ("nobox", {"cutoff": cutoff}),
            ]:
                try:
                    obj = cls(**kwargs)
                except Exception as e:
                    rows_sweep.append({"class_name": cname, "cutoff": cutoff, "recipe_name": rname, "ok": False, "stage": "construct", "error": repr(e)[:500]})
                    continue

                for dist in distances:
                    res = call_method(obj, "energy", coords_two(dist), box_length=box_vec(10.0))
                    rows_sweep.append({
                        "class_name": cname,
                        "signature": sig,
                        "cutoff": cutoff,
                        "recipe_name": rname,
                        "distance": dist,
                        "ok": res.get("ok"),
                        "energy": res.get("value"),
                        "style": res.get("style", ""),
                        "error": res.get("error", ""),
                    })

                near_r = 1.1
                e2 = call_method(obj, "energy", coords_two(near_r), box_length=box_vec(10.0))
                e2v = e2.get("value")
                for r_far in [1.2, cutoff - 0.05, cutoff + 0.05, cutoff + 1.0, 8.0]:
                    e3 = call_method(obj, "energy", coords_three(near_r, r_far), box_length=box_vec(10.0))
                    e3v = e3.get("value")
                    delta = None
                    if isinstance(e2v, (int, float)) and isinstance(e3v, (int, float)):
                        delta = e3v - e2v
                    rows_far.append({
                        "class_name": cname,
                        "cutoff": cutoff,
                        "recipe_name": rname,
                        "near_pair_distance": near_r,
                        "third_particle_distance": r_far,
                        "pair_energy_ok": e2.get("ok"),
                        "three_energy_ok": e3.get("ok"),
                        "pair_energy": e2v,
                        "three_particle_energy": e3v,
                        "delta_three_minus_pair": delta,
                        "three_style": e3.get("style", ""),
                        "error": e3.get("error", "") or e2.get("error", ""),
                    })

    print(json.dumps({"lj_cutoff_sweep": rows_sweep, "lj_far_particle_effect": rows_far}, default=str))

main()
'''


MW_SW_CODE = COMMON_CODE + r'''
def main():
    include_lambda = sys.argv[1].lower() == "true"
    rows = []
    for m in ["bgmat.systems.monatomic_water", "bgmat.systems.stillinger_weber"]:
        try:
            classes = target_classes(m)
        except Exception as e:
            rows.append({"module": m, "class_name": "", "ok": False, "stage": "import", "error": repr(e)})
            continue

        for cname, cls, sig in classes:
            if "energy" not in cname.lower() and "potential" not in cname.lower():
                continue

            recipes = [("zero_arg", {}), ("box20", {"box_length": box_vec(20.0)})]
            if "stillinger" in m:
                recipes += [
                    ("mw_params", {"sigma": 2.3925, "epsilon": 6.189, "lambda_three_body": 23.15, "box_length": box_vec(20.0)}),
                    ("si_params", {"sigma": 2.0951, "epsilon": 50.003, "lambda_three_body": 21.0, "box_length": box_vec(20.0)}),
                ]

            for rname, kwargs in recipes:
                try:
                    obj = cls(**kwargs)
                except Exception as e:
                    rows.append({"module": m, "class_name": cname, "recipe_name": rname, "ok": False, "stage": "construct", "error": repr(e)[:600]})
                    continue

                for vname, coords in coord_variants(n=4, spacing=2.5).items():
                    res = call_method(obj, "energy", coords, box_length=box_vec(20.0))
                    rows.append({
                        "module": m,
                        "class_name": cname,
                        "recipe_name": rname,
                        "coord_variant": vname,
                        "method_requested": "energy",
                        **res,
                    })

                    if include_lambda:
                        lambdas = jnp.array([21.0])
                        res2 = call_method(obj, "energy_with_lambda", coords, box_length=box_vec(20.0), lambdas_three_body=lambdas)
                        rows.append({
                            "module": m,
                            "class_name": cname,
                            "recipe_name": rname,
                            "coord_variant": vname,
                            "method_requested": "energy_with_lambda",
                            **res2,
                        })

    print(json.dumps(rows, default=str))

main()
'''


def as_float(x: Any) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) else None


def summarize_lj(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("ok") is True and isinstance(r.get("energy"), (int, float))]
    beyond = []
    inside = []
    for r in ok:
        cutoff = float(r.get("cutoff"))
        dist = float(r.get("distance"))
        e = abs(float(r.get("energy")))
        if dist > cutoff:
            beyond.append(e)
        elif dist < cutoff:
            inside.append(e)
    frac = None
    if beyond:
        frac = sum(1 for e in beyond if e < 1e-8) / len(beyond)
    return {
        "ok_points": len(ok),
        "classes": sorted(set(str(r.get("class_name")) for r in ok)),
        "cutoffs": sorted(set(float(r.get("cutoff")) for r in ok if r.get("cutoff") is not None)),
        "inside_points": len(inside),
        "beyond_points": len(beyond),
        "beyond_cutoff_near_zero_fraction": frac,
    }


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def write_recommendations(path: Path, constructor_rows: List[Dict[str, Any]], shape_rows: List[Dict[str, Any]], lj_rows: List[Dict[str, Any]], far_rows: List[Dict[str, Any]], mw_rows: List[Dict[str, Any]]) -> None:
    ctor_ok = [r for r in constructor_rows if r.get("ok")]
    shape_ok = [r for r in shape_rows if r.get("ok")]
    lj_summary = summarize_lj(lj_rows)
    far_ok = [r for r in far_rows if r.get("pair_energy_ok") and r.get("three_energy_ok")]
    mw_ok = [r for r in mw_rows if r.get("ok")]

    lines = [
        "# Probe 06 recommendations for Probe 07",
        "",
        "Probe 06 was a tiny explicit energy/locality stress test.",
        "",
        "## Constructor successes",
        "",
    ]
    if ctor_ok:
        lines += md_table(["module", "class", "recipe", "kwargs"], [[r.get("module"), r.get("class_name"), r.get("recipe_name"), r.get("kwargs_keys")] for r in ctor_ok[:80]])
    else:
        lines.append("_None._")

    lines += ["", "## Energy shape acceptance successes", ""]
    if shape_ok:
        lines += md_table(
            ["module", "class", "recipe", "coords", "method", "style", "value"],
            [[r.get("module"), r.get("class_name"), r.get("recipe_name"), r.get("coord_variant"), r.get("method"), r.get("style"), r.get("value")] for r in shape_ok[:100]]
        )
    else:
        lines.append("_None._")

    lines += [
        "",
        "## LJ cutoff summary",
        "",
        f"- `{lj_summary}`",
        "",
        "## Far-particle effect",
        "",
    ]
    if far_ok:
        lines += md_table(
            ["class", "recipe", "cutoff", "near r", "third r", "E2", "E3", "delta"],
            [[r.get("class_name"), r.get("recipe_name"), r.get("cutoff"), r.get("near_pair_distance"), r.get("third_particle_distance"), r.get("pair_energy"), r.get("three_particle_energy"), r.get("delta_three_minus_pair")] for r in far_ok[:120]]
        )
    else:
        lines.append("_No successful far-particle comparison._")

    lines += ["", "## mW / SW tiny energy successes", ""]
    if mw_ok:
        lines += md_table(
            ["module", "class", "recipe", "coords", "method", "style", "value"],
            [[r.get("module"), r.get("class_name"), r.get("recipe_name"), r.get("coord_variant"), r.get("method_requested"), r.get("style"), r.get("value")] for r in mw_ok[:100]]
        )
    else:
        lines.append("_None._")

    lines += [
        "",
        "## Suggested Probe 07",
        "",
        "Lock onto the first class/method with stable energy evaluation.",
        "",
        "Recommended:",
        "",
        "1. LJ pairwise cutoff stress: vary `cutoff`, `box_length`, and pair distance.",
        "2. Far-particle invariance: third particle outside cutoff should not affect energy if the locality boundary is hard.",
        "3. Repeat the same geometry with mW/SW if smoke succeeded.",
        "",
        "This gets directly to the paper seam: fixed/local neighborhoods imply information outside the local radius is invisible unless reintroduced through conditioning or global variables.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, out_dir: Path, official_repo: Path, constructor_rows: List[Dict[str, Any]], shape_rows: List[Dict[str, Any]], lj_rows: List[Dict[str, Any]], far_rows: List[Dict[str, Any]], mw_rows: List[Dict[str, Any]]) -> None:
    lj_summary = summarize_lj(lj_rows)
    lines = [
        "# PROBE 06 SUMMARY",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Official repo: `{rel(official_repo)}`",
        "",
        "## What this probe did",
        "",
        "- Dynamically discovered and instantiated tiny system energy classes.",
        "- Tested coordinate shapes for energy methods.",
        "- Swept Lennard-Jones pair distance across cutoff boundaries.",
        "- Compared two-particle energy to three-particle energy with a third particle near/far.",
        "- Smoked mW/SW tiny energy methods.",
        "",
        "## Headline counts",
        "",
        f"- Constructor attempts: **{len(constructor_rows)}**",
        f"- Constructor OK: **{sum(1 for r in constructor_rows if r.get('ok'))}**",
        f"- Energy shape attempts: **{len(shape_rows)}**",
        f"- Energy shape OK: **{sum(1 for r in shape_rows if r.get('ok'))}**",
        f"- LJ cutoff sweep rows: **{len(lj_rows)}**",
        f"- LJ cutoff OK rows: **{sum(1 for r in lj_rows if r.get('ok'))}**",
        f"- LJ far-particle rows: **{len(far_rows)}**",
        f"- LJ far-particle OK rows: **{sum(1 for r in far_rows if r.get('pair_energy_ok') and r.get('three_energy_ok'))}**",
        f"- mW/SW rows: **{len(mw_rows)}**",
        f"- mW/SW OK rows: **{sum(1 for r in mw_rows if r.get('ok'))}**",
        "",
        "## LJ cutoff diagnostic",
        "",
        f"- `{lj_summary}`",
        "",
        "## Generated files",
        "",
        "- `constructor_attempts.csv/json`",
        "- `energy_shape_acceptance.csv/json`",
        "- `lj_cutoff_sweep.csv/json`",
        "- `lj_far_particle_effect.csv/json`",
        "- `mw_sw_energy_smoke.csv/json`",
        "- `probe07_recommendations.md`",
        "- `logs/*.stdout.txt`, `logs/*.stderr.txt`",
        "",
        "## Next",
        "",
        "Open `probe07_recommendations.md`, then lock Probe 07 to the first stable energy/locality path.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe 06: tiny cutoff/locality stress test.")
    p.add_argument("--official-repo", default="../bgmat")
    p.add_argument("--out-root", default="probe_runs")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--include-sw-lambda", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = resolve_from_root(args.official_repo)
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe06_tiny_cutoff_locality_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")

    print("=" * 100)
    print("CURRENT PAPER PROBE 06 — TINY CUTOFF / LOCALITY STRESS")
    print("=" * 100)
    print("[ROOT]          .")
    print(f"[OFFICIAL REPO] {rel(official_repo)}")
    print(f"[OUT]           {rel(out_dir)}")

    print("[RUN] constructors + shape acceptance")
    res1 = run_subprocess(CONSTRUCTOR_CODE, official_repo, args.timeout, log_dir, "constructors_shape_acceptance")
    payload1 = parse_payload(res1, {"constructor_attempts": [], "energy_shape_acceptance": []})
    constructor_rows = payload1.get("constructor_attempts", [])
    shape_rows = payload1.get("energy_shape_acceptance", [])

    print("[RUN] LJ cutoff sweep + far-particle effect")
    res2 = run_subprocess(LJ_CODE, official_repo, args.timeout, log_dir, "lj_cutoff_far_particle")
    payload2 = parse_payload(res2, {"lj_cutoff_sweep": [], "lj_far_particle_effect": []})
    lj_rows = payload2.get("lj_cutoff_sweep", [])
    far_rows = payload2.get("lj_far_particle_effect", [])

    print("[RUN] mW/SW energy smoke")
    res3 = run_subprocess(MW_SW_CODE, official_repo, args.timeout, log_dir, "mw_sw_energy_smoke", args=["true" if args.include_sw_lambda else "false"])
    mw_rows = parse_payload(res3, [])

    if not constructor_rows and not shape_rows and not res1.get("ok"):
        constructor_rows = [{"ok": False, "stage": "subprocess", "error": str(res1.get("stderr", ""))[-1000:] or res1.get("error", ""), "stdout_log": res1.get("stdout_log"), "stderr_log": res1.get("stderr_log")}]
    if not lj_rows and not far_rows and not res2.get("ok"):
        lj_rows = [{"ok": False, "stage": "subprocess", "error": str(res2.get("stderr", ""))[-1000:] or res2.get("error", ""), "stdout_log": res2.get("stdout_log"), "stderr_log": res2.get("stderr_log")}]
    if not mw_rows and not res3.get("ok"):
        mw_rows = [{"ok": False, "stage": "subprocess", "error": str(res3.get("stderr", ""))[-1000:] or res3.get("error", ""), "stdout_log": res3.get("stdout_log"), "stderr_log": res3.get("stderr_log")}]

    write_csv(out_dir / "constructor_attempts.csv", constructor_rows)
    write_json(out_dir / "constructor_attempts.json", constructor_rows)
    write_csv(out_dir / "energy_shape_acceptance.csv", shape_rows)
    write_json(out_dir / "energy_shape_acceptance.json", shape_rows)
    write_csv(out_dir / "lj_cutoff_sweep.csv", lj_rows)
    write_json(out_dir / "lj_cutoff_sweep.json", lj_rows)
    write_csv(out_dir / "lj_far_particle_effect.csv", far_rows)
    write_json(out_dir / "lj_far_particle_effect.json", far_rows)
    write_csv(out_dir / "mw_sw_energy_smoke.csv", mw_rows)
    write_json(out_dir / "mw_sw_energy_smoke.json", mw_rows)

    write_recommendations(out_dir / "probe07_recommendations.md", constructor_rows, shape_rows, lj_rows, far_rows, mw_rows)
    write_summary(out_dir / "PROBE06_SUMMARY.md", out_dir, official_repo, constructor_rows, shape_rows, lj_rows, far_rows, mw_rows)

    print("\n[SUMMARY]")
    print(f"  constructor_attempts : {len(constructor_rows)}")
    print(f"  constructor_ok       : {sum(1 for r in constructor_rows if r.get('ok'))}")
    print(f"  energy_shape_rows    : {len(shape_rows)}")
    print(f"  energy_shape_ok      : {sum(1 for r in shape_rows if r.get('ok'))}")
    print(f"  lj_sweep_rows        : {len(lj_rows)}")
    print(f"  lj_sweep_ok          : {sum(1 for r in lj_rows if r.get('ok'))}")
    print(f"  far_particle_rows    : {len(far_rows)}")
    print(f"  far_particle_ok      : {sum(1 for r in far_rows if r.get('pair_energy_ok') and r.get('three_energy_ok'))}")
    print(f"  mw_sw_rows           : {len(mw_rows)}")
    print(f"  mw_sw_ok             : {sum(1 for r in mw_rows if r.get('ok'))}")
    print(f"  output               : {rel(out_dir)}")
    print("\nNext:")
    print(f"  open {rel(out_dir / 'PROBE06_SUMMARY.md')}")
    print(f"  open {rel(out_dir / 'probe07_recommendations.md')}")
    print(f"  open {rel(out_dir / 'lj_cutoff_sweep.csv')}")


if __name__ == "__main__":
    main()
