
#!/usr/bin/env python3
# current_probe_05_tiny_energy_seam_probe.py
#
# Current paper Probe 05
# Portable tiny-code execution probe for bgmat energy/locality seams.
#
# Run from repo root:
#   python .\current_probe_05_tiny_energy_seam_probe.py
#
# Optional tiny dummy method smoke:
#   python .\current_probe_05_tiny_energy_seam_probe.py --method-smoke
#
# No training. No hardcoded local paths. Outputs under probe_runs/.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent

DEFAULT_TARGET_MODULES = [
    "bgmat.systems.energies",
    "bgmat.systems.lennard_jones",
    "bgmat.systems.monatomic_water",
    "bgmat.systems.stillinger_weber",
    "bgmat.utils.lattice_utils",
    "bgmat.utils.marginal",
    "bgmat.utils.observable_utils",
    "bgmat.experiments.utils",
    "bgmat.models.particle_models",
    "bgmat.models.gnn_conditioner",
    "bgmat.models.augmented_coupling_flows",
    "bgmat.models.conditional_split_coupling",
    "bgmat.experiments.augmented_lennard_jones_config",
    "bgmat.experiments.augmented_monatomic_water_config",
    "bgmat.experiments.augmented_monatomic_water_config_lambda_vol_conditional",
]

LOCALITY_PARAM_PATTERNS = [
    "cutoff", "r_cut", "rcut", "radius", "neighbor", "neighbour",
    "n_neighbor", "n_neighbour", "n_edges", "edges", "senders", "receivers",
    "box", "cell", "lattice", "volume", "density", "pbc", "periodic",
    "sigma", "epsilon", "lambda", "lambda3", "temperature", "pressure",
    "n_particles", "n_atoms", "num_particles", "num_atoms", "n_nodes",
    "aux", "auxiliary", "n_aux", "num_aux", "marginal",
]

ENERGY_METHOD_NAMES = [
    "__call__", "energy", "energies", "potential", "potential_energy",
    "energy_fn", "log_prob", "force", "forces", "gradient", "compute", "evaluate",
]

HEAVY_NAME_HINTS = [
    "train", "training", "fit", "epoch", "optimizer", "checkpoint",
    "sample_many", "simulation", "minimize", "md", "langevin",
]


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


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def module_to_path(module: str, official_repo: Path) -> Path:
    parts = module.split(".")
    if parts and parts[0] == "bgmat":
        parts = parts[1:]
    package_dir = official_repo / "bgmat"
    if not parts:
        return package_dir / "__init__.py"
    as_file = package_dir.joinpath(*parts).with_suffix(".py")
    as_init = package_dir.joinpath(*parts) / "__init__.py"
    return as_file if as_file.exists() else as_init


def safe_unparse(node: ast.AST, max_len: int = 220) -> str:
    try:
        s = ast.unparse(node)
    except Exception:
        s = node.__class__.__name__
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def first_doc_line(node: ast.AST) -> str:
    try:
        d = ast.get_docstring(node) or ""
    except Exception:
        d = ""
    return d.strip().splitlines()[0][:240] if d.strip() else ""


def contains_param_hint(s: str) -> bool:
    low = s.lower()
    return any(p.lower() in low for p in LOCALITY_PARAM_PATTERNS)


def is_probably_heavy_name(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in HEAVY_NAME_HINTS)


def signature_from_ast_func(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    pos = list(node.args.posonlyargs) + list(node.args.args)
    defaults = list(node.args.defaults)
    default_offset = len(pos) - len(defaults)
    for i, a in enumerate(pos):
        s = a.arg
        if a.annotation:
            s += ": " + safe_unparse(a.annotation, 80)
        if i >= default_offset:
            s += "=" + safe_unparse(defaults[i - default_offset], 80)
        args.append(s)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults):
        s = a.arg
        if a.annotation:
            s += ": " + safe_unparse(a.annotation, 80)
        if d is not None:
            s += "=" + safe_unparse(d, 80)
        args.append(s)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return f"{node.name}(" + ", ".join(args) + ")"


def required_args_from_ast_func(node: ast.FunctionDef, drop_self: bool) -> int:
    args = list(node.args.posonlyargs) + list(node.args.args)
    if drop_self and args and args[0].arg in {"self", "cls"}:
        args = args[1:]
    defaults = list(node.args.defaults)
    req_pos = max(0, len(args) - len(defaults))
    req_kw = sum(1 for d in node.args.kw_defaults if d is None)
    return req_pos + req_kw


def class_zero_required_args(cls: ast.ClassDef) -> Tuple[bool, int, str]:
    init_node = None
    for b in cls.body:
        if isinstance(b, ast.FunctionDef) and b.name == "__init__":
            init_node = b
            break
    if init_node is None:
        return True, 0, "object_default_init"
    req = required_args_from_ast_func(init_node, drop_self=True)
    return req == 0, req, signature_from_ast_func(init_node)


def analyze_static_module(module: str, official_repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    path = module_to_path(module, official_repo)
    text = read_text(path)
    symbols: List[Dict[str, Any]] = []
    params: List[Dict[str, Any]] = []

    if not path.exists():
        return ([{
            "module": module, "file": rel(path), "kind": "module", "name": "",
            "lineno": "", "signature": "", "doc_first_line": "", "error": "file_missing",
        }], [])

    try:
        tree = ast.parse(text, filename=str(path))
    except Exception as exc:
        return ([{
            "module": module, "file": rel(path), "kind": "module", "name": "",
            "lineno": "", "signature": "", "doc_first_line": "",
            "error": f"parse_error: {exc!r}",
        }], [])

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            zero, req, init_sig = class_zero_required_args(node)
            bases = ";".join(safe_unparse(b, 100) for b in node.bases)
            symbols.append({
                "module": module,
                "file": rel(path),
                "kind": "class",
                "name": node.name,
                "lineno": getattr(node, "lineno", ""),
                "signature": f"class {node.name}; init={init_sig}; required_init_args={req}; zero_arg={zero}",
                "doc_first_line": first_doc_line(node),
                "bases": bases,
                "heavy_name_hint": is_probably_heavy_name(node.name),
                "zero_arg_class": zero,
                "required_init_args": req,
                "error": "",
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = signature_from_ast_func(node)
            req = required_args_from_ast_func(node, drop_self=False)
            symbols.append({
                "module": module,
                "file": rel(path),
                "kind": "function",
                "name": node.name,
                "lineno": getattr(node, "lineno", ""),
                "signature": sig,
                "doc_first_line": first_doc_line(node),
                "bases": "",
                "heavy_name_hint": is_probably_heavy_name(node.name),
                "zero_arg_class": "",
                "required_args_guess": req,
                "error": "",
            })

            for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                if contains_param_hint(arg.arg):
                    params.append({
                        "module": module,
                        "file": rel(path),
                        "owner_kind": "function_or_method",
                        "owner_name": node.name,
                        "lineno": getattr(node, "lineno", ""),
                        "param_or_symbol": arg.arg,
                        "signature": sig,
                        "default_or_value": "",
                        "hint_type": "signature_parameter",
                    })

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: List[str] = []
            value = ""
            if isinstance(node, ast.Assign):
                targets = [safe_unparse(t, 100) for t in node.targets]
                value = safe_unparse(node.value, 180)
            else:
                targets = [safe_unparse(node.target, 100)]
                value = safe_unparse(node.value, 180) if node.value else ""
            for t in targets:
                if contains_param_hint(t):
                    params.append({
                        "module": module,
                        "file": rel(path),
                        "owner_kind": "assignment",
                        "owner_name": "",
                        "lineno": getattr(node, "lineno", ""),
                        "param_or_symbol": t,
                        "signature": "",
                        "default_or_value": value,
                        "hint_type": "assignment",
                    })
    return symbols, params


def run_subprocess(cmd: Sequence[str], cwd: Path, official_repo: Path, timeout: int) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(official_repo) + os.pathsep + env.get("PYTHONPATH", "")
    rec = {
        "cmd": list(cmd), "cwd": rel(cwd), "timeout_s": timeout, "ok": False,
        "returncode": None, "stdout": "", "stderr": "", "error": "",
    }
    try:
        p = subprocess.run(
            list(cmd), cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, env=env,
        )
        rec.update(ok=(p.returncode == 0), returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def write_logs(log_dir: Path, prefix: str, result: Dict[str, Any]) -> Dict[str, str]:
    ensure_dir(log_dir)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("_")[:110]
    h = sha16(str(result.get("cmd", "")) + str(result.get("cwd", "")))
    stdout = log_dir / f"{safe}_{h}.stdout.txt"
    stderr = log_dir / f"{safe}_{h}.stderr.txt"
    stdout.write_text(str(result.get("stdout", "")), encoding="utf-8")
    stderr.write_text(str(result.get("stderr", "")), encoding="utf-8")
    return {"stdout_log": rel(stdout), "stderr_log": rel(stderr)}


DYNAMIC_INSPECT_CODE = '''
import importlib, inspect, json, sys, time
module_name = sys.argv[1]
t0 = time.perf_counter()
out = {"module": module_name, "ok": False, "seconds": None, "error": "", "file": ""}
symbols = []
try:
    mod = importlib.import_module(module_name)
    out["ok"] = True
    out["seconds"] = time.perf_counter() - t0
    out["file"] = str(getattr(mod, "__file__", ""))
    for name, obj in inspect.getmembers(mod):
        if name.startswith("__") and name != "__call__":
            continue
        if inspect.isclass(obj):
            kind = "class"
        elif inspect.isfunction(obj):
            kind = "function"
        else:
            continue
        try:
            sig = str(inspect.signature(obj))
        except Exception as e:
            sig = f"<signature_error {e!r}>"
        try:
            doc = inspect.getdoc(obj) or ""
            docline = doc.splitlines()[0][:240] if doc else ""
        except Exception:
            docline = ""
        try:
            source_file = inspect.getsourcefile(obj) or ""
        except Exception:
            source_file = ""
        symbols.append({
            "module": module_name, "kind": kind, "name": name, "signature": sig,
            "doc_first_line": docline, "source_file": source_file,
        })
except Exception as e:
    out["seconds"] = time.perf_counter() - t0
    out["error"] = repr(e)
print(json.dumps({"import": out, "symbols": symbols}, default=str))
if not out["ok"]:
    raise SystemExit(1)
'''


INSTANTIATE_CODE = '''
import importlib, json, sys, time
module_name = sys.argv[1]
class_names = sys.argv[2].split("|") if sys.argv[2] else []
rows = []
try:
    mod = importlib.import_module(module_name)
except Exception as e:
    for c in class_names:
        rows.append({"module": module_name, "class_name": c, "ok": False, "seconds": 0, "error": f"module_import_failed {e!r}", "repr": ""})
    print(json.dumps(rows))
    raise SystemExit(1)

for c in class_names:
    t0 = time.perf_counter()
    try:
        cls = getattr(mod, c)
        obj = cls()
        rows.append({
            "module": module_name, "class_name": c, "ok": True,
            "seconds": time.perf_counter() - t0, "error": "",
            "repr": repr(obj)[:400], "type": str(type(obj)),
        })
    except Exception as e:
        rows.append({
            "module": module_name, "class_name": c, "ok": False,
            "seconds": time.perf_counter() - t0, "error": repr(e), "repr": "", "type": "",
        })
print(json.dumps(rows, default=str))
'''


METHOD_SMOKE_CODE = '''
import importlib, inspect, json, sys, time
import numpy as np
try:
    import jax.numpy as jnp
except Exception:
    jnp = None

module_name = sys.argv[1]
class_names = sys.argv[2].split("|") if sys.argv[2] else []
energy_method_names = set(sys.argv[3].split("|"))
rows = []

def dummy_values():
    arr_np_4x3 = np.zeros((4, 3), dtype=float)
    arr_np_batch = np.zeros((1, 12), dtype=float)
    box_np = np.eye(3, dtype=float) * 10.0
    vol_np = 1000.0
    if jnp is not None:
        arr_jnp_4x3 = jnp.zeros((4, 3))
        arr_jnp_batch = jnp.zeros((1, 12))
        box_jnp = jnp.eye(3) * 10.0
    else:
        arr_jnp_4x3 = arr_np_4x3
        arr_jnp_batch = arr_np_batch
        box_jnp = box_np
    return {
        "positions_np_4x3": arr_np_4x3, "positions_jnp_4x3": arr_jnp_4x3,
        "flat_np_batch": arr_np_batch, "flat_jnp_batch": arr_jnp_batch,
        "box_np": box_np, "box_jnp": box_jnp, "volume": vol_np,
    }

def result_summary(x):
    try:
        return {"repr": repr(x)[:300], "shape": str(getattr(x, "shape", "")), "dtype": str(getattr(x, "dtype", ""))}
    except Exception:
        return {"repr": repr(x)[:300], "shape": "", "dtype": ""}

def required_params(sig):
    req = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            req.append(name)
    return req

def build_arg_sets(param_names):
    d = dummy_values()
    if len(param_names) == 0:
        return [("no_args", [], {})]
    if len(param_names) == 1:
        p = param_names[0].lower()
        if "volume" in p or p in {"vol"}:
            return [("volume", [d["volume"]], {})]
        if "box" in p or "cell" in p:
            return [("box_jnp", [d["box_jnp"]], {}), ("box_np", [d["box_np"]], {})]
        return [
            ("positions_jnp_4x3", [d["positions_jnp_4x3"]], {}),
            ("flat_jnp_batch", [d["flat_jnp_batch"]], {}),
            ("positions_np_4x3", [d["positions_np_4x3"]], {}),
            ("flat_np_batch", [d["flat_np_batch"]], {}),
        ]
    if len(param_names) == 2:
        return [
            ("positions_box_jnp", [d["positions_jnp_4x3"], d["box_jnp"]], {}),
            ("flat_volume_jnp", [d["flat_jnp_batch"], d["volume"]], {}),
            ("positions_box_np", [d["positions_np_4x3"], d["box_np"]], {}),
            ("flat_volume_np", [d["flat_np_batch"], d["volume"]], {}),
        ]
    return []

try:
    mod = importlib.import_module(module_name)
except Exception as e:
    for c in class_names:
        rows.append({"module": module_name, "class_name": c, "method": "", "ok": False, "error": f"module_import_failed {e!r}"})
    print(json.dumps(rows))
    raise SystemExit(1)

for c in class_names:
    try:
        cls = getattr(mod, c)
        obj = cls()
    except Exception as e:
        rows.append({"module": module_name, "class_name": c, "method": "", "ok": False, "error": f"instantiate_failed {e!r}"})
        continue

    for mname in energy_method_names:
        try:
            meth = obj if mname == "__call__" and callable(obj) else getattr(obj, mname, None)
            if meth is None or not callable(meth):
                continue
            try:
                sig = inspect.signature(meth)
            except Exception as e:
                rows.append({"module": module_name, "class_name": c, "method": mname, "ok": False, "error": f"signature_failed {e!r}"})
                continue
            req = required_params(sig)
            if len(req) > 2:
                rows.append({"module": module_name, "class_name": c, "method": mname, "ok": False, "signature": str(sig), "required_params": ",".join(req), "error": "too_many_required_params"})
                continue
            arg_sets = build_arg_sets(req)
            if not arg_sets:
                rows.append({"module": module_name, "class_name": c, "method": mname, "ok": False, "signature": str(sig), "required_params": ",".join(req), "error": "no_dummy_arg_set"})
                continue
            success = False
            last_err = ""
            for label, args, kwargs in arg_sets:
                t0 = time.perf_counter()
                try:
                    y = meth(*args, **kwargs)
                    summ = result_summary(y)
                    rows.append({
                        "module": module_name, "class_name": c, "method": mname, "ok": True,
                        "arg_set": label, "seconds": time.perf_counter() - t0,
                        "signature": str(sig), "required_params": ",".join(req),
                        "result_repr": summ["repr"], "result_shape": summ["shape"],
                        "result_dtype": summ["dtype"], "error": "",
                    })
                    success = True
                    break
                except Exception as e:
                    last_err = repr(e)
            if not success:
                rows.append({
                    "module": module_name, "class_name": c, "method": mname, "ok": False,
                    "signature": str(sig), "required_params": ",".join(req),
                    "error": last_err[:500],
                })
        except Exception as e:
            rows.append({"module": module_name, "class_name": c, "method": mname, "ok": False, "error": f"outer_failed {e!r}"})

print(json.dumps(rows, default=str))
'''


def dynamic_inspect_module(module: str, official_repo: Path, timeout: int, log_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    res = run_subprocess([sys.executable, "-c", DYNAMIC_INSPECT_CODE, module], cwd=REPO_ROOT, official_repo=official_repo, timeout=timeout)
    logs = write_logs(log_dir, f"dynamic_inspect_{module}", res)
    payload = {"import": {"module": module, "ok": False, "error": "no_json"}, "symbols": []}
    try:
        for line in reversed(str(res.get("stdout", "")).splitlines()):
            if line.strip().startswith("{"):
                payload = json.loads(line)
                break
    except Exception as exc:
        payload = {"import": {"module": module, "ok": False, "error": f"json_parse_error {exc!r}"}, "symbols": []}
    import_row = payload.get("import", {})
    import_row.update({
        "returncode": res.get("returncode"),
        "stderr_tail": str(res.get("stderr", ""))[-900:],
        "process_error": res.get("error", ""),
        **logs,
    })
    dyn_symbols = payload.get("symbols", [])
    for s in dyn_symbols:
        s["import_ok"] = import_row.get("ok")
    return import_row, dyn_symbols


def zero_arg_instantiation(module: str, class_names: Sequence[str], official_repo: Path, timeout: int, log_dir: Path) -> List[Dict[str, Any]]:
    if not class_names:
        return []
    joined = "|".join(class_names)
    res = run_subprocess([sys.executable, "-c", INSTANTIATE_CODE, module, joined], cwd=REPO_ROOT, official_repo=official_repo, timeout=timeout)
    logs = write_logs(log_dir, f"instantiate_{module}", res)
    rows: List[Dict[str, Any]] = []
    try:
        for line in reversed(str(res.get("stdout", "")).splitlines()):
            if line.strip().startswith("["):
                rows = json.loads(line)
                break
    except Exception as exc:
        rows = [{"module": module, "class_name": "", "ok": False, "seconds": "", "error": f"json_parse_error {exc!r}", "repr": "", "type": ""}]
    for r in rows:
        r.update({
            "returncode": res.get("returncode"),
            "process_error": res.get("error", ""),
            "stderr_tail": str(res.get("stderr", ""))[-900:],
            **logs,
        })
    return rows


def method_smoke(module: str, class_names: Sequence[str], official_repo: Path, timeout: int, log_dir: Path) -> List[Dict[str, Any]]:
    if not class_names:
        return []
    joined = "|".join(class_names)
    methods = "|".join(ENERGY_METHOD_NAMES)
    res = run_subprocess([sys.executable, "-c", METHOD_SMOKE_CODE, module, joined, methods], cwd=REPO_ROOT, official_repo=official_repo, timeout=timeout)
    logs = write_logs(log_dir, f"method_smoke_{module}", res)
    rows: List[Dict[str, Any]] = []
    try:
        for line in reversed(str(res.get("stdout", "")).splitlines()):
            if line.strip().startswith("["):
                rows = json.loads(line)
                break
    except Exception as exc:
        rows = [{"module": module, "class_name": "", "method": "", "ok": False, "error": f"json_parse_error {exc!r}"}]
    for r in rows:
        r.update({
            "returncode": res.get("returncode"),
            "process_error": res.get("error", ""),
            "stderr_tail": str(res.get("stderr", ""))[-900:],
            **logs,
        })
    return rows


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def write_recommendations(path: Path, imports: List[Dict[str, Any]], params: List[Dict[str, Any]], inst: List[Dict[str, Any]], method_rows: List[Dict[str, Any]]) -> None:
    ok_modules = {r["module"] for r in imports if r.get("ok")}
    failed_modules = [(r["module"], r.get("error") or r.get("stderr_tail", "")) for r in imports if not r.get("ok")]

    param_by_module: Dict[str, int] = {}
    for p in params:
        param_by_module[p["module"]] = param_by_module.get(p["module"], 0) + 1

    inst_ok = [r for r in inst if r.get("ok")]
    method_ok = [r for r in method_rows if r.get("ok")]

    lines = [
        "# Probe 05 tiny path recommendations",
        "",
        "This file picks the next execution target based on tiny import/constructor/method evidence.",
        "",
        "## Import-OK target modules",
        "",
    ]
    for m in sorted(ok_modules):
        lines.append(f"- `{m}`")

    lines += ["", "## Failed target modules", ""]
    if not failed_modules:
        lines.append("_None._")
    else:
        for m, e in failed_modules:
            lines.append(f"- `{m}` — `{str(e)[:240]}`")

    lines += ["", "## Locality / cutoff / geometry parameter density", ""]
    table = [[m, n] for m, n in sorted(param_by_module.items(), key=lambda kv: (-kv[1], kv[0]))[:30]]
    lines += md_table(["module", "parameter/assignment hits"], table) if table else ["_No parameter hits._"]

    lines += ["", "## Zero-arg constructors that worked", ""]
    if inst_ok:
        lines += md_table(["module", "class", "repr"], [[r.get("module"), r.get("class_name"), str(r.get("repr", ""))[:160]] for r in inst_ok[:40]])
    else:
        lines.append("_None, or not run._")

    lines += ["", "## Method smoke successes", ""]
    if method_ok:
        lines += md_table(
            ["module", "class", "method", "arg_set", "result_shape", "result_repr"],
            [[r.get("module"), r.get("class_name"), r.get("method"), r.get("arg_set"), r.get("result_shape"), str(r.get("result_repr", ""))[:160]] for r in method_ok[:40]]
        )
    else:
        lines.append("_None, or not run._")

    lines += [
        "",
        "## Suggested Probe 06 target",
        "",
        "Start with the simplest import-OK system energy module that has locality/cutoff geometry parameters and either:",
        "",
        "1. a working zero-arg constructor, or",
        "2. a clear constructor signature from `dynamic_symbol_signatures.csv`.",
        "",
        "Likely first targets, unless Probe 05 output says otherwise:",
        "",
        "- `bgmat.systems.lennard_jones` for simplest pairwise/local cutoff seam.",
        "- `bgmat.systems.monatomic_water` for mW ice/source-data relevance.",
        "- `bgmat.systems.stillinger_weber` for phase/λ seam.",
        "- `bgmat.utils.lattice_utils` for controlled geometry/cell generation.",
        "- `bgmat.utils.marginal` for auxiliary marginal estimator seam.",
        "",
        "Probe 06 should instantiate one explicit small system and evaluate energy/locality behavior across tiny sizes/cutoffs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, out_dir: Path, official_repo: Path, imports: List[Dict[str, Any]], static_symbols: List[Dict[str, Any]], dynamic_symbols: List[Dict[str, Any]], params: List[Dict[str, Any]], inst: List[Dict[str, Any]], method_rows: List[Dict[str, Any]]) -> None:
    import_ok = sum(1 for r in imports if r.get("ok"))
    import_fail = len(imports) - import_ok
    inst_ok = sum(1 for r in inst if r.get("ok"))
    method_ok = sum(1 for r in method_rows if r.get("ok"))

    top_param_modules: Dict[str, int] = {}
    for p in params:
        top_param_modules[p["module"]] = top_param_modules.get(p["module"], 0) + 1

    lines = [
        "# PROBE 05 SUMMARY",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Official repo: `{rel(official_repo)}`",
        "",
        "## What this probe did",
        "",
        "- Inspected target system/model/utility modules from Probe 04.",
        "- Imported each module through repo-relative `PYTHONPATH`.",
        "- Cataloged static and dynamic symbols/signatures.",
        "- Extracted locality/cutoff/geometry/marginal parameter seams.",
        "- Instantiated zero-required-argument classes.",
        "- Optionally method-smoked tiny energy-like calls.",
        "",
        "## Headline counts",
        "",
        f"- Target modules: **{len(imports)}**",
        f"- Imports OK: **{import_ok}**",
        f"- Imports failed: **{import_fail}**",
        f"- Static symbols: **{len(static_symbols)}**",
        f"- Dynamic symbols: **{len(dynamic_symbols)}**",
        f"- Locality/geometry/marginal parameter hits: **{len(params)}**",
        f"- Zero-arg instantiation attempts: **{len(inst)}**",
        f"- Zero-arg instantiations OK: **{inst_ok}**",
        f"- Method smoke rows: **{len(method_rows)}**",
        f"- Method smoke successes: **{method_ok}**",
        "",
        "## Top parameter-seam modules",
        "",
    ]
    for m, n in sorted(top_param_modules.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        lines.append(f"- `{m}`: {n} hit(s)")
    lines += [
        "",
        "## Generated files",
        "",
        "- `target_module_imports.csv/json`",
        "- `static_symbol_catalog.csv/json`",
        "- `dynamic_symbol_signatures.csv/json`",
        "- `locality_parameter_catalog.csv/json`",
        "- `zero_arg_instantiation.csv/json`",
        "- `method_smoke_results.csv/json`",
        "- `tiny_path_recommendations.md`",
        "- `logs/*.stdout.txt`, `logs/*.stderr.txt`",
        "",
        "## Next",
        "",
        "Open `tiny_path_recommendations.md` and `locality_parameter_catalog.csv`.",
        "Probe 06 should be a fixed, explicit tiny system/cutoff/locality stress test.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Current paper Probe 05: tiny energy/locality seam probe.")
    p.add_argument("--official-repo", default="../bgmat", help="Official repo folder relative to repo root unless absolute.")
    p.add_argument("--out-root", default="probe_runs", help="Output root relative to repo root unless absolute.")
    p.add_argument("--module", action="append", default=[], help="Override/add target module. Repeatable.")
    p.add_argument("--only-user-modules", action="store_true", help="Use only --module values, not default targets.")
    p.add_argument("--import-timeout", type=int, default=60, help="Timeout seconds per module dynamic inspection.")
    p.add_argument("--instantiate-timeout", type=int, default=60, help="Timeout seconds per module zero-arg instantiation batch.")
    p.add_argument("--method-smoke", action="store_true", help="Attempt tiny dummy calls on zero-arg energy-like methods.")
    p.add_argument("--method-timeout", type=int, default=90, help="Timeout seconds per module method-smoke batch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    official_repo = resolve_from_root(args.official_repo)
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe05_tiny_energy_seams_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")

    target_modules = list(dict.fromkeys(args.module if args.only_user_modules else DEFAULT_TARGET_MODULES + args.module))

    print("=" * 100)
    print("CURRENT PAPER PROBE 05 — TINY ENERGY / LOCALITY SEAM PROBE")
    print("=" * 100)
    print("[ROOT]          .")
    print(f"[OFFICIAL REPO] {rel(official_repo)}")
    print(f"[OUT]           {rel(out_dir)}")
    print(f"[TARGETS]       {len(target_modules)} modules")

    static_symbols: List[Dict[str, Any]] = []
    params: List[Dict[str, Any]] = []
    for m in target_modules:
        sym, par = analyze_static_module(m, official_repo)
        static_symbols.extend(sym)
        params.extend(par)

    imports: List[Dict[str, Any]] = []
    dynamic_symbols: List[Dict[str, Any]] = []
    print("[DYNAMIC] module imports + signatures")
    for i, m in enumerate(target_modules, start=1):
        print(f"  [{i:02d}/{len(target_modules):02d}] {m}")
        imp, dyn = dynamic_inspect_module(m, official_repo, timeout=args.import_timeout, log_dir=log_dir)
        imports.append(imp)
        dynamic_symbols.extend(dyn)

    import_ok_modules = {r["module"] for r in imports if r.get("ok")}
    zero_by_module: Dict[str, List[str]] = {}
    for s in static_symbols:
        if s.get("kind") == "class" and s.get("zero_arg_class") is True and s.get("module") in import_ok_modules:
            if not is_probably_heavy_name(str(s.get("name", ""))):
                zero_by_module.setdefault(str(s["module"]), []).append(str(s["name"]))

    inst_rows: List[Dict[str, Any]] = []
    print("[INSTANTIATE] zero-arg classes")
    for m, classes in sorted(zero_by_module.items()):
        classes = list(dict.fromkeys(classes))
        if not classes:
            continue
        print(f"  {m}: {', '.join(classes[:8])}{'...' if len(classes) > 8 else ''}")
        inst_rows.extend(zero_arg_instantiation(m, classes, official_repo, timeout=args.instantiate_timeout, log_dir=log_dir))

    method_rows: List[Dict[str, Any]] = []
    if args.method_smoke:
        print("[METHOD] tiny dummy method smoke")
        ok_classes_by_module: Dict[str, List[str]] = {}
        for r in inst_rows:
            if r.get("ok"):
                cname = str(r.get("class_name", ""))
                m = str(r.get("module", ""))
                if any(k in cname.lower() for k in ["energy", "potential", "lj", "water", "silicon"]):
                    ok_classes_by_module.setdefault(m, []).append(cname)
        for m, classes in sorted(ok_classes_by_module.items()):
            print(f"  {m}: {', '.join(classes)}")
            method_rows.extend(method_smoke(m, classes, official_repo, timeout=args.method_timeout, log_dir=log_dir))

    write_csv(out_dir / "target_module_imports.csv", imports)
    write_json(out_dir / "target_module_imports.json", imports)
    write_csv(out_dir / "static_symbol_catalog.csv", static_symbols)
    write_json(out_dir / "static_symbol_catalog.json", static_symbols)
    write_csv(out_dir / "dynamic_symbol_signatures.csv", dynamic_symbols)
    write_json(out_dir / "dynamic_symbol_signatures.json", dynamic_symbols)
    write_csv(out_dir / "locality_parameter_catalog.csv", params)
    write_json(out_dir / "locality_parameter_catalog.json", params)
    write_csv(out_dir / "zero_arg_instantiation.csv", inst_rows)
    write_json(out_dir / "zero_arg_instantiation.json", inst_rows)
    write_csv(out_dir / "method_smoke_results.csv", method_rows)
    write_json(out_dir / "method_smoke_results.json", method_rows)

    write_recommendations(out_dir / "tiny_path_recommendations.md", imports, params, inst_rows, method_rows)
    write_summary(out_dir / "PROBE05_SUMMARY.md", out_dir, official_repo, imports, static_symbols, dynamic_symbols, params, inst_rows, method_rows)

    print("\n[SUMMARY]")
    print(f"  target_modules       : {len(imports)}")
    print(f"  imports_ok           : {sum(1 for r in imports if r.get('ok'))}")
    print(f"  imports_failed       : {sum(1 for r in imports if not r.get('ok'))}")
    print(f"  static_symbols       : {len(static_symbols)}")
    print(f"  dynamic_symbols      : {len(dynamic_symbols)}")
    print(f"  parameter_hits       : {len(params)}")
    print(f"  instantiation_rows   : {len(inst_rows)}")
    print(f"  instantiation_ok     : {sum(1 for r in inst_rows if r.get('ok'))}")
    print(f"  method_smoke_rows    : {len(method_rows)}")
    print(f"  method_smoke_ok      : {sum(1 for r in method_rows if r.get('ok'))}")
    print(f"  output               : {rel(out_dir)}")
    print("\nNext:")
    print(f"  open {rel(out_dir / 'PROBE05_SUMMARY.md')}")
    print(f"  open {rel(out_dir / 'tiny_path_recommendations.md')}")
    print(f"  open {rel(out_dir / 'locality_parameter_catalog.csv')}")


if __name__ == "__main__":
    main()
