#!/usr/bin/env python3
r"""
current_probe_04_module_seam_map.py

Current paper Probe 04
======================

Purpose
-------
Portable, no-training module/import/seam map for the current paper sandbox.

This probe moves past packaging and into the actual code surface:

- Which bgmat modules import cleanly?
- Which modules touch JAX / Haiku / Distrax / OpenMM?
- Which files control locality, cutoff, neighbor graph, auxiliary variables,
  marginal sampling, free-energy estimators, system size, and phase variables?
- Where are the likely smallest non-training execution paths?
- Which classes/functions exist, and which zero-arg constructors can optionally
  instantiate safely?

It does not run training.
It does not install dependencies.
It does not hardcode local machine paths.

Run from repo root:

  python .\current_probe_04_module_seam_map.py

Recommended after modern dependency compatibility is green:

  python .\current_probe_04_module_seam_map.py --import-modules

Optional extra class smoke, still conservative:

  python .\current_probe_04_module_seam_map.py --import-modules --instantiate-zero-arg

Outputs
-------
  probe_runs/current_probe04_module_seams_<timestamp>/
    PROBE04_SUMMARY.md
    module_static_seams.csv
    module_static_seams.json
    module_import_results.csv
    module_import_results.json
    symbol_catalog.csv
    symbol_catalog.json
    class_signature_catalog.csv
    class_signature_catalog.json
    instantiation_results.csv
    seam_index.md
    jax_openmm_entrypoints.md
    candidate_tiny_paths.md
    logs/*.stdout.txt
    logs/*.stderr.txt

Probe contract
--------------
Probe 04 is a code-surface map. It establishes where to aim Probe 05.
It does not validate the paper's scientific claims yet.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import datetime as _dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Seam vocabulary
# -----------------------------------------------------------------------------

SEAM_KEYWORDS: Dict[str, List[str]] = {
    "locality": [
        "local", "local_env", "neighbour", "neighbor", "neighborhood", "radius",
        "cutoff", "r_cut", "rcut", "edge", "edges", "graph", "adjacency",
        "senders", "receivers", "sparse", "mask",
    ],
    "geometry_periodic": [
        "periodic", "pbc", "box", "cell", "lattice", "minimum_image",
        "minimum image", "displacement", "distance", "wrap", "fractional",
    ],
    "augmentation_flow": [
        "augmented", "aux", "auxiliary", "split", "coupling", "flow",
        "bijection", "inverse", "forward", "log_det", "logdet", "conditioner",
    ],
    "jax_tracing": [
        "jax.jit", "@jax.jit", "jit(", "jax.vmap", "vmap(", "jax.grad",
        "grad(", "value_and_grad", "jacobian", "jacfwd", "jacrev",
        "lax.scan", "lax.fori_loop", "pmap", "xmap",
    ],
    "haiku_model": [
        "haiku", "hk.", "transform", "without_apply_rng", "Module", "Linear",
        "MLP", "Graph", "GNN", "message", "conditioner",
    ],
    "openmm_energy": [
        "openmm", "System", "Context", "Simulation", "Platform",
        "CustomExternalForce", "CustomNonbondedForce", "NonbondedForce",
        "MonteCarloBarostat", "LangevinIntegrator",
    ],
    "free_energy": [
        "free_energy", "free energy", "gibbs", "helmholtz", "mbar", "tfep",
        "ess", "effective_sample", "effective sample", "reweight", "weights",
        "delta_g", "delta_f", "log_weight", "log_weights",
    ],
    "marginal": [
        "marginal", "importance", "auxiliary_samples", "n_aux", "num_aux",
        "sample_aux", "logsumexp", "M=", "M =",
    ],
    "phase_conditioning": [
        "lambda", "lambda3", "temperature", "pressure", "volume", "density",
        "cutoff_radius", "cutoff radius", "phase", "coexistence",
        "diamond", "beta", "tin", "bcc", "fcc", "hcp", "ice", "water",
    ],
    "training": [
        "train", "training", "optimizer", "learning_rate", "lr", "epoch",
        "epochs", "batch", "checkpoint", "wandb", "tensorboard",
    ],
}

IMPORT_GROUPS: Dict[str, List[str]] = {
    "jax": ["jax", "jax.numpy", "jnp"],
    "haiku": ["haiku", "hk"],
    "distrax": ["distrax"],
    "openmm": ["openmm"],
    "tfp": ["tensorflow_probability", "tfp"],
    "numpy": ["numpy", "np"],
    "scipy": ["scipy"],
    "matplotlib": ["matplotlib", "plt"],
    "ase": ["ase"],
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class ModuleStaticRecord:
    module: str
    rel_path: str
    line_count: int
    role_guess: str
    imports: str
    import_groups: str
    seam_scores: str
    total_seam_hits: int
    classes: str
    functions: str
    top_level_assignments: str
    has_main_guard: bool
    maybe_heavy: bool
    recommended_probe05_candidate: bool


@dataclasses.dataclass
class SymbolRecord:
    module: str
    rel_path: str
    kind: str
    name: str
    lineno: int
    signature_guess: str
    seam_tags: str
    doc_first_line: str


@dataclasses.dataclass
class ClassSignatureRecord:
    module: str
    rel_path: str
    class_name: str
    lineno: int
    init_arg_count: int
    required_arg_count: int
    all_args_default_or_variadic: bool
    zero_arg_candidate: bool
    base_classes: str
    seam_tags: str


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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
    if path.is_absolute():
        return path
    return REPO_ROOT / path


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
                    fieldnames.append(k)
                    seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def read_text_safe(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
        if len(txt) > max_bytes:
            return txt[:max_bytes]
        return txt
    except Exception:
        return ""


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def module_name_from_path(py_path: Path, package_root: Path) -> str:
    relp = py_path.relative_to(package_root)
    parts = list(relp.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["bgmat"] + parts)


def safe_unparse(node: ast.AST, max_len: int = 160) -> str:
    try:
        s = ast.unparse(node)
    except Exception:
        s = node.__class__.__name__
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def first_doc_line(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node) or ""
    except Exception:
        doc = ""
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:220]


def ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = ast_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Call):
        return ast_name(node.func)
    if isinstance(node, ast.Subscript):
        return ast_name(node.value)
    return ""


def import_aliases(tree: ast.AST) -> Tuple[List[str], Dict[str, str]]:
    imports: List[str] = []
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append(a.name)
                aliases[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.append(mod)
            for a in node.names:
                aliases[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
    imports = sorted(set([i for i in imports if i]))
    return imports, aliases


def detect_import_groups(imports: Sequence[str], aliases: Dict[str, str]) -> List[str]:
    hay = " ".join(list(imports) + [f"{k}:{v}" for k, v in aliases.items()]).lower()
    groups = []
    for group, needles in IMPORT_GROUPS.items():
        for n in needles:
            nlow = n.lower()
            if nlow in hay:
                groups.append(group)
                break
    return sorted(set(groups))


def seam_hit_counts(text: str) -> Dict[str, int]:
    low = text.lower()
    counts: Dict[str, int] = {}
    for seam, needles in SEAM_KEYWORDS.items():
        n = 0
        for needle in needles:
            n += low.count(needle.lower())
        counts[seam] = n
    return counts


def seam_tags_for_text(text: str, min_hits: int = 1) -> List[str]:
    counts = seam_hit_counts(text)
    return [k for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if v >= min_hits]


def role_guess(module: str, rel_path: str, text: str, seam_counts: Dict[str, int]) -> str:
    hay = f"{module} {rel_path} {text[:4000]}".lower()
    roles = []
    if "config" in hay:
        roles.append("config")
    if any(w in hay for w in ["train", "training", "optimizer", "epoch"]):
        roles.append("training")
    if any(w in hay for w in ["evaluate", "eval", "ess", "free_energy", "mbar", "tfep"]):
        roles.append("evaluation")
    if any(w in hay for w in ["lennard", "water", "stillinger", "weber", "silicon", "system", "energy"]):
        roles.append("system_energy")
    if any(w in hay for w in ["flow", "coupling", "conditioner", "gnn", "particle"]):
        roles.append("flow_model")
    if any(w in hay for w in ["marginal", "auxiliary"]):
        roles.append("marginal_aux")
    if any(w in hay for w in ["plot", "figure", "matplotlib"]):
        roles.append("plotting")
    if seam_counts.get("locality", 0) >= 5:
        roles.append("locality")
    if seam_counts.get("openmm_energy", 0) >= 3:
        roles.append("openmm_bridge")
    return ";".join(dict.fromkeys(roles)) if roles else "unknown"


def maybe_heavy_module(role: str, text: str) -> bool:
    low = text.lower()
    if "training" in role:
        return True
    heavy_terms = ["for epoch", "wandb", "checkpoint", "train(", "optimizer", "submitit", "slurm"]
    return any(t in low for t in heavy_terms)


def recommended_candidate(role: str, imports: Sequence[str], seam_counts: Dict[str, int], maybe_heavy: bool) -> bool:
    if maybe_heavy:
        return False
    if "config" in role or "system_energy" in role or "marginal_aux" in role:
        return True
    if seam_counts.get("locality", 0) >= 8 and seam_counts.get("training", 0) < 5:
        return True
    if "openmm" in detect_import_groups(imports, {}):
        return True
    return False


def function_signature_guess(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts = []
    args = list(node.args.posonlyargs) + list(node.args.args)
    defaults = list(node.args.defaults)
    default_offset = len(args) - len(defaults)
    for i, arg in enumerate(args):
        s = arg.arg
        if arg.annotation:
            s += ": " + safe_unparse(arg.annotation, 60)
        if i >= default_offset:
            s += "=" + safe_unparse(defaults[i - default_offset], 60)
        parts.append(s)
    if node.args.vararg:
        parts.append("*" + node.args.vararg.arg)
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        s = arg.arg
        if arg.annotation:
            s += ": " + safe_unparse(arg.annotation, 60)
        if default is not None:
            s += "=" + safe_unparse(default, 60)
        parts.append(s)
    if node.args.kwarg:
        parts.append("**" + node.args.kwarg.arg)
    return f"{node.name}(" + ", ".join(parts) + ")"


def class_init_info(cls: ast.ClassDef) -> Tuple[int, int, bool, bool]:
    init_node: Optional[ast.FunctionDef] = None
    for body_node in cls.body:
        if isinstance(body_node, ast.FunctionDef) and body_node.name == "__init__":
            init_node = body_node
            break
    if init_node is None:
        # object default constructor.
        return 0, 0, True, True

    args = list(init_node.args.posonlyargs) + list(init_node.args.args)
    # remove self/cls if present
    if args and args[0].arg in {"self", "cls"}:
        args = args[1:]

    defaults = list(init_node.args.defaults)
    required_pos = max(0, len(args) - len(defaults))
    required_kw = sum(1 for d in init_node.args.kw_defaults if d is None)
    required = required_pos + required_kw

    total = len(args) + len(init_node.args.kwonlyargs)
    all_default_or_variadic = required == 0
    zero_arg_candidate = required == 0
    return total, required, all_default_or_variadic, zero_arg_candidate


# -----------------------------------------------------------------------------
# Static analysis
# -----------------------------------------------------------------------------

def analyze_module(py_path: Path, package_root: Path) -> Tuple[ModuleStaticRecord, List[SymbolRecord], List[ClassSignatureRecord]]:
    text = read_text_safe(py_path)
    relp = rel(py_path)
    module = module_name_from_path(py_path, package_root)
    line_count = len(text.splitlines())

    symbols: List[SymbolRecord] = []
    classes: List[ClassSignatureRecord] = []

    try:
        tree = ast.parse(text, filename=str(py_path))
    except Exception as exc:
        rec = ModuleStaticRecord(
            module=module,
            rel_path=relp,
            line_count=line_count,
            role_guess="parse_error",
            imports="",
            import_groups="",
            seam_scores=f"parse_error={repr(exc)}",
            total_seam_hits=0,
            classes="",
            functions="",
            top_level_assignments="",
            has_main_guard=False,
            maybe_heavy=False,
            recommended_probe05_candidate=False,
        )
        return rec, symbols, classes

    imports, aliases = import_aliases(tree)
    groups = detect_import_groups(imports, aliases)
    seam_counts = seam_hit_counts(text)
    total_hits = sum(seam_counts.values())
    role = role_guess(module, relp, text, seam_counts)

    class_names: List[str] = []
    func_names: List[str] = []
    assignments: List[str] = []
    has_main = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_names.append(node.name)
            tag_text = ast.get_source_segment(text, node) or node.name
            tags = seam_tags_for_text(tag_text)
            bases = [safe_unparse(b, 80) for b in node.bases]
            init_arg_count, req_arg_count, all_default, zero_arg = class_init_info(node)
            classes.append(ClassSignatureRecord(
                module=module,
                rel_path=relp,
                class_name=node.name,
                lineno=getattr(node, "lineno", 0),
                init_arg_count=init_arg_count,
                required_arg_count=req_arg_count,
                all_args_default_or_variadic=all_default,
                zero_arg_candidate=zero_arg,
                base_classes=";".join(bases),
                seam_tags=";".join(tags),
            ))
            symbols.append(SymbolRecord(
                module=module,
                rel_path=relp,
                kind="class",
                name=node.name,
                lineno=getattr(node, "lineno", 0),
                signature_guess=f"class {node.name}",
                seam_tags=";".join(tags),
                doc_first_line=first_doc_line(node),
            ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.append(node.name)
            tag_text = ast.get_source_segment(text, node) or node.name
            tags = seam_tags_for_text(tag_text)
            symbols.append(SymbolRecord(
                module=module,
                rel_path=relp,
                kind="function",
                name=node.name,
                lineno=getattr(node, "lineno", 0),
                signature_guess=function_signature_guess(node),
                seam_tags=";".join(tags),
                doc_first_line=first_doc_line(node),
            ))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_s = ""
            if isinstance(node, ast.Assign):
                target_s = ",".join(safe_unparse(t, 50) for t in node.targets)
            else:
                target_s = safe_unparse(node.target, 50)
            if target_s and not target_s.startswith("__"):
                assignments.append(target_s[:80])
        elif isinstance(node, ast.If):
            test_s = safe_unparse(node.test, 120)
            if "__name__" in test_s and "__main__" in test_s:
                has_main = True

    heavy = maybe_heavy_module(role, text)
    rec = ModuleStaticRecord(
        module=module,
        rel_path=relp,
        line_count=line_count,
        role_guess=role,
        imports=";".join(imports[:80]),
        import_groups=";".join(groups),
        seam_scores=";".join(f"{k}={v}" for k, v in sorted(seam_counts.items()) if v),
        total_seam_hits=total_hits,
        classes=";".join(class_names[:80]),
        functions=";".join(func_names[:80]),
        top_level_assignments=";".join(assignments[:120]),
        has_main_guard=has_main,
        maybe_heavy=heavy,
        recommended_probe05_candidate=recommended_candidate(role, imports, seam_counts, heavy),
    )
    return rec, symbols, classes


def discover_modules(package_root: Path) -> List[Path]:
    if not package_root.exists():
        return []
    out = []
    for p in sorted(package_root.rglob("*.py")):
        if "__pycache__" in p.parts or ".git" in p.parts:
            continue
        out.append(p)
    return out


# -----------------------------------------------------------------------------
# Dynamic import and instantiation smoke
# -----------------------------------------------------------------------------

def run_subprocess(
    cmd: Sequence[str],
    cwd: Path,
    timeout: int,
    env_extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    rec: Dict[str, Any] = {
        "cmd": list(cmd),
        "cwd": rel(cwd),
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": "",
        "timeout_s": timeout,
    }
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        rec.update(ok=(proc.returncode == 0), returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def write_logs(log_dir: Path, prefix: str, res: Dict[str, Any]) -> Dict[str, str]:
    ensure_dir(log_dir)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("_")[:120]
    h = sha16(str(res.get("cmd", "")) + str(res.get("cwd", "")))
    stdout = log_dir / f"{safe}_{h}.stdout.txt"
    stderr = log_dir / f"{safe}_{h}.stderr.txt"
    stdout.write_text(str(res.get("stdout", "")), encoding="utf-8")
    stderr.write_text(str(res.get("stderr", "")), encoding="utf-8")
    return {"stdout_log": rel(stdout), "stderr_log": rel(stderr)}


def import_module_smoke(module: str, official_repo: Path, timeout: int, log_dir: Path) -> Dict[str, Any]:
    code = (
        "import importlib, json, time\n"
        f"m={module!r}\n"
        "t=time.perf_counter()\n"
        "try:\n"
        "    mod=importlib.import_module(m)\n"
        "    dt=time.perf_counter()-t\n"
        "    print(json.dumps({'module':m,'ok':True,'seconds':dt,'file':str(getattr(mod,'__file__',''))}))\n"
        "except Exception as e:\n"
        "    dt=time.perf_counter()-t\n"
        "    print(json.dumps({'module':m,'ok':False,'seconds':dt,'error':repr(e)}))\n"
        "    raise\n"
    )
    py_path = str(official_repo)
    env_extra = {"PYTHONPATH": py_path + os.pathsep + os.environ.get("PYTHONPATH", "")}
    res = run_subprocess([sys.executable, "-c", code], cwd=REPO_ROOT, timeout=timeout, env_extra=env_extra)
    logs = write_logs(log_dir, f"import_{module}", res)
    parsed: Dict[str, Any] = {}
    try:
        # Last JSON-looking line.
        for line in reversed(res.get("stdout", "").splitlines()):
            if line.strip().startswith("{"):
                parsed = json.loads(line)
                break
    except Exception:
        parsed = {}

    return {
        "module": module,
        "ok": bool(res.get("ok")),
        "returncode": res.get("returncode"),
        "seconds": parsed.get("seconds", ""),
        "module_file": parsed.get("file", ""),
        "parsed_error": parsed.get("error", ""),
        "error": res.get("error", ""),
        "stderr_tail": str(res.get("stderr", ""))[-700:],
        **logs,
    }


def instantiate_zero_arg_smoke(class_rec: ClassSignatureRecord, official_repo: Path, timeout: int, log_dir: Path) -> Dict[str, Any]:
    module = class_rec.module
    cls_name = class_rec.class_name
    code = (
        "import importlib, json, time\n"
        f"m={module!r}; c={cls_name!r}\n"
        "t=time.perf_counter()\n"
        "try:\n"
        "    mod=importlib.import_module(m)\n"
        "    cls=getattr(mod,c)\n"
        "    obj=cls()\n"
        "    dt=time.perf_counter()-t\n"
        "    print(json.dumps({'module':m,'class':c,'ok':True,'seconds':dt,'repr':repr(obj)[:300]}))\n"
        "except Exception as e:\n"
        "    dt=time.perf_counter()-t\n"
        "    print(json.dumps({'module':m,'class':c,'ok':False,'seconds':dt,'error':repr(e)}))\n"
        "    raise\n"
    )
    env_extra = {"PYTHONPATH": str(official_repo) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    res = run_subprocess([sys.executable, "-c", code], cwd=REPO_ROOT, timeout=timeout, env_extra=env_extra)
    logs = write_logs(log_dir, f"instantiate_{module}_{cls_name}", res)
    parsed: Dict[str, Any] = {}
    try:
        for line in reversed(res.get("stdout", "").splitlines()):
            if line.strip().startswith("{"):
                parsed = json.loads(line)
                break
    except Exception:
        parsed = {}
    return {
        "module": module,
        "class_name": cls_name,
        "ok": bool(res.get("ok")),
        "returncode": res.get("returncode"),
        "seconds": parsed.get("seconds", ""),
        "repr": parsed.get("repr", ""),
        "parsed_error": parsed.get("error", ""),
        "error": res.get("error", ""),
        "stderr_tail": str(res.get("stderr", ""))[-700:],
        **logs,
    }


# -----------------------------------------------------------------------------
# Markdown outputs
# -----------------------------------------------------------------------------

def md_table(rows: List[List[str]], headers: List[str]) -> List[str]:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", " ") for x in row) + " |")
    return out


def write_seam_index(path: Path, static_rows: List[ModuleStaticRecord], symbol_rows: List[SymbolRecord]) -> None:
    by_seam: Dict[str, List[ModuleStaticRecord]] = {k: [] for k in SEAM_KEYWORDS}
    for r in static_rows:
        score_map = {}
        for part in r.seam_scores.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    score_map[k] = int(v)
                except Exception:
                    score_map[k] = 0
        for seam, v in score_map.items():
            if v > 0 and seam in by_seam:
                by_seam[seam].append(r)

    lines = [
        "# Probe 04 seam index",
        "",
        "This is the static code map of where paper-mechanism seams appear in `bgmat`.",
        "",
    ]

    for seam, rows in by_seam.items():
        rows = sorted(rows, key=lambda r: -dict((p.split("=")[0], int(p.split("=")[1])) for p in r.seam_scores.split(";") if "=" in p).get(seam, 0))
        lines += [f"## {seam}", ""]
        if not rows:
            lines += ["_No hits._", ""]
            continue
        table_rows = []
        for r in rows[:20]:
            score = ""
            for part in r.seam_scores.split(";"):
                if part.startswith(seam + "="):
                    score = part.split("=", 1)[1]
            table_rows.append([r.module, r.role_guess, score, r.import_groups, r.rel_path])
        lines += md_table(table_rows, ["module", "role", "hits", "imports", "file"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_entrypoints_md(path: Path, static_rows: List[ModuleStaticRecord], import_rows: List[Dict[str, Any]]) -> None:
    import_ok = {r["module"]: r.get("ok") for r in import_rows}
    lines = [
        "# Probe 04 — JAX/OpenMM/free-energy entrypoints",
        "",
        "These are likely code entrypoints for the implementation mechanisms.",
        "",
    ]

    for label, predicate in [
        ("JAX / tracing / Haiku / Distrax", lambda r: any(g in r.import_groups.split(";") for g in ["jax", "haiku", "distrax", "tfp"]) or "jax_tracing=" in r.seam_scores),
        ("OpenMM energy bridge", lambda r: "openmm" in r.import_groups.split(";") or "openmm_energy=" in r.seam_scores),
        ("Locality / neighbor / cutoff", lambda r: "locality=" in r.seam_scores),
        ("Marginal / auxiliary sampling", lambda r: "marginal=" in r.seam_scores or "augmentation_flow=" in r.seam_scores),
        ("Free energy / ESS / reweighting", lambda r: "free_energy=" in r.seam_scores),
    ]:
        lines += [f"## {label}", ""]
        selected = [r for r in static_rows if predicate(r)]
        selected = sorted(selected, key=lambda r: (-r.total_seam_hits, r.module))
        if not selected:
            lines += ["_No static hits._", ""]
            continue
        table = []
        for r in selected[:30]:
            table.append([
                r.module,
                "YES" if import_ok.get(r.module) else ("NO" if r.module in import_ok else "not run"),
                r.role_guess,
                r.seam_scores[:160],
                r.rel_path,
            ])
        lines += md_table(table, ["module", "imports?", "role", "seam scores", "file"])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_candidate_tiny_paths_md(
    path: Path,
    static_rows: List[ModuleStaticRecord],
    class_rows: List[ClassSignatureRecord],
    import_rows: List[Dict[str, Any]],
) -> None:
    ok_modules = {r["module"] for r in import_rows if r.get("ok")}
    candidates = [r for r in static_rows if r.recommended_probe05_candidate]
    candidates = sorted(candidates, key=lambda r: (r.maybe_heavy, -r.total_seam_hits, r.module))

    zero_arg = [c for c in class_rows if c.zero_arg_candidate]
    zero_arg_by_module: Dict[str, List[str]] = {}
    for c in zero_arg:
        zero_arg_by_module.setdefault(c.module, []).append(c.class_name)

    lines = [
        "# Probe 04 — candidate tiny paths for Probe 05",
        "",
        "These are not executed training runs. They are likely small targets for the next probe.",
        "",
        "## Best candidate modules",
        "",
    ]

    if not candidates:
        lines += ["_No candidates selected._", ""]
    else:
        table = []
        for r in candidates[:40]:
            table.append([
                r.module,
                "YES" if r.module in ok_modules else ("NO" if ok_modules else "not run"),
                r.role_guess,
                r.import_groups,
                ";".join(zero_arg_by_module.get(r.module, [])[:8]),
                r.rel_path,
            ])
        lines += md_table(table, ["module", "import ok", "role", "imports", "zero-arg classes", "file"])
        lines.append("")

    lines += [
        "## Suggested Probe 05 strategy",
        "",
        "1. Start with imported config/system modules, not training scripts.",
        "2. Instantiate only explicit small systems or zero-arg/simple constructors.",
        "3. Locate the energy function and neighbor/cutoff construction path.",
        "4. Build a tiny two-size locality stress test before touching full training.",
        "",
        "Candidate command style:",
        "",
        "```powershell",
        "python .\\current_probe_05_tiny_system_smoke.py",
        "```",
        "",
        "Probe 05 should be fixed and explicit, based on this file's selected modules.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_md(
    path: Path,
    out_dir: Path,
    official_repo: Path,
    package_root: Path,
    static_rows: List[ModuleStaticRecord],
    symbol_rows: List[SymbolRecord],
    class_rows: List[ClassSignatureRecord],
    import_rows: List[Dict[str, Any]],
    inst_rows: List[Dict[str, Any]],
) -> None:
    ok_imports = sum(1 for r in import_rows if r.get("ok"))
    fail_imports = len(import_rows) - ok_imports
    modules_by_group: Dict[str, int] = {}
    for r in static_rows:
        for g in r.import_groups.split(";"):
            if g:
                modules_by_group[g] = modules_by_group.get(g, 0) + 1

    top_seams: Dict[str, int] = {k: 0 for k in SEAM_KEYWORDS}
    for r in static_rows:
        for part in r.seam_scores.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    top_seams[k] = top_seams.get(k, 0) + int(v)
                except Exception:
                    pass

    candidates = [r for r in static_rows if r.recommended_probe05_candidate]

    lines = [
        "# PROBE 04 SUMMARY",
        "",
        f"Output folder: `{rel(out_dir)}`",
        f"Official repo: `{rel(official_repo)}`",
        f"Package root: `{rel(package_root)}`",
        "",
        "## What this probe did",
        "",
        "- Statically scanned every `bgmat` Python module.",
        "- Mapped imports, symbols, classes, seam keywords, and likely code roles.",
        "- Optionally imported each module in a subprocess using `PYTHONPATH`.",
        "- Optionally instantiated zero-required-argument classes.",
        "- Wrote targeted seam maps for JAX/OpenMM/locality/marginal/free-energy mechanisms.",
        "",
        "## Headline counts",
        "",
        f"- Python modules scanned: **{len(static_rows)}**",
        f"- Symbols cataloged: **{len(symbol_rows)}**",
        f"- Classes cataloged: **{len(class_rows)}**",
        f"- Probe 05 candidate modules: **{len(candidates)}**",
        f"- Module import tests run: **{len(import_rows)}**",
        f"- Module imports OK: **{ok_imports}**",
        f"- Module imports failed: **{fail_imports}**",
        f"- Instantiation tests run: **{len(inst_rows)}**",
        f"- Instantiations OK: **{sum(1 for r in inst_rows if r.get('ok'))}**",
        "",
        "## Import groups",
        "",
    ]
    for g, n in sorted(modules_by_group.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{g}`: {n} module(s)")
    lines += ["", "## Seam hit totals", ""]
    for s, n in sorted(top_seams.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{s}`: {n}")
    lines += [
        "",
        "## Most important next files",
        "",
    ]
    for r in sorted(candidates, key=lambda r: (-r.total_seam_hits, r.module))[:15]:
        lines.append(f"- `{r.module}` — role `{r.role_guess}`, imports `{r.import_groups}`, file `{r.rel_path}`")
    lines += [
        "",
        "## Generated files",
        "",
        "- `module_static_seams.csv/json`",
        "- `module_import_results.csv/json`",
        "- `symbol_catalog.csv/json`",
        "- `class_signature_catalog.csv/json`",
        "- `instantiation_results.csv`",
        "- `seam_index.md`",
        "- `jax_openmm_entrypoints.md`",
        "- `candidate_tiny_paths.md`",
        "- `logs/*.stdout.txt` and `logs/*.stderr.txt`",
        "",
        "## Recommended next move",
        "",
        "Open `candidate_tiny_paths.md` and `jax_openmm_entrypoints.md`.",
        "Probe 05 should pick one small config/system path and instantiate an energy/locality object without training.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Current paper Probe 04: module/import/seam map.")
    p.add_argument("--official-repo", default="../bgmat", help="Official repo folder relative to repo root unless absolute.")
    p.add_argument("--out-root", default="probe_runs", help="Output root relative to repo root unless absolute.")
    p.add_argument("--import-modules", action="store_true", help="Import every discovered bgmat module in a subprocess.")
    p.add_argument("--import-timeout", type=int, default=45, help="Timeout seconds per module import.")
    p.add_argument("--instantiate-zero-arg", action="store_true", help="Attempt zero-required-argument class constructors in subprocesses.")
    p.add_argument("--instantiate-timeout", type=int, default=45, help="Timeout seconds per class instantiation.")
    p.add_argument("--max-instantiations", type=int, default=80, help="Safety cap for instantiation attempts.")
    p.add_argument("--max-modules", type=int, default=0, help="Optional cap for scanned modules; 0 means all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    official_repo = resolve_from_root(args.official_repo)
    package_root = official_repo / "bgmat"
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe04_module_seams_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")

    print("=" * 100)
    print("CURRENT PAPER PROBE 04 — MODULE IMPORT + MECHANISM SEAM MAP")
    print("=" * 100)
    print("[ROOT]          .")
    print(f"[OFFICIAL REPO] {rel(official_repo)}")
    print(f"[PACKAGE ROOT]  {rel(package_root)}")
    print(f"[OUT]           {rel(out_dir)}")

    if not package_root.exists():
        print(f"[ERROR] Missing package root: {rel(package_root)}", file=sys.stderr)
        sys.exit(2)

    module_paths = discover_modules(package_root)
    if args.max_modules and args.max_modules > 0:
        module_paths = module_paths[: args.max_modules]

    print(f"[SCAN] modules={len(module_paths)}")

    static_records: List[ModuleStaticRecord] = []
    symbols: List[SymbolRecord] = []
    classes: List[ClassSignatureRecord] = []

    for p in module_paths:
        rec, sym, cls = analyze_module(p, package_root)
        static_records.append(rec)
        symbols.extend(sym)
        classes.extend(cls)

    import_rows: List[Dict[str, Any]] = []
    if args.import_modules:
        print("[IMPORT] importing modules in subprocesses")
        for i, rec in enumerate(static_records, start=1):
            print(f"  [{i:03d}/{len(static_records):03d}] {rec.module}")
            import_rows.append(import_module_smoke(rec.module, official_repo, timeout=args.import_timeout, log_dir=log_dir))

    inst_rows: List[Dict[str, Any]] = []
    if args.instantiate_zero_arg:
        zero_arg = [c for c in classes if c.zero_arg_candidate]
        zero_arg = zero_arg[: max(1, args.max_instantiations)]
        print(f"[INSTANTIATE] zero-arg candidates={len(zero_arg)}")
        for i, c in enumerate(zero_arg, start=1):
            print(f"  [{i:03d}/{len(zero_arg):03d}] {c.module}.{c.class_name}")
            inst_rows.append(instantiate_zero_arg_smoke(c, official_repo, timeout=args.instantiate_timeout, log_dir=log_dir))

    static_dicts = [dataclasses.asdict(r) for r in static_records]
    symbol_dicts = [dataclasses.asdict(r) for r in symbols]
    class_dicts = [dataclasses.asdict(r) for r in classes]

    write_csv(out_dir / "module_static_seams.csv", static_dicts)
    write_json(out_dir / "module_static_seams.json", static_dicts)
    write_csv(out_dir / "symbol_catalog.csv", symbol_dicts)
    write_json(out_dir / "symbol_catalog.json", symbol_dicts)
    write_csv(out_dir / "class_signature_catalog.csv", class_dicts)
    write_json(out_dir / "class_signature_catalog.json", class_dicts)
    write_csv(out_dir / "module_import_results.csv", import_rows)
    write_json(out_dir / "module_import_results.json", import_rows)
    write_csv(out_dir / "instantiation_results.csv", inst_rows)
    write_json(out_dir / "instantiation_results.json", inst_rows)

    write_seam_index(out_dir / "seam_index.md", static_records, symbols)
    write_entrypoints_md(out_dir / "jax_openmm_entrypoints.md", static_records, import_rows)
    write_candidate_tiny_paths_md(out_dir / "candidate_tiny_paths.md", static_records, classes, import_rows)
    write_summary_md(
        out_dir / "PROBE04_SUMMARY.md",
        out_dir=out_dir,
        official_repo=official_repo,
        package_root=package_root,
        static_rows=static_records,
        symbol_rows=symbols,
        class_rows=classes,
        import_rows=import_rows,
        inst_rows=inst_rows,
    )

    ok_imports = sum(1 for r in import_rows if r.get("ok"))
    fail_imports = len(import_rows) - ok_imports
    candidates = [r for r in static_records if r.recommended_probe05_candidate]

    print("\n[SUMMARY]")
    print(f"  modules_scanned       : {len(static_records)}")
    print(f"  symbols_cataloged     : {len(symbols)}")
    print(f"  classes_cataloged     : {len(classes)}")
    print(f"  probe05_candidates    : {len(candidates)}")
    print(f"  import_tests_run      : {len(import_rows)}")
    print(f"  import_ok             : {ok_imports}")
    print(f"  import_failed         : {fail_imports}")
    print(f"  instantiation_tests   : {len(inst_rows)}")
    print(f"  output                : {rel(out_dir)}")
    print("\nNext:")
    print(f"  open {rel(out_dir / 'PROBE04_SUMMARY.md')}")
    print(f"  open {rel(out_dir / 'candidate_tiny_paths.md')}")
    print(f"  open {rel(out_dir / 'jax_openmm_entrypoints.md')}")


if __name__ == "__main__":
    main()
