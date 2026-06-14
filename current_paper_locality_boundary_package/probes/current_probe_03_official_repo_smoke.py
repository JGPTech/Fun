#!/usr/bin/env python3
r"""
current_probe_03_official_repo_smoke.py

Current paper Probe 03
======================

Purpose
-------
Portable official-repo smoke planner / runner for the current paper sandbox.

This probe is intentionally careful:

- No hardcoded local paths.
- Defaults are relative to this repo root.
- Generated outputs go under probe_runs/.
- It does NOT train by default.
- It does NOT install dependencies by default.
- It does NOT execute README commands by default.

It audits the official code checkout, identifies runnable entry points, checks
dependency/import status, and can optionally run safe `--help` smoke tests or a
single explicit user-supplied command.

Current paper:
  "Scalable Boltzmann generators for equilibrium sampling of large-scale materials"
  Schebek, Noé, Rogal, Nature Communications (2026) 17:5010
  DOI: 10.1038/s41467-026-73900-9

Expected repo layout
--------------------
Your repo root should contain this script and, optionally, an official-code
checkout folder such as:

  ./bgmat/

Portable usage
--------------
From the repo root:

  python .\current_probe_03_official_repo_smoke.py

If the official repo folder has a different name:

  python .\current_probe_03_official_repo_smoke.py --official-repo path/to/bgmat

Import/dependency smoke:

  python .\current_probe_03_official_repo_smoke.py --import-smoke

Optional editable install of official repo:

  python .\current_probe_03_official_repo_smoke.py --editable-install --import-smoke

Optional safe help smoke for discovered Python scripts:

  python .\current_probe_03_official_repo_smoke.py --run-help-smoke

Optional explicit command, run inside the official repo:

  python .\current_probe_03_official_repo_smoke.py --run-command "python some_script.py --help"

Outputs
-------
  probe_runs/current_probe03_official_smoke_<timestamp>/
    PROBE03_SUMMARY.md
    official_repo_audit.json
    official_repo_audit.md
    dependency_report.json
    candidate_commands.csv
    candidate_scripts.csv
    command_results.csv
    logs/*.stdout.txt
    logs/*.stderr.txt

Probe contract
--------------
This probe establishes the official-code execution surface. It does not validate
paper results yet. The next probe should run the smallest actual official example
selected from Probe 03's candidates and compare its outputs to Probe 02 source
data where possible.
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import datetime as _dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import textwrap
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_rel(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:
        try:
            return str(path.relative_to(base)).replace("\\", "/")
        except Exception:
            return str(path)


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
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def run_subprocess(
    cmd: str | Sequence[str],
    cwd: Optional[Path],
    timeout: int,
    shell: bool = False,
    env_extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    rec: Dict[str, Any] = {
        "cmd": cmd if isinstance(cmd, str) else list(cmd),
        "cwd": repo_rel(cwd) if cwd else "",
        "timeout_s": timeout,
        "shell": shell,
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": "",
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            shell=shell,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
        rec.update(
            ok=(proc.returncode == 0),
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def write_command_logs(log_dir: Path, prefix: str, result: Dict[str, Any]) -> Dict[str, str]:
    ensure_dir(log_dir)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("_")[:90]
    h = sha256_text(str(result.get("cmd", "")) + str(result.get("cwd", "")))
    stdout_path = log_dir / f"{safe}_{h}.stdout.txt"
    stderr_path = log_dir / f"{safe}_{h}.stderr.txt"
    stdout_path.write_text(str(result.get("stdout", "")), encoding="utf-8")
    stderr_path.write_text(str(result.get("stderr", "")), encoding="utf-8")
    return {"stdout_log": repo_rel(stdout_path), "stderr_log": repo_rel(stderr_path)}


def read_text_safe(path: Path, max_bytes: int = 1_500_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def iter_text_files(root: Path, max_files: int = 10000) -> Iterable[Path]:
    allowed = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".sh", ".ps1"}
    n = 0
    for path in root.rglob("*"):
        if n >= max_files:
            return
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in allowed:
            continue
        n += 1
        yield path


# -----------------------------------------------------------------------------
# Dependency / package parsing
# -----------------------------------------------------------------------------

CORE_IMPORTS = [
    "jax",
    "jaxlib",
    "haiku",
    "distrax",
    "openmm",
    "numpy",
    "scipy",
    "matplotlib",
    "ase",
]


def module_import_status(module_name: str) -> Dict[str, Any]:
    rec = {"module": module_name, "importable": False, "version": "", "origin": "", "error": ""}
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            rec["error"] = "not found"
            return rec
        rec["origin"] = str(spec.origin)
        mod = __import__(module_name)
        rec["importable"] = True
        rec["version"] = str(getattr(mod, "__version__", ""))
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def parse_pyproject(repo: Path) -> Dict[str, Any]:
    path = repo / "pyproject.toml"
    out: Dict[str, Any] = {"exists": path.exists(), "path": repo_rel(path), "project_name": "", "dependencies": [], "optional_dependencies": {}}
    if not path.exists():
        return out

    try:
        import tomllib  # Python 3.11+
        data = tomllib.loads(read_text_safe(path))
        project = data.get("project", {}) if isinstance(data, dict) else {}
        out["project_name"] = project.get("name", "")
        out["dependencies"] = project.get("dependencies", []) or []
        out["optional_dependencies"] = project.get("optional-dependencies", {}) or {}
        # Some projects use tool-specific sections.
        if not out["dependencies"]:
            poetry = (((data.get("tool") or {}).get("poetry") or {}).get("dependencies") or {})
            if poetry:
                out["project_name"] = out["project_name"] or ((data.get("tool") or {}).get("poetry") or {}).get("name", "")
                out["dependencies"] = [str(k) for k in poetry.keys() if str(k).lower() != "python"]
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def parse_requirements(repo: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in ["requirements.txt", "requirements-dev.txt", "environment.yml", "environment.yaml"]:
        path = repo / name
        if not path.exists():
            continue
        txt = read_text_safe(path)
        for i, line in enumerate(txt.splitlines(), start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append({"file": repo_rel(path), "line": i, "requirement": s})
    return rows


def infer_imports_from_python(repo: Path) -> Dict[str, Any]:
    imports: Dict[str, int] = {}
    files_scanned = 0
    parse_errors: List[Dict[str, Any]] = []

    for path in repo.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        files_scanned += 1
        txt = read_text_safe(path)
        try:
            tree = ast.parse(txt, filename=str(path))
        except Exception as exc:
            parse_errors.append({"file": repo_rel(path, repo), "error": repr(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    imports[top] = imports.get(top, 0) + 1
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top = node.module.split(".")[0]
                    imports[top] = imports.get(top, 0) + 1

    top_imports = [{"module": k, "count": v} for k, v in sorted(imports.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"files_scanned": files_scanned, "top_imports": top_imports[:200], "parse_errors": parse_errors[:50]}


# -----------------------------------------------------------------------------
# Candidate discovery
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class CandidateCommand:
    source: str
    command: str
    cwd: str
    reason: str
    risk: str
    recommended_first: bool = False

    def asdict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class CandidateScript:
    path: str
    has_main_guard: bool
    uses_argparse: bool
    imports_jax: bool
    imports_openmm: bool
    line_count: int
    role_guess: str
    help_command: str
    recommended_help_smoke: bool

    def asdict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def extract_markdown_commands(repo: Path) -> List[CandidateCommand]:
    out: List[CandidateCommand] = []
    md_files = [p for p in repo.rglob("*.md") if ".git" not in p.parts]
    command_re = re.compile(r"^\s*((?:python|python3|py|pip|pytest|jupyter|ipython|conda|mamba|uv)\b.*)$", re.IGNORECASE)

    for path in md_files:
        txt = read_text_safe(path)
        in_fence = False
        for i, line in enumerate(txt.splitlines(), start=1):
            s = line.strip()
            if s.startswith("```"):
                in_fence = not in_fence
                continue
            m = command_re.match(line)
            if not m:
                continue
            cmd = m.group(1).strip()
            low = cmd.lower()
            risk = "low"
            reason = f"README/markdown command at {repo_rel(path, repo)}:{i}"
            recommended = False

            if "install" in low or low.startswith(("pip ", "conda ", "mamba ", "uv ")):
                risk = "install"
            elif any(word in low for word in ["train", "training", "--epochs", "submit", "slurm"]):
                risk = "heavy"
            elif "--help" in low or "-h" in low or "pytest" in low:
                risk = "low"
                recommended = True
            elif "tutorial" in low or "example" in low or "demo" in low:
                risk = "medium"
                recommended = True

            out.append(CandidateCommand(
                source=repo_rel(path, repo),
                command=cmd,
                cwd="official_repo_root",
                reason=reason,
                risk=risk,
                recommended_first=recommended,
            ))
    return out


def guess_script_role(path: Path, text: str) -> str:
    hay = (str(path).lower() + "\n" + text[:4000].lower())
    roles = []
    for key, terms in [
        ("train", ["train", "training", "trainer", "optimizer", "epochs"]),
        ("evaluate", ["eval", "evaluate", "ess", "free energy", "mbar", "tfep"]),
        ("sample", ["sample", "generate", "generator"]),
        ("system", ["lennard", "lj", "stillinger", "weber", "silicon", "mw", "ice", "fcc", "hcp", "bcc"]),
        ("tutorial", ["tutorial", "example", "demo"]),
        ("plot", ["plot", "figure", "matplotlib"]),
        ("utility", ["utils", "helper", "config"]),
    ]:
        if any(t in hay for t in terms):
            roles.append(key)
    return ";".join(roles) if roles else "unknown"


def discover_python_scripts(repo: Path) -> List[CandidateScript]:
    rows: List[CandidateScript] = []
    for path in repo.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        txt = read_text_safe(path)
        relp = repo_rel(path, repo)
        has_main = "__main__" in txt
        uses_argparse = "argparse" in txt or "click." in txt or "typer." in txt
        imports_jax = re.search(r"^\s*(import|from)\s+jax\b", txt, re.MULTILINE) is not None
        imports_openmm = re.search(r"^\s*(import|from)\s+openmm\b", txt, re.MULTILINE) is not None
        line_count = len(txt.splitlines())
        role = guess_script_role(path, txt)
        recommended = has_main and uses_argparse and not any(h in role for h in ["train"]) and line_count < 2000
        help_command = f"{shlex.quote(sys.executable)} {shlex.quote(relp)} --help"
        rows.append(CandidateScript(
            path=relp,
            has_main_guard=has_main,
            uses_argparse=uses_argparse,
            imports_jax=imports_jax,
            imports_openmm=imports_openmm,
            line_count=line_count,
            role_guess=role,
            help_command=help_command,
            recommended_help_smoke=recommended,
        ))
    rows.sort(key=lambda r: (not r.recommended_help_smoke, r.path))
    return rows


def build_probe_commands(repo: Path, scripts: List[CandidateScript], md_cmds: List[CandidateCommand]) -> List[CandidateCommand]:
    out = list(md_cmds)

    # Always include a pip metadata/help surface; it is a low-risk command.
    out.append(CandidateCommand(
        source="probe03",
        command=f"{shlex.quote(sys.executable)} -m pip show bgmat",
        cwd="repo_root",
        reason="Check whether official package is already installed.",
        risk="low",
        recommended_first=True,
    ))

    # Help commands for recommended scripts.
    for s in scripts:
        if s.recommended_help_smoke:
            out.append(CandidateCommand(
                source=s.path,
                command=s.help_command,
                cwd="official_repo_root",
                reason="Discovered Python script with main guard/argparse; --help should be non-heavy.",
                risk="low",
                recommended_first=True,
            ))

    # De-duplicate by command.
    seen = set()
    unique: List[CandidateCommand] = []
    for c in out:
        key = (c.command, c.cwd)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


# -----------------------------------------------------------------------------
# Repo audit
# -----------------------------------------------------------------------------

def audit_official_repo(repo: Path) -> Dict[str, Any]:
    audit: Dict[str, Any] = {
        "official_repo_arg": "",
        "exists": repo.exists(),
        "path_repo_relative": repo_rel(repo),
        "status": "missing",
    }
    if not repo.exists():
        return audit
    if not repo.is_dir():
        audit["status"] = "not_a_directory"
        return audit

    audit["status"] = "found"
    audit["is_git_repo"] = (repo / ".git").exists()
    audit["top_level_files"] = sorted([p.name for p in repo.iterdir() if p.is_file()])
    audit["top_level_dirs"] = sorted([p.name for p in repo.iterdir() if p.is_dir() and p.name != ".git"])

    stats = {"n_files": 0, "n_py": 0, "n_md": 0, "n_notebooks": 0, "total_bytes": 0}
    for p in repo.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        stats["n_files"] += 1
        stats["total_bytes"] += p.stat().st_size if p.exists() else 0
        if p.suffix.lower() == ".py":
            stats["n_py"] += 1
        elif p.suffix.lower() == ".md":
            stats["n_md"] += 1
        elif p.suffix.lower() == ".ipynb":
            stats["n_notebooks"] += 1
    audit["stats"] = stats

    expected = ["experiments", "models", "systems", "tutorial", "utils"]
    audit["expected_paths"] = {name: (repo / name).exists() for name in expected}

    pyproject = parse_pyproject(repo)
    audit["pyproject"] = pyproject
    audit["requirements"] = parse_requirements(repo)
    audit["inferred_imports"] = infer_imports_from_python(repo)

    if audit["is_git_repo"]:
        audit["git"] = {
            "head": run_subprocess(["git", "rev-parse", "HEAD"], cwd=repo, timeout=20),
            "branch": run_subprocess(["git", "branch", "--show-current"], cwd=repo, timeout=20),
            "status_short": run_subprocess(["git", "status", "--short"], cwd=repo, timeout=20),
            "remote_v": run_subprocess(["git", "remote", "-v"], cwd=repo, timeout=20),
        }

    return audit


def write_audit_md(path: Path, audit: Dict[str, Any], candidates: List[CandidateCommand], scripts: List[CandidateScript]) -> None:
    lines = [
        "# Current Probe 03 — official repo smoke audit",
        "",
        f"Official repo path: `{audit.get('path_repo_relative')}`",
        f"Status: **{audit.get('status')}**",
        "",
    ]

    if audit.get("status") != "found":
        lines += [
            "The official repo folder was not found. Clone it beside this script or pass `--official-repo`.",
            "",
            "Portable example:",
            "",
            "```powershell",
            "git clone https://github.com/maxschebek/bgmat bgmat",
            "python .\\current_probe_03_official_repo_smoke.py --official-repo bgmat",
            "```",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "## Top-level structure",
        "",
        "### Directories",
        "",
    ]
    for d in audit.get("top_level_dirs", []):
        lines.append(f"- `{d}/`")
    lines += ["", "### Files", ""]
    for f in audit.get("top_level_files", []):
        lines.append(f"- `{f}`")
    lines.append("")

    lines += [
        "## Expected paper-code directories",
        "",
    ]
    for k, v in audit.get("expected_paths", {}).items():
        lines.append(f"- `{k}`: {'YES' if v else 'NO'}")
    lines.append("")

    pyproject = audit.get("pyproject", {})
    lines += [
        "## Package metadata",
        "",
        f"- pyproject exists: `{pyproject.get('exists')}`",
        f"- project name: `{pyproject.get('project_name', '')}`",
        f"- dependency count: `{len(pyproject.get('dependencies', []) or [])}`",
        "",
    ]

    deps = pyproject.get("dependencies", []) or []
    if deps:
        lines += ["### Dependencies from pyproject", ""]
        for d in deps:
            lines.append(f"- `{d}`")
        lines.append("")

    lines += [
        "## Candidate commands",
        "",
        "These are discovered, not automatically trusted. Heavy commands should not be run casually.",
        "",
    ]
    for i, c in enumerate(candidates[:80]):
        tag = "⭐ " if c.recommended_first else ""
        lines.append(f"{i}. {tag}`{c.command}` — risk: `{c.risk}`, cwd: `{c.cwd}`, source: `{c.source}`")
    if len(candidates) > 80:
        lines.append(f"- ... {len(candidates)-80} more")
    lines.append("")

    lines += [
        "## Candidate Python scripts",
        "",
    ]
    for s in scripts[:120]:
        rec = "⭐ " if s.recommended_help_smoke else ""
        lines.append(
            f"- {rec}`{s.path}` role=`{s.role_guess}` main={s.has_main_guard} argparse={s.uses_argparse} lines={s.line_count}"
        )
    if len(scripts) > 120:
        lines.append(f"- ... {len(scripts)-120} more")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Execution modes
# -----------------------------------------------------------------------------

def editable_install(repo: Path, log_dir: Path, timeout: int) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    res = run_subprocess(cmd, cwd=repo, timeout=timeout, shell=False)
    res.update(write_command_logs(log_dir, "editable_install", res))
    return res


def import_smoke(log_dir: Path, official_repo: Optional[Path], extra_imports: Sequence[str]) -> List[Dict[str, Any]]:
    modules = list(dict.fromkeys(CORE_IMPORTS + list(extra_imports)))
    rows: List[Dict[str, Any]] = []

    # First use find_spec in current interpreter.
    for mod in modules:
        rows.append(module_import_status(mod))

    # Then run a subprocess with PYTHONPATH including official repo, so local packages can be imported.
    if official_repo and official_repo.exists():
        code = (
            "import importlib, json; mods="
            + repr(modules)
            + "; out=[]\n"
              "for m in mods:\n"
              "    r={'module':m,'subprocess_import_ok':False,'version':'','error':''}\n"
              "    try:\n"
              "        mod=importlib.import_module(m); r['subprocess_import_ok']=True; r['version']=str(getattr(mod,'__version__',''))\n"
              "    except Exception as e:\n"
              "        r['error']=repr(e)\n"
              "    out.append(r)\n"
              "print(json.dumps(out))"
        )
        env_extra = {"PYTHONPATH": str(official_repo) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        res = run_subprocess([sys.executable, "-c", code], cwd=official_repo, timeout=60, env_extra=env_extra)
        res.update(write_command_logs(log_dir, "import_smoke_subprocess", res))
        try:
            subrows = json.loads(res.get("stdout", "[]"))
        except Exception:
            subrows = []
        sub_by_mod = {r.get("module"): r for r in subrows if isinstance(r, dict)}
        for r in rows:
            sub = sub_by_mod.get(r["module"], {})
            r["subprocess_import_ok_with_repo_pythonpath"] = sub.get("subprocess_import_ok", "")
            r["subprocess_version"] = sub.get("version", "")
            r["subprocess_error"] = sub.get("error", "")

    return rows


def run_help_smoke(repo: Path, candidates: List[CandidateCommand], log_dir: Path, max_commands: int, timeout: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected = [c for c in candidates if c.risk == "low" and c.recommended_first]
    # Prefer --help Python script commands, skip pip show because it is not official script help.
    selected = [c for c in selected if "--help" in c.command or " -h" in c.command][:max_commands]
    for i, c in enumerate(selected):
        res = run_subprocess(c.command, cwd=repo, timeout=timeout, shell=True)
        logs = write_command_logs(log_dir, f"help_smoke_{i:03d}", res)
        rows.append({
            "kind": "help_smoke",
            "index": i,
            "source": c.source,
            "command": c.command,
            "cwd": "official_repo_root",
            "risk": c.risk,
            "ok": res.get("ok"),
            "returncode": res.get("returncode"),
            "error": res.get("error"),
            **logs,
        })
    return rows


def run_explicit_command(repo: Path, command: str, log_dir: Path, timeout: int) -> Dict[str, Any]:
    res = run_subprocess(command, cwd=repo, timeout=timeout, shell=True)
    logs = write_command_logs(log_dir, "explicit_command", res)
    return {
        "kind": "explicit_command",
        "index": 0,
        "source": "user_supplied",
        "command": command,
        "cwd": "official_repo_root",
        "risk": "user_explicit",
        "ok": res.get("ok"),
        "returncode": res.get("returncode"),
        "error": res.get("error"),
        **logs,
    }


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def write_summary_md(
    path: Path,
    out_dir: Path,
    official_repo: Path,
    audit: Dict[str, Any],
    deps: List[Dict[str, Any]],
    candidates: List[CandidateCommand],
    scripts: List[CandidateScript],
    command_results: List[Dict[str, Any]],
) -> None:
    deps_missing = [d["module"] for d in deps if not d.get("importable")]
    help_ok = sum(1 for r in command_results if r.get("kind") == "help_smoke" and r.get("ok"))
    help_total = sum(1 for r in command_results if r.get("kind") == "help_smoke")
    explicit = [r for r in command_results if r.get("kind") == "explicit_command"]

    lines = [
        "# PROBE 03 SUMMARY",
        "",
        f"Output folder: `{repo_rel(out_dir)}`",
        f"Official repo: `{repo_rel(official_repo)}`",
        "",
        "## What this probe did",
        "",
        "- Audited the official code checkout.",
        "- Parsed dependency/package metadata when available.",
        "- Scanned README/Markdown and Python scripts for runnable candidates.",
        "- Checked import status when requested.",
        "- Optionally ran safe help-smoke commands or one explicit user command.",
        "",
        "## Headline status",
        "",
        f"- Official repo status: **{audit.get('status')}**",
        f"- Candidate commands found: **{len(candidates)}**",
        f"- Python scripts found: **{len(scripts)}**",
        f"- Recommended help-smoke scripts: **{sum(1 for s in scripts if s.recommended_help_smoke)}**",
        f"- Missing core imports: **{', '.join(deps_missing) if deps_missing else 'none'}**",
        "",
    ]

    if help_total:
        lines += [
            "## Help-smoke results",
            "",
            f"- OK: **{help_ok}/{help_total}**",
            "",
        ]

    if explicit:
        r = explicit[0]
        lines += [
            "## Explicit command result",
            "",
            f"- Command: `{r.get('command')}`",
            f"- OK: `{r.get('ok')}`",
            f"- Return code: `{r.get('returncode')}`",
            f"- Stdout log: `{r.get('stdout_log')}`",
            f"- Stderr log: `{r.get('stderr_log')}`",
            "",
        ]

    lines += [
        "## Interpretation",
        "",
        "Probe 03 is an execution-surface probe. A green Probe 03 means we know how to call the official code safely.",
        "It does not mean the paper has been reproduced yet.",
        "",
        "## Recommended next move",
        "",
        "Pick the smallest non-heavy official candidate from `candidate_commands.csv` or `official_repo_audit.md`.",
        "Run it explicitly with `--run-command` if it looks safe, then turn that into Probe 04 as a fixed reproducible path.",
        "",
        "Example:",
        "",
        "```powershell",
        "python .\\current_probe_03_official_repo_smoke.py --run-command \"python path/from/bgmat.py --help\"",
        "```",
        "",
        "## Generated files",
        "",
        "- `official_repo_audit.json`",
        "- `official_repo_audit.md`",
        "- `dependency_report.json`",
        "- `candidate_commands.csv`",
        "- `candidate_scripts.csv`",
        "- `command_results.csv`",
        "- `logs/*.stdout.txt`",
        "- `logs/*.stderr.txt`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Args / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Current paper Probe 03: official repo smoke planner/runner.")
    p.add_argument("--official-repo", default="../bgmat", help="Official repo folder, relative to this repo root unless absolute.")
    p.add_argument("--out-root", default="probe_runs", help="Output root, relative to this repo root unless absolute.")
    p.add_argument("--import-smoke", action="store_true", help="Check core dependency import status.")
    p.add_argument("--extra-import", action="append", default=[], help="Additional module to check. Repeatable.")
    p.add_argument("--editable-install", action="store_true", help="Run `python -m pip install -e .` inside official repo before import smoke.")
    p.add_argument("--install-timeout", type=int, default=600, help="Timeout seconds for editable install.")
    p.add_argument("--run-help-smoke", action="store_true", help="Run safe discovered `--help` commands only.")
    p.add_argument("--max-help-commands", type=int, default=20, help="Max help-smoke commands to run.")
    p.add_argument("--help-timeout", type=int, default=30, help="Timeout seconds per help-smoke command.")
    p.add_argument("--run-command", default="", help="Explicit command to run inside official repo. Not run unless provided.")
    p.add_argument("--command-timeout", type=int, default=120, help="Timeout seconds for explicit command.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    official_repo = resolve_from_root(args.official_repo)
    out_root = resolve_from_root(args.out_root)
    out_dir = ensure_dir(out_root / f"current_probe03_official_smoke_{now_stamp()}")
    log_dir = ensure_dir(out_dir / "logs")

    print("=" * 100)
    print("CURRENT PAPER PROBE 03 — OFFICIAL REPO SMOKE PLANNER/RUNNER")
    print("=" * 100)
    print(f"[ROOT]          {repo_rel(REPO_ROOT)}")
    print(f"[OFFICIAL REPO] {repo_rel(official_repo)}")
    print(f"[OUT]           {repo_rel(out_dir)}")

    audit = audit_official_repo(official_repo)

    scripts: List[CandidateScript] = []
    md_cmds: List[CandidateCommand] = []
    candidates: List[CandidateCommand] = []
    command_results: List[Dict[str, Any]] = []
    deps: List[Dict[str, Any]] = []

    if audit.get("status") == "found":
        scripts = discover_python_scripts(official_repo)
        md_cmds = extract_markdown_commands(official_repo)
        candidates = build_probe_commands(official_repo, scripts, md_cmds)

    if args.editable_install:
        if audit.get("status") != "found":
            command_results.append({
                "kind": "editable_install",
                "index": 0,
                "source": "probe03",
                "command": "python -m pip install -e .",
                "cwd": "official_repo_root",
                "risk": "install",
                "ok": False,
                "returncode": "",
                "error": "official repo missing",
                "stdout_log": "",
                "stderr_log": "",
            })
        else:
            print("[INSTALL] python -m pip install -e .")
            res = editable_install(official_repo, log_dir=log_dir, timeout=args.install_timeout)
            command_results.append({
                "kind": "editable_install",
                "index": 0,
                "source": "probe03",
                "command": f"{sys.executable} -m pip install -e .",
                "cwd": "official_repo_root",
                "risk": "install",
                "ok": res.get("ok"),
                "returncode": res.get("returncode"),
                "error": res.get("error"),
                "stdout_log": res.get("stdout_log"),
                "stderr_log": res.get("stderr_log"),
            })

    if args.import_smoke:
        print("[IMPORT] import smoke")
        deps = import_smoke(log_dir=log_dir, official_repo=official_repo if audit.get("status") == "found" else None, extra_imports=args.extra_import)
    else:
        # Lightweight find_spec only for core imports; no subprocess.
        deps = [module_import_status(m) for m in CORE_IMPORTS + list(args.extra_import)]

    if args.run_help_smoke:
        if audit.get("status") != "found":
            print("[HELP] skipped: official repo missing")
        else:
            print("[HELP] running safe --help smoke commands")
            command_results.extend(
                run_help_smoke(
                    official_repo,
                    candidates,
                    log_dir=log_dir,
                    max_commands=max(1, args.max_help_commands),
                    timeout=max(5, args.help_timeout),
                )
            )

    if args.run_command.strip():
        if audit.get("status") != "found":
            command_results.append({
                "kind": "explicit_command",
                "index": 0,
                "source": "user_supplied",
                "command": args.run_command,
                "cwd": "official_repo_root",
                "risk": "user_explicit",
                "ok": False,
                "returncode": "",
                "error": "official repo missing",
                "stdout_log": "",
                "stderr_log": "",
            })
        else:
            print(f"[RUN] {args.run_command}")
            command_results.append(
                run_explicit_command(
                    official_repo,
                    command=args.run_command,
                    log_dir=log_dir,
                    timeout=max(5, args.command_timeout),
                )
            )

    # Persist outputs.
    write_json(out_dir / "official_repo_audit.json", audit)
    write_audit_md(out_dir / "official_repo_audit.md", audit, candidates, scripts)
    write_json(out_dir / "dependency_report.json", deps)
    write_csv(out_dir / "candidate_commands.csv", [c.asdict() for c in candidates])
    write_csv(out_dir / "candidate_scripts.csv", [s.asdict() for s in scripts])
    write_csv(out_dir / "command_results.csv", command_results)
    write_summary_md(
        out_dir / "PROBE03_SUMMARY.md",
        out_dir=out_dir,
        official_repo=official_repo,
        audit=audit,
        deps=deps,
        candidates=candidates,
        scripts=scripts,
        command_results=command_results,
    )

    missing = [d["module"] for d in deps if not d.get("importable")]
    print("\n[SUMMARY]")
    print(f"  official_repo_status : {audit.get('status')}")
    print(f"  official_repo        : {repo_rel(official_repo)}")
    print(f"  candidate_commands   : {len(candidates)}")
    print(f"  candidate_scripts    : {len(scripts)}")
    print(f"  missing_core_imports : {missing if missing else 'none'}")
    print(f"  command_results      : {len(command_results)}")
    print(f"  output               : {repo_rel(out_dir)}")
    print("\nNext:")
    print(f"  open {repo_rel(out_dir / 'PROBE03_SUMMARY.md')}")
    print(f"  open {repo_rel(out_dir / 'official_repo_audit.md')}")
    print(f"  open {repo_rel(out_dir / 'candidate_commands.csv')}")


if __name__ == "__main__":
    main()
