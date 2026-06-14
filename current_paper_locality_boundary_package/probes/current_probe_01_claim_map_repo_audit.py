#!/usr/bin/env python3
r"""
current_probe_01_claim_map_repo_audit.py

Current paper Probe 01
======================

Purpose
-------
Make the first standalone, non-heavy probe for:

  "Scalable Boltzmann generators for equilibrium sampling of large-scale materials"
  Schebek, Noe, Rogal, Nature Communications (2026) 17:5010
  DOI: 10.1038/s41467-026-73900-9

This probe does NOT train a Boltzmann Generator yet.

It does three things:

1. Builds a claim ledger from the paper's stated claims.
2. Audits the official code repository structure and local environment.
3. Emits a sequenced probe plan aimed at finding the exact mechanism that
   blocks purely/local-environment BGs from working on long-range structure.

Why this exists
---------------
Before running expensive JAX/OpenMM training, we want the same discipline as the
DHA/PaiNN probe: exact claims first, smoke/audit second, then surgical tests.

Expected usage
--------------
Run from the probe directory:

  python .\current_probe_01_claim_map_repo_audit.py --clone-if-missing

If git is not installed or you want to point at an already-cloned repo:

  python .\current_probe_01_claim_map_repo_audit.py --repo-dir bgmat --no-clone

Optional import smoke after installing repo deps:

  python .\current_probe_01_claim_map_repo_audit.py --repo-dir bgmat --import-smoke

Outputs
-------
A timestamped folder under probe_runs/, containing:

  claim_ledger.json
  claim_ledger.csv
  claim_ledger.md
  environment_report.json
  repo_audit.json
  repo_audit.md
  next_probe_plan.md
  PROBE01_SUMMARY.md

Notes
-----
The official paper says the code is available at:
  https://github.com/maxschebek/bgmat

This script only audits. It does not assume the repo is installed, and it does
not run training/evaluation scripts.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def resolve_from_root(p: "str | Path") -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        rec.update(
            ok=(proc.returncode == 0),
            returncode=proc.returncode,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
        )
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=str)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def maybe_version(modname: str) -> Dict[str, Any]:
    rec = {"module": modname, "import_ok": False, "version": None, "error": None}
    try:
        mod = importlib.import_module(modname)
        rec["import_ok"] = True
        rec["version"] = getattr(mod, "__version__", None)
    except Exception as exc:
        rec["error"] = repr(exc)
    return rec


def flatten_text_files(root: Path, max_files: int = 5000) -> Iterable[Path]:
    allowed = {
        ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json",
        ".ipynb", ".cfg", ".ini", ".sh", ".ps1",
    }
    count = 0
    for p in root.rglob("*"):
        if count >= max_files:
            break
        if not p.is_file():
            continue
        if ".git" in p.parts:
            continue
        if p.suffix.lower() not in allowed:
            continue
        try:
            if p.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue
        count += 1
        yield p


# -----------------------------------------------------------------------------
# Claim ledger
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class Claim:
    claim_id: str
    category: str
    paper_anchor: str
    stated_claim: str
    metric_or_object: str
    exact_probe_implication: str
    failure_mechanism_to_watch: str
    priority: str = "medium"

    def asdict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


CLAIMS: List[Claim] = [
    Claim(
        claim_id="C01_architecture_scalability",
        category="headline",
        paper_anchor="Abstract / page 1",
        stated_claim=(
            "The paper claims a Boltzmann Generator architecture that overcomes prior "
            "large-system scaling limits by combining augmented coupling flows with GNNs "
            "over local environments."
        ),
        metric_or_object="Architecture: augmented coupling flow + GNN local environment embeddings.",
        exact_probe_implication=(
            "Probe must verify the official code exposes augmented-flow and local-GNN components "
            "as first-class model pieces, not only as post-hoc analysis language."
        ),
        failure_mechanism_to_watch=(
            "If the implementation depends on hidden global information, absolute-size-specific "
            "features, or system-specific scripts, the local-transfer claim weakens."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C02_energy_based_no_target_samples",
        category="training",
        paper_anchor="Results / page 2",
        stated_claim=(
            "The local augmented coupling-flow setup enables energy-based training using the "
            "target potential, without requiring target-distribution samples."
        ),
        metric_or_object="Training objective / KL loss / target energy evaluations.",
        exact_probe_implication=(
            "Later probes should distinguish true energy-based training from any path that "
            "implicitly consumes MD target samples for training."
        ),
        failure_mechanism_to_watch=(
            "If configs from MD/reference ensembles are needed to stabilize training, the "
            "energy-only claim becomes conditional."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C03_local_not_full_configuration",
        category="locality",
        paper_anchor="Results / page 2",
        stated_claim=(
            "The generative process is based on local environments rather than the full "
            "configuration."
        ),
        metric_or_object="Neighbor lists / fixed local neighborhoods / no all-pairs attention.",
        exact_probe_implication=(
            "We need a locality audit: neighbor cutoff/count, receptive field, periodic handling, "
            "and whether any global lattice/box conditioning sneaks in the missing signal."
        ),
        failure_mechanism_to_watch=(
            "Locally indistinguishable configurations can have different long-range energies "
            "or phase weights; this is a primary candidate failure mechanism."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C04_augmented_flow_preserves_3d_distances",
        category="architecture",
        paper_anchor="Results / page 2",
        stated_claim=(
            "Augmented flows split physical and auxiliary variables, retaining full 3D coordinates "
            "so interatomic distances can be computed despite coupling-flow splits."
        ),
        metric_or_object="Physical variables x, auxiliary variables a, noised-copy auxiliary base.",
        exact_probe_implication=(
            "Probe should inspect whether physical/auxiliary update blocks are separable and "
            "whether local-distance computation uses auxiliary or physical states as stated."
        ),
        failure_mechanism_to_watch=(
            "Auxiliary marginal mismatch: joint density can look good while physical marginal "
            "or long-range correlations degrade."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C05_linear_scaling_fixed_neighbors",
        category="scaling",
        paper_anchor="Results / page 2 and Fig. 1",
        stated_claim=(
            "The local approach achieves linear scaling with system size by fixing the number "
            "of neighbors, unlike global attention-style architectures that scale quadratically."
        ),
        metric_or_object="Runtime/memory vs N; fixed neighbor count.",
        exact_probe_implication=(
            "Later probes should run timing/memory sweeps over N while holding neighbor count "
            "fixed, and also vary the neighbor count to see when accuracy changes."
        ),
        failure_mechanism_to_watch=(
            "Linear scaling can be bought by truncating information; the mechanism may be a "
            "cutoff-induced missing collective mode."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C06_displacements_from_ideal_lattice",
        category="representation",
        paper_anchor="Results / page 2",
        stated_claim=(
            "The model uses displacements from the ideal crystal lattice rather than absolute "
            "positions so bijector inputs do not scale with system size."
        ),
        metric_or_object="Coordinate representation: lattice displacement field.",
        exact_probe_implication=(
            "Probe must check whether success is tied to crystalline solids with known reference "
            "lattices; this may not transfer to liquids or molecular crystals without new DOF."
        ),
        failure_mechanism_to_watch=(
            "Representation dependence: long-wavelength strain/phonon or defect modes may be "
            "handled differently than local displacement noise."
        ),
        priority="medium",
    ),
    Claim(
        claim_id="C07_marginal_density_M200",
        category="density/free_energy",
        paper_anchor="Results / page 3",
        stated_claim=(
            "Only the joint augmented density is exact; physical marginal density is approximated "
            "by drawing auxiliary samples, using M=200 unless otherwise stated."
        ),
        metric_or_object="Joint density, marginal density, M auxiliary samples.",
        exact_probe_implication=(
            "We need to track joint-vs-marginal ESS/free energy gaps and test sensitivity to M."
        ),
        failure_mechanism_to_watch=(
            "The physical samples can look structurally correct while marginal-density estimates "
            "carry the real failure."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C08_TFEP_free_energy",
        category="free_energy",
        paper_anchor="Results / page 3 and Methods",
        stated_claim=(
            "The trained flow is used within targeted free energy perturbation to estimate "
            "Helmholtz/Gibbs free energies."
        ),
        metric_or_object="TFEP weights, partition-function/free-energy estimator.",
        exact_probe_implication=(
            "Later probes should compute weight distributions, log-weight variance, ESS, and "
            "free-energy bias under controlled long-range terms."
        ),
        failure_mechanism_to_watch=(
            "A small subset of samples can dominate TFEP when long-range energy errors are "
            "extensive, causing low ESS or hidden bias."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C09_size_transfer_systems",
        category="transfer",
        paper_anchor="Results / page 3",
        stated_claim=(
            "Models trained on mW ice N=216 and FCC Lennard-Jones N=256 are applied to much "
            "larger systems: mW 512/1000/1728 and LJ 500/864/1000/1372."
        ),
        metric_or_object="Train N vs transfer N; same learned local rule.",
        exact_probe_implication=(
            "Probe series should reproduce a tiny train/small-transfer path first, then stress "
            "transfer by increasing N and adding long-range terms."
        ),
        failure_mechanism_to_watch=(
            "Constant small per-particle errors become extensive with N, so transfer can fail "
            "through exponentially worse Boltzmann weights."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C10_RDF_energy_hist_no_reweighting",
        category="sampling",
        paper_anchor="Results / page 4 and Fig. 2",
        stated_claim=(
            "For the largest investigated transferred systems, local BG samples reproduce RDFs "
            "and potential-energy histograms close to MD, already without reweighting."
        ),
        metric_or_object="RDF g(r), reduced energy histogram, no reweighting.",
        exact_probe_implication=(
            "Our validation cannot stop at RDF; we must also compare energy histograms, "
            "long-range observables, and weight-based diagnostics."
        ),
        failure_mechanism_to_watch=(
            "RDF can look correct while higher-order/long-range correlations or free-energy "
            "weights fail."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C11_ESS_decreases_with_N",
        category="diagnostic",
        paper_anchor="Results / page 4 and Table 1",
        stated_claim=(
            "ESS naturally decreases with system size because even constant per-particle errors "
            "produce exponentially amplified Boltzmann-distribution errors."
        ),
        metric_or_object="ESS vs system size.",
        exact_probe_implication=(
            "ESS decay is not a side note; it is one of the main windows into the failure "
            "mechanism. We should model ESS vs N and vs injected long-range strength."
        ),
        failure_mechanism_to_watch=(
            "The exact mechanism may be extensive log-weight error accumulation."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C12_local_beats_global_cost_efficiency",
        category="comparison",
        paper_anchor="Results / page 4 and Fig. 3",
        stated_claim=(
            "Local BGs train faster and reach higher ESS than global attention-style BGs at "
            "comparable training time; reported mW N=216 convergence is about four GPU days."
        ),
        metric_or_object="Training time, ESS curve, local vs global.",
        exact_probe_implication=(
            "We do not need full global reproduction immediately, but should preserve a baseline "
            "for cost/ESS once the local path runs."
        ),
        failure_mechanism_to_watch=(
            "A local model can outperform global under short-range potentials while still failing "
            "under nonlocal interactions."
        ),
        priority="medium",
    ),
    Claim(
        claim_id="C13_system_dependent_performance",
        category="paper_admitted_seam",
        paper_anchor="Results / page 5",
        stated_claim=(
            "The paper notes BGs do not perform equally well across systems; mW ice is easier "
            "than FCC LJ, and even structures under the same potential can differ."
        ),
        metric_or_object="ESS/performance by potential and structure.",
        exact_probe_implication=(
            "Probe should not assume one success case generalizes. We should intentionally search "
            "for the structural variable that separates easy from hard cases."
        ),
        failure_mechanism_to_watch=(
            "Different crystals may require different long-range/collective modes despite similar "
            "local neighborhoods."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C14_free_energy_low_ESS_warning",
        category="paper_admitted_seam",
        paper_anchor="Results / page 5",
        stated_claim=(
            "The paper warns that low-ESS free energy estimates can have small variance even when "
            "reliability is hard to verify; it suggests mean ESS about 0.1% was sufficient in their examples."
        ),
        metric_or_object="ESS threshold, free-energy reliability, variance trap.",
        exact_probe_implication=(
            "We need adversarial diagnostics where variance looks stable but bias appears against "
            "known reference."
        ),
        failure_mechanism_to_watch=(
            "Hidden free-energy bias under low ESS, especially after adding long-range terms."
        ),
        priority="high",
    ),
    Claim(
        claim_id="C15_energy_eval_vs_wallclock_caveat",
        category="cost",
        paper_anchor="Results / page 6",
        stated_claim=(
            "The paper emphasizes energy-evaluation savings do not always translate to wall-clock "
            "savings for cheap potentials; benefits become more favorable for expensive ML/DFT-like potentials."
        ),
        metric_or_object="Energy eval count vs wall-clock.",
        exact_probe_implication=(
            "Report both wall-clock and energy evaluations. Do not oversell speed from eval counts alone."
        ),
        failure_mechanism_to_watch=(
            "Optimization overhead dominates cheap-potential toy systems."
        ),
        priority="medium",
    ),
    Claim(
        claim_id="C16_conditional_NPT_phase",
        category="conditioning",
        paper_anchor="Results / pages 6-8",
        stated_claim=(
            "The paper conditions flows on box shape, interaction parameters, and thermodynamic variables "
            "to evaluate Gibbs free energies and phase diagrams."
        ),
        metric_or_object="Shape conditioning, lambda_3 SW conditioning, T/P conditioning.",
        exact_probe_implication=(
            "We can defer full phase diagrams, but should audit that the code contains conditional "
            "training/evaluation paths."
        ),
        failure_mechanism_to_watch=(
            "Global box/shape conditioning may carry low-dimensional nonlocal information that pure "
            "local neighborhoods do not."
        ),
        priority="medium",
    ),
    Claim(
        claim_id="C17_long_range_limitation",
        category="paper_admitted_boundary",
        paper_anchor="Discussion / page 8",
        stated_claim=(
            "The paper explicitly names long-range interactions such as electrostatics as an important "
            "future direction that cannot be represented within any purely local formulation."
        ),
        metric_or_object="Long-range interaction terms and observables.",
        exact_probe_implication=(
            "This becomes our main scalpel target: construct controlled long-range terms and find "
            "which metric fails first and why."
        ),
        failure_mechanism_to_watch=(
            "Local environments can be identical within cutoff while the correct Boltzmann weight depends "
            "on distant charges/collective fields."
        ),
        priority="highest",
    ),
    Claim(
        claim_id="C18_future_molecular_crystals_orientations",
        category="paper_admitted_boundary",
        paper_anchor="Discussion / page 8",
        stated_claim=(
            "The paper says molecular crystals would need more elaborate architectural modifications "
            "for orientational degrees of freedom such as rigid-body constraints."
        ),
        metric_or_object="Orientational DOF / rigid-body constraints.",
        exact_probe_implication=(
            "Do not jump to molecular crystals yet; if we do, separate orientation failure from "
            "long-range-interaction failure."
        ),
        failure_mechanism_to_watch=(
            "Missing orientation variables can masquerade as a long-range failure."
        ),
        priority="low",
    ),
]


def claim_rows() -> List[Dict[str, str]]:
    return [c.asdict() for c in CLAIMS]


def write_claim_markdown(path: Path, claims: List[Claim]) -> None:
    lines = [
        "# Current Probe 01 — claim ledger",
        "",
        "This ledger is the exact test contract for the current paper sandbox.",
        "Probe 01 does not validate the claims; it maps them into measurable audit/probe targets.",
        "",
        "## Highest-priority seam",
        "",
        (
            "The main seam is C17: the paper explicitly marks purely local formulations as "
            "unable to represent long-range interactions such as electrostatics. The probe series "
            "will search for the first metric that fails when controlled long-range structure is added."
        ),
        "",
        "## Claims",
        "",
    ]
    for c in claims:
        lines += [
            f"### {c.claim_id} — {c.category} — priority: {c.priority}",
            "",
            f"**Paper anchor:** {c.paper_anchor}",
            "",
            f"**Stated claim:** {c.stated_claim}",
            "",
            f"**Metric/object:** {c.metric_or_object}",
            "",
            f"**Probe implication:** {c.exact_probe_implication}",
            "",
            f"**Failure mechanism to watch:** {c.failure_mechanism_to_watch}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Environment and repo audit
# -----------------------------------------------------------------------------

EXPECTED_DIRS = ["experiments", "models", "systems", "tutorial", "utils"]
EXPECTED_CONFIG_TERMS = [
    "lennard", "lj", "mW", "monatomic", "stillinger", "weber", "silicon",
    "diamond", "beta", "tin", "bcc", "fcc", "hcp", "hexagonal", "cubic",
]
SEARCH_TERMS = [
    "augmented",
    "auxiliary",
    "local",
    "neighbor",
    "neighbour",
    "cutoff",
    "rational",
    "spline",
    "ESS",
    "effective sample",
    "marginal",
    "joint",
    "TFEP",
    "free energy",
    "openmm",
    "jax",
    "haiku",
    "distrax",
    "lennard",
    "mW",
    "Stillinger",
    "Weber",
    "silicon",
    "N=216",
    "N = 216",
    "N=256",
    "N = 256",
]


def collect_environment(import_smoke: bool) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "timestamp": now_stamp(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "cpu_count": os.cpu_count(),
        "git_path": shutil.which("git"),
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "commands": {},
        "imports": [],
    }
    if report["git_path"]:
        report["commands"]["git_version"] = run_cmd(["git", "--version"], timeout=20)
    if report["nvidia_smi_path"]:
        report["commands"]["nvidia_smi"] = run_cmd(["nvidia-smi"], timeout=20)

    modules = ["numpy", "scipy", "matplotlib"]
    if import_smoke:
        modules += ["jax", "jaxlib", "haiku", "distrax", "openmm", "torch", "ase"]
    report["imports"] = [maybe_version(m) for m in modules]
    return report


def clone_or_find_repo(args: argparse.Namespace, out_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    audit: Dict[str, Any] = {
        "repo_url": args.repo_url,
        "requested_repo_dir": args.repo_dir,
        "clone_if_missing": bool(args.clone_if_missing),
        "no_clone": bool(args.no_clone),
        "repo_path": None,
        "clone_attempt": None,
        "exists": False,
        "is_git_repo": False,
    }
    repo = Path(args.repo_dir).expanduser()
    if not repo.is_absolute():
        repo = Path.cwd() / repo
    audit["repo_path"] = str(repo)

    if repo.exists():
        audit["exists"] = True
        return repo, audit

    if args.no_clone or not args.clone_if_missing:
        return None, audit

    git = shutil.which("git")
    if git is None:
        audit["clone_attempt"] = {"ok": False, "error": "git not found on PATH"}
        return None, audit

    ensure_dir(repo.parent)
    audit["clone_attempt"] = run_cmd(["git", "clone", "--depth", "1", args.repo_url, str(repo)], timeout=args.clone_timeout)
    if repo.exists():
        audit["exists"] = True
        return repo, audit
    return None, audit


def audit_repo(repo: Optional[Path], base_audit: Dict[str, Any]) -> Dict[str, Any]:
    audit = dict(base_audit)
    if repo is None or not repo.exists():
        audit["status"] = "repo_missing"
        return audit

    audit["status"] = "repo_found"
    audit["root"] = str(repo)
    audit["expected_dirs"] = {d: (repo / d).exists() for d in EXPECTED_DIRS}
    audit["top_level_files"] = sorted([p.name for p in repo.iterdir() if p.is_file()])[:200]
    audit["top_level_dirs"] = sorted([p.name for p in repo.iterdir() if p.is_dir()])[:200]
    audit["is_git_repo"] = (repo / ".git").exists()

    if audit["is_git_repo"]:
        audit["git"] = {
            "rev_parse_head": run_cmd(["git", "rev-parse", "HEAD"], cwd=repo, timeout=20),
            "branch": run_cmd(["git", "branch", "--show-current"], cwd=repo, timeout=20),
            "status_short": run_cmd(["git", "status", "--short"], cwd=repo, timeout=20),
            "remote_v": run_cmd(["git", "remote", "-v"], cwd=repo, timeout=20),
        }

    setup_candidates = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "environment.yml", "README.md"]
    audit["setup_files"] = {}
    for name in setup_candidates:
        p = repo / name
        audit["setup_files"][name] = p.exists()

    experiment_files = []
    for p in repo.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        low = str(p.relative_to(repo)).lower()
        if any(term.lower() in low for term in EXPECTED_CONFIG_TERMS):
            if p.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".toml", ".ipynb", ".md"}:
                experiment_files.append(rel(p, repo))
    audit["experiment_related_files"] = sorted(experiment_files)[:500]

    term_hits: Dict[str, List[Dict[str, Any]]] = {term: [] for term in SEARCH_TERMS}
    for p in flatten_text_files(repo):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        low_txt = txt.lower()
        for term in SEARCH_TERMS:
            count = low_txt.count(term.lower())
            if count:
                term_hits[term].append({"file": rel(p, repo), "count": count})

    for term in list(term_hits.keys()):
        term_hits[term] = sorted(term_hits[term], key=lambda x: x["count"], reverse=True)[:15]
    audit["term_hits_top15"] = term_hits

    # Lightweight repo-size stats.
    n_files = 0
    n_py = 0
    n_nb = 0
    total_bytes = 0
    for p in repo.rglob("*"):
        if ".git" in p.parts or not p.is_file():
            continue
        n_files += 1
        if p.suffix.lower() == ".py":
            n_py += 1
        if p.suffix.lower() == ".ipynb":
            n_nb += 1
        try:
            total_bytes += p.stat().st_size
        except OSError:
            pass
    audit["repo_stats"] = {
        "n_files": n_files,
        "n_python_files": n_py,
        "n_notebooks": n_nb,
        "total_bytes": total_bytes,
    }

    return audit


def write_repo_markdown(path: Path, audit: Dict[str, Any]) -> None:
    lines = [
        "# Current Probe 01 — repo audit",
        "",
        f"Repo URL: `{audit.get('repo_url')}`",
        f"Repo path: `{audit.get('repo_path')}`",
        f"Status: **{audit.get('status')}**",
        "",
    ]

    if audit.get("clone_attempt") is not None:
        ca = audit["clone_attempt"]
        lines += [
            "## Clone attempt",
            "",
            f"OK: `{ca.get('ok')}`",
            f"Return code: `{ca.get('returncode')}`",
            "",
        ]
        if ca.get("stderr"):
            lines += ["```", str(ca.get("stderr"))[:4000], "```", ""]

    if audit.get("status") != "repo_found":
        lines += [
            "Repo was not found/audited. Run with `--clone-if-missing`, install git, or pass `--repo-dir`.",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "## Expected directories",
        "",
    ]
    for d, ok in audit.get("expected_dirs", {}).items():
        lines.append(f"- `{d}`: {'YES' if ok else 'NO'}")
    lines.append("")

    lines += [
        "## Setup files",
        "",
    ]
    for f, ok in audit.get("setup_files", {}).items():
        lines.append(f"- `{f}`: {'YES' if ok else 'NO'}")
    lines.append("")

    lines += [
        "## Repo stats",
        "",
        "```json",
        json.dumps(audit.get("repo_stats", {}), indent=2),
        "```",
        "",
    ]

    lines += [
        "## Experiment/config-related files",
        "",
    ]
    files = audit.get("experiment_related_files", [])
    if files:
        for f in files[:120]:
            lines.append(f"- `{f}`")
        if len(files) > 120:
            lines.append(f"- ... {len(files) - 120} more")
    else:
        lines.append("_No obvious experiment files found by filename scan._")
    lines.append("")

    lines += [
        "## Term-hit summary",
        "",
        "Top matching files for each term; this is a crude audit, not proof.",
        "",
    ]
    for term, hits in audit.get("term_hits_top15", {}).items():
        total = sum(h["count"] for h in hits)
        if not hits:
            continue
        lines += [f"### `{term}` — top-hit total shown: {total}", ""]
        for h in hits[:8]:
            lines.append(f"- `{h['file']}`: {h['count']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_next_probe_plan(path: Path) -> None:
    text = """# Current paper — next probe plan

## Probe 01 status

Probe 01 is a claims/code/environment audit. It does not train or validate the paper.

## Probe 02 candidate: official tiny tutorial smoke

Goal:
- Install the official repository in an isolated environment.
- Run the smallest available tutorial or example.
- Confirm sample generation, energy evaluation, log weights, ESS calculation, and output artifacts.

Success:
- The official tiny path runs end-to-end.
- We can identify the exact files/functions for:
  - model construction,
  - local neighbor embedding,
  - augmented variables,
  - energy-based loss,
  - ESS/free-energy evaluation.

Failure:
- Install break, dependency mismatch, missing data, undocumented checkpoint path, or no tiny runnable example.

## Probe 03 candidate: local-neighborhood audit inside the official model

Goal:
- Trace the model's information path:
  local neighbor list -> GNN embedding -> coupling transform -> logdet/log weight.
- Record neighbor count/cutoff and whether any global/box/lattice feature enters.

Success:
- We can explicitly say what information each particle update can and cannot access.

## Probe 04 candidate: size-transfer minimal reproduction

Goal:
- Train/evaluate the smallest feasible local model.
- Transfer to a larger N.
- Compare RDF, energy histogram, log-weight distribution, ESS, and wall-clock.

Success:
- A tiny version of the paper's size-transfer trick works.

## Probe 05 candidate: locality stress

Goal:
- Vary neighbor count/cutoff and identify which metric fails first.
- Keep the potential unchanged initially.

Success:
- We know whether failure tracks cutoff/receptive field.

## Probe 06 candidate: controlled long-range injection

Goal:
- Add a controlled long-range term where local neighborhoods are insufficient.
- Measure the failure order:
  RDF -> energy hist -> log-weight tails -> ESS -> free energy.

Success:
- We isolate the mechanism preventing purely local transfer from working long range.
"""
    path.write_text(text, encoding="utf-8")


def write_summary(path: Path, out_dir: Path, env: Dict[str, Any], audit: Dict[str, Any]) -> None:
    missing_dirs = []
    if audit.get("status") == "repo_found":
        missing_dirs = [d for d, ok in audit.get("expected_dirs", {}).items() if not ok]

    import_rows = env.get("imports", [])
    import_summary = ", ".join(
        f"{r['module']}={'ok' if r['import_ok'] else 'missing'}"
        for r in import_rows
    )

    text = f"""# PROBE 01 SUMMARY

Output folder: `{out_dir}`

## What this probe did

- Wrote a claim ledger from the current paper.
- Collected local environment information.
- Audited the official repository path if available.
- Wrote the next probe plan.

## Environment

- Platform: `{env.get('platform')}`
- Python: `{env.get('python_executable')}`
- Git: `{env.get('git_path')}`
- NVIDIA SMI: `{env.get('nvidia_smi_path')}`
- Import smoke: {import_summary}

## Repo audit

- Status: `{audit.get('status')}`
- Path: `{audit.get('repo_path')}`
- Missing expected dirs: `{missing_dirs}`

## Main seam to pursue

The current paper succeeds by imposing a local-environment bottleneck.
The admitted boundary is long-range interactions that cannot be represented
inside any purely local formulation. The probe series should now locate the
first measurable failure mode under controlled nonlocal structure.

## Generated files

- `claim_ledger.json`
- `claim_ledger.csv`
- `claim_ledger.md`
- `environment_report.json`
- `repo_audit.json`
- `repo_audit.md`
- `next_probe_plan.md`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Current paper Probe 01: claim map + repo/environment audit.")
    p.add_argument("--out-root", default="probe_runs", help="Output root directory, relative to this script unless absolute.")
    p.add_argument("--repo-url", default="https://github.com/maxschebek/bgmat", help="Official repo URL to clone/audit.")
    p.add_argument("--repo-dir", default="bgmat", help="Local repo directory to audit.")
    p.add_argument("--clone-if-missing", action="store_true", help="Clone repo if --repo-dir is missing.")
    p.add_argument("--no-clone", action="store_true", help="Do not clone even if repo is missing.")
    p.add_argument("--clone-timeout", type=int, default=300, help="Seconds allowed for git clone.")
    p.add_argument("--import-smoke", action="store_true", help="Try importing JAX/Haiku/Distrax/OpenMM/etc.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(resolve_from_root(args.out_root) / f"current_probe01_{now_stamp()}")

    print("=" * 100)
    print("CURRENT PAPER PROBE 01 — CLAIM MAP + REPO AUDIT")
    print("=" * 100)
    print(f"[OUT] {out_dir}")

    # Claims.
    rows = claim_rows()
    write_json(out_dir / "claim_ledger.json", rows)
    write_csv(out_dir / "claim_ledger.csv", rows)
    write_claim_markdown(out_dir / "claim_ledger.md", CLAIMS)
    print(f"[CLAIMS] wrote {len(rows)} claims")

    # Environment.
    env = collect_environment(import_smoke=bool(args.import_smoke))
    write_json(out_dir / "environment_report.json", env)
    print("[ENV] collected")

    # Repo.
    repo, base_audit = clone_or_find_repo(args, out_dir)
    audit = audit_repo(repo, base_audit)
    write_json(out_dir / "repo_audit.json", audit)
    write_repo_markdown(out_dir / "repo_audit.md", audit)
    print(f"[REPO] status={audit.get('status')} path={audit.get('repo_path')}")

    # Plan and summary.
    write_next_probe_plan(out_dir / "next_probe_plan.md")
    write_summary(out_dir / "PROBE01_SUMMARY.md", out_dir, env, audit)

    print("\n[SUMMARY]")
    print(f"  claims              : {len(rows)}")
    print(f"  repo_status         : {audit.get('status')}")
    print(f"  repo_path           : {audit.get('repo_path')}")
    print(f"  import_smoke        : {bool(args.import_smoke)}")
    print(f"  output              : {out_dir}")
    print("\nNext:")
    print(f"  open {out_dir / 'PROBE01_SUMMARY.md'}")
    print(f"  open {out_dir / 'claim_ledger.md'}")
    print(f"  open {out_dir / 'repo_audit.md'}")


if __name__ == "__main__":
    main()
