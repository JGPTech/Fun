#!/usr/bin/env python3
"""
run_all_probes_recommended_settings.py

Repo-root runner for the current-paper locality-boundary audit.

This version is GitHub-friendly:

- You do NOT need to commit bgmat/ or the Nature source-data folder.
- On first run, if bgmat/ is missing, it fetches the official bgmat repo.
- On first run, if 41467_2026_73900_MOESM3_ESM/ is missing or incomplete,
  it downloads and extracts the official Source Data ZIP.
- It keeps generated probe outputs under <repo-root>/probe_runs.
- It seeds PYTHONPATH with <repo-root>/bgmat and <repo-root> for subprocesses.
- It runs Probes 01-11 in the recommended order.

Run from repo root:

    python .\run_all_probes_recommended_settings.py

Dry run commands only:

    python .\run_all_probes_recommended_settings.py --dry-run

Fetch dependencies only:

    python .\run_all_probes_recommended_settings.py --fetch-only

Disable fetching:

    python .\run_all_probes_recommended_settings.py --no-fetch

Continue after a nonzero return code:

    python .\run_all_probes_recommended_settings.py --continue-on-error
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple


BGMAT_GIT_URL = "https://github.com/maxschebek/bgmat.git"
BGMAT_ZIP_URLS = [
    "https://github.com/maxschebek/bgmat/archive/refs/tags/v1.0.0.zip",
    "https://github.com/maxschebek/bgmat/archive/refs/heads/main.zip",
]
SOURCE_DATA_URL = (
    "https://static-content.springer.com/esm/"
    "art%3A10.1038%2Fs41467-026-73900-9/"
    "MediaObjects/41467_2026_73900_MOESM3_ESM.zip"
)

SOURCE_DIR_NAME = "41467_2026_73900_MOESM3_ESM"
EXPECTED_SOURCE_WORKBOOKS = [
    "Figure2.xlsx",
    "Figure3.xlsx",
    "Figure4.xlsx",
    "Figure5.xlsx",
    "Figure6.xlsx",
    "Figure7.xlsx",
    "SupplementaryFigure1.xlsx",
    "SupplementaryFigure2.xlsx",
    "SupplementaryFigure3.xlsx",
    "SupplementaryFigure4.xlsx",
    "SupplementaryFigure5.xlsx",
]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def repo_root_from_here() -> Path:
    here = Path(__file__).resolve().parent

    # Normal placement: repo root.
    if (here / "probes").is_dir():
        return here

    # Accept being placed inside probes by mistake.
    if here.name.lower() == "probes":
        return here.parent

    # Last chance: current working directory.
    cwd = Path.cwd().resolve()
    if (cwd / "probes").is_dir():
        return cwd

    return here


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if rows:
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    else:
        fieldnames = ["step", "name", "status", "returncode"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def ps_cmd(cmd: List[str]) -> str:
    # Display-only. Actual subprocess uses list args, not shell=True.
    parts = []
    for item in cmd:
        s = str(item)
        if " " in s or "\\" in s or ":" in s:
            parts.append(f'"{s}"')
        else:
            parts.append(s)
    return " ".join(parts)


def add_pythonpath(env: Dict[str, str], repo_root: Path, official_repo: Path) -> Dict[str, str]:
    existing = env.get("PYTHONPATH", "")
    pieces = [
        str(official_repo.resolve()),  # contains bgmat/ package directory
        str(repo_root.resolve()),
    ]
    if existing:
        pieces.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pieces)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def find_script(probes_dir: Path, filename: str) -> Path:
    path = probes_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing probe script: {path}")
    return path


# -----------------------------------------------------------------------------
# First-run fetch helpers
# -----------------------------------------------------------------------------

def bgmat_present(path: Path) -> bool:
    return (path / "setup.py").exists() and (path / "bgmat").is_dir()


def source_data_present(path: Path) -> Tuple[bool, List[str]]:
    missing = [name for name in EXPECTED_SOURCE_WORKBOOKS if not (path / name).exists()]
    return (len(missing) == 0), missing


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    ensure_dir(dest.parent)
    print(f"[DOWNLOAD] {url}")
    print(f"[TO]       {dest}")

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 locality-boundary-audit-runner",
            "Accept": "*/*",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None

        with dest.open("wb") as f:
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_int:
                    pct = downloaded * 100.0 / total_int
                    print(f"\r  {downloaded/1_000_000:.1f} MB / {total_int/1_000_000:.1f} MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r  {downloaded/1_000_000:.1f} MB", end="")
        print()


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    dest_resolved = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = dest_resolved / member.filename
            try:
                member_path.resolve().relative_to(dest_resolved)
            except Exception as exc:
                raise RuntimeError(f"Unsafe zip path: {member.filename}") from exc
        zf.extractall(dest_resolved)


def copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def fetch_bgmat(repo_root: Path, args: argparse.Namespace) -> bool:
    target = repo_root / "bgmat"
    if bgmat_present(target) and not args.force_fetch:
        print(f"[FETCH] bgmat already present: {target}")
        return True

    if target.exists() and args.force_fetch:
        print(f"[FETCH] Removing existing bgmat due to --force-fetch: {target}")
        shutil.rmtree(target)

    downloads = ensure_dir(repo_root / "_downloads")
    git_exe = shutil.which("git")

    if git_exe and not args.no_git:
        cmd = [
            git_exe,
            "clone",
            "--depth", "1",
            "--branch", args.bgmat_ref,
            args.bgmat_git_url,
            str(target),
        ]
        print("[FETCH] Cloning bgmat with git:")
        print("        " + ps_cmd(cmd))
        p = subprocess.run(cmd, cwd=str(repo_root), text=True, encoding="utf-8", errors="replace", capture_output=True)
        if p.returncode == 0 and bgmat_present(target):
            print("[FETCH] bgmat clone OK")
            return True

        print("[FETCH] git clone failed; trying fallback clone without --branch")
        if target.exists():
            shutil.rmtree(target)
        cmd = [git_exe, "clone", "--depth", "1", args.bgmat_git_url, str(target)]
        print("        " + ps_cmd(cmd))
        p = subprocess.run(cmd, cwd=str(repo_root), text=True, encoding="utf-8", errors="replace", capture_output=True)
        if p.returncode == 0 and bgmat_present(target):
            print("[FETCH] bgmat clone OK")
            return True

        print("[FETCH] git fallback failed.")
        if p.stdout:
            print("[git stdout]", p.stdout[-2000:])
        if p.stderr:
            print("[git stderr]", p.stderr[-2000:])
        if target.exists():
            shutil.rmtree(target)

    print("[FETCH] Downloading bgmat ZIP fallback")
    urls = [args.bgmat_zip_url] if args.bgmat_zip_url else BGMAT_ZIP_URLS
    last_error = None
    for i, url in enumerate(urls, start=1):
        zip_path = downloads / f"bgmat_fetch_{i}.zip"
        extract_dir = downloads / f"bgmat_fetch_{i}_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        try:
            download_file(url, zip_path, timeout=args.fetch_timeout)
            safe_extract_zip(zip_path, extract_dir)

            candidates = []
            for p in extract_dir.rglob("setup.py"):
                candidate = p.parent
                if (candidate / "bgmat").is_dir():
                    candidates.append(candidate)

            if not candidates:
                raise RuntimeError("Could not find extracted bgmat repo root containing setup.py and bgmat/")

            copytree_replace(candidates[0], target)
            if bgmat_present(target):
                print(f"[FETCH] bgmat ZIP extraction OK: {target}")
                return True
        except Exception as exc:
            last_error = exc
            print(f"[FETCH] bgmat ZIP attempt failed: {exc!r}")

    raise RuntimeError(f"Could not fetch bgmat. Last error: {last_error!r}")


def fetch_source_data(repo_root: Path, args: argparse.Namespace) -> bool:
    target = repo_root / SOURCE_DIR_NAME
    ok, missing = source_data_present(target)
    if ok and not args.force_fetch:
        print(f"[FETCH] source data already present: {target}")
        return True

    if target.exists() and args.force_fetch:
        print(f"[FETCH] Removing existing source data due to --force-fetch: {target}")
        shutil.rmtree(target)

    if target.exists():
        print(f"[FETCH] source data folder exists but is incomplete. Missing: {missing}")

    downloads = ensure_dir(repo_root / "_downloads")
    zip_path = downloads / f"{SOURCE_DIR_NAME}.zip"
    extract_dir = downloads / f"{SOURCE_DIR_NAME}_extract"

    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    download_file(args.source_data_url, zip_path, timeout=args.fetch_timeout)
    safe_extract_zip(zip_path, extract_dir)

    # The ZIP may contain the XLSX files directly or nested under one folder.
    found: Dict[str, Path] = {}
    for name in EXPECTED_SOURCE_WORKBOOKS:
        matches = list(extract_dir.rglob(name))
        if matches:
            found[name] = matches[0]

    missing_after_extract = [name for name in EXPECTED_SOURCE_WORKBOOKS if name not in found]
    if missing_after_extract:
        raise RuntimeError(
            "Source Data ZIP extracted, but expected workbooks were missing: "
            + ", ".join(missing_after_extract)
        )

    ensure_dir(target)
    for name, src in found.items():
        shutil.copy2(src, target / name)

    ok, missing = source_data_present(target)
    if not ok:
        raise RuntimeError(f"Source data copy incomplete. Missing: {missing}")

    print(f"[FETCH] source data extraction OK: {target}")
    return True


def ensure_external_assets(repo_root: Path, args: argparse.Namespace) -> None:
    if args.no_fetch:
        print("[FETCH] Disabled by --no-fetch")
        return

    print("=" * 100)
    print("FIRST-RUN ASSET CHECK")
    print("=" * 100)
    print("[INFO] This runner can fetch bgmat/ and Source Data if they are missing.")
    print("[INFO] If you redistribute this repo, keep third-party license/citation terms clear.")
    print()

    fetch_bgmat(repo_root, args)
    fetch_source_data(repo_root, args)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_step(
    idx: int,
    name: str,
    cmd: List[str],
    repo_root: Path,
    logs_dir: Path,
    env: Dict[str, str],
    timeout: Optional[int],
) -> Dict[str, Any]:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:100]
    stdout_log = logs_dir / f"{idx:02d}_{safe}.stdout.txt"
    stderr_log = logs_dir / f"{idx:02d}_{safe}.stderr.txt"

    print()
    print("=" * 100)
    print(f"[{idx:02d}] {name}")
    print("=" * 100)
    print("[CMD]", ps_cmd(cmd))
    print("[CWD]", repo_root)
    print()

    rec: Dict[str, Any] = {
        "step": idx,
        "name": name,
        "cmd": cmd,
        "cmd_text": ps_cmd(cmd),
        "cwd": str(repo_root),
        "status": "not_started",
        "returncode": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "error": "",
    }

    try:
        p = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        stdout_log.write_text(p.stdout or "", encoding="utf-8")
        stderr_log.write_text(p.stderr or "", encoding="utf-8")

        rec["returncode"] = p.returncode
        rec["status"] = "pass" if p.returncode == 0 else "fail"

        if p.stdout:
            print(p.stdout, end="" if p.stdout.endswith("\n") else "\n")
        if p.stderr:
            print("\n[stderr]")
            print(p.stderr, end="" if p.stderr.endswith("\n") else "\n")

    except subprocess.TimeoutExpired as exc:
        rec["returncode"] = "timeout"
        rec["status"] = "timeout"
        rec["error"] = repr(exc)
        stdout_log.write_text(exc.stdout or "", encoding="utf-8")
        stderr_log.write_text((exc.stderr or "") + "\n\nTIMEOUT\n" + repr(exc), encoding="utf-8")
        print(f"[TIMEOUT] {exc!r}")

    except Exception as exc:
        rec["returncode"] = "error"
        rec["status"] = "error"
        rec["error"] = repr(exc)
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(repr(exc), encoding="utf-8")
        print(f"[ERROR] {exc!r}")

    print()
    print(f"[RESULT] {rec['status']} returncode={rec['returncode']}")
    return rec


def latest_dir(root: Path, prefix: str) -> Optional[Path]:
    if not root.exists():
        return None
    matches = sorted([p for p in root.glob(prefix + "*") if p.is_dir()], key=lambda p: p.name)
    return matches[-1] if matches else None


def write_summary(run_dir: Path, repo_root: Path, out_root: Path, records: List[Dict[str, Any]]) -> None:
    passed = [r for r in records if r.get("status") == "pass"]
    failed = [r for r in records if r.get("status") != "pass"]

    lines = [
        "# Run-all probes summary",
        "",
        f"Run folder: `{rel(run_dir, repo_root)}`",
        f"Probe output root: `{rel(out_root, repo_root)}`",
        "",
        f"- Steps run: **{len(records)}**",
        f"- Passed: **{len(passed)}**",
        f"- Failed / timeout / error: **{len(failed)}**",
        "",
        "## Step table",
        "",
        "| step | status | name | returncode | stdout | stderr |",
        "|---:|---|---|---:|---|---|",
    ]

    for r in records:
        lines.append(
            f"| {r['step']} | {r['status']} | {r['name']} | {r['returncode']} | "
            f"`{rel(Path(r['stdout_log']), repo_root)}` | `{rel(Path(r['stderr_log']), repo_root)}` |"
        )

    lines += ["", "## Latest capstone outputs", ""]

    p10 = latest_dir(out_root, "current_probe10_locality_collision_alias_")
    p11 = latest_dir(out_root, "current_probe11_physical_long_range_alias_")

    if p10:
        lines += [
            f"- Latest Probe 10: `{rel(p10, repo_root)}`",
            f"  - `{rel(p10 / 'alias_comparisons.csv', repo_root)}`",
            f"  - `{rel(p10 / 'PROBE10_LOCALITY_COLLISION_REPORT.md', repo_root)}`",
        ]
    else:
        lines.append("- Latest Probe 10: not found")

    if p11:
        lines += [
            f"- Latest Probe 11: `{rel(p11, repo_root)}`",
            f"  - `{rel(p11 / 'capstone_claim_table.csv', repo_root)}`",
            f"  - `{rel(p11 / 'PROBE11_PHYSICAL_ALIAS_CONFIRMATION_REPORT.md', repo_root)}`",
        ]
    else:
        lines.append("- Latest Probe 11: not found")

    if failed:
        lines += ["", "## Failures", ""]
        for r in failed:
            lines.append(f"- Step {r['step']} `{r['name']}`: status={r['status']} returncode={r['returncode']}")
            if r.get("error"):
                lines.append(f"  - error: `{r['error']}`")

    (run_dir / "RUN_ALL_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def build_steps(args: argparse.Namespace, repo_root: Path) -> List[Dict[str, Any]]:
    py = sys.executable if not args.python else str(Path(args.python).expanduser().resolve())
    probes = repo_root / "probes"
    bgmat = repo_root / "bgmat"
    source = repo_root / SOURCE_DIR_NAME
    out = repo_root / "probe_runs"

    n_local_args: List[str] = []
    for n in args.n_local:
        n_local_args += ["--n-local", str(n)]

    steps: List[Dict[str, Any]] = []

    def script(name: str) -> str:
        return str(find_script(probes, name))

    def add(name: str, filename: str, extra: List[str], timeout: Optional[int]) -> None:
        steps.append({
            "name": name,
            "cmd": [py, script(filename)] + extra,
            "timeout": timeout,
        })

    if not args.skip_patch:
        patch_extra = ["--official-repo", str(bgmat)]
        if args.apply_patch:
            patch_extra.append("--apply")
        add("compatibility patch preflight" + (" apply" if args.apply_patch else " dry-run"),
            "current_patch_compat_modern_deps.py", patch_extra, 120)

    add("probe 01 claim map + repo audit",
        "current_probe_01_claim_map_repo_audit.py",
        ["--out-root", str(out), "--repo-dir", str(bgmat), "--no-clone", "--import-smoke"],
        180)

    add("probe 02 source-data audit",
        "current_probe_02_source_data_audit.py",
        ["--source-dir", str(source), "--out-root", str(out), "--strict-main-figures"],
        300)

    add("probe 03 official repo smoke",
        "current_probe_03_official_repo_smoke.py",
        ["--official-repo", str(bgmat), "--out-root", str(out), "--import-smoke", "--extra-import", "bgmat"],
        300)

    add("probe 04 module seam map + import modules",
        "current_probe_04_module_seam_map.py",
        ["--official-repo", str(bgmat), "--out-root", str(out), "--import-modules"],
        1800)

    add("probe 05 tiny energy/locality seam probe",
        "current_probe_05_tiny_energy_seam_probe.py",
        ["--official-repo", str(bgmat), "--out-root", str(out)],
        900)

    p6_extra = ["--official-repo", str(bgmat), "--out-root", str(out)]
    if args.include_sw_lambda:
        p6_extra.append("--include-sw-lambda")
    add("probe 06 tiny cutoff/locality stress",
        "current_probe_06_tiny_cutoff_locality_stress.py",
        p6_extra,
        900)

    add("probe 07 clean explicit-energy locality witness",
        "current_probe_07_clean_locality_witness.py",
        ["--official-repo", str(bgmat), "--out-root", str(out)],
        600)

    add("probe 08 extracted neighbor graph locality witness",
        "current_probe_08_neighbor_graph_locality_witness_v2.py",
        ["--official-repo", str(bgmat), "--out-root", str(out)] + n_local_args,
        600)

    add("probe 09 joint energy + neighbor locality report",
        "current_probe_09_joint_energy_neighbor_locality_report.py",
        ["--out-root", str(out)],
        180)

    add("probe 10 locality collision / long-range alias",
        "current_probe_10_locality_collision_long_range_alias.py",
        ["--official-repo", str(bgmat), "--out-root", str(out)] + n_local_args,
        600)

    add("probe 11 physical long-range alias confirmation",
        "current_probe_11_physical_long_range_alias_confirmation.py",
        ["--out-root", str(out)],
        300)

    return steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all current-paper locality-boundary probes with recommended settings.")
    p.add_argument("--python", default="", help="Python executable. Default: current interpreter.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue after failed steps.")
    p.add_argument("--skip-patch", action="store_true", help="Skip compatibility patch preflight.")
    p.add_argument("--apply-patch", action="store_true", help="Run compatibility patch with --apply instead of dry-run.")
    p.add_argument("--include-sw-lambda", action="store_true", help="Pass --include-sw-lambda to Probe 06.")
    p.add_argument("--n-local", action="append", type=int, default=[1, 2, 3], help="Repeatable n_local. Default: 1,2,3.")
    p.add_argument("--dry-run", action="store_true", help="Print commands but do not run.")

    # First-run fetch controls.
    p.add_argument("--no-fetch", action="store_true", help="Do not fetch missing bgmat/source-data assets.")
    p.add_argument("--fetch-only", action="store_true", help="Fetch/validate external assets, then exit before running probes.")
    p.add_argument("--force-fetch", action="store_true", help="Delete and refetch bgmat/source-data even if present.")
    p.add_argument("--fetch-timeout", type=int, default=180, help="Download timeout in seconds.")
    p.add_argument("--no-git", action="store_true", help="Do not use git clone; use ZIP fallback for bgmat.")
    p.add_argument("--bgmat-git-url", default=BGMAT_GIT_URL, help="Git URL for official bgmat repo.")
    p.add_argument("--bgmat-ref", default="v1.0.0", help="Git tag/branch to clone first.")
    p.add_argument("--bgmat-zip-url", default="", help="Optional bgmat ZIP URL override.")
    p.add_argument("--source-data-url", default=SOURCE_DATA_URL, help="Source Data ZIP URL.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = repo_root_from_here()
    probes = repo_root / "probes"
    bgmat = repo_root / "bgmat"
    source = repo_root / SOURCE_DIR_NAME
    out = ensure_dir(repo_root / "probe_runs")
    run_dir = ensure_dir(out / f"run_all_probes_{now_stamp()}")
    logs = ensure_dir(run_dir / "logs")

    print("=" * 100)
    print("RUN ALL CURRENT-PAPER LOCALITY-BOUNDARY PROBES")
    print("=" * 100)
    print(f"[REPO ROOT]     {repo_root}")
    print(f"[PROBES DIR]    {probes}")
    print(f"[OFFICIAL REPO] {bgmat}")
    print(f"[SOURCE DATA]   {source}")
    print(f"[OUT ROOT]      {out}")
    print(f"[RUN DIR]       {run_dir}")
    print(f"[PYTHON]        {args.python or sys.executable}")
    print()

    if not probes.exists():
        print(f"[MISSING] probes folder: {probes}")
        return 2

    try:
        ensure_external_assets(repo_root, args)
    except Exception as exc:
        print(f"[FETCH ERROR] {exc!r}")
        print("Re-run with --no-fetch if you want to provide bgmat/source-data manually.")
        return 2

    if args.fetch_only:
        print("[DONE] --fetch-only requested; not running probes.")
        return 0

    missing = []
    for label, path in [("bgmat", bgmat), ("source data", source)]:
        if not path.exists():
            missing.append((label, path))
            print(f"[MISSING] {label}: {path}")
    ok_source, missing_workbooks = source_data_present(source)
    if not ok_source:
        missing.append(("source data workbooks", source))
        print(f"[MISSING] source data workbooks: {missing_workbooks}")

    if not bgmat_present(bgmat):
        missing.append(("bgmat repo contents", bgmat))
        print(f"[MISSING] bgmat/setup.py or bgmat/bgmat missing under: {bgmat}")

    if missing:
        print("\nCannot run until missing paths are fixed.")
        return 2

    steps = build_steps(args, repo_root)
    env = add_pythonpath(os.environ.copy(), repo_root, bgmat)

    write_json(run_dir / "run_manifest.json", {
        "repo_root": str(repo_root),
        "probes": str(probes),
        "bgmat": str(bgmat),
        "source_data": str(source),
        "out_root": str(out),
        "run_dir": str(run_dir),
        "python": args.python or sys.executable,
        "pythonpath": env.get("PYTHONPATH", ""),
        "fetch": {
            "no_fetch": args.no_fetch,
            "force_fetch": args.force_fetch,
            "bgmat_git_url": args.bgmat_git_url,
            "bgmat_ref": args.bgmat_ref,
            "source_data_url": args.source_data_url,
        },
        "steps": steps,
    })

    if args.dry_run:
        print("[DRY RUN]")
        for i, step in enumerate(steps, start=1):
            print(f"{i:02d}. {step['name']}")
            print("    " + ps_cmd(step["cmd"]))
        print(f"\nManifest: {run_dir / 'run_manifest.json'}")
        return 0

    records: List[Dict[str, Any]] = []
    for i, step in enumerate(steps, start=1):
        rec = run_step(i, step["name"], step["cmd"], repo_root, logs, env, step.get("timeout"))
        records.append(rec)

        write_json(run_dir / "run_records.json", records)
        write_csv(run_dir / "run_records.csv", records)
        write_summary(run_dir, repo_root, out, records)

        if rec["status"] != "pass" and not args.continue_on_error:
            print()
            print("[STOP] Step failed. Use --continue-on-error to keep going.")
            print(f"[SUMMARY] {run_dir / 'RUN_ALL_SUMMARY.md'}")
            return 1

    failed = [r for r in records if r["status"] != "pass"]

    print()
    print("=" * 100)
    print("RUN COMPLETE")
    print("=" * 100)
    print(f"Steps run: {len(records)}")
    print(f"Passed:    {len(records) - len(failed)}")
    print(f"Failed:    {len(failed)}")
    print(f"Summary:   {run_dir / 'RUN_ALL_SUMMARY.md'}")
    print(f"Records:   {run_dir / 'run_records.csv'}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
