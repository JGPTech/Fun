#!/usr/bin/env python
"""Resumable staged runner for run_unified_pipeline.py.

The suite is ordered from cheap integration checks to a paper-scale run.  Each
test writes into its own output directory and the runner updates a manifest
before and after each test, so completed tests remain archived if the process is
interrupted.  The current suite exercises causal insertion decay: lag-slope
memory decay and a hybrid T_ins/chi-guided thermal envelope are recorded as
separate factors, while finite-clock readiness remains a K2 anchor diagnostic
rather than an observable multiplier.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_AUTHOR_NPZ = "data/fig3/data_better.npz"
DEFAULT_AUTHOR_TEMP_KEY = "tem"
DEFAULT_AUTHOR_OBS_KEY = "M_MC"
DEFAULT_TEMPS = "0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96"


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def linspace_csv(start: float, stop: float, count: int) -> str:
    if count <= 1:
        return f"{start:.6f}"
    step = (stop - start) / (count - 1)
    return ",".join(f"{start + step * i:.6f}" for i in range(count))


@dataclass(frozen=True)
class TestSpec:
    test_id: int
    name: str
    description: str
    args: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.test_id:02d}_{self.name}"


def build_suite() -> list[TestSpec]:
    common = {
        "Psi-scale-pi": 0.85,
        "proposal-id": 2,
        "proposal-delta-rad": 0.1,
        "window-radius": 1,
        "init": "ordered",
        "seed": 1234,
        "forward-seed": 987654,
        "threads-per-block": 128,
        "tap-lags": "1,2,3",
        "tap-weights": "0.5,0.3,0.2",
        "beta-pi": 0.5,
        "null-scale": 0.49,
    }
    return [
        TestSpec(1, "q512_cuda_smoke", "Cheap CUDA smoke gate with stable non-edge selector bracket.", {
            **common, "L": 8, "q": 512, "n-traj": 16, "repeats": 1,
            "burn-steps": 30_000, "sample-steps": 12_000, "sample-stride": 1_000,
            "checkpoint-steps": 3_000, "forward-steps": 8_000, "forward-sample-stride": 1_000,
            "temps": "0.40,0.64,0.72,0.84,0.96", "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(2, "q512_short", "Short q=512 finite-clock validation.", {
            **common, "L": 8, "q": 512, "n-traj": 16, "repeats": 1,
            "burn-steps": 60_000, "sample-steps": 22_000, "sample-stride": 1_000,
            "checkpoint-steps": 5_500, "forward-steps": 10_000, "forward-sample-stride": 1_000,
            "temps": "0.40,0.64,0.72,0.84,0.96", "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(3, "q512_smoke_insertion_decay", "Smoke test for explicit lag and thermal-envelope decay fields.", {
            **common, "L": 8, "q": 512, "n-traj": 32, "repeats": 1,
            "burn-steps": 120_000, "sample-steps": 44_000, "sample-stride": 2_000,
            "checkpoint-steps": 11_000, "forward-steps": 20_000, "forward-sample-stride": 1_000,
            "temps": DEFAULT_TEMPS, "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(4, "q512_mid_freeze", "Mid-depth version of the frozen candidate.", {
            **common, "L": 8, "q": 512, "n-traj": 64, "repeats": 1,
            "burn-steps": 300_000, "sample-steps": 110_000, "sample-stride": 2_000,
            "checkpoint-steps": 11_000, "forward-steps": 30_000, "forward-sample-stride": 1_000,
            "temps": DEFAULT_TEMPS, "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(5, "q512_frozen_candidate", "Current frozen L=8/q=512 evidence depth.", {
            **common, "L": 8, "q": 512, "n-traj": 64, "repeats": 1,
            "burn-steps": 600_000, "sample-steps": 220_000, "sample-stride": 2_000,
            "checkpoint-steps": 22_000, "forward-steps": 50_000, "forward-sample-stride": 1_000,
            "temps": DEFAULT_TEMPS, "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(6, "high_temp_bracket", "Raw thermodynamic bracket above the current grid.", {
            **common, "L": 8, "q": 512, "n-traj": 64, "repeats": 1,
            "burn-steps": 600_000, "sample-steps": 220_000, "sample-stride": 2_000,
            "checkpoint-steps": 22_000, "forward-steps": 50_000, "forward-sample-stride": 1_000,
            "temps": "0.72,0.84,0.96,1.08,1.20,1.32,1.44", "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(7, "q1024_continuum_check", "q=1024 finite-clock continuum check at L=8.", {
            **common, "L": 8, "q": 1024, "n-traj": 64, "repeats": 1,
            "burn-steps": 300_000, "sample-steps": 110_000, "sample-stride": 2_000,
            "checkpoint-steps": 11_000, "forward-steps": 30_000, "forward-sample-stride": 1_000,
            "temps": DEFAULT_TEMPS, "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(8, "L16_medium", "First larger-lattice check with explicit lag and thermal-envelope decay.", {
            **common, "L": 16, "q": 512, "n-traj": 64, "repeats": 1,
            "burn-steps": 2_000_000, "sample-steps": 800_000, "sample-stride": 5_000,
            "checkpoint-steps": 80_000, "forward-steps": 100_000, "forward-sample-stride": 2_000,
            "temps": DEFAULT_TEMPS, "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(9, "L32_deep", "Deep pre-paper lattice run; intended overnight-class test.", {
            **common, "L": 32, "q": 512, "n-traj": 100, "repeats": 1,
            "burn-steps": 20_000_000, "sample-steps": 8_000_000, "sample-stride": 20_000,
            "checkpoint-steps": 800_000, "forward-steps": 250_000, "forward-sample-stride": 5_000,
            "temps": DEFAULT_TEMPS, "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
        TestSpec(10, "paper_settings", "Paper-scale L=100, n=100, 60-temperature grid, 200L^4 iteration schedule; q=1024 finite-clock proxy.", {
            **common, "L": 100, "q": 1024, "n-traj": 100, "repeats": 1,
            "burn-steps": 15_000_000_000, "burn-chunk-steps": 1_000_000_000,
            "sample-steps": 5_000_000_000, "sample-stride": 100_000,
            "checkpoint-steps": 500_000_000, "forward-steps": 1_000_000, "forward-sample-stride": 20_000,
            "temps": linspace_csv(0.1, 1.2, 60), "rho": 0.10, "lambda-safety-factor": 3.0,
        }),
    ]


def parse_test_selection(selection: str | None, max_id: int) -> list[int]:
    if selection is None or selection.strip().lower() in {"", "all"}:
        return list(range(1, max_id + 1))
    selected: set[int] = set()
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    bad = sorted(i for i in selected if i < 1 or i > max_id)
    if bad:
        raise ValueError(f"Unknown test id(s): {bad}; valid range is 1-{max_id}")
    return sorted(selected)


def count_temps(spec: TestSpec) -> int:
    return len([part for part in str(spec.args["temps"]).split(",") if part.strip()])


def progress_bar(done: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * min(done, total) / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def command_for_test(
    *,
    python_exe: str,
    pipeline: Path,
    kernel: Path,
    test_out: Path,
    spec: TestSpec,
    author_npz: Path | None,
    author_temp_key: str,
    author_obs_key: str,
) -> list[str]:
    cmd = [python_exe, "-u", str(pipeline), "--kernel", str(kernel), "--out", str(test_out)]
    for key, value in spec.args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    if author_npz is not None:
        cmd.extend([
            "--author-npz", str(author_npz),
            "--author-temp-key", author_temp_key,
            "--author-observable-key", author_obs_key,
        ])
    return cmd


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at_utc": now_utc(), "tests": {}}


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)


def find_evidence(test_out: Path) -> str | None:
    candidates = sorted(test_out.glob("unified_main_pipeline/*/unified_evidence_*.json"))
    return str(candidates[-1]) if candidates else None


def run_one_test(
    *,
    spec: TestSpec,
    selected_index: int,
    selected_total: int,
    command: list[str],
    test_out: Path,
    log_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> int:
    test_key = str(spec.test_id)
    temp_total = count_temps(spec)
    temp_done = 0
    entry = {
        "id": spec.test_id,
        "name": spec.name,
        "description": spec.description,
        "status": "running",
        "started_at_utc": now_utc(),
        "out_dir": str(test_out),
        "log": str(log_path),
        "command": command,
    }
    manifest["tests"][test_key] = entry
    save_manifest(manifest_path, manifest)

    print()
    print(f"Series {progress_bar(selected_index - 1, selected_total)}")
    print(f"Starting test {spec.test_id}/{selected_total}: {spec.name}")
    print(f"Output: {test_out}")
    print(f"Log:    {log_path}")
    print(f"Temps:  {temp_total}")

    test_out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    temp_line_re = re.compile(r"^\[UNIFIED\]\s+T=")

    proc: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write("$ " + " ".join(command) + "\n\n")
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                if temp_line_re.match(line):
                    temp_done += 1
                    print(f"Test {spec.test_id} temps {progress_bar(temp_done, temp_total)}", flush=True)
            returncode = proc.wait()
    except KeyboardInterrupt:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        entry.update({
            "status": "interrupted",
            "finished_at_utc": now_utc(),
            "returncode": None,
            "evidence": find_evidence(test_out),
        })
        save_manifest(manifest_path, manifest)
        print(f"\nInterrupted during test {spec.test_id}. Completed previous tests are preserved.")
        raise

    evidence = find_evidence(test_out)
    entry.update({
        "status": "completed" if returncode == 0 else "failed",
        "finished_at_utc": now_utc(),
        "returncode": returncode,
        "evidence": evidence,
    })
    save_manifest(manifest_path, manifest)
    print(f"Finished test {spec.test_id}: {entry['status']} {progress_bar(selected_index, selected_total)}")
    if evidence:
        print(f"Evidence: {evidence}")
    return returncode


def main() -> int:
    suite = build_suite()
    ap = argparse.ArgumentParser(description="Run staged unified-pipeline tests with resumable progress.")
    ap.add_argument("selection", nargs="*", help='Optional shorthand: "tests 8-10" or "1,3,5-7".')
    ap.add_argument("--tests", default=None, help='Test ids/ranges, e.g. "8-10" or "1,3,5-7". Defaults to all.')
    ap.add_argument("--series-dir", default=None, help="Existing/new series directory. Enables organized resume.")
    ap.add_argument("--out-root", default="analysis/unified_test_series", help="Root directory for new series.")
    ap.add_argument("--kernel", default="kernel/kernel_finite_clock.cu")
    ap.add_argument("--pipeline", default="run_unified_pipeline.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--author-npz", default=DEFAULT_AUTHOR_NPZ)
    ap.add_argument("--author-temp-key", default=DEFAULT_AUTHOR_TEMP_KEY)
    ap.add_argument("--author-observable-key", default=DEFAULT_AUTHOR_OBS_KEY)
    ap.add_argument("--no-author", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Skip completed tests in the selected series directory.")
    ap.add_argument("--continue-on-failure", action="store_true", help="Continue to later tests after a failure.")
    ap.add_argument("--dry-run", action="store_true", help="Print selected tests and commands without running.")
    args = ap.parse_args()

    selection = args.tests
    if selection is None and args.selection:
        if args.selection[0].lower() == "tests":
            if len(args.selection) < 2:
                raise SystemExit('Expected a range after "tests", for example: tests 8-10')
            selection = args.selection[1]
        else:
            selection = args.selection[0]
    selected_ids = parse_test_selection(selection, max(spec.test_id for spec in suite))
    specs_by_id = {spec.test_id: spec for spec in suite}
    selected_specs = [specs_by_id[i] for i in selected_ids]

    series_dir = Path(args.series_dir) if args.series_dir else Path(args.out_root) / f"series_{stamp()}"
    manifest_path = series_dir / "series_manifest.json"
    manifest = load_manifest(manifest_path)
    manifest.update({
        "series_dir": str(series_dir),
        "updated_at_utc": now_utc(),
        "selected_tests": selected_ids,
        "runner_version": 9,
    })
    save_manifest(manifest_path, manifest)

    pipeline = Path(args.pipeline)
    kernel = Path(args.kernel)
    author_npz = None if args.no_author else Path(args.author_npz)

    print(f"Series directory: {series_dir}")
    print(f"Manifest:         {manifest_path}")
    print(f"Selected tests:   {','.join(str(i) for i in selected_ids)}")
    print("Use Ctrl+C to stop; completed test folders and manifest entries are preserved.")

    for index, spec in enumerate(selected_specs, start=1):
        test_out = series_dir / spec.label
        log_path = test_out / f"{spec.label}.log"
        existing = manifest.get("tests", {}).get(str(spec.test_id), {})
        if args.resume and existing.get("status") == "completed" and existing.get("returncode") == 0:
            evidence = existing.get("evidence")
            if not evidence or Path(evidence).exists():
                print(f"Skipping completed test {spec.test_id}: {spec.name}")
                continue
        cmd = command_for_test(
            python_exe=args.python,
            pipeline=pipeline,
            kernel=kernel,
            test_out=test_out,
            spec=spec,
            author_npz=author_npz,
            author_temp_key=args.author_temp_key,
            author_obs_key=args.author_observable_key,
        )
        if args.dry_run:
            print()
            print(f"Test {spec.test_id}: {spec.name}")
            print(spec.description)
            print(" ".join(cmd))
            continue
        try:
            returncode = run_one_test(
                spec=spec,
                selected_index=index,
                selected_total=len(selected_specs),
                command=cmd,
                test_out=test_out,
                log_path=log_path,
                manifest_path=manifest_path,
                manifest=manifest,
            )
        except KeyboardInterrupt:
            return 130
        if returncode != 0 and not args.continue_on_failure:
            print(f"Stopping after failed test {spec.test_id}. Use --continue-on-failure to run later tests anyway.")
            return returncode

    manifest["updated_at_utc"] = now_utc()
    save_manifest(manifest_path, manifest)
    print()
    print(f"Series complete or dry-run complete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
