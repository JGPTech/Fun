# Current Paper Locality-Boundary Audit

This repository is a staged, runnable **mechanism-boundary audit** of a fixed-local-neighborhood Boltzmann-generator code path.

It does **not** claim to reproduce the full paper.

The repo reconstructs only the minimum path needed to ask one specific question:

> What information can a fixed local/top-n neighborhood representation erase when the target physics contains long-range structure?

The capstone result is:

> In controlled toy configurations, extracted local/top-n neighborhood signatures can remain identical while neutral long-range physical target witnesses change.

The identified mechanism is:

> **finite local neighborhood aliasing** — fixed local/top-n neighborhoods can erase long-range physical target information.

---

## What this repo includes

This repository intentionally keeps the upload small and clean.

It includes:

```text
CURRENT_PAPER_LOCALITY_BOUNDARY_PACKAGE/
├── docs/
│   ├── checkpoint_process_report_01-09.md
│   └── checkpoint_process_report_10-11.md
├── probes/
│   ├── current_patch_compat_modern_deps.py
│   ├── current_probe_01_claim_map_repo_audit.py
│   ├── current_probe_02_source_data_audit.py
│   ├── current_probe_03_official_repo_smoke.py
│   ├── current_probe_04_module_seam_map.py
│   ├── current_probe_05_tiny_energy_seam_probe.py
│   ├── current_probe_06_tiny_cutoff_locality_stress.py
│   ├── current_probe_07_clean_locality_witness.py
│   ├── current_probe_08_neighbor_graph_locality_witness_v2.py
│   ├── current_probe_09_joint_energy_neighbor_locality_report.py
│   ├── current_probe_10_locality_collision_long_range_alias.py
│   └── current_probe_11_physical_long_range_alias_confirmation.py
├── README.md
└── run_all_probes_recommended_settings.py
```

It does **not** commit the official `bgmat/` repository or the Nature source-data spreadsheets.

Those external assets are fetched automatically on first run.

---

## What gets downloaded on first run

When `run_all_probes_recommended_settings.py` runs, it checks for these folders:

```text
bgmat/
41467_2026_73900_MOESM3_ESM/
```

If they are missing, the runner fetches them.

After the first run, the working tree may look like this:

```text
CURRENT_PAPER_LOCALITY_BOUNDARY_PACKAGE/
├── 41467_2026_73900_MOESM3_ESM/
│   ├── Figure2.xlsx
│   ├── Figure3.xlsx
│   ├── Figure4.xlsx
│   ├── Figure5.xlsx
│   ├── Figure6.xlsx
│   ├── Figure7.xlsx
│   ├── SupplementaryFigure1.xlsx
│   ├── SupplementaryFigure2.xlsx
│   ├── SupplementaryFigure3.xlsx
│   ├── SupplementaryFigure4.xlsx
│   └── SupplementaryFigure5.xlsx
├── bgmat/
│   ├── bgmat/
│   ├── figs/
│   ├── tutorial/
│   ├── LICENSE
│   ├── README.md
│   └── setup.py
├── docs/
├── probes/
├── probe_runs/
├── _downloads/
├── README.md
└── run_all_probes_recommended_settings.py
```

Generated folders are not meant to be committed.

---

## First read

Read these in order:

1. [`docs/checkpoint_process_report_01-09.md`](docs/checkpoint_process_report_01-09.md)
2. [`docs/checkpoint_process_report_10-11.md`](docs/checkpoint_process_report_10-11.md)
3. This `README.md`
4. The probe scripts in [`probes/`](probes/)

The checkpoint reports explain the full audit path and the claim boundary.

---

## Correct interpretation

The capstone supports this limited claim:

> A fixed local/top-n representation can alias a controlled long-range physical target.

It does **not** support these stronger claims:

- the paper is wrong;
- the full Boltzmann generator fails;
- the benchmark systems necessarily contain this alias;
- the reported figures, ESS, TFEP, transfer, or free-energy results are invalid;
- this repository reproduces the paper.

This repository identifies a **mechanism-boundary condition**, not a benchmark failure.

---

## Why this matters

The current paper relies on local environments to scale Boltzmann generators to large systems.

These probes identify a specific boundary condition:

> When the target observable depends on information outside the fixed local neighborhood, a fixed local/top-n representation can assign identical local signatures to physically different states.

The suggested correction target is conditional:

> For systems where long-range physics matters, the architecture should include or verify a nonlocal, multiscale, reciprocal-space, global-conditioning, or long-range interaction channel — or include diagnostics showing that local aliases are irrelevant for the claimed target observables.

---

## Requirements

Recommended environment:

```text
Python 3.12
Git
Internet access for the first run
```

The runner fetches external code/data, but it does not magically fix every Python environment. If dependency imports fail on a fresh machine, use the compatibility patch workflow below.

---

## Quick start: one-command audit

From the repository root:

```powershell
python .\run_all_probes_recommended_settings.py
```

On first run, this will:

```text
1. fetch bgmat/ if missing
2. fetch 41467_2026_73900_MOESM3_ESM/ if missing
3. run compatibility patch preflight in dry-run mode
4. run Probes 01-11
5. write outputs under probe_runs/
```

A successful full run ends with:

```text
RUN COMPLETE
Steps run: 12
Passed:    12
Failed:    0
```

---

## First-run fetch only

To download/validate external assets without running probes:

```powershell
python .\run_all_probes_recommended_settings.py --fetch-only
```

To force a clean re-download:

```powershell
python .\run_all_probes_recommended_settings.py --force-fetch --fetch-only
```

To disable fetching because you placed the assets manually:

```powershell
python .\run_all_probes_recommended_settings.py --no-fetch
```

---

## Dry run

To inspect the planned commands without running the probes:

```powershell
python .\run_all_probes_recommended_settings.py --dry-run
```

To continue through later probes after a nonzero return code:

```powershell
python .\run_all_probes_recommended_settings.py --continue-on-error
```

---

## Dependency setup for a fresh environment

If imports fail, first fetch the external assets:

```powershell
python .\run_all_probes_recommended_settings.py --fetch-only
```

Then write the modern Python 3.12 compatibility requirements:

```powershell
python .\probes\current_patch_compat_modern_deps.py --official-repo .\bgmat --apply
```

Then install:

```powershell
python -m pip install -U -r .\requirements_modern_py312.txt
```

Then rerun:

```powershell
python .\run_all_probes_recommended_settings.py
```

This is a modern compatibility path, not an exact frozen-environment reproduction.

---

## What the runner executes

The recommended runner executes:

```text
compatibility patch preflight, dry-run
Probe 01  claim map + repo audit
Probe 02  source-data audit
Probe 03  official repo smoke/import planner
Probe 04  module/seam map with import tests
Probe 05  tiny energy/locality seam probe
Probe 06  tiny cutoff/locality stress
Probe 07  clean explicit-energy locality witness
Probe 08  extracted neighbor-graph locality witness
Probe 09  joint locality report
Probe 10  locality collision / long-range alias witness
Probe 11  physical long-range alias confirmation
```

The runner writes outputs under:

```text
probe_runs/
```

and writes its own run summary under:

```text
probe_runs/run_all_probes_<timestamp>/
├── RUN_ALL_SUMMARY.md
├── run_manifest.json
├── run_records.csv
├── run_records.json
└── logs/
```

---

## Minimal capstone rerun

To rerun only the final mechanism-boundary result, run Probes 10 and 11 from the repository root.

### Probe 10 — Locality collision / long-range alias witness

```powershell
python .\probes\current_probe_10_locality_collision_long_range_alias.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs `
  --n-local 1 `
  --n-local 2 `
  --n-local 3
```

Probe 10 asks:

> Can globally different configurations have identical extracted local/top-n signatures?

The key output is:

```text
probe_runs\current_probe10_locality_collision_alias_<timestamp>\alias_comparisons.csv
```

The important condition is:

```text
local_graph_same == True
local_distance_shell_same == True
internal_cluster_signature_same == True
both_topn_graphs_internal_only == True
long_range_delta_nonzero == True
alias_detected == True
```

### Probe 11 — Physical long-range alias confirmation

```powershell
python .\probes\current_probe_11_physical_long_range_alias_confirmation.py `
  --out-root .\probe_runs
```

Probe 11 auto-detects the latest Probe 10 output.

Probe 11 asks:

> Do those local aliases also differ under neutral long-range physical target witnesses?

The key output is:

```text
probe_runs\current_probe11_physical_long_range_alias_<timestamp>\capstone_claim_table.csv
```

Expected capstone result:

```text
P11-C01 PASS
P11-C02 PASS
P11-C03 PASS
P11-C04 SUPPORTED_IN_CONTROLLED_WITNESS
P11-C05 INFO
```

---

## Full manual probe walkthrough

The recommended path is to run `run_all_probes_recommended_settings.py`.

The commands below show the equivalent manual sequence from the repository root, after external assets are already fetched.

### Probe 01 — Claim map and repo audit

```powershell
python .\probes\current_probe_01_claim_map_repo_audit.py `
  --out-root .\probe_runs `
  --repo-dir .\bgmat `
  --no-clone `
  --import-smoke
```

Purpose:

```text
Build a claim ledger and verify that the official repository is present.
```

This probe does not train or reproduce the paper.

---

### Probe 02 — Source-data audit

```powershell
python .\probes\current_probe_02_source_data_audit.py `
  --source-dir .\41467_2026_73900_MOESM3_ESM `
  --out-root .\probe_runs `
  --strict-main-figures
```

Purpose:

```text
Inventory the source-data spreadsheets and map figure data to claim targets.
```

This probe does not test the model.

Expected source-data files:

```text
Figure2.xlsx
Figure3.xlsx
Figure4.xlsx
Figure5.xlsx
Figure6.xlsx
Figure7.xlsx
SupplementaryFigure1.xlsx
SupplementaryFigure2.xlsx
SupplementaryFigure3.xlsx
SupplementaryFigure4.xlsx
SupplementaryFigure5.xlsx
```

---

### Probe 03 — Official repo smoke/import planner

```powershell
python .\probes\current_probe_03_official_repo_smoke.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs `
  --import-smoke `
  --extra-import bgmat
```

Purpose:

```text
Audit official repo structure, candidate commands, dependency metadata, and import readiness.
```

The local setup used here is a modern compatibility sandbox, not exact frozen-environment reproduction.

---

### Probe 04 — Module/seam map

```powershell
python .\probes\current_probe_04_module_seam_map.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs `
  --import-modules
```

Purpose:

```text
Map energy, locality, GNN, marginal, augmentation, OpenMM, and free-energy seams in the codebase.
```

---

### Probe 05 — Tiny energy/locality seam probe

```powershell
python .\probes\current_probe_05_tiny_energy_seam_probe.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs
```

Purpose:

```text
Find usable tiny system-energy paths and identify the full-model dependency gate.
```

Important result:

```text
full GNN/config path is gated by e3nn_jax
```

---

### Probe 06 — Tiny cutoff/locality stress

```powershell
python .\probes\current_probe_06_tiny_cutoff_locality_stress.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs
```

Purpose:

```text
Stress-test explicit Lennard-Jones cutoff behavior.
```

This establishes that the explicit local energy path behaves locally.

---

### Probe 07 — Clean explicit-energy locality witness

```powershell
python .\probes\current_probe_07_clean_locality_witness.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs
```

Purpose:

```text
Cleanly separate far/blind third-particle cases from inside/local third-particle cases.
```

Expected pattern:

```text
outside local cutoff → energy unchanged
inside local cutoff  → energy changes
```

---

### Probe 08 — Extracted neighbor-graph locality witness

```powershell
python .\probes\current_probe_08_neighbor_graph_locality_witness_v2.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs `
  --n-local 1 `
  --n-local 2 `
  --n-local 3
```

Purpose:

```text
Extract and test the model-side top-n neighbor selector without requiring e3nn_jax.
```

Expected pattern:

```text
far node  → existing local graph unchanged for comparable n_local
near node → local graph rewires
```

---

### Probe 09 — Joint locality report

```powershell
python .\probes\current_probe_09_joint_energy_neighbor_locality_report.py `
  --out-root .\probe_runs
```

Purpose:

```text
Combine Probe 07 and Probe 08 into a single locality-control report.
```

This is still a control, not the capstone.

---

### Probe 10 — Locality collision / long-range alias witness

```powershell
python .\probes\current_probe_10_locality_collision_long_range_alias.py `
  --official-repo .\bgmat `
  --out-root .\probe_runs `
  --n-local 1 `
  --n-local 2 `
  --n-local 3
```

Purpose:

```text
Construct paired configurations with identical local/top-n signatures but different long-range/global witnesses.
```

This is where the audit moves from:

```text
locality works
```

to:

```text
locality can erase information
```

---

### Probe 11 — Physical long-range alias confirmation

```powershell
python .\probes\current_probe_11_physical_long_range_alias_confirmation.py `
  --out-root .\probe_runs
```

Purpose:

```text
Attach neutral Coulomb-like physical long-range target witnesses to the Probe 10 alias pairs.
```

This is the capstone probe.

---

## Capstone claim table

| claim_id | result | claim | evidence |
| --- | --- | --- | --- |
| P11-C01 | PASS | Probe 10 produced local/top-n alias rows suitable for physical confirmation. | 12/12 Probe 10 comparison rows had alias_detected=True. |
| P11-C02 | PASS | At least one local/top-n alias pair is separated by a neutral Coulomb-like physical witness. | 36/36 charge-model confirmation rows had physical_alias_confirmed=True. |
| P11-C03 | PASS | Every Probe 10 alias pair/n_local group is physically separated by at least one neutral long-range witness. | 12/12 alias pair/n_local groups had physical_alias_confirmed_any_model=True. |
| P11-C04 | SUPPORTED_IN_CONTROLLED_WITNESS | Specific mechanism identified: finite local/top-n neighborhoods can erase long-range physical target information. | Rows require Probe10 alias_detected=True plus a nonzero primary neutral long-range physical delta. |
| P11-C05 | INFO | Paper implication is conditional. | This probe does not train or run the full BG; it provides a minimal counterexample class for fixed local representations. |

---

## Troubleshooting

### Fetch fails

The first run needs internet access.

To fetch only:

```powershell
python .\run_all_probes_recommended_settings.py --fetch-only
```

To retry from scratch:

```powershell
python .\run_all_probes_recommended_settings.py --force-fetch --fetch-only
```

To provide assets manually instead:

```powershell
python .\run_all_probes_recommended_settings.py --no-fetch
```

Manual asset names expected:

```text
bgmat/
41467_2026_73900_MOESM3_ESM/
```

### `bgmat` import problems

The recommended runner sets `PYTHONPATH` for subprocesses automatically.

Manual runs that need import smoke may require:

```powershell
$env:PYTHONPATH = (Resolve-Path .\bgmat).Path
```

Then rerun the probe.

### `e3nn_jax` missing

That is expected for the minimal probe path.

Probes 08, 10, and 11 avoid the full `e3nn_jax` dependency by extracting only the neighbor-selection helper needed for the locality-boundary test.

This is why the result is described as:

```text
extracted local/top-n representation
```

not:

```text
full model reproduction
```

### Probe 11 cannot find Probe 10

Run Probe 10 first, or pass the Probe 10 folder explicitly:

```powershell
python .\probes\current_probe_11_physical_long_range_alias_confirmation.py `
  --probe10-dir .\probe_runs\current_probe10_locality_collision_alias_<timestamp> `
  --out-root .\probe_runs
```

---

## Generated outputs

The recommended runner writes output folders under:

```text
probe_runs/
```

Typical output folders:

```text
probe_runs/current_probe01_<timestamp>/
probe_runs/current_probe02_source_data_audit_<timestamp>/
probe_runs/current_probe03_official_smoke_<timestamp>/
probe_runs/current_probe04_module_seams_<timestamp>/
probe_runs/current_probe05_tiny_energy_seams_<timestamp>/
probe_runs/current_probe06_tiny_cutoff_locality_<timestamp>/
probe_runs/current_probe07_clean_locality_witness_<timestamp>/
probe_runs/current_probe08_neighbor_graph_locality_<timestamp>/
probe_runs/current_probe09_joint_locality_report_<timestamp>/
probe_runs/current_probe10_locality_collision_alias_<timestamp>/
probe_runs/current_probe11_physical_long_range_alias_<timestamp>/
probe_runs/run_all_probes_<timestamp>/
```

The most important generated files are:

```text
Probe 10:
  alias_comparisons.csv
  PROBE10_LOCALITY_COLLISION_REPORT.md

Probe 11:
  capstone_claim_table.csv
  physical_alias_confirmations.csv
  PROBE11_PHYSICAL_ALIAS_CONFIRMATION_REPORT.md

Runner:
  RUN_ALL_SUMMARY.md
  run_records.csv
```

---

## Suggested `.gitignore`

```gitignore
bgmat/
41467_2026_73900_MOESM3_ESM/
_downloads/
probe_runs/
__pycache__/
*.pyc
.venv/
venv/
~$*.xlsx
COMPATIBILITY_PATCH_REPORT.md
requirements_modern_py312.txt
```

---

## Bottom line

This repository is a clean mechanism-boundary audit.

The final result is not:

```text
the paper is wrong
```

The final result is:

```text
fixed local/top-n neighborhoods can alias neutral long-range physical target information in controlled witnesses
```

That is the specific mechanism future full-model tests should check.
