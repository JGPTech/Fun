# CHECKPOINT PROCESS REPORT 01–09

**Project:** Current paper sandbox — `cool_paper / bgmat`  
**Paper under audit:** *Scalable Boltzmann generators for equilibrium sampling of large-scale materials*  
**Checkpoint file:** `checkpoint_process_report_01-09.md`  
**Status:** mechanism-level audit checkpoint, not paper reproduction  
**Date:** 2026-06-13

---

## 0. Executive checkpoint

This checkpoint covers Probes 01–09.

The current result should be described carefully:

> We did **not** reproduce the paper.  
> We tested an **adjacent code-level locality witness** that supports one mechanism the paper relies on: finite local neighborhoods.

What we have demonstrated so far:

1. The official repository can be located and imported under a modern compatibility sandbox.
2. The source-data spreadsheets can be inventoried and mapped to paper claims.
3. The official repo has old/frozen dependency metadata that is not Python 3.12 friendly.
4. A modern dependency path can import the core package and system-energy modules.
5. The explicit Lennard-Jones energy implementation has hard cutoff behavior.
6. A clean far/near perturbation witness confirms locality at the energy level.
7. An extracted model-side neighbor selector shows stable behavior under far perturbations and rewiring under near perturbations for comparable unsaturated `n_local` values.
8. Saturated `n_local` settings need separate interpretation because they enter an all/self-slot regime.

The correct claim is:

> Current evidence supports a **mechanism-level locality witness** adjacent to the paper’s core architecture. It does not validate paper-level training, sampling, transfer, ESS, TFEP, figure reproduction, or full Boltzmann-generator performance claims.

---

## 1. Scope and claim boundary

### What this checkpoint supports

This checkpoint supports a narrow, defensible statement:

> The codebase contains explicit local/cutoff mechanisms and finite-neighbor graph mechanisms. In small controlled witnesses, those mechanisms behave as locality predicts: far perturbations can be invisible/stable, while near perturbations can alter energy or rewire the neighbor graph.

### What this checkpoint does **not** support yet

This checkpoint does **not** establish:

- reproduction of paper figures;
- reproduction of training results;
- correctness of learned Boltzmann generator samples;
- transfer from small to large systems;
- ESS trends with system size;
- TFEP or free-energy estimates;
- wall-clock or energy-evaluation scaling claims;
- NPT phase behavior;
- complete import/use of the full GNN/flow stack with `e3nn_jax`;
- validation of model architecture performance.

This is a seam audit and mechanism witness, not a reproduction claim.

---

## 2. Environment and setup status

### Local project layout

Working root:

```text
C:\atomstuff\cool_paper
```

Expected layout:

```text
cool_paper/
  Figure2.xlsx ... Figure7.xlsx
  SupplementaryFigure1.xlsx ... SupplementaryFigure5.xlsx
  current_probe_01_claim_map_repo_audit.py
  current_probe_02_source_data_audit.py
  current_probe_03_official_repo_smoke.py
  current_patch_compat_modern_deps.py
  current_probe_04_module_seam_map.py
  current_probe_05_tiny_energy_seam_probe.py
  current_probe_06_tiny_cutoff_locality_stress.py
  current_probe_07_clean_locality_witness.py
  current_probe_08_neighbor_graph_locality_witness_v2.py
  current_probe_09_joint_energy_neighbor_locality_report.py
  bgmat/
  probe_runs/
```

Recommended `.gitignore`:

```gitignore
probe_runs/
current_probe_runs/
__pycache__/
*.pyc
.venv/
venv/
~$*.xlsx
```

### Compatibility status

The official repository dependency pins were not cleanly compatible with the active Python 3.12 environment. We treated the local setup as a **modern compatibility sandbox**, not an exact frozen-environment reproduction.

Important dependency observations:

- old editable install path tried to pull incompatible/frozen pins;
- `numpy==1.25.2` failed under Python 3.12 tooling;
- core package import works through repo-relative `PYTHONPATH`;
- core energy/system modules can import;
- full model/GNN config path is gated by missing `e3nn_jax`.

---

## 3. Probe summary table

| Probe | Purpose | Result | Status |
|---|---|---:|---|
| Probe 01 | Claim map and repo audit | Official repo found; claim map established | ✅ complete |
| Probe 02 | Source-data spreadsheet audit | Figure/source-data inventory and claim targets mapped | ✅ complete |
| Probe 03 | Official repo smoke/import planner | Core imports green after modern compatibility path | ✅ complete |
| Patch | Modern dependency compatibility | Avoided old pins; modern setup path established | ✅ complete |
| Probe 04 | Module/seam map | Located energy, locality, marginal, GNN, OpenMM seams | ✅ complete |
| Probe 05 | Tiny energy/locality seam probe | Identified usable system-energy modules and `e3nn_jax` gate | ✅ complete |
| Probe 06 | Tiny cutoff/locality stress | LJ cutoff hard-locality confirmed; mW/SW smoke usable | ✅ complete |
| Probe 07 | Clean energy locality witness | Far outside cutoff leaves energy unchanged; near inside cutoff changes energy | ✅ complete |
| Probe 08 | Neighbor graph locality witness | Extracted top-n neighbor selector; far/near behavior observed | ✅ complete |
| Probe 09 | Joint locality report | Energy + neighbor witness combined; saturated cases separated | ✅ complete |

---

## 4. Probe-by-probe record

## Probe 01 — Claim map + official repo audit

**Script:** `current_probe_01_claim_map_repo_audit.py`

### Goal

Create a paper claim ledger and audit the official repository without training.

### Key results

- Official repo found.
- Repo path confirmed under local project.
- Claim map created for paper-level claims.
- Initial import smoke identified missing packages before dependency fixes.

### Claim IDs established

| Claim ID | Theme |
|---|---|
| C01 | Architecture scalability |
| C02 | Energy-based training without target samples |
| C03 | Local rather than full configuration |
| C04 | Augmented flow preserves 3D distances |
| C05 | Linear scaling with fixed neighbors |
| C06 | Displacements from ideal lattice |
| C07 | Marginal density with auxiliary samples |
| C08 | TFEP |
| C09 | Size transfer systems |
| C10 | RDF/energy histograms without reweighting |
| C11 | ESS decreases with system size |
| C12 | Local beats global cost |
| C13 | System-dependent performance |
| C14 | Low-ESS warning |
| C15 | Energy-evaluation vs wallclock caveat |
| C16 | NPT phase behavior |
| C17 | Long-range limitation |
| C18 | Molecular crystals/orientation limitation |

### Checkpoint interpretation

Probe 01 established the audit frame. It did not test scientific claims yet.

---

## Probe 02 — Source-data audit

**Script:** `current_probe_02_source_data_audit.py`

### Goal

Audit the source-data spreadsheets and map paper figures to claims.

### Output family

```text
PROBE02_SUMMARY.md
source_data_inventory.csv
workbook_sheet_summary.csv
workbook_column_summary.csv
figure_claim_targets.md
figure_claim_targets.json
sheet_csv/*.csv
plots/*.png
parsed_workbooks.json
parsed_cells.json
```

### Figure-to-claim mapping

| Figure | Mapped theme |
|---|---|
| Figure 2 | RDF/energy histograms, transferred large systems, no reweighting |
| Figure 3 | ESS versus training/local-global comparison |
| Figure 4 | Helmholtz free energies and cubic-vs-hex mW |
| Figure 5 | Gibbs free energy / density versus LJ cutoff radius |
| Figure 6 | Stillinger-Weber `lambda3` phase diagram |
| Figure 7 | Silicon temperature/pressure phase diagram |

### Checkpoint interpretation

Probe 02 established source-data structure and claim targets. It did not reproduce figures yet.

---

## Probe 03 — Official repo smoke/import planner

**Script:** `current_probe_03_official_repo_smoke.py`

### Goal

Inspect official repo structure, dependency metadata, candidate commands, and import readiness.

### Important results

Initial run showed missing core imports:

```text
jax
jaxlib
haiku
distrax
openmm
```

After modern compatibility setup, import smoke became green:

```text
missing_core_imports : none
```

TensorFlow Probability/JAX backend warnings appeared but were non-fatal.

### Important caveat

This became a modern compatibility sandbox:

```text
modern compatibility sandbox ≠ exact frozen-environment reproduction
```

### Checkpoint interpretation

Probe 03 established that the codebase could be imported and inspected after dependency adjustment. It did not run training or validate paper outputs.

---

## Compatibility patch — Modern dependency path

**Script:** `current_patch_compat_modern_deps.py`

### Goal

Avoid old dependency pins that break under Python 3.12 and create a modern compatibility path.

### Compatibility path

```powershell
python -m pip install -U -r .\requirements_modern_py312.txt
python -m pip install --no-build-isolation --no-deps -e .\bgmat
python .\current_probe_03_official_repo_smoke.py --import-smoke --extra-import bgmat
```

### Key stance

Do not claim exact reproduction from this environment. It is a practical modern sandbox for mechanism probing.

---

## Probe 04 — Module/seam map

**Script:** `current_probe_04_module_seam_map.py`

### Goal

Map the code surface:

- importable modules;
- failing modules;
- JAX/Haiku/Distrax/OpenMM seams;
- locality/cutoff/neighbor graph seams;
- marginal/auxiliary/free-energy seams;
- candidate Probe 05 paths.

### Key results

```text
Python modules scanned: 28
Symbols cataloged: 132
Classes cataloged: 35
Probe 05 candidate modules: 11
Module imports OK: 16
Module imports failed: 12
```

### Most important next files

| Module | Role |
|---|---|
| `bgmat.systems.energies` | energy abstractions |
| `bgmat.systems.lennard_jones` | LJ energy/cutoff seam |
| `bgmat.systems.monatomic_water` | mW energy seam |
| `bgmat.systems.stillinger_weber` | SW / lambda seam |
| `bgmat.utils.lattice_utils` | geometry/lattice helpers |
| `bgmat.utils.marginal` | auxiliary/marginal seam |
| `bgmat.utils.observable_utils` | observables/free-energy helpers |
| `bgmat.models.gnn_conditioner` | neighbor/local GNN seam |
| `bgmat.models.particle_models` | particle model seam |

### Checkpoint interpretation

Probe 04 found the mechanism map. It did not test the mechanisms yet.

---

## Probe 05 — Tiny energy/locality seam probe

**Script:** `current_probe_05_tiny_energy_seam_probe.py`

### Goal

Test which system/utility modules import, inspect signatures, and identify explicit energy/locality targets.

### Key results

```text
Target modules: 15
Imports OK: 9
Imports failed: 6
Static symbols: 207
Dynamic symbols: 60
Locality/geometry/marginal parameter hits: 266
Zero-arg instantiation attempts: 6
Zero-arg instantiations OK: 4
Method smoke successes: 0
```

### Import-OK target modules

```text
bgmat.experiments.utils
bgmat.models.particle_models
bgmat.systems.energies
bgmat.systems.lennard_jones
bgmat.systems.monatomic_water
bgmat.systems.stillinger_weber
bgmat.utils.lattice_utils
bgmat.utils.marginal
bgmat.utils.observable_utils
```

### Failed modules

All failed through the same gate:

```text
ModuleNotFoundError("No module named 'e3nn_jax'")
```

Affected modules:

```text
bgmat.models.gnn_conditioner
bgmat.models.augmented_coupling_flows
bgmat.models.conditional_split_coupling
bgmat.experiments.augmented_lennard_jones_config
bgmat.experiments.augmented_monatomic_water_config
bgmat.experiments.augmented_monatomic_water_config_lambda_vol_conditional
```

### Zero-arg constructors that worked

| Module | Class |
|---|---|
| `bgmat.systems.monatomic_water` | `_TwoBodyEnergy` |
| `bgmat.systems.monatomic_water` | `MonatomicWaterEnergy` |
| `bgmat.systems.stillinger_weber` | `MonatomicWaterEnergySW` |
| `bgmat.systems.stillinger_weber` | `SiliconEnergySW` |

### Checkpoint interpretation

Probe 05 narrowed the path to explicit system-energy modules and identified the full model-side GNN gate.

---

## Probe 06 — Tiny cutoff/locality stress

**Script:** `current_probe_06_tiny_cutoff_locality_stress.py`

### Goal

Execute explicit tiny energy/cutoff tests.

### Key results

```text
Constructor attempts: 32
Constructor OK: 18
Energy shape attempts: 180
Energy shape OK: 84
LJ cutoff sweep rows: 112
LJ cutoff OK rows: 112
LJ far-particle rows: 40
LJ far-particle OK rows: 40
mW/SW rows: 60
mW/SW OK rows: 18
```

### LJ cutoff diagnostic

```text
ok_points: 112
classes: ['LennardJonesEnergy']
cutoffs: [1.0, 1.5, 2.5, 4.0]
inside_points: 60
beyond_points: 48
beyond_cutoff_near_zero_fraction: 1.0
```

### Important interpretation correction

The LJ cutoff sweep itself was clean and meaningful.

The far-particle comparison in Probe 06 was not yet clean because some “far” placements were still local to another particle or affected by periodic wrapping. This was fixed in Probe 07.

### Checkpoint interpretation

Probe 06 confirmed hard cutoff behavior in explicit LJ pair energy, but its far-particle witness needed cleaner geometry.

---

## Probe 07 — Clean energy locality witness

**Script:** `current_probe_07_clean_locality_witness.py`

### Goal

Cleanly separate far/blind third-particle cases from inside/local third-particle cases using a large periodic box and explicit minimum-image distance checks.

### Key results

```text
Locality witness rows: 16
Pair graph rows: 40
Far/blind cases: 4
Far/blind near-zero deltas: 4
Inside/local cases: 8
Inside/local nonzero deltas: 8
Near-zero tolerance: 1e-07
```

### Far/blind results

| Cutoff | Case | Energy delta |
|---:|---|---:|
| 1.0 | `third_far_all` | 0.0 |
| 1.5 | `third_far_all` | 0.0 |
| 2.5 | `third_far_all` | 0.0 |
| 4.0 | `third_far_all` | 0.0 |

### Inside/local results

All eight inside/local cases produced nonzero deltas.

Examples:

| Cutoff | Case | Interacting pairs | Delta |
|---:|---|---|---:|
| 1.0 | `third_inside_particle0` | `0-2` | 103.80254554748535 |
| 1.5 | `third_inside_particle0` | `0-2` | -0.9998189806938171 |
| 2.5 | `third_inside_particle0` | `0-2;1-2` | -0.1070079356431961 |
| 4.0 | `third_inside_particle0` | `0-2;1-2` | -0.0065010422840714455 |

### Checkpoint interpretation

Probe 07 is a clean explicit-energy locality witness:

```text
outside local cutoff → energy unchanged
inside local cutoff  → energy changes
```

This is still not the learned model. It is an adjacent explicit-energy mechanism witness.

---

## Probe 08 — Neighbor graph locality witness

**Script:** `current_probe_08_neighbor_graph_locality_witness_v2.py`

### Goal

Test the model-side finite-neighbor selector without installing `e3nn_jax`.

### Method

Probe 08 used an AST shim:

1. Read `bgmat/bgmat/models/gnn_conditioner.py`.
2. Extract:
   - `get_senders_and_receivers_fully_connected`
   - `get_top_n_connections_vectorized`
3. Shim the missing periodic helper:
   - `pairwise_difference_pbc`
4. Run clean base/far/near geometries through the top-n neighbor selector.

### Fully connected sanity

| Nodes | Edges |
|---:|---:|
| 3 | 6 |
| 4 | 12 |
| 5 | 20 |

### Neighbor result summary

For comparable unsaturated settings:

```text
n_local = 1
  far node  → existing graph unchanged
  near node → graph rewired / new-node edges appear

n_local = 2
  far node  → existing graph unchanged
  near node → graph rewired / new-node edges appear
```

For saturated setting:

```text
n_local = 3 with base3 is saturated.
It enters an all/self-slot regime and must be interpreted separately.
```

### Key rows

| n_local | Case | Existing graph unchanged? | New-node edges |
|---:|---|---|---|
| 1 | `add_far_node` | True | `1->3` |
| 1 | `add_near0_node` | False | `3->0;3->1;0->3` |
| 1 | `add_near1_node` | False | `3->1;1->3` |
| 2 | `add_far_node` | True | `1->3;2->3` |
| 2 | `add_near0_node` | False | `3->0;3->1;3->2;0->3;1->3` |
| 2 | `add_near1_node` | False | `3->1;1->3;0->3` |

### Checkpoint interpretation

Probe 08 is a model-side neighbor-selection witness, but still isolated/shimmed:

```text
far node  → stable existing graph for comparable n_local
near node → local graph rewires
```

This is closer to the paper mechanism than Probe 07, but it is not the full GNN/flow model.

---

## Probe 09 — Joint energy + neighbor locality report

**Script:** `current_probe_09_joint_energy_neighbor_locality_report.py`

### Goal

Combine Probe 07 and Probe 08 into one joint locality report.

### Key result table

| Claim ID | Result | Claim | Evidence |
|---|---|---|---|
| J01 | PASS | Explicit LJ energy is blind to a third particle outside cutoff. | 4/4 far/blind cases had near-zero energy delta. |
| J02 | PASS | Explicit LJ energy responds when a third particle enters the local cutoff neighborhood. | 8/8 inside/local cases had nonzero energy delta. |
| J03 | PASS | Model-side top-n neighbor graph is stable to far-node insertion for comparable `n_local` values. | 2/2 comparable far-node cases kept existing graph unchanged. |
| J04 | PASS | Model-side top-n neighbor graph reacts to near-node insertion for comparable `n_local` values. | 4/4 comparable near-node cases touched/rewired the local graph. |
| J05 | INFO | Saturated `n_local` values should be interpreted separately. | 4 saturated rows were excluded from pass/fail accounting. |

### Checkpoint interpretation

Probe 09 is the current checkpoint artifact.

It supports this statement:

> The explicit energy cutoff and extracted top-n neighbor selector both show local behavior under controlled far/near perturbations.

It does **not** support this stronger statement:

> The paper’s full Boltzmann generator, transfer claims, ESS results, or figure-level outputs have been reproduced.

---

## 5. Current evidence level

### Evidence ladder

| Level | Description | Current status |
|---|---|---|
| L0 | Repo exists and imports | ✅ done |
| L1 | Source-data inventory mapped | ✅ done |
| L2 | Code seams identified | ✅ done |
| L3 | Explicit mechanism witness | ✅ done |
| L4 | Extracted model-side helper witness | ✅ done |
| L5 | Full model import with all dependencies | ⚠️ not done |
| L6 | Tiny full-flow forward pass | ❌ not done |
| L7 | Sampling/evaluation reproduction | ❌ not done |
| L8 | Figure reproduction | ❌ not done |
| L9 | Paper-level claim validation | ❌ not done |

Current level:

```text
L4 — extracted model-side helper witness
```

---

## 6. What we should say publicly / internally

### Safe internal wording

```text
We performed a staged code audit of the bgmat repository and built a mechanism-level locality witness.

The explicit Lennard-Jones energy implementation behaved as a hard cutoff local potential in controlled far/near perturbation tests. An extracted top-n neighbor selector from the model-side GNN conditioner showed stable existing-neighbor behavior for far-node insertions and rewiring for near-node insertions under comparable unsaturated n_local settings.

This supports the presence and behavior of the locality seam used by the architecture, but it is not a reproduction of the full paper, learned generator, training pipeline, sampling results, ESS trends, TFEP estimates, or figures.
```

### Unsafe / overclaiming wording to avoid

```text
We reproduced the paper.
We validated the model.
We confirmed the scalability claims.
We confirmed the ESS results.
We confirmed TFEP/free-energy claims.
We confirmed transfer to large systems.
The paper is correct.
```

---

## 7. Immediate next options

### Option A — Conservative next probe: larger statistical neighbor witness

Keep the shim path and run many random geometries.

Goal:

```text
Measure neighbor-graph stability under far perturbations versus near perturbations across many random point clouds and n_local values.
```

Pros:

- no new dependency installs;
- fast;
- strengthens mechanism witness statistically;
- good checkpoint follow-up.

Cons:

- still not full model import.

### Option B — Full dependency gate: install/import `e3nn_jax`

Goal:

```text
Import full bgmat.models.gnn_conditioner and compare full function behavior to shimmed helper behavior.
```

Pros:

- moves closer to full model path;
- reduces “shimmed witness” caveat.

Cons:

- more dependency sludge likely;
- may create new JAX/Windows issues;
- still not training or reproduction.

### Option C — Tiny full-flow forward pass

Goal:

```text
Instantiate a minimal model/config and run a tiny forward/log-prob/sample path.
```

Pros:

- moves from helper witness toward architecture execution.

Cons:

- likely blocked by `e3nn_jax`;
- may require config surgery;
- more fragile.

Recommended next step:

```text
Option A first, then Option B.
```

Reason:

> We should strengthen the locality witness statistically before opening a dependency swamp.

---

## 8. Current final checkpoint statement

The most accurate checkpoint statement is:

> Probes 01–09 completed a staged audit from repo discovery to a joint locality witness. The current evidence supports an adjacent mechanism-level claim: explicit energy cutoffs and extracted top-n neighbor selection behave locally under controlled far/near perturbations. This is consistent with the paper’s locality-based architecture, but it is not a reproduction or validation of the full paper-level results.

---

## 9. Files produced so far

Representative files:

```text
current_probe_01_claim_map_repo_audit.py
current_probe_02_source_data_audit.py
current_probe_03_official_repo_smoke.py
current_patch_compat_modern_deps.py
current_probe_04_module_seam_map.py
current_probe_05_tiny_energy_seam_probe.py
current_probe_06_tiny_cutoff_locality_stress.py
current_probe_07_clean_locality_witness.py
current_probe_08_neighbor_graph_locality_witness_v2.py
current_probe_09_joint_energy_neighbor_locality_report.py
```

Representative output folders:

```text
probe_runs/current_probe03_official_smoke_*
probe_runs/current_probe04_module_seams_*
probe_runs/current_probe05_tiny_energy_seams_*
probe_runs/current_probe06_tiny_cutoff_locality_*
probe_runs/current_probe07_clean_locality_witness_20260613_114735
probe_runs/current_probe08_neighbor_graph_locality_20260613_120602
probe_runs/current_probe09_joint_locality_report_20260613_121123
```

---

## 10. Closing checkpoint

This is a good pause point.

The audit has moved from:

```text
paper PDF → source data → repo import → module seams → explicit energy witness → extracted neighbor witness → joint locality report
```

Current conclusion:

```text
mechanism witness achieved
paper reproduction not yet attempted
full model path still gated by e3nn_jax
```

Recommended next move after checkpoint:

```text
Probe 10 — statistical neighbor locality witness across random point clouds
```
