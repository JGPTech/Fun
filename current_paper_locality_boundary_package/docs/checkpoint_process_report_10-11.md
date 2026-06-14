# CHECKPOINT PROCESS REPORT 10–11

**Project:** Current paper locality-boundary audit  
**Checkpoint file:** `checkpoint_process_report_10-11.md`  
**Status:** capstone mechanism-boundary checkpoint  
**Date:** 2026-06-13  
**Scope:** Probes 10–11 only, continuing from `checkpoint_process_report_01-09.md`

---

## 0. Executive checkpoint

This checkpoint covers the reset back to the main objective.

The main objective was not merely to confirm that the current paper’s local machinery behaves locally. Probes 01–09 already established that baseline.

The real objective was:

> Find the specific mechanism by which fixed local/top-n neighborhoods can fail to represent long-range physical information.

Probes 10–11 are the capstone sequence.

They move the audit from:

```text
locality works as expected
```

to:

```text
locality can alias long-range physical target information
```

The current capstone result is:

> In controlled toy configurations, identical extracted local/top-n neighborhood signatures can correspond to different neutral long-range physical target witnesses.

The mechanism identified is:

> **finite local neighborhood aliasing** — once all local/top-n slots are occupied by nearby structure, distant/global information can be excluded from the representation even when a long-range physical witness changes.

This is a mechanism-boundary result.

It is **not** a full paper reproduction, paper failure, benchmark refutation, or figure-level validation result.

---

## 1. Mission reset

The original question was about the current paper’s local-vs-long-range boundary.

The working hypothesis was:

```text
A fixed local/top-n neighborhood representation may treat two globally distinct configurations as identical if their local neighborhoods match.
```

The more precise failure shape was:

```text
same local/top-n representation
different long-range/global or physical witness
```

Probes 01–09 reconstructed the minimal local machinery needed to ask that question properly:

```text
claim map
→ repo/source-data audit
→ import/dependency path
→ seam map
→ explicit energy locality witness
→ extracted neighbor-graph locality witness
→ joint locality report
```

That established the control:

```text
the local machinery is actually local
```

Probes 10–11 then asked the actual boundary question:

```text
what does that locality erase?
```

---

## 2. Evidence ladder update

| Level | Description | Status after Probe 09 | Status after Probe 11 |
|---|---|---:|---:|
| L0 | Repo exists and imports | ✅ | ✅ |
| L1 | Source-data inventory mapped | ✅ | ✅ |
| L2 | Code seams identified | ✅ | ✅ |
| L3 | Explicit locality witness | ✅ | ✅ |
| L4 | Extracted model-side helper witness | ✅ | ✅ |
| L5 | Locality collision / representation alias | ❌ | ✅ |
| L6 | Physical long-range alias confirmation | ❌ | ✅ |
| L7 | Full model import with all dependencies | ⚠️ not done | ⚠️ not done |
| L8 | Tiny full-flow forward pass | ❌ | ❌ |
| L9 | Sampling/evaluation reproduction | ❌ | ❌ |
| L10 | Figure/source-data reproduction | ❌ | ❌ |
| L11 | Paper-level claim validation | ❌ | ❌ |

Current level:

```text
L6 — physical long-range alias confirmation in controlled toy witnesses
```

---

## 3. Probe 10 — Locality collision / long-range alias witness

**Script:** `current_probe_10_locality_collision_long_range_alias.py`

### Goal

Probe 10 was the first direct test of the main objective.

It asked:

> Can two configurations share identical extracted local/top-n signatures while differing in long-range/global witnesses?

### Method

Probe 10 constructed paired two-cluster configurations.

Each case used the same internal local motif. The global placement of the second motif changed.

The probe then extracted the same top-n neighbor helper path from the current paper’s model-side `gnn_conditioner` logic and compared:

- local graph signatures;
- local distance-shell signatures;
- internal cluster distance signatures;
- whether top-n graphs remained internal-only;
- long-range/global witnesses.

### Probe 10 alias condition

An alias was counted when:

```text
local_graph_same == True
local_distance_shell_same == True
internal_cluster_signature_same == True
both_topn_graphs_internal_only == True
long_range_delta_nonzero == True
alias_detected == True
```

In plain language:

> The local machinery sees the same object, but a long-range/global witness changes.

### Configuration families

Probe 10 compared cases such as:

| Pair | Meaning |
|---|---|
| `sep20_x` vs `sep45_x` | same motif, different cluster separation |
| `sep20_x` vs `sep20_y` | same motif, same nominal separation magnitude, different global axis |
| `sep20_x` vs `sep20_diag_xy` | same motif, diagonal global placement |
| `sep20_x` vs `sep20_z` | same motif, different z-axis placement |

### Key Probe 10 result

Probe 10 produced:

```text
12/12 alias comparison rows with alias_detected=True
```

This included all tested `n_local` values:

```text
n_local = 1
n_local = 2
n_local = 3
```

### Strongest Probe 10 example

For `sep20_x` vs `sep45_x`:

```text
local_graph_same = True
local_distance_shell_same = True
internal_cluster_signature_same = True
both_topn_graphs_internal_only = True
long_range_delta_nonzero = True
alias_detected = True
```

Representative long-range/global deltas included:

```text
tail_sum_inv_r_delta_abs      = 0.4435342164051259
radius_gyration_sq_delta_abs  = 406.25
max_cov_component_delta_abs   = 406.25
```

Interpretation:

> The extracted local/top-n representation preserved the local motif and erased the distant separation scale.

### Axis/anisotropy examples

For cases such as `sep20_x` vs `sep20_y` and `sep20_x` vs `sep20_z`, the local graph remained identical while global covariance components changed strongly.

This supports a second flavor of aliasing:

> The local representation can also alias global orientation/anisotropy when local neighborhoods remain unchanged.

### What Probe 10 proved

Probe 10 proved a representation-level locality collision:

```text
same extracted local/top-n representation
different global/long-range witness
```

### What Probe 10 did not prove

Probe 10 did not prove that the full paper model fails.

It did not test:

- full GNN conditioner import;
- full augmented flow;
- training;
- sampling;
- ESS;
- TFEP;
- free energies;
- figure reproduction;
- transfer claims.

Probe 10 only established the local representation alias shape.

---

## 4. Probe 11 — Physical long-range alias confirmation

**Script:** `current_probe_11_physical_long_range_alias_confirmation.py`

### Goal

Probe 11 was the capstone confirmation.

It asked:

> Do the Probe 10 local aliases remain aliases when attached to simple neutral long-range physical target witnesses?

This converted the Probe 10 result from:

```text
global witness differs
```

to:

```text
neutral long-range physical target differs
```

### Method

Probe 11 consumed Probe 10’s `alias_comparisons.csv`.

It reconstructed the aliased two-cluster configurations and assigned neutral charge models to each identical motif.

The charge models were controlled toy witnesses, not claims about the benchmark systems.

Representative neutral charge models:

```text
neutral_dipole
neutral_quadrupole
neutral_asymmetric
```

Each cluster had zero net charge.

### Physical witnesses

Probe 11 computed neutral Coulomb-like long-range target witnesses including:

- cross-cluster Coulomb-like energy;
- absolute cross-cluster pair-force sum;
- net force magnitude on cluster A;
- L1 particle-force response on cluster A;
- torque magnitude on cluster A;
- field magnitude at cluster A centroid due to cluster B;
- covariance/global anisotropy descriptors.

### Probe 11 confirmation condition

A physical alias confirmation row required:

```text
Probe10 alias_detected == True
primary_physical_delta_nonzero == True
physical_alias_confirmed == True
```

In plain language:

> The local/top-n representation aliases the pair, and a neutral long-range physical witness separates it.

---

## 5. Probe 11 capstone claim table

| claim_id | result | claim | evidence |
|---|---|---|---|
| P11-C01 | PASS | Probe 10 produced local/top-n alias rows suitable for physical confirmation. | 12/12 Probe 10 comparison rows had alias_detected=True. |
| P11-C02 | PASS | At least one local/top-n alias pair is separated by a neutral Coulomb-like physical witness. | 36/36 charge-model confirmation rows had physical_alias_confirmed=True. |
| P11-C03 | PASS | Every Probe 10 alias pair/n_local group is physically separated by at least one neutral long-range witness. | 12/12 alias pair/n_local groups had physical_alias_confirmed_any_model=True. |
| P11-C04 | SUPPORTED_IN_CONTROLLED_WITNESS | Specific mechanism identified: finite local/top-n neighborhoods can erase long-range physical target information. | Rows require Probe10 alias_detected=True plus a nonzero primary neutral long-range physical delta. |
| P11-C05 | INFO | Paper implication is conditional. | This probe does not train or run the full BG; it provides a minimal counterexample class for fixed local representations. |

---

## 6. What Probe 11 proves

Probe 11 supports this controlled mechanism-boundary claim:

> In controlled toy configurations, identical extracted local/top-n neighborhood signatures can correspond to different neutral long-range physical target witnesses.

A shorter version:

> Fixed local/top-n neighborhoods can alias long-range physical target information.

The mechanism is specific:

```text
finite local neighborhood aliasing
```

or more explicitly:

```text
once local/top-n slots are filled by nearby structure,
distant/global physical information can be excluded from the representation
even when a neutral long-range target witness changes
```

---

## 7. What Probe 11 does not prove

Probe 11 does **not** prove:

- the paper is wrong;
- the benchmark systems contain this alias;
- the full Boltzmann generator fails;
- the paper’s figures are invalid;
- ESS or TFEP claims are wrong;
- transfer results are wrong;
- local architectures are useless.

It also does not execute:

- full model import with `e3nn_jax`;
- full flow forward pass;
- training;
- sampling;
- reweighting;
- free-energy estimation;
- phase-diagram reproduction.

Probe 11 is a capstone mechanism witness, not a reproduction failure.

---

## 8. Paper implication

The implication is conditional and specific.

The current paper’s architecture relies on local environments to scale. Probes 10–11 identify a concrete boundary condition:

> If the target physics depends on information outside the fixed local/top-n neighborhood, then the representation can alias physically distinct states.

The suggested correction target is also specific:

> Add or verify a nonlocal, multiscale, reciprocal-space, global conditioning, or long-range interaction channel, or include diagnostics showing that such aliases are irrelevant for the claimed target systems.

This should be framed as:

```text
a candidate mechanism-boundary correction
```

not as:

```text
a proof that the paper is invalid
```

---

## 9. Relation to the original Paper 1 question

The original Paper 1 issue involved pair-level hotspot mismatch versus response-level or neighborhood-level long-range force structure.

The current paper was treated standalone.

Only after Probes 10–11 is there a possible bridge back:

```text
current paper boundary:
  fixed local/top-n representations can erase long-range physical target information

Paper 1 question:
  can force-response / hotspot / neighborhood-response witnesses detect the erased long-range signal?
```

This bridge remains future work.

No Paper 1 integration claim is made in Probes 10–11.

---

## 10. GitHub packaging status

After Probe 11, the package is ready for GitHub preparation.

Generated package:

```text
current_paper_locality_boundary_package.zip
```

Package contents include:

```text
README.md
CLAIM_BOUNDARY.md
METHODS.md
RESULTS_SUMMARY.md
PROBE_INDEX.md
PACKAGE_CHECKLIST.md
docs/checkpoint_process_report_01-09.md
results/capstone_claim_table.csv
probes/current_probe_01...
...
probes/current_probe_11...
```

The repository headline should be:

```text
finite local/top-n neighborhoods can erase long-range physical target information
```

The first-screen caveat should be:

```text
This repository does not reproduce the full paper. It provides a staged mechanism-boundary audit.
```

---

## 11. Recommended README claim

Use this wording:

```text
This repository does not claim to reproduce the full paper.

It provides a staged mechanism-boundary audit of a fixed-local-neighborhood Boltzmann-generator architecture.

The capstone probes show that, in controlled toy configurations, extracted local/top-n neighborhood signatures can remain identical while neutral long-range physical target witnesses change.

This identifies a concrete boundary condition: finite local neighborhood aliasing.
```

---

## 12. Recommended public-safe summary

Use this wording for posts, README, or issue descriptions:

```text
I audited the local/long-range boundary of a scalable local Boltzmann-generator code path.

The probes do not reproduce the full paper. They reconstruct the minimal local-neighborhood machinery needed to test a specific question: can fixed local/top-n neighborhoods alias long-range physical information?

In controlled toy configurations, the extracted local/top-n signatures remain identical while neutral Coulomb-like long-range witnesses change. This identifies a concrete mechanism-boundary condition: finite local neighborhood aliasing.

The result suggests a conditional correction target for systems where long-range physics matters: add or verify nonlocal, multiscale, reciprocal-space, or global conditioning channels, or include diagnostics showing such aliases are irrelevant for the target observables.
```

---

## 13. Recommended next technical steps

### Before GitHub

1. Rerun all probes once from a clean folder.
2. Confirm no hardcoded local paths remain.
3. Confirm Probe 10 and Probe 11 auto-detect latest relevant folders.
4. Confirm generated docs use the non-reproduction caveat.
5. Remove PDFs, source-data spreadsheets, and any copyrighted/non-redistributable files.
6. Commit scripts and generated summaries only.

### After GitHub

Optional future probes:

| Future probe | Purpose |
|---|---|
| Probe 12 | Full `e3nn_jax` import and full GNN conditioner comparison |
| Probe 13 | Tiny full-flow forward pass |
| Probe 14 | Physical alias test inside full model path |
| Probe 15 | Paper 1 response/hotspot witness applied to alias pairs |
| Probe 16 | Larger randomized alias statistics |

---

## 14. Final checkpoint statement

The final checkpoint statement for Probes 10–11 is:

> Probes 10–11 identified and confirmed a controlled mechanism-boundary condition for fixed local/top-n representations. Probe 10 showed that globally distinct configurations can share identical extracted local/top-n signatures. Probe 11 showed that those local aliases can correspond to different neutral long-range physical target witnesses. This supports the specific claim that finite local neighborhoods can erase long-range physical target information. The implication for the current paper is conditional: full-model and benchmark-specific tests are still required, but the capstone probes identify the exact correction target to check.

---

## 15. Closing

This is a clean capstone point.

The audit has moved from:

```text
paper question
→ repo and source-data map
→ local mechanism reconstruction
→ locality control witnesses
→ representation-level alias
→ physical long-range alias confirmation
```

Current conclusion:

```text
capstone mechanism-boundary witness achieved
paper reproduction not claimed
GitHub packaging ready
```
