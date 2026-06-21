# CUDA finite-clock replay for Hamiltonian non-reciprocal interactions

This repository contains a CUDA finite-clock replay and evidence pipeline for the paper:

**Yu-Bo Shi, Roderich Moessner, Ricard Alert, and Marin Bukov,
“Hamiltonian description of non-reciprocal interactions,” Nature Physics (2026).**
Paper: https://doi.org/10.1038/s41567-026-03317-0
Official data/code record: https://doi.org/10.5281/zenodo.19427734

The goal of this repository is to provide a reproducible, resource-bounded replay of the Fig. 3 comparison workflow using a CUDA finite-clock implementation of the constrained-Hamiltonian Glauber update rule, a selfish-energy control branch, and a frozen causal insertion-decay observable.

## What this repository contains

```text
.
├── kernel/
│   └── kernel_finite_clock.cu
├── run_unified_pipeline.py
├── run_unified_test_series.py
├── results.md
├── analysis/
│   └── unified_test_series/
│       └── series_20260621_121023/
│           ├── 01_q512_cuda_smoke/
│           ├── 02_q512_short/
│           ├── 03_q512_smoke_insertion_decay/
│           ├── 04_q512_mid_freeze/
│           ├── 05_q512_frozen_candidate/
│           ├── 06_high_temp_bracket/
│           ├── 07_q1024_continuum_check/
│           ├── 08_L16_medium/
│           ├── 09_L32_deep/
│           └── series_manifest.json
└── data/
    └── fig3/
        ├── data_better.npz
        └── data_fig7_selfish.npz
```

The author data files shown above are **not included in this repository**. See [Data setup](#data-setup).

## What this replay does

The unified pipeline performs the following stages:

1. Builds small finite-clock diagnostic artifacts.
2. Builds projection-base states.
3. Runs paired CUDA constrained/selfish finite-clock projections over a temperature grid.
4. Saves local K1 witness checkpoints.
5. Selects `T_ins` using the predeclared scout selector.
6. Builds coherence-thinned causal delay taps and freezes the insertion operator.
7. Applies the frozen K2 insertion on device.
8. Evolves live, post-hoc, and null insertion arms.
9. Optionally compares the observable against the author-provided processed Fig. 3 data.

The main comparison observable reported in `results.md` is:

```text
c_mean_m_insertable = c_mean_m_raw * D_insert(T)
```

where `D_insert(T)` is the frozen `causal_insertion_decay_v7` decay factor. The raw constrained projection values are preserved separately beside the insertable observable.

## Scientific boundary

This repository should be read as a **resource-bounded replay and extension**, not as a full paper-scale rerun.

Completed evidence covers tests 1–9:

* progression/smoke tests: tests 1–4
* high-temperature stress bracket: test 6
* main writeup-grade comparison set: tests 5, 7, 8, and 9
* q-refinement check: L8, q512 to q1024
* lattice scaling check: L8 to L16 to L32

The full paper-scale test 10 is registered in the runner, but was not run locally because of compute limits.

The strongest completed holdout is:

```text
Test 09: L32_deep
L = 32
q = 512
n_traj = 100
insertable MAE vs author Fig. 3 M_MC = 0.04180
raw MAE vs author Fig. 3 M_MC = 0.19474
```

The claim supported by this repository is:

> A CUDA finite-clock constrained/selfish replay, combined with a frozen causal insertion-decay observable, tracks the author processed Fig. 3 constrained-MC magnetization curve across the completed main comparison set, with insertable MAE approximately 0.04–0.05.

The repository does **not** claim that the raw constrained projection alone reproduces the full paper-scale Fig. 3 curve.

The v7 insertion-decay constants are phenomenological global constants selected during development and then frozen for the completed series:

```text
lag_decay_strength = 0.825
thermal_guide_strength = 0.925
```

They are not refit per test, per temperature, or at runtime.

## Data setup

The author data is not redistributed in this repository.

To run author-data comparisons, obtain the official archive from the Zenodo record:

https://doi.org/10.5281/zenodo.19427734

After downloading/extracting the archive, place the required processed Fig. 3 files at:

```text
data/fig3/data_better.npz
data/fig3/data_fig7_selfish.npz
```

The runner expects `data_better.npz` by default at:

```text
data/fig3/data_better.npz
```

with:

```text
temperature key: tem
observable key: M_MC
```

If your local data path differs, pass it explicitly:

```powershell
python run_unified_test_series.py `
  --tests 5,7,8,9 `
  --author-npz path\to\data_better.npz `
  --author-temp-key tem `
  --author-observable-key M_MC
```

To run without author-data comparison:

```powershell
python run_unified_test_series.py --tests 1-3 --no-author
```

## Environment

This project requires:

* Python 3.10+
* NumPy
* CuPy with CUDA support
* NVIDIA CUDA-capable GPU
* A working CUDA driver/runtime compatible with the installed CuPy build

Example install pattern:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy cupy-cuda12x
```

Use the CuPy package matching your local CUDA runtime.

## Running the staged suite

From the repository root:

```powershell
python run_unified_test_series.py --tests 1-3
```

Run the main completed comparison set:

```powershell
python run_unified_test_series.py --tests 5,7,8,9
```

Resume an interrupted series:

```powershell
python run_unified_test_series.py `
  --series-dir analysis\unified_test_series\series_YYYYMMDD_HHMMSS `
  --tests 5,7,8,9 `
  --resume
```

Dry-run the planned commands without executing them:

```powershell
python run_unified_test_series.py --tests 5,7,8,9 --dry-run
```

## Running a single unified pipeline job

The staged runner is recommended, but a direct run is also possible:

```powershell
python run_unified_pipeline.py `
  --kernel kernel\kernel_finite_clock.cu `
  --out analysis `
  --L 8 `
  --q 512 `
  --n-traj 64 `
  --repeats 1 `
  --burn-steps 600000 `
  --sample-steps 220000 `
  --sample-stride 2000 `
  --checkpoint-steps 22000 `
  --forward-steps 50000 `
  --forward-sample-stride 1000 `
  --temps 0.40,0.56,0.64,0.72,0.76,0.80,0.84,0.88,0.96 `
  --Psi-scale-pi 0.85 `
  --proposal-id 2 `
  --proposal-delta-rad 0.1 `
  --window-radius 1 `
  --init ordered `
  --seed 1234 `
  --forward-seed 987654 `
  --threads-per-block 128 `
  --tap-lags 1,2,3 `
  --tap-weights 0.5,0.3,0.2 `
  --beta-pi 0.5 `
  --rho 0.10 `
  --lambda-safety-factor 3.0 `
  --null-scale 0.49 `
  --author-npz data\fig3\data_better.npz `
  --author-temp-key tem `
  --author-observable-key M_MC
```

## Existing results

The included `results.md` summarizes completed tests 1–9.

Main comparison set:

```text
Test 05: q512_frozen_candidate       insertable MAE = 0.04709
Test 07: q1024_continuum_check       insertable MAE = 0.04028
Test 08: L16_medium                  insertable MAE = 0.04747
Test 09: L32_deep                    insertable MAE = 0.04180
```

Main-set insertable MAE range:

```text
0.04028 to 0.04747
```

Main-set raw MAE range:

```text
0.19474 to 0.28478
```

These results use the author processed Fig. 3 constrained-MC magnetization array `M_MC` from `data_better.npz`.

## Reproducibility notes

* The CUDA kernel uses finite-clock states `k_i in {0, ..., q-1}`.
* `proposal_id = 2` samples a continuous update range first, then quantizes onto the finite clock.
* The support size is set by `proposal_delta_rad = 0.1`.
* The vision-cone angle is `Psi = 0.85π`.
* The paired CUDA branch tracks constrained-Hamiltonian Glauber and selfish-energy dynamics side by side.
* The insertable observable is a frozen post-scout observable with fixed v7 constants, not a hidden runtime fit to the author data.
* The null arm is retained as a control diagnostic.

## Citation

If using this repository, cite the original paper and data archive:

```bibtex
@article{shi2026hamiltonian,
  title = {Hamiltonian description of non-reciprocal interactions},
  author = {Shi, Yu-Bo and Moessner, Roderich and Alert, Ricard and Bukov, Marin},
  journal = {Nature Physics},
  year = {2026},
  doi = {10.1038/s41567-026-03317-0}
}

@dataset{shi2026zenodo19427734,
  title = {Hamiltonian description of nonreciprocal interactions},
  author = {Shi, Yu-Bo and Moessner, Roderich and Alert, Ricard and Bukov, Marin},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19427734}
}
```

## License

The original paper is open access under a Creative Commons Attribution 4.0 International License. The author data/code archive should be used according to the terms stated on the official Zenodo record.
