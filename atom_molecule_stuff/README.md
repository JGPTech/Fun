# DHA PaiNN Pairwise Force Attribution Probe

A small public sandbox for probing the MLFF side of the paper **“How Atoms Interact Within Molecules”** (`arXiv:2605.28960v1`) using public DHA data, SchNetPack PaiNN, and pairwise force-attribution diagnostics.

This is not an official reproduction of the paper and it does **not** reproduce the private/expensive SQ-MBD pairwise finite-difference decomposition. It is a focused public-data probe of the PaiNN/MLFF attribution layer described in the paper:

\[
F_{ij} = -\frac{\partial E_j}{\partial R_i}
\]

where `E_j` is a learned per-atom energy contribution and `F_ij` is interpreted as atom `j`'s contribution to the force on atom `i`.

## What this probe does

The script:

1. Downloads public DHA data from Hugging Face / ColabFit (`colabfit/MD22_DHA`) on first run.
2. Caches the data locally as `.npz`.
3. Builds a SchNetPack-compatible ASE database.
4. Trains a SchNetPack PaiNN model with paper-like DHA settings.
5. Extracts pairwise force-attribution vectors using

   \[
   F_{ij} = -\frac{\partial E_j}{\partial R_i}
   \]

   with autograd when possible and finite differences when the installed SchNetPack build detaches the per-atom output graph.

6. Reconstructs the summed force from the pairwise attributions and compares it to the model force.
7. Measures interaction-depth scatter and long-range anisotropy.
8. Runs simple zero-sum energy-gauge perturbation tests.
9. Compares attribution hotspots against a total-force response witness:

   \[
   \left\|\frac{\partial F_i}{\partial R_j}\right\|
   \]

## Why this exists

The paper reports that pairwise force contributions in molecules show broad scatter and persistent long-range anisotropy, and that MLFFs qualitatively reproduce much of this structure.

The goal here was to test two things on the public MLFF side:

1. Can a public-data PaiNN probe reproduce the qualitative scatter / anisotropy signature?
2. Are the learned pairwise attribution hotspots stable under simple perturbations, and do they agree with a more direct force-response witness?

## Representative local result

A representative run was performed on Windows with an RTX 3090 using:

```powershell
python dha_painn_schnetpack_probe_v6.py --nframes 2500 --max-epochs 50 --device cuda --force-train --train-only --num-workers 0

python dha_painn_schnetpack_probe_v6.py --nframes 2500 --analysis-frames 24 --response-frames 4 --device cuda --fij-method finite-diff --fij-fd-delta 0.002 --response-fd-delta 0.002
```

Representative summary:

```text
eval_E_MAE                                             : 0.1360
eval_F_MAE                                             : 0.0818
n_pairs                                                : 73920
analysis_frames                                        : 24
mean_log10_force_iqr_by_1A_bins                        : 0.3773
overall_log10_force_range                              : 4.6030
aligned_fraction_theta_gt150_dist_ge10                 : 0.0965
anisotropic_fraction_theta_lt150_dist_ge10             : 0.9035
p95_log10_deviation_from_rminus7                       : 1.3094
p99_log10_deviation_from_rminus7                       : 1.8914
mean_reconstruction_best_scalar_alpha_sumj_to_model_Fi : 0.9938
mean_reconstruction_mae_scaled_sumj_Fij_equals_model_Fi: 0.0680
mean_reconstruction_cosine_sumj_vs_model_Fi            : 0.9969
gauge_0.3_mean_topk_jaccard                            : 0.9149
gauge_0.3_mean_spearman                                : 0.9996
response_mean_topk_jaccard_attr_vs_response            : 0.0
response_mean_spearman_attr_vs_response                : -0.6887
```

## Preliminary interpretation

This probe **qualitatively reproduces the public-data MLFF-side signal**:

- pairwise force-attribution magnitudes span multiple orders of magnitude;
- long-range contributions remain strongly anisotropic;
- the broad scatter / anisotropy pattern survives across finite-difference settings.

The initial suspicion was that the learned per-atom-energy attribution might be fragile under simple zero-sum energy-gauge perturbations. In this PaiNN probe, that suspicion was mostly **not supported**:

- gauge perturbation at strength `0.3` preserved the hotspot ranking well;
- top-k Jaccard stayed around `0.91`;
- Spearman rank correlation stayed near `0.9996`.

The more interesting robustness question came from the force-response comparison:

- attribution hotspots and total-force response hotspots had zero top-k overlap in the representative run;
- Spearman correlation was strongly negative;
- the summed attribution still reconstructed the model force direction well, with cosine around `0.997`.

That suggests a possible strengthening test for hotspot claims:

> Define robust molecular hotspots as regions where pairwise attribution and force-response sensitivity agree, not attribution alone.

For the MLFF side, this is probably a validation-layer issue rather than a fatal problem. The open question is whether the same attribution/response separation appears in the SQ-MBD pairwise decomposition, where the physical interpretation is stronger.

## Caveats

This is exploratory.

Important limitations:

- This repo does **not** reproduce the paper's SQ-MBD decomposition.
- This repo does **not** use the authors' trained models or private intermediate tensors.
- The trained PaiNN model here is not yet as accurate as the paper's reported DHA MLFF model.
- The force-response witness is finite-difference based and can be expensive.
- The results should be read as a robustness probe, not a claim that the paper is wrong.

## Files

Expected minimal repo layout:

```text
atom_molecule_stuff/
├── README.md
└── dha_painn_schnetpack_probe_v6.py
```

Generated files are intentionally not tracked by default. Typical generated paths:

```text
dha_probe_cache/
├── md22_dha_colabfit.npz
├── md22_dha_schnetpack.db
└── md22_dha_split.npz

dha_painn_runs/
└── dha_painn_probe_<timestamp>/
    ├── best_inference_model/
    ├── analysis_summary.json
    ├── final_report.json
    ├── eval_subset.csv
    ├── frame_records.csv
    ├── gauge_trials.csv
    ├── response_trials.csv
    ├── interaction_depth_hist2d.png
    ├── anisotropy_hist2d.png
    ├── rminus7_deviation_hist2d.png
    ├── gauge_stability_boxplot.png
    └── response_witness_overlap.png
```

## Install

Recommended: use a fresh virtual environment.

Windows / PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy scipy pandas pyarrow datasets huggingface_hub matplotlib tqdm ase torchmetrics pytorch-lightning schnetpack
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch numpy scipy pandas pyarrow datasets huggingface_hub matplotlib tqdm ase torchmetrics pytorch-lightning schnetpack
```

## Basic usage

### 1. Smoke test

```powershell
python dha_painn_schnetpack_probe_v6.py --nframes 500 --max-epochs 2 --analysis-frames 2 --response-frames 0 --device cuda --force-train
```

Use `--device cpu` if CUDA is unavailable.

### 2. Train only

On Windows, keep `--num-workers 0`. Higher worker counts caused multiprocessing spawn/import problems during local testing.

```powershell
python dha_painn_schnetpack_probe_v6.py --nframes 2500 --max-epochs 50 --device cuda --force-train --train-only --num-workers 0
```

### 3. Analysis using latest trained model

```powershell
python dha_painn_schnetpack_probe_v6.py --nframes 2500 --analysis-frames 24 --response-frames 4 --device cuda --fij-method finite-diff --fij-fd-delta 0.002 --response-fd-delta 0.002
```

### 4. Analysis using an explicit model path

```powershell
python dha_painn_schnetpack_probe_v6.py --nframes 2500 --analysis-frames 24 --response-frames 4 --device cuda --model-path "dha_painn_runs\dha_painn_probe_YYYYMMDD_HHMMSS\best_inference_model" --fij-method finite-diff --fij-fd-delta 0.002 --response-fd-delta 0.002
```

## Windows notes

PyTorch Lightning may warn that the dataloaders should use many workers. On Windows, that advice can backfire because multiprocessing uses spawn semantics and re-imports the script/package stack in every child process.

For this probe, the stable local setting was:

```powershell
--num-workers 0
```

If you want to experiment with workers anyway:

```powershell
--allow-windows-workers --num-workers 2
```

But if the process starts spamming import warnings, freezing, or hitting memory errors, go back to `--num-workers 0`.

## Main metrics

The most useful output file is:

```text
dha_painn_runs/<run>/final_report.json
```

Key fields:

| Metric | Meaning |
|---|---|
| `overall_log10_force_range` | Spread of pairwise attribution magnitudes in log10 units. |
| `anisotropic_fraction_theta_lt150_dist_ge10` | Fraction of long-range pairs with angle less than 150 degrees. |
| `aligned_fraction_theta_gt150_dist_ge10` | Fraction of long-range pairs strongly aligned with the interatomic axis. |
| `p95_log10_deviation_from_rminus7` | 95th percentile excess over the fitted `R^-7` baseline. |
| `mean_reconstruction_cosine_sumj_vs_model_Fi` | Directional agreement between summed pairwise attributions and model force. |
| `gauge_0.3_mean_topk_jaccard` | Stability of hotspot top-k under zero-sum energy-gauge perturbation. |
| `response_mean_topk_jaccard_attr_vs_response` | Top-k overlap between attribution hotspots and force-response hotspots. |
| `response_mean_spearman_attr_vs_response` | Rank correlation between attribution score and force-response score. |

## Method sketch

For each analyzed frame:

1. Compute pair distances `R_ij`.
2. Extract or finite-difference per-atom energy contributions `E_j`.
3. Estimate pairwise attribution vectors:

   \[
   F_{ij,\alpha} \approx -\frac{E_j(R_{i,\alpha}+\Delta)-E_j(R_{i,\alpha}-\Delta)}{2\Delta}
   \]

4. Compute magnitude:

   \[
   \|F_{ij}\|_2
   \]

5. Compute angle between `F_ij` and the interatomic displacement vector.
6. Fit an `R^-7` baseline over the requested distance window.
7. Rank hotspots by deviation or magnitude.
8. Perturb the per-atom energy partition using simple zero-sum transformations and rerank.
9. Compute a total-force response witness by finite difference:

   \[
   \left\|\frac{\partial F_i}{\partial R_j}\right\|
   \]

10. Compare attribution hotspots against response hotspots.

## Related resources

- Paper: `How Atoms Interact Within Molecules`, arXiv:2605.28960v1
- Dataset: `colabfit/MD22_DHA`
- MLFF package: SchNetPack PaiNN

## Suggested reading of the result

The clean read is not “the paper is wrong.”

The clean read is:

> A public-data PaiNN probe reproduces the qualitative scatter/anisotropy signature, and simple energy-gauge perturbations do not destroy the attribution ranking. However, attribution hotspots and total-force response hotspots separate strongly in this probe. A useful robustness extension would be to validate molecular hotspots against a response-stability witness, especially before carrying hotspot interpretation into SQ-MBD or biological mechanism claims.

