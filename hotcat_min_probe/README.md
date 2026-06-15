# Simple summary

This repo is a small, start-to-finish sandbox for one narrow question in **"To Cool, or Not to Cool? Displacement Sensing with Hot Quantum States"**:

> Does the paper's reduced hot-ECD branch formula track the actual displacement sensitivity of a finite Hilbert-space hot-cat state?

The pipeline is intentionally short:

1. **Rebuild the paper's simplified hot-ECD protocol model.**  
   The script sweeps cooling time and cat-generation time using the paper's Fig. 2-style trapped-ion parameters.

2. **Find the best reduced-model operating point.**  
   It computes `QFI / (cool_time + cat_time)` and records the best grid point. This checks the paper-level claim that the simplified model can prefer little or no cooling plus finite cat-generation time.

3. **Build finite-cutoff hot-ECD cat states.**  
   For several thermal occupations `nbar` and branch-separation values `alpha`, the script constructs a finite-dimensional ECD-style hot cat.

4. **Compare the paper's branch formula against two witnesses.**  
   The reduced formula is checked against exact spectral mixed-state QFI and against a finite-displacement Bures response witness.

5. **Group the result by branch-separation strength.**  
   The key control parameter is `eta = alpha^2 / (2 nbar + 1)`. Low `eta` means stronger branch overlap, where the approximation should be less reliable. Higher `eta` is the cleaner branch-pseudospin regime.

6. **Write plain CSVs and one human-readable summary.**  
   All outputs go into `analysis/`. The main file to read first is `analysis/probe_summary.md`.

The intended interpretation is modest: this is not a challenge to the whole paper. It is a compact dual-witness boundary check for the paper's large-separation branch approximation.

---

# Minimal hot-ECD dual-witness QFI probe

This is a small sandbox for the paper **"To Cool, or Not to Cool? Displacement Sensing with Hot Quantum States"**.

The goal is deliberately narrow: reproduce the paper's simplified hot-ECD sensing model and check the main branch-approximation boundary with one easy-to-read finite Hilbert-space probe.

No long pipeline. No training loop. No external data dependency.

## What this tests

The paper argues that an ECD hot cat can retain displacement sensitivity because the dominant contribution lives in coherent interference between two displaced thermal branches, while thermal mixedness remains in local/intrabranch degrees of freedom. In the large-separation approximation, the paper gives

```text
F_Q ~= 16 alpha^2 + 4/(2 nbar + 1)
```

where `alpha` is cat displacement and `nbar` is thermal occupation.

This probe asks one simple question:

> When we build a finite-cutoff ECD-style hot cat, does the reduced branch formula track both exact spectral QFI and an independent finite-displacement response witness?

The control parameter is

```text
eta = alpha^2 / (2 nbar + 1)
```

Small `eta` means the displaced thermal branches overlap more strongly. Large `eta` is the regime where the branch-pseudospin approximation should behave better.

## Why dual witness?

The first version compared the reduced formula against exact mixed-state spectral QFI. That is useful, but still QFI-to-QFI.

This version adds a second witness: apply a small displacement to the finite-cutoff state, compute the Uhlmann fidelity between the original and displaced states, and convert the Bures response back into a local QFI estimate:

```text
F_Q_from_response ~= 8 * (1 - sqrt(Fidelity(rho, rho_delta))) / delta^2
```

So the branch formula is checked two ways:

1. **Spectral witness:** exact finite-cutoff mixed-state QFI.
2. **Response witness:** actual finite-displacement state distinguishability.

A strong result is when the branch formula agrees with both witnesses, and the two witnesses agree with each other.

## What the script writes

`probe_hot_ecd_qfi.py` writes these outputs into `analysis/`:

| Output | Meaning |
| --- | --- |
| `fig2_like_protocol_grid.csv` | Grid over cooling time and cat time using the paper's simplified trapped-ion objective. |
| `fig2_like_best_point.csv` | Best grid point for `QFI / (cool_time + cat_time)`. |
| `dual_witness_ecd_sweep.csv` | Row-by-row branch formula vs spectral QFI vs finite Bures-response witness. |
| `dual_witness_summary.csv` | Compact summary grouped by `eta`. |
| `probe_summary.md` | Human-readable summary of the result. |

For compatibility with the first sandbox version, the script also writes `exact_ecd_qfi_sweep.csv` as an alias of the dual-witness sweep.

Optionally, it can also write two simple plots.

## Paper-matched parameters

The reduced protocol grid uses the paper's Fig. 2-style trapped-ion parameters:

```text
initial thermal occupation nbar0 = 10
cooling rate = 1
motional heating rate = 1e-3
spin dephasing rate = 1e-2
```

For those parameters, the reduced model is expected to prefer near-zero cooling and finite cat-generation time.

## Install

```bash
python -m pip install numpy scipy matplotlib
```

`matplotlib` is only needed for `--plots`.

## Run

From this repo folder:

```bash
python probe_hot_ecd_qfi.py
```

With plots:

```bash
python probe_hot_ecd_qfi.py --plots
```

To use the alternative component-normalized state model instead of the postselected map:

```bash
python probe_hot_ecd_qfi.py --component-normalized
```

To change the finite displacement used by the response witness:

```bash
python probe_hot_ecd_qfi.py --finite-signal-delta 0.001
```

## How to read the result

Open:

```text
analysis/probe_summary.md
```

The most important table is the dual-witness check by `eta`.

A clean outcome looks like this:

- larger errors at low `eta`, where branch overlap is not negligible;
- smaller errors as `eta` increases, where the large-separation branch model should become valid;
- spectral QFI and finite-response QFI agreeing with each other;
- a best reduced-model cooling time close to zero for the paper's Fig. 2-style parameters.

That does **not** falsify the paper. It maps the boundary of the assumption in the same language as the paper.

## Minimal claim language

This sandbox checks whether the reduced ECD branch formula is a sufficient predictor of both exact displacement QFI and finite-displacement response in a finite Hilbert-space construction.

It is not a full experimental simulation, not a replacement for the paper's master-equation simulations, and not a broad challenge to hot-state sensing.

The useful boundary is simple:

> The branch formula is strongest when `eta = alpha^2/(2 nbar+1)` is large. When `eta` is small, finite branch overlap and postselection details can matter. The dual-witness check asks whether that boundary appears in both the infinitesimal-QFI witness and the finite-response witness.
