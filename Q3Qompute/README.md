# EchoKey — Unified Reversible Coin (TSP & Chemistry)

**One-liner:** A small-N, measurement-free, **reversible** core (Bennett compute→use→uncompute) driving an **inhomogeneous coined walk**. We show it twice: **TSP** (anchored tours, adjacent swaps, exact ΔL) and **Chemistry** (finite sandbox, exact ΔE). No hype. No hidden oracles. No erasure.

> **Expectation (up front):** This is a **truth machine**, not a speed machine. Tiny instances, exact math, transparent logs. We make **no scaling or advantage claims**.

---

## Repository layout

```
./
├─ chem_track/
│  ├─ chem_reversible_coin.py
│  └─ chem_YYYYMMDD_HHMMSS/               # per-run artifacts (example below)
│     ├─ metrics.json
│     ├─ lut_int.csv                      # ΔE quantized (two’s complement ints)
│     ├─ lut_ev.csv                       # ΔE in eV
│     ├─ gate_counts.csv
│     ├─ U_reversible_chem.qpy            # QPY dump of the reversible core
│     ├─ fig_*.png                        # optional figures (if matplotlib available)
│     └─ manifest.json                    # list of generated files
├─ tsp_track/
│  ├─ tsp_reversible_coin.py
│  └─ tsp_YYYYMMDD_HHMMSS/                # per-run artifacts
│     ├─ tsp_metrics.json
│     ├─ tsp_routes_summary.csv
│     ├─ manifest.json
│     ├─ fig_*.png                        # optional figures (if matplotlib available)
│     └─ tsp_U.qpy                        # QPY dump of coined-walk unitary (when --prove)
├─ README.md
├─ LICENSE.md                             # CC0-1.0 (Public Domain)
├─ Ethics.md                              # short ethics doc
├─ USE_POLICY.md                          # allowed/prohibited uses
├─ EchoKey_Unified_Reversible_Coin_Framework.pdf     # main write-up (PDF)
└─ Ethics.pdf                                         # ethics supplement (PDF)
# (You will also add one dependency file in repo root: requirements.txt or environment.yml)
```

---

## Install

Python **3.10+** recommended.

### Option A: `requirements.txt` (recommended for most)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: `environment.yml` (conda/mamba users)

```bash
mamba env create -f environment.yml
mamba activate echokey
```

**Packages expected** (pinned in your dependency file):

* `qiskit>=1.4,<2` • `qiskit-aer>=0.15` (for local sims) • `numpy` • `matplotlib`
* For chemistry Nature path (optional): `qiskit-nature`, `pyscf` (if using `--nature`)

> If Aer isn’t installed, Qiskit’s basic simulator will still run; figures are optional and saved only if `matplotlib` is present.

---

## Quickstart

### TSP (N=4 demo, with reversibility proof + artifacts)

```bash
# Save outputs under a track folder (recommended for neatness)
python tsp_track/tsp_reversible_coin.py \
  --N 4 --steps 8 --beta 0.25 --stay_gap 3.0 \
  --routes 5 --repeats 2 --prove \
  --outdir tsp_track/tsp_20250913_074447
```

### Chemistry (toy sandbox; Nature optional)

```bash
# Try Nature, fallback to mock automatically; save artifacts under chem_track/
python chem_track/chem_reversible_coin.py --nature \
  --geomA 0.735 --geomB 0.900 --seed 7 \
  --outdir chem_track/chem_20250913_073421

# Or force mock energies
python chem_track/chem_reversible_coin.py --mock --seed 7 \
  --outdir chem_track/chem_20250913_073421
```

> By default the scripts can also save under `runs/<timestamp>/`. We show `chem_track/…` / `tsp_track/…` to keep video assets organized per track.

---

## CLI overview

### Shared flags

| Flag         | Meaning                                                 |
| ------------ | ------------------------------------------------------- |
| `--steps`    | Number of coined-walk steps                             |
| `--beta`     | Inverse-temperature for softmax coin (internally ramps) |
| `--stay_gap` | Margin below the best move for the “stay” coin outcome  |
| `--seed`     | RNG base seed                                           |
| `--outdir`   | Directory for artifacts (metrics/CSVs/figures/QPY)      |

### TSP-specific

| Flag                                           | Meaning                                                               |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| `--N`                                          | Number of cities (anchored at 0); tiny $4–8$                          |
| `--routes` / `--repeats` / `--shots`           | Multi-run harness controls                                            |
| `--init {exp,uniform,topk}` `--alpha` `--topk` | Position init bias modes                                              |
| `--wprofile {default,robust}`                  | Weight profile for EchoKey-7                                          |
| `--prove`                                      | Build unitary $U$, show `init/compute/full/U†U`, save proof artifacts |
| `--skip_exact`                                 | Skip Held–Karp baseline (no gap/hit metrics)                          |

### Chemistry-specific

| Flag                                 | Meaning                                                               |
| ------------------------------------ | --------------------------------------------------------------------- |
| `--nature`                           | Try Qiskit Nature (VQE + QEOM/VQD); fallback to mock if not available |
| `--mock`                             | Force mock energies                                                   |
| `--energies EN_A EN_B ENp1_A ENp1_B` | Manual energy override (eV)                                           |
| `--bits` `--delta_ev` `--lambda`     | Δ register width (two’s complement), quantization step (eV), RY angle |
| `--geomA` `--geomB`                  | H–H distances for H₂ (Å) when using Nature                            |

Run `-h` on either script to see the full list.

---

## What you get (artifacts)

### Chemistry (`chem_track/chem_YYYYMMDD_HHMMSS/`)

* `metrics.json` — args, energies + provenance, ΔE LUT stats, zero-register probabilities, gate depth/opcounts, `fidelity_UdaggerU`.
* `lut_ev.csv` / `lut_int.csv` — ΔE tables (eV and quantized two’s-complement ints).
* `gate_counts.csv` — decomposed op histogram for the reversible stage.
* `U_reversible_chem.qpy` — QPY dump (loadable by Qiskit) of the Bennett sandwich.
* `fig_*.png` — video-friendly visuals (energies table, ΔE heatmaps, zero-prob bars, gate histogram, circuit).
* `manifest.json` — index of all files saved.

### TSP (`tsp_track/tsp_YYYYMMDD_HHMMSS/`)

* `tsp_metrics.json` — args, per-route summaries (best L, gaps, hits, invalid), aggregates.
* `tsp_routes_summary.csv` — per-route table, ready to chart.
* `manifest.json` — list of outputs.
* If `--prove`:

  * `tsp_proof_metrics.json` — `P(coin==0)` across `init/compute/full/U†U`, gate depth/opcounts, `fidelity_UdaggerU`.
  * `tsp_gate_counts.csv`, `fig_tsp_circuit_U.png`, `fig_tsp_coin_zero_probs.png`, `tsp_U.qpy`.
  * Plus `fig_route0_distance_heatmap.png`, `route0_D.csv` (for the proof route).

> Figures are created only if `matplotlib` is installed. All numeric artifacts (JSON/CSV/QPY) are always saved.

---

## Interpreting the proof readouts (low-key)

* **Reversibility:**
  `fidelity_UdaggerU ≈ 1.0` (within FP tolerance) and registers that were temporarily populated during “compute” (e.g., Δ or coin) **return to zero** after the sandwich / after `U†U`.
* **Padded mass:**
  Padded states self-map and don’t leak; invalid fractions in TSP should be low and stable.
* **TSP gaps & hits:**
  Gaps are against Held–Karp (when not skipped). Hits count the exact optimum *or its reverse* (symmetry).

---

## Ethics & Use Policy (quick reminders)

* **Allowed:** pedagogy, reversible-logic demos, toy routing on synthetic data, benign chemistry sandboxes.
* **Prohibited:** people-routing/surveillance; hazardous chemical/biological design; misleading scaling/advantage/energy claims.
* **Review required:** moving to real data, coupling to decision systems, any synthesis/toxicity work.

See **Ethics.md** and **USE\_POLICY.md** for the full no-nonsense policy.

---

## Reproducibility & logging

* Seeds and key args are recorded in `metrics.json` / `tsp_metrics.json`.
* Circuits are exportable/reloadable via **QPY** (`*_U.qpy`).
* Each run folder includes a **manifest** for easy video assembly.
* If you want environment pinning, use the root **dependency file** (`requirements.txt` or `environment.yml`) and keep it updated with exact versions.

---

## Troubleshooting

* **Qiskit Nature missing?** Use `--mock` or add `qiskit-nature` + `pyscf` to your dependency file.
* **No figures?** Install `matplotlib` (or skip—metrics/CSVs/QPY are sufficient).
* **Aer not found?** `pip install qiskit-aer` (or run on the default simulator, slower but fine for tiny N).
* **Windows paths:** avoid deep nested folders or enable long path support if needed.

---

## Cite / License

* **License:** CC0-1.0 (Public Domain). No rights reserved.
* If you reference this work, cite the framework PDF (**EchoKey\_Unified\_Reversible\_Coin\_Framework.pdf**), the ethics supplement (**Ethics.pdf**), and the repository.

---

**Contact / Issues:** Open a GitHub issue. PRs for better logging, figures, or additional tiny sandboxes welcome—keep it explicit and reversible.
