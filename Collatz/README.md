# Collatz as Arithmetic Topology

Two papers and four computational probes exploring the shortened Collatz map $T_3$ as a dynamical system on 2-adic probability measures, with a proposed bridge to Minhyong Kim's arithmetic Chern–Simons theory.

All code is **CC0** (public domain). All results are empirical, computed on finite ranges, and do not assume the Collatz conjecture.

---

## Repository contents

| File | Role |
|------|------|
| `collatz_2adic_probe.py` | **Probe 0a.** First-pass empirical study: tracks $\mu_2(S) = \sum_{x \in S} \lvert x \rvert_2$ under $T_3$ and compares with the 2-adic shift $\sigma$. Records Cesàro compression, ratio windows, collapse to $\{1,2\}$, and the conjugacy gap. |
| `2-Adic_Expectation_Dynamics.py` | **Probe 0b.** Refined expectation-based version of Probe 0a. Switches from $\mu_2$ to the normalized expectation $E_k[f]$ and adds the five-measure vacuum-detection suite ($M_1$–$M_5$), including the Orzechowski four-piece threshold. |
| `p1_collatz_arithmetic.pdf` | **Paper 1.** *Collatz Dynamics as Arithmetic Topology.* Introduces the scalar-shadow holonomy $\mathrm{hol}_{\mathrm{add}}(\gamma_n) = \sum_j v_2(\gamma_n^{(j)})$, the supermartingale conjecture, and the three-way translation table (Collatz / M2KR / Kim). |
| `p1_collatz_arithmetic_probe.py` | Reproduces every table in Paper 1: §5.1 supermartingale sweep, §5.2 $H$ on small trajectories, §5.3 $H$ near the attractor / tail $H$, §5.4 four-piece ($\lvert S_k \rvert \leq 4$) crossing. |
| `p2_collatz_echokey.pdf` | **Paper 2.** *EchoKey Seven-Operator Structural Analysis of the Kim-Holonomy Open Problem.* Assigns each of the seven EchoKey v2 operators (C, R, F, G, S, Π, O) a specific object in the Kim/Collatz setting and tests each against a Paper 1 anchor. |
| `p2_collatz_echokey_probe.py` | Runs the seven structural-consistency tests. Depends on `p1_collatz_arithmetic_probe.py` as its data source. |

---

## The short version

**Object of study.** The shortened Collatz map

$$T_3(n) = \begin{cases} n/2 & n \text{ even} \\ (3n+1)/2 & n \text{ odd} \end{cases}$$

acting on probability measures over $\mathbb{N} \hookrightarrow \mathbb{Z}_2$. The primary observable is $f(n) = \lvert n \rvert_2 = 2^{-v_2(n)}$.

**Paper 1 claim.** The sum-of-valuations quantity $\mathrm{hol}_{\mathrm{add}}(\gamma_n) = \sum_{j=0}^{\lvert \gamma_n \rvert - 1} v_2(\gamma_n^{(j)})$ is a candidate scalar shadow of a $\mathbb{Z}_p$-valued holonomy in Kim's arithmetic Chern–Simons framework. Empirical signatures:

- Pearson $r(\mathrm{hol}_{\mathrm{add}}, \lvert \gamma \rvert) = 0.96$ over $n \leq 100$.
- Attractor signature $\mathrm{hol}_{\mathrm{norm}}(\{1,2\}) = 1/3$.
- Tail-holonomy convergence to $1.18 \pm 0.18$ at $K = 20$.
- Supermartingale signature stable across $N \in \{50, 100, 200, 500, 1000, 2000\}$ with $\sum_k \varepsilon_k < 2.7$.
- Orzechowski four-piece threshold crossed at 96.2% of total collapse time, coinciding with a sharp drop $\Delta \bar v_2 = -0.48$ in the $v_2$-mean.

**Paper 2 claim.** The seven operators of the EchoKey v2 specification, via principled assignments to Kim/Collatz objects, produce predictions matching all Paper 1 anchors within pre-declared tolerances (7/7). The strongest test is Outliers, which recovers the Orzechowski crossing location independently (step $k^* = 76$ vs. Paper 1's step 75) from the $v_2$-mean trajectory alone. This is framed as **structural consistency**, not an internal proof in Kim's framework.

---

## Running the probes

Requirements: Python ≥ 3.8, NumPy. No other dependencies.

```bash
# Probe 0a: first-pass 2-adic mass sweep
python collatz_2adic_probe.py

# Probe 0b: expectation + vacuum detection
python 2-Adic_Expectation_Dynamics.py

# Paper 1 tables (§5.1–§5.4)
python p1_collatz_arithmetic_probe.py
python p1_collatz_arithmetic_probe.py --save       # also writes CSVs to ./results/

# Paper 2 seven-operator verdicts
python p2_collatz_echokey_probe.py
python p2_collatz_echokey_probe.py --save          # also writes verdicts CSV
```

Probe 2 imports Probe 1, so keep them in the same directory.

Each probe is self-contained and deterministic; output tables match the PDFs verbatim.

---

## Reading order

If you are coming to this cold, the recommended order is:

1. Skim `collatz_2adic_probe.py` for the 2-adic machinery and the maps $T_3, \sigma$.
2. Read `2-Adic_Expectation_Dynamics.py` for the expectation framework and the five-measure vacuum suite.
3. Read **Paper 1** (`p1_collatz_arithmetic.pdf`) for the holonomy definition, the M2KR / Kim dictionary, and the empirical results.
4. Run `p1_collatz_arithmetic_probe.py` to reproduce §5 of Paper 1.
5. Read **Paper 2** (`p2_collatz_echokey.pdf`) for the EchoKey operator assignments.
6. Run `p2_collatz_echokey_probe.py` to reproduce the seven verdicts.

---

## Status

Nothing here is a theorem in Kim's framework. Paper 1 lists seven open problems (OP1–OP7) required to lift the scalar shadow to a genuine $\mathbb{Z}_p$-valued invariant; Paper 2 lists six further open problems (OP-EK1–OP-EK6) required to upgrade structural consistency to a Kim-internal proof. The probes are measurement tools, not proof tools.

## License

CC0 1.0 Universal. Use, modify, redistribute freely, with or without attribution.

## Citation

If this work is useful to you:

> JGPTech, *Collatz Dynamics as Arithmetic Topology* and *EchoKey Seven-Operator Structural Analysis of the Kim-Holonomy Open Problem*, 2026. https://github.com/JGPTech/Fun
