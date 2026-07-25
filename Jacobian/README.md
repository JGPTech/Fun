# Local Fidelity, Global Alias

## A pipeline proof and exact generator for the three-dimensional Jacobian counterexample

This repository presents:

1. a self-contained proof of an explicit polynomial map
   \(F:\mathbb{C}^3\rightarrow\mathbb{C}^3\) with constant nonzero Jacobian
   determinant that is not injective; and
2. **ACEG**, an Arbitrary Counterexample Generator implemented independently
   in Python, Julia, and the Wolfram Language.

The proof derives the compact certificate from a marked-factor representation:

> marked state → visible product → gauge normalization → local fidelity →  
> global alias → boundary closure → finite certificate

The central mechanism is simple:

> A polynomial representation may preserve every infinitesimal degree of
> freedom while erasing a discrete global state label. The Jacobian detects
> the first fact. A collision detects the second.

All decisive identities are checked with exact polynomial arithmetic over the
rational numbers. No floating-point inference is used.

---

## The compact certificate

Define \(F=(F_1,F_2,F_3)\) by

$$
F_1(x,y,t)
=(1+xy)^3t+y^2(1+xy)(4+3xy),
$$

$$
F_2(x,y,t)
=y+3x(1+xy)^2t+3xy^2(4+3xy),
$$

$$
F_3(x,y,t)
=2x-3x^2y-x^3t.
$$

Then

$$
\det DF\equiv -2.
$$

The three distinct rational points

$$
p_0=\left(0,0,-\frac14\right),\qquad
p_1=\left(1,-\frac32,\frac{13}{2}\right),\qquad
p_{-1}=\left(-1,\frac32,\frac{13}{2}\right)
$$

have the same image:

$$
F(p_0)=F(p_1)=F(p_{-1})
=\left(-\frac14,0,0\right).
$$

Thus \(F\) has constant nonzero Jacobian determinant but is not injective.
The product map

$$
\left(F,\mathrm{id}_{\mathbb{C}^{\,n-3}}\right)
$$

extends the same certificate to every dimension \(n\geq 3\).

The complete derivation, including gauge control, local fidelity, affine
boundary closure, structural Jacobian argument, direct collision substitution,
and closure audit, is in the proof document.

---

## What ACEG generates

ACEG first derives the base map \(F\) from the marked-factor pipeline. It then
constructs maps

$$
G=B\circ F\circ A,
$$

where \(A\) and \(B\) are compositions of elementary polynomial shears with
Jacobian determinant \(1\).

For every accepted map, ACEG:

- expands all three coordinates exactly;
- recomputes the full symbolic Jacobian determinant and requires \(-2\);
- transports the three rational collision witnesses through \(A^{-1}\);
- verifies the common image by exact substitution;
- computes a canonical SHA-256 hash of the polynomial map;
- rejects duplicate hashes and candidates exceeding configured complexity
  caps; and
- writes a machine-verifiable JSON manifest.

ACEG generates a polynomial-automorphism orbit of certified formulas. It does
**not** claim that the generated maps are inequivalent under polynomial
coordinate changes, and it does not claim a new geometric mechanism for each
formula.

No GPU, numerical solver, or third-party symbolic algebra package is required.

---

## Repository contents

| File | Purpose |
| --- | --- |
| [`Local_Fidelity_Global_Alias_Pipeline_Proof.pdf`](Local_Fidelity_Global_Alias_Pipeline_Proof.pdf) | Typeset proof |
| [`Local_Fidelity_Global_Alias_Pipeline_Proof.tex`](Local_Fidelity_Global_Alias_Pipeline_Proof.tex) | LaTeX source |
| [`aceg.py`](aceg.py) | Python ACEG implementation |
| [`aceg.jl`](aceg.jl) | Julia sister implementation |
| [`aceg.wl`](aceg.wl) | Wolfram Language / Mathematica sister implementation |
| [`aceg_manifest.json`](aceg_manifest.json) | Python-generated example manifest |
| [`aceg_julia_manifest.json`](aceg_julia_manifest.json) | Julia-generated example manifest |
| [`aceg_mathematica_manifest.json`](aceg_mathematica_manifest.json) | Mathematica-generated example manifest |

---

## Quick start

Run all commands from the repository directory.

### Python

Python uses only the standard library.

```powershell
python aceg.py base
python aceg.py generate --count 5 --seed 20260724 --output aceg_manifest.json
python aceg.py verify aceg_manifest.json
```

### Julia

The Julia implementation uses only standard libraries.

```powershell
julia aceg.jl selftest aceg_manifest.json
julia aceg.jl generate --count 5 --seed 20260724 --output aceg_julia_manifest.json
julia aceg.jl verify aceg_julia_manifest.json
```

### Wolfram Language / Mathematica

Use the `key=value` generation syntax on Windows. Some `wolframscript -file`
configurations consume double-dash options before the script receives them.

```powershell
wolframscript -file aceg.wl selftest aceg_manifest.json
wolframscript -file aceg.wl generate count=5 seed=20260724 output=aceg_mathematica_manifest.json
wolframscript -file aceg.wl verify aceg_mathematica_manifest.json
```

The Mathematica generator checks that the exported file exists before printing
`generation complete`.

### Rebuild the proof PDF

With a LaTeX installation containing the packages named in the source:

```powershell
latexmk -pdf Local_Fidelity_Global_Alias_Pipeline_Proof.tex
```

---

## Cross-language verification contract

All three ACEG editions use the manifest schema

```text
jgptech.aceg.manifest.v1
```

and agree on the canonical base-map SHA-256:

```text
ce70ce88ad5ef1553386ebcfc9ff5b4b1c6d7b239defc514cb66c41bc07423c7
```

A verifier reconstructs each stored map from its source and target shears
rather than trusting the expanded formula. It then checks:

1. the manifest schema and pipeline-derived base map;
2. canonical polynomial serialization and SHA-256;
3. reconstruction of \(G=B\circ F\circ A\);
4. exact equality with the stored coordinate polynomials;
5. the full determinant identity \(\det DG\equiv-2\);
6. transport and distinctness of all collision witnesses; and
7. exact equality of their images.

Python, Julia, and Wolfram Language intentionally use their native random
number generators. The same seed reproduces a run within one implementation
but need not choose the same orbit representatives across different languages.
Cross-language agreement is established through the base hash, shared schema,
canonical rational serialization, reconstruction, and exact verification.

---

## Complexity controls

Polynomial composition can grow rapidly with shear depth and degree. ACEG
therefore applies conservative defaults for:

- coordinate term count;
- composition-work estimates;
- Jacobian expansion-work estimates; and
- total candidate attempts.

Candidates that exceed a cap are rejected before acceptance. Increase the
limits cautiously when exploring deeper or higher-degree automorphism
compositions.

---

## Update 

Added a blind marked-factor search for the polynomial Jacobian counterexample check point showing the pipeline can be used in a search algo to find a counter example. The search is blind with respect to the final certificate: it does not assume a coefficient slice, boundary modulus, polynomial chart, degree-seven map, or specific collision witness.

The structural input is the marked factorization

L = a*U + b*V,

Q = c*U**2 + d*U*V + e*V**2,

together with its visible cubic coefficients and the resultant normalization R(L,Q) = 1, which removes the continuous scaling gauge.

---

## Provenance

The explicit degree-seven counterexample was publicly attributed to the Fable
AI system in July 2026 and subsequently discussed in geometric form by
[Terence Tao](https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/)
and
[David E. Speyer](https://sbseminar.wordpress.com/2026/07/20/the-new-counterexample-to-the-jacobian-conjecture/).

This repository does not claim discovery of the displayed map. Its contribution
is an independently organized proof through Jon Poplett's established pipeline,
together with three exact, interoperable ACEG implementations.

- **Primary author and investigator:** Jon Poplett
- **Assistance:** AI-assisted proof construction, algebra checking, software
  implementation, and document preparation
- **Verification standard:** exact rational polynomial arithmetic

---

## Suggested citation

> Jon Poplett, *Local Fidelity, Global Alias: A Pipeline Proof of the
> Three-Dimensional Jacobian Counterexample*, JGPTech Research Note,
> 24 July 2026.
