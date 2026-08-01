# Degree-Six Keller Nonproperness Components

This repository contains the Lean 4 formalization and synchronized paper for
a degree-six family of polynomial Keller maps `Fh : C^3 -> C^3`.

For every nonzero deformation polynomial `h : Complex[X]`, the formalization
proves:

- the polynomial coordinate model evaluates exactly to the point map `Fh h`;
- the Jacobian determinant is the constant polynomial `-2`;
- the genuine function-field generic degree is six;
- the map is not a polynomial automorphism;
- the sequence-defined nonproperness set is exactly the finite component union
  the reduced zero locus of `p^6 h(p)`;
- the finite component is closed, irreducible, an affine hypersurface, of
  topological Krull dimension two, dominant over the `p`-line, and contains no
  vertical hyperplane; and
- the actual irreducible-component count is
  `2 + (nonzeroRoots h).card`.

The component count uses mathlib's genuine `irreducibleComponents`.

## Main theorems

`DegreeSixKeller/MainTheorems.lean` exports two parameter-free declarations:

- `DegreeSixKeller.theoremA` certifies CEX-004 and CEX-006 as degree-six
  Keller counterexamples, proves their nonproperness component counts are
  respectively `3` and `2`, and proves they are not polynomially left-right
  equivalent.
- `DegreeSixKeller.theoremB` certifies an explicit product family with count
  `m + 2`, proves its range is infinite, and proves pairwise polynomial
  left-right inequivalence on that range.

All nonproperness and zero-locus equalities are equalities of underlying
reduced sets. No scheme-theoretic multiplicity statement is asserted.

## Read the paper

- `DegreeSixKeller_AF5_Final.pdf` is the final 19-page paper.
- `DegreeSixKeller_AF5_Final.tex` is its LaTeX source.

The paper includes the human proof, the implemented Lean proof route, the
main declaration map, and the exact trust boundary.

## Build

The repository is pinned to:

- Lean `v4.32.2`;
- mathlib commit `905b95818eb32af7874a58b427f50c1711a5e96c`.

With Lean installed through `elan`, build the public artifact with:

```text
lake build DegreeSixKeller DegreeSixKeller.AxiomAudit
```

On PowerShell, the enforced build-and-audit entry point is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\build_and_audit.ps1
```

The audit rejects project-local `axiom`, `sorry`, or `admit`, requires exactly
92 unique `#print axioms` reports, and permits only:

- `propext`;
- `Classical.choice`;
- `Quot.sound`.

## Repository layout

- `DegreeSixKeller.lean`: public umbrella module.
- `DegreeSixKeller/`: the complete transitive production source graph and
  `AxiomAudit.lean`.
- `lean-toolchain`, `lakefile.lean`, `lake-manifest.json`: pinned environment.
- `build_and_audit.ps1`: reproducible enforced audit.
- `PUBLIC_SOURCE_MANIFEST.sha256`: hashes for every other file in this archive.

Two production dependencies retain their historical filenames
`FunctionFieldSpike.lean` and `GenericDegreeGeneratorSpike.lean`. They are
required by the generic-degree proof. All other experimental Spike modules,
checkpoint archives, planning notes, and historical process records have been
excluded from this public package.

