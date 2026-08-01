import DegreeSixKeller.CriticalImageGeometry
import DegreeSixKeller.PairSpecificComponents
import Mathlib.Algebra.Polynomial.Roots
import Mathlib.Tactic

/-!
# Pair-specific discriminant hypersurfaces

This file closes the two remaining irredundancy obligations for the reduced
CEX-004 and CEX-006 candidates.  The critical image is contained in an exact
nonvertical elimination hypersurface.  Every vertical slice of that
hypersurface remains cut out by a nonzero univariate polynomial in `q`, so no
whole hyperplane `p = alpha` can be contained in the finite component.

The common family polynomial below is the nonvertical factor obtained after
removing the leading factor `p^22 * h(p)^3` from the resultant of `omega` and
its derivative.  For CEX-006 we multiply by the nonzero scalar `-8/9` so that
its normalization agrees with the integral resultant factor and the `q^2`
coefficient is `216`.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Ideal MvPolynomial Polynomial Set Topology
open scoped Polynomial

noncomputable section

/-! ## The exact nonvertical family factor -/

/-- The scalar polynomial expression underlying the nonvertical resultant
factor.  Its arguments are `t = h(p)`, then `p`, `q`, and `r`. -/
def deltaValue (t p q r : Complex) : Complex :=
    243 * p ^ 14 * r ^ 5 * t ^ 3
  + 2187 * p ^ 10 * r ^ 4 * t ^ 2
  - 4860 * p ^ 9 * q * r ^ 3 * t ^ 2
  + 6750 * p ^ 9 * r ^ 2 * t ^ 2
  - 216 * p ^ 8 * q ^ 3 * r ^ 3 * t ^ 2
  + 2700 * p ^ 8 * q ^ 2 * r ^ 2 * t ^ 2
  - 5625 * p ^ 8 * q * r * t ^ 2
  + 3125 * p ^ 8 * t ^ 2
  + 6561 * p ^ 6 * r ^ 3 * t
  - 16038 * p ^ 5 * q * r ^ 2 * t
  + 4050 * p ^ 5 * r * t
  + 1620 * p ^ 4 * q ^ 3 * r ^ 2 * t
  + 7290 * p ^ 4 * q ^ 2 * r * t
  - 6750 * p ^ 4 * q * t
  - 1296 * p ^ 3 * q ^ 4 * r * t
  + 1200 * p ^ 3 * q ^ 3 * t
  + 48 * p ^ 2 * q ^ 6 * r * t
  - 48 * p ^ 2 * q ^ 5 * t
  + 6561 * p ^ 2 * r ^ 2
  - 4374 * p * q * r
  + 3888 * p
  + 243 * q ^ 3 * r
  - 243 * q ^ 2

/-- Embed a univariate polynomial as a polynomial in the first target
coordinate `p`.  This is the three-variable analogue of `polynomialAtP` from
`CriticalImageGeometry`. -/
def polynomialAtP3 (h : Complex[X]) : MvPolynomial (Fin 3) Complex :=
  h.eval₂ MvPolynomial.C (MvPolynomial.X 0)

@[simp]
theorem eval_polynomialAtP3 (h : Complex[X]) (x : C3) :
    MvPolynomial.eval x (polynomialAtP3 h) = h.eval (x 0) := by
  induction h using Polynomial.induction_on' with
  | add p q hp hq =>
      rw [show polynomialAtP3 (p + q) =
          polynomialAtP3 p + polynomialAtP3 q by
        simp [polynomialAtP3]]
      rw [map_add, hp, hq, Polynomial.eval_add]
  | monomial n a =>
      simp [polynomialAtP3]

/-- The nonvertical elimination factor for the general degree-six deformation
`h`.  Specializing this definition at `eta004` gives the exact integral
CEX-004 factor. -/
noncomputable def deltaFamily (h : Complex[X]) :
    MvPolynomial (Fin 3) Complex :=
  let p : MvPolynomial (Fin 3) Complex := MvPolynomial.X (0 : Fin 3)
  let q : MvPolynomial (Fin 3) Complex := MvPolynomial.X (1 : Fin 3)
  let r : MvPolynomial (Fin 3) Complex := MvPolynomial.X (2 : Fin 3)
  let t : MvPolynomial (Fin 3) Complex := polynomialAtP3 h
  MvPolynomial.C 243 * p ^ 14 * r ^ 5 * t ^ 3
    + MvPolynomial.C 2187 * p ^ 10 * r ^ 4 * t ^ 2
    - MvPolynomial.C 4860 * p ^ 9 * q * r ^ 3 * t ^ 2
    + MvPolynomial.C 6750 * p ^ 9 * r ^ 2 * t ^ 2
    - MvPolynomial.C 216 * p ^ 8 * q ^ 3 * r ^ 3 * t ^ 2
    + MvPolynomial.C 2700 * p ^ 8 * q ^ 2 * r ^ 2 * t ^ 2
    - MvPolynomial.C 5625 * p ^ 8 * q * r * t ^ 2
    + MvPolynomial.C 3125 * p ^ 8 * t ^ 2
    + MvPolynomial.C 6561 * p ^ 6 * r ^ 3 * t
    - MvPolynomial.C 16038 * p ^ 5 * q * r ^ 2 * t
    + MvPolynomial.C 4050 * p ^ 5 * r * t
    + MvPolynomial.C 1620 * p ^ 4 * q ^ 3 * r ^ 2 * t
    + MvPolynomial.C 7290 * p ^ 4 * q ^ 2 * r * t
    - MvPolynomial.C 6750 * p ^ 4 * q * t
    - MvPolynomial.C 1296 * p ^ 3 * q ^ 4 * r * t
    + MvPolynomial.C 1200 * p ^ 3 * q ^ 3 * t
    + MvPolynomial.C 48 * p ^ 2 * q ^ 6 * r * t
    - MvPolynomial.C 48 * p ^ 2 * q ^ 5 * t
    + MvPolynomial.C 6561 * p ^ 2 * r ^ 2
    - MvPolynomial.C 4374 * p * q * r
    + MvPolynomial.C 3888 * p
    + MvPolynomial.C 243 * q ^ 3 * r
    - MvPolynomial.C 243 * q ^ 2

@[simp]
theorem deltaFamily_aeval (h : Complex[X]) (b : C3) :
    MvPolynomial.aeval b (deltaFamily h) =
      deltaValue (h.eval (b 0)) (b 0) (b 1) (b 2) := by
  rw [MvPolynomial.aeval_eq_eval]
  simp [deltaFamily, deltaValue]

/-- The family factor vanishes identically on the finite-multiple-root
parametrization. -/
theorem deltaFamily_criticalTarget
    (h : Complex[X]) (p s : Complex) (hs : s ≠ 0) :
    MvPolynomial.aeval
      (criticalTarget h p s) (deltaFamily h) = 0 := by
  rw [deltaFamily_aeval]
  simp [deltaValue, criticalQ, criticalR]
  field_simp [hs]
  ring

/-- Exact CEX-004 nonvertical resultant factor. -/
noncomputable def delta004 : MvPolynomial (Fin 3) Complex :=
  deltaFamily eta004

/-- Exact CEX-006 nonvertical resultant factor, normalized integrally. -/
noncomputable def delta006 : MvPolynomial (Fin 3) Complex :=
  MvPolynomial.C (-8 / 9 : Complex) * deltaFamily eta006

/-- The CEX-004 factor vanishes on its critical parametrization. -/
theorem delta004_criticalTarget
    (p s : Complex) (hs : s ≠ 0) :
    MvPolynomial.aeval
      (criticalTarget eta004 p s) delta004 = 0 := by
  simpa [delta004] using deltaFamily_criticalTarget eta004 p s hs

/-- The CEX-006 factor vanishes on its critical parametrization. -/
theorem delta006_criticalTarget
    (p s : Complex) (hs : s ≠ 0) :
    MvPolynomial.aeval
      (criticalTarget eta006 p s) delta006 = 0 := by
  rw [delta006, map_mul, deltaFamily_criticalTarget eta006 p s hs]
  simp

/-! ## Algebraic zero loci and closure -/

/-- The hypersurface cut out by one multivariate polynomial. -/
def deltaZeroLocus
    (f : MvPolynomial (Fin 3) Complex) : Set C3 :=
  MvPolynomial.zeroLocus Complex (Ideal.span {f})

/-- A one-polynomial zero locus is Zariski closed. -/
theorem deltaZeroLocus_isClosed
    (f : MvPolynomial (Fin 3) Complex) :
    IsClosed (zariskiLift (deltaZeroLocus f)) := by
  simpa [deltaZeroLocus, zariskiLift] using
    (zariskiLiftAffine_zeroLocus_isClosed
      (Fin 3) (Ideal.span ({f} : Set (MvPolynomial (Fin 3) Complex))))

/-- Membership in a one-polynomial zero locus is exactly vanishing of that
polynomial. -/
theorem mem_deltaZeroLocus_iff
    (f : MvPolynomial (Fin 3) Complex) (b : C3) :
    b ∈ deltaZeroLocus f ↔ MvPolynomial.aeval b f = 0 := by
  rw [deltaZeroLocus, MvPolynomial.zeroLocus_span]
  simp

/-- A closed hypersurface containing every directly parametrized critical
point also contains the Zariski closure of the critical image. -/
theorem finiteComponent_subset_delta_of_criticalTarget
    (h : Complex[X]) (f : MvPolynomial (Fin 3) Complex)
    (hVanish : ∀ p s : Complex, s ≠ 0 ->
      MvPolynomial.aeval (criticalTarget h p s) f = 0) :
    finiteComponent h ⊆ deltaZeroLocus f := by
  intro b hb
  have hImage :
      zariskiLift (criticalImage h) ⊆
        zariskiLift (deltaZeroLocus f) := by
    intro z hz
    change ofZariskiC3 z ∈ criticalImage h at hz
    rcases hz with ⟨p, s, hs, hEq⟩
    change ofZariskiC3 z ∈ deltaZeroLocus f
    rw [mem_deltaZeroLocus_iff]
    rw [hEq]
    exact hVanish p s hs
  have hClosure :
      closure (zariskiLift (criticalImage h)) ⊆
        zariskiLift (deltaZeroLocus f) :=
    closure_minimal hImage (deltaZeroLocus_isClosed f)
  change toZariskiC3 b ∈ closure (zariskiLift (criticalImage h)) at hb
  have hz := hClosure hb
  change ofZariskiC3 (toZariskiC3 b) ∈ deltaZeroLocus f at hz
  simpa only [ofZariskiC3_toZariskiC3] using hz

/-- The CEX-004 finite component lies in `V(delta004)`. -/
theorem finiteComponent004_subset_delta :
    finiteComponent eta004 ⊆ deltaZeroLocus delta004 :=
  finiteComponent_subset_delta_of_criticalTarget eta004 delta004
    delta004_criticalTarget

/-- The CEX-006 finite component lies in `V(delta006)`. -/
theorem finiteComponent006_subset_delta :
    finiteComponent eta006 ⊆ deltaZeroLocus delta006 :=
  finiteComponent_subset_delta_of_criticalTarget eta006 delta006
    delta006_criticalTarget

/-! ## Nonzero vertical slices -/

/-- The point `(alpha,q,0)` in target affine space. -/
def deltaSlicePoint (alpha q : Complex) : C3 :=
  ![alpha, q, 0]

/-- The `r = 0`, `p = alpha` slice of the CEX-004 factor. -/
noncomputable def delta004Slice (alpha : Complex) : Complex[X] :=
    Polynomial.monomial 0
      (3125 * alpha ^ 8 * (1 + 4 * alpha) ^ 2 + 3888 * alpha)
  + Polynomial.monomial 1 (-6750 * alpha ^ 4 * (1 + 4 * alpha))
  + Polynomial.monomial 2 (-243 : Complex)
  + Polynomial.monomial 3 (1200 * alpha ^ 3 * (1 + 4 * alpha))
  + Polynomial.monomial 5 (-48 * alpha ^ 2 * (1 + 4 * alpha))

/-- The `r = 0`, `p = alpha` slice of the normalized CEX-006 factor. -/
noncomputable def delta006Slice (alpha : Complex) : Complex[X] :=
    Polynomial.monomial 0 (-6250 * alpha ^ 8 - 3456 * alpha)
  + Polynomial.monomial 1 (-9000 * alpha ^ 4)
  + Polynomial.monomial 2 (216 : Complex)
  + Polynomial.monomial 3 (1600 * alpha ^ 3)
  + Polynomial.monomial 5 (-64 * alpha ^ 2)

/-- Evaluation of the CEX-004 slice agrees with evaluation of `delta004` at
`(alpha,q,0)`. -/
theorem delta004Slice_eval (alpha q : Complex) :
    (delta004Slice alpha).eval q =
      MvPolynomial.aeval (deltaSlicePoint alpha q) delta004 := by
  rw [delta004, deltaFamily_aeval]
  simp [delta004Slice, deltaSlicePoint, deltaValue, eta004_eval]
  ring

/-- Evaluation of the CEX-006 slice agrees with evaluation of `delta006` at
`(alpha,q,0)`. -/
theorem delta006Slice_eval (alpha q : Complex) :
    (delta006Slice alpha).eval q =
      MvPolynomial.aeval (deltaSlicePoint alpha q) delta006 := by
  rw [delta006, map_mul, deltaFamily_aeval]
  simp [delta006Slice, deltaSlicePoint, deltaValue, eta006_eval]
  ring

/-- The fixed quadratic coefficient in every CEX-004 vertical slice. -/
theorem delta004Slice_coeff_two (alpha : Complex) :
    (delta004Slice alpha).coeff 2 = (-243 : Complex) := by
  simp only [delta004Slice, Polynomial.coeff_add,
    Polynomial.coeff_monomial]
  norm_num

/-- The fixed quadratic coefficient in every CEX-006 vertical slice. -/
theorem delta006Slice_coeff_two (alpha : Complex) :
    (delta006Slice alpha).coeff 2 = (216 : Complex) := by
  simp only [delta006Slice, Polynomial.coeff_add,
    Polynomial.coeff_monomial]
  norm_num

/-- Every CEX-004 vertical slice is a nonzero polynomial in `q`. -/
theorem delta004Slice_ne_zero (alpha : Complex) :
    delta004Slice alpha ≠ 0 := by
  intro hZero
  have hCoeff : (-243 : Complex) = 0 := by
    calc
      (-243 : Complex) = (delta004Slice alpha).coeff 2 :=
        (delta004Slice_coeff_two alpha).symm
      _ = (0 : Complex[X]).coeff 2 := congrArg (fun f : Complex[X] => f.coeff 2) hZero
      _ = 0 := by simp
  norm_num at hCoeff

/-- Every CEX-006 vertical slice is a nonzero polynomial in `q`. -/
theorem delta006Slice_ne_zero (alpha : Complex) :
    delta006Slice alpha ≠ 0 := by
  intro hZero
  have hCoeff : (216 : Complex) = 0 := by
    calc
      (216 : Complex) = (delta006Slice alpha).coeff 2 :=
        (delta006Slice_coeff_two alpha).symm
      _ = (0 : Complex[X]).coeff 2 := congrArg (fun f : Complex[X] => f.coeff 2) hZero
      _ = 0 := by simp
  norm_num at hCoeff

/-- A nonzero complex polynomial is nonzero at some complex input. -/
theorem exists_eval_ne_zero
    (f : Complex[X]) (hf : f ≠ 0) :
    ∃ q : Complex, f.eval q ≠ 0 := by
  by_contra hExists
  have hAll : ∀ q : Complex, f.eval q = 0 := by
    intro q
    by_contra hq
    exact hExists ⟨q, hq⟩
  have hRoots : {q : Complex | f.IsRoot q} = Set.univ := by
    ext q
    simp [Polynomial.IsRoot, hAll q]
  have hFinite := Polynomial.finite_setOf_isRoot hf
  rw [hRoots] at hFinite
  exact (Set.infinite_univ : Set.Infinite (Set.univ : Set Complex)) hFinite

/-- Every vertical hyperplane has a point outside `V(delta004)`. -/
theorem exists_in_pHyperplane_not_delta004
    (alpha : Complex) :
    ∃ b ∈ pHyperplane alpha, b ∉ deltaZeroLocus delta004 := by
  obtain ⟨q, hq⟩ := exists_eval_ne_zero
    (delta004Slice alpha) (delta004Slice_ne_zero alpha)
  let b : C3 := deltaSlicePoint alpha q
  refine ⟨b, ?_, ?_⟩
  · simp [b, deltaSlicePoint, pHyperplane]
  · intro hb
    have hZero := (mem_deltaZeroLocus_iff delta004 b).1 hb
    have hSlice : (delta004Slice alpha).eval q = 0 := by
      simpa [b] using (delta004Slice_eval alpha q).trans hZero
    exact hq hSlice

/-- Every vertical hyperplane has a point outside `V(delta006)`. -/
theorem exists_in_pHyperplane_not_delta006
    (alpha : Complex) :
    ∃ b ∈ pHyperplane alpha, b ∉ deltaZeroLocus delta006 := by
  obtain ⟨q, hq⟩ := exists_eval_ne_zero
    (delta006Slice alpha) (delta006Slice_ne_zero alpha)
  let b : C3 := deltaSlicePoint alpha q
  refine ⟨b, ?_, ?_⟩
  · simp [b, deltaSlicePoint, pHyperplane]
  · intro hb
    have hZero := (mem_deltaZeroLocus_iff delta006 b).1 hb
    have hSlice : (delta006Slice alpha).eval q = 0 := by
      simpa [b] using (delta006Slice_eval alpha q).trans hZero
    exact hq hSlice

/-! ## Pair-specific endpoints -/

/-- No vertical hyperplane is contained in the CEX-004 finite component. -/
theorem cex004_noVerticalHyperplane :
    NoVerticalHyperplaneInFiniteComponent eta004 := by
  intro alpha hSubset
  obtain ⟨b, hbVertical, hbNotDelta⟩ :=
    exists_in_pHyperplane_not_delta004 alpha
  exact hbNotDelta
    (finiteComponent004_subset_delta (hSubset hbVertical))

/-- No vertical hyperplane is contained in the CEX-006 finite component. -/
theorem cex006_noVerticalHyperplane :
    NoVerticalHyperplaneInFiniteComponent eta006 := by
  intro alpha hSubset
  obtain ⟨b, hbVertical, hbNotDelta⟩ :=
    exists_in_pHyperplane_not_delta006 alpha
  exact hbNotDelta
    (finiteComponent006_subset_delta (hSubset hbVertical))

/-- The reduced CEX-004 candidate now has an unconditional component count. -/
theorem cex004_candidate_componentCount :
    algebraicComponentCount reducedCandidate004 = 3 :=
  reducedCandidate004_componentCount
    cex004_finiteComponentIrreducible
    cex004_noVerticalHyperplane

/-- The reduced CEX-006 candidate now has an unconditional component count. -/
theorem cex006_candidate_componentCount :
    algebraicComponentCount reducedCandidate006 = 2 :=
  reducedCandidate006_componentCount
    cex006_finiteComponentIrreducible
    cex006_noVerticalHyperplane

end

end DegreeSixKeller
