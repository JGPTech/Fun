import DegreeSixKeller.ResultantCertificates
import Mathlib.Topology.Algebra.MvPolynomial

/-!
# Universal geometry of the finite multiple-root component

This module packages the family-wide consequences of the critical-image and
resultant machinery.  In particular, the finite component is irreducible,
dominates the first-coordinate line, and contains no vertical hyperplane.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Filter Ideal MvPolynomial Polynomial Set Topology
open scoped Polynomial

noncomputable section

/-! ## Irreducibility and the universal eliminant -/

/-- The finite multiple-root component is irreducible for every deformation
polynomial. -/
theorem finiteComponent_isIrreducible (h : Complex[X]) :
    IsIrreducible (zariskiLift (finiteComponent h)) :=
  finiteComponent_isIrreducible_of_criticalImage h
    (criticalImage_isIrreducible h)

/-- The finite component is contained in the zero locus of the universal
nonvertical eliminant. -/
theorem finiteComponent_subset_deltaFamily (h : Complex[X]) :
    finiteComponent h ⊆ deltaZeroLocus (deltaFamily h) :=
  finiteComponent_subset_delta_of_criticalTarget h (deltaFamily h)
    (deltaFamily_criticalTarget h)

/-- Universal specialization of the reduced resultant certificate. -/
theorem reducedResultantFamily_factorization
    (h : Complex[X]) (p q r : Complex) :
    resultant (cubicRemainder p q r)
        (quadraticRemainder (aCoeff h p) p q r) 3 2 =
      27 * p ^ 10 *
        MvPolynomial.aeval (![p, q, r] : C3) (deltaFamily h) := by
  rw [reducedResultant_factorization, discriminantCore_aCoeff,
    deltaFamily_aeval]
  simp
  ring

/-- Off the leading-coefficient locus, every point of the finite component is
already in the directly parametrized critical image. -/
theorem finiteComponent_mem_criticalImage_of_aCoeff_ne_zero
    (h : Complex[X]) {b : C3}
    (hb : b ∈ finiteComponent h)
    (ha : aCoeff h (b 0) ≠ 0) :
    b ∈ criticalImage h := by
  have hDelta := finiteComponent_subset_deltaFamily h hb
  have hDeltaZero : MvPolynomial.aeval b (deltaFamily h) = 0 :=
    (mem_deltaZeroLocus_iff (deltaFamily h) b).1 hDelta
  have hp : b 0 ≠ 0 := by
    intro hp
    apply ha
    simp [hp, aCoeff]
  have hbvec : (![b 0, b 1, b 2] : C3) = b := by
    funext i
    fin_cases i <;> simp
  have hDeltaVec :
      MvPolynomial.aeval (![b 0, b 1, b 2] : C3) (deltaFamily h) = 0 := by
    rw [hbvec]
    exact hDeltaZero
  have hRes : resultant (cubicRemainder (b 0) (b 1) (b 2))
      (quadraticRemainder (aCoeff h (b 0)) (b 0) (b 1) (b 2)) 3 2 = 0 := by
    rw [reducedResultantFamily_factorization]
    simp [hDeltaVec]
  obtain ⟨s, hCubic, hQuadratic⟩ :=
    exists_common_root_of_reduced_resultant_eq_zero
      (aCoeff h (b 0)) (b 0) (b 1) (b 2) hp hRes
  obtain ⟨hOmega, hDerivative⟩ :=
    common_root_of_remainders h (b 0) (b 1) (b 2) s hp hCubic hQuadratic
  have hs := common_root_ne_zero h (b 0) (b 1) (b 2) s hDerivative
  refine ⟨b 0, s, hs, ?_⟩
  have hEq := eq_criticalTarget_of_common_root h
    (b 0) (b 1) (b 2) s hs hOmega hDerivative
  calc
    b = (![b 0, b 1, b 2] : C3) := hbvec.symm
    _ = criticalTarget h (b 0) s := hEq

/-- Away from `p = 0`, every zero of the universal eliminant is directly
parametrized by a finite multiple root.  Unlike the preceding theorem, this
does not require the degree-six leading coefficient to be nonzero. -/
theorem deltaFamily_mem_criticalImage_of_p_ne_zero
    (h : Complex[X]) {b : C3}
    (hb : b ∈ deltaZeroLocus (deltaFamily h))
    (hp : b 0 ≠ 0) :
    b ∈ criticalImage h := by
  have hDeltaZero : MvPolynomial.aeval b (deltaFamily h) = 0 :=
    (mem_deltaZeroLocus_iff (deltaFamily h) b).1 hb
  have hbvec : (![b 0, b 1, b 2] : C3) = b := by
    funext i
    fin_cases i <;> simp
  have hDeltaVec :
      MvPolynomial.aeval (![b 0, b 1, b 2] : C3) (deltaFamily h) = 0 := by
    rw [hbvec]
    exact hDeltaZero
  have hRes : resultant (cubicRemainder (b 0) (b 1) (b 2))
      (quadraticRemainder (aCoeff h (b 0)) (b 0) (b 1) (b 2)) 3 2 = 0 := by
    rw [reducedResultantFamily_factorization]
    simp [hDeltaVec]
  obtain ⟨s, hCubic, hQuadratic⟩ :=
    exists_common_root_of_reduced_resultant_eq_zero
      (aCoeff h (b 0)) (b 0) (b 1) (b 2) hp hRes
  obtain ⟨hOmega, hDerivative⟩ :=
    common_root_of_remainders h (b 0) (b 1) (b 2) s hp hCubic hQuadratic
  have hs := common_root_ne_zero h (b 0) (b 1) (b 2) s hDerivative
  refine ⟨b 0, s, hs, ?_⟩
  have hEq := eq_criticalTarget_of_common_root h
    (b 0) (b 1) (b 2) s hs hOmega hDerivative
  calc
    b = (![b 0, b 1, b 2] : C3) := hbvec.symm
    _ = criticalTarget h (b 0) s := hEq

/-! ## Universal nonvertical slices -/

/-- The `r = 0`, `p = alpha` slice of the family eliminant. -/
noncomputable def deltaFamilySlice
    (h : Complex[X]) (alpha : Complex) : Complex[X] :=
    Polynomial.monomial 0
      (3125 * alpha ^ 8 * (h.eval alpha) ^ 2 + 3888 * alpha)
  + Polynomial.monomial 1 (-6750 * alpha ^ 4 * h.eval alpha)
  + Polynomial.monomial 2 (-243 : Complex)
  + Polynomial.monomial 3 (1200 * alpha ^ 3 * h.eval alpha)
  + Polynomial.monomial 5 (-48 * alpha ^ 2 * h.eval alpha)

/-- Evaluation of the universal slice agrees with evaluation of the family
eliminant at `(alpha,q,0)`. -/
theorem deltaFamilySlice_eval
    (h : Complex[X]) (alpha q : Complex) :
    (deltaFamilySlice h alpha).eval q =
      MvPolynomial.aeval (deltaSlicePoint alpha q) (deltaFamily h) := by
  rw [deltaFamily_aeval]
  simp [deltaFamilySlice, deltaSlicePoint, deltaValue]
  ring

/-- Every universal vertical slice has the fixed nonzero quadratic
coefficient `-243`. -/
theorem deltaFamilySlice_coeff_two
    (h : Complex[X]) (alpha : Complex) :
    (deltaFamilySlice h alpha).coeff 2 = (-243 : Complex) := by
  simp only [deltaFamilySlice, Polynomial.coeff_add,
    Polynomial.coeff_monomial]
  norm_num

/-- Every vertical slice of the family eliminant is a nonzero polynomial in
the second target coordinate. -/
theorem deltaFamilySlice_ne_zero
    (h : Complex[X]) (alpha : Complex) :
    deltaFamilySlice h alpha ≠ 0 := by
  intro hZero
  have hCoeff : (-243 : Complex) = 0 := by
    calc
      (-243 : Complex) = (deltaFamilySlice h alpha).coeff 2 :=
        (deltaFamilySlice_coeff_two h alpha).symm
      _ = (0 : Complex[X]).coeff 2 :=
        congrArg (fun f : Complex[X] => f.coeff 2) hZero
      _ = 0 := by simp
  norm_num at hCoeff

/-- Every vertical hyperplane has a point outside the universal eliminant. -/
theorem exists_in_pHyperplane_not_deltaFamily
    (h : Complex[X]) (alpha : Complex) :
    ∃ b ∈ pHyperplane alpha, b ∉ deltaZeroLocus (deltaFamily h) := by
  obtain ⟨q, hq⟩ := exists_eval_ne_zero
    (deltaFamilySlice h alpha) (deltaFamilySlice_ne_zero h alpha)
  let b : C3 := deltaSlicePoint alpha q
  refine ⟨b, ?_, ?_⟩
  · simp [b, deltaSlicePoint, pHyperplane]
  · intro hb
    have hZero := (mem_deltaZeroLocus_iff (deltaFamily h) b).1 hb
    have hSlice : (deltaFamilySlice h alpha).eval q = 0 := by
      simpa [b] using (deltaFamilySlice_eval h alpha q).trans hZero
    exact hq hSlice

/-- No vertical hyperplane is contained in the finite component, uniformly in
the deformation polynomial. -/
theorem finiteComponent_noVerticalHyperplane
    (h : Complex[X]) :
    NoVerticalHyperplaneInFiniteComponent h := by
  intro alpha hSubset
  obtain ⟨b, hbVertical, hbNotDelta⟩ :=
    exists_in_pHyperplane_not_deltaFamily h alpha
  exact hbNotDelta
    (finiteComponent_subset_deltaFamily h (hSubset hbVertical))

/-! ## Exact hypersurface description -/

/-- The identity from Euclidean affine space to the type carrying its affine
Zariski topology is continuous. -/
theorem continuous_toZariskiC3_euclidean :
    Continuous toZariskiC3 := by
  rw [continuous_iff_isClosed]
  intro Z hZ
  have hRawClosed :
      @IsClosed C3 (affineZariskiTopology (Fin 3))
        (toZariskiC3 ⁻¹' Z) := by
    letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
    exact hZ.preimage
      (WithTopology.continuous_toTopology (affineZariskiTopology (Fin 3)))
  rcases (isClosed_affineZariski_iff (Fin 3) _).mp hRawClosed with ⟨I, hI⟩
  rw [hI]
  change IsClosed {x : C3 | ∀ p ∈ I, MvPolynomial.aeval x p = 0}
  simp only [MvPolynomial.aeval_eq_eval]
  rw [show {x : C3 | ∀ p ∈ I, MvPolynomial.eval x p = 0} =
      ⋂ p : MvPolynomial (Fin 3) Complex,
        ⋂ (_hp : p ∈ I), {x : C3 | MvPolynomial.eval x p = 0} by
    ext x
    simp]
  exact isClosed_iInter fun p => isClosed_iInter fun _hp =>
    isClosed_eq (MvPolynomial.continuous_eval p) continuous_const

/-- A first-coordinate-zero point lies on the universal eliminant precisely
on the two reduced branches `q = 0` and `q*r = 1`. -/
theorem mem_deltaZeroLocus_deltaFamily_of_p_zero_iff
    (h : Complex[X]) (q r : Complex) :
    (![0, q, r] : C3) ∈ deltaZeroLocus (deltaFamily h) ↔
      q = 0 ∨ q * r = 1 := by
  rw [mem_deltaZeroLocus_iff, deltaFamily_aeval]
  simp [deltaValue]
  constructor
  · intro hZero
    have hFactor : q ^ 2 * (q * r - 1) = 0 := by
      linear_combination hZero / 243
    rcases mul_eq_zero.mp hFactor with hq | hqr
    · exact Or.inl (sq_eq_zero_iff.mp hq)
    · exact Or.inr (sub_eq_zero.mp hqr)
  · rintro (rfl | hqr)
    · ring
    · calc
        243 * q ^ 3 * r - 243 * q ^ 2 =
            243 * q ^ 2 * (q * r - 1) := by ring
        _ = 0 := by rw [hqr]; ring

/-- First coordinate used to approach the exceptional line `p=q=0` from the
critical parametrization. -/
def finiteLineApproxP (r ε : Complex) : Complex :=
  ε ^ 2 - r * ε ^ 3

/-- Polynomial model for the second coordinate of the exceptional-line
approximation. -/
def finiteLineApproxQModel
    (h : Complex[X]) (r ε : Complex) : Complex :=
  ε ^ 8 * (1 - r * ε) ^ 6 * h.eval (finiteLineApproxP r ε) +
    4 * ε - 3 * r * ε ^ 2

/-- Polynomial model for the third coordinate of the exceptional-line
approximation. -/
def finiteLineApproxRModel
    (h : Complex[X]) (r ε : Complex) : Complex :=
  r - (2 / 3 : Complex) * ε ^ 6 * (1 - r * ε) ^ 6 *
    h.eval (finiteLineApproxP r ε)

theorem criticalQ_finiteLineApprox
    (h : Complex[X]) (r ε : Complex) (hε : ε ≠ 0) :
    criticalQ h (finiteLineApproxP r ε) ε⁻¹ =
      finiteLineApproxQModel h r ε := by
  simp only [criticalQ, finiteLineApproxP, finiteLineApproxQModel]
  field_simp [hε]
  ring

theorem criticalR_finiteLineApprox
    (h : Complex[X]) (r ε : Complex) (hε : ε ≠ 0) :
    criticalR h (finiteLineApproxP r ε) ε⁻¹ =
      finiteLineApproxRModel h r ε := by
  simp only [criticalR, finiteLineApproxP, finiteLineApproxRModel]
  field_simp [hε]
  ring

theorem finiteLineApproxP_tendsto_zero (r : Complex) :
    Tendsto (fun n => finiteLineApproxP r (escapeEps n)) atTop (nhds 0) := by
  simpa [finiteLineApproxP] using
    (escapeEps_tendsto_zero.pow 2).sub
      (tendsto_const_nhds.mul (escapeEps_tendsto_zero.pow 3))

theorem finiteLineApproxQModel_tendsto_zero
    (h : Complex[X]) (r : Complex) :
    Tendsto (fun n => finiteLineApproxQModel h r (escapeEps n))
      atTop (nhds 0) := by
  have hp := finiteLineApproxP_tendsto_zero r
  have hh : Tendsto
      (fun n => h.eval (finiteLineApproxP r (escapeEps n))) atTop
      (nhds (h.eval 0)) :=
    (h.continuous.tendsto 0).comp hp
  have hOne : Tendsto (fun n => 1 - r * escapeEps n) atTop (nhds 1) := by
    simpa using tendsto_const_nhds.sub
      (tendsto_const_nhds.mul escapeEps_tendsto_zero)
  have hLead : Tendsto
      (fun n => escapeEps n ^ 8 * (1 - r * escapeEps n) ^ 6 *
        h.eval (finiteLineApproxP r (escapeEps n))) atTop (nhds 0) := by
    simpa [mul_assoc] using
      (escapeEps_tendsto_zero.pow 8).mul ((hOne.pow 6).mul hh)
  have hFour : Tendsto (fun _ : Nat => (4 : Complex)) atTop (nhds 4) :=
    tendsto_const_nhds
  have hLinear : Tendsto (fun n => 4 * escapeEps n) atTop (nhds 0) := by
    simpa using hFour.mul escapeEps_tendsto_zero
  have hThree : Tendsto (fun _ : Nat => (3 : Complex)) atTop (nhds 3) :=
    tendsto_const_nhds
  have hR : Tendsto (fun _ : Nat => r) atTop (nhds r) :=
    tendsto_const_nhds
  have hQuadratic : Tendsto
      (fun n => 3 * r * escapeEps n ^ 2) atTop (nhds 0) := by
    simpa using
      (hThree.mul hR).mul (escapeEps_tendsto_zero.pow 2)
  simpa [finiteLineApproxQModel] using
    (hLead.add hLinear).sub hQuadratic

theorem finiteLineApproxRModel_tendsto
    (h : Complex[X]) (r : Complex) :
    Tendsto (fun n => finiteLineApproxRModel h r (escapeEps n))
      atTop (nhds r) := by
  have hp := finiteLineApproxP_tendsto_zero r
  have hh : Tendsto
      (fun n => h.eval (finiteLineApproxP r (escapeEps n))) atTop
      (nhds (h.eval 0)) :=
    (h.continuous.tendsto 0).comp hp
  have hOne : Tendsto (fun n => 1 - r * escapeEps n) atTop (nhds 1) := by
    simpa using tendsto_const_nhds.sub
      (tendsto_const_nhds.mul escapeEps_tendsto_zero)
  have hTwoThird : Tendsto (fun _ : Nat => (2 / 3 : Complex)) atTop
      (nhds (2 / 3 : Complex)) := tendsto_const_nhds
  have hError : Tendsto
      (fun n => (2 / 3 : Complex) * escapeEps n ^ 6 *
        (1 - r * escapeEps n) ^ 6 *
          h.eval (finiteLineApproxP r (escapeEps n))) atTop (nhds 0) := by
    simpa using
      ((hTwoThird.mul (escapeEps_tendsto_zero.pow 6)).mul
        (hOne.pow 6)).mul hh
  have hR : Tendsto (fun _ : Nat => r) atTop (nhds r) :=
    tendsto_const_nhds
  simpa [finiteLineApproxRModel] using
    hR.sub hError

/-- Critical targets approach every point of the exceptional line `p=q=0`. -/
theorem criticalTarget_finiteLineApprox_tendsto
    (h : Complex[X]) (r : Complex) :
    Tendsto
      (fun n => criticalTarget h (finiteLineApproxP r (escapeEps n))
        (escapeEps n)⁻¹)
      atTop (nhds (![0, 0, r] : C3)) := by
  apply tendsto_pi_nhds.2
  intro i
  fin_cases i
  · simpa [criticalTarget] using finiteLineApproxP_tendsto_zero r
  · apply Tendsto.congr'
      (Filter.Eventually.of_forall fun n => by
        simpa [criticalTarget] using
          (criticalQ_finiteLineApprox h r (escapeEps n)
            (escapeEps_ne_zero n)).symm)
      (finiteLineApproxQModel_tendsto_zero h r)
  · apply Tendsto.congr'
      (Filter.Eventually.of_forall fun n => by
        simpa [criticalTarget] using
          (criticalR_finiteLineApprox h r (escapeEps n)
            (escapeEps_ne_zero n)).symm)
      (finiteLineApproxRModel_tendsto h r)

/-- The exceptional line `p=q=0` lies in the finite component. -/
theorem finiteLine_subset_finiteComponent
    (h : Complex[X]) (r : Complex) :
    (![0, 0, r] : C3) ∈ finiteComponent h := by
  change toZariskiC3 (![0, 0, r] : C3) ∈
    closure (zariskiLift (criticalImage h))
  apply mem_closure_of_tendsto
    (continuous_toZariskiC3_euclidean.continuousAt.tendsto.comp
      (criticalTarget_finiteLineApprox_tendsto h r))
  exact Filter.Eventually.of_forall fun n => by
    change criticalTarget h (finiteLineApproxP r (escapeEps n))
      (escapeEps n)⁻¹ ∈ criticalImage h
    exact ⟨finiteLineApproxP r (escapeEps n), (escapeEps n)⁻¹,
      inv_ne_zero (escapeEps_ne_zero n), rfl⟩

/-- The universal eliminant cuts out exactly the finite component. -/
theorem finiteComponent_eq_deltaZeroLocus
    (h : Complex[X]) :
    finiteComponent h = deltaZeroLocus (deltaFamily h) := by
  apply Set.Subset.antisymm (finiteComponent_subset_deltaFamily h)
  intro b hb
  by_cases hp : b 0 = 0
  · have hbvec : b = (![0, b 1, b 2] : C3) := by
      funext i
      fin_cases i <;> simp [hp]
    have hSlice : (![0, b 1, b 2] : C3) ∈
        deltaZeroLocus (deltaFamily h) := by
      rwa [← hbvec]
    rcases (mem_deltaZeroLocus_deltaFamily_of_p_zero_iff h (b 1) (b 2)).1
      hSlice with hq | hqr
    · rw [hbvec, hq]
      exact finiteLine_subset_finiteComponent h (b 2)
    · have hr : b 2 ≠ 0 := by
        intro hr
        rw [hr, mul_zero] at hqr
        exact zero_ne_one hqr
      have hq : b 1 = 1 / b 2 := by
        apply (eq_div_iff hr).2
        simpa [mul_comm] using hqr
      rw [hbvec, hq]
      exact subset_closure
        (⟨0, b 2, hr, by simp [criticalTarget, criticalQ, criticalR]⟩ :
          (![0, 1 / b 2, b 2] : C3) ∈ criticalImage h)
  · exact subset_closure
      (deltaFamily_mem_criticalImage_of_p_ne_zero h hb hp)

/-- The family eliminant is nonzero. -/
theorem deltaFamily_ne_zero (h : Complex[X]) :
    deltaFamily h ≠ 0 := by
  intro hZero
  have hEval := congrArg
    (fun f : MvPolynomial (Fin 3) Complex =>
      MvPolynomial.aeval (![0, 1, 0] : C3) f) hZero
  rw [deltaFamily_aeval] at hEval
  have : (-243 : Complex) = 0 := by
    simp [deltaValue] at hEval
  norm_num at this

/-- A subset of affine three-space is a hypersurface when it is exactly the
zero locus of one nonzero polynomial. -/
def IsAffineHypersurface (S : Set C3) : Prop :=
  ∃ f : MvPolynomial (Fin 3) Complex,
    f ≠ 0 ∧ S = deltaZeroLocus f

/-- The finite component is an exact affine hypersurface. -/
theorem finiteComponent_isHypersurface
    (h : Complex[X]) :
    IsAffineHypersurface (finiteComponent h) :=
  ⟨deltaFamily h, deltaFamily_ne_zero h,
    finiteComponent_eq_deltaZeroLocus h⟩

/-! ## Dominance and distinctness -/

/-- A subset of target affine three-space dominates the first-coordinate line
when its first-coordinate projection is surjective. -/
def DominatesPLine (S : Set C3) : Prop :=
  Function.Surjective (fun b : S => b.1 0)

/-- The finite component dominates the first-coordinate line. -/
theorem finiteComponent_dominatesPLine
    (h : Complex[X]) :
    DominatesPLine (finiteComponent h) := by
  intro p
  let b : C3 := criticalTarget h p 1
  have hbImage : b ∈ criticalImage h :=
    ⟨p, 1, one_ne_zero, rfl⟩
  have hbFinite : b ∈ finiteComponent h :=
    subset_closure hbImage
  refine ⟨⟨b, hbFinite⟩, ?_⟩
  simp [b]

/-- The finite component is distinct from every vertical hyperplane. -/
theorem finiteComponent_ne_pHyperplane
    (h : Complex[X]) (alpha : Complex) :
    finiteComponent h ≠ pHyperplane alpha := by
  intro hEq
  exact finiteComponent_not_subset_pHyperplane h alpha hEq.le

end

end DegreeSixKeller
