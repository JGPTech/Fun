import DegreeSixKeller.AsymptoticSources
import Mathlib.FieldTheory.IsAlgClosed.Basic
import Mathlib.RingTheory.Polynomial.Resultant.Basic
import Mathlib.Tactic

/-!
# Resultant certificates for the pair-specific finite components

The original eleven-by-eleven Sylvester determinant is deliberately avoided.
A pair of exact Euclidean-reduction identities replaces `omega` and its
derivative by a cubic and a quadratic.  Their five-by-five Sylvester
determinant is then evaluated explicitly.  Off the leading locus, a common
root of the cubic and quadratic reconstructs a common root of `omega` and its
derivative.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 8000000

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

/-- A nonzero polynomial over an algebraically closed field whose ordinary
resultant with another polynomial vanishes has a common root with it. -/
theorem exists_common_root_of_resultant_eq_zero
    (f g : Complex[X]) (_hf : f ≠ 0)
    (hres : resultant f g = 0) :
    ∃ s : Complex, f.eval s = 0 ∧ g.eval s = 0 := by
  have hNotCoprime : ¬ IsCoprime f g :=
    (resultant_eq_zero_iff.mp hres).2
  rw [Polynomial.isCoprime_iff_aeval_ne_zero_of_isAlgClosed
      (k := Complex) (K := Complex) f g] at hNotCoprime
  push Not at hNotCoprime
  rcases hNotCoprime with ⟨s, hs1, hs2⟩
  exact ⟨s, by simpa [Polynomial.aeval_def] using hs1,
    by simpa [Polynomial.aeval_def] using hs2⟩

/-! ## The reduced cubic/quadratic certificate -/

/-- A cubic written in ascending coefficient data. -/
def cubicPolynomial (a0 a1 a2 a3 : Complex) : Complex[X] :=
  C a3 * X ^ 3 + C a2 * X ^ 2 + C a1 * X + C a0

/-- A quadratic written in ascending coefficient data. -/
def quadraticPolynomial (b0 b1 b2 : Complex) : Complex[X] :=
  C b2 * X ^ 2 + C b1 * X + C b0

/-- The explicit resultant of a cubic and a quadratic.  This is only a
five-by-five determinant. -/
theorem resultant_cubic_quadratic
    (a0 a1 a2 a3 b0 b1 b2 : Complex) :
    resultant (cubicPolynomial a0 a1 a2 a3)
        (quadraticPolynomial b0 b1 b2) 3 2 =
      a0 ^ 2 * b2 ^ 3
      - a0 * a1 * b1 * b2 ^ 2
      - 2 * a0 * a2 * b0 * b2 ^ 2
      + a0 * a2 * b1 ^ 2 * b2
      + 3 * a0 * a3 * b0 * b1 * b2
      - a0 * a3 * b1 ^ 3
      + a1 ^ 2 * b0 * b2 ^ 2
      - a1 * a2 * b0 * b1 * b2
      - 2 * a1 * a3 * b0 ^ 2 * b2
      + a1 * a3 * b0 * b1 ^ 2
      + a2 ^ 2 * b0 ^ 2 * b2
      - a2 * a3 * b0 ^ 2 * b1
      + a3 ^ 2 * b0 ^ 3 := by
  let f := cubicPolynomial a0 a1 a2 a3
  let g := quadraticPolynomial b0 b1 b2
  have hsyl : Polynomial.sylvester f g 3 2 =
      !![b0, 0,  0,  a0, 0;
         b1, b0, 0,  a1, a0;
         b2, b1, b0, a2, a1;
         0,  b2, b1, a3, a2;
         0,  0,  b2, 0,  a3] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Polynomial.sylvester, f, g, cubicPolynomial,
        quadraticPolynomial, Set.mem_Icc, Fin.addCases,
        Polynomial.coeff_X]
  rw [Polynomial.resultant, hsyl]
  simp [Matrix.det_succ_row_zero, Fin.sum_univ_succ,
    Matrix.submatrix_apply, Function.comp_apply, Fin.succAbove]
  ring

/-- The cubic remainder `H₃` obtained from
`6 * omega - X * omega' = 2 * H₃`. -/
def cubicRemainder (p q r : Complex) : Complex[X] :=
  cubicPolynomial (-3 * r) 5 (-2 * q) (3 * p)

/-- Coefficient of `X²` in the quadratic remainder. -/
def quadraticCoeffTwo (a p q r : Complex) : Complex :=
  27 * a * p ^ 2 * r - 60 * a * p * q + 8 * a * q ^ 3 + 27 * p ^ 4

/-- Coefficient of `X` in the quadratic remainder. -/
def quadraticCoeffOne (a p q r : Complex) : Complex :=
  18 * a * p * q * r + 75 * a * p - 20 * a * q ^ 2 - 9 * p ^ 3 * q

/-- Constant coefficient of the quadratic remainder. -/
def quadraticCoeffZero (a p q r : Complex) : Complex :=
  -45 * a * p * r + 12 * a * q ^ 2 * r + 9 * p ^ 3

/-- The quadratic remainder `H₂`. -/
def quadraticRemainder (a p q r : Complex) : Complex[X] :=
  quadraticPolynomial
    (quadraticCoeffZero a p q r)
    (quadraticCoeffOne a p q r)
    (quadraticCoeffTwo a p q r)

/-- The scalar core of the reduced resultant. -/
def discriminantCore (a p q r : Complex) : Complex :=
    729 * a ^ 3 * r ^ 5
  + 2187 * a ^ 2 * p ^ 2 * r ^ 4
  - 4860 * a ^ 2 * p * q * r ^ 3
  + 6750 * a ^ 2 * p * r ^ 2
  - 216 * a ^ 2 * q ^ 3 * r ^ 3
  + 2700 * a ^ 2 * q ^ 2 * r ^ 2
  - 5625 * a ^ 2 * q * r
  + 3125 * a ^ 2
  + 2187 * a * p ^ 4 * r ^ 3
  - 5346 * a * p ^ 3 * q * r ^ 2
  + 1350 * a * p ^ 3 * r
  + 540 * a * p ^ 2 * q ^ 3 * r ^ 2
  + 2430 * a * p ^ 2 * q ^ 2 * r
  - 2250 * a * p ^ 2 * q
  - 432 * a * p * q ^ 4 * r
  + 400 * a * p * q ^ 3
  + 16 * a * q ^ 6 * r
  - 16 * a * q ^ 5
  + 729 * p ^ 6 * r ^ 2
  - 486 * p ^ 5 * q * r
  + 432 * p ^ 5
  + 27 * p ^ 4 * q ^ 3 * r
  - 27 * p ^ 4 * q ^ 2

/-- Exact five-by-five resultant certificate. -/
theorem reducedResultant_factorization (a p q r : Complex) :
    resultant (cubicRemainder p q r)
        (quadraticRemainder a p q r) 3 2 =
      243 * p ^ 6 * discriminantCore a p q r := by
  rw [show cubicRemainder p q r =
      cubicPolynomial (-3 * r) 5 (-2 * q) (3 * p) from rfl]
  rw [show quadraticRemainder a p q r =
      quadraticPolynomial
        (quadraticCoeffZero a p q r)
        (quadraticCoeffOne a p q r)
        (quadraticCoeffTwo a p q r) from rfl]
  rw [resultant_cubic_quadratic]
  simp only [quadraticCoeffZero, quadraticCoeffOne, quadraticCoeffTwo,
    discriminantCore]
  ring

/-- Substituting `a = p^6 h(p)/3` turns the reduced core into the exact
nonvertical factor from Upgrade 2. -/
theorem discriminantCore_aCoeff
    (h : Complex[X]) (p q r : Complex) :
    discriminantCore (aCoeff h p) p q r =
      p ^ 4 / 9 * deltaValue (h.eval p) p q r := by
  simp only [discriminantCore, aCoeff, deltaValue]
  ring

/-- Pair-specific reduced resultant certificate for CEX-004. -/
theorem reducedResultant004_factorization (p q r : Complex) :
    resultant (cubicRemainder p q r)
        (quadraticRemainder (aCoeff eta004 p) p q r) 3 2 =
      27 * p ^ 10 * MvPolynomial.aeval (![p, q, r] : C3) delta004 := by
  rw [reducedResultant_factorization, discriminantCore_aCoeff,
    delta004, deltaFamily_aeval]
  simp
  ring

/-- Pair-specific reduced resultant certificate for CEX-006, using the
integral normalization of `delta006`. -/
theorem reducedResultant006_factorization (p q r : Complex) :
    resultant (cubicRemainder p q r)
        (quadraticRemainder (aCoeff eta006 p) p q r) 3 2 =
      (-243 / 8 : Complex) * p ^ 10 *
        MvPolynomial.aeval (![p, q, r] : C3) delta006 := by
  rw [reducedResultant_factorization, discriminantCore_aCoeff,
    delta006, map_mul, deltaFamily_aeval]
  simp
  ring

/-- The cubic remainder has exact degree three off `p = 0`. -/
theorem cubicRemainder_natDegree (p q r : Complex) (hp : p ≠ 0) :
    (cubicRemainder p q r).natDegree = 3 := by
  simp only [cubicRemainder, cubicPolynomial]
  compute_degree!

/-- The quadratic remainder has degree at most two. -/
theorem quadraticRemainder_natDegree_le (a p q r : Complex) :
    (quadraticRemainder a p q r).natDegree ≤ 2 := by
  simp only [quadraticRemainder, quadraticPolynomial]
  compute_degree

/-- Vanishing of the fixed `3 × 2` resultant yields an ordinary common root,
provided the cubic leading coefficient is nonzero. -/
theorem exists_common_root_of_reduced_resultant_eq_zero
    (a p q r : Complex) (hp : p ≠ 0)
    (hres : resultant (cubicRemainder p q r)
      (quadraticRemainder a p q r) 3 2 = 0) :
    ∃ s : Complex,
      (cubicRemainder p q r).eval s = 0 ∧
      (quadraticRemainder a p q r).eval s = 0 := by
  let f := cubicRemainder p q r
  let g := quadraticRemainder a p q r
  have hfdeg : f.natDegree = 3 := cubicRemainder_natDegree p q r hp
  have hgdeg : g.natDegree ≤ 2 := quadraticRemainder_natDegree_le a p q r
  have hfcoeffValue : f.coeff 3 = 3 * p := by
    simp only [f, cubicRemainder, cubicPolynomial, Polynomial.coeff_add,
      Polynomial.coeff_C_mul_X_pow, Polynomial.coeff_C_mul_X,
      Polynomial.coeff_C]
    norm_num
  have hfcoeff : f.coeff 3 ≠ 0 := by
    rw [hfcoeffValue]
    exact mul_ne_zero (by norm_num) hp
  change resultant f g 3 2 = 0 at hres
  have hraise := Polynomial.resultant_add_right_deg
    f g 3 g.natDegree (2 - g.natDegree) le_rfl
  rw [Nat.add_sub_of_le hgdeg] at hraise
  rw [hres] at hraise
  have hpow : f.coeff 3 ^ (2 - g.natDegree) ≠ 0 :=
    pow_ne_zero _ hfcoeff
  have hdefaultFixed : resultant f g 3 g.natDegree = 0 :=
    (mul_eq_zero.mp hraise.symm).resolve_left hpow
  have hdefault : resultant f g = 0 := by
    simpa [hfdeg] using hdefaultFixed
  have hf : f ≠ 0 := by
    intro hz
    have := congrArg (fun u : Complex[X] => u.coeff 3) hz
    exact hfcoeff (by simpa using this)
  simpa [f, g] using exists_common_root_of_resultant_eq_zero f g hf hdefault

/-! ## Recovering the original multiple root -/

/-- First Euclidean identity: `6 Ω - X Ω' = 2 H₃`. -/
theorem omega_cubicRemainder_identity
    (h : Complex[X]) (p q r s : Complex) :
    6 * (omega h p q r).eval s
      - s * (derivative (omega h p q r)).eval s =
        2 * (cubicRemainder p q r).eval s := by
  rw [omega_eval, omega_derivative_eval]
  simp [cubicRemainder, cubicPolynomial, aCoeff]
  ring

/-- Second Euclidean identity. -/
theorem derivative_quadraticRemainder_identity
    (h : Complex[X]) (p q r s : Complex) :
    9 * p ^ 3 * (derivative (omega h p q r)).eval s
      - 2 * aCoeff h p *
          (9 * p ^ 2 * s ^ 2 + 6 * p * q * s - 15 * p + 4 * q ^ 2) *
          (cubicRemainder p q r).eval s =
        2 * (quadraticRemainder (aCoeff h p) p q r).eval s := by
  rw [omega_derivative_eval]
  simp [cubicRemainder, cubicPolynomial, quadraticRemainder,
    quadraticPolynomial, quadraticCoeffZero, quadraticCoeffOne,
    quadraticCoeffTwo, aCoeff]
  ring

/-- A common root of the reduced cubic/quadratic pair is a common root of
`omega` and its derivative, off `p = 0`. -/
theorem common_root_of_remainders
    (h : Complex[X]) (p q r s : Complex) (hp : p ≠ 0)
    (hCubic : (cubicRemainder p q r).eval s = 0)
    (hQuadratic :
      (quadraticRemainder (aCoeff h p) p q r).eval s = 0) :
    (omega h p q r).eval s = 0 ∧
      (derivative (omega h p q r)).eval s = 0 := by
  have hDerivative : (derivative (omega h p q r)).eval s = 0 := by
    have hrel := derivative_quadraticRemainder_identity h p q r s
    rw [hCubic, hQuadratic] at hrel
    simp only [mul_zero, sub_zero] at hrel
    have hcoef : 9 * p ^ 3 ≠ 0 :=
      mul_ne_zero (by norm_num) (pow_ne_zero 3 hp)
    exact (mul_eq_zero.mp hrel).resolve_left hcoef
  have hOmega : (omega h p q r).eval s = 0 := by
    have hrel := omega_cubicRemainder_identity h p q r s
    rw [hDerivative, hCubic] at hrel
    linear_combination hrel / 6
  exact ⟨hOmega, hDerivative⟩

/-- A common nonzero root of `omega` and its derivative is exactly a point of
the finite critical parametrization. -/
theorem eq_criticalTarget_of_common_root
    (h : Complex[X]) (p q r s : Complex)
    (hs : s ≠ 0)
    (hOmega : (omega h p q r).eval s = 0)
    (hDerivative : (derivative (omega h p q r)).eval s = 0) :
    (![p, q, r] : C3) = criticalTarget h p s := by
  have hq : q = criticalQ h p s := by
    rw [omega_derivative_eval] at hDerivative
    unfold criticalQ
    field_simp [hs]
    linear_combination -hDerivative / 2
  have hr : r = criticalR h p s := by
    rw [omega_eval, hq] at hOmega
    unfold criticalQ at hOmega
    unfold criticalR
    unfold aCoeff at hOmega
    field_simp [hs] at hOmega ⊢
    ring_nf at hOmega ⊢
    exact (sub_eq_zero.mp hOmega).symm
  funext i
  fin_cases i <;> simp [criticalTarget, hq, hr]

/-- The common root supplied by the reduced resultant is nonzero because the
constant term of the derivative is `2`. -/
theorem common_root_ne_zero
    (h : Complex[X]) (p q r s : Complex)
    (hDerivative : (derivative (omega h p q r)).eval s = 0) :
    s ≠ 0 := by
  intro hs
  subst s
  rw [omega_derivative_eval] at hDerivative
  norm_num at hDerivative

/-! ## Pair-specific finite-component inclusions -/

/-- Off its leading locus, every CEX-004 point of the finite component is
already in the directly parametrized critical image. -/
theorem finiteComponent004_mem_criticalImage_of_aCoeff_ne_zero
    {b : C3} (hb : b ∈ finiteComponent eta004)
    (ha : aCoeff eta004 (b 0) ≠ 0) :
    b ∈ criticalImage eta004 := by
  have hDelta := finiteComponent004_subset_delta hb
  have hDeltaZero : MvPolynomial.aeval b delta004 = 0 :=
    (mem_deltaZeroLocus_iff delta004 b).1 hDelta
  have hp : b 0 ≠ 0 := by
    intro hp
    apply ha
    simp [hp, aCoeff]
  have hbvec : (![b 0, b 1, b 2] : C3) = b := by
    funext i
    fin_cases i <;> simp
  have hDeltaVec :
      MvPolynomial.aeval (![b 0, b 1, b 2] : C3) delta004 = 0 := by
    rw [hbvec]
    exact hDeltaZero
  have hRes : resultant (cubicRemainder (b 0) (b 1) (b 2))
      (quadraticRemainder (aCoeff eta004 (b 0)) (b 0) (b 1) (b 2)) 3 2 = 0 := by
    rw [reducedResultant004_factorization]
    simp [hDeltaVec]
  obtain ⟨s, hCubic, hQuadratic⟩ :=
    exists_common_root_of_reduced_resultant_eq_zero
      (aCoeff eta004 (b 0)) (b 0) (b 1) (b 2) hp hRes
  obtain ⟨hOmega, hDerivative⟩ :=
    common_root_of_remainders eta004 (b 0) (b 1) (b 2) s hp hCubic hQuadratic
  have hs := common_root_ne_zero eta004 (b 0) (b 1) (b 2) s hDerivative
  refine ⟨b 0, s, hs, ?_⟩
  have hEq := eq_criticalTarget_of_common_root eta004
    (b 0) (b 1) (b 2) s hs hOmega hDerivative
  calc
    b = (![b 0, b 1, b 2] : C3) := hbvec.symm
    _ = criticalTarget eta004 (b 0) s := hEq

/-- Off its leading locus, every CEX-006 point of the finite component is
already in the directly parametrized critical image. -/
theorem finiteComponent006_mem_criticalImage_of_aCoeff_ne_zero
    {b : C3} (hb : b ∈ finiteComponent eta006)
    (ha : aCoeff eta006 (b 0) ≠ 0) :
    b ∈ criticalImage eta006 := by
  have hDelta := finiteComponent006_subset_delta hb
  have hDeltaZero : MvPolynomial.aeval b delta006 = 0 :=
    (mem_deltaZeroLocus_iff delta006 b).1 hDelta
  have hp : b 0 ≠ 0 := by
    intro hp
    apply ha
    simp [hp, aCoeff]
  have hbvec : (![b 0, b 1, b 2] : C3) = b := by
    funext i
    fin_cases i <;> simp
  have hDeltaVec :
      MvPolynomial.aeval (![b 0, b 1, b 2] : C3) delta006 = 0 := by
    rw [hbvec]
    exact hDeltaZero
  have hRes : resultant (cubicRemainder (b 0) (b 1) (b 2))
      (quadraticRemainder (aCoeff eta006 (b 0)) (b 0) (b 1) (b 2)) 3 2 = 0 := by
    rw [reducedResultant006_factorization]
    simp [hDeltaVec]
  obtain ⟨s, hCubic, hQuadratic⟩ :=
    exists_common_root_of_reduced_resultant_eq_zero
      (aCoeff eta006 (b 0)) (b 0) (b 1) (b 2) hp hRes
  obtain ⟨hOmega, hDerivative⟩ :=
    common_root_of_remainders eta006 (b 0) (b 1) (b 2) s hp hCubic hQuadratic
  have hs := common_root_ne_zero eta006 (b 0) (b 1) (b 2) s hDerivative
  refine ⟨b 0, s, hs, ?_⟩
  have hEq := eq_criticalTarget_of_common_root eta006
    (b 0) (b 1) (b 2) s hs hOmega hDerivative
  calc
    b = (![b 0, b 1, b 2] : C3) := hbvec.symm
    _ = criticalTarget eta006 (b 0) s := hEq

/-- The full CEX-004 finite component lies in the actual nonproperness set. -/
theorem finiteComponent004_subset_nonproperness :
    finiteComponent eta004 ⊆ NonpropernessSet F004 := by
  intro b hb
  by_cases ha : aCoeff eta004 (b 0) = 0
  · rw [aCoeff_eta004_zero_iff] at ha
    rcases ha with hp | hp
    · exact pHyperplane_zero_subset_nonproperness004 hp
    · have hroot : b ∈ pHyperplane alpha004 := by
        simpa [pHyperplane, alpha004] using hp
      exact pHyperplane_root_subset_nonproperness004 hroot
  · exact criticalImage_subset_nonproperness eta004
      (finiteComponent004_mem_criticalImage_of_aCoeff_ne_zero hb ha)

/-- The full CEX-006 finite component lies in the actual nonproperness set. -/
theorem finiteComponent006_subset_nonproperness :
    finiteComponent eta006 ⊆ NonpropernessSet F006 := by
  intro b hb
  by_cases ha : aCoeff eta006 (b 0) = 0
  · rw [aCoeff_eta006_zero_iff] at ha
    exact pHyperplane_zero_subset_nonproperness006 ha
  · exact criticalImage_subset_nonproperness eta006
      (finiteComponent006_mem_criticalImage_of_aCoeff_ne_zero hb ha)

end

end DegreeSixKeller
