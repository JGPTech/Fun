import DegreeSixKeller.PairSpecificDiscriminants
import Mathlib.Analysis.Polynomial.CauchyBound
import Mathlib.Topology.Algebra.Polynomial
import Mathlib.Tactic.FieldSimp

/-!
# Inverse chart and reconstruction for the degree-six Keller family

This module contains the exact algebraic identities used in the asymptotic
analysis.  No topology or component counting is used here.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial
open scoped Polynomial

noncomputable section

/-- The deformation term in the second target coordinate on the `A ≠ 0`
chart. -/
def phi (h : Complex[X]) (p s : Complex) : Complex :=
  3 * p * s + p ^ 6 * h.eval p * s ^ 4

/-- The deformation term in the third target coordinate on the `A ≠ 0`
chart. -/
def theta (h : Complex[X]) (p s : Complex) : Complex :=
  p * s ^ 3 + (2 / 3 : Complex) * p ^ 6 * h.eval p * s ^ 6

/-- The affine chart variable `s = x/A`. -/
def chartS (x y : Complex) : Complex :=
  x / baseA x y

/-- The reconstruction denominator `D = 1 - sy`. -/
def rootD (h : Complex[X]) (p q s : Complex) : Complex :=
  1 - s * (q - phi h p s)

/-- The source point reconstructed from a finite simple inverse root. -/
def reconstruct (h : Complex[X]) (p q s : Complex) : C3 :=
  let y := q - phi h p s
  let D := rootD h p q s
  ![s / D,
    y,
    p * D ^ 3 - y ^ 2 * (4 - s * y) * D]

@[simp]
theorem reconstruct_zero (h : Complex[X]) (p q s : Complex) :
    reconstruct h p q s 0 = s / rootD h p q s := by
  simp [reconstruct]

@[simp]
theorem reconstruct_one (h : Complex[X]) (p q s : Complex) :
    reconstruct h p q s 1 = q - phi h p s := by
  simp [reconstruct]

@[simp]
theorem reconstruct_two (h : Complex[X]) (p q s : Complex) :
    reconstruct h p q s 2 =
      p * rootD h p q s ^ 3 -
        (q - phi h p s) ^ 2 *
          (4 - s * (q - phi h p s)) * rootD h p q s := by
  simp [reconstruct]

/-- Multiplying the chart coordinate by the marked factor recovers `x`. -/
theorem chartS_mul_baseA
    (x y : Complex) (hA : baseA x y ≠ 0) :
    chartS x y * baseA x y = x := by
  unfold chartS
  field_simp [hA]

/-- The reconstruction denominator evaluated at a source chart point is
`A⁻¹`. -/
theorem one_sub_chartS_mul
    (x y : Complex) (hA : baseA x y ≠ 0) :
    1 - chartS x y * y = (baseA x y)⁻¹ := by
  unfold chartS
  field_simp [hA]
  unfold baseA
  ring

/-- The undeformed second coordinate becomes `y + 3ps` on the chart. -/
theorem baseQ_chart_identity
    (x y z : Complex) (hA : baseA x y ≠ 0) :
    baseQ x y z = y + 3 * pCoord x y z * chartS x y := by
  unfold baseQ pCoord chartS
  field_simp [hA]

/-- The full second coordinate becomes `y + phi(p,s)` on the chart. -/
theorem qCoord_chart_identity
    (h : Complex[X]) (x y z : Complex) (hA : baseA x y ≠ 0) :
    qCoord h x y z = y + phi h (pCoord x y z) (chartS x y) := by
  rw [qCoord, baseQ_chart_identity x y z hA]
  unfold phi pCoord chartS
  field_simp [hA]
  ring

/-- The undeformed third coordinate becomes `2s - ys^2 - ps^3`. -/
theorem baseR_chart_identity
    (x y z : Complex) (hA : baseA x y ≠ 0) :
    baseR x y z =
      2 * chartS x y - y * chartS x y ^ 2 -
        pCoord x y z * chartS x y ^ 3 := by
  unfold baseR chartS pCoord
  field_simp [hA]
  unfold baseB baseA
  ring

/-- The full third coordinate becomes `2s - ys^2 - theta(p,s)`. -/
theorem rCoord_chart_identity
    (h : Complex[X]) (x y z : Complex) (hA : baseA x y ≠ 0) :
    rCoord h x y z =
      2 * chartS x y - y * chartS x y ^ 2 -
        theta h (pCoord x y z) (chartS x y) := by
  rw [rCoord, baseR_chart_identity x y z hA]
  unfold theta pCoord chartS
  field_simp [hA]
  ring

/-- Every source point on the `A ≠ 0` chart produces a root of `omega`. -/
theorem omega_of_source
    (h : Complex[X]) (x y z : Complex) (hA : baseA x y ≠ 0) :
    (omega h
      (pCoord x y z)
      (qCoord h x y z)
      (rCoord h x y z)).eval (chartS x y) = 0 := by
  rw [omega_eval, qCoord_chart_identity h x y z hA,
    rCoord_chart_identity h x y z hA]
  unfold aCoeff phi theta
  ring

/-- The derivative at the source root is exactly `2/A`. -/
theorem omegaDerivative_of_source
    (h : Complex[X]) (x y z : Complex) (hA : baseA x y ≠ 0) :
    (derivative (omega h
      (pCoord x y z)
      (qCoord h x y z)
      (rCoord h x y z))).eval (chartS x y) =
      2 / baseA x y := by
  rw [omega_derivative_eval, qCoord_chart_identity h x y z hA]
  calc
    2 * pCoord x y z ^ 6 * h.eval (pCoord x y z) * chartS x y ^ 5 +
          6 * pCoord x y z * chartS x y ^ 2 -
          2 * (y + phi h (pCoord x y z) (chartS x y)) * chartS x y + 2 =
        2 * (1 - chartS x y * y) := by
          unfold phi
          ring
    _ = 2 * (baseA x y)⁻¹ := by
          rw [one_sub_chartS_mul x y hA]
    _ = 2 / baseA x y := by
          rfl

/-- The reconstruction denominator is half the evaluated derivative. -/
theorem rootD_eq_derivative_div_two
    (h : Complex[X]) (p q r s : Complex)
    (_hOmega : (omega h p q r).eval s = 0) :
    rootD h p q s = (derivative (omega h p q r)).eval s / 2 := by
  rw [omega_derivative_eval]
  unfold rootD phi
  ring

/-- Reconstruction has marked factor `A = D⁻¹`. -/
theorem baseA_reconstruct
    (h : Complex[X]) (p q s : Complex)
    (hD : rootD h p q s ≠ 0) :
    baseA (reconstruct h p q s 0) (reconstruct h p q s 1) =
      (rootD h p q s)⁻¹ := by
  simp only [baseA, reconstruct_zero, reconstruct_one]
  field_simp [hD]
  unfold rootD
  ring

/-- Reconstruction has marked factor `B = pD`. -/
theorem baseB_reconstruct
    (h : Complex[X]) (p q s : Complex)
    (hD : rootD h p q s ≠ 0) :
    baseB
      (reconstruct h p q s 0)
      (reconstruct h p q s 1)
      (reconstruct h p q s 2) = p * rootD h p q s := by
  rw [baseB]
  rw [baseA_reconstruct h p q s hD]
  simp only [reconstruct_zero, reconstruct_one, reconstruct_two]
  field_simp [hD]
  unfold rootD
  ring

/-- Reconstruction recovers the first target coordinate. -/
theorem pCoord_reconstruct
    (h : Complex[X]) (p q s : Complex)
    (hD : rootD h p q s ≠ 0) :
    pCoord
      (reconstruct h p q s 0)
      (reconstruct h p q s 1)
      (reconstruct h p q s 2) = p := by
  unfold pCoord
  rw [baseA_reconstruct h p q s hD,
    baseB_reconstruct h p q s hD]
  field_simp [hD]

/-- Reconstruction recovers its chart coordinate. -/
theorem chartS_reconstruct
    (h : Complex[X]) (p q s : Complex)
    (hD : rootD h p q s ≠ 0) :
    chartS
      (reconstruct h p q s 0)
      (reconstruct h p q s 1) = s := by
  rw [chartS, baseA_reconstruct h p q s hD, reconstruct_zero]
  field_simp [hD]

/-- The reconstructed marked factor is nonzero. -/
theorem baseA_reconstruct_ne_zero
    (h : Complex[X]) (p q s : Complex)
    (hD : rootD h p q s ≠ 0) :
    baseA (reconstruct h p q s 0) (reconstruct h p q s 1) ≠ 0 := by
  rw [baseA_reconstruct h p q s hD]
  exact inv_ne_zero hD

/-- Reconstruction recovers the second target coordinate. -/
theorem qCoord_reconstruct
    (h : Complex[X]) (p q s : Complex)
    (hD : rootD h p q s ≠ 0) :
    qCoord h
      (reconstruct h p q s 0)
      (reconstruct h p q s 1)
      (reconstruct h p q s 2) = q := by
  rw [qCoord_chart_identity h _ _ _
    (baseA_reconstruct_ne_zero h p q s hD)]
  rw [pCoord_reconstruct h p q s hD, chartS_reconstruct h p q s hD,
    reconstruct_one]
  ring

/-- Reconstruction recovers the third target coordinate whenever `s` is a root. -/
theorem rCoord_reconstruct
    (h : Complex[X]) (p q r s : Complex)
    (hOmega : (omega h p q r).eval s = 0)
    (hD : rootD h p q s ≠ 0) :
    rCoord h
      (reconstruct h p q s 0)
      (reconstruct h p q s 1)
      (reconstruct h p q s 2) = r := by
  rw [rCoord_chart_identity h _ _ _
    (baseA_reconstruct_ne_zero h p q s hD)]
  rw [pCoord_reconstruct h p q s hD, chartS_reconstruct h p q s hD,
    reconstruct_one]
  rw [omega_eval] at hOmega
  unfold aCoeff phi theta at *
  linear_combination hOmega

/-- A finite simple root reconstructs an exact source point. -/
theorem reconstruct_maps_to
    (h : Complex[X]) (p q r s : Complex)
    (hOmega : (omega h p q r).eval s = 0)
    (hSimple : (derivative (omega h p q r)).eval s ≠ 0) :
    Fh h (reconstruct h p q s) = ![p, q, r] := by
  have hD : rootD h p q s ≠ 0 := by
    rw [rootD_eq_derivative_div_two h p q r s hOmega]
    exact div_ne_zero hSimple (by norm_num)
  funext i
  fin_cases i
  · simpa [Fh] using pCoord_reconstruct h p q s hD
  · simpa [Fh] using qCoord_reconstruct h p q s hD
  · simpa [Fh] using rCoord_reconstruct h p q r s hOmega hD

/-- At a source chart point the reconstruction denominator is `A⁻¹`. -/
theorem rootD_of_source
    (h : Complex[X]) (x y z : Complex) (hA : baseA x y ≠ 0) :
    rootD h (pCoord x y z) (qCoord h x y z) (chartS x y) =
      (baseA x y)⁻¹ := by
  rw [rootD, qCoord_chart_identity h x y z hA]
  simpa using one_sub_chartS_mul x y hA

/-- A source point in the finite chart equals the reconstruction from its
chart root. -/
theorem source_eq_reconstruct_of_chart
    (h : Complex[X]) (x y z : Complex) (hA : baseA x y ≠ 0) :
    (![x, y, z] : C3) = reconstruct h
      (pCoord x y z) (qCoord h x y z) (chartS x y) := by
  have h0 :
      x = reconstruct h
        (pCoord x y z) (qCoord h x y z) (chartS x y) 0 := by
    rw [reconstruct_zero, rootD_of_source h x y z hA]
    unfold chartS
    field_simp [hA]
  have h1 :
      y = reconstruct h
        (pCoord x y z) (qCoord h x y z) (chartS x y) 1 := by
    rw [reconstruct_one, qCoord_chart_identity h x y z hA]
    ring
  have h2 :
      z = reconstruct h
        (pCoord x y z) (qCoord h x y z) (chartS x y) 2 := by
    rw [reconstruct_two, rootD_of_source h x y z hA,
      qCoord_chart_identity h x y z hA]
    simp only [add_sub_cancel_right]
    unfold chartS pCoord
    field_simp [hA]
    unfold baseB baseA
    ring
  funext i
  fin_cases i
  · simpa using h0
  · simpa using h1
  · simpa using h2

end

end DegreeSixKeller
