import DegreeSixKeller.PolynomialModel
import DegreeSixKeller.EliminationCertificates
import Mathlib.Algebra.MvPolynomial.Funext
import Mathlib.Algebra.MvPolynomial.PDeriv
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Tactic

/-!
# Formal Jacobian certificate for the degree-six Keller family

The determinant is computed in the multivariate polynomial ring, for an
arbitrary deformation polynomial `h`.  The pointwise maps used by the geometry
are connected to this model by `eval_FhPolynomial`.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open MvPolynomial Polynomial

noncomputable section

/-- The formal Jacobian matrix of a polynomial self-map of affine three-space.
Rows are target coordinates and columns are source variables. -/
def formalJacobian (P : PolynomialMap3) :
    Matrix (Fin 3) (Fin 3) (MvPolynomial (Fin 3) Complex) :=
  fun i j => MvPolynomial.pderiv j (P i)

/-- Backwards-readable alias for `formalJacobian`. -/
abbrev jacobianMatrix := formalJacobian

/-- The determinant of the formal Jacobian matrix. -/
def jacobianDet (P : PolynomialMap3) : MvPolynomial (Fin 3) Complex :=
  (formalJacobian P).det

/-- A polynomial map is Keller when its Jacobian determinant is a nonzero
constant. -/
def IsKeller (P : PolynomialMap3) : Prop :=
  ∃ c : Complex, c ≠ 0 ∧ jacobianDet P = MvPolynomial.C c

@[simp] private theorem C_natCast (n : Nat) :
    MvPolynomial.C (n : Complex) =
      (n : MvPolynomial (Fin 3) Complex) := by
  exact map_natCast MvPolynomial.C n

@[simp] private theorem C_ofNat (n : Nat) [n.AtLeastTwo] :
    MvPolynomial.C (ofNat(n) : Complex) =
      (ofNat(n) : MvPolynomial (Fin 3) Complex) := by
  exact _root_.map_ofNat MvPolynomial.C n

@[simp] private theorem pderiv_ofNat
    (i : Fin 3) (n : Nat) [n.AtLeastTwo] :
    MvPolynomial.pderiv i
        (ofNat(n) : MvPolynomial (Fin 3) Complex) = 0 := by
  rw [← C_ofNat]
  exact MvPolynomial.pderiv_C

/-- Chain rule for a univariate polynomial substituted into an
`MvPolynomial`, differentiated in one source variable. -/
theorem pderiv_polynomial_eval₂
    (h : Complex[X]) (P : MvPolynomial (Fin 3) Complex) (i : Fin 3) :
    MvPolynomial.pderiv i (h.eval₂ MvPolynomial.C P) =
      h.derivative.eval₂ MvPolynomial.C P * MvPolynomial.pderiv i P := by
  induction h using Polynomial.induction_on' with
  | add p q hp hq =>
      simp only [Polynomial.eval₂_add, map_add, hp, hq, add_mul]
  | monomial n a =>
      simp only [Polynomial.eval₂_monomial, MvPolynomial.pderiv_mul,
        MvPolynomial.pderiv_C, zero_mul, zero_add, MvPolynomial.pderiv_pow,
        Polynomial.derivative_monomial, Polynomial.eval₂_monomial]
      simp [map_mul, map_natCast, mul_comm, mul_left_comm, mul_assoc]

@[simp] theorem pderiv_deformationPolynomial
    (h : Complex[X]) (i : Fin 3) :
    MvPolynomial.pderiv i (deformationPolynomial h) =
      h.derivative.eval₂ MvPolynomial.C pCoordPolynomial *
        MvPolynomial.pderiv i pCoordPolynomial := by
  exact pderiv_polynomial_eval₂ h pCoordPolynomial i

private def sourceXDerivative : Fin 3 -> MvPolynomial (Fin 3) Complex
  | 0 => 1
  | 1 => 0
  | 2 => 0

private def baseADerivative : Fin 3 -> MvPolynomial (Fin 3) Complex
  | 0 => sourceY
  | 1 => sourceX
  | 2 => 0

private def baseBDerivative : Fin 3 -> MvPolynomial (Fin 3) Complex
  | 0 =>
      2 * baseAPolynomial * sourceY * sourceZ + 3 * sourceY ^ 3
  | 1 =>
      2 * baseAPolynomial * sourceX * sourceZ + 8 * sourceY +
        9 * sourceX * sourceY ^ 2
  | 2 => baseAPolynomial ^ 2

private def pCoordDerivative (i : Fin 3) :
    MvPolynomial (Fin 3) Complex :=
  baseADerivative i * baseBPolynomial +
    baseAPolynomial * baseBDerivative i

private def baseQDerivative : Fin 3 -> MvPolynomial (Fin 3) Complex
  | 0 =>
      3 * baseBPolynomial + 3 * sourceX * baseBDerivative 0
  | 1 => 1 + 3 * sourceX * baseBDerivative 1
  | 2 => 3 * sourceX * baseBDerivative 2

private def baseRDerivative : Fin 3 -> MvPolynomial (Fin 3) Complex
  | 0 =>
      2 - 6 * sourceX * sourceY - 3 * sourceX ^ 2 * sourceZ
  | 1 => -(3 * sourceX ^ 2)
  | 2 => -(sourceX ^ 3)

private def deformationDerivative (h : Complex[X]) (i : Fin 3) :
    MvPolynomial (Fin 3) Complex :=
  deformationPolynomial h.derivative * pCoordDerivative i

private def qCoordDerivative (h : Complex[X]) (i : Fin 3) :
    MvPolynomial (Fin 3) Complex :=
  baseQDerivative i +
    (2 * baseAPolynomial * baseADerivative i) *
      sourceX ^ 4 * baseBPolynomial ^ 6 * deformationPolynomial h +
    baseAPolynomial ^ 2 *
      (4 * sourceX ^ 3 * sourceXDerivative i) *
      baseBPolynomial ^ 6 * deformationPolynomial h +
    baseAPolynomial ^ 2 * sourceX ^ 4 *
      (6 * baseBPolynomial ^ 5 * baseBDerivative i) *
      deformationPolynomial h +
    baseAPolynomial ^ 2 * sourceX ^ 4 * baseBPolynomial ^ 6 *
      deformationDerivative h i

private def rCoordDerivative (h : Complex[X]) (i : Fin 3) :
    MvPolynomial (Fin 3) Complex :=
  baseRDerivative i - MvPolynomial.C (2 / 3 : Complex) *
    ((6 * sourceX ^ 5 * sourceXDerivative i) *
        baseBPolynomial ^ 6 * deformationPolynomial h +
      sourceX ^ 6 *
        (6 * baseBPolynomial ^ 5 * baseBDerivative i) *
        deformationPolynomial h +
      sourceX ^ 6 * baseBPolynomial ^ 6 * deformationDerivative h i)

@[simp] theorem pderiv_baseAPolynomial (i : Fin 3) :
    MvPolynomial.pderiv i baseAPolynomial = baseADerivative i := by
  fin_cases i <;>
    simp [baseAPolynomial, baseADerivative, sourceX, sourceY]

@[simp] theorem pderiv_sourceX (i : Fin 3) :
    MvPolynomial.pderiv i sourceX = sourceXDerivative i := by
  fin_cases i <;> simp [sourceX, sourceXDerivative]

@[simp] theorem pderiv_baseBPolynomial (i : Fin 3) :
    MvPolynomial.pderiv i baseBPolynomial = baseBDerivative i := by
  fin_cases i <;>
    simp [baseBPolynomial, baseBDerivative, baseADerivative,
      sourceX, sourceY, sourceZ]
  all_goals ring

@[simp] theorem pderiv_pCoordPolynomial (i : Fin 3) :
    MvPolynomial.pderiv i pCoordPolynomial = pCoordDerivative i := by
  simp [pCoordPolynomial, pCoordDerivative]
  ring

@[simp] theorem pderiv_baseQPolynomial (i : Fin 3) :
    MvPolynomial.pderiv i baseQPolynomial = baseQDerivative i := by
  fin_cases i <;>
    simp [baseQPolynomial, baseQDerivative, sourceX, sourceY]
  all_goals ring

@[simp] theorem pderiv_baseRPolynomial (i : Fin 3) :
    MvPolynomial.pderiv i baseRPolynomial = baseRDerivative i := by
  fin_cases i <;>
    simp [baseRPolynomial, baseRDerivative, sourceX, sourceY, sourceZ]
  all_goals ring

@[simp] theorem pderiv_deformationPolynomial_eq
    (h : Complex[X]) (i : Fin 3) :
    MvPolynomial.pderiv i (deformationPolynomial h) =
      deformationDerivative h i := by
  rw [pderiv_deformationPolynomial, pderiv_pCoordPolynomial]
  rfl

@[simp] theorem pderiv_qCoordPolynomial
    (h : Complex[X]) (i : Fin 3) :
    MvPolynomial.pderiv i (qCoordPolynomial h) = qCoordDerivative h i := by
  simp only [qCoordPolynomial, map_add, MvPolynomial.pderiv_mul,
    MvPolynomial.pderiv_pow, pderiv_baseAPolynomial,
    pderiv_baseBPolynomial, pderiv_baseQPolynomial,
    pderiv_deformationPolynomial_eq, pderiv_sourceX]
  unfold qCoordDerivative
  ring

@[simp] theorem pderiv_rCoordPolynomial
    (h : Complex[X]) (i : Fin 3) :
    MvPolynomial.pderiv i (rCoordPolynomial h) = rCoordDerivative h i := by
  simp only [rCoordPolynomial, map_sub, MvPolynomial.pderiv_mul,
    MvPolynomial.pderiv_pow, MvPolynomial.pderiv_C,
    pderiv_baseBPolynomial, pderiv_baseRPolynomial,
    pderiv_deformationPolynomial_eq, pderiv_sourceX, zero_mul, zero_add]
  unfold rCoordDerivative
  ring

/-- The formal Jacobian determinant is the constant `-2`, independently of
the deformation polynomial. -/
theorem Fh_jacobianDet (h : Complex[X]) :
    jacobianDet (FhPolynomial h) = MvPolynomial.C (-2 : Complex) := by
  rw [jacobianDet, Matrix.det_fin_three]
  simp only [formalJacobian, FhPolynomial_zero, FhPolynomial_one,
    FhPolynomial_two, pderiv_pCoordPolynomial, pderiv_qCoordPolynomial,
    pderiv_rCoordPolynomial]
  simp [pCoordDerivative, qCoordDerivative, rCoordDerivative,
    deformationDerivative, baseADerivative, baseBDerivative,
    baseQDerivative, baseRDerivative, sourceXDerivative]
  apply MvPolynomial.funext
  intro u
  simp only [map_add, map_sub, map_mul, map_pow, map_neg,
    MvPolynomial.eval_C, eval_sourceX, eval_sourceY, eval_sourceZ,
    eval_baseAPolynomial, eval_baseBPolynomial, eval_deformationPolynomial]
  ring_nf
  simp [baseA, baseB]
  ring

/-- Every member of the family is Keller. -/
theorem Fh_isKeller (h : Complex[X]) : IsKeller (FhPolynomial h) := by
  refine ⟨-2, by norm_num, Fh_jacobianDet h⟩

/-- Polynomial model of CEX-004. -/
noncomputable def F004Polynomial : PolynomialMap3 := FhPolynomial eta004

/-- Polynomial model of CEX-006. -/
noncomputable def F006Polynomial : PolynomialMap3 := FhPolynomial eta006

/-- The CEX-004 polynomial model has determinant `-2`. -/
theorem F004_jacobianDet :
    jacobianDet F004Polynomial = MvPolynomial.C (-2 : Complex) := by
  simpa [F004Polynomial] using Fh_jacobianDet eta004

/-- The CEX-006 polynomial model has determinant `-2`. -/
theorem F006_jacobianDet :
    jacobianDet F006Polynomial = MvPolynomial.C (-2 : Complex) := by
  simpa [F006Polynomial] using Fh_jacobianDet eta006

/-- CEX-004 has nonzero constant Jacobian determinant. -/
theorem F004_isKeller : IsKeller F004Polynomial := by
  simpa [F004Polynomial] using Fh_isKeller eta004

/-- CEX-006 has nonzero constant Jacobian determinant. -/
theorem F006_isKeller : IsKeller F006Polynomial := by
  simpa [F006Polynomial] using Fh_isKeller eta006

end

end DegreeSixKeller
