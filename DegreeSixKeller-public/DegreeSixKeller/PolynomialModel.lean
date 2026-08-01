import DegreeSixKeller.FetaCore
import Mathlib.Algebra.MvPolynomial.Eval
import Mathlib.Algebra.MvPolynomial.Equiv
import Mathlib.Tactic.FinCases

/-!
# Polynomial model of the degree-six Keller family

This module reifies the already established pointwise map `Fh` as a triple of
multivariate polynomials and proves that evaluation of that triple is exactly
the old map.  It deliberately contains no Jacobian or function-field theory.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open MvPolynomial Polynomial

noncomputable section

/-- A polynomial self-map of complex affine three-space. -/
abbrev PolynomialMap3 := Fin 3 -> MvPolynomial (Fin 3) Complex

namespace PolynomialMap3

/-- Evaluate a polynomial coordinate triple at a point of `C3`. -/
def eval (P : PolynomialMap3) (u : C3) : C3 :=
  fun i => MvPolynomial.eval u (P i)

end PolynomialMap3

/-- The three source coordinate polynomials. -/
def sourceX : MvPolynomial (Fin 3) Complex := MvPolynomial.X 0
def sourceY : MvPolynomial (Fin 3) Complex := MvPolynomial.X 1
def sourceZ : MvPolynomial (Fin 3) Complex := MvPolynomial.X 2

/-- The polynomial `A = 1 + xy`. -/
def baseAPolynomial : MvPolynomial (Fin 3) Complex :=
  1 + sourceX * sourceY

/-- The polynomial `B = A^2 z + y^2 (4 + 3xy)`. -/
def baseBPolynomial : MvPolynomial (Fin 3) Complex :=
  baseAPolynomial ^ 2 * sourceZ +
    sourceY ^ 2 * (MvPolynomial.C 4 + MvPolynomial.C 3 * sourceX * sourceY)

/-- The first target coordinate `P = AB`. -/
def pCoordPolynomial : MvPolynomial (Fin 3) Complex :=
  baseAPolynomial * baseBPolynomial

/-- The undeformed second target coordinate. -/
def baseQPolynomial : MvPolynomial (Fin 3) Complex :=
  sourceY + MvPolynomial.C 3 * sourceX * baseBPolynomial

/-- The undeformed third target coordinate. -/
def baseRPolynomial : MvPolynomial (Fin 3) Complex :=
  MvPolynomial.C 2 * sourceX -
    MvPolynomial.C 3 * sourceX ^ 2 * sourceY - sourceX ^ 3 * sourceZ

/-- Substitute the polynomial `P` into the univariate deformation polynomial. -/
def deformationPolynomial (h : Complex[X]) : MvPolynomial (Fin 3) Complex :=
  h.eval₂ MvPolynomial.C pCoordPolynomial

/-- The deformed second target coordinate as a multivariate polynomial. -/
def qCoordPolynomial (h : Complex[X]) : MvPolynomial (Fin 3) Complex :=
  baseQPolynomial + baseAPolynomial ^ 2 * sourceX ^ 4 *
    baseBPolynomial ^ 6 * deformationPolynomial h

/-- The deformed third target coordinate as a multivariate polynomial. -/
def rCoordPolynomial (h : Complex[X]) : MvPolynomial (Fin 3) Complex :=
  baseRPolynomial - MvPolynomial.C (2 / 3 : Complex) * sourceX ^ 6 *
    baseBPolynomial ^ 6 * deformationPolynomial h

/-- The canonical polynomial-coordinate representation of `Fh`. -/
def FhPolynomial (h : Complex[X]) : PolynomialMap3
  | 0 => pCoordPolynomial
  | 1 => qCoordPolynomial h
  | 2 => rCoordPolynomial h

@[simp] theorem eval_sourceX (u : C3) :
    MvPolynomial.eval u sourceX = u 0 := by simp [sourceX]

@[simp] theorem eval_sourceY (u : C3) :
    MvPolynomial.eval u sourceY = u 1 := by simp [sourceY]

@[simp] theorem eval_sourceZ (u : C3) :
    MvPolynomial.eval u sourceZ = u 2 := by simp [sourceZ]

@[simp] theorem eval_baseAPolynomial (u : C3) :
    MvPolynomial.eval u baseAPolynomial = baseA (u 0) (u 1) := by
  simp [baseAPolynomial, baseA]

@[simp] theorem eval_baseBPolynomial (u : C3) :
    MvPolynomial.eval u baseBPolynomial = baseB (u 0) (u 1) (u 2) := by
  simp [baseBPolynomial, baseB]

@[simp] theorem eval_pCoordPolynomial (u : C3) :
    MvPolynomial.eval u pCoordPolynomial = pCoord (u 0) (u 1) (u 2) := by
  simp [pCoordPolynomial, pCoord]

@[simp] theorem eval_baseQPolynomial (u : C3) :
    MvPolynomial.eval u baseQPolynomial = baseQ (u 0) (u 1) (u 2) := by
  simp [baseQPolynomial, baseQ]

@[simp] theorem eval_baseRPolynomial (u : C3) :
    MvPolynomial.eval u baseRPolynomial = baseR (u 0) (u 1) (u 2) := by
  simp [baseRPolynomial, baseR]

@[simp] theorem eval_deformationPolynomial (h : Complex[X]) (u : C3) :
    MvPolynomial.eval u (deformationPolynomial h) =
      h.eval (pCoord (u 0) (u 1) (u 2)) := by
  unfold deformationPolynomial
  rw [Polynomial.hom_eval₂]
  have hcomp :
      (MvPolynomial.eval u).comp MvPolynomial.C = RingHom.id Complex := by
    ext c
    simp
  rw [hcomp, eval_pCoordPolynomial]
  rfl

@[simp] theorem eval_qCoordPolynomial (h : Complex[X]) (u : C3) :
    MvPolynomial.eval u (qCoordPolynomial h) =
      qCoord h (u 0) (u 1) (u 2) := by
  simp [qCoordPolynomial, qCoord]

@[simp] theorem eval_rCoordPolynomial (h : Complex[X]) (u : C3) :
    MvPolynomial.eval u (rCoordPolynomial h) =
      rCoord h (u 0) (u 1) (u 2) := by
  simp [rCoordPolynomial, rCoord]

@[simp] theorem FhPolynomial_zero (h : Complex[X]) :
    FhPolynomial h 0 = pCoordPolynomial := rfl

@[simp] theorem FhPolynomial_one (h : Complex[X]) :
    FhPolynomial h 1 = qCoordPolynomial h := rfl

@[simp] theorem FhPolynomial_two (h : Complex[X]) :
    FhPolynomial h 2 = rCoordPolynomial h := rfl

/-- Evaluation of the polynomial model is exactly the established point map. -/
theorem eval_FhPolynomial (h : Complex[X]) (u : C3) :
    PolynomialMap3.eval (FhPolynomial h) u = Fh h u := by
  funext i
  fin_cases i <;> simp [PolynomialMap3.eval, Fh]

end

end DegreeSixKeller
