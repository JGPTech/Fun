import Mathlib.Algebra.Polynomial.Derivative
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Degree-six Keller family: core formulas

This module encodes the explicit maps and the univariate inverse polynomial
used by the human proof.  It proves only exact algebraic identities.  It does
not assert the geometric nonproperness theorem or generic-degree statement.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial
open scoped Polynomial

/-- Complex affine three-space. -/
abbrev C3 := Fin 3 -> Complex

/-- The marked factor `A = 1 + xy`. -/
def baseA (x y : Complex) : Complex := 1 + x * y

/-- The marked factor `B = A^2 z + y^2 (4 + 3xy)`. -/
def baseB (x y z : Complex) : Complex :=
  baseA x y ^ 2 * z + y ^ 2 * (4 + 3 * x * y)

/-- The first target coordinate `P = AB`. -/
def pCoord (x y z : Complex) : Complex :=
  baseA x y * baseB x y z

/-- The undeformed second coordinate `Q = y + 3xB`. -/
def baseQ (x y z : Complex) : Complex :=
  y + 3 * x * baseB x y z

/-- The undeformed third coordinate `R = 2x - 3x^2y - x^3z`. -/
def baseR (x y z : Complex) : Complex :=
  2 * x - 3 * x ^ 2 * y - x ^ 3 * z

/-- The degree-six deformation of the second coordinate. -/
def qCoord (h : Complex[X]) (x y z : Complex) : Complex :=
  baseQ x y z
    + baseA x y ^ 2 * x ^ 4 * baseB x y z ^ 6 * h.eval (pCoord x y z)

/-- The degree-six deformation of the third coordinate. -/
noncomputable def rCoord (h : Complex[X]) (x y z : Complex) : Complex :=
  baseR x y z
    - (2 / 3 : Complex) * x ^ 6 * baseB x y z ^ 6 * h.eval (pCoord x y z)

/-- The explicit polynomial map `F_h : C^3 -> C^3`. -/
noncomputable def Fh (h : Complex[X]) (u : C3) : C3 :=
  let x := u 0
  let y := u 1
  let z := u 2
  ![pCoord x y z, qCoord h x y z, rCoord h x y z]

/-- Leading coefficient of the inverse polynomial in the chart variable. -/
noncomputable def aCoeff (h : Complex[X]) (p : Complex) : Complex :=
  (1 / 3 : Complex) * p ^ 6 * h.eval p

/--
The inverse-root polynomial
`Omega(s) = a_h(p)s^6 + 2ps^3 - qs^2 + 2s - r`.
-/
noncomputable def omega (h : Complex[X]) (p q r : Complex) : Complex[X] :=
  C (aCoeff h p) * X ^ 6
    + C (2 * p) * X ^ 3
    - C q * X ^ 2
    + C 2 * X
    - C r

@[simp]
theorem omega_eval (h : Complex[X]) (p q r s : Complex) :
    (omega h p q r).eval s =
      aCoeff h p * s ^ 6 + 2 * p * s ^ 3 - q * s ^ 2 + 2 * s - r := by
  simp [omega]

@[simp]
theorem omega_coeff_six (h : Complex[X]) (p q r : Complex) :
    (omega h p q r).coeff 6 = aCoeff h p := by
  simp only [omega, coeff_sub, coeff_add, coeff_C_mul_X_pow,
    coeff_C_mul_X, coeff_C]
  all_goals norm_num

/-- Exact evaluated derivative of the inverse polynomial. -/
theorem omega_derivative_eval (h : Complex[X]) (p q r s : Complex) :
    (derivative (omega h p q r)).eval s =
      2 * p ^ 6 * h.eval p * s ^ 5 + 6 * p * s ^ 2 - 2 * q * s + 2 := by
  simp only [omega, derivative_sub, derivative_add, derivative_C_mul_X_pow,
    derivative_C_mul_X, derivative_C]
  simp [aCoeff]
  all_goals ring

/-- The vertical leading-coefficient locus is exactly `p = 0` or `h(p) = 0`. -/
theorem aCoeff_eq_zero_iff (h : Complex[X]) (p : Complex) :
    aCoeff h p = 0 ↔ p = 0 ∨ h.eval p = 0 := by
  simp [aCoeff, mul_eq_zero]

end DegreeSixKeller
