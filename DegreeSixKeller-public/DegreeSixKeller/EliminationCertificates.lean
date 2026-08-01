import DegreeSixKeller.FetaCore

/-!
# Exact specializations for CEX-004 and CEX-006

This module certifies the univariate parameter polynomials and their distinct
nonzero roots.  It is the exact-algebra input to the geometric component count.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial
open scoped Polynomial

/-- CEX-004 has `eta_4(T) = 1 + 4T`. -/
noncomputable def eta004 : Complex[X] := C 1 + C 4 * X

/-- CEX-006 has the constant parameter `eta_4(T) = -3/2`. -/
noncomputable def eta006 : Complex[X] := C (-3 / 2 : Complex)

@[simp]
theorem eta004_eval (p : Complex) : eta004.eval p = 1 + 4 * p := by
  simp [eta004]

@[simp]
theorem eta006_eval (p : Complex) : eta006.eval p = (-3 / 2 : Complex) := by
  simp [eta006]

theorem eta004_root_iff (p : Complex) :
    eta004.eval p = 0 ↔ p = (-1 / 4 : Complex) := by
  rw [eta004_eval]
  constructor
  · intro hp
    apply (eq_div_iff (show (4 : Complex) ≠ 0 by norm_num)).2
    linear_combination hp
  · rintro rfl
    norm_num

theorem eta006_not_root (p : Complex) : eta006.eval p ≠ 0 := by
  rw [eta006_eval]
  norm_num

theorem aCoeff_eta004_zero_iff (p : Complex) :
    aCoeff eta004 p = 0 ↔ p = 0 ∨ p = (-1 / 4 : Complex) := by
  rw [aCoeff_eq_zero_iff, eta004_root_iff]

theorem aCoeff_eta006_zero_iff (p : Complex) :
    aCoeff eta006 p = 0 ↔ p = 0 := by
  rw [aCoeff_eq_zero_iff]
  simp

/-- The actual CEX-004 map. -/
noncomputable def F004 : C3 -> C3 := Fh eta004

/-- The actual CEX-006 map. -/
noncomputable def F006 : C3 -> C3 := Fh eta006

end DegreeSixKeller
