import DegreeSixKeller.FunctionFieldSpike

/-!
# Disposable generic-degree generator spike

This file instantiates the actual degree-six inverse polynomial over the
parameter function field.  It is deliberately isolated from the production
generic-degree module.
-/

set_option autoImplicit false

namespace DegreeSixKeller.GenericDegreeGeneratorSpike

open Polynomial
open DegreeSixKeller.FunctionFieldSpike

noncomputable section

abbrev K := ParameterFunctionField

abbrev L := SimpleExtensionFunctionField

/-- The two canonical algebraically independent parameters. -/
def p : K := parameterCoordinate 0

def q : K := parameterCoordinate 1

/-- Evaluation of a complex polynomial at the transcendental parameter `p`. -/
def evaluateAtP (h : Polynomial Complex) : K :=
  Polynomial.aeval p h

theorem p_transcendental : Transcendental Complex p := by
  exact parameterCoordinate_algebraicIndependent.transcendental 0

theorem evaluateAtP_ne_zero {h : Polynomial Complex} (hh : h ≠ 0) :
    evaluateAtP h ≠ 0 := by
  exact (map_ne_zero_iff (Polynomial.aeval p)
    (transcendental_iff_injective.mp p_transcendental)).mpr hh

theorem p_ne_zero : p ≠ 0 := by
  simpa [evaluateAtP] using
    (evaluateAtP_ne_zero (h := (Polynomial.X : Polynomial Complex))
      Polynomial.X_ne_zero)

/-- The coefficient of the sixth-degree term in the inverse polynomial. -/
def inverseLeadingCoefficient (h : Polynomial Complex) : K :=
  ((1 : K) / 3) * p ^ 6 * evaluateAtP h

theorem inverseLeadingCoefficient_ne_zero {h : Polynomial Complex}
    (hh : h ≠ 0) : inverseLeadingCoefficient h ≠ 0 := by
  exact mul_ne_zero
    (mul_ne_zero (by norm_num) (pow_ne_zero 6 p_ne_zero))
    (evaluateAtP_ne_zero hh)

/--
The univariate inverse equation

`(1/3) p^6 h(p) S^6 + 2 p S^3 - q S^2 + 2 S`.
-/
def inversePolynomial (h : Polynomial Complex) : Polynomial K :=
  C (inverseLeadingCoefficient h) * X ^ 6 +
    C (2 * p) * X ^ 3 - C q * X ^ 2 + C 2 * X

theorem inversePolynomial_natDegree_le (h : Polynomial Complex) :
    (inversePolynomial h).natDegree ≤ 6 := by
  unfold inversePolynomial
  refine (natDegree_add_le _ _).trans (max_le ?_ ?_)
  · refine (natDegree_sub_le _ _).trans (max_le ?_ ?_)
    · refine (natDegree_add_le _ _).trans (max_le ?_ ?_)
      · exact natDegree_C_mul_X_pow_le _ _
      · exact (natDegree_C_mul_X_pow_le _ _).trans (by omega)
    · exact (natDegree_C_mul_X_pow_le _ _).trans (by omega)
  · simpa only [pow_one] using
      (natDegree_C_mul_X_pow_le (2 : K) 1).trans (by omega)

@[simp]
theorem inversePolynomial_coeff_six (h : Polynomial Complex) :
    (inversePolynomial h).coeff 6 = inverseLeadingCoefficient h := by
  simp only [inversePolynomial, coeff_add, coeff_sub,
    coeff_C_mul_X_pow, coeff_C_mul_X]
  norm_num

theorem inversePolynomial_natDegree {h : Polynomial Complex} (hh : h ≠ 0) :
    (inversePolynomial h).natDegree = 6 := by
  exact natDegree_eq_of_le_of_coeff_ne_zero
    (inversePolynomial_natDegree_le h)
    (inversePolynomial_coeff_six h |>.trans_ne
      (inverseLeadingCoefficient_ne_zero hh))

theorem inversePolynomial_leadingCoeff {h : Polynomial Complex} (hh : h ≠ 0) :
    (inversePolynomial h).leadingCoeff = inverseLeadingCoefficient h := by
  rw [leadingCoeff, inversePolynomial_natDegree hh]
  exact inversePolynomial_coeff_six h

theorem inversePolynomial_leadingCoeff_ne_zero {h : Polynomial Complex}
    (hh : h ≠ 0) : (inversePolynomial h).leadingCoeff ≠ 0 := by
  rw [inversePolynomial_leadingCoeff hh]
  exact inverseLeadingCoefficient_ne_zero hh

/-- The inverse polynomial embedded in the staged rational-function field. -/
def inverseGenerator (h : Polynomial Complex) : L :=
  algebraMap (Polynomial K) L (inversePolynomial h)

@[simp]
theorem inverseGenerator_num (h : Polynomial Complex) :
    (inverseGenerator h).num = inversePolynomial h := by
  exact RatFunc.num_algebraMap _

@[simp]
theorem inverseGenerator_denom (h : Polynomial Complex) :
    (inverseGenerator h).denom = (1 : Polynomial K) := by
  exact RatFunc.denom_algebraMap _

theorem inverseGenerator_num_natDegree {h : Polynomial Complex} (hh : h ≠ 0) :
    (inverseGenerator h).num.natDegree = 6 := by
  rw [inverseGenerator_num, inversePolynomial_natDegree hh]

theorem inverseGenerator_denom_natDegree (h : Polynomial Complex) :
    (inverseGenerator h).denom.natDegree = 0 := by
  simp

/-- The pinned RatFunc degree theorem gives generic degree exactly six. -/
theorem inverseGenerator_finrank {h : Polynomial Complex} (hh : h ≠ 0) :
    Module.finrank (generatedIntermediateField (inverseGenerator h)) L = 6 := by
  rw [ratFunc_finrank_eq_max_natDegree]
  simp [inversePolynomial_natDegree hh]

end

end DegreeSixKeller.GenericDegreeGeneratorSpike
