import Mathlib.Algebra.MvPolynomial.Equiv
import Mathlib.Data.Complex.Basic
import Mathlib.FieldTheory.RatFunc.IntermediateField
import Mathlib.RingTheory.AlgebraicIndependent.TranscendenceBasis
import Mathlib.RingTheory.Localization.Algebra
import Mathlib.RingTheory.Localization.LocalizationLocalization

/-!
# Disposable function-field API spike

This file isolates the function-field types and theorem applications needed by
the later generic-degree certificate.  It deliberately does not depend on the
project's polynomial-map implementation.
-/

set_option autoImplicit false

namespace DegreeSixKeller.FunctionFieldSpike

open Function IntermediateField algebraAdjoinAdjoin MvPolynomial Polynomial Set
open scoped IntermediateField RatFunc

noncomputable section

/-! ## The three-variable source function field -/

abbrev SourcePolynomialRing := MvPolynomial (Fin 3) Complex

abbrev SourceFunctionField := FractionRing SourcePolynomialRing

example : Field SourceFunctionField := inferInstance

example : Algebra Complex SourceFunctionField := inferInstance

example : Algebra SourcePolynomialRing SourceFunctionField := inferInstance

example : IsScalarTower Complex SourcePolynomialRing SourceFunctionField :=
  inferInstance

/-- The canonical inclusion of the polynomial ring in its fraction field,
regarded as a `Complex`-algebra homomorphism. -/
def sourcePolynomialInclusion :
    SourcePolynomialRing →ₐ[Complex] SourceFunctionField :=
  IsScalarTower.toAlgHom Complex SourcePolynomialRing SourceFunctionField

/-- The three canonical rational coordinate functions. -/
def sourceCoordinate : Fin 3 → SourceFunctionField :=
  sourcePolynomialInclusion ∘ MvPolynomial.X

@[simp]
theorem sourceCoordinate_eq_algebraMap (i : Fin 3) :
    sourceCoordinate i =
      algebraMap SourcePolynomialRing SourceFunctionField (MvPolynomial.X i) :=
  rfl

/-- The canonical rational coordinates remain algebraically independent after
passing from the polynomial ring to its fraction field. -/
theorem sourceCoordinate_algebraicIndependent :
    AlgebraicIndependent Complex sourceCoordinate := by
  exact (MvPolynomial.algebraicIndependent_X (Fin 3) Complex).map'
    (FaithfulSMul.algebraMap_injective
      SourcePolynomialRing SourceFunctionField)

theorem sourceCoordinate_transcendental (i : Fin 3) :
    Transcendental Complex (sourceCoordinate i) :=
  sourceCoordinate_algebraicIndependent.transcendental i

noncomputable def sourceCoordinateField :
    IntermediateField Complex SourceFunctionField :=
  IntermediateField.adjoin Complex (Set.range sourceCoordinate)

/-! `finSuccEquiv` separates one variable before passage to fraction fields. -/

abbrev SplitSourcePolynomialRing :=
  Polynomial (MvPolynomial (Fin 2) Complex)

abbrev SplitSourceFunctionField := FractionRing SplitSourcePolynomialRing

noncomputable def sourcePolynomialFinSuccEquiv :
    SourcePolynomialRing ≃ₐ[Complex] SplitSourcePolynomialRing :=
  MvPolynomial.finSuccEquiv Complex 2

noncomputable def sourceFractionFinSuccEquiv :
    SourceFunctionField ≃ₐ[Complex] SplitSourceFunctionField :=
  IsFractionRing.algEquivOfAlgEquiv sourcePolynomialFinSuccEquiv

/-- A last-coordinate-oriented variant of `finSuccEquiv`.  The preliminary
swap sends `Fin.last 2` to the zero coordinate singled out by Mathlib's
`finSuccEquiv`. -/
noncomputable def sourcePolynomialLastEquiv :
    AlgEquiv Complex SourcePolynomialRing SplitSourcePolynomialRing :=
  (MvPolynomial.renameEquiv Complex
    (Equiv.swap (Fin.last 2) 0)).trans sourcePolynomialFinSuccEquiv

noncomputable def sourceFractionLastEquiv :
    AlgEquiv Complex SourceFunctionField SplitSourceFunctionField :=
  IsFractionRing.algEquivOfAlgEquiv sourcePolynomialLastEquiv

/-! ## A staged `Complex(p,q)(s)` model -/

abbrev ParameterPolynomialRing := MvPolynomial (Fin 2) Complex

abbrev ParameterFunctionField := FractionRing ParameterPolynomialRing

def parameterPolynomialInclusion :
    ParameterPolynomialRing →ₐ[Complex] ParameterFunctionField :=
  IsScalarTower.toAlgHom Complex ParameterPolynomialRing ParameterFunctionField

def parameterCoordinate : Fin 2 → ParameterFunctionField :=
  parameterPolynomialInclusion ∘ MvPolynomial.X

theorem parameterCoordinate_algebraicIndependent :
    AlgebraicIndependent Complex parameterCoordinate := by
  exact (MvPolynomial.algebraicIndependent_X (Fin 2) Complex).map'
    (FaithfulSMul.algebraMap_injective
      ParameterPolynomialRing ParameterFunctionField)

abbrev SimpleExtensionFunctionField := RatFunc ParameterFunctionField

/-! ## Canonical comparison with the split source fraction field

The coefficient map from `ParameterPolynomialRing` to
`ParameterFunctionField`
extends coefficientwise to polynomials and then into `RatFunc`.  This is the
ring map over which the latter is the fraction field of
`SplitSourcePolynomialRing`.
-/

noncomputable def splitPolynomialToRatFuncRingHom :
    RingHom SplitSourcePolynomialRing SimpleExtensionFunctionField :=
  (algebraMap (Polynomial ParameterFunctionField)
      SimpleExtensionFunctionField).comp
    (Polynomial.mapRingHom
      (algebraMap ParameterPolynomialRing ParameterFunctionField))

noncomputable local instance splitPolynomialMapAlgebra :
    Algebra SplitSourcePolynomialRing
      (Polynomial ParameterFunctionField) :=
  (Polynomial.mapRingHom
    (algebraMap ParameterPolynomialRing ParameterFunctionField)).toAlgebra

/-! Polynomial rings preserve localization.  Composing that localization with
the ordinary RatFunc localization shows that `RatFunc (FractionRing A)` is a
fraction ring of `Polynomial A` itself. -/
noncomputable local instance splitPolynomialRatFuncIsFractionRing :
    IsFractionRing SplitSourcePolynomialRing
      SimpleExtensionFunctionField := by
  letI : IsLocalization
      ((nonZeroDivisors ParameterPolynomialRing).map Polynomial.C)
      (Polynomial ParameterFunctionField) :=
    Polynomial.isLocalization _ _
  have hLocalization :
      IsLocalization
        ((nonZeroDivisors (Polynomial ParameterFunctionField)).comap
          (algebraMap SplitSourcePolynomialRing
            (Polynomial ParameterFunctionField)))
        SimpleExtensionFunctionField :=
    IsLocalization.localization_localization_isLocalization_of_has_all_units
      ((nonZeroDivisors ParameterPolynomialRing).map Polynomial.C)
      (nonZeroDivisors (Polynomial ParameterFunctionField))
      SimpleExtensionFunctionField fun x hx =>
        mem_nonZeroDivisors_iff_ne_zero.mpr hx.ne_zero
  convert hLocalization using 1
  ext p
  simp only [Submonoid.mem_comap, mem_nonZeroDivisors_iff_ne_zero]
  exact (map_ne_zero_iff _
    (FaithfulSMul.algebraMap_injective SplitSourcePolynomialRing
      (Polynomial ParameterFunctionField))).symm

/-- The canonical equivalence between fractioning the whole split polynomial
ring and fractioning its coefficient ring before forming rational functions. -/
noncomputable def splitFractionToRatFuncEquiv :
    AlgEquiv Complex SplitSourceFunctionField
      SimpleExtensionFunctionField :=
  (FractionRing.algEquiv SplitSourcePolynomialRing
    SimpleExtensionFunctionField).restrictScalars Complex

/-! The split total fraction field also contains the parameter fraction field
canonically, by the universal property of `FractionRing`. -/
noncomputable local instance parameterFunctionSplitAlgebra :
    Algebra ParameterFunctionField SplitSourceFunctionField :=
  FractionRing.liftAlgebra ParameterPolynomialRing
    SplitSourceFunctionField

/-- The same comparison, now bundled over the full parameter function field.
This is the strongest compatibility needed by the generic-degree argument. -/
noncomputable def splitFractionToRatFuncParameterEquiv :
    AlgEquiv ParameterFunctionField SplitSourceFunctionField
      SimpleExtensionFunctionField :=
  { splitFractionToRatFuncEquiv with
    commutes' := fun a =>
      (IsFractionRing.algEquiv_commutes
        (AlgEquiv.refl : AlgEquiv ParameterPolynomialRing
          ParameterFunctionField ParameterFunctionField)
        (FractionRing.algEquiv SplitSourcePolynomialRing
          SimpleExtensionFunctionField) a).symm }

/-- The exact `finSuccEquiv`-oriented staged presentation; its singled-out
source coordinate is coordinate zero. -/
noncomputable def sourceFunctionFieldFinSuccToRatFuncEquiv :
    AlgEquiv Complex SourceFunctionField SimpleExtensionFunctionField :=
  sourceFractionFinSuccEquiv.trans splitFractionToRatFuncEquiv

/-- The staged `Complex(p,q)(s)` presentation of the original three-variable
rational function field, oriented so that the last source coordinate maps to
the RatFunc variable. -/
noncomputable def sourceFunctionFieldToRatFuncEquiv :
    AlgEquiv Complex SourceFunctionField SimpleExtensionFunctionField :=
  sourceFractionLastEquiv.trans splitFractionToRatFuncEquiv

@[simp]
theorem splitFractionToRatFuncEquiv_algebraMap
    (p : SplitSourcePolynomialRing) :
    splitFractionToRatFuncEquiv
        (algebraMap SplitSourcePolynomialRing SplitSourceFunctionField p) =
      algebraMap SplitSourcePolynomialRing SimpleExtensionFunctionField p :=
  (FractionRing.algEquiv SplitSourcePolynomialRing
    SimpleExtensionFunctionField).commutes p

@[simp]
theorem algebraMap_splitPolynomial_X :
    algebraMap SplitSourcePolynomialRing SimpleExtensionFunctionField
        Polynomial.X = RatFunc.X := by
  change algebraMap (Polynomial ParameterFunctionField)
      SimpleExtensionFunctionField
        (Polynomial.map
          (algebraMap ParameterPolynomialRing ParameterFunctionField)
          Polynomial.X) = RatFunc.X
  simp

@[simp]
theorem sourceFunctionFieldFinSuccToRatFuncEquiv_coordinate_zero :
    sourceFunctionFieldFinSuccToRatFuncEquiv (sourceCoordinate 0) =
      RatFunc.X := by
  rw [sourceCoordinate_eq_algebraMap]
  simp [sourceFunctionFieldFinSuccToRatFuncEquiv,
    sourceFractionFinSuccEquiv, sourcePolynomialFinSuccEquiv,
    splitFractionToRatFuncEquiv,
    MvPolynomial.finSuccEquiv_X_zero]

@[simp]
theorem sourceFunctionFieldFinSuccToRatFuncEquiv_symm_X :
    sourceFunctionFieldFinSuccToRatFuncEquiv.symm RatFunc.X =
      sourceCoordinate 0 := by
  calc
    sourceFunctionFieldFinSuccToRatFuncEquiv.symm RatFunc.X =
        sourceFunctionFieldFinSuccToRatFuncEquiv.symm
          (sourceFunctionFieldFinSuccToRatFuncEquiv (sourceCoordinate 0)) :=
      congrArg sourceFunctionFieldFinSuccToRatFuncEquiv.symm
        sourceFunctionFieldFinSuccToRatFuncEquiv_coordinate_zero.symm
    _ = sourceCoordinate 0 :=
      sourceFunctionFieldFinSuccToRatFuncEquiv.symm_apply_apply _

@[simp]
theorem sourceFunctionFieldToRatFuncEquiv_coordinate_last :
    sourceFunctionFieldToRatFuncEquiv
        (sourceCoordinate (Fin.last 2)) = RatFunc.X := by
  rw [sourceCoordinate_eq_algebraMap]
  simp [sourceFunctionFieldToRatFuncEquiv, sourceFractionLastEquiv,
    sourcePolynomialLastEquiv, sourcePolynomialFinSuccEquiv,
    splitFractionToRatFuncEquiv,
    MvPolynomial.finSuccEquiv_X_zero]

@[simp]
theorem sourceFunctionFieldToRatFuncEquiv_symm_X :
    sourceFunctionFieldToRatFuncEquiv.symm RatFunc.X =
      sourceCoordinate (Fin.last 2) := by
  calc
    sourceFunctionFieldToRatFuncEquiv.symm RatFunc.X =
        sourceFunctionFieldToRatFuncEquiv.symm
          (sourceFunctionFieldToRatFuncEquiv
            (sourceCoordinate (Fin.last 2))) :=
      congrArg sourceFunctionFieldToRatFuncEquiv.symm
        sourceFunctionFieldToRatFuncEquiv_coordinate_last.symm
    _ = sourceCoordinate (Fin.last 2) :=
      sourceFunctionFieldToRatFuncEquiv.symm_apply_apply _

example : Field SimpleExtensionFunctionField := inferInstance

example : Algebra ParameterFunctionField SimpleExtensionFunctionField :=
  inferInstance

example : Algebra Complex SimpleExtensionFunctionField := inferInstance

example : IsScalarTower
    Complex ParameterFunctionField SimpleExtensionFunctionField :=
  inferInstance

/-- The chart root `s` in the staged univariate rational-function field. -/
def extensionCoordinate : SimpleExtensionFunctionField := RatFunc.X

theorem extensionCoordinate_transcendental :
    Transcendental ParameterFunctionField extensionCoordinate := by
  simpa [extensionCoordinate] using
    (RatFunc.transcendental_X (K := ParameterFunctionField))

/-- The intermediate field generated by `f`, used by the RatFunc degree theorem. -/
def generatedIntermediateField (f : SimpleExtensionFunctionField) :
    IntermediateField ParameterFunctionField SimpleExtensionFunctionField :=
  IntermediateField.adjoin ParameterFunctionField {f}

/-! ## Exact RatFunc theorem applications -/

theorem ratFunc_finrank_eq_max_natDegree
    (f : SimpleExtensionFunctionField) :
    Module.finrank (generatedIntermediateField f) SimpleExtensionFunctionField =
      max f.num.natDegree f.denom.natDegree :=
  f.finrank_eq_max_natDegree

theorem ratFunc_irreducible_minpolyX
    (f : SimpleExtensionFunctionField)
    (hf : ¬ ∃ c : ParameterFunctionField, f = RatFunc.C c) :
    Irreducible
      (f.minpolyX
        (IntermediateField.adjoin ParameterFunctionField {f})) :=
  f.irreducible_minpolyX hf

theorem ratFunc_X_not_constant :
    ¬ ∃ c : ParameterFunctionField,
      (RatFunc.X : SimpleExtensionFunctionField) = RatFunc.C c := by
  intro h
  have hDegree :=
    (RatFunc.eq_C_iff (RatFunc.X : SimpleExtensionFunctionField)).mp h
  simp at hDegree

theorem ratFunc_X_minpolyX_irreducible :
    Irreducible
      ((RatFunc.X : SimpleExtensionFunctionField).minpolyX
        (IntermediateField.adjoin ParameterFunctionField
          {(RatFunc.X : SimpleExtensionFunctionField)})) :=
  (RatFunc.X : SimpleExtensionFunctionField).irreducible_minpolyX
    ratFunc_X_not_constant

/-! A concrete degree-six check of the finrank formula. -/

def sixthPowerGenerator : SimpleExtensionFunctionField :=
  algebraMap (Polynomial ParameterFunctionField)
    SimpleExtensionFunctionField (Polynomial.X ^ 6)

@[simp]
theorem num_sixthPowerGenerator :
    sixthPowerGenerator.num = (Polynomial.X : Polynomial ParameterFunctionField) ^ 6 := by
  exact RatFunc.num_algebraMap _

@[simp]
theorem denom_sixthPowerGenerator :
    sixthPowerGenerator.denom = (1 : Polynomial ParameterFunctionField) := by
  exact RatFunc.denom_algebraMap _

theorem sixthPowerGenerator_not_constant :
    ¬ ∃ c : ParameterFunctionField, sixthPowerGenerator = RatFunc.C c := by
  intro h
  have hDegree := (RatFunc.eq_C_iff sixthPowerGenerator).mp h
  simp only [num_sixthPowerGenerator, denom_sixthPowerGenerator] at hDegree
  simp at hDegree

theorem sixthPowerGenerator_finrank :
    Module.finrank (generatedIntermediateField sixthPowerGenerator)
      SimpleExtensionFunctionField = 6 := by
  rw [ratFunc_finrank_eq_max_natDegree]
  simp only [num_sixthPowerGenerator, denom_sixthPowerGenerator]
  simp

theorem sixthPowerGenerator_minpolyX_irreducible :
    Irreducible
      (sixthPowerGenerator.minpolyX
        (IntermediateField.adjoin ParameterFunctionField
          {sixthPowerGenerator})) :=
  sixthPowerGenerator.irreducible_minpolyX
    sixthPowerGenerator_not_constant

end

end DegreeSixKeller.FunctionFieldSpike
