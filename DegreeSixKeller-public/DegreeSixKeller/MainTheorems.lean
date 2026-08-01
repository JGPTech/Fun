import DegreeSixKeller.InfiniteFamily

/-!
# Final theorem assembly

This file assembles the accepted Keller, generic-degree, nonautomorphism,
component-count, and polynomial-left-right inequivalence endpoints into the
two parameter-free main statements.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

/-- A deformation parameter whose associated polynomial map has nonzero
constant Jacobian, generic degree six, and no polynomial inverse. -/
def IsDegreeSixKellerCounterexample (h : Complex[X]) : Prop :=
  IsKeller (FhPolynomial h) ∧
    genericDegree (FhPolynomial h) = 6 ∧
      ¬ IsPolynomialAutomorphism (FhPolynomial h)

theorem F004_isDegreeSixKellerCounterexample :
    IsDegreeSixKellerCounterexample eta004 := by
  exact ⟨Fh_isKeller eta004,
    Fh_genericDegree_six eta004 eta004_ne_zero,
    Fh_notPolynomialAutomorphism eta004 eta004_ne_zero⟩

theorem F006_isDegreeSixKellerCounterexample :
    IsDegreeSixKellerCounterexample eta006 := by
  exact ⟨Fh_isKeller eta006,
    Fh_genericDegree_six eta006 eta006_ne_zero,
    Fh_notPolynomialAutomorphism eta006 eta006_ne_zero⟩

theorem hFamily_isDegreeSixKellerCounterexample (m : Nat) :
    IsDegreeSixKellerCounterexample (hFamily m) := by
  exact ⟨hFamily_isKeller m, hFamily_genericDegree_six m,
    hFamily_notPolynomialAutomorphism m⟩

/-- The two concrete degree-six Keller counterexamples have respectively
three and two irreducible nonproperness components and are not polynomially
left-right equivalent. -/
theorem theoremA :
    IsDegreeSixKellerCounterexample eta004 ∧
      IsDegreeSixKellerCounterexample eta006 ∧
        algebraicComponentCount (NonpropernessSet F004) = 3 ∧
          algebraicComponentCount (NonpropernessSet F006) = 2 ∧
            ¬ PolynomialLeftRightEquivalent F004 F006 := by
  exact ⟨F004_isDegreeSixKellerCounterexample,
    F006_isDegreeSixKellerCounterexample,
    cex004_actual_componentCount_unconditional,
    cex006_actual_componentCount_unconditional,
    cex004_cex006_not_polynomialLeftRightEquivalent⟩

/-- For every index, the explicit family member is a degree-six Keller
counterexample with `m + 2` nonproperness components; the point maps form a
literal infinite set whose distinct members are pairwise polynomially
left-right inequivalent. -/
theorem theoremB :
    (∀ m : Nat,
      IsDegreeSixKellerCounterexample (hFamily m) ∧
        algebraicComponentCount
          (NonpropernessSet (Fh (hFamily m))) = m + 2) ∧
      (Set.Infinite (Set.range (fun m : Nat => Fh (hFamily m))) ∧
        (Set.range (fun m : Nat => Fh (hFamily m))).Pairwise
          (fun F G => ¬ PolynomialLeftRightEquivalent F G)) := by
  constructor
  · intro m
    exact ⟨hFamily_isDegreeSixKellerCounterexample m,
      hFamily_componentCount m⟩
  · exact hFamily_infinite_pairwise_polynomialLeftRightInequivalent

end

end DegreeSixKeller
