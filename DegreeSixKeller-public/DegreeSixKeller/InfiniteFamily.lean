import DegreeSixKeller.GeneralComponents
import DegreeSixKeller.GenericDegreeAutomorphism

/-!
# An infinite family of inequivalent Keller maps

For each `m`, the deformation polynomial has the `m` distinct positive
integer roots `1, ..., m`.  The resulting nonproperness sets have `m + 2`
irreducible components, which separates the maps under polynomial
left-right equivalence.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

/-- The explicit set of `m` distinct positive integral roots. -/
def hFamilyRoots (m : Nat) : Finset Complex :=
  (Finset.range m).image (fun i => ((i + 1 : Nat) : Complex))

/-- The deformation polynomial with roots `1, ..., m`. -/
def hFamily (m : Nat) : Complex[X] :=
  (hFamilyRoots m).prod (fun alpha => X - C alpha)

theorem positiveNatRoot_injective :
    Function.Injective (fun i : Nat => ((i + 1 : Nat) : Complex)) := by
  intro i j hij
  have hnat : i + 1 = j + 1 := Nat.cast_injective hij
  omega

@[simp]
theorem hFamilyRoots_zero : hFamilyRoots 0 = ∅ := by
  simp [hFamilyRoots]

@[simp]
theorem hFamilyRoots_card (m : Nat) :
    (hFamilyRoots m).card = m := by
  rw [hFamilyRoots,
    Finset.card_image_of_injective _ positiveNatRoot_injective,
    Finset.card_range]

@[simp]
theorem zero_not_mem_hFamilyRoots (m : Nat) :
    0 ∉ hFamilyRoots m := by
  intro hmem
  rcases Finset.mem_image.mp hmem with ⟨i, _hi, hEq⟩
  have hEq' : ((i + 1 : Nat) : Complex) = ((0 : Nat) : Complex) := by
    simpa using hEq
  have hnat : i + 1 = 0 := Nat.cast_injective hEq'
  omega

theorem hFamily_ne_zero (m : Nat) : hFamily m ≠ 0 := by
  unfold hFamily
  exact Finset.prod_ne_zero_iff.mpr (fun alpha _halpha => X_sub_C_ne_zero alpha)

@[simp]
theorem hFamily_zero : hFamily 0 = 1 := by
  simp [hFamily, hFamilyRoots]

/-- The multiset of roots has multiplicity one at exactly the prescribed
parameters. -/
theorem hFamily_roots (m : Nat) :
    (hFamily m).roots = (hFamilyRoots m).val := by
  exact Polynomial.roots_prod_X_sub_C (hFamilyRoots m)

@[simp]
theorem hFamily_roots_toFinset (m : Nat) :
    (hFamily m).roots.toFinset = hFamilyRoots m := by
  rw [hFamily_roots]
  simp

/-- The accepted distinct-nonzero-root index recovers the prescribed set. -/
@[simp]
theorem nonzeroRoots_hFamily (m : Nat) :
    nonzeroRoots (hFamily m) = hFamilyRoots m := by
  unfold nonzeroRoots
  rw [hFamily_roots_toFinset]
  apply Finset.filter_eq_self.mpr
  intro alpha halpha hzero
  subst alpha
  exact zero_not_mem_hFamilyRoots m halpha

@[simp]
theorem nonzeroRoots_hFamily_card (m : Nat) :
    (nonzeroRoots (hFamily m)).card = m := by
  rw [nonzeroRoots_hFamily, hFamilyRoots_card]

theorem hFamily_isKeller (m : Nat) :
    IsKeller (FhPolynomial (hFamily m)) :=
  Fh_isKeller (hFamily m)

@[simp]
theorem hFamily_genericDegree (m : Nat) :
    genericDegree (FhPolynomial (hFamily m)) = 6 :=
  Fh_genericDegree_six (hFamily m) (hFamily_ne_zero m)

theorem hFamily_genericDegree_six (m : Nat) :
    genericDegree (FhPolynomial (hFamily m)) = 6 :=
  hFamily_genericDegree m

theorem hFamily_notPolynomialAutomorphism (m : Nat) :
    ¬ IsPolynomialAutomorphism (FhPolynomial (hFamily m)) :=
  Fh_notPolynomialAutomorphism (hFamily m) (hFamily_ne_zero m)

@[simp]
theorem hFamily_componentCount (m : Nat) :
    algebraicComponentCount
        (NonpropernessSet (Fh (hFamily m))) = m + 2 := by
  rw [Fh_componentCount (hFamily m) (hFamily_ne_zero m),
    nonzeroRoots_hFamily_card]
  omega

/-- Polynomial left-right equivalence preserves algebraic component count. -/
theorem algebraicComponentCount_eq_of_polynomialLeftRightEquivalent
    {F G : C3 → C3} (hEq : PolynomialLeftRightEquivalent F G) :
    algebraicComponentCount (NonpropernessSet G) =
      algebraicComponentCount (NonpropernessSet F) :=
  algebraicComponentCount_eq_of_algebraicEscapeLeftRightEquivalent
    (algebraicEscapeLeftRightEquivalent_of_polynomialLeftRightEquivalent hEq)

/-- Different family indices cannot be polynomial-left-right equivalent. -/
theorem hFamily_not_polynomialLeftRightEquivalent_of_ne
    {m n : Nat} (hmn : m ≠ n) :
    ¬ PolynomialLeftRightEquivalent
      (Fh (hFamily m)) (Fh (hFamily n)) := by
  intro hEq
  have hcount :=
    algebraicComponentCount_eq_of_polynomialLeftRightEquivalent hEq
  rw [hFamily_componentCount n, hFamily_componentCount m] at hcount
  omega

/-- Distinct indices give distinct point maps. -/
theorem hFamily_map_injective :
    Function.Injective (fun m : Nat => Fh (hFamily m)) := by
  intro m n hmaps
  have hcount := congrArg
    (fun F : C3 → C3 => algebraicComponentCount (NonpropernessSet F)) hmaps
  rw [hFamily_componentCount m, hFamily_componentCount n] at hcount
  omega

/-- The component-count invariant itself takes infinitely many values on the
family. -/
theorem hFamily_componentCount_range_infinite :
    Set.Infinite (Set.range (fun m : Nat =>
      algebraicComponentCount (NonpropernessSet (Fh (hFamily m))))) := by
  apply Set.infinite_range_of_injective
  intro m n hmn
  simpa only [hFamily_componentCount, Nat.add_right_cancel_iff] using hmn

/-- A literal infinite set of point maps whose distinct members are pairwise
inequivalent under polynomial source and target automorphisms. -/
theorem hFamily_infinite_pairwise_polynomialLeftRightInequivalent :
    Set.Infinite (Set.range (fun m : Nat => Fh (hFamily m))) ∧
      (Set.range (fun m : Nat => Fh (hFamily m))).Pairwise
        (fun F G => ¬ PolynomialLeftRightEquivalent F G) := by
  constructor
  · exact Set.infinite_range_of_injective hFamily_map_injective
  · rintro F ⟨m, rfl⟩ G ⟨n, rfl⟩ hmaps
    apply hFamily_not_polynomialLeftRightEquivalent_of_ne
    intro hmn
    subst n
    exact hmaps rfl

end

end DegreeSixKeller
