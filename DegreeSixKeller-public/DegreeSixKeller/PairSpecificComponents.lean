import DegreeSixKeller.PairSpecificGeometry

/-!
# Genuine pair-specific irreducible-component counts

This module applies the general finite-decomposition theorem to the actual
mathlib `irreducibleComponents` of the two reduced algebraic subspaces.  The
remaining geometric frontier is visible in the hypotheses: the reduced-set
equality, irreducibility of the finite component, and exclusion of a whole
vertical hyperplane from that finite component.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set Topology
open scoped Polynomial

noncomputable section

/-- The exact nonverticality condition needed for irredundancy of the
component decomposition. -/
def NoVerticalHyperplaneInFiniteComponent (h : Complex[X]) : Prop :=
  ∀ alpha : Complex, ¬ pHyperplane alpha ⊆ finiteComponent h

theorem reducedCandidate004_componentCount
    (hFiniteIrreducible : FiniteComponentIrreducible004)
    (hNoVertical : NoVerticalHyperplaneInFiniteComponent eta004) :
    algebraicComponentCount reducedCandidate004 = 3 := by
  classical
  let D := finiteComponent eta004
  let H₀ := pHyperplane 0
  let H₁ := pHyperplane (-1 / 4 : Complex)
  let C : Finset (Set C3) := {D, H₀, H₁}
  have hClosed : ∀ A ∈ C, IsClosed (zariskiLift A) := by
    intro A hA
    simp only [C, Finset.mem_insert, Finset.mem_singleton] at hA
    rcases hA with rfl | rfl | rfl
    · exact finiteComponent_isClosed eta004
    · exact pHyperplane_isClosed 0
    · exact pHyperplane_isClosed (-1 / 4 : Complex)
  have hIrreducible : ∀ A ∈ C, IsIrreducible (zariskiLift A) := by
    intro A hA
    simp only [C, Finset.mem_insert, Finset.mem_singleton] at hA
    rcases hA with rfl | rfl | rfl
    · exact hFiniteIrreducible
    · exact pHyperplane_isIrreducible 0
    · exact pHyperplane_isIrreducible (-1 / 4 : Complex)
  have hCover : ⋃₀ (C : Set (Set C3)) = reducedCandidate004 := by
    ext x
    simp [C, D, H₀, H₁, reducedCandidate004, or_assoc]
  have hIrredundant :
      ∀ A ∈ C, ∀ B ∈ C, A ⊆ B -> A = B := by
    intro A hA B hB hAB
    simp only [C, Finset.mem_insert, Finset.mem_singleton] at hA hB
    rcases hA with rfl | rfl | rfl <;>
      rcases hB with rfl | rfl | rfl
    · rfl
    · exact False.elim
        (finiteComponent_not_subset_pHyperplane eta004 0 hAB)
    · exact False.elim
        (finiteComponent_not_subset_pHyperplane eta004
          (-1 / 4 : Complex) hAB)
    · exact False.elim (hNoVertical 0 hAB)
    · rfl
    · have hEq := pHyperplane_subset_iff.mp hAB
      norm_num at hEq
    · exact False.elim (hNoVertical (-1 / 4 : Complex) hAB)
    · have hEq := pHyperplane_subset_iff.mp hAB
      norm_num at hEq
    · rfl
  have hCount := algebraicComponentCount_eq_finset_card
    reducedCandidate004 C hClosed hIrreducible hCover hIrredundant
  have hD₀ : D ≠ H₀ := by
    intro hEq
    exact finiteComponent_not_subset_pHyperplane eta004 0 hEq.le
  have hD₁ : D ≠ H₁ := by
    intro hEq
    exact finiteComponent_not_subset_pHyperplane eta004
      (-1 / 4 : Complex) hEq.le
  have hH : H₀ ≠ H₁ := by
    intro hEq
    have := pHyperplane_eq_iff.mp hEq
    norm_num at this
  simpa [C, hD₀, hD₁, hH] using hCount

theorem reducedCandidate006_componentCount
    (hFiniteIrreducible : FiniteComponentIrreducible006)
    (hNoVertical : NoVerticalHyperplaneInFiniteComponent eta006) :
    algebraicComponentCount reducedCandidate006 = 2 := by
  classical
  let D := finiteComponent eta006
  let H₀ := pHyperplane 0
  let C : Finset (Set C3) := {D, H₀}
  have hClosed : ∀ A ∈ C, IsClosed (zariskiLift A) := by
    intro A hA
    simp only [C, Finset.mem_insert, Finset.mem_singleton] at hA
    rcases hA with rfl | rfl
    · exact finiteComponent_isClosed eta006
    · exact pHyperplane_isClosed 0
  have hIrreducible : ∀ A ∈ C, IsIrreducible (zariskiLift A) := by
    intro A hA
    simp only [C, Finset.mem_insert, Finset.mem_singleton] at hA
    rcases hA with rfl | rfl
    · exact hFiniteIrreducible
    · exact pHyperplane_isIrreducible 0
  have hCover : ⋃₀ (C : Set (Set C3)) = reducedCandidate006 := by
    ext x
    simp [C, D, H₀, reducedCandidate006]
  have hIrredundant :
      ∀ A ∈ C, ∀ B ∈ C, A ⊆ B -> A = B := by
    intro A hA B hB hAB
    simp only [C, Finset.mem_insert, Finset.mem_singleton] at hA hB
    rcases hA with rfl | rfl <;> rcases hB with rfl | rfl
    · rfl
    · exact False.elim
        (finiteComponent_not_subset_pHyperplane eta006 0 hAB)
    · exact False.elim (hNoVertical 0 hAB)
    · rfl
  have hCount := algebraicComponentCount_eq_finset_card
    reducedCandidate006 C hClosed hIrreducible hCover hIrredundant
  have hD₀ : D ≠ H₀ := by
    intro hEq
    exact finiteComponent_not_subset_pHyperplane eta006 0 hEq.le
  simpa [C, hD₀] using hCount

/-- Actual component count for CEX-004, conditional only on the three named
pair-specific geometric obligations. -/
theorem cex004_actual_componentCount
    (hEquality : ReducedNonpropernessEquality004)
    (hFiniteIrreducible : FiniteComponentIrreducible004)
    (hNoVertical : NoVerticalHyperplaneInFiniteComponent eta004) :
    algebraicComponentCount (NonpropernessSet F004) = 3 := by
  rw [hEquality]
  exact reducedCandidate004_componentCount hFiniteIrreducible hNoVertical

/-- Actual component count for CEX-006, conditional only on the three named
pair-specific geometric obligations. -/
theorem cex006_actual_componentCount
    (hEquality : ReducedNonpropernessEquality006)
    (hFiniteIrreducible : FiniteComponentIrreducible006)
    (hNoVertical : NoVerticalHyperplaneInFiniteComponent eta006) :
    algebraicComponentCount (NonpropernessSet F006) = 2 := by
  rw [hEquality]
  exact reducedCandidate006_componentCount hFiniteIrreducible hNoVertical

end

end DegreeSixKeller
