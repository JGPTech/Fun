import DegreeSixKeller.AsymptoticExclusion
import DegreeSixKeller.CEX004_CEX006_Inequivalent

/-!
# Exact reduced nonproperness sets for CEX-004 and CEX-006
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Set

noncomputable section

/-- Forward inclusion for CEX-004. -/
theorem cex004_nonproperness_subset_candidate :
    NonpropernessSet F004 ⊆ reducedCandidate004 := by
  intro b hb
  have h := nonproperness_subset_finiteComponent_union_leading eta004 hb
  rcases h with hbFinite | hbLead
  · exact Or.inl (Or.inl hbFinite)
  · change aCoeff eta004 (b 0) = 0 at hbLead
    rw [aCoeff_eta004_zero_iff] at hbLead
    rcases hbLead with hp0 | hpRoot
    · exact Or.inl (Or.inr hp0)
    · exact Or.inr hpRoot

/-- Forward inclusion for CEX-006. -/
theorem cex006_nonproperness_subset_candidate :
    NonpropernessSet F006 ⊆ reducedCandidate006 := by
  intro b hb
  have h := nonproperness_subset_finiteComponent_union_leading eta006 hb
  rcases h with hbFinite | hbLead
  · exact Or.inl hbFinite
  · change aCoeff eta006 (b 0) = 0 at hbLead
    rw [aCoeff_eta006_zero_iff] at hbLead
    exact Or.inr hbLead

/-- Reverse inclusion for CEX-004. -/
theorem cex004_candidate_subset_nonproperness :
    reducedCandidate004 ⊆ NonpropernessSet F004 := by
  intro b hb
  rcases hb with (hbFinite | hbZero) | hbRoot
  · exact finiteComponent004_subset_nonproperness hbFinite
  · exact pHyperplane_zero_subset_nonproperness004 hbZero
  · have hbAlpha : b ∈ pHyperplane alpha004 := by
      simpa [alpha004] using hbRoot
    exact pHyperplane_root_subset_nonproperness004 hbAlpha

/-- Reverse inclusion for CEX-006. -/
theorem cex006_candidate_subset_nonproperness :
    reducedCandidate006 ⊆ NonpropernessSet F006 := by
  intro b hb
  rcases hb with hbFinite | hbZero
  · exact finiteComponent006_subset_nonproperness hbFinite
  · exact pHyperplane_zero_subset_nonproperness006 hbZero

/-- Exact reduced nonproperness set for CEX-004. -/
theorem cex004_reducedNonpropernessEquality :
    ReducedNonpropernessEquality004 :=
  cex004_reducedNonpropernessEquality_of_inclusions
    cex004_nonproperness_subset_candidate
    cex004_candidate_subset_nonproperness

/-- Exact reduced nonproperness set for CEX-006. -/
theorem cex006_reducedNonpropernessEquality :
    ReducedNonpropernessEquality006 :=
  cex006_reducedNonpropernessEquality_of_inclusions
    cex006_nonproperness_subset_candidate
    cex006_candidate_subset_nonproperness

/-- The actual CEX-004 nonproperness set has three irreducible components. -/
theorem cex004_actual_componentCount_unconditional :
    algebraicComponentCount (NonpropernessSet F004) = 3 :=
  cex004_actual_componentCount
    cex004_reducedNonpropernessEquality
    cex004_finiteComponentIrreducible
    cex004_noVerticalHyperplane

/-- The actual CEX-006 nonproperness set has two irreducible components. -/
theorem cex006_actual_componentCount_unconditional :
    algebraicComponentCount (NonpropernessSet F006) = 2 :=
  cex006_actual_componentCount
    cex006_reducedNonpropernessEquality
    cex006_finiteComponentIrreducible
    cex006_noVerticalHyperplane

/-- Unconditional inequivalence in the current algebraic-escape interface. -/
theorem cex004_cex006_not_algebraicEscapeEquivalent :
    ¬ AlgebraicEscapeLeftRightEquivalent F004 F006 :=
  cex004_cex006_inequivalent_of_pair_geometry
    cex004_reducedNonpropernessEquality
    cex006_reducedNonpropernessEquality
    cex004_finiteComponentIrreducible
    cex006_finiteComponentIrreducible
    cex004_noVerticalHyperplane
    cex006_noVerticalHyperplane

end

end DegreeSixKeller
