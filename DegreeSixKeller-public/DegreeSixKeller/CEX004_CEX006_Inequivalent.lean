import DegreeSixKeller.EliminationCertificates
import DegreeSixKeller.Irreducibility
import DegreeSixKeller.PairSpecificComponents

/-!
# Final component-count contradiction for CEX-004 and CEX-006

This theorem is deliberately conditional on the two geometric component-count
certificates.  The hypotheses are visible and there are no placeholders or
undeclared axioms.  Discharging those hypotheses is the remaining geometric
formalization frontier.
-/

set_option autoImplicit false

namespace DegreeSixKeller

theorem cex004_cex006_inequivalent_of_component_counts
    (componentCount : Set C3 -> Nat)
    (hInv : ComponentCountInvariant componentCount)
    (h004 : componentCount (NonpropernessSet F004) = 3)
    (h006 : componentCount (NonpropernessSet F006) = 2) :
    ¬ EscapeLeftRightEquivalent F004 F006 := by
  intro hEq
  have hcount :=
    componentCount_eq_of_escapeLeftRightEquivalent componentCount hInv hEq
  rw [h006, h004] at hcount
  omega

/-- Unified pair-specific theorem using the genuine affine-Zariski component
count.  Its six geometric hypotheses are precisely the two reduced-set
equalities, the two finite-component irreducibility statements, and the two
nonverticality statements. -/
theorem cex004_cex006_inequivalent_of_pair_geometry
    (hEquality004 : ReducedNonpropernessEquality004)
    (hEquality006 : ReducedNonpropernessEquality006)
    (hIrreducible004 : FiniteComponentIrreducible004)
    (hIrreducible006 : FiniteComponentIrreducible006)
    (hNoVertical004 : NoVerticalHyperplaneInFiniteComponent eta004)
    (hNoVertical006 : NoVerticalHyperplaneInFiniteComponent eta006) :
    ¬ AlgebraicEscapeLeftRightEquivalent F004 F006 := by
  intro hEquivalent
  have hCountEq :=
    algebraicComponentCount_eq_of_algebraicEscapeLeftRightEquivalent
      hEquivalent
  have hCount004 := cex004_actual_componentCount hEquality004
    hIrreducible004 hNoVertical004
  have hCount006 := cex006_actual_componentCount hEquality006
    hIrreducible006 hNoVertical006
  rw [hCount006, hCount004] at hCountEq
  omega

end DegreeSixKeller
