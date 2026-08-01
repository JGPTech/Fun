import DegreeSixKeller.AsymptoticExclusion
import DegreeSixKeller.FiberLoss
import DegreeSixKeller.GeneralFiniteGeometry
import DegreeSixKeller.LeadingFiberBounds

/-!
# General reduced nonproperness equality

For every nonzero deformation polynomial, the Euclidean nonproperness set of
the degree-six Keller map is exactly the union of the finite multiple-root
component and the vanishing locus of the degree-six leading coefficient.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

/-- The target locus on which the degree-six leading coefficient vanishes,
written without the harmless scalar factor appearing in `aCoeff`. -/
def zeroLocusOfP6MulH (h : Complex[X]) : Set C3 :=
  {b : C3 | b 0 ^ 6 * h.eval (b 0) = 0}

/-- The family-wide reduced candidate for the nonproperness set. -/
def generalReducedCandidate (h : Complex[X]) : Set C3 :=
  finiteComponent h ∪ zeroLocusOfP6MulH h

/-- The explicit leading locus is exactly the zero locus of `aCoeff`. -/
theorem mem_zeroLocusOfP6MulH_iff_aCoeff_eq_zero
    (h : Complex[X]) (b : C3) :
    b ∈ zeroLocusOfP6MulH h ↔ aCoeff h (b 0) = 0 := by
  rw [zeroLocusOfP6MulH, Set.mem_setOf_eq, aCoeff_eq_zero_iff]
  simp only [mul_eq_zero, pow_eq_zero_iff (by norm_num : 6 ≠ 0)]

theorem zeroLocusOfP6MulH_eq_aCoeffZero
    (h : Complex[X]) :
    zeroLocusOfP6MulH h = {b : C3 | aCoeff h (b 0) = 0} := by
  ext b
  exact mem_zeroLocusOfP6MulH_iff_aCoeff_eq_zero h b

/-- Every asymptotic value lies in the reduced candidate. -/
theorem Fh_nonproperness_subset_generalReducedCandidate
    (h : Complex[X]) :
    NonpropernessSet (Fh h) ⊆ generalReducedCandidate h := by
  rw [generalReducedCandidate, zeroLocusOfP6MulH_eq_aCoeffZero]
  exact nonproperness_subset_finiteComponent_union_leading h

/-- For a nonzero deformation, both pieces of the reduced candidate consist
of genuine nonproper values. -/
theorem generalReducedCandidate_subset_Fh_nonproperness
    (h : Complex[X]) (hh : h ≠ 0) :
    generalReducedCandidate h ⊆ NonpropernessSet (Fh h) := by
  intro b hb
  rcases hb with hbFinite | hbLeading
  · by_cases ha : aCoeff h (b 0) = 0
    · apply mem_nonproperness_of_Fh_fiber_ncard_lt_six h hh
      have hLeading : b 0 ^ 6 * h.eval (b 0) = 0 :=
        (mem_zeroLocusOfP6MulH_iff_aCoeff_eq_zero h b).2 ha
      have hBound := leading_fiber_ncard_le_three h b hLeading
      omega
    · exact criticalImage_subset_nonproperness h
        (finiteComponent_mem_criticalImage_of_aCoeff_ne_zero h hbFinite ha)
  · apply mem_nonproperness_of_Fh_fiber_ncard_lt_six h hh
    have hBound := leading_fiber_ncard_le_three h b hbLeading
    omega

/-- Exact family-wide reduced nonproperness formula. -/
theorem Fh_reducedNonpropernessEquality
    (h : Complex[X]) (hh : h ≠ 0) :
    NonpropernessSet (Fh h) = generalReducedCandidate h :=
  Set.Subset.antisymm
    (Fh_nonproperness_subset_generalReducedCandidate h)
    (generalReducedCandidate_subset_Fh_nonproperness h hh)

end

end DegreeSixKeller
