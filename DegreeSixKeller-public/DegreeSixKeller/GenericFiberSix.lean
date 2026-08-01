import DegreeSixKeller.LeadingFiberBounds
import Mathlib.Analysis.Complex.Polynomial.Basic
import Mathlib.FieldTheory.Separable

/-!
# Exact generic fiber cardinality

The inverse chart identifies a fiber over a nonvertical target with the
distinct roots of its degree-six inverse polynomial.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

/-- The inverse roots associated to a target point. -/
def inverseRoots (h : Complex[X]) (b : C3) : Set Complex :=
  (omega h (b 0) (b 1) (b 2)).rootSet Complex

/-- The inverse-chart coordinate on source space. -/
def sourceChart (u : C3) : Complex :=
  chartS (u 0) (u 1)

theorem omega_natDegree_eq_six
    (h : Complex[X]) (p q r : Complex) (ha : aCoeff h p ≠ 0) :
    (omega h p q r).natDegree = 6 := by
  apply natDegree_eq_of_le_of_coeff_ne_zero
  · unfold omega
    compute_degree
  · simpa only [omega_coeff_six] using ha

theorem fiber_baseA_ne_zero {h : Complex[X]} {b u : C3}
    (hp : b 0 ≠ 0) (hu : u ∈ fiber h b) :
    baseA (u 0) (u 1) ≠ 0 := by
  intro hA
  have hfirst := fiber_coordinate_zero hu
  unfold pCoord at hfirst
  rw [hA] at hfirst
  exact hp (by simpa using hfirst.symm)

theorem sourceChart_mem_inverseRoots
    {h : Complex[X]} {b u : C3} (hp : b 0 ≠ 0) (hu : u ∈ fiber h b) :
    sourceChart u ∈ inverseRoots h b := by
  have hroot := omega_of_source h (u 0) (u 1) (u 2)
    (fiber_baseA_ne_zero hp hu)
  rw [fiber_coordinate_zero hu, fiber_coordinate_one hu,
    fiber_coordinate_two hu] at hroot
  change sourceChart u ∈ (omega h (b 0) (b 1) (b 2)).rootSet Complex
  rw [Polynomial.mem_rootSet_of_ne]
  · simpa [sourceChart, Polynomial.aeval_def] using hroot
  · exact omega_ne_zero h (b 0) (b 1) (b 2)

theorem source_eq_reconstruct
    {h : Complex[X]} {b u : C3} (hp : b 0 ≠ 0) (hu : u ∈ fiber h b) :
    u = reconstruct h (b 0) (b 1) (sourceChart u) := by
  simpa [sourceChart] using
    source_eq_reconstruct_of_fiber hu (fiber_baseA_ne_zero hp hu)

theorem sourceChart_injOn (h : Complex[X]) (b : C3) (hp : b 0 ≠ 0) :
    Set.InjOn sourceChart (fiber h b) := by
  intro u hu v hv huv
  have huRec := source_eq_reconstruct hp hu
  have hvRec := source_eq_reconstruct hp hv
  rw [huv] at huRec
  exact huRec.trans hvRec.symm

theorem derivative_ne_zero_at_root_of_separable
    {f : Complex[X]} (hsep : f.Separable) {s : Complex}
    (hs : f.eval s = 0) : f.derivative.eval s ≠ 0 := by
  intro hderivative
  rcases hsep with ⟨a, b, hab⟩
  have heval := congrArg (fun g : Complex[X] => g.eval s) hab
  simp [hs, hderivative] at heval

theorem target_eq_coordinates (b : C3) :
    (![b 0, b 1, b 2] : C3) = b := by
  funext i
  fin_cases i <;> simp

theorem sourceChart_surjOn
    (h : Complex[X]) (b : C3)
    (hsep : (omega h (b 0) (b 1) (b 2)).Separable) :
    Set.SurjOn sourceChart (fiber h b) (inverseRoots h b) := by
  intro s hs
  have hf0 : omega h (b 0) (b 1) (b 2) ≠ 0 := hsep.ne_zero
  have hroot : (omega h (b 0) (b 1) (b 2)).eval s = 0 := by
    have haeval : Polynomial.aeval s (omega h (b 0) (b 1) (b 2)) = 0 := by
      rw [← Polynomial.mem_rootSet_of_ne hf0]
      simpa [inverseRoots] using hs
    simpa [Polynomial.aeval_def] using haeval
  have hderivative := derivative_ne_zero_at_root_of_separable hsep hroot
  let u := reconstruct h (b 0) (b 1) s
  have hmaps : Fh h u = b := by
    exact (reconstruct_maps_to h (b 0) (b 1) (b 2) s hroot hderivative).trans
      (target_eq_coordinates b)
  have hD : rootD h (b 0) (b 1) s ≠ 0 := by
    rw [rootD_eq_derivative_div_two h (b 0) (b 1) (b 2) s hroot]
    exact div_ne_zero hderivative (by norm_num)
  refine ⟨u, hmaps, ?_⟩
  exact chartS_reconstruct h (b 0) (b 1) s hD

/-- The chart coordinate bijects the fiber with the inverse root set. -/
theorem sourceChart_bijOn
    (h : Complex[X]) (b : C3) (hp : b 0 ≠ 0)
    (hsep : (omega h (b 0) (b 1) (b 2)).Separable) :
    Set.BijOn sourceChart (fiber h b) (inverseRoots h b) :=
  ⟨fun _ hu => sourceChart_mem_inverseRoots hp hu,
    sourceChart_injOn h b hp,
    sourceChart_surjOn h b hsep⟩

theorem inverseRoots_ncard_eq_six
    (h : Complex[X]) (b : C3) (ha : aCoeff h (b 0) ≠ 0)
    (hsep : (omega h (b 0) (b 1) (b 2)).Separable) :
    (inverseRoots h b).ncard = 6 := by
  unfold inverseRoots
  calc
    ((omega h (b 0) (b 1) (b 2)).rootSet Complex).ncard =
        Fintype.card ((omega h (b 0) (b 1) (b 2)).rootSet Complex) :=
      (Set.fintypeCard_eq_ncard
        ((omega h (b 0) (b 1) (b 2)).rootSet Complex)).symm
    _ = (omega h (b 0) (b 1) (b 2)).natDegree := by
      exact Polynomial.card_rootSet_eq_natDegree hsep
        (IsAlgClosed.splits_domain (f := algebraMap Complex Complex)
          (omega h (b 0) (b 1) (b 2)))
    _ = 6 := omega_natDegree_eq_six h (b 0) (b 1) (b 2) ha

/-- A nonvertical target with separable degree-six inverse polynomial has
exactly six source points. -/
theorem fiber_ncard_eq_six
    (h : Complex[X]) (b : C3) (hp : b 0 ≠ 0)
    (ha : aCoeff h (b 0) ≠ 0)
    (hsep : (omega h (b 0) (b 1) (b 2)).Separable) :
    ({u : C3 | Fh h u = b} : Set C3).ncard = 6 := by
  change (fiber h b).ncard = 6
  rw [(sourceChart_bijOn h b hp hsep).ncard_eq]
  exact inverseRoots_ncard_eq_six h b ha hsep

end

end DegreeSixKeller
