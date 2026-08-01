import DegreeSixKeller.InverseChart
import Mathlib.Data.Set.Card
import Mathlib.Tactic.ComputeDegree

/-!
# Leading-locus fiber bounds

This module proves exact cardinal bounds for fibers on the vanishing locus of
the degree-six leading coefficient, using the explicit inverse chart and
elementary polynomial root bounds.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

def fiber (h : Complex[X]) (b : C3) : Set C3 :=
  {u | Fh h u = b}

def finiteChartFiber (h : Complex[X]) (b : C3) : Set C3 :=
  {u | Fh h u = b ∧ baseA (u 0) (u 1) ≠ 0}

def singularChartFiber (h : Complex[X]) (b : C3) : Set C3 :=
  {u | Fh h u = b ∧ baseA (u 0) (u 1) = 0}

theorem fiber_eq_chart_union_singular (h : Complex[X]) (b : C3) :
    fiber h b = finiteChartFiber h b ∪ singularChartFiber h b := by
  ext u
  simp only [fiber, finiteChartFiber, singularChartFiber, Set.mem_setOf_eq,
    Set.mem_union]
  by_cases hA : baseA (u 0) (u 1) = 0 <;> aesop

theorem fiber_coordinate_zero {h : Complex[X]} {b u : C3}
    (hu : u ∈ fiber h b) : pCoord (u 0) (u 1) (u 2) = b 0 := by
  exact congrFun hu 0

theorem fiber_coordinate_one {h : Complex[X]} {b u : C3}
    (hu : u ∈ fiber h b) : qCoord h (u 0) (u 1) (u 2) = b 1 := by
  exact congrFun hu 1

theorem fiber_coordinate_two {h : Complex[X]} {b u : C3}
    (hu : u ∈ fiber h b) : rCoord h (u 0) (u 1) (u 2) = b 2 := by
  exact congrFun hu 2

theorem source_eq_reconstruct_of_fiber {h : Complex[X]} {b u : C3}
    (hu : u ∈ fiber h b) (hA : baseA (u 0) (u 1) ≠ 0) :
    u = reconstruct h (b 0) (b 1) (chartS (u 0) (u 1)) := by
  have hsource := source_eq_reconstruct_of_chart h (u 0) (u 1) (u 2) hA
  rw [fiber_coordinate_zero hu, fiber_coordinate_one hu] at hsource
  calc
    u = ![u 0, u 1, u 2] := by
      funext i
      fin_cases i <;> simp
    _ = reconstruct h (b 0) (b 1) (chartS (u 0) (u 1)) := hsource

theorem omega_ne_zero (h : Complex[X]) (p q r : Complex) :
    omega h p q r ≠ 0 := by
  intro homega
  have hcoeff := congrArg (fun g : Complex[X] => g.coeff 1) homega
  simp only [omega, coeff_sub, coeff_add, coeff_C_mul_X_pow,
    coeff_C_mul_X, coeff_C] at hcoeff
  norm_num at hcoeff

theorem finiteChartFiber_chartS_mem_rootSet
    {h : Complex[X]} {b u : C3} (hu : u ∈ finiteChartFiber h b) :
    chartS (u 0) (u 1) ∈ (omega h (b 0) (b 1) (b 2)).rootSet Complex := by
  have homega := omega_of_source h (u 0) (u 1) (u 2) hu.2
  have huFiber : u ∈ fiber h b := hu.1
  rw [fiber_coordinate_zero huFiber, fiber_coordinate_one huFiber,
    fiber_coordinate_two huFiber] at homega
  rw [Polynomial.mem_rootSet_of_ne (omega_ne_zero h (b 0) (b 1) (b 2))]
  simpa [Polynomial.aeval_def] using homega

theorem finiteChartFiber_chartS_injOn (h : Complex[X]) (b : C3) :
    Set.InjOn (fun u : C3 => chartS (u 0) (u 1))
      (finiteChartFiber h b) := by
  intro u hu v hv hs
  have huRec := source_eq_reconstruct_of_fiber hu.1 hu.2
  have hvRec := source_eq_reconstruct_of_fiber hv.1 hv.2
  change chartS (u 0) (u 1) = chartS (v 0) (v 1) at hs
  rw [hs] at huRec
  exact huRec.trans hvRec.symm

theorem finiteChartFiber_finite (h : Complex[X]) (b : C3) :
    (finiteChartFiber h b).Finite := by
  exact Set.Finite.of_injOn
    (fun _ hu => finiteChartFiber_chartS_mem_rootSet hu)
    (finiteChartFiber_chartS_injOn h b)
    (Polynomial.rootSet_finite _ _)

theorem finiteChartFiber_ncard_le_natDegree (h : Complex[X]) (b : C3) :
    (finiteChartFiber h b).ncard ≤
      (omega h (b 0) (b 1) (b 2)).natDegree := by
  calc
    (finiteChartFiber h b).ncard ≤
        ((omega h (b 0) (b 1) (b 2)).rootSet Complex).ncard :=
      Set.ncard_le_ncard_of_injOn
        (fun u : C3 => chartS (u 0) (u 1))
        (fun _ hu => finiteChartFiber_chartS_mem_rootSet hu)
        (finiteChartFiber_chartS_injOn h b)
        (Polynomial.rootSet_finite _ _)
    _ ≤ (omega h (b 0) (b 1) (b 2)).natDegree :=
      Polynomial.ncard_rootSet_le _ _

theorem omega_natDegree_le_three_of_aCoeff_zero
    (h : Complex[X]) (p q r : Complex) (ha : aCoeff h p = 0) :
    (omega h p q r).natDegree ≤ 3 := by
  rw [omega, ha]
  simp only [C_0, zero_mul, zero_add]
  compute_degree

theorem omega_natDegree_le_two_of_p_zero
    (h : Complex[X]) (q r : Complex) :
    (omega h 0 q r).natDegree ≤ 2 := by
  have ha : aCoeff h 0 = 0 := by simp [aCoeff]
  rw [omega, ha]
  simp only [C_0, zero_mul, mul_zero, zero_add]
  compute_degree

theorem finiteChartFiber_ncard_le_three_of_aCoeff_zero
    (h : Complex[X]) (b : C3) (ha : aCoeff h (b 0) = 0) :
    (finiteChartFiber h b).ncard ≤ 3 :=
  (finiteChartFiber_ncard_le_natDegree h b).trans
    (omega_natDegree_le_three_of_aCoeff_zero h (b 0) (b 1) (b 2) ha)

theorem finiteChartFiber_ncard_le_two_of_p_zero
    (h : Complex[X]) (b : C3) (hp : b 0 = 0) :
    (finiteChartFiber h b).ncard ≤ 2 := by
  exact (finiteChartFiber_ncard_le_natDegree h b).trans (by
    rw [hp]
    exact omega_natDegree_le_two_of_p_zero h (b 1) (b 2))

theorem first_ne_zero_of_baseA_zero {x y : Complex}
    (hA : baseA x y = 0) : x ≠ 0 := by
  intro hx
  subst x
  simp [baseA] at hA

theorem second_ne_zero_of_baseA_zero {x y : Complex}
    (hA : baseA x y = 0) : y ≠ 0 := by
  intro hy
  subst y
  simp [baseA] at hA

theorem qCoord_of_baseA_zero (h : Complex[X]) (x y z : Complex)
    (hA : baseA x y = 0) : qCoord h x y z = -2 * y := by
  unfold qCoord baseQ pCoord baseB
  rw [hA]
  simp only [zero_pow (by norm_num : (2 : Nat) ≠ 0), zero_mul,
    zero_add]
  unfold baseA at hA
  linear_combination (3 * y * (1 + 3 * x * y)) * hA

theorem rCoord_injective_last_of_baseA_zero
    (h : Complex[X]) (x y z z' : Complex) (hA : baseA x y = 0)
    (hr : rCoord h x y z = rCoord h x y z') : z = z' := by
  have hx : x ≠ 0 := first_ne_zero_of_baseA_zero hA
  apply (mul_left_cancel₀ (pow_ne_zero 3 hx) :
    x ^ 3 * z = x ^ 3 * z' → z = z')
  unfold rCoord baseR pCoord baseB at hr
  rw [hA] at hr
  simp only [zero_pow (by norm_num : (2 : Nat) ≠ 0), zero_mul,
    zero_add] at hr
  linear_combination -hr

theorem singularChartFiber_subsingleton (h : Complex[X]) (b : C3) :
    (singularChartFiber h b).Subsingleton := by
  intro u hu v hv
  have hqu := fiber_coordinate_one hu.1
  have hqv := fiber_coordinate_one hv.1
  rw [qCoord_of_baseA_zero h (u 0) (u 1) (u 2) hu.2] at hqu
  rw [qCoord_of_baseA_zero h (v 0) (v 1) (v 2) hv.2] at hqv
  have hy : u 1 = v 1 := by
    linear_combination (-1 / 2 : Complex) * hqu + (1 / 2 : Complex) * hqv
  have hAu := hu.2
  have hAv := hv.2
  rw [← hy] at hAv
  have hx : u 0 = v 0 := by
    apply (mul_right_cancel₀ (second_ne_zero_of_baseA_zero hAu) :
      u 0 * u 1 = v 0 * u 1 → u 0 = v 0)
    unfold baseA at hAu hAv
    linear_combination hAu - hAv
  have hru := fiber_coordinate_two hu.1
  have hrv := fiber_coordinate_two hv.1
  rw [← hx, ← hy] at hrv
  have hz : u 2 = v 2 :=
    rCoord_injective_last_of_baseA_zero h (u 0) (u 1) (u 2) (v 2)
      hu.2 (hru.trans hrv.symm)
  funext i
  fin_cases i
  · exact hx
  · exact hy
  · exact hz

theorem singularChartFiber_finite (h : Complex[X]) (b : C3) :
    (singularChartFiber h b).Finite :=
  (singularChartFiber_subsingleton h b).finite

theorem singularChartFiber_ncard_le_one (h : Complex[X]) (b : C3) :
    (singularChartFiber h b).ncard ≤ 1 := by
  have hs := singularChartFiber_subsingleton h b
  exact (Set.ncard_le_one (singularChartFiber_finite h b)).2 (by
    intro u hu v hv
    exact hs hu hv)

theorem fiber_eq_finiteChartFiber_of_first_ne_zero
    (h : Complex[X]) (b : C3) (hp : b 0 ≠ 0) :
    fiber h b = finiteChartFiber h b := by
  ext u
  constructor
  · intro hu
    refine ⟨hu, ?_⟩
    intro hA
    have hpCoord := fiber_coordinate_zero hu
    unfold pCoord at hpCoord
    rw [hA] at hpCoord
    exact hp (by simpa using hpCoord.symm)
  · exact fun hu => hu.1

theorem fiber_finite (h : Complex[X]) (b : C3) :
    (fiber h b).Finite := by
  rw [fiber_eq_chart_union_singular]
  exact (finiteChartFiber_finite h b).union (singularChartFiber_finite h b)

theorem fiber_finite_of_leading_zero
    (h : Complex[X]) (b : C3)
    (hLeading : b 0 ^ 6 * h.eval (b 0) = 0) :
    (fiber h b).Finite := by
  have _ : b 0 ^ 6 * h.eval (b 0) = 0 := hLeading
  exact fiber_finite h b

theorem fiber_ncard_le_three_of_nonzero_root
    (h : Complex[X]) (b : C3) (hp : b 0 ≠ 0)
    (hh : h.eval (b 0) = 0) :
    (fiber h b).ncard ≤ 3 := by
  rw [fiber_eq_finiteChartFiber_of_first_ne_zero h b hp]
  apply finiteChartFiber_ncard_le_three_of_aCoeff_zero
  simp [aCoeff, hh]

theorem fiber_ncard_le_three_of_first_zero
    (h : Complex[X]) (b : C3) (hp : b 0 = 0) :
    (fiber h b).ncard ≤ 3 := by
  rw [fiber_eq_chart_union_singular]
  exact (Set.ncard_union_le _ _).trans (by
    have hfinite := finiteChartFiber_ncard_le_two_of_p_zero h b hp
    have hsingular := singularChartFiber_ncard_le_one h b
    omega)

theorem fiber_ncard_le_three_of_leading_zero
    (h : Complex[X]) (b : C3)
    (hLeading : b 0 ^ 6 * h.eval (b 0) = 0) :
    (fiber h b).ncard ≤ 3 := by
  by_cases hp : b 0 = 0
  · exact fiber_ncard_le_three_of_first_zero h b hp
  · have hh : h.eval (b 0) = 0 := by
      exact (mul_eq_zero.mp hLeading).resolve_left (pow_ne_zero 6 hp)
    exact fiber_ncard_le_three_of_nonzero_root h b hp hh

theorem leading_fiber_finite
    (h : Complex[X]) (b : C3)
    (hLeading : b 0 ^ 6 * h.eval (b 0) = 0) :
    ({u : C3 | Fh h u = b} : Set C3).Finite := by
  exact fiber_finite_of_leading_zero h b hLeading

theorem leading_fiber_encard_le_three
    (h : Complex[X]) (b : C3)
    (hLeading : b 0 ^ 6 * h.eval (b 0) = 0) :
    ({u : C3 | Fh h u = b} : Set C3).encard ≤ ((3 : ℕ) : ℕ∞) := by
  exact Set.encard_le_coe_iff_finite_ncard_le.mpr
    ⟨leading_fiber_finite h b hLeading,
      fiber_ncard_le_three_of_leading_zero h b hLeading⟩

theorem leading_fiber_ncard_le_three
    (h : Complex[X]) (b : C3)
    (hLeading : b 0 ^ 6 * h.eval (b 0) = 0) :
    ({u : C3 | Fh h u = b} : Set C3).ncard ≤ 3 := by
  exact (Set.encard_le_coe_iff_finite_ncard_le.mp
    (leading_fiber_encard_le_three h b hLeading)).2

end

end DegreeSixKeller
