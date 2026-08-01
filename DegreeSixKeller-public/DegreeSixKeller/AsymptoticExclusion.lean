import DegreeSixKeller.ResultantCertificates
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Analysis.Polynomial.CauchyBound
import Mathlib.Topology.MetricSpace.Sequences
import Mathlib.Topology.Order.OrderClosed

/-!
# Excluding asymptotic values off the reduced candidate

The main theorem in this module is the analytic half of the reduced
nonproperness calculation.  If the leading coefficient of the inverse
polynomial stays nonzero and the target is outside the finite multiple-root
component, an escaping source sequence cannot have a convergent image.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 12000000

namespace DegreeSixKeller

open Filter Polynomial Set Topology
open scoped Topology NNReal

noncomputable section

/-- A totalized chart root.  On the eventual locus `A ≠ 0` it is the genuine
chart coordinate `x/A`; elsewhere its value is irrelevant. -/
def sourceChartRoot (u : C3) : Complex :=
  if baseA (u 0) (u 1) = 0 then 0 else chartS (u 0) (u 1)

lemma sourceChartRoot_eq
    {u : C3} (hA : baseA (u 0) (u 1) ≠ 0) :
    sourceChartRoot u = chartS (u 0) (u 1) := by
  simp [sourceChartRoot, hA]

/-- Continuity of the leading coefficient in the target first coordinate. -/
theorem tendsto_aCoeff
    (h : Complex[X]) {p : Nat -> Complex} {p0 : Complex}
    (hp : Tendsto p atTop (nhds p0)) :
    Tendsto (fun n => aCoeff h (p n)) atTop (nhds (aCoeff h p0)) := by
  have hh : Tendsto (fun n => h.eval (p n)) atTop (nhds (h.eval p0)) :=
    (h.continuous.tendsto p0).comp hp
  have hc : Tendsto (fun _ : Nat => (1 / 3 : Complex)) atTop
      (nhds (1 / 3 : Complex)) := tendsto_const_nhds
  simpa [aCoeff, mul_assoc] using hc.mul ((hp.pow 6).mul hh)

/-- A direct norm bound for a root of the degree-six inverse polynomial.
The estimate is deliberately coarse, but depends continuously on the target
coefficients and the reciprocal of the leading coefficient. -/
theorem omega_root_norm_le
    (h : Complex[X]) (p q r s : Complex)
    (ha : aCoeff h p ≠ 0)
    (hs : (omega h p q r).eval s = 0) :
    norm s ≤ max 1
      ((2 * norm p + norm q + 2 + norm r) / norm (aCoeff h p)) := by
  by_cases hsOne : norm s ≤ 1
  · exact hsOne.trans (le_max_left _ _)
  have hsOne' : 1 < norm s := lt_of_not_ge hsOne
  have hs0 : s ≠ 0 := by
    intro hs0
    subst s
    norm_num at hsOne'
  have haPos : 0 < norm (aCoeff h p) := norm_pos_iff.mpr ha
  have hsPos : 0 < norm s := norm_pos_iff.mpr hs0
  have hEq :
      aCoeff h p * s ^ 6 =
        -(2 * p * s ^ 3 - q * s ^ 2 + 2 * s - r) := by
    rw [omega_eval] at hs
    linear_combination hs
  have hLead :
      norm (aCoeff h p) * norm s ^ 6 ≤
        2 * norm p * norm s ^ 3 + norm q * norm s ^ 2 +
          2 * norm s + norm r := by
    calc
      norm (aCoeff h p) * norm s ^ 6 =
          norm (aCoeff h p * s ^ 6) := by simp [norm_pow]
      _ = norm (2 * p * s ^ 3 - q * s ^ 2 + 2 * s - r) := by
        rw [hEq, norm_neg]
      _ ≤ norm (2 * p * s ^ 3) + norm (q * s ^ 2) +
            norm (2 * s) + norm r := by
        calc
          norm (2 * p * s ^ 3 - q * s ^ 2 + 2 * s - r) ≤
              norm (2 * p * s ^ 3 - q * s ^ 2 + 2 * s) + norm r :=
            norm_sub_le _ _
          _ ≤ norm (2 * p * s ^ 3 - q * s ^ 2) + norm (2 * s) + norm r := by
            gcongr
            exact norm_add_le _ _
          _ ≤ (norm (2 * p * s ^ 3) + norm (q * s ^ 2)) +
                norm (2 * s) + norm r := by
            gcongr
            exact norm_sub_le _ _
      _ = 2 * norm p * norm s ^ 3 + norm q * norm s ^ 2 +
            2 * norm s + norm r := by
        simp [norm_pow]
  have hsOneLe : 1 ≤ norm s := le_of_lt hsOne'
  have h13 : norm s ^ 3 ≤ norm s ^ 5 := by
    exact pow_le_pow_right₀ hsOneLe (by norm_num)
  have h12 : norm s ^ 2 ≤ norm s ^ 5 := by
    exact pow_le_pow_right₀ hsOneLe (by norm_num)
  have h11 : norm s ≤ norm s ^ 5 := by
    simpa using (pow_le_pow_right₀ hsOneLe (show 1 ≤ 5 by norm_num))
  have h10 : 1 ≤ norm s ^ 5 := by
    simpa using (pow_le_pow_right₀ hsOneLe (show 0 ≤ 5 by norm_num))
  let C : Real := 2 * norm p + norm q + 2 + norm r
  have hRhs :
      2 * norm p * norm s ^ 3 + norm q * norm s ^ 2 +
          2 * norm s + norm r ≤ C * norm s ^ 5 := by
    dsimp [C]
    have hp0 : 0 ≤ norm p := norm_nonneg _
    have hq0 : 0 ≤ norm q := norm_nonneg _
    have hr0 : 0 ≤ norm r := norm_nonneg _
    nlinarith
  have hMain : norm (aCoeff h p) * norm s ≤ C := by
    have hpow : 0 < norm s ^ 5 := pow_pos hsPos 5
    have hMul :
        (norm (aCoeff h p) * norm s) * norm s ^ 5 ≤
          C * norm s ^ 5 := by
      calc
        (norm (aCoeff h p) * norm s) * norm s ^ 5 =
            norm (aCoeff h p) * norm s ^ 6 := by ring
        _ ≤ 2 * norm p * norm s ^ 3 + norm q * norm s ^ 2 +
              2 * norm s + norm r := hLead
        _ ≤ C * norm s ^ 5 := hRhs
    exact le_of_mul_le_mul_right hMul hpow
  have hDiv : norm s ≤ C / norm (aCoeff h p) := by
    exact (le_div_iff₀ haPos).2 (by simpa [mul_comm] using hMain)
  exact hDiv.trans (le_max_right _ _)

/-- A convergent target sequence with nonzero limiting leading coefficient has
all eventual inverse roots in one closed ball. -/
theorem omega_roots_eventually_bounded
    (h : Complex[X]) {v : Nat -> C3} {b : C3}
    (hv : Tendsto v atTop (nhds b))
    (ha : aCoeff h (b 0) ≠ 0)
    {s : Nat -> Complex}
    (hRoot : ∀ᶠ n in atTop,
      (omega h (v n 0) (v n 1) (v n 2)).eval (s n) = 0) :
    ∃ R : Real, ∀ᶠ n in atTop, s n ∈ Metric.closedBall 0 R := by
  have hv0 : Tendsto (fun n => v n 0) atTop (nhds (b 0)) :=
    (tendsto_pi_nhds.1 hv) 0
  have hv1 : Tendsto (fun n => v n 1) atTop (nhds (b 1)) :=
    (tendsto_pi_nhds.1 hv) 1
  have hv2 : Tendsto (fun n => v n 2) atTop (nhds (b 2)) :=
    (tendsto_pi_nhds.1 hv) 2
  obtain ⟨M0, hM0Pos, hM0⟩ :=
    (Metric.isBounded_range_of_tendsto _ hv0).exists_pos_norm_le
  obtain ⟨M1, hM1Pos, hM1⟩ :=
    (Metric.isBounded_range_of_tendsto _ hv1).exists_pos_norm_le
  obtain ⟨M2, hM2Pos, hM2⟩ :=
    (Metric.isBounded_range_of_tendsto _ hv2).exists_pos_norm_le
  have haLim := tendsto_aCoeff h hv0
  have haNorm : Tendsto (fun n => norm (aCoeff h (v n 0))) atTop
      (nhds (norm (aCoeff h (b 0)))) := haLim.norm
  have haPos : 0 < norm (aCoeff h (b 0)) := norm_pos_iff.mpr ha
  have haLower :
      ∀ᶠ n in atTop,
        norm (aCoeff h (b 0)) / 2 < norm (aCoeff h (v n 0)) := by
    obtain ⟨N, hN⟩ := Metric.tendsto_atTop.1 haNorm
      (norm (aCoeff h (b 0)) / 2) (half_pos haPos)
    filter_upwards [eventually_ge_atTop N] with n hn
    have hdist := hN n hn
    rw [Real.dist_eq, abs_lt] at hdist
    linarith
  let R : Real := max 1
    ((2 * M0 + M1 + 2 + M2) / (norm (aCoeff h (b 0)) / 2))
  refine ⟨R, ?_⟩
  filter_upwards [hRoot, haLower] with n hnRoot hnLower
  have han : aCoeff h (v n 0) ≠ 0 := by
    apply norm_pos_iff.mp
    exact (half_pos haPos).trans hnLower
  have hBound := omega_root_norm_le h (v n 0) (v n 1) (v n 2) (s n) han hnRoot
  have hCoeff :
      (2 * norm (v n 0) + norm (v n 1) + 2 + norm (v n 2)) /
          norm (aCoeff h (v n 0)) ≤
        (2 * M0 + M1 + 2 + M2) /
          (norm (aCoeff h (b 0)) / 2) := by
    apply div_le_div₀
    · positivity
    · nlinarith [hM0 (v n 0) ⟨n, rfl⟩,
        hM1 (v n 1) ⟨n, rfl⟩, hM2 (v n 2) ⟨n, rfl⟩]
    · exact half_pos haPos
    · exact le_of_lt hnLower
  have : norm (s n) ≤ R :=
    hBound.trans (max_le_max_left _ hCoeff)
  simpa [Metric.mem_closedBall, dist_zero_right] using this

/-- Reconstruction varies continuously along convergent parameter sequences
provided the limiting denominator is nonzero. -/
theorem reconstruct_tendsto
    (h : Complex[X]) {p q s : Nat -> Complex} {p0 q0 s0 : Complex}
    (hp : Tendsto p atTop (nhds p0))
    (hq : Tendsto q atTop (nhds q0))
    (hs : Tendsto s atTop (nhds s0))
    (hD : rootD h p0 q0 s0 ≠ 0) :
    Tendsto (fun n => reconstruct h (p n) (q n) (s n)) atTop
      (nhds (reconstruct h p0 q0 s0)) := by
  have hh : Tendsto (fun n => h.eval (p n)) atTop (nhds (h.eval p0)) :=
    (h.continuous.tendsto p0).comp hp
  have hthree : Tendsto (fun _ : Nat => (3 : Complex)) atTop
      (nhds (3 : Complex)) := tendsto_const_nhds
  have hfour : Tendsto (fun _ : Nat => (4 : Complex)) atTop
      (nhds (4 : Complex)) := tendsto_const_nhds
  have hone : Tendsto (fun _ : Nat => (1 : Complex)) atTop
      (nhds (1 : Complex)) := tendsto_const_nhds
  have hphi : Tendsto (fun n => phi h (p n) (s n)) atTop
      (nhds (phi h p0 s0)) := by
    have hfirst : Tendsto (fun n => 3 * p n * s n) atTop
        (nhds (3 * p0 * s0)) := (hthree.mul hp).mul hs
    have hsecond : Tendsto
        (fun n => p n ^ 6 * h.eval (p n) * s n ^ 4) atTop
        (nhds (p0 ^ 6 * h.eval p0 * s0 ^ 4)) :=
      ((hp.pow 6).mul hh).mul (hs.pow 4)
    simpa [phi, mul_assoc] using hfirst.add hsecond
  have hy : Tendsto (fun n => q n - phi h (p n) (s n)) atTop
      (nhds (q0 - phi h p0 s0)) := hq.sub hphi
  have hrootD : Tendsto (fun n => rootD h (p n) (q n) (s n)) atTop
      (nhds (rootD h p0 q0 s0)) := by
    simpa [rootD] using hone.sub (hs.mul hy)
  have hfourMinus : Tendsto
      (fun n => 4 - s n * (q n - phi h (p n) (s n))) atTop
      (nhds (4 - s0 * (q0 - phi h p0 s0))) :=
    hfour.sub (hs.mul hy)
  apply tendsto_pi_nhds.2
  intro i
  fin_cases i
  · change Tendsto
      (fun n => s n / rootD h (p n) (q n) (s n)) atTop
      (nhds (s0 / rootD h p0 q0 s0))
    exact hs.div hrootD hD
  · change Tendsto (fun n => q n - phi h (p n) (s n)) atTop
      (nhds (q0 - phi h p0 s0))
    exact hy
  · change Tendsto
      (fun n => p n * rootD h (p n) (q n) (s n) ^ 3 -
        (q n - phi h (p n) (s n)) ^ 2 *
          (4 - s n * (q n - phi h (p n) (s n))) *
            rootD h (p n) (q n) (s n)) atTop
      (nhds (p0 * rootD h p0 q0 s0 ^ 3 -
        (q0 - phi h p0 s0) ^ 2 *
          (4 - s0 * (q0 - phi h p0 s0)) * rootD h p0 q0 s0))
    exact (hp.mul (hrootD.pow 3)).sub
      (((hy.pow 2).mul hfourMinus).mul hrootD)

/-- Off the leading-coefficient locus, every asymptotic value belongs to the
finite multiple-root component. -/
theorem nonproperness_mem_finiteComponent_of_aCoeff_ne_zero
    (h : Complex[X]) {b : C3}
    (hb : b ∈ NonpropernessSet (Fh h))
    (ha : aCoeff h (b 0) ≠ 0) :
    b ∈ finiteComponent h := by
  rcases hb with ⟨u, huEsc, huLim⟩
  let v : Nat -> C3 := fun n => Fh h (u n)
  let s : Nat -> Complex := fun n => sourceChartRoot (u n)
  have hv : Tendsto v atTop (nhds b) := huLim
  have hv0 : Tendsto (fun n => v n 0) atTop (nhds (b 0)) :=
    (tendsto_pi_nhds.1 hv) 0
  have hb0 : b 0 ≠ 0 := by
    intro hb0
    apply ha
    simp [hb0, aCoeff]
  have hpEventually : ∀ᶠ n in atTop, v n 0 ≠ 0 :=
    hv0.eventually_ne hb0
  have hAEventually :
      ∀ᶠ n in atTop, baseA (u n 0) (u n 1) ≠ 0 := by
    filter_upwards [hpEventually] with n hp
    intro hA
    apply hp
    simp [v, Fh, pCoord, hA]
  have hRoot : ∀ᶠ n in atTop,
      (omega h (v n 0) (v n 1) (v n 2)).eval (s n) = 0 := by
    filter_upwards [hAEventually] with n hA
    simpa [v, Fh, s, sourceChartRoot_eq hA] using
      omega_of_source h (u n 0) (u n 1) (u n 2) hA
  obtain ⟨R, hR⟩ := omega_roots_eventually_bounded h hv ha hRoot
  have hFreq : ∃ᶠ n in atTop, s n ∈ Metric.closedBall 0 R := hR.frequently
  obtain ⟨s0, hs0Closure, φ, hφ, hs0⟩ :=
    tendsto_subseq_of_frequently_bounded
      (s := Metric.closedBall (0 : Complex) R)
      Metric.isBounded_closedBall hFreq
  have hφTop : Tendsto φ atTop atTop := hφ.tendsto_atTop
  have hvφ : Tendsto (v ∘ φ) atTop (nhds b) := hv.comp hφTop
  have huφEsc : Escapes (u ∘ φ) := huEsc.comp hφTop
  have hsφ : Tendsto (s ∘ φ) atTop (nhds s0) := hs0
  have hRootφ : ∀ᶠ n in atTop,
      (omega h ((v ∘ φ) n 0) ((v ∘ φ) n 1) ((v ∘ φ) n 2)).eval
        ((s ∘ φ) n) = 0 := hφTop.eventually hRoot
  have hOmegaLimit :
      Tendsto (fun n =>
        (omega h ((v ∘ φ) n 0) ((v ∘ φ) n 1) ((v ∘ φ) n 2)).eval
          ((s ∘ φ) n)) atTop
        (nhds ((omega h (b 0) (b 1) (b 2)).eval s0)) := by
    have hp := (tendsto_pi_nhds.1 hvφ) 0
    have hq := (tendsto_pi_nhds.1 hvφ) 1
    have hr := (tendsto_pi_nhds.1 hvφ) 2
    have haT := tendsto_aCoeff h hp
    have htwo : Tendsto (fun _ : Nat => (2 : Complex)) atTop
        (nhds (2 : Complex)) := tendsto_const_nhds
    have hLeadTerm : Tendsto (fun n =>
        aCoeff h ((v ∘ φ) n 0) * ((s ∘ φ) n) ^ 6) atTop
        (nhds (aCoeff h (b 0) * s0 ^ 6)) := haT.mul (hsφ.pow 6)
    have hCubicTerm : Tendsto (fun n =>
        2 * ((v ∘ φ) n 0) * ((s ∘ φ) n) ^ 3) atTop
        (nhds (2 * (b 0) * s0 ^ 3)) := (htwo.mul hp).mul (hsφ.pow 3)
    have hQuadraticTerm : Tendsto (fun n =>
        ((v ∘ φ) n 1) * ((s ∘ φ) n) ^ 2) atTop
        (nhds ((b 1) * s0 ^ 2)) := hq.mul (hsφ.pow 2)
    have hLinearTerm : Tendsto (fun n => 2 * ((s ∘ φ) n)) atTop
        (nhds (2 * s0)) := htwo.mul hsφ
    simpa [omega_eval, mul_assoc] using
      (((hLeadTerm.add hCubicTerm).sub hQuadraticTerm).add hLinearTerm).sub hr
  have hOmega0 : (omega h (b 0) (b 1) (b 2)).eval s0 = 0 := by
    have hZero : Tendsto (fun _ : Nat => (0 : Complex)) atTop
        (nhds (0 : Complex)) := tendsto_const_nhds
    have hEvalZero : Tendsto (fun n =>
        (omega h ((v ∘ φ) n 0) ((v ∘ φ) n 1) ((v ∘ φ) n 2)).eval
          ((s ∘ φ) n)) atTop (nhds (0 : Complex)) :=
      Tendsto.congr' (hRootφ.mono fun _ hn => hn.symm) hZero
    exact tendsto_nhds_unique hOmegaLimit hEvalZero
  have hDerivative0 :
      (derivative (omega h (b 0) (b 1) (b 2))).eval s0 = 0 := by
    by_contra hne
    have hD0 : rootD h (b 0) (b 1) s0 ≠ 0 := by
      rw [rootD_eq_derivative_div_two h (b 0) (b 1) (b 2) s0 hOmega0]
      exact div_ne_zero hne (by norm_num)
    have hAφ :
        ∀ᶠ n in atTop,
          baseA ((u ∘ φ) n 0) ((u ∘ φ) n 1) ≠ 0 :=
      hφTop.eventually hAEventually
    have hSourceEq :
        ∀ᶠ n in atTop,
          (u ∘ φ) n = reconstruct h
            ((v ∘ φ) n 0) ((v ∘ φ) n 1) ((s ∘ φ) n) := by
      filter_upwards [hAφ] with n hA
      calc
        (u ∘ φ) n =
            ![((u ∘ φ) n 0), ((u ∘ φ) n 1), ((u ∘ φ) n 2)] := by
          funext i
          fin_cases i <;> rfl
        _ = reconstruct h
            (pCoord ((u ∘ φ) n 0) ((u ∘ φ) n 1) ((u ∘ φ) n 2))
            (qCoord h ((u ∘ φ) n 0) ((u ∘ φ) n 1) ((u ∘ φ) n 2))
            (chartS ((u ∘ φ) n 0) ((u ∘ φ) n 1)) :=
          source_eq_reconstruct_of_chart h
            ((u ∘ φ) n 0) ((u ∘ φ) n 1) ((u ∘ φ) n 2) hA
        _ = reconstruct h
            (pCoord ((u ∘ φ) n 0) ((u ∘ φ) n 1) ((u ∘ φ) n 2))
            (qCoord h ((u ∘ φ) n 0) ((u ∘ φ) n 1) ((u ∘ φ) n 2))
            (sourceChartRoot ((u ∘ φ) n)) := by
          rw [sourceChartRoot_eq hA]
        _ = reconstruct h
            ((v ∘ φ) n 0) ((v ∘ φ) n 1) ((s ∘ φ) n) := by
          simp [v, s, Fh, Function.comp_apply]
    have hRecLim := reconstruct_tendsto h
      ((tendsto_pi_nhds.1 hvφ) 0)
      ((tendsto_pi_nhds.1 hvφ) 1) hsφ hD0
    have hSourceEq' :
        (fun n => reconstruct h
          ((v ∘ φ) n 0) ((v ∘ φ) n 1) ((s ∘ φ) n)) =ᶠ[atTop]
          (u ∘ φ) :=
      hSourceEq.mono fun _ hn => hn.symm
    have hSourceLim : Tendsto (u ∘ φ) atTop
        (nhds (reconstruct h (b 0) (b 1) s0)) :=
      Tendsto.congr' hSourceEq' hRecLim
    have hNormFinite := hSourceLim.norm
    have hNormInf : Tendsto (fun n => norm ((u ∘ φ) n)) atTop atTop := huφEsc
    exact hNormFinite.not_tendsto (disjoint_nhds_atTop _) hNormInf
  have hs0ne : s0 ≠ 0 :=
    common_root_ne_zero h (b 0) (b 1) (b 2) s0 hDerivative0
  have hbCritical : b ∈ criticalImage h := by
    refine ⟨b 0, s0, hs0ne, ?_⟩
    have hEq := eq_criticalTarget_of_common_root h
      (b 0) (b 1) (b 2) s0 hs0ne hOmega0 hDerivative0
    calc
      b = ![b 0, b 1, b 2] := by
        funext i
        fin_cases i <;> rfl
      _ = criticalTarget h (b 0) s0 := hEq
  exact subset_closure hbCritical

/-- Universal forward inclusion: the only possible asymptotic values are the
finite component and the leading-coefficient locus. -/
theorem nonproperness_subset_finiteComponent_union_leading
    (h : Complex[X]) :
    NonpropernessSet (Fh h) ⊆
      finiteComponent h ∪ {b : C3 | aCoeff h (b 0) = 0} := by
  intro b hb
  by_cases ha : aCoeff h (b 0) = 0
  · exact Or.inr ha
  · exact Or.inl (nonproperness_mem_finiteComponent_of_aCoeff_ne_zero h hb ha)

end

end DegreeSixKeller
