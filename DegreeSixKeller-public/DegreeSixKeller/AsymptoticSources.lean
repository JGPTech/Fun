import DegreeSixKeller.InverseChart
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Analysis.Normed.Group.Constructions
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Topology.MetricSpace.Sequences
import Mathlib.Topology.Sequences
import Mathlib.FieldTheory.IsAlgClosed.Basic
import Mathlib.Tactic.FunProp

/-!
# Explicit asymptotic source families

This module proves closedness of the sequence-defined nonproperness set and
constructs the escaping source families needed for the finite critical image
and the pair-specific vertical components.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Filter Polynomial Set Topology
open scoped Topology

noncomputable section

/-- The standard nonzero parameter tending to zero. -/
def escapeEps (n : Nat) : Complex :=
  (((n + 1 : Nat) : Complex))⁻¹

lemma escapeEps_ne_zero (n : Nat) : escapeEps n ≠ 0 := by
  apply inv_ne_zero
  exact_mod_cast Nat.succ_ne_zero n

@[simp]
lemma norm_escapeEps (n : Nat) :
    norm (escapeEps n) =
      (1 : Real) / (((n + 1 : Nat) : Real)) := by
  unfold escapeEps
  rw [norm_inv, norm_natCast_eq_mul_norm_one]
  norm_num [div_eq_mul_inv]

lemma escapeEps_tendsto_zero :
    Tendsto escapeEps atTop (nhds 0) := by
  rw [tendsto_zero_iff_norm_tendsto_zero]
  have hReal :
      Tendsto (fun n : Nat =>
        (1 : Real) / (((n + 1 : Nat) : Real))) atTop (nhds 0) := by
    simpa only [Nat.cast_add, Nat.cast_one] using
      tendsto_one_div_add_atTop_nhds_zero_nat
  simpa only [norm_escapeEps] using hReal

/-- A diagonal witness used to prove closedness of `NonpropernessSet`. -/
private theorem exists_diagonal_escape_point
    (F : C3 -> C3) (bseq : Nat -> C3)
    (hb : ∀ n, bseq n ∈ NonpropernessSet F) :
    ∀ n : Nat, ∃ x : C3,
      (n : Real) < norm x ∧
      dist (F x) (bseq n) <
        (1 : Real) / (((n + 1 : Nat) : Real)) := by
  intro n
  rcases hb n with ⟨u, huEsc, huLim⟩
  have hEscEventually : ∀ᶠ k in atTop, (n : Real) < norm (u k) := by
    filter_upwards [tendsto_atTop.1 huEsc ((n : Real) + 1)] with k hk
    linarith
  have hPos :
      (0 : Real) < 1 / (((n + 1 : Nat) : Real)) := by positivity
  have hLimEventually :
      ∀ᶠ k in atTop,
        dist (F (u k)) (bseq n) <
          (1 : Real) / (((n + 1 : Nat) : Real)) := by
    rcases Metric.tendsto_atTop.1 huLim _ hPos with ⟨N, hN⟩
    exact Filter.eventually_atTop.2 ⟨N, hN⟩
  rcases Filter.eventually_atTop.1 hEscEventually with ⟨Ne, hNe⟩
  rcases Filter.eventually_atTop.1 hLimEventually with ⟨Nl, hNl⟩
  let k := max Ne Nl
  exact ⟨u k, hNe k (le_max_left _ _), hNl k (le_max_right _ _)⟩

/-- The sequence definition of nonproperness is Euclidean closed. -/
theorem nonpropernessSet_isClosed
    (F : C3 -> C3) :
    IsClosed (NonpropernessSet F) := by
  apply IsSeqClosed.isClosed
  intro bseq b hbseq hbLim
  have hChoice := exists_diagonal_escape_point F bseq hbseq
  choose x hxEsc hxNear using hChoice
  refine ⟨x, ?_, ?_⟩
  · rw [Escapes]
    apply tendsto_atTop.2
    intro R
    obtain ⟨N, hN⟩ : ∃ N : Nat, R < N := exists_nat_gt R
    refine Filter.eventually_atTop.2 ⟨N, ?_⟩
    intro n hn
    have hNn : (N : Real) ≤ n := by exact_mod_cast hn
    exact le_of_lt (hN.trans (hNn.trans_lt (hxEsc n)))
  · apply Metric.tendsto_atTop.2
    intro ε hε
    have hHalf : 0 < ε / 2 := by linarith
    rcases Metric.tendsto_atTop.1 hbLim _ hHalf with ⟨Nb, hbN⟩
    have hInv :
        Tendsto (fun n : Nat =>
          (1 : Real) / (((n + 1 : Nat) : Real))) atTop (nhds 0) := by
      simpa only [Nat.cast_add, Nat.cast_one] using
        tendsto_one_div_add_atTop_nhds_zero_nat
    rcases Metric.tendsto_atTop.1 hInv _ hHalf with ⟨Ni, hiN⟩
    refine ⟨max Nb Ni, ?_⟩
    intro n hn
    have hnb : Nb ≤ n := le_trans (le_max_left _ _) hn
    have hni : Ni ≤ n := le_trans (le_max_right _ _) hn
    have hi : (1 : Real) / (((n + 1 : Nat) : Real)) < ε / 2 := by
      have hiDist := hiN n hni
      have hfrac :
          (0 : Real) < 1 / (((n + 1 : Nat) : Real)) := by positivity
      simpa only [Real.dist_eq, sub_zero, abs_of_pos hfrac] using hiDist
    calc
      dist (F (x n)) b ≤
          dist (F (x n)) (bseq n) + dist (bseq n) b := dist_triangle _ _ _
      _ < ε / 2 + ε / 2 :=
        add_lt_add ((hxNear n).trans hi) (hbN n hnb)
      _ = ε := by ring

/-- The explicit source family approaching a finite critical target. -/
def criticalEscapeSource
    (_h : Complex[X]) (p t ε : Complex) : C3 :=
  let y := (1 - ε) / t
  ![t / ε,
    y,
    p * ε ^ 3 - y ^ 2 * (4 - t * y) * ε]

@[simp]
theorem criticalEscapeSource_zero
    (h : Complex[X]) (p t ε : Complex) :
    criticalEscapeSource h p t ε 0 = t / ε := by
  simp [criticalEscapeSource]

@[simp]
theorem criticalEscapeSource_one
    (h : Complex[X]) (p t ε : Complex) :
    criticalEscapeSource h p t ε 1 = (1 - ε) / t := by
  simp [criticalEscapeSource]

/-- Exact image of the finite-critical escaping source. -/
theorem criticalEscape_maps
    (h : Complex[X]) (p t ε : Complex)
    (ht : t ≠ 0) (hε : ε ≠ 0) :
    Fh h (criticalEscapeSource h p t ε) =
      ![p,
        criticalQ h p t - ε / t,
        criticalR h p t + ε * t] := by
  let u := criticalEscapeSource h p t ε
  have hAeq : baseA (u 0) (u 1) = ε⁻¹ := by
    simp [u, criticalEscapeSource, baseA]
    field_simp [ht, hε]
    ring
  have hA : baseA (u 0) (u 1) ≠ 0 := by
    rw [hAeq]
    exact inv_ne_zero hε
  have hP : pCoord (u 0) (u 1) (u 2) = p := by
    simp [u, criticalEscapeSource, pCoord, baseA, baseB]
    field_simp [ht, hε]
    ring
  have hS : chartS (u 0) (u 1) = t := by
    rw [chartS, hAeq]
    simp [u, criticalEscapeSource]
    field_simp [hε]
  have h0 : Fh h u 0 = p := by
    simpa [Fh] using hP
  have h1 : Fh h u 1 = criticalQ h p t - ε / t := by
    rw [show Fh h u 1 = qCoord h (u 0) (u 1) (u 2) by rfl]
    rw [qCoord_chart_identity h _ _ _ hA, hP, hS]
    simp [u, criticalEscapeSource, criticalQ, phi]
    field_simp [ht]
    ring
  have h2 : Fh h u 2 = criticalR h p t + ε * t := by
    rw [show Fh h u 2 = rCoord h (u 0) (u 1) (u 2) by rfl]
    rw [rCoord_chart_identity h _ _ _ hA, hP, hS]
    simp [u, criticalEscapeSource, criticalR, theta]
    field_simp [ht]
    ring
  funext i
  fin_cases i
  · simpa [u] using h0
  · simpa [u] using h1
  · simpa [u] using h2

/-- The finite-critical source sequence escapes. -/
theorem criticalEscape_escapes
    (_h : Complex[X]) (p t : Complex) (ht : t ≠ 0) :
    Escapes (fun n => criticalEscapeSource _h p t (escapeEps n)) := by
  rw [Escapes]
  have hNat : Tendsto (fun n : Nat => ((n + 1 : Nat) : Real)) atTop atTop :=
    tendsto_natCast_atTop_atTop.comp (tendsto_add_atTop_nat 1)
  have htNorm : 0 < norm t := norm_pos_iff.mpr ht
  have hScaled :
      Tendsto (fun n : Nat => norm t * ((n + 1 : Nat) : Real)) atTop atTop :=
    (tendsto_const_mul_atTop_of_pos htNorm).2 hNat
  have hInvNorm :
      Tendsto (fun n => norm (t / escapeEps n)) atTop atTop := by
    have hfun :
        (fun n : Nat => norm (t / escapeEps n)) =
          fun n : Nat => norm t * ((n + 1 : Nat) : Real) := by
      funext n
      rw [norm_div, norm_escapeEps]
      have hn : (((n + 1 : Nat) : Real)) ≠ 0 := by positivity
      field_simp [hn]
    rw [hfun]
    exact hScaled
  exact tendsto_atTop_mono' atTop
    (Filter.Eventually.of_forall fun n =>
      norm_le_pi_norm (criticalEscapeSource _h p t (escapeEps n)) 0)
    hInvNorm

/-- Every directly parametrized finite critical target is nonproper. -/
theorem criticalImage_subset_nonproperness
    (h : Complex[X]) :
    criticalImage h ⊆ NonpropernessSet (Fh h) := by
  intro b hb
  rcases hb with ⟨p, t, ht, rfl⟩
  refine ⟨fun n => criticalEscapeSource h p t (escapeEps n),
    criticalEscape_escapes h p t ht, ?_⟩
  have hImage :
      (fun n => Fh h (criticalEscapeSource h p t (escapeEps n))) =
        fun n => ![p,
          criticalQ h p t - escapeEps n / t,
          criticalR h p t + escapeEps n * t] := by
    funext n
    exact criticalEscape_maps h p t (escapeEps n) ht (escapeEps_ne_zero n)
  rw [hImage]
  apply tendsto_pi_nhds.2
  intro i
  fin_cases i
  · simp [criticalTarget]
  · simpa [criticalTarget] using
      (tendsto_const_nhds.sub (escapeEps_tendsto_zero.div_const t))
  · simpa [criticalTarget] using
      (tendsto_const_nhds.add (escapeEps_tendsto_zero.mul_const t))

/-! ## The vertical component `p = 0`

The direct `A = 0` family only approaches the line `q = 0`; it does not
prove nonproperness of the full vertical hyperplane.  For the two maps used
here we instead construct a genuine lost degree-six branch.  With
`ε → 0`, put `s = ε⁻⁵` and `p = ε³(c + d ε²)`.  The constants are chosen so
that the divergent terms in the target `q` coordinate cancel.
-/

/-- A chosen fifth root in `Complex`. -/
noncomputable def chosenFifthRoot (a : Complex) : Complex :=
  Classical.choose (IsAlgClosed.exists_pow_nat_eq a (by norm_num : 0 < 5))

lemma chosenFifthRoot_pow_five (a : Complex) :
    chosenFifthRoot a ^ 5 = a :=
  Classical.choose_spec (IsAlgClosed.exists_pow_nat_eq a (by norm_num : 0 < 5))

lemma chosenFifthRoot_ne_zero {a : Complex} (ha : a ≠ 0) :
    chosenFifthRoot a ≠ 0 := by
  intro hz
  apply ha
  calc
    a = chosenFifthRoot a ^ 5 := (chosenFifthRoot_pow_five a).symm
    _ = 0 := by rw [hz]; simp

/-- Cancellation constant for CEX-004, satisfying `c^5 = -6`. -/
noncomputable def zeroBranchC004 : Complex := chosenFifthRoot (-6)

/-- Cancellation constant for CEX-006, satisfying `c^5 = 4`. -/
noncomputable def zeroBranchC006 : Complex := chosenFifthRoot 4

lemma zeroBranchC004_pow_five : zeroBranchC004 ^ 5 = (-6 : Complex) :=
  chosenFifthRoot_pow_five (-6)

lemma zeroBranchC006_pow_five : zeroBranchC006 ^ 5 = (4 : Complex) :=
  chosenFifthRoot_pow_five 4

lemma zeroBranchC004_ne_zero : zeroBranchC004 ≠ 0 :=
  chosenFifthRoot_ne_zero (by norm_num)

lemma zeroBranchC006_ne_zero : zeroBranchC006 ≠ 0 :=
  chosenFifthRoot_ne_zero (by norm_num)

/-- The correction carrying the prescribed limiting target coordinate `q`. -/
def zeroBranchD (q : Complex) : Complex := -q / 10

/-- The bounded rescaled first-coordinate parameter. -/
def zeroBranchW (c q ε : Complex) : Complex :=
  c + zeroBranchD q * ε ^ 2

/-- Target first coordinate of the lost branch. -/
def zeroBranchP (c q ε : Complex) : Complex :=
  ε ^ 3 * zeroBranchW c q ε

/-- Large inverse-root coordinate. -/
def zeroBranchS (ε : Complex) : Complex :=
  ε⁻¹ ^ 5

/-- Target second coordinate chosen so that `zeroBranchS ε` is exactly a
root of the inverse polynomial. -/
def zeroBranchQ
    (h : Complex[X]) (c q r ε : Complex) : Complex :=
  let p := zeroBranchP c q ε
  let s := zeroBranchS ε
  aCoeff h p * s ^ 4 + 2 * p * s + 2 / s - r / s ^ 2

/-- The chosen large value is an exact inverse root. -/
theorem omega_zeroBranch
    (h : Complex[X]) (c q r ε : Complex) (hε : ε ≠ 0) :
    (omega h (zeroBranchP c q ε) (zeroBranchQ h c q r ε) r).eval
      (zeroBranchS ε) = 0 := by
  simp [zeroBranchQ, omega_eval]
  field_simp [zeroBranchS, hε]
  ring

/-- The standard sixth-power difference quotient. -/
def sixthDiffSum (x c : Complex) : Complex :=
  x ^ 5 + x ^ 4 * c + x ^ 3 * c ^ 2 +
    x ^ 2 * c ^ 3 + x * c ^ 4 + c ^ 5

lemma sixthDiffSum_self (c : Complex) :
    sixthDiffSum c c = 6 * c ^ 5 := by
  unfold sixthDiffSum
  ring

/-- Cancellation model for the CEX-004 target `q` coordinate. -/
def zeroBranchQModel004 (q r ε : Complex) : Complex :=
  let w := zeroBranchW zeroBranchC004 q ε
  (1 / 3 : Complex) * zeroBranchD q * sixthDiffSum w zeroBranchC004 +
    2 * zeroBranchD q +
    (4 / 3 : Complex) * ε * w ^ 7 + 2 * ε ^ 5 - r * ε ^ 10

/-- Cancellation model for the CEX-006 target `q` coordinate. -/
def zeroBranchQModel006 (q r ε : Complex) : Complex :=
  let w := zeroBranchW zeroBranchC006 q ε
  (-1 / 2 : Complex) * zeroBranchD q * sixthDiffSum w zeroBranchC006 +
    2 * zeroBranchD q + 2 * ε ^ 5 - r * ε ^ 10

lemma zeroBranchC004_pow_six :
    zeroBranchC004 ^ 6 = -6 * zeroBranchC004 := by
  calc
    zeroBranchC004 ^ 6 = zeroBranchC004 ^ 5 * zeroBranchC004 := by ring
    _ = -6 * zeroBranchC004 := by rw [zeroBranchC004_pow_five]

lemma zeroBranchC006_pow_six :
    zeroBranchC006 ^ 6 = 4 * zeroBranchC006 := by
  calc
    zeroBranchC006 ^ 6 = zeroBranchC006 ^ 5 * zeroBranchC006 := by ring
    _ = 4 * zeroBranchC006 := by rw [zeroBranchC006_pow_five]

lemma zeroBranchQ004_eq_model
    (q r ε : Complex) (hε : ε ≠ 0) :
    zeroBranchQ eta004 zeroBranchC004 q r ε =
      zeroBranchQModel004 q r ε := by
  simp [zeroBranchQ, zeroBranchS, zeroBranchP, zeroBranchW,
    zeroBranchQModel004, sixthDiffSum, aCoeff, eta004_eval]
  field_simp [hε]
  ring_nf
  rw [zeroBranchC004_pow_six]
  ring

lemma zeroBranchQ006_eq_model
    (q r ε : Complex) (hε : ε ≠ 0) :
    zeroBranchQ eta006 zeroBranchC006 q r ε =
      zeroBranchQModel006 q r ε := by
  simp [zeroBranchQ, zeroBranchS, zeroBranchP, zeroBranchW,
    zeroBranchQModel006, sixthDiffSum, aCoeff, eta006_eval]
  field_simp [hε]
  ring_nf
  rw [zeroBranchC006_pow_six]
  ring

lemma zeroBranchP_tendsto_zero (c q : Complex) :
    Tendsto (fun n => zeroBranchP c q (escapeEps n)) atTop (nhds 0) := by
  have hcont : ContinuousAt (fun ε : Complex => zeroBranchP c q ε) 0 := by
    unfold zeroBranchP zeroBranchW zeroBranchD
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto (fun n => zeroBranchP c q (escapeEps n)) atTop
    (nhds (zeroBranchP c q 0)) at hlim
  simpa [zeroBranchP, zeroBranchW] using hlim

lemma zeroBranchQModel004_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchQModel004 q r (escapeEps n))
      atTop (nhds q) := by
  have hcont : ContinuousAt (fun ε : Complex => zeroBranchQModel004 q r ε) 0 := by
    unfold zeroBranchQModel004 zeroBranchW zeroBranchD sixthDiffSum
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto (fun n => zeroBranchQModel004 q r (escapeEps n)) atTop
    (nhds (zeroBranchQModel004 q r 0)) at hlim
  have hzero : zeroBranchQModel004 q r 0 = q := by
    simp [zeroBranchQModel004, zeroBranchW]
    rw [sixthDiffSum_self, zeroBranchC004_pow_five]
    simp [zeroBranchD]
    ring
  simpa only [hzero] using hlim

lemma zeroBranchQModel006_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchQModel006 q r (escapeEps n))
      atTop (nhds q) := by
  have hcont : ContinuousAt (fun ε : Complex => zeroBranchQModel006 q r ε) 0 := by
    unfold zeroBranchQModel006 zeroBranchW zeroBranchD sixthDiffSum
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto (fun n => zeroBranchQModel006 q r (escapeEps n)) atTop
    (nhds (zeroBranchQModel006 q r 0)) at hlim
  have hzero : zeroBranchQModel006 q r 0 = q := by
    simp [zeroBranchQModel006, zeroBranchW]
    rw [sixthDiffSum_self, zeroBranchC006_pow_five]
    simp [zeroBranchD]
    ring
  simpa only [hzero] using hlim

lemma zeroBranchQ004_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchQ eta004 zeroBranchC004 q r (escapeEps n))
      atTop (nhds q) := by
  exact (zeroBranchQModel004_tendsto q r).congr'
    (Filter.Eventually.of_forall fun n =>
      (zeroBranchQ004_eq_model q r (escapeEps n) (escapeEps_ne_zero n)).symm)

lemma zeroBranchQ006_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchQ eta006 zeroBranchC006 q r (escapeEps n))
      atTop (nhds q) := by
  exact (zeroBranchQModel006_tendsto q r).congr'
    (Filter.Eventually.of_forall fun n =>
      (zeroBranchQ006_eq_model q r (escapeEps n) (escapeEps_ne_zero n)).symm)

/-- Evaluated derivative along a zero-vertical lost branch. -/
def zeroBranchDerivative
    (h : Complex[X]) (c q r ε : Complex) : Complex :=
  (derivative (omega h (zeroBranchP c q ε)
    (zeroBranchQ h c q r ε) r)).eval (zeroBranchS ε)

/-- Cancellation model for `ε⁷ Ω'(s)`. -/
def zeroBranchDerivativeModel
    (h : Complex[X]) (c q r ε : Complex) : Complex :=
  let w := zeroBranchW c q ε
  (4 / 3 : Complex) * w ^ 6 * h.eval (zeroBranchP c q ε) +
    2 * w - 2 * ε ^ 7 + 2 * r * ε ^ 12

lemma zeroBranchDerivative_scaled
    (h : Complex[X]) (c q r ε : Complex) (hε : ε ≠ 0) :
    zeroBranchDerivative h c q r ε * ε ^ 7 =
      zeroBranchDerivativeModel h c q r ε := by
  simp [zeroBranchDerivative, zeroBranchDerivativeModel,
    zeroBranchQ, zeroBranchS, zeroBranchP, zeroBranchW,
    omega_derivative_eval, aCoeff]
  field_simp [hε]
  ring

lemma zeroBranchDerivativeModel004_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchDerivativeModel eta004 zeroBranchC004 q r
      (escapeEps n)) atTop (nhds (-6 * zeroBranchC004)) := by
  have hcont : ContinuousAt
      (fun ε : Complex => zeroBranchDerivativeModel eta004 zeroBranchC004 q r ε) 0 := by
    simp only [zeroBranchDerivativeModel, zeroBranchW, zeroBranchP, zeroBranchD,
      eta004_eval]
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto
    (fun n => zeroBranchDerivativeModel eta004 zeroBranchC004 q r (escapeEps n))
    atTop (nhds (zeroBranchDerivativeModel eta004 zeroBranchC004 q r 0)) at hlim
  have hzero :
      zeroBranchDerivativeModel eta004 zeroBranchC004 q r 0 =
        -6 * zeroBranchC004 := by
    simp [zeroBranchDerivativeModel, zeroBranchP, zeroBranchW, eta004_eval]
    rw [zeroBranchC004_pow_six]
    ring
  simpa only [hzero] using hlim

lemma zeroBranchDerivativeModel006_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchDerivativeModel eta006 zeroBranchC006 q r
      (escapeEps n)) atTop (nhds (-6 * zeroBranchC006)) := by
  have hcont : ContinuousAt
      (fun ε : Complex => zeroBranchDerivativeModel eta006 zeroBranchC006 q r ε) 0 := by
    simp only [zeroBranchDerivativeModel, zeroBranchW, zeroBranchP, zeroBranchD,
      eta006_eval]
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto
    (fun n => zeroBranchDerivativeModel eta006 zeroBranchC006 q r (escapeEps n))
    atTop (nhds (zeroBranchDerivativeModel eta006 zeroBranchC006 q r 0)) at hlim
  have hzero :
      zeroBranchDerivativeModel eta006 zeroBranchC006 q r 0 =
        -6 * zeroBranchC006 := by
    simp [zeroBranchDerivativeModel, zeroBranchP, zeroBranchW, eta006_eval]
    rw [zeroBranchC006_pow_six]
    ring
  simpa only [hzero] using hlim

lemma zeroBranchDerivative004_eventually_ne_zero (q r : Complex) :
    ∀ᶠ n : Nat in atTop,
      zeroBranchDerivative eta004 zeroBranchC004 q r (escapeEps n) ≠ 0 := by
  have hnon : ∀ᶠ n : Nat in atTop,
      zeroBranchDerivativeModel eta004 zeroBranchC004 q r (escapeEps n) ≠ 0 :=
    (zeroBranchDerivativeModel004_tendsto q r).eventually_ne
      (mul_ne_zero (by norm_num) zeroBranchC004_ne_zero)
  filter_upwards [hnon] with n hn
  intro hz
  apply hn
  rw [← zeroBranchDerivative_scaled eta004 zeroBranchC004 q r
    (escapeEps n) (escapeEps_ne_zero n)]
  simp [hz]

lemma zeroBranchDerivative006_eventually_ne_zero (q r : Complex) :
    ∀ᶠ n : Nat in atTop,
      zeroBranchDerivative eta006 zeroBranchC006 q r (escapeEps n) ≠ 0 := by
  have hnon : ∀ᶠ n : Nat in atTop,
      zeroBranchDerivativeModel eta006 zeroBranchC006 q r (escapeEps n) ≠ 0 :=
    (zeroBranchDerivativeModel006_tendsto q r).eventually_ne
      (mul_ne_zero (by norm_num) zeroBranchC006_ne_zero)
  filter_upwards [hnon] with n hn
  intro hz
  apply hn
  rw [← zeroBranchDerivative_scaled eta006 zeroBranchC006 q r
    (escapeEps n) (escapeEps_ne_zero n)]
  simp [hz]

/-- Cancellation model for the reconstructed `y` coordinate multiplied by
`ε²`. -/
def zeroBranchYModel
    (h : Complex[X]) (c q r ε : Complex) : Complex :=
  -zeroBranchW c q ε -
      (2 / 3 : Complex) * (zeroBranchW c q ε) ^ 6 *
        h.eval (zeroBranchP c q ε) +
    2 * ε ^ 7 - r * ε ^ 12

lemma zeroBranch_y_scaled
    (h : Complex[X]) (c q r ε : Complex) (hε : ε ≠ 0) :
    reconstruct h (zeroBranchP c q ε) (zeroBranchQ h c q r ε)
        (zeroBranchS ε) 1 * ε ^ 2 =
      zeroBranchYModel h c q r ε := by
  simp [reconstruct, zeroBranchYModel, zeroBranchQ, phi,
    zeroBranchS, zeroBranchP, zeroBranchW, aCoeff]
  field_simp [hε]
  ring

lemma zeroBranchYModel004_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchYModel eta004 zeroBranchC004 q r
      (escapeEps n)) atTop (nhds (3 * zeroBranchC004)) := by
  have hcont : ContinuousAt
      (fun ε : Complex => zeroBranchYModel eta004 zeroBranchC004 q r ε) 0 := by
    simp only [zeroBranchYModel, zeroBranchW, zeroBranchP, zeroBranchD, eta004_eval]
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto
    (fun n => zeroBranchYModel eta004 zeroBranchC004 q r (escapeEps n))
    atTop (nhds (zeroBranchYModel eta004 zeroBranchC004 q r 0)) at hlim
  have hzero : zeroBranchYModel eta004 zeroBranchC004 q r 0 =
      3 * zeroBranchC004 := by
    simp [zeroBranchYModel, zeroBranchP, zeroBranchW, eta004_eval]
    rw [zeroBranchC004_pow_six]
    ring
  simpa only [hzero] using hlim

lemma zeroBranchYModel006_tendsto (q r : Complex) :
    Tendsto (fun n => zeroBranchYModel eta006 zeroBranchC006 q r
      (escapeEps n)) atTop (nhds (3 * zeroBranchC006)) := by
  have hcont : ContinuousAt
      (fun ε : Complex => zeroBranchYModel eta006 zeroBranchC006 q r ε) 0 := by
    simp only [zeroBranchYModel, zeroBranchW, zeroBranchP, zeroBranchD, eta006_eval]
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto
    (fun n => zeroBranchYModel eta006 zeroBranchC006 q r (escapeEps n))
    atTop (nhds (zeroBranchYModel eta006 zeroBranchC006 q r 0)) at hlim
  have hzero : zeroBranchYModel eta006 zeroBranchC006 q r 0 =
      3 * zeroBranchC006 := by
    simp [zeroBranchYModel, zeroBranchP, zeroBranchW, eta006_eval]
    rw [zeroBranchC006_pow_six]
    ring
  simpa only [hzero] using hlim

lemma zeroBranch_y_scaled_tendsto004 (q r : Complex) :
    Tendsto (fun n =>
      reconstruct eta004
        (zeroBranchP zeroBranchC004 q (escapeEps n))
        (zeroBranchQ eta004 zeroBranchC004 q r (escapeEps n))
        (zeroBranchS (escapeEps n)) 1 * escapeEps n ^ 2)
      atTop (nhds (3 * zeroBranchC004)) := by
  exact (zeroBranchYModel004_tendsto q r).congr'
    (Filter.Eventually.of_forall fun n =>
      (zeroBranch_y_scaled eta004 zeroBranchC004 q r
        (escapeEps n) (escapeEps_ne_zero n)).symm)

lemma zeroBranch_y_scaled_tendsto006 (q r : Complex) :
    Tendsto (fun n =>
      reconstruct eta006
        (zeroBranchP zeroBranchC006 q (escapeEps n))
        (zeroBranchQ eta006 zeroBranchC006 q r (escapeEps n))
        (zeroBranchS (escapeEps n)) 1 * escapeEps n ^ 2)
      atTop (nhds (3 * zeroBranchC006)) := by
  exact (zeroBranchYModel006_tendsto q r).congr'
    (Filter.Eventually.of_forall fun n =>
      (zeroBranch_y_scaled eta006 zeroBranchC006 q r
        (escapeEps n) (escapeEps_ne_zero n)).symm)

/-- A nonzero quadratic rescaling limit forces the reconstructed sources to
escape. -/
theorem zeroBranchReconstruct_escapes
    (h : Complex[X]) (c q r : Complex) (hc : c ≠ 0)
    (hyScaled : Tendsto (fun n =>
      reconstruct h
        (zeroBranchP c q (escapeEps n))
        (zeroBranchQ h c q r (escapeEps n))
        (zeroBranchS (escapeEps n)) 1 * escapeEps n ^ 2)
      atTop (nhds (3 * c))) :
    Escapes (fun n => reconstruct h
      (zeroBranchP c q (escapeEps n))
      (zeroBranchQ h c q r (escapeEps n))
      (zeroBranchS (escapeEps n))) := by
  rw [Escapes]
  let y : Nat -> Complex := fun n =>
    reconstruct h
      (zeroBranchP c q (escapeEps n))
      (zeroBranchQ h c q r (escapeEps n))
      (zeroBranchS (escapeEps n)) 1
  have hy : Tendsto (fun n => y n * escapeEps n ^ 2)
      atTop (nhds (3 * c)) := by
    simpa [y] using hyScaled
  have hNorm : Tendsto (fun n => norm (y n * escapeEps n ^ 2)) atTop
      (nhds (norm (3 * c))) := hy.norm
  have h3c : (3 : Complex) * c ≠ 0 := mul_ne_zero (by norm_num) hc
  have hK : 0 < norm (3 * c) / 2 := by
    exact div_pos (norm_pos_iff.mpr h3c) (by norm_num)
  have hLower : ∀ᶠ n in atTop,
      norm (3 * c) / 2 < norm (y n * escapeEps n ^ 2) := by
    rcases Metric.tendsto_atTop.1 hNorm (norm (3 * c) / 2) hK with ⟨N, hN⟩
    refine Filter.eventually_atTop.2 ⟨N, ?_⟩
    intro n hn
    have hd := hN n hn
    rw [Real.dist_eq] at hd
    have hlo := (abs_lt.mp hd).1
    linarith
  have hNat : Tendsto (fun n : Nat => ((n + 1 : Nat) : Real)) atTop atTop :=
    tendsto_natCast_atTop_atTop.comp (tendsto_add_atTop_nat 1)
  have hSquare : Tendsto (fun n : Nat => (((n + 1 : Nat) : Real)) ^ 2)
      atTop atTop := by
    exact tendsto_atTop_mono' atTop
      (Filter.Eventually.of_forall fun n => by
        have hn : (1 : Real) ≤ (n + 1 : Nat) := by exact_mod_cast Nat.succ_le_succ (Nat.zero_le n)
        nlinarith)
      hNat
  have hGrowth : Tendsto
      (fun n : Nat => norm (3 * c) / 2 * (((n + 1 : Nat) : Real)) ^ 2)
      atTop atTop :=
    (tendsto_const_mul_atTop_of_pos hK).2 hSquare
  apply tendsto_atTop_mono' atTop
    (Filter.Eventually.of_forall fun n => norm_le_pi_norm _ 1)
  exact tendsto_atTop_mono' atTop (hLower.mono fun n hn => by
    have hden : (0 : Real) < (((n + 1 : Nat) : Real)) ^ 2 := by positivity
    have hEq : norm (y n * escapeEps n ^ 2) =
        norm (y n) / (((n + 1 : Nat) : Real)) ^ 2 := by
      rw [norm_mul, norm_pow, norm_escapeEps]
      have hn0 : (((n + 1 : Nat) : Real)) ≠ 0 := by positivity
      field_simp [hn0]
    rw [hEq] at hn
    exact le_of_lt ((lt_div_iff₀ hden).mp hn)) hGrowth

lemma zeroBranchReconstruct_escapes004 (q r : Complex) :
    Escapes (fun n => reconstruct eta004
      (zeroBranchP zeroBranchC004 q (escapeEps n))
      (zeroBranchQ eta004 zeroBranchC004 q r (escapeEps n))
      (zeroBranchS (escapeEps n))) :=
  zeroBranchReconstruct_escapes eta004 zeroBranchC004 q r
    zeroBranchC004_ne_zero (zeroBranch_y_scaled_tendsto004 q r)

lemma zeroBranchReconstruct_escapes006 (q r : Complex) :
    Escapes (fun n => reconstruct eta006
      (zeroBranchP zeroBranchC006 q (escapeEps n))
      (zeroBranchQ eta006 zeroBranchC006 q r (escapeEps n))
      (zeroBranchS (escapeEps n))) :=
  zeroBranchReconstruct_escapes eta006 zeroBranchC006 q r
    zeroBranchC006_ne_zero (zeroBranch_y_scaled_tendsto006 q r)

/-- Every target on `p = 0` is nonproper for CEX-004. -/
theorem pHyperplane_zero_subset_nonproperness004 :
    pHyperplane 0 ⊆ NonpropernessSet F004 := by
  intro b hb
  have hp : b 0 = 0 := hb
  let u : Nat -> C3 := fun n => reconstruct eta004
    (zeroBranchP zeroBranchC004 (b 1) (escapeEps n))
    (zeroBranchQ eta004 zeroBranchC004 (b 1) (b 2) (escapeEps n))
    (zeroBranchS (escapeEps n))
  refine ⟨u, ?_, ?_⟩
  · simpa [u] using zeroBranchReconstruct_escapes004 (b 1) (b 2)
  · have hMap : ∀ᶠ n in atTop,
        F004 (u n) = ![
          zeroBranchP zeroBranchC004 (b 1) (escapeEps n),
          zeroBranchQ eta004 zeroBranchC004 (b 1) (b 2) (escapeEps n),
          b 2] := by
      filter_upwards [zeroBranchDerivative004_eventually_ne_zero (b 1) (b 2)] with n hn
      exact reconstruct_maps_to eta004 _ _ (b 2) _
        (omega_zeroBranch eta004 zeroBranchC004 (b 1) (b 2)
          (escapeEps n) (escapeEps_ne_zero n)) hn
    have hTarget : Tendsto (fun n => ![
        zeroBranchP zeroBranchC004 (b 1) (escapeEps n),
        zeroBranchQ eta004 zeroBranchC004 (b 1) (b 2) (escapeEps n),
        b 2]) atTop (nhds b) := by
      apply tendsto_pi_nhds.2
      intro i
      fin_cases i
      · simpa [hp] using zeroBranchP_tendsto_zero zeroBranchC004 (b 1)
      · simpa using zeroBranchQ004_tendsto (b 1) (b 2)
      · simp
    exact hTarget.congr' (hMap.mono fun n hn => hn.symm)

/-- Every target on `p = 0` is nonproper for CEX-006. -/
theorem pHyperplane_zero_subset_nonproperness006 :
    pHyperplane 0 ⊆ NonpropernessSet F006 := by
  intro b hb
  have hp : b 0 = 0 := hb
  let u : Nat -> C3 := fun n => reconstruct eta006
    (zeroBranchP zeroBranchC006 (b 1) (escapeEps n))
    (zeroBranchQ eta006 zeroBranchC006 (b 1) (b 2) (escapeEps n))
    (zeroBranchS (escapeEps n))
  refine ⟨u, ?_, ?_⟩
  · simpa [u] using zeroBranchReconstruct_escapes006 (b 1) (b 2)
  · have hMap : ∀ᶠ n in atTop,
        F006 (u n) = ![
          zeroBranchP zeroBranchC006 (b 1) (escapeEps n),
          zeroBranchQ eta006 zeroBranchC006 (b 1) (b 2) (escapeEps n),
          b 2] := by
      filter_upwards [zeroBranchDerivative006_eventually_ne_zero (b 1) (b 2)] with n hn
      exact reconstruct_maps_to eta006 _ _ (b 2) _
        (omega_zeroBranch eta006 zeroBranchC006 (b 1) (b 2)
          (escapeEps n) (escapeEps_ne_zero n)) hn
    have hTarget : Tendsto (fun n => ![
        zeroBranchP zeroBranchC006 (b 1) (escapeEps n),
        zeroBranchQ eta006 zeroBranchC006 (b 1) (b 2) (escapeEps n),
        b 2]) atTop (nhds b) := by
      apply tendsto_pi_nhds.2
      intro i
      fin_cases i
      · simpa [hp] using zeroBranchP_tendsto_zero zeroBranchC006 (b 1)
      · simpa using zeroBranchQ006_tendsto (b 1) (b 2)
      · simp
    exact hTarget.congr' (hMap.mono fun n hn => hn.symm)


/-! ## The additional CEX-004 vertical component `p = -1/4` -/

/-- The fixed nonzero root of `eta004`. -/
def alpha004 : Complex := -1 / 4

/-- First asymptotic correction in the lost degree-six branches. -/
def alpha004C : Complex := 1536

/-- Second asymptotic correction, carrying the target `q`. -/
def alpha004D (q : Complex) : Complex := 3072 * q

/-- The target first-coordinate branch, written in the large-root variable
`s`. -/
def rootVerticalP (q s : Complex) : Complex :=
  alpha004 + alpha004C / s ^ 3 + alpha004D q / s ^ 4

/-- The target second coordinate chosen so that `s` is exactly a root of
`omega`; it converges to the prescribed `q`. -/
def rootVerticalQ (q r s : Complex) : Complex :=
  aCoeff eta004 (rootVerticalP q s) * s ^ 4 +
    2 * rootVerticalP q s * s + 2 / s - r / s ^ 2

/-- Exact root identity for the additional vertical branch. -/
theorem omega_rootVertical
    (q r s : Complex) (hs : s ≠ 0) :
    (omega eta004 (rootVerticalP q s) (rootVerticalQ q r s) r).eval s = 0 := by
  simp [rootVerticalQ, aCoeff, omega_eval]
  field_simp [hs]
  ring

/-- The same first-coordinate branch in the small parameter `ε = 1/s`. -/
def rootVerticalPEps (q ε : Complex) : Complex :=
  alpha004 + alpha004C * ε ^ 3 + alpha004D q * ε ^ 4

lemma rootVerticalP_inv
    (q ε : Complex) (_hε : ε ≠ 0) :
    rootVerticalP q ε⁻¹ = rootVerticalPEps q ε := by
  simp [rootVerticalP, rootVerticalPEps, div_eq_mul_inv]

/-- The quotient `(p(ε)^6-alpha^6)/ε`, after cancelling its visible factor
of `ε`. -/
def powSixSum (x a : Complex) : Complex :=
  x ^ 5 + x ^ 4 * a + x ^ 3 * a ^ 2 + x ^ 2 * a ^ 3 + x * a ^ 4 + a ^ 5

def rootVerticalPowQuot (q ε : Complex) : Complex :=
  (alpha004C * ε ^ 2 + alpha004D q * ε ^ 3) *
    powSixSum (rootVerticalPEps q ε) alpha004

lemma rootVertical_powQuot
    (q ε : Complex) (hε : ε ≠ 0) :
    (rootVerticalPEps q ε ^ 6 - alpha004 ^ 6) / ε =
      rootVerticalPowQuot q ε := by
  simp [rootVerticalPEps, rootVerticalPowQuot, powSixSum]
  field_simp [hε]
  ring

lemma rootVerticalPEps_tendsto
    (q : Complex) :
    Tendsto (fun n => rootVerticalPEps q (escapeEps n))
      atTop (nhds alpha004) := by
  have hcont : ContinuousAt (fun ε : Complex => rootVerticalPEps q ε) 0 := by
    unfold rootVerticalPEps alpha004C alpha004D
    fun_prop
  have hlim := hcont.tendsto.comp escapeEps_tendsto_zero
  change Tendsto (fun n => rootVerticalPEps q (escapeEps n)) atTop
    (nhds (rootVerticalPEps q 0)) at hlim
  have hzero : rootVerticalPEps q 0 = alpha004 := by
    simp [rootVerticalPEps]
  simpa only [hzero] using hlim

lemma rootVerticalPowQuot_tendsto_zero
    (q : Complex) :
    Tendsto (fun n => rootVerticalPowQuot q (escapeEps n))
      atTop (nhds 0) := by
  have he := escapeEps_tendsto_zero
  have hp := rootVerticalPEps_tendsto q
  have hsmall : Tendsto
      (fun n => alpha004C * escapeEps n ^ 2 +
        alpha004D q * escapeEps n ^ 3) atTop (nhds 0) := by
    simpa using
      (tendsto_const_nhds.mul (he.pow 2)).add
        (tendsto_const_nhds.mul (he.pow 3))
  have hsum : Tendsto
      (fun n => powSixSum (rootVerticalPEps q (escapeEps n)) alpha004)
      atTop (nhds (powSixSum alpha004 alpha004)) := by
    simpa [powSixSum] using
      (((((hp.pow 5).add ((hp.pow 4).mul_const alpha004)).add
        ((hp.pow 3).mul_const (alpha004 ^ 2))).add
        ((hp.pow 2).mul_const (alpha004 ^ 3))).add
        (hp.mul_const (alpha004 ^ 4))).add tendsto_const_nhds
  simpa [rootVerticalPowQuot] using hsmall.mul hsum

/-- A cancellation-normal form for the second target coordinate. -/
def rootVerticalQModel (q r ε : Complex) : Complex :=
  (4 / 3 : Complex) * alpha004C * rootVerticalPowQuot q ε +
    (4 / 3 : Complex) * alpha004D q * rootVerticalPEps q ε ^ 6 +
    2 * alpha004C * ε ^ 2 + 2 * alpha004D q * ε ^ 3 +
    2 * ε - r * ε ^ 2

lemma rootVerticalQ_inv_eq_model
    (q r ε : Complex) (hε : ε ≠ 0) :
    rootVerticalQ q r ε⁻¹ = rootVerticalQModel q r ε := by
  unfold rootVerticalQ
  rw [rootVerticalP_inv q ε hε]
  simp [rootVerticalQModel, rootVerticalPowQuot, powSixSum, aCoeff,
    eta004_eval, rootVerticalPEps, alpha004, alpha004C, alpha004D,
    div_eq_mul_inv]
  field_simp [hε]
  ring

lemma rootVerticalQModel_tendsto
    (q r : Complex) :
    Tendsto (fun n => rootVerticalQModel q r (escapeEps n))
      atTop (nhds q) := by
  have he := escapeEps_tendsto_zero
  have hp := rootVerticalPEps_tendsto q
  have hquot := rootVerticalPowQuot_tendsto_zero q
  have hlim : Tendsto (fun n => rootVerticalQModel q r (escapeEps n)) atTop
      (nhds ((4 / 3 : Complex) * alpha004D q * alpha004 ^ 6)) := by
    simpa [rootVerticalQModel] using
      (((((tendsto_const_nhds.mul hquot).add
        ((tendsto_const_nhds.mul (hp.pow 6)))).add
        (tendsto_const_nhds.mul (he.pow 2))).add
        (tendsto_const_nhds.mul (he.pow 3))).add
        (tendsto_const_nhds.mul he)).sub
        (tendsto_const_nhds.mul (he.pow 2))
  convert hlim using 1
  simp [alpha004D, alpha004]
  ring

/-- The branch first coordinate tends to `-1/4`. -/
theorem rootVerticalP_tendsto
    (q : Complex) :
    Tendsto (fun n => rootVerticalP q (((n + 1 : Nat) : Complex)))
      atTop (nhds alpha004) := by
  apply Tendsto.congr' _ (rootVerticalPEps_tendsto q)
  filter_upwards [] with n
  rw [← rootVerticalP_inv q (escapeEps n) (escapeEps_ne_zero n)]
  simp [escapeEps]

/-- The branch second coordinate tends to the prescribed `q`. -/
theorem rootVerticalQ_tendsto
    (q r : Complex) :
    Tendsto (fun n => rootVerticalQ q r (((n + 1 : Nat) : Complex)))
      atTop (nhds q) := by
  apply Tendsto.congr' _ (rootVerticalQModel_tendsto q r)
  filter_upwards [] with n
  rw [← rootVerticalQ_inv_eq_model q r (escapeEps n) (escapeEps_ne_zero n)]
  simp [escapeEps]

/-- Evaluated derivative along the additional CEX-004 branch. -/
def rootVerticalDerivative (q r s : Complex) : Complex :=
  (derivative (omega eta004 (rootVerticalP q s) (rootVerticalQ q r s) r)).eval s

/-- Cancellation-normal form for `ε² Omega'(1/ε)`. -/
def rootVerticalDerivativeModel (q r ε : Complex) : Complex :=
  8 * alpha004C * rootVerticalPEps q ε ^ 6 +
    8 * alpha004D q * ε * rootVerticalPEps q ε ^ 6 +
    6 * rootVerticalPEps q ε -
    2 * rootVerticalQModel q r ε * ε + 2 * ε ^ 2

lemma rootVerticalDerivative_scaled
    (q r ε : Complex) (hε : ε ≠ 0) :
    rootVerticalDerivative q r ε⁻¹ * ε ^ 2 =
      rootVerticalDerivativeModel q r ε := by
  unfold rootVerticalDerivative
  rw [rootVerticalP_inv q ε hε, rootVerticalQ_inv_eq_model q r ε hε]
  rw [omega_derivative_eval]
  rw [show eta004.eval (rootVerticalPEps q ε) =
      4 * (rootVerticalPEps q ε - alpha004) by
    simp [eta004_eval, alpha004]
    ring]
  simp [rootVerticalDerivativeModel, rootVerticalPEps]
  field_simp [hε]
  ring

lemma rootVerticalDerivativeModel_tendsto
    (q r : Complex) :
    Tendsto (fun n => rootVerticalDerivativeModel q r (escapeEps n))
      atTop (nhds (3 / 2 : Complex)) := by
  have he := escapeEps_tendsto_zero
  have hp := rootVerticalPEps_tendsto q
  have hq := rootVerticalQModel_tendsto q r
  have hlim : Tendsto
      (fun n => rootVerticalDerivativeModel q r (escapeEps n)) atTop
      (nhds (8 * alpha004C * alpha004 ^ 6 + 6 * alpha004)) := by
    simpa [rootVerticalDerivativeModel] using
      ((((tendsto_const_nhds.mul (hp.pow 6)).add
        (((tendsto_const_nhds.mul he).mul (hp.pow 6)))).add
        (tendsto_const_nhds.mul hp)).sub
        ((tendsto_const_nhds.mul hq).mul he)).add
        (tendsto_const_nhds.mul (he.pow 2))
  convert hlim using 1
  simp [alpha004C, alpha004]
  ring

/-- The branch derivative is eventually nonzero. -/
theorem rootVerticalDerivative_eventually_ne_zero
    (q r : Complex) :
    ∀ᶠ n : Nat in atTop,
      rootVerticalDerivative q r (((n + 1 : Nat) : Complex)) ≠ 0 := by
  have hNonzero : ∀ᶠ n in atTop,
      rootVerticalDerivativeModel q r (escapeEps n) ≠ 0 :=
    (rootVerticalDerivativeModel_tendsto q r).eventually_ne
      (by norm_num : (3 / 2 : Complex) ≠ 0)
  filter_upwards [hNonzero] with n hn
  intro hzero
  apply hn
  calc
    rootVerticalDerivativeModel q r (escapeEps n) =
        rootVerticalDerivative q r (escapeEps n)⁻¹ * escapeEps n ^ 2 :=
      (rootVerticalDerivative_scaled q r (escapeEps n)
        (escapeEps_ne_zero n)).symm
    _ = rootVerticalDerivative q r (((n + 1 : Nat) : Complex)) *
        escapeEps n ^ 2 := by simp [escapeEps]
    _ = 0 := by rw [hzero]; simp

/-- Exact cancellation formula for the reconstructed `y` coordinate. -/
lemma rootVertical_y_scaled_eq_model
    (q r ε : Complex) (hε : ε ≠ 0) :
    reconstruct eta004
        (rootVerticalP q ε⁻¹)
        (rootVerticalQ q r ε⁻¹)
        ε⁻¹ 1 * ε =
      rootVerticalQModel q r ε * ε -
        3 * rootVerticalPEps q ε -
        4 * alpha004C * rootVerticalPEps q ε ^ 6 -
        4 * alpha004D q * ε * rootVerticalPEps q ε ^ 6 := by
  rw [rootVerticalP_inv q ε hε, rootVerticalQ_inv_eq_model q r ε hε]
  simp [reconstruct, phi, eta004_eval, alpha004, alpha004C, alpha004D,
    rootVerticalPEps]
  field_simp [hε]
  ring

/-- The reconstructed `y` coordinate, multiplied by `ε`, has the nonzero
limit `-3/4`. -/
theorem rootVertical_y_scaled_tendsto
    (q r : Complex) :
    Tendsto (fun n =>
      (reconstruct eta004
        (rootVerticalP q (((n + 1 : Nat) : Complex)))
        (rootVerticalQ q r (((n + 1 : Nat) : Complex)))
        (((n + 1 : Nat) : Complex)) 1) * escapeEps n)
      atTop (nhds (-3 / 4 : Complex)) := by
  have he := escapeEps_tendsto_zero
  have hp := rootVerticalPEps_tendsto q
  have hq := rootVerticalQModel_tendsto q r
  have hlim : Tendsto (fun n =>
      rootVerticalQModel q r (escapeEps n) * escapeEps n -
        3 * rootVerticalPEps q (escapeEps n) -
        4 * alpha004C * rootVerticalPEps q (escapeEps n) ^ 6 -
        4 * alpha004D q * escapeEps n *
          rootVerticalPEps q (escapeEps n) ^ 6)
      atTop (nhds (-3 * alpha004 - 4 * alpha004C * alpha004 ^ 6)) := by
    simpa using
      (((hq.mul he).sub (tendsto_const_nhds.mul hp)).sub
        (tendsto_const_nhds.mul (hp.pow 6))).sub
        (((tendsto_const_nhds.mul he).mul (hp.pow 6)))
  have hmodel : Tendsto (fun n =>
      (reconstruct eta004
        (rootVerticalP q (((n + 1 : Nat) : Complex)))
        (rootVerticalQ q r (((n + 1 : Nat) : Complex)))
        (((n + 1 : Nat) : Complex)) 1) * escapeEps n)
      atTop (nhds (-3 * alpha004 - 4 * alpha004C * alpha004 ^ 6)) := by
    exact hlim.congr' (Filter.Eventually.of_forall fun n => by
      have hε := escapeEps_ne_zero n
      have hEq := rootVertical_y_scaled_eq_model q r (escapeEps n) hε
      simpa [escapeEps] using hEq.symm)
  convert hmodel using 1
  simp [alpha004C, alpha004]
  ring

/-- The reconstructed branch sources escape. -/
theorem rootVerticalReconstruct_escapes
    (q r : Complex) :
    Escapes (fun n => reconstruct eta004
      (rootVerticalP q (((n + 1 : Nat) : Complex)))
      (rootVerticalQ q r (((n + 1 : Nat) : Complex)))
      (((n + 1 : Nat) : Complex))) := by
  rw [Escapes]
  let y : Nat -> Complex := fun n =>
    reconstruct eta004
      (rootVerticalP q (((n + 1 : Nat) : Complex)))
      (rootVerticalQ q r (((n + 1 : Nat) : Complex)))
      (((n + 1 : Nat) : Complex)) 1
  have hyScaled : Tendsto (fun n => y n * escapeEps n)
      atTop (nhds (-3 / 4 : Complex)) := by
    simpa [y] using rootVertical_y_scaled_tendsto q r
  have hNorm : Tendsto (fun n => norm (y n * escapeEps n)) atTop
      (nhds (3 / 4 : Real)) := by
    convert hyScaled.norm using 1
    norm_num
  have hLower : ∀ᶠ n in atTop, (3 / 8 : Real) < norm (y n * escapeEps n) := by
    rcases Metric.tendsto_atTop.1 hNorm (3 / 8 : Real) (by norm_num) with ⟨N, hN⟩
    refine Filter.eventually_atTop.2 ⟨N, ?_⟩
    intro n hn
    have hd := hN n hn
    rw [Real.dist_eq] at hd
    have hlo := (abs_lt.mp hd).1
    linarith
  have hNat : Tendsto (fun n : Nat => (n : Real) + 1) atTop atTop := by
    simpa [Function.comp_def] using
      ((tendsto_natCast_atTop_atTop (R := Real)).comp
        (tendsto_add_atTop_nat 1))
  have hLinear : Tendsto
      (fun n : Nat => (3 / 8 : Real) * ((n : Real) + 1))
      atTop atTop :=
    (tendsto_const_mul_atTop_of_pos (by norm_num)).2 hNat
  have hyInf : Tendsto (fun n => norm (y n)) atTop atTop := by
    exact tendsto_atTop_mono' atTop (hLower.mono fun n hn => by
      have hden : (0 : Real) < (n : Real) + 1 := by positivity
      have hEq : norm (y n * escapeEps n) =
          norm (y n) / ((n : Real) + 1) := by
        rw [norm_mul, norm_escapeEps]
        simp only [Nat.cast_add, Nat.cast_one]
        ring
      rw [hEq] at hn
      exact le_of_lt ((lt_div_iff₀ hden).mp hn)) hLinear
  exact tendsto_atTop_mono' atTop
    (Filter.Eventually.of_forall fun n => norm_le_pi_norm _ 1) hyInf

/-- The whole additional vertical hyperplane of CEX-004 is nonproper. -/
theorem pHyperplane_root_subset_nonproperness004 :
    pHyperplane alpha004 ⊆ NonpropernessSet F004 := by
  intro b hb
  have hp : b 0 = alpha004 := hb
  let s : Nat -> Complex := fun n => ((n + 1 : Nat) : Complex)
  let p : Nat -> Complex := fun n => rootVerticalP (b 1) (s n)
  let q : Nat -> Complex := fun n => rootVerticalQ (b 1) (b 2) (s n)
  let u : Nat -> C3 := fun n => reconstruct eta004 (p n) (q n) (s n)
  refine ⟨u, ?_, ?_⟩
  · simpa [u, p, q, s] using rootVerticalReconstruct_escapes (b 1) (b 2)
  · have hEventuallySimple := rootVerticalDerivative_eventually_ne_zero (b 1) (b 2)
    have hEventuallyMap :
        ∀ᶠ n in atTop, F004 (u n) = ![p n, q n, b 2] := by
      filter_upwards [hEventuallySimple] with n hSimple
      have hs : s n ≠ 0 := by
        dsimp [s]
        exact_mod_cast Nat.succ_ne_zero n
      exact reconstruct_maps_to eta004 (p n) (q n) (b 2) (s n)
        (omega_rootVertical (b 1) (b 2) (s n) hs) hSimple
    have hTarget : Tendsto (fun n => ![p n, q n, b 2]) atTop (nhds b) := by
      apply tendsto_pi_nhds.2
      intro i
      fin_cases i
      · simpa [p, s, hp] using rootVerticalP_tendsto (b 1)
      · simpa [q, s] using rootVerticalQ_tendsto (b 1) (b 2)
      · simp
    exact hTarget.congr' (hEventuallyMap.mono fun n hn => hn.symm)

end

end DegreeSixKeller
