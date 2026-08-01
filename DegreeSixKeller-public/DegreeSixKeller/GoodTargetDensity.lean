import DegreeSixKeller.InverseChart
import Mathlib.FieldTheory.IsAlgClosed.Basic
import Mathlib.Topology.Separation.Basic

/-!
# Density of targets with simple inverse roots

For a nonzero deformation polynomial, the first target coordinate can be
perturbed away from the finite leading-coefficient exceptional set and the
third coordinate away from the finite critical-value set.  The resulting
inverse polynomial has only simple roots; the middle coordinate stays fixed.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set
open scoped Polynomial

noncomputable section

/-- The value of the inverse polynomial before subtracting its target
coordinate `r`. -/
def omegaCriticalValue
    (h : Complex[X]) (p q s : Complex) : Complex :=
  aCoeff h p * s ^ 6 + 2 * p * s ^ 3 - q * s ^ 2 + 2 * s

/-- The derivative of every inverse polynomial is nonzero, since its constant
coefficient is `2`. -/
theorem omega_derivative_ne_zero
    (h : Complex[X]) (p q r : Complex) :
    derivative (omega h p q r) ≠ 0 := by
  intro hzero
  have hcoeff := congrArg (fun f : Complex[X] => f.coeff 0) hzero
  simp [omega] at hcoeff

/-- Critical values of `r` obtained from roots of the derivative.  The
derivative is independent of `r`, so it is enough to use `r = 0` here. -/
def omegaBadRValues (h : Complex[X]) (p q : Complex) : Set Complex :=
  omegaCriticalValue h p q ''
    (derivative (omega h p q 0)).rootSet Complex

theorem omegaBadRValues_finite
    (h : Complex[X]) (p q : Complex) :
    (omegaBadRValues h p q).Finite := by
  exact (Polynomial.rootSet_finite _ _).image _

/-- Outside the finite critical-value set, every inverse root is simple. -/
theorem omega_roots_simple_of_r_not_mem
    (h : Complex[X]) (p q r : Complex)
    (hr : r ∉ omegaBadRValues h p q) :
    ∀ s : Complex, (omega h p q r).eval s = 0 →
      (derivative (omega h p q r)).eval s ≠ 0 := by
  intro s hs hderiv
  have hderivZero : (derivative (omega h p q 0)).eval s = 0 := by
    rw [omega_derivative_eval] at hderiv ⊢
    exact hderiv
  have hsRoot :
      s ∈ (derivative (omega h p q 0)).rootSet Complex := by
    rw [Polynomial.mem_rootSet_of_ne (omega_derivative_ne_zero h p q 0)]
    exact hderivZero
  apply hr
  refine ⟨s, hsRoot, ?_⟩
  rw [omega_eval] at hs
  exact sub_eq_zero.mp (by simpa [omegaCriticalValue] using hs)

/-- Over `Complex`, the pointwise simple-root condition implies separability. -/
theorem separable_of_derivative_ne_zero_at_roots
    {f : Complex[X]}
    (hsimple : ∀ s : Complex, f.eval s = 0 → f.derivative.eval s ≠ 0) :
    f.Separable := by
  rw [Polynomial.separable_def]
  rw [Polynomial.isCoprime_iff_aeval_ne_zero_of_isAlgClosed
    (k := Complex) Complex]
  intro s
  by_cases hs : f.eval s = 0
  · right
    simpa [Polynomial.aeval_def] using hsimple s hs
  · left
    simpa [Polynomial.aeval_def] using hs

theorem omega_separable_of_r_not_mem
    (h : Complex[X]) (p q r : Complex)
    (hr : r ∉ omegaBadRValues h p q) :
    (omega h p q r).Separable :=
  separable_of_derivative_ne_zero_at_roots
    (omega_roots_simple_of_r_not_mem h p q r hr)

/-- A target is good when its first coordinate avoids the leading locus and
all roots of its inverse polynomial are simple. -/
def IsGoodTarget (h : Complex[X]) (b : C3) : Prop :=
  b 0 ≠ 0 ∧
    aCoeff h (b 0) ≠ 0 ∧
      ∀ s : Complex, (omega h (b 0) (b 1) (b 2)).eval s = 0 →
        (derivative (omega h (b 0) (b 1) (b 2))).eval s ≠ 0

theorem IsGoodTarget.omega_separable
    {h : Complex[X]} {b : C3} (hb : IsGoodTarget h b) :
    (omega h (b 0) (b 1) (b 2)).Separable :=
  separable_of_derivative_ne_zero_at_roots hb.2.2

/-- Every target has arbitrarily close good targets.  The construction keeps
the middle coordinate fixed. -/
theorem exists_goodTarget_dist_lt
    (h : Complex[X]) (hh : h ≠ 0) (b : C3)
    (ε : Real) (hε : 0 < ε) :
    ∃ b' : C3, dist b' b < ε ∧ IsGoodTarget h b' := by
  let exceptionalP : Set Complex := {0} ∪ h.rootSet Complex
  have hExceptionalP : exceptionalP.Finite :=
    Set.finite_singleton 0 |>.union (Polynomial.rootSet_finite _ _)
  have hDenseP : Dense ((Set.univ : Set Complex) \ exceptionalP) :=
    dense_univ.sdiff_finite hExceptionalP
  rcases (Metric.dense_iff.mp hDenseP (b 0) ε hε) with
    ⟨p, hpBall, hpOutside⟩
  have hpNotExceptional : p ∉ exceptionalP := hpOutside.2
  have hpZero : p ≠ 0 := by
    intro hp
    apply hpNotExceptional
    exact Or.inl hp
  have hpNotRoot : p ∉ h.rootSet Complex := by
    intro hp
    apply hpNotExceptional
    exact Or.inr hp
  have hpEval : h.eval p ≠ 0 := by
    intro hp
    exact hpNotRoot ((Polynomial.mem_rootSet_of_ne hh).mpr hp)
  have hpCoeff : aCoeff h p ≠ 0 := by
    intro hp
    rcases (aCoeff_eq_zero_iff h p).mp hp with hp | hp
    · exact hpZero hp
    · exact hpEval hp
  have hDenseR : Dense ((Set.univ : Set Complex) \
      omegaBadRValues h p (b 1)) :=
    dense_univ.sdiff_finite (omegaBadRValues_finite h p (b 1))
  rcases (Metric.dense_iff.mp hDenseR (b 2) ε hε) with
    ⟨r, hrBall, hrOutside⟩
  let b' : C3 := ![p, b 1, r]
  refine ⟨b', ?_, ?_⟩
  · rw [dist_pi_lt_iff hε]
    intro i
    fin_cases i
    · simpa [b'] using hpBall
    · simp [b', hε]
    · simpa [b'] using hrBall
  · refine ⟨?_, ?_, ?_⟩
    · simpa [b'] using hpZero
    · simpa [b'] using hpCoeff
    · simpa [b'] using
        omega_roots_simple_of_r_not_mem h p (b 1) r hrOutside.2

/-- The set of good targets is dense in the target affine space. -/
theorem goodTarget_dense (h : Complex[X]) (hh : h ≠ 0) :
    Dense {b : C3 | IsGoodTarget h b} := by
  rw [Metric.dense_iff]
  intro b ε hε
  rcases exists_goodTarget_dist_lt h hh b ε hε with
    ⟨b', hb'Dist, hb'Good⟩
  exact ⟨b', hb'Dist, hb'Good⟩

end

end DegreeSixKeller
