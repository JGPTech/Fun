import DegreeSixKeller.ComponentCounting
import DegreeSixKeller.EliminationCertificates
import DegreeSixKeller.ReducedNonproperness

/-!
# Pair-specific geometric loci for CEX-004 and CEX-006

The public definitions in this file are the exact loci occurring in the human
proof.  In particular, `finiteComponent h` is the Zariski closure of the
finite-multiple-root parametrization and the reduced candidates have the
displayed vertical hyperplanes for the two examples.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Filter Polynomial Set Topology

noncomputable section

/-- The second coordinate of the finite-multiple-root parametrization. -/
def criticalQ (h : Complex[X]) (p s : Complex) : Complex :=
  p ^ 6 * h.eval p * s ^ 4 + 3 * p * s + 1 / s

/-- The third coordinate of the finite-multiple-root parametrization. -/
def criticalR (h : Complex[X]) (p s : Complex) : Complex :=
  s - p * s ^ 3 - (2 / 3 : Complex) * p ^ 6 * h.eval p * s ^ 6

/-- The target parametrization obtained by solving
`Omega = derivative Omega = 0`. -/
def criticalTarget (h : Complex[X]) (p s : Complex) : C3 :=
  ![p, criticalQ h p s, criticalR h p s]

@[simp]
theorem criticalTarget_zero (h : Complex[X]) (p s : Complex) :
    criticalTarget h p s 0 = p := by
  simp [criticalTarget]

@[simp]
theorem criticalTarget_one (h : Complex[X]) (p s : Complex) :
    criticalTarget h p s 1 = criticalQ h p s := by
  simp [criticalTarget]

@[simp]
theorem criticalTarget_two (h : Complex[X]) (p s : Complex) :
    criticalTarget h p s 2 = criticalR h p s := by
  simp [criticalTarget]

theorem omega_criticalTarget (h : Complex[X]) (p s : Complex)
    (_hs : s ≠ 0) :
    (omega h
      (criticalTarget h p s 0)
      (criticalTarget h p s 1)
      (criticalTarget h p s 2)).eval s = 0 := by
  simp [criticalQ, criticalR, aCoeff]
  field_simp [_hs]
  ring

theorem omega_derivative_criticalTarget (h : Complex[X]) (p s : Complex)
    (hs : s ≠ 0) :
    (derivative (omega h
      (criticalTarget h p s 0)
      (criticalTarget h p s 1)
      (criticalTarget h p s 2))).eval s = 0 := by
  rw [omega_derivative_eval]
  simp [criticalQ]
  field_simp [hs]
  ring

/-- Targets directly produced by the finite-multiple-root parametrization. -/
def criticalImage (h : Complex[X]) : Set C3 :=
  {b | ∃ p s : Complex, s ≠ 0 ∧ b = criticalTarget h p s}

/-- Zariski closure of the finite-multiple-root parametrization. -/
def finiteComponent (h : Complex[X]) : Set C3 :=
  {b | toZariskiC3 b ∈ closure (zariskiLift (criticalImage h))}

theorem zariskiLift_finiteComponent (h : Complex[X]) :
    zariskiLift (finiteComponent h) =
      closure (zariskiLift (criticalImage h)) := by
  ext b
  rfl

theorem finiteComponent_isClosed (h : Complex[X]) :
    IsClosed (zariskiLift (finiteComponent h)) := by
  rw [zariskiLift_finiteComponent]
  exact isClosed_closure

theorem finiteComponent_isIrreducible_of_criticalImage
    (h : Complex[X])
    (hImage : IsIrreducible (zariskiLift (criticalImage h))) :
    IsIrreducible (zariskiLift (finiteComponent h)) := by
  rw [zariskiLift_finiteComponent]
  exact hImage.closure

/-- A vertical target hyperplane `p = alpha`. -/
def pHyperplane (alpha : Complex) : Set C3 :=
  {b | b 0 = alpha}

theorem zariskiLift_pHyperplane (alpha : Complex) :
    zariskiLift (pHyperplane alpha) =
      zariskiLift
        (MvPolynomial.zeroLocus Complex (firstCoordinateIdeal alpha)) := by
  rw [zeroLocus_firstCoordinateIdeal]
  rfl

theorem pHyperplane_isClosed (alpha : Complex) :
    IsClosed (zariskiLift (pHyperplane alpha)) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  have hSet : pHyperplane alpha =
      MvPolynomial.zeroLocus Complex (firstCoordinateIdeal alpha) := by
    exact (zeroLocus_firstCoordinateIdeal alpha).symm
  have hRaw : IsClosed (pHyperplane alpha) := by
    rw [hSet]
    exact (isClosed_affineZariski_iff (Fin 3)
      (MvPolynomial.zeroLocus Complex (firstCoordinateIdeal alpha))).mpr
        ⟨firstCoordinateIdeal alpha, rfl⟩
  exact hRaw.preimage
    (WithTopology.continuous_ofTopology (affineZariskiTopology (Fin 3)))

theorem pHyperplane_isIrreducible (alpha : Complex) :
    IsIrreducible (zariskiLift (pHyperplane alpha)) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  have hSet : pHyperplane alpha =
      MvPolynomial.zeroLocus Complex (firstCoordinateIdeal alpha) := by
    exact (zeroLocus_firstCoordinateIdeal alpha).symm
  have hRaw : IsIrreducible (pHyperplane alpha) := by
    rw [hSet]
    exact isIrreducible_zeroLocus_of_isPrime (Fin 3)
      (firstCoordinateIdeal alpha)
  have hImage := hRaw.image
    (WithTopology.toTopology (affineZariskiTopology (Fin 3)))
    (WithTopology.continuous_toTopology
      (affineZariskiTopology (Fin 3))).continuousOn
  rw [WithTopology.image_toTopology] at hImage
  exact hImage

theorem pHyperplane_eq_iff {alpha beta : Complex} :
    pHyperplane alpha = pHyperplane beta ↔ alpha = beta := by
  constructor
  · intro h
    let x : C3 := ![alpha, 0, 0]
    have hx : x ∈ pHyperplane alpha := by
      simp [x, pHyperplane]
    rw [h] at hx
    simpa [x, pHyperplane] using hx
  · rintro rfl
    rfl

theorem pHyperplane_subset_iff {alpha beta : Complex} :
    pHyperplane alpha ⊆ pHyperplane beta ↔ alpha = beta := by
  constructor
  · intro h
    let x : C3 := ![alpha, 0, 0]
    have hx : x ∈ pHyperplane alpha := by
      simp [x, pHyperplane]
    simpa [x, pHyperplane] using h hx
  · rintro rfl
    exact Set.Subset.rfl

/-- The parametrized finite component cannot be contained in a vertical
hyperplane because its first coordinate is the free parameter `p`. -/
theorem finiteComponent_not_subset_pHyperplane
    (h : Complex[X]) (alpha : Complex) :
    ¬ finiteComponent h ⊆ pHyperplane alpha := by
  intro hSubset
  let p : Complex := alpha + 1
  let b : C3 := criticalTarget h p 1
  have hOne : (1 : Complex) ≠ 0 := one_ne_zero
  have hbImage : b ∈ criticalImage h :=
    ⟨p, 1, hOne, rfl⟩
  have hbFinite : b ∈ finiteComponent h := by
    exact subset_closure hbImage
  have hbVertical := hSubset hbFinite
  have hpNe : p ≠ alpha := by
    dsimp [p]
    intro hp
    have : (1 : Complex) = 0 := by
      linear_combination hp
    exact one_ne_zero this
  exact hpNe (by simpa [b, pHyperplane] using hbVertical)

theorem leadingLocus004 :
    {b : C3 | aCoeff eta004 (b 0) = 0} =
      pHyperplane 0 ∪ pHyperplane (-1 / 4 : Complex) := by
  ext b
  rw [Set.mem_setOf_eq, aCoeff_eta004_zero_iff]
  rfl

theorem leadingLocus006 :
    {b : C3 | aCoeff eta006 (b 0) = 0} = pHyperplane 0 := by
  ext b
  rw [Set.mem_setOf_eq, aCoeff_eta006_zero_iff]
  rfl

/-- The exact reduced-set candidate for CEX-004. -/
def reducedCandidate004 : Set C3 :=
  finiteComponent eta004 ∪ pHyperplane 0 ∪
    pHyperplane (-1 / 4 : Complex)

/-- The exact reduced-set candidate for CEX-006. -/
def reducedCandidate006 : Set C3 :=
  finiteComponent eta006 ∪ pHyperplane 0

/-- Exact target proposition for the CEX-004 reduced nonproperness equality. -/
def ReducedNonpropernessEquality004 : Prop :=
  NonpropernessSet F004 = reducedCandidate004

/-- Exact target proposition for the CEX-006 reduced nonproperness equality. -/
def ReducedNonpropernessEquality006 : Prop :=
  NonpropernessSet F006 = reducedCandidate006

/-- Exact target proposition for irreducibility of the CEX-004 nonvertical
piece in the affine Zariski topology. -/
def FiniteComponentIrreducible004 : Prop :=
  IsIrreducible (zariskiLift (finiteComponent eta004))

/-- Exact target proposition for irreducibility of the CEX-006 nonvertical
piece in the affine Zariski topology. -/
def FiniteComponentIrreducible006 : Prop :=
  IsIrreducible (zariskiLift (finiteComponent eta006))

theorem cex004_reducedNonpropernessEquality_of_inclusions
    (hForward : NonpropernessSet F004 ⊆ reducedCandidate004)
    (hReverse : reducedCandidate004 ⊆ NonpropernessSet F004) :
    ReducedNonpropernessEquality004 :=
  Set.Subset.antisymm hForward hReverse

theorem cex006_reducedNonpropernessEquality_of_inclusions
    (hForward : NonpropernessSet F006 ⊆ reducedCandidate006)
    (hReverse : reducedCandidate006 ⊆ NonpropernessSet F006) :
    ReducedNonpropernessEquality006 :=
  Set.Subset.antisymm hForward hReverse

theorem cex004_finiteComponentIrreducible_of_criticalImage
    (hImage : IsIrreducible (zariskiLift (criticalImage eta004))) :
    FiniteComponentIrreducible004 :=
  finiteComponent_isIrreducible_of_criticalImage eta004 hImage

theorem cex006_finiteComponentIrreducible_of_criticalImage
    (hImage : IsIrreducible (zariskiLift (criticalImage eta006))) :
    FiniteComponentIrreducible006 :=
  finiteComponent_isIrreducible_of_criticalImage eta006 hImage

end

end DegreeSixKeller
