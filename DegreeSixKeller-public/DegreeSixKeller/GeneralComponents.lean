import DegreeSixKeller.ComponentCounting
import DegreeSixKeller.GeneralNonproperness

/-!
# Irreducible components for the general degree-six family

The vertical components are indexed by the distinct nonzero roots of `h`,
together with the always-present parameter `0`.  Multiplicity in `h.roots`
is deliberately erased by `Multiset.toFinset`.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Polynomial Set Topology
open scoped Polynomial

noncomputable section

/-- Distinct nonzero roots of the deformation polynomial. -/
def nonzeroRoots (h : Complex[X]) : Finset Complex :=
  h.roots.toFinset.filter (fun alpha ↦ alpha ≠ 0)

/-- Membership in `nonzeroRoots` is evaluation-theoretic for a nonzero
polynomial. -/
theorem mem_nonzeroRoots_iff
    {h : Complex[X]} (hh : h ≠ 0) {alpha : Complex} :
    alpha ∈ nonzeroRoots h ↔ alpha ≠ 0 ∧ h.eval alpha = 0 := by
  simp [nonzeroRoots, Polynomial.mem_roots hh, Polynomial.IsRoot,
    and_comm]

/-- The root at zero is excluded from the index set. -/
@[simp]
theorem zero_not_mem_nonzeroRoots (h : Complex[X]) :
    0 ∉ nonzeroRoots h := by
  simp [nonzeroRoots]

/-- A nonzero constant has no vertical roots. -/
@[simp]
theorem nonzeroRoots_C (c : Complex) :
    nonzeroRoots (C c) = ∅ := by
  ext alpha
  simp [nonzeroRoots, Polynomial.roots_C]

/-- A pure zero root contributes no nonzero vertical component. -/
@[simp]
theorem nonzeroRoots_X :
    nonzeroRoots (X : Complex[X]) = ∅ := by
  ext alpha
  simp [nonzeroRoots, Polynomial.roots_X]

/-- Adding any multiplicity of the root at zero does not change the nonzero
root index. -/
theorem nonzeroRoots_X_mul
    {h : Complex[X]} (hh : h ≠ 0) :
    nonzeroRoots (X * h) = nonzeroRoots h := by
  ext alpha
  rw [mem_nonzeroRoots_iff (mul_ne_zero X_ne_zero hh),
    mem_nonzeroRoots_iff hh]
  constructor
  · rintro ⟨halpha, hroot⟩
    refine ⟨halpha, ?_⟩
    simpa [halpha] using hroot
  · rintro ⟨halpha, hroot⟩
    exact ⟨halpha, by simp [hroot]⟩

/-- Repeating every root changes multiplicities but not the component index
set. -/
theorem nonzeroRoots_pow
    (h : Complex[X]) {n : Nat} (hn : n ≠ 0) :
    nonzeroRoots (h ^ n) = nonzeroRoots h := by
  ext alpha
  simp [nonzeroRoots, Polynomial.roots_pow, hn]

/-- Vertical parameter values: zero and the distinct nonzero roots of `h`. -/
def verticalParameters (h : Complex[X]) : Finset Complex :=
  insert 0 (nonzeroRoots h)

/-- Vertical hyperplanes in the reduced nonproperness set. -/
def verticalComponents (h : Complex[X]) : Finset (Set C3) :=
  (verticalParameters h).image pHyperplane

theorem mem_verticalComponents_iff
    {h : Complex[X]} {A : Set C3} :
    A ∈ verticalComponents h ↔
      ∃ alpha ∈ verticalParameters h, A = pHyperplane alpha := by
  simp [verticalComponents, eq_comm]

/-- Distinct vertical parameters give distinct components, so the vertical
component count is one plus the number of distinct nonzero roots. -/
theorem verticalComponents_card (h : Complex[X]) :
    (verticalComponents h).card = 1 + (nonzeroRoots h).card := by
  rw [verticalComponents, Finset.card_image_iff.mpr]
  · simp [verticalParameters, Nat.add_comm]
  · intro alpha _ beta _ hEq
    exact pHyperplane_eq_iff.mp hEq

/-- The leading-coefficient locus is the union of its vertical
hyperplanes. -/
theorem zeroLocusOfP6MulH_eq_sUnion_verticalComponents
    (h : Complex[X]) (hh : h ≠ 0) :
    zeroLocusOfP6MulH h =
      ⋃₀ (verticalComponents h : Set (Set C3)) := by
  ext b
  constructor
  · intro hb
    change b 0 ^ 6 * h.eval (b 0) = 0 at hb
    rcases mul_eq_zero.mp hb with hp | hhroot
    · have hp0 : b 0 = 0 :=
        (pow_eq_zero_iff (by norm_num : 6 ≠ 0)).mp hp
      apply Set.mem_sUnion_of_mem
        (show b ∈ pHyperplane 0 by simpa [pHyperplane] using hp0)
      rw [Finset.mem_coe, mem_verticalComponents_iff]
      exact ⟨0, by simp [verticalParameters], rfl⟩
    · by_cases hp0 : b 0 = 0
      · apply Set.mem_sUnion_of_mem
          (show b ∈ pHyperplane 0 by simpa [pHyperplane] using hp0)
        rw [Finset.mem_coe, mem_verticalComponents_iff]
        exact ⟨0, by simp [verticalParameters], rfl⟩
      · have hroot : b 0 ∈ nonzeroRoots h :=
          (mem_nonzeroRoots_iff hh).2 ⟨hp0, hhroot⟩
        apply Set.mem_sUnion_of_mem
          (show b ∈ pHyperplane (b 0) by rfl)
        rw [Finset.mem_coe, mem_verticalComponents_iff]
        exact ⟨b 0, by simp [verticalParameters, hroot], rfl⟩
  · intro hb
    rcases Set.mem_sUnion.mp hb with ⟨A, hA, hbA⟩
    rw [Finset.mem_coe, mem_verticalComponents_iff] at hA
    rcases hA with ⟨alpha, halpha, rfl⟩
    change b 0 ^ 6 * h.eval (b 0) = 0
    have hbAlpha : b 0 = alpha := hbA
    rcases Finset.mem_insert.mp halpha with rfl | halpha
    · simp [hbAlpha]
    · have hroot := (mem_nonzeroRoots_iff hh).1 halpha
      rw [hbAlpha, hroot.2]
      simp

/-! ## The full finite component family -/

/-- The finite component together with all vertical components. -/
def generalComponents (h : Complex[X]) : Finset (Set C3) :=
  insert (finiteComponent h) (verticalComponents h)

theorem finiteComponent_not_mem_verticalComponents (h : Complex[X]) :
    finiteComponent h ∉ verticalComponents h := by
  intro hmem
  rw [mem_verticalComponents_iff] at hmem
  rcases hmem with ⟨alpha, _halpha, hEq⟩
  exact finiteComponent_ne_pHyperplane h alpha hEq

theorem generalComponents_card (h : Complex[X]) :
    (generalComponents h).card = 2 + (nonzeroRoots h).card := by
  rw [generalComponents,
    Finset.card_insert_of_notMem (finiteComponent_not_mem_verticalComponents h),
    verticalComponents_card]
  omega

theorem verticalComponent_isClosed
    (h : Complex[X]) {A : Set C3} (hA : A ∈ verticalComponents h) :
    IsClosed (zariskiLift A) := by
  rw [mem_verticalComponents_iff] at hA
  rcases hA with ⟨alpha, _halpha, rfl⟩
  exact pHyperplane_isClosed alpha

theorem verticalComponent_isIrreducible
    (h : Complex[X]) {A : Set C3} (hA : A ∈ verticalComponents h) :
    IsIrreducible (zariskiLift A) := by
  rw [mem_verticalComponents_iff] at hA
  rcases hA with ⟨alpha, _halpha, rfl⟩
  exact pHyperplane_isIrreducible alpha

theorem generalComponent_isClosed
    (h : Complex[X]) {A : Set C3} (hA : A ∈ generalComponents h) :
    IsClosed (zariskiLift A) := by
  rw [generalComponents, Finset.mem_insert] at hA
  rcases hA with rfl | hA
  · exact finiteComponent_isClosed h
  · exact verticalComponent_isClosed h hA

theorem generalComponent_isIrreducible
    (h : Complex[X]) {A : Set C3} (hA : A ∈ generalComponents h) :
    IsIrreducible (zariskiLift A) := by
  rw [generalComponents, Finset.mem_insert] at hA
  rcases hA with rfl | hA
  · exact finiteComponent_isIrreducible h
  · exact verticalComponent_isIrreducible h hA

theorem generalComponents_sUnion
    (h : Complex[X]) (hh : h ≠ 0) :
    ⋃₀ (generalComponents h : Set (Set C3)) =
      NonpropernessSet (Fh h) := by
  calc
    ⋃₀ (generalComponents h : Set (Set C3)) =
        finiteComponent h ∪
          ⋃₀ (verticalComponents h : Set (Set C3)) := by
      ext b
      simp [generalComponents]
    _ = finiteComponent h ∪ zeroLocusOfP6MulH h := by
      rw [← zeroLocusOfP6MulH_eq_sUnion_verticalComponents h hh]
    _ = generalReducedCandidate h := rfl
    _ = NonpropernessSet (Fh h) :=
      (Fh_reducedNonpropernessEquality h hh).symm

theorem generalComponents_irredundant
    (h : Complex[X]) :
    ∀ A ∈ generalComponents h, ∀ B ∈ generalComponents h,
      A ⊆ B → A = B := by
  intro A hA B hB hAB
  rw [generalComponents, Finset.mem_insert] at hA hB
  rcases hA with rfl | hA
  · rcases hB with rfl | hB
    · rfl
    · rw [mem_verticalComponents_iff] at hB
      rcases hB with ⟨beta, _hbeta, rfl⟩
      exact False.elim (finiteComponent_not_subset_pHyperplane h beta hAB)
  · rw [mem_verticalComponents_iff] at hA
    rcases hA with ⟨alpha, _halpha, rfl⟩
    rcases hB with rfl | hB
    · exact False.elim ((finiteComponent_noVerticalHyperplane h alpha) hAB)
    · rw [mem_verticalComponents_iff] at hB
      rcases hB with ⟨beta, _hbeta, rfl⟩
      exact pHyperplane_eq_iff.mpr (pHyperplane_subset_iff.mp hAB)

/-- The nonproperness set has one finite component, one zero hyperplane, and
one further vertical component for each distinct nonzero root of `h`. -/
theorem Fh_componentCount
    (h : Complex[X]) (hh : h ≠ 0) :
    algebraicComponentCount (NonpropernessSet (Fh h)) =
      2 + (nonzeroRoots h).card := by
  have hCount := algebraicComponentCount_eq_finset_card
    (NonpropernessSet (Fh h)) (generalComponents h)
    (fun A hA ↦ generalComponent_isClosed h hA)
    (fun A hA ↦ generalComponent_isIrreducible h hA)
    (generalComponents_sUnion h hh)
    (generalComponents_irredundant h)
  rw [hCount]
  exact generalComponents_card h

/-! ## Semantic checks -/

/-- A nonzero constant has exactly the finite component and the zero
hyperplane. -/
theorem Fh_componentCount_C
    (c : Complex) (hc : c ≠ 0) :
    algebraicComponentCount (NonpropernessSet (Fh (C c))) = 2 := by
  rw [Fh_componentCount (C c) (C_ne_zero.mpr hc), nonzeroRoots_C]
  simp

/-- A deformation whose only root is zero still has exactly two
components. -/
theorem Fh_componentCount_X :
    algebraicComponentCount (NonpropernessSet (Fh (X : Complex[X]))) = 2 := by
  rw [Fh_componentCount (X : Complex[X]) X_ne_zero, nonzeroRoots_X]
  simp

/-- Adding a root at zero does not change the component-count formula. -/
theorem Fh_componentCount_X_mul
    (h : Complex[X]) (hh : h ≠ 0) :
    algebraicComponentCount (NonpropernessSet (Fh (X * h))) =
      2 + (nonzeroRoots h).card := by
  rw [Fh_componentCount (X * h) (mul_ne_zero X_ne_zero hh),
    nonzeroRoots_X_mul hh]

/-- Repeated roots count once: taking a positive power does not change the
number of components. -/
theorem Fh_componentCount_pow
    (h : Complex[X]) (hh : h ≠ 0) {n : Nat} (hn : n ≠ 0) :
    algebraicComponentCount (NonpropernessSet (Fh (h ^ n))) =
      2 + (nonzeroRoots h).card := by
  rw [Fh_componentCount (h ^ n) (pow_ne_zero n hh),
    nonzeroRoots_pow h hn]

end

end DegreeSixKeller
