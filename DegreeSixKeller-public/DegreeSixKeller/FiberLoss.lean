import DegreeSixKeller.GenericFiberSix
import DegreeSixKeller.GoodTargetDensity
import DegreeSixKeller.KellerLocalHomeomorph
import DegreeSixKeller.ReducedNonproperness
import Mathlib.Analysis.Normed.Module.Connected
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Topology.Algebra.MvPolynomial
import Mathlib.Topology.Homotopy.Lifting
import Mathlib.Topology.Maps.Proper.CompactlyGenerated
import Mathlib.Topology.MetricSpace.Bounded

/-!
# Fiber loss away from the nonproperness set

This file isolates the analytic compactness input for the general fiber-loss
theorem.  Negating the sequential definition of `NonpropernessSet` gives a
uniform norm bound on the inverse image of some Euclidean neighborhood.  In
particular, every fiber away from the nonproperness set is compact.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Filter Polynomial Set Topology
open scoped Polynomial

noncomputable section

/-- Evaluation of a polynomial coordinate map is Euclidean-continuous. -/
theorem continuous_polynomialMap_eval (F : PolynomialMap3) :
    Continuous (PolynomialMap3.eval F) := by
  apply continuous_pi
  intro i
  exact (F i).continuous_eval

/-- If `b` is not a nonproper value, the inverse image of a sufficiently
small ball around `b` has a uniform source norm bound.  This is the exact
local properness consequence of the project's sequential definition. -/
theorem exists_local_source_norm_bound_of_not_mem_nonproperness
    {F : C3 → C3} {b : C3}
    (hb : b ∉ NonpropernessSet F) :
    ∃ ε : ℝ, 0 < ε ∧ ∃ R : ℝ, ∀ u : C3,
      dist (F u) b < ε → norm u ≤ R := by
  by_contra hBound
  push Not at hBound
  have hUnbounded (n : ℕ) :
      ∃ u : C3,
        dist (F u) b < 1 / ((n : ℝ) + 1) ∧ (n : ℝ) < norm u := by
    exact hBound (1 / ((n : ℝ) + 1)) (by positivity) (n : ℝ)
  choose u huDist huNorm using hUnbounded
  apply hb
  refine ⟨u, ?_, ?_⟩
  · exact tendsto_atTop_mono (fun n ↦ (huNorm n).le)
      tendsto_natCast_atTop_atTop
  · apply tendsto_iff_dist_tendsto_zero.mpr
    exact squeeze_zero (fun n ↦ dist_nonneg)
      (fun n ↦ (huDist n).le)
      tendsto_one_div_add_atTop_nhds_zero_nat

/-- Sequential nonproperness is exactly failure of a uniform source bound
over every sufficiently small target neighborhood. -/
theorem not_mem_nonproperness_iff_exists_local_source_norm_bound
    {F : C3 → C3} {b : C3} :
    b ∉ NonpropernessSet F ↔
      ∃ ε : ℝ, 0 < ε ∧ ∃ R : ℝ, ∀ u : C3,
        dist (F u) b < ε → norm u ≤ R := by
  constructor
  · exact exists_local_source_norm_bound_of_not_mem_nonproperness
  · rintro ⟨ε, hε, R, hR⟩ ⟨u, huEscapes, huTendsto⟩
    have hNear : ∀ᶠ n in atTop, dist (F (u n)) b < ε := by
      have hBall := huTendsto.eventually (Metric.ball_mem_nhds b hε)
      simpa [Metric.mem_ball, dist_comm] using hBall
    have hFar : ∀ᶠ n in atTop, R < norm (u n) :=
      huEscapes.eventually_gt_atTop R
    obtain ⟨n, hnNear, hnFar⟩ := (hNear.and hFar).exists
    exact (not_lt_of_ge (hR (u n) hnNear)) hnFar

/-- Every point fiber of a polynomial map away from its nonproperness set is
compact in the Euclidean source. -/
theorem polynomialMap_fiber_isCompact_of_not_mem_nonproperness
    {F : PolynomialMap3} {b : C3}
    (hb : b ∉ NonpropernessSet (PolynomialMap3.eval F)) :
    IsCompact {u : C3 | PolynomialMap3.eval F u = b} := by
  obtain ⟨ε, hε, R, hR⟩ :=
    exists_local_source_norm_bound_of_not_mem_nonproperness hb
  rw [Metric.isCompact_iff_isClosed_bounded]
  constructor
  · exact isClosed_singleton.preimage (continuous_polynomialMap_eval F)
  · rw [isBounded_iff_forall_norm_le]
    refine ⟨R, fun u hu ↦ hR u ?_⟩
    change PolynomialMap3.eval F u = b at hu
    rw [hu, dist_self]
    exact hε

/-- Away from the nonproperness set, a whole closed target ball has compact
inverse image.  This is the local properness input needed by a covering-space
proof of fiber constancy. -/
theorem exists_compact_closedBall_preimage_of_not_mem_nonproperness
    {F : PolynomialMap3} {b : C3}
    (hb : b ∉ NonpropernessSet (PolynomialMap3.eval F)) :
    ∃ δ : ℝ, 0 < δ ∧
      IsCompact (PolynomialMap3.eval F ⁻¹' Metric.closedBall b δ) := by
  obtain ⟨ε, hε, R, hR⟩ :=
    exists_local_source_norm_bound_of_not_mem_nonproperness hb
  refine ⟨ε / 2, half_pos hε, ?_⟩
  rw [Metric.isCompact_iff_isClosed_bounded]
  constructor
  · exact Metric.isClosed_closedBall.preimage (continuous_polynomialMap_eval F)
  · rw [isBounded_iff_forall_norm_le]
    refine ⟨R, fun u hu ↦ hR u ?_⟩
    exact lt_of_le_of_lt hu (half_lt_self hε)

/-- Restricting a Keller polynomial map over a sufficiently small target
ball gives a proper map.  The use of an open ball here is important: its
source and target subtype inclusions are local homeomorphisms, while the
compact preimage of the corresponding closed ball supplies properness. -/
private theorem restrictedBall_isProperMap
    {F : PolynomialMap3} {b : C3} {δ : ℝ}
    (hcompact : IsCompact
      (PolynomialMap3.eval F ⁻¹' Metric.closedBall b δ)) :
    IsProperMap
      ((Metric.ball b δ).restrictPreimage (PolynomialMap3.eval F)) := by
  rw [isProperMap_iff_isCompact_preimage]
  constructor
  · exact (continuous_polynomialMap_eval F).restrictPreimage
  · intro K hK
    rw [Topology.IsEmbedding.subtypeVal.isCompact_iff,
      image_val_preimage_restrictPreimage]
    have hImageCompact : IsCompact (Subtype.val '' K) :=
      hK.image continuous_subtype_val
    apply hcompact.of_isClosed_subset
    · exact hImageCompact.isClosed.preimage (continuous_polynomialMap_eval F)
    · intro u hu
      rcases hu with ⟨y, hy, huy⟩
      change PolynomialMap3.eval F u ∈ Metric.closedBall b δ
      rw [← huy]
      exact Metric.ball_subset_closedBall y.prop

/-- Restriction over an open target ball preserves the local-homeomorphism
property of a Keller polynomial map. -/
private theorem restrictedBall_isLocalHomeomorph
    {F : PolynomialMap3} (hKeller : IsKeller F) (b : C3) (δ : ℝ) :
    IsLocalHomeomorph
      ((Metric.ball b δ).restrictPreimage (PolynomialMap3.eval F)) := by
  let s : Set C3 := Metric.ball b δ
  let f : C3 → C3 := PolynomialMap3.eval F
  let g := s.restrictPreimage f
  have hfcont : Continuous f := continuous_polynomialMap_eval F
  have hsopen : IsOpen s := Metric.isOpen_ball
  have hpreopen : IsOpen (f ⁻¹' s) := hsopen.preimage hfcont
  have hsource : IsLocalHomeomorph ((↑) : (f ⁻¹' s) → C3) :=
    hpreopen.isOpenEmbedding_subtypeVal.isLocalHomeomorph
  have htarget : IsLocalHomeomorph ((↑) : s → C3) :=
    hsopen.isOpenEmbedding_subtypeVal.isLocalHomeomorph
  have hcomp : IsLocalHomeomorph (((↑) : s → C3) ∘ g) := by
    simpa [f, g, s, Function.comp_def] using
      (isLocalHomeomorph_polynomialMap_eval hKeller).comp hsource
  have hgcont : Continuous g := hfcont.restrictPreimage
  simpa [g, s, f] using hcomp.of_comp htarget hgcont

/-- A point fiber of a restriction over `s` is canonically equivalent to
the corresponding point fiber of the original map. -/
private def sourceFiberEquivRestrictPreimage
    {α β : Type*} {f : α → β} {s : Set β} {z : β} (hz : z ∈ s) :
    {u : α // f u = z} ≃
      (s.restrictPreimage f ⁻¹' {(⟨z, hz⟩ : s)}) where
  toFun u := ⟨⟨u, by
      change f (u : α) ∈ s
      rw [u.prop]
      exact hz⟩, by
    change s.restrictPreimage f ⟨u, by
      change f (u : α) ∈ s
      rw [u.prop]
      exact hz⟩ = ⟨z, hz⟩
    apply Subtype.ext
    exact u.prop⟩
  invFun u := ⟨u.1.1, by
    have hu : s.restrictPreimage f u.1 = ⟨z, hz⟩ := by
      simpa only [Set.mem_preimage, Set.mem_singleton_iff] using u.2
    exact congrArg Subtype.val hu⟩
  left_inv u := by ext; rfl
  right_inv u := by ext; rfl

/-- On some ball about every proper target value of a Keller polynomial
map, all fibers have the same cardinality.  This is the analytic
covering-space layer of fiber loss; it does not use function fields. -/
theorem exists_ball_fiber_ncard_eq_of_not_mem_nonproperness
    {F : PolynomialMap3} (hKeller : IsKeller F) {b : C3}
    (hb : b ∉ NonpropernessSet (PolynomialMap3.eval F)) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ y : C3, y ∈ Metric.ball b δ →
      Set.ncard {u : C3 | PolynomialMap3.eval F u = y} =
        Set.ncard {u : C3 | PolynomialMap3.eval F u = b} := by
  obtain ⟨δ, hδ, hcompact⟩ :=
    exists_compact_closedBall_preimage_of_not_mem_nonproperness hb
  refine ⟨δ, hδ, ?_⟩
  let s : Set C3 := Metric.ball b δ
  let f : C3 → C3 := PolynomialMap3.eval F
  let g := s.restrictPreimage f
  have hproper : IsProperMap g := by
    simpa [g, s, f] using restrictedBall_isProperMap hcompact
  have hlocal : IsLocalHomeomorph g := by
    simpa [g, s, f] using restrictedBall_isLocalHomeomorph hKeller b δ
  have hfiberFinite (x : s) : (g ⁻¹' {x}).Finite := by
    apply (hproper.isCompact_preimage isCompact_singleton).finite
    apply IsDiscrete.of_openPartialHomeomorph g subset_rfl
    intro e _he
    obtain ⟨φ, heφ, hφ⟩ := hlocal e
    exact ⟨φ, heφ, hφ.symm⟩
  have hcover : IsCoveringMap g := by
    rw [isCoveringMap_iff_isCoveringMapOn_univ]
    exact hproper.isClosedMap.isCoveringMapOn_of_isLocalHomeomorphOn
      (fun x _hx ↦ hfiberFinite x) hlocal.isLocalHomeomorphOn
  have hbBall : b ∈ s := by simpa [s] using hδ
  intro y hy
  have hyBall : y ∈ s := by simpa [s] using hy
  have hjoined : Joined (⟨b, hbBall⟩ : s) ⟨y, hyBall⟩ := by
    have hjoinedIn : JoinedIn (Metric.ball b δ) b y :=
      (Metric.isPathConnected_ball (x := b) (r := δ) hδ).joinedIn
        b (Metric.mem_ball_self hδ) y hy
    simpa [s] using hjoinedIn.joined_subtype
  let γ : Path (⟨b, hbBall⟩ : s) ⟨y, hyBall⟩ := hjoined.somePath
  let monodromyEquiv :
      (g ⁻¹' {(⟨b, hbBall⟩ : s)}) ≃
        (g ⁻¹' {(⟨y, hyBall⟩ : s)}) :=
    Equiv.ofBijective
      (hcover.monodromy (Path.Homotopic.Quotient.mk γ))
      (hcover.monodromy_bijective (Path.Homotopic.Quotient.mk γ))
  exact Set.ncard_congr'
    ((sourceFiberEquivRestrictPreimage (f := f) hyBall).trans
      (monodromyEquiv.symm.trans
        (sourceFiberEquivRestrictPreimage (f := f) hbBall).symm))

/-- Away from the nonproperness set, every fiber of a nontrivial member of
the degree-six Keller family has all six generic points. -/
theorem Fh_fiber_ncard_eq_six_of_not_mem_nonproperness
    (h : Complex[X]) (hh : h ≠ 0) {b : C3}
    (hb : b ∉ NonpropernessSet (Fh h)) :
    Set.ncard {u : C3 | Fh h u = b} = 6 := by
  have hEval : PolynomialMap3.eval (FhPolynomial h) = Fh h :=
    funext (eval_FhPolynomial h)
  have hbPolynomial :
      b ∉ NonpropernessSet (PolynomialMap3.eval (FhPolynomial h)) := by
    rwa [hEval]
  obtain ⟨δ, hδ, hFiberConst⟩ :=
    exists_ball_fiber_ncard_eq_of_not_mem_nonproperness
      (Fh_isKeller h) hbPolynomial
  obtain ⟨b', hb'Dist, hb'Good⟩ :=
    exists_goodTarget_dist_lt h hh b δ hδ
  have hb'Ball : b' ∈ Metric.ball b δ := by
    simpa [Metric.mem_ball] using hb'Dist
  have hEq :
      Set.ncard {u : C3 | Fh h u = b'} =
        Set.ncard {u : C3 | Fh h u = b} := by
    simpa only [hEval] using hFiberConst b' hb'Ball
  calc
    Set.ncard {u : C3 | Fh h u = b} =
        Set.ncard {u : C3 | Fh h u = b'} := hEq.symm
    _ = 6 := fiber_ncard_eq_six h b' hb'Good.1 hb'Good.2.1
      hb'Good.omega_separable

/-- Strict loss from the generic six-point fiber forces nonproperness. -/
theorem mem_nonproperness_of_Fh_fiber_ncard_lt_six
    (h : Complex[X]) (hh : h ≠ 0) {b : C3}
    (hFiber : Set.ncard {u : C3 | Fh h u = b} < 6) :
    b ∈ NonpropernessSet (Fh h) := by
  by_contra hb
  have hSix := Fh_fiber_ncard_eq_six_of_not_mem_nonproperness h hh hb
  omega

end

end DegreeSixKeller
