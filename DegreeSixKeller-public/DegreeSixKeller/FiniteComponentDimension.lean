import DegreeSixKeller.GeneralFiniteGeometry
import DegreeSixKeller.PolynomialAutomorphism
import Mathlib.RingTheory.Ideal.Quotient.Operations
import Mathlib.RingTheory.KrullDimension.NonZeroDivisors
import Mathlib.RingTheory.KrullDimension.Field
import Mathlib.RingTheory.KrullDimension.Polynomial
import Mathlib.RingTheory.Spectrum.Prime.Topology
import Mathlib.Topology.KrullDimension

namespace DegreeSixKeller

open Filter Ideal MvPolynomial Polynomial Set Topology
open scoped nonZeroDivisors

noncomputable section

private abbrev C3CoordinateRing := MvPolynomial (Fin 3) Complex

private abbrev IrreducibleClosed (T : Type*) [TopologicalSpace T] :=
  TopologicalSpace.IrreducibleCloseds T

/-- The ordinary affine points underlying an irreducible closed subset of
`ZariskiC3`. -/
private def affineRawCarrier (Z : IrreducibleClosed ZariskiC3) : Set C3 :=
  {x | toZariskiC3 x ∈ Z}

private theorem affineRawCarrier_image (Z : IrreducibleClosed ZariskiC3) :
    ofZariskiC3 '' (Z : Set ZariskiC3) = affineRawCarrier Z := by
  ext x
  constructor
  · rintro ⟨z, hz, rfl⟩
    exact hz
  · intro hx
    exact ⟨toZariskiC3 x, hx, rfl⟩

private theorem affineRawCarrier_isIrreducible
    (Z : IrreducibleClosed ZariskiC3) :
    @IsIrreducible C3 (affineZariskiTopology (Fin 3)) (affineRawCarrier Z) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  have hImage := Z.isIrreducible.image ofZariskiC3
    (WithTopology.continuous_ofTopology (affineZariskiTopology (Fin 3))).continuousOn
  rwa [affineRawCarrier_image] at hImage

private theorem affineRawCarrier_isClosed
    (Z : IrreducibleClosed ZariskiC3) :
    @IsClosed C3 (affineZariskiTopology (Fin 3)) (affineRawCarrier Z) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  exact Z.isClosed.preimage
    (WithTopology.continuous_toTopology (affineZariskiTopology (Fin 3)))

private theorem affineRawCarrier_eq_zeroLocus_vanishingIdeal
    (Z : IrreducibleClosed ZariskiC3) :
    affineRawCarrier Z = MvPolynomial.zeroLocus Complex
      (MvPolynomial.vanishingIdeal Complex (affineRawCarrier Z)) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  obtain ⟨I, hI⟩ :=
    (isClosed_affineZariski_iff (Fin 3) (affineRawCarrier Z)).mp
      (affineRawCarrier_isClosed Z)
  apply Set.Subset.antisymm
  · exact MvPolynomial.zeroLocus_vanishingIdeal_le (affineRawCarrier Z)
  · rw [hI]
    exact MvPolynomial.zeroLocus_anti_mono
      (by simpa [hI] using
        (MvPolynomial.le_vanishingIdeal_zeroLocus
          (k := Complex) (K := Complex) I))

private theorem affineRawCarrier_vanishingIdeal_isPrime
    (Z : IrreducibleClosed ZariskiC3) :
    (MvPolynomial.vanishingIdeal Complex (affineRawCarrier Z)).IsPrime := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  rw [Ideal.isPrime_iff]
  constructor
  · rw [ne_eq, Ideal.eq_top_iff_one]
    intro hone
    obtain ⟨x, hx⟩ := (affineRawCarrier_isIrreducible Z).nonempty
    have := (MvPolynomial.mem_vanishingIdeal_iff.mp hone) x hx
    simp at this
  · intro a b hab
    let Va : Set C3 := {x | MvPolynomial.aeval x a = 0}
    let Vb : Set C3 := {x | MvPolynomial.aeval x b = 0}
    have hVa : IsClosed Va := by
      have hEq : Va = MvPolynomial.zeroLocus Complex (Ideal.span {a}) := by
        ext x
        simp [Va, MvPolynomial.zeroLocus_span]
      rw [hEq]
      exact (isClosed_affineZariski_iff (Fin 3) _).mpr ⟨_, rfl⟩
    have hVb : IsClosed Vb := by
      have hEq : Vb = MvPolynomial.zeroLocus Complex (Ideal.span {b}) := by
        ext x
        simp [Vb, MvPolynomial.zeroLocus_span]
      rw [hEq]
      exact (isClosed_affineZariski_iff (Fin 3) _).mpr ⟨_, rfl⟩
    have hCover : affineRawCarrier Z ⊆ Va ∪ Vb := by
      intro x hx
      have hzero := (MvPolynomial.mem_vanishingIdeal_iff.mp hab) x hx
      rw [map_mul] at hzero
      rcases mul_eq_zero.mp hzero with ha | hb
      · exact Or.inl ha
      · exact Or.inr hb
    rcases (isPreirreducible_iff_isClosed_union_isClosed.mp
      (affineRawCarrier_isIrreducible Z).isPreirreducible)
        Va Vb hVa hVb hCover with ha | hb
    · exact Or.inl (MvPolynomial.mem_vanishingIdeal_iff.mpr fun x hx ↦ ha hx)
    · exact Or.inr (MvPolynomial.mem_vanishingIdeal_iff.mpr fun x hx ↦ hb hx)

private def finiteLineSet : Set C3 :=
  {b | b 0 = 0 ∧ b 1 = 0}

private theorem finiteLineSet_isClosed :
    IsClosed (zariskiLift finiteLineSet) := by
  have h0 : IsClosed (zariskiLift {b : C3 | b 0 = 0}) := by
    have h := zariskiLiftAffine_zeroLocus_isClosed
      (Fin 3) (coordinateIdeal (0 : Fin 3) 0)
    rw [zeroLocus_coordinateIdeal] at h
    exact h
  have h1 : IsClosed (zariskiLift {b : C3 | b 1 = 0}) := by
    have h := zariskiLiftAffine_zeroLocus_isClosed
      (Fin 3) (coordinateIdeal (1 : Fin 3) 0)
    rw [zeroLocus_coordinateIdeal] at h
    exact h
  have hEq : zariskiLift finiteLineSet =
      zariskiLift {b : C3 | b 0 = 0} ∩
        zariskiLift {b : C3 | b 1 = 0} := by
    ext z
    rfl
  rw [hEq]
  exact h0.inter h1

private theorem zariskiLift_singleton_isClosed (b : C3) :
    IsClosed (zariskiLift ({b} : Set C3)) := by
  have hi (i : Fin 3) :
      IsClosed (zariskiLift {x : C3 | x i = b i}) := by
    have h := zariskiLiftAffine_zeroLocus_isClosed
      (Fin 3) (coordinateIdeal i (b i))
    rw [zeroLocus_coordinateIdeal] at h
    exact h
  have hEq : zariskiLift ({b} : Set C3) =
      ⋂ i : Fin 3, zariskiLift {x : C3 | x i = b i} := by
    ext z
    constructor
    · intro hz
      rw [Set.mem_iInter]
      intro i
      exact congrFun hz i
    · intro hz
      apply _root_.funext
      intro i
      exact Set.mem_iInter.mp hz i
  rw [hEq]
  exact isClosed_iInter hi

private def finiteLineCoordinates :
    Fin 3 → MvPolynomial (Fin 3) Complex :=
  ![0, 0, MvPolynomial.X 2]

@[simp]
private theorem polynomialCoordinateMap_finiteLineCoordinates (x : C3) :
    polynomialCoordinateMap finiteLineCoordinates x = (![0, 0, x 2] : C3) := by
  funext i
  fin_cases i <;> simp [polynomialCoordinateMap, finiteLineCoordinates]

private def finiteLineParameter
    (h : Complex[X]) (z : ZariskiC3) :
    ZariskiSubspace (finiteComponent h) :=
  ⟨zariskiPolynomialCoordinateMap finiteLineCoordinates z, by
    change polynomialCoordinateMap finiteLineCoordinates (ofZariskiC3 z) ∈
      finiteComponent h
    rw [polynomialCoordinateMap_finiteLineCoordinates]
    exact finiteLine_subset_finiteComponent h ((ofZariskiC3 z) 2)⟩

private theorem continuous_finiteLineParameter (h : Complex[X]) :
    Continuous (finiteLineParameter h) :=
  continuous_zariskiPolynomialCoordinateMap finiteLineCoordinates |>.subtype_mk _

private theorem range_finiteLineParameter (h : Complex[X]) :
    Set.range (finiteLineParameter h) =
      Subtype.val ⁻¹' zariskiLift finiteLineSet := by
  ext z
  constructor
  · rintro ⟨w, rfl⟩
    change polynomialCoordinateMap finiteLineCoordinates (ofZariskiC3 w) ∈
      finiteLineSet
    rw [polynomialCoordinateMap_finiteLineCoordinates]
    simp [finiteLineSet]
  · intro hz
    let w : ZariskiC3 := toZariskiC3 (![0, 0, (ofZariskiC3 z.1) 2] : C3)
    refine ⟨w, Subtype.ext ?_⟩
    apply WithTopology.ext
    change polynomialCoordinateMap finiteLineCoordinates
      (![0, 0, (ofZariskiC3 z.1) 2] : C3) = ofZariskiC3 z.1
    rw [polynomialCoordinateMap_finiteLineCoordinates]
    change ofZariskiC3 z.1 ∈ finiteLineSet at hz
    simp only [finiteLineSet, Set.mem_setOf_eq] at hz
    funext i
    fin_cases i <;> simp [hz.1, hz.2]

private theorem finiteLineInComponent_isIrreducible (h : Complex[X]) :
    IsIrreducible
      (Subtype.val ⁻¹' zariskiLift finiteLineSet :
        Set (ZariskiSubspace (finiteComponent h))) := by
  have hImage := (isIrreducible_zariskiAffineSpace (Fin 3)).image
    (finiteLineParameter h) (continuous_finiteLineParameter h).continuousOn
  rw [Set.image_univ, range_finiteLineParameter h] at hImage
  exact hImage

private def finiteLineInComponent
    (h : Complex[X]) :
    IrreducibleClosed (ZariskiSubspace (finiteComponent h)) where
  carrier := Subtype.val ⁻¹' zariskiLift finiteLineSet
  isIrreducible' := finiteLineInComponent_isIrreducible h
  isClosed' := finiteLineSet_isClosed.preimage continuous_subtype_val

private def finiteOrigin (h : Complex[X]) :
    ZariskiSubspace (finiteComponent h) :=
  ⟨toZariskiC3 (![0, 0, 0] : C3), finiteLine_subset_finiteComponent h 0⟩

private def finiteOriginInComponent
    (h : Complex[X]) :
    IrreducibleClosed (ZariskiSubspace (finiteComponent h)) where
  carrier := {finiteOrigin h}
  isIrreducible' := isIrreducible_singleton
  isClosed' := by
    have hc := (zariskiLift_singleton_isClosed (![0, 0, 0] : C3)).preimage
      (continuous_subtype_val : Continuous
        (Subtype.val : ZariskiSubspace (finiteComponent h) → ZariskiC3))
    have hEq :
        Subtype.val ⁻¹' zariskiLift ({(![0, 0, 0] : C3)} : Set C3) =
          ({finiteOrigin h} : Set (ZariskiSubspace (finiteComponent h))) := by
      ext z
      constructor
      · intro hz
        apply Set.mem_singleton_iff.mpr
        apply Subtype.ext
        apply WithTopology.ext
        exact hz
      · intro hz
        rw [Set.mem_singleton_iff] at hz
        subst z
        rfl
    rwa [hEq] at hc

private def finiteComponentWhole
    (h : Complex[X]) :
    IrreducibleClosed (ZariskiSubspace (finiteComponent h)) where
  carrier := Set.univ
  isIrreducible' := by
    letI : IrreducibleSpace (ZariskiSubspace (finiteComponent h)) :=
      Subtype.irreducibleSpace (finiteComponent_isIrreducible h)
    exact IrreducibleSpace.isIrreducible_univ _
  isClosed' := isClosed_univ

private theorem finiteOrigin_lt_finiteLine (h : Complex[X]) :
    finiteOriginInComponent h < finiteLineInComponent h := by
  apply lt_of_le_of_ne
  · intro z hz
    change z = finiteOrigin h at hz
    subst z
    change (![0, 0, 0] : C3) ∈ finiteLineSet
    simp [finiteLineSet]
  · intro hEqual
    have hReverse : finiteLineInComponent h ≤ finiteOriginInComponent h := by
      rw [hEqual]
    let z : ZariskiSubspace (finiteComponent h) :=
      ⟨toZariskiC3 (![0, 0, 1] : C3), finiteLine_subset_finiteComponent h 1⟩
    have hzLine : z ∈ finiteLineInComponent h := by
      change (![0, 0, 1] : C3) ∈ finiteLineSet
      simp [finiteLineSet]
    have hzOrigin := hReverse hzLine
    change z = finiteOrigin h at hzOrigin
    have hzRaw := congrArg (fun w ↦ (ofZariskiC3 w.1) 2) hzOrigin
    simp [z, finiteOrigin] at hzRaw

private theorem finiteLine_lt_finiteComponentWhole (h : Complex[X]) :
    finiteLineInComponent h < finiteComponentWhole h := by
  apply lt_of_le_of_ne
  · intro z _
    exact Set.mem_univ z
  · intro hEqual
    have hReverse : finiteComponentWhole h ≤ finiteLineInComponent h := by
      rw [hEqual]
    obtain ⟨b, hb0⟩ := finiteComponent_dominatesPLine h 1
    let z : ZariskiSubspace (finiteComponent h) := ⟨toZariskiC3 b.1, b.2⟩
    have hzWhole : z ∈ finiteComponentWhole h := by
      exact Set.mem_univ z
    have hzLine := hReverse hzWhole
    have hzRaw : b.1 ∈ finiteLineSet := hzLine
    exact one_ne_zero (hb0.symm.trans hzRaw.1)

/-- The prime ideal of an irreducible affine closed set.  The order dual is
used because larger point sets have smaller vanishing ideals. -/
private def affineIrreducibleClosedToPrimeDual
    (Z : IrreducibleClosed ZariskiC3) :
    (PrimeSpectrum C3CoordinateRing)ᵒᵈ :=
  OrderDual.toDual
    ⟨MvPolynomial.vanishingIdeal Complex (affineRawCarrier Z),
      affineRawCarrier_vanishingIdeal_isPrime Z⟩

private theorem affineIrreducibleClosedToPrimeDual_mono :
    Monotone affineIrreducibleClosedToPrimeDual := by
  intro Z W hZW
  exact MvPolynomial.vanishingIdeal_anti_mono fun x hx ↦ hZW hx

private theorem affineIrreducibleClosedToPrimeDual_injective :
    Function.Injective affineIrreducibleClosedToPrimeDual := by
  intro Z W hZW
  have hIdeals :
      MvPolynomial.vanishingIdeal Complex (affineRawCarrier Z) =
        MvPolynomial.vanishingIdeal Complex (affineRawCarrier W) := by
    exact congrArg PrimeSpectrum.asIdeal hZW
  have hRaw : affineRawCarrier Z = affineRawCarrier W := by
    rw [affineRawCarrier_eq_zeroLocus_vanishingIdeal Z,
      affineRawCarrier_eq_zeroLocus_vanishingIdeal W, hIdeals]
  apply TopologicalSpace.IrreducibleCloseds.ext
  ext z
  have hz := Set.ext_iff.mp hRaw (ofZariskiC3 z)
  simpa [affineRawCarrier] using hz

private theorem affineIrreducibleClosedToPrimeDual_strictMono :
    StrictMono affineIrreducibleClosedToPrimeDual :=
  affineIrreducibleClosedToPrimeDual_mono.strictMono_of_injective
    affineIrreducibleClosedToPrimeDual_injective

private def finiteComponentIrreducibleClosedToAmbient
    (h : Complex[X])
    (Z : IrreducibleClosed (ZariskiSubspace (finiteComponent h))) :
    IrreducibleClosed ZariskiC3 :=
  TopologicalSpace.IrreducibleCloseds.map Subtype.val continuous_subtype_val Z

private theorem finiteComponent_ambient_rawCarrier_subset
    (h : Complex[X])
    (Z : IrreducibleClosed (ZariskiSubspace (finiteComponent h))) :
    affineRawCarrier (finiteComponentIrreducibleClosedToAmbient h Z) ⊆
      finiteComponent h := by
  intro x hx
  have hClosure :
      closure (Subtype.val '' (Z : Set (ZariskiSubspace (finiteComponent h)))) ⊆
        zariskiLift (finiteComponent h) := by
    apply closure_minimal
    · rintro z ⟨y, hy, rfl⟩
      exact y.property
    · exact finiteComponent_isClosed h
  exact hClosure hx

/-- Every irreducible closed subset of the finite component determines a
prime of its principal coordinate ring, represented here as a prime of the
ambient polynomial ring containing the defining ideal. -/
private def finiteComponentIrreducibleClosedToPrimeDual
    (h : Complex[X])
    (Z : IrreducibleClosed (ZariskiSubspace (finiteComponent h))) :
    (PrimeSpectrum.zeroLocus
      (R := C3CoordinateRing) (Ideal.span {deltaFamily h}))ᵒᵈ := by
  let A := finiteComponentIrreducibleClosedToAmbient h Z
  let P : PrimeSpectrum C3CoordinateRing :=
    OrderDual.ofDual (affineIrreducibleClosedToPrimeDual A)
  have hZero : affineRawCarrier A ⊆
      MvPolynomial.zeroLocus Complex (Ideal.span {deltaFamily h}) := by
    intro x hx
    have hxFinite := finiteComponent_ambient_rawCarrier_subset h Z hx
    simpa [finiteComponent_eq_deltaZeroLocus h, deltaZeroLocus] using hxFinite
  have hIdeal : Ideal.span {deltaFamily h} ≤ P.asIdeal := by
    exact (MvPolynomial.le_zeroLocus_iff_le_vanishingIdeal
      (k := Complex) (K := Complex)).mp hZero
  exact OrderDual.toDual ⟨P, hIdeal⟩

private theorem finiteComponentIrreducibleClosedToPrimeDual_strictMono
    (h : Complex[X]) :
    StrictMono (finiteComponentIrreducibleClosedToPrimeDual h) := by
  intro Z W hZW
  have hAmbient :
      finiteComponentIrreducibleClosedToAmbient h Z <
        finiteComponentIrreducibleClosedToAmbient h W :=
    (TopologicalSpace.IrreducibleCloseds.map_strictMono_of_isInducing
      Topology.IsInducing.subtypeVal) hZW
  exact affineIrreducibleClosedToPrimeDual_strictMono hAmbient

/-- The coordinate ring of the finite component, presented by its single equation. -/
abbrev finiteComponentCoordinateRing (h : Complex[X]) :=
  C3CoordinateRing ⧸ Ideal.span {deltaFamily h}

theorem finiteComponentCoordinateRing_ringKrullDim_le_two (h : Complex[X]) :
    ringKrullDim (finiteComponentCoordinateRing h) ≤ 2 := by
  have hregular : deltaFamily h ∈ C3CoordinateRing⁰ := by
    exact mem_nonZeroDivisors_iff_ne_zero.mpr (deltaFamily_ne_zero h)
  have hquot := ringKrullDim_quotient_succ_le_of_nonZeroDivisor hregular
  have hquot' : ringKrullDim (finiteComponentCoordinateRing h) + 1 ≤
      (3 : WithBot ℕ∞) := by
    simpa [finiteComponentCoordinateRing, C3CoordinateRing,
      ringKrullDim_eq_zero_of_field] using hquot
  have hlt : ringKrullDim (finiteComponentCoordinateRing h) <
      (3 : WithBot ℕ∞) := ENat.WithBot.add_one_le_natCast_iff.mp hquot'
  apply ENat.WithBot.lt_add_one_iff.mp
  norm_num at hlt ⊢
  exact hlt

theorem finiteComponent_topologicalKrullDim_le_two (h : Complex[X]) :
    topologicalKrullDim (ZariskiSubspace (finiteComponent h)) ≤ 2 := by
  change Order.krullDim
    (IrreducibleClosed (ZariskiSubspace (finiteComponent h))) ≤ 2
  calc
    Order.krullDim
        (IrreducibleClosed (ZariskiSubspace (finiteComponent h))) ≤
        Order.krullDim
          ((PrimeSpectrum.zeroLocus
            (R := C3CoordinateRing) (Ideal.span {deltaFamily h}))ᵒᵈ) :=
      Order.krullDim_le_of_strictMono
        (finiteComponentIrreducibleClosedToPrimeDual h)
        (finiteComponentIrreducibleClosedToPrimeDual_strictMono h)
    _ = Order.krullDim
          (PrimeSpectrum.zeroLocus
            (R := C3CoordinateRing) (Ideal.span {deltaFamily h})) :=
      Order.krullDim_orderDual
    _ = ringKrullDim (finiteComponentCoordinateRing h) := by
      symm
      exact ringKrullDim_quotient (Ideal.span {deltaFamily h})
    _ ≤ 2 := finiteComponentCoordinateRing_ringKrullDim_le_two h

theorem two_le_finiteComponent_topologicalKrullDim (h : Complex[X]) :
    2 ≤ topologicalKrullDim (ZariskiSubspace (finiteComponent h)) := by
  change 2 ≤ Order.krullDim
    (IrreducibleClosed (ZariskiSubspace (finiteComponent h)))
  apply Order.le_krullDim_iff.mpr
  let p : LTSeries
      (IrreducibleClosed (ZariskiSubspace (finiteComponent h))) :=
    RelSeries.fromListIsChain
      [finiteOriginInComponent h, finiteLineInComponent h, finiteComponentWhole h]
      (List.cons_ne_nil (finiteOriginInComponent h)
        [finiteLineInComponent h, finiteComponentWhole h])
      (List.IsChain.cons_cons (finiteOrigin_lt_finiteLine h)
        (List.isChain_pair.mpr (finiteLine_lt_finiteComponentWhole h)))
  exact ⟨p, by simp [p]⟩

/-- The finite multiple-root component is a genuine affine surface. -/
theorem finiteComponent_topologicalKrullDim (h : Complex[X]) :
    topologicalKrullDim (ZariskiSubspace (finiteComponent h)) = 2 :=
  le_antisymm (finiteComponent_topologicalKrullDim_le_two h)
    (two_le_finiteComponent_topologicalKrullDim h)

end

end DegreeSixKeller
