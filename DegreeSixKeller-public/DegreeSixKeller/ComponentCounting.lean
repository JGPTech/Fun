import DegreeSixKeller.AffineZariski

/-!
# Exact finite irreducible-component counting

This module proves the topological lemma used to turn a finite, irredundant
decomposition into the actual `irreducibleComponents` set and its cardinality.
It is independent of the special formulas for CEX-004 and CEX-006.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Set Topology

noncomputable section

theorem irreducibleComponents_eq_finset
    {X : Type*} [TopologicalSpace X]
    (C : Finset (Set X))
    (hClosed : ∀ Z ∈ C, IsClosed Z)
    (hIrreducible : ∀ Z ∈ C, IsIrreducible Z)
    (hCover : ⋃₀ (C : Set (Set X)) = Set.univ)
    (hIrredundant :
      ∀ A ∈ C, ∀ B ∈ C, A ⊆ B -> A = B) :
    irreducibleComponents X = (C : Set (Set X)) := by
  classical
  ext Z
  constructor
  · intro hZ
    have hZCover : Z ⊆ ⋃₀ (C : Set (Set X)) := by
      rw [hCover]
      exact Set.subset_univ Z
    obtain ⟨A, hAC, hZA⟩ :=
      isIrreducible_iff_sUnion_isClosed.mp hZ.1 C hClosed hZCover
    have hAZ : A ⊆ Z := hZ.2 (hIrreducible A hAC) hZA
    have hEq : Z = A := Set.Subset.antisymm hZA hAZ
    simpa [hEq] using hAC
  · intro hZC
    refine ⟨hIrreducible Z hZC, ?_⟩
    intro Y hY hZY
    have hYCover : Y ⊆ ⋃₀ (C : Set (Set X)) := by
      rw [hCover]
      exact Set.subset_univ Y
    obtain ⟨B, hBC, hYB⟩ :=
      isIrreducible_iff_sUnion_isClosed.mp hY C hClosed hYCover
    have hZB : Z ⊆ B := hZY.trans hYB
    have hEq : Z = B := hIrredundant Z hZC B hBC hZB
    simpa [hEq] using hYB

theorem irreducibleComponentCount_eq_finset_card
    {X : Type*} [TopologicalSpace X]
    (C : Finset (Set X))
    (hClosed : ∀ Z ∈ C, IsClosed Z)
    (hIrreducible : ∀ Z ∈ C, IsIrreducible Z)
    (hCover : ⋃₀ (C : Set (Set X)) = Set.univ)
    (hIrredundant :
      ∀ A ∈ C, ∀ B ∈ C, A ⊆ B -> A = B) :
    (irreducibleComponents X).ncard = C.card := by
  rw [irreducibleComponents_eq_finset C hClosed hIrreducible hCover hIrredundant]
  exact Set.ncard_coe_finset C

/-- The part of an ambient algebraic set `A` viewed inside the algebraic
subspace carried by `S`. -/
def restrictedPiece (S A : Set C3) : Set (ZariskiSubspace S) :=
  Subtype.val ⁻¹' zariskiLift A

theorem restrictedPiece_isClosed {S A : Set C3}
    (hA : IsClosed (zariskiLift A)) :
    IsClosed (restrictedPiece S A) :=
  hA.preimage continuous_subtype_val

theorem restrictedPiece_isIrreducible {S A : Set C3}
    (hAS : A ⊆ S)
    (hA : IsIrreducible (zariskiLift A)) :
    IsIrreducible (restrictedPiece S A) := by
  letI : IrreducibleSpace (ZariskiSubspace A) :=
    Subtype.irreducibleSpace hA
  let f : ZariskiSubspace A -> ZariskiSubspace S :=
    fun x => ⟨x.1, hAS x.2⟩
  have hf : Continuous f :=
    continuous_subtype_val.subtype_mk _
  have hImage :=
    (IrreducibleSpace.isIrreducible_univ (ZariskiSubspace A)).image
      f hf.continuousOn
  convert hImage using 1
  ext x
  constructor
  · intro hx
    exact ⟨⟨x.1, hx⟩, Set.mem_univ _, rfl⟩
  · rintro ⟨y, -, rfl⟩
    exact y.2

/-- A finite irreducible, closed, irredundant ambient decomposition computes
the genuine irreducible-component count of the corresponding subspace. -/
theorem algebraicComponentCount_eq_finset_card
    (S : Set C3) (C : Finset (Set C3))
    (hClosed : ∀ A ∈ C, IsClosed (zariskiLift A))
    (hIrreducible : ∀ A ∈ C, IsIrreducible (zariskiLift A))
    (hCover : ⋃₀ (C : Set (Set C3)) = S)
    (hIrredundant :
      ∀ A ∈ C, ∀ B ∈ C, A ⊆ B -> A = B) :
    algebraicComponentCount S = C.card := by
  classical
  let pieces : Finset (Set (ZariskiSubspace S)) :=
    C.image (restrictedPiece S)
  have hSubset (A : Set C3) (hAC : A ∈ C) : A ⊆ S := by
    rw [← hCover]
    exact Set.subset_sUnion_of_mem hAC
  have hPieceInjective : Set.InjOn (restrictedPiece S) (C : Set (Set C3)) := by
    intro A hAC B hBC hEq
    apply Set.Subset.antisymm
    · intro x hx
      let z : ZariskiSubspace S :=
        ⟨toZariskiC3 x, hSubset A hAC hx⟩
      have hzA : z ∈ restrictedPiece S A := hx
      have hzB : z ∈ restrictedPiece S B := by
        rw [← hEq]
        exact hzA
      exact hzB
    · intro x hx
      let z : ZariskiSubspace S :=
        ⟨toZariskiC3 x, hSubset B hBC hx⟩
      have hzB : z ∈ restrictedPiece S B := hx
      have hzA : z ∈ restrictedPiece S A := by
        rw [hEq]
        exact hzB
      exact hzA
  have hPiecesCard : pieces.card = C.card := by
    exact Finset.card_image_iff.mpr hPieceInjective
  have hPiecesClosed : ∀ Z ∈ pieces, IsClosed Z := by
    intro Z hZ
    rcases Finset.mem_image.mp hZ with ⟨A, hAC, rfl⟩
    exact restrictedPiece_isClosed (hClosed A hAC)
  have hPiecesIrreducible : ∀ Z ∈ pieces, IsIrreducible Z := by
    intro Z hZ
    rcases Finset.mem_image.mp hZ with ⟨A, hAC, rfl⟩
    exact restrictedPiece_isIrreducible (hSubset A hAC)
      (hIrreducible A hAC)
  have hPiecesCover : ⋃₀ (pieces : Set (Set (ZariskiSubspace S))) = Set.univ := by
    ext x
    constructor
    · intro _
      exact Set.mem_univ x
    · intro _
      have hxS : ofZariskiC3 x.1 ∈ S := x.2
      have hxUnion : ofZariskiC3 x.1 ∈ ⋃₀ (C : Set (Set C3)) :=
        hCover.ge hxS
      rcases Set.mem_sUnion.mp hxUnion with ⟨A, hAC, hxA⟩
      have hxPiece : x ∈ restrictedPiece S A := hxA
      exact Set.mem_sUnion_of_mem hxPiece
        (Finset.mem_coe.mpr (Finset.mem_image.mpr ⟨A, hAC, rfl⟩))
  have hPiecesIrredundant :
      ∀ A ∈ pieces, ∀ B ∈ pieces, A ⊆ B -> A = B := by
    intro A hA B hB hAB
    rcases Finset.mem_image.mp hA with ⟨A₀, hA₀, rfl⟩
    rcases Finset.mem_image.mp hB with ⟨B₀, hB₀, rfl⟩
    have hAmbient : A₀ ⊆ B₀ := by
      intro x hx
      let z : ZariskiSubspace S :=
        ⟨toZariskiC3 x, hSubset A₀ hA₀ hx⟩
      have hzA : z ∈ restrictedPiece S A₀ := hx
      have hzB : z ∈ restrictedPiece S B₀ := hAB hzA
      exact hzB
    rw [hIrredundant A₀ hA₀ B₀ hB₀ hAmbient]
  rw [algebraicComponentCount]
  calc
    (irreducibleComponents (ZariskiSubspace S)).ncard = pieces.card :=
      irreducibleComponentCount_eq_finset_card pieces hPiecesClosed
        hPiecesIrreducible hPiecesCover hPiecesIrredundant
    _ = C.card := hPiecesCard

end

end DegreeSixKeller
