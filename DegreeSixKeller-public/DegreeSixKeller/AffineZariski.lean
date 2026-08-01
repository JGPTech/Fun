import DegreeSixKeller.FetaCore
import Mathlib.Analysis.Complex.Polynomial.Basic
import Mathlib.Algebra.MvPolynomial.Polynomial
import Mathlib.RingTheory.Nullstellensatz
import Mathlib.RingTheory.Polynomial.Ideal
import Mathlib.Topology.Irreducible
import Mathlib.Topology.WithTopology

/-!
# The point-set Zariski topology used by the pair-specific proof

`C3` already carries its Euclidean topology, which is needed by the sequence
definition of nonproperness.  This file constructs the genuine affine Zariski
topology on the same underlying points and places it on the type synonym
`ZariskiC3`.  Thus the analytic and algebraic topologies cannot be confused by
typeclass inference.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Ideal Set Topology

noncomputable section

/-- A subset of affine space is algebraically closed when it is the common
zero locus of an ideal of multivariate polynomials. -/
def AffineClosed (σ : Type*) (Z : Set (σ -> Complex)) : Prop :=
  ∃ I : Ideal (MvPolynomial σ Complex),
    Z = MvPolynomial.zeroLocus Complex I

theorem affineClosed_univ (σ : Type*) :
    AffineClosed σ (Set.univ : Set (σ -> Complex)) := by
  refine ⟨⊥, ?_⟩
  exact (MvPolynomial.zeroLocus_bot (k := Complex) (K := Complex)).symm

theorem affineClosed_empty (σ : Type*) :
    AffineClosed σ (∅ : Set (σ -> Complex)) := by
  refine ⟨⊤, ?_⟩
  exact (MvPolynomial.zeroLocus_top (k := Complex) (K := Complex)).symm

theorem affineClosed_sInter (σ : Type*)
    (S : Set (Set (σ -> Complex)))
    (hS : ∀ Z ∈ S, AffineClosed σ Z) :
    AffineClosed σ (⋂₀ S) := by
  classical
  let I : S -> Ideal (MvPolynomial σ Complex) :=
    fun Z => Classical.choose (hS Z.1 Z.2)
  have hI (Z : S) :
      Z.1 = MvPolynomial.zeroLocus Complex (I Z) :=
    Classical.choose_spec (hS Z.1 Z.2)
  let generators : Set (MvPolynomial σ Complex) :=
    ⋃ Z : S, (I Z : Set (MvPolynomial σ Complex))
  refine ⟨Ideal.span generators, ?_⟩
  rw [MvPolynomial.zeroLocus_span]
  ext x
  constructor
  · intro hx p hp
    rcases Set.mem_iUnion.1 hp with ⟨Z, hpZ⟩
    have hxZ : x ∈ Z.1 := (Set.mem_sInter.1 hx) Z.1 Z.2
    rw [hI Z] at hxZ
    exact hxZ p hpZ
  · intro hx
    rw [Set.mem_sInter]
    intro Z hZS
    change x ∈ (⟨Z, hZS⟩ : S).1
    rw [hI ⟨Z, hZS⟩]
    intro p hp
    exact hx p (Set.mem_iUnion.2 ⟨⟨Z, hZS⟩, hp⟩)

/-- The zero locus of a product of ideals is the union of their zero loci. -/
theorem zeroLocus_mul_ideal (σ : Type*)
    (I J : Ideal (MvPolynomial σ Complex)) :
    MvPolynomial.zeroLocus Complex (I * J) =
      MvPolynomial.zeroLocus Complex I ∪
        MvPolynomial.zeroLocus Complex J := by
  classical
  ext x
  constructor
  · intro hx
    by_contra hUnion
    rw [Set.mem_union, not_or] at hUnion
    have hI : ¬ ∀ p ∈ I, MvPolynomial.aeval x p = 0 := hUnion.1
    have hJ : ¬ ∀ p ∈ J, MvPolynomial.aeval x p = 0 := hUnion.2
    push Not at hI hJ
    obtain ⟨p, hpI, hp⟩ := hI
    obtain ⟨q, hqJ, hq⟩ := hJ
    have hpq := hx (p * q) (Ideal.mul_mem_mul hpI hqJ)
    exact (mul_ne_zero hp hq) (by simpa using hpq)
  · rintro (hx | hx) p hp
    · exact hx p (Ideal.mul_le_right hp)
    · exact hx p (Ideal.mul_le_left hp)

theorem affineClosed_union (σ : Type*)
    (A B : Set (σ -> Complex))
    (hA : AffineClosed σ A) (hB : AffineClosed σ B) :
    AffineClosed σ (A ∪ B) := by
  rcases hA with ⟨I, rfl⟩
  rcases hB with ⟨J, rfl⟩
  exact ⟨I * J, (zeroLocus_mul_ideal σ I J).symm⟩

/-- The affine Zariski topology on complex points. -/
@[implicit_reducible]
def affineZariskiTopology (σ : Type*) : TopologicalSpace (σ -> Complex) :=
  TopologicalSpace.ofClosed
    {Z | AffineClosed σ Z}
    (affineClosed_empty σ)
    (affineClosed_sInter σ)
    (by
      intro A hA B hB
      exact affineClosed_union σ A B hA hB)

theorem isOpen_affineZariski_iff (σ : Type*)
    (U : Set (σ -> Complex)) :
    @IsOpen (σ -> Complex) (affineZariskiTopology σ) U ↔
      AffineClosed σ Uᶜ := by
  rfl

theorem isClosed_affineZariski_iff (σ : Type*)
    (Z : Set (σ -> Complex)) :
    @IsClosed (σ -> Complex) (affineZariskiTopology σ) Z ↔
      AffineClosed σ Z := by
  letI : TopologicalSpace (σ -> Complex) := affineZariskiTopology σ
  rw [← isOpen_compl_iff, isOpen_affineZariski_iff, compl_compl]

/-- Over an algebraically closed field, the zero locus of a prime ideal is an
irreducible affine algebraic set.  This is the point-set counterpart of the
prime-spectrum result used in algebraic geometry. -/
theorem isIrreducible_zeroLocus_of_isPrime
    (σ : Type*) [Finite σ]
    (P : Ideal (MvPolynomial σ Complex)) [P.IsPrime] :
    @IsIrreducible (σ -> Complex) (affineZariskiTopology σ)
      (MvPolynomial.zeroLocus Complex P) := by
  letI : TopologicalSpace (σ -> Complex) := affineZariskiTopology σ
  have hPrime : P.IsPrime := inferInstance
  have hNonempty : (MvPolynomial.zeroLocus Complex P).Nonempty := by
    by_contra h
    have hEmpty : MvPolynomial.zeroLocus Complex P = ∅ :=
      Set.not_nonempty_iff_eq_empty.mp h
    have hTop : P = ⊤ := by
      rw [← MvPolynomial.IsPrime.vanishingIdeal_zeroLocus
        (k := Complex) (K := Complex) P, hEmpty,
        MvPolynomial.vanishingIdeal_empty]
    exact hPrime.ne_top hTop
  refine ⟨hNonempty, isPreirreducible_iff_isClosed_union_isClosed.mpr ?_⟩
  intro Z₁ Z₂ hZ₁ hZ₂ hCover
  rcases (isClosed_affineZariski_iff σ Z₁).mp hZ₁ with ⟨I, rfl⟩
  rcases (isClosed_affineZariski_iff σ Z₂).mp hZ₂ with ⟨J, rfl⟩
  rw [← zeroLocus_mul_ideal σ I J] at hCover
  have hMul : I * J ≤ P := by
    have h :=
      (MvPolynomial.le_zeroLocus_iff_le_vanishingIdeal
        (k := Complex) (K := Complex)).mp hCover
    simpa using h
  rcases hPrime.mul_le.mp hMul with hIP | hJP
  · exact Or.inl (MvPolynomial.zeroLocus_anti_mono hIP)
  · exact Or.inr (MvPolynomial.zeroLocus_anti_mono hJP)

theorem isIrreducible_affineSpace (σ : Type*) [Finite σ] :
    @IsIrreducible (σ -> Complex) (affineZariskiTopology σ) Set.univ := by
  simpa using
    (isIrreducible_zeroLocus_of_isPrime σ
      (⊥ : Ideal (MvPolynomial σ Complex)))

/-- Affine space with its point-set Zariski topology.  This generic type
synonym is used for parameter spaces as well as the target `ZariskiC3`. -/
abbrev ZariskiAffine (σ : Type*) :=
  WithTopology (σ -> Complex) (affineZariskiTopology σ)

/-- Regard an ordinary affine point as the same point with the Zariski
topology. -/
def toZariskiAffine {σ : Type*} : (σ -> Complex) -> ZariskiAffine σ :=
  WithTopology.toTopology (affineZariskiTopology σ)

/-- Forget the Zariski topology on an affine point. -/
def ofZariskiAffine {σ : Type*} : ZariskiAffine σ -> (σ -> Complex) :=
  WithTopology.ofTopology

@[simp]
theorem ofZariskiAffine_toZariskiAffine {σ : Type*} (x : σ -> Complex) :
    ofZariskiAffine (toZariskiAffine x) = x := rfl

@[simp]
theorem toZariskiAffine_ofZariskiAffine {σ : Type*} (x : ZariskiAffine σ) :
    toZariskiAffine (ofZariskiAffine x) = x := rfl

/-- Lift a set of ordinary affine points to the corresponding Zariski type
synonym. -/
def zariskiLiftAffine {σ : Type*} (S : Set (σ -> Complex)) :
    Set (ZariskiAffine σ) :=
  ofZariskiAffine ⁻¹' S

/-- Every algebraic zero locus is closed after lifting it to the type synonym
carrying the affine Zariski topology. -/
theorem zariskiLiftAffine_zeroLocus_isClosed
    (σ : Type*) (I : Ideal (MvPolynomial σ Complex)) :
    IsClosed
      (zariskiLiftAffine (MvPolynomial.zeroLocus Complex I) :
        Set (ZariskiAffine σ)) := by
  letI : TopologicalSpace (σ -> Complex) := affineZariskiTopology σ
  have hRaw : IsClosed (MvPolynomial.zeroLocus Complex I) := by
    exact (isClosed_affineZariski_iff σ
      (MvPolynomial.zeroLocus Complex I)).mpr ⟨I, rfl⟩
  exact hRaw.preimage
    (WithTopology.continuous_ofTopology (affineZariskiTopology σ))

/-- Affine space remains irreducible after moving its topology to the
`WithTopology` type synonym. -/
theorem isIrreducible_zariskiAffineSpace
    (σ : Type*) [Finite σ] :
    IsIrreducible (Set.univ : Set (ZariskiAffine σ)) := by
  letI : TopologicalSpace (σ -> Complex) := affineZariskiTopology σ
  have hRaw : IsIrreducible (Set.univ : Set (σ -> Complex)) :=
    isIrreducible_affineSpace σ
  have hImage := hRaw.image
    (WithTopology.toTopology (affineZariskiTopology σ))
    (WithTopology.continuous_toTopology
      (affineZariskiTopology σ)).continuousOn
  have hImageSet :
      WithTopology.toTopology (affineZariskiTopology σ) ''
          (Set.univ : Set (σ -> Complex)) =
        (Set.univ : Set (ZariskiAffine σ)) := by
    ext x
    constructor
    · intro _
      exact Set.mem_univ x
    · intro _
      refine ⟨x.ofTopology, Set.mem_univ _, ?_⟩
      cases x
      rfl
  rw [hImageSet] at hImage
  exact hImage

/-- The ideal of all polynomials vanishing on the coordinate hyperplane
`x i = alpha`.  Defining it as a vanishing ideal gives a uniform construction
in every finite affine dimension. -/
def coordinateIdeal
    {n : Nat} (i : Fin n) (alpha : Complex) :
    Ideal (MvPolynomial (Fin n) Complex) :=
  MvPolynomial.vanishingIdeal Complex
    {x : Fin n -> Complex | x i = alpha}

/-- The zero locus of `coordinateIdeal i alpha` is exactly the coordinate
hyperplane `x i = alpha`. -/
theorem zeroLocus_coordinateIdeal
    {n : Nat} (i : Fin n) (alpha : Complex) :
    MvPolynomial.zeroLocus Complex (coordinateIdeal i alpha) =
      {x : Fin n -> Complex | x i = alpha} := by
  classical
  apply Set.Subset.antisymm
  · intro x hx
    let g : MvPolynomial (Fin n) Complex :=
      MvPolynomial.X i - MvPolynomial.C alpha
    have hg : g ∈ coordinateIdeal i alpha := by
      rw [coordinateIdeal, MvPolynomial.mem_vanishingIdeal_iff]
      intro y hy
      simpa [g, sub_eq_zero, MvPolynomial.aeval_eq_eval] using hy
    have hzero := hx g hg
    simpa [g, sub_eq_zero, MvPolynomial.aeval_eq_eval] using hzero
  · simpa [coordinateIdeal] using
      (MvPolynomial.zeroLocus_vanishingIdeal_le
        (k := Complex) (K := Complex)
        ({x : Fin n -> Complex | x i = alpha}))

/-- The prime ideal obtained by evaluating the first coordinate at `alpha`.
The codomain is the polynomial ring in the remaining two coordinates. -/
def firstCoordinateIdeal (alpha : Complex) :
    Ideal (MvPolynomial (Fin 3) Complex) :=
  RingHom.ker
    ((Polynomial.evalRingHom (MvPolynomial.C alpha)).comp
      (MvPolynomial.finSuccEquiv Complex 2).toRingHom)

instance firstCoordinateIdeal_isPrime (alpha : Complex) :
    (firstCoordinateIdeal alpha).IsPrime :=
  RingHom.ker_isPrime _

/-- The zero locus of the first-coordinate evaluation ideal is exactly the
affine hyperplane whose first coordinate is `alpha`. -/
theorem zeroLocus_firstCoordinateIdeal (alpha : Complex) :
    MvPolynomial.zeroLocus Complex (firstCoordinateIdeal alpha) =
      {x : Fin 3 -> Complex | x 0 = alpha} := by
  classical
  ext x
  constructor
  · intro hx
    let g : MvPolynomial (Fin 3) Complex :=
      MvPolynomial.X 0 - MvPolynomial.C alpha
    have hg : g ∈ firstCoordinateIdeal alpha := by
      change Polynomial.eval (MvPolynomial.C alpha)
        (MvPolynomial.finSuccEquiv Complex 2 g) = 0
      have hC :
          MvPolynomial.finSuccEquiv Complex 2 (MvPolynomial.C alpha) =
            Polynomial.C (MvPolynomial.C alpha) := by
        simpa [MvPolynomial.algebraMap_eq, Polynomial.algebraMap_apply] using
          (MvPolynomial.finSuccEquiv Complex 2).commutes alpha
      dsimp [g]
      rw [map_sub, MvPolynomial.finSuccEquiv_X_zero, hC]
      simp
    have hzero := hx g hg
    simpa [g, sub_eq_zero] using hzero
  · intro hx p hp
    have hpEval :
        Polynomial.eval (MvPolynomial.C alpha)
          (MvPolynomial.finSuccEquiv Complex 2 p) = 0 := by
      simpa [firstCoordinateIdeal, RingHom.mem_ker] using hp
    let tail : Fin 2 -> Complex := fun i => x i.succ
    have hEval := congrArg (MvPolynomial.eval tail) hpEval
    have hPoint : Fin.cases alpha tail = x := by
      funext i
      refine Fin.cases ?_ ?_ i
      · exact hx.symm
      · intro j
        rfl
    rw [map_zero, MvPolynomial.eval_polynomial_eval_finSuccEquiv,
      MvPolynomial.eval_C, hPoint] at hEval
    simpa [MvPolynomial.aeval_eq_eval] using hEval

/-- The original first-coordinate kernel construction agrees with the generic
coordinate-hyperplane ideal. -/
theorem firstCoordinateIdeal_eq_coordinateIdeal (alpha : Complex) :
    firstCoordinateIdeal alpha =
      coordinateIdeal (0 : Fin 3) alpha := by
  rw [coordinateIdeal, ← zeroLocus_firstCoordinateIdeal]
  symm
  exact MvPolynomial.IsPrime.vanishingIdeal_zeroLocus
    (k := Complex) (K := Complex) (firstCoordinateIdeal alpha)

/-- Complex affine three-space equipped with the Zariski rather than Euclidean
topology. -/
abbrev ZariskiC3 := ZariskiAffine (Fin 3)

/-- Regard a Euclidean `C3` point as the same point in `ZariskiC3`. -/
def toZariskiC3 : C3 -> ZariskiC3 :=
  toZariskiAffine

/-- Forget the Zariski topology on a point. -/
def ofZariskiC3 : ZariskiC3 -> C3 :=
  ofZariskiAffine

@[simp]
theorem ofZariskiC3_toZariskiC3 (x : C3) :
    ofZariskiC3 (toZariskiC3 x) = x := rfl

@[simp]
theorem toZariskiC3_ofZariskiC3 (x : ZariskiC3) :
    toZariskiC3 (ofZariskiC3 x) = x := rfl

/-- Lift a set of ordinary complex points to the Zariski type synonym. -/
def zariskiLift (S : Set C3) : Set ZariskiC3 :=
  zariskiLiftAffine S

/-- The algebraic subspace carried by a set of complex points. -/
abbrev ZariskiSubspace (S : Set C3) := zariskiLift S

/-- The actual number of irreducible components of the reduced algebraic
subspace carried by `S`.  This is no longer an arbitrary counting function. -/
noncomputable def algebraicComponentCount (S : Set C3) : Nat :=
  (irreducibleComponents (ZariskiSubspace S)).ncard

end

end DegreeSixKeller
