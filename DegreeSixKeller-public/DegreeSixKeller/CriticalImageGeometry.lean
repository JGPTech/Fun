import DegreeSixKeller.PairSpecificGeometry
import Mathlib.Algebra.MvPolynomial.Eval
import Mathlib.Algebra.Polynomial.Eval.Coeff
import Mathlib.Tactic

/-!
# Irreducibility of the finite-multiple-root image

This file discharges the two finite-component irreducibility obligations.  The
critical parametrization is treated as a regular map on the principal open
`{s ≠ 0}` in affine two-space.  The source is a nonempty open subset of an
irreducible affine space, and the critical image is its continuous image.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Ideal MvPolynomial Polynomial Set Topology

noncomputable section

/-! ## The structured parameter domain -/

/-- Ordinary complex affine two-space. -/
abbrev C2 := Fin 2 -> Complex

/-- Complex affine two-space with the point-set Zariski topology. -/
abbrev ZariskiC2 := ZariskiAffine (Fin 2)

/-- Regard a Euclidean `C2` point as the same point in `ZariskiC2`. -/
def toZariskiC2 : C2 -> ZariskiC2 :=
  toZariskiAffine

/-- Forget the Zariski topology on a parameter point. -/
def ofZariskiC2 : ZariskiC2 -> C2 :=
  ofZariskiAffine

@[simp]
theorem ofZariskiC2_toZariskiC2 (u : C2) :
    ofZariskiC2 (toZariskiC2 u) = u := rfl

@[simp]
theorem toZariskiC2_ofZariskiC2 (u : ZariskiC2) :
    toZariskiC2 (ofZariskiC2 u) = u := rfl

/-- The principal-open parameter set `s ≠ 0` on ordinary affine points. -/
def criticalDomainSet : Set C2 :=
  {u | u 1 ≠ 0}

/-- The same principal-open parameter set in the affine Zariski topology. -/
def criticalDomainZSet : Set ZariskiC2 :=
  ofZariskiC2 ⁻¹' criticalDomainSet

/-- The structured source of the critical parametrization. -/
abbrev CriticalDomain := criticalDomainZSet

/-- The free first parameter `p`. -/
def criticalP (u : CriticalDomain) : Complex :=
  ofZariskiC2 u.1 0

/-- The nonzero second parameter `s`. -/
def criticalS (u : CriticalDomain) : Complex :=
  ofZariskiC2 u.1 1

@[simp]
theorem criticalS_ne_zero (u : CriticalDomain) :
    criticalS u ≠ 0 := by
  have hu := u.2
  change ofZariskiC2 u.1 1 ≠ 0 at hu
  exact hu

/-- The old existential definition of the critical image is exactly the range
of the structured parameter domain. -/
theorem mem_criticalImage_iff_exists_domain
    (h : Complex[X]) (b : C3) :
    b ∈ criticalImage h ↔
      ∃ u : CriticalDomain,
        b = criticalTarget h (criticalP u) (criticalS u) := by
  constructor
  · rintro ⟨p, s, hs, rfl⟩
    let x : C2 := ![p, s]
    let u : CriticalDomain :=
      ⟨toZariskiC2 x, by simpa [criticalDomainZSet, criticalDomainSet, x] using hs⟩
    refine ⟨u, ?_⟩
    simp [u, x, criticalP, criticalS]
  · rintro ⟨u, rfl⟩
    exact ⟨criticalP u, criticalS u, criticalS_ne_zero u, rfl⟩

/-! ## The source is a nonempty irreducible principal open -/

/-- The excluded coordinate hyperplane `s = 0` is Zariski closed. -/
theorem critical_s_zero_isClosed :
    IsClosed {u : ZariskiC2 | ofZariskiC2 u 1 = 0} := by
  have hClosed := zariskiLiftAffine_zeroLocus_isClosed
    (Fin 2) (coordinateIdeal (1 : Fin 2) 0)
  rw [zeroLocus_coordinateIdeal] at hClosed
  simpa [zariskiLiftAffine, ofZariskiC2] using hClosed

/-- The critical parameter domain `s ≠ 0` is Zariski open. -/
theorem criticalDomain_isOpen :
    IsOpen criticalDomainZSet := by
  have hCompl :
      criticalDomainZSetᶜ =
        {u : ZariskiC2 | ofZariskiC2 u 1 = 0} := by
    ext u
    simp [criticalDomainZSet, criticalDomainSet]
  rw [← isClosed_compl_iff, hCompl]
  exact critical_s_zero_isClosed

/-- The point `(p,s) = (0,1)` lies in the principal-open domain. -/
theorem criticalDomain_nonempty :
    Nonempty CriticalDomain := by
  let x : C2 := ![0, 1]
  exact ⟨⟨toZariskiC2 x, by simp [criticalDomainZSet, criticalDomainSet, x]⟩⟩

/-- A nonempty open subset of irreducible affine two-space is irreducible. -/
theorem criticalDomainZSet_isIrreducible :
    IsIrreducible criticalDomainZSet := by
  have hAmbient : IsIrreducible (Set.univ : Set ZariskiC2) :=
    isIrreducible_zariskiAffineSpace (Fin 2)
  have hNonempty : criticalDomainZSet.Nonempty := by
    rcases criticalDomain_nonempty with ⟨u⟩
    exact ⟨u.1, u.2⟩
  exact hAmbient.isPreirreducible.subset_irreducible
    hNonempty criticalDomain_isOpen (Set.Subset.rfl) (Set.subset_univ _)

/-- The structured source is itself an irreducible topological space. -/
theorem criticalDomain_isIrreducible :
    IsIrreducible (Set.univ : Set CriticalDomain) := by
  letI : IrreducibleSpace CriticalDomain :=
    Subtype.irreducibleSpace criticalDomainZSet_isIrreducible
  exact IrreducibleSpace.isIrreducible_univ CriticalDomain

/-! ## Regular maps on a principal open -/

/-- One coordinate of a regular map on the principal open where `g ≠ 0`,
written as a polynomial numerator divided by a power of `g`. -/
structure PrincipalOpenCoordinate {σ : Type*}
    (g : MvPolynomial σ Complex) where
  numerator : MvPolynomial σ Complex
  denominatorPower : Nat

namespace PrincipalOpenCoordinate

/-- Evaluate a principal-open coordinate at an ordinary affine point. -/
def value {σ : Type*} {g : MvPolynomial σ Complex}
    (c : PrincipalOpenCoordinate g) (x : σ -> Complex) : Complex :=
  MvPolynomial.eval x c.numerator /
    (MvPolynomial.eval x g) ^ c.denominatorPower

end PrincipalOpenCoordinate

/-- The principal open `D(g)` in the affine Zariski type synonym. -/
def principalOpenSet {σ : Type*} (g : MvPolynomial σ Complex) :
    Set (ZariskiAffine σ) :=
  {x | MvPolynomial.eval (ofZariskiAffine x) g ≠ 0}

/-- The structured principal-open domain `D(g)`. -/
abbrev PrincipalOpenDomain {σ : Type*} (g : MvPolynomial σ Complex) :=
  principalOpenSet g

/-- Evaluating any target polynomial after a principal-open coordinate map
produces a single polynomial numerator over a power of the denominator. -/
theorem exists_principalOpen_pullback_fraction
    {σ τ : Type*}
    (g : MvPolynomial σ Complex)
    (coord : τ -> PrincipalOpenCoordinate g)
    (P : MvPolynomial τ Complex) :
    ∃ N : MvPolynomial σ Complex, ∃ k : Nat,
      ∀ x : σ -> Complex, MvPolynomial.eval x g ≠ 0 ->
        MvPolynomial.eval
            (fun i => PrincipalOpenCoordinate.value (coord i) x) P =
          MvPolynomial.eval x N / (MvPolynomial.eval x g) ^ k := by
  induction P using MvPolynomial.induction_on with
  | C a =>
      refine ⟨MvPolynomial.C a, 0, ?_⟩
      intro x hx
      simp [PrincipalOpenCoordinate.value]
  | add P Q hP hQ =>
      rcases hP with ⟨NP, kP, hP⟩
      rcases hQ with ⟨NQ, kQ, hQ⟩
      refine ⟨NP * g ^ kQ + NQ * g ^ kP, kP + kQ, ?_⟩
      intro x hx
      rw [map_add, hP x hx, hQ x hx]
      simp only [map_add, map_mul, map_pow]
      field_simp [pow_ne_zero _ hx, pow_add]
      ring
  | mul_X P i hP =>
      rcases hP with ⟨NP, kP, hP⟩
      refine ⟨NP * (coord i).numerator,
        kP + (coord i).denominatorPower, ?_⟩
      intro x hx
      rw [map_mul, MvPolynomial.eval_X, hP x hx]
      simp only [map_mul, PrincipalOpenCoordinate.value]
      field_simp [pow_ne_zero _ hx, pow_add]
      ring

/-- Chosen numerator of the pullback fraction. -/
noncomputable def principalOpenPullbackNumerator
    {σ τ : Type*}
    (g : MvPolynomial σ Complex)
    (coord : τ -> PrincipalOpenCoordinate g)
    (P : MvPolynomial τ Complex) : MvPolynomial σ Complex :=
  Classical.choose (exists_principalOpen_pullback_fraction g coord P)

/-- Chosen denominator power of the pullback fraction. -/
noncomputable def principalOpenPullbackPower
    {σ τ : Type*}
    (g : MvPolynomial σ Complex)
    (coord : τ -> PrincipalOpenCoordinate g)
    (P : MvPolynomial τ Complex) : Nat :=
  Classical.choose
    (Classical.choose_spec (exists_principalOpen_pullback_fraction g coord P))

/-- Specification of the chosen pullback numerator and denominator power. -/
theorem principalOpenPullback_spec
    {σ τ : Type*}
    (g : MvPolynomial σ Complex)
    (coord : τ -> PrincipalOpenCoordinate g)
    (P : MvPolynomial τ Complex)
    (x : σ -> Complex)
    (hx : MvPolynomial.eval x g ≠ 0) :
    MvPolynomial.eval
        (fun i => PrincipalOpenCoordinate.value (coord i) x) P =
      MvPolynomial.eval x (principalOpenPullbackNumerator g coord P) /
        (MvPolynomial.eval x g) ^
          (principalOpenPullbackPower g coord P) := by
  exact Classical.choose_spec
    (Classical.choose_spec (exists_principalOpen_pullback_fraction g coord P)) x hx

/-- The map determined by finitely many principal-open coordinates. -/
noncomputable def principalOpenMap
    {σ τ : Type*}
    (g : MvPolynomial σ Complex)
    (coord : τ -> PrincipalOpenCoordinate g) :
    PrincipalOpenDomain g -> ZariskiAffine τ :=
  fun u => toZariskiAffine
    (fun i => PrincipalOpenCoordinate.value (coord i)
      (ofZariskiAffine u.1))

/-- A vector-valued map whose coordinates are polynomial-over-a-power-of-`g`
is continuous for the affine Zariski topologies on `D(g)` and the target. -/
theorem continuous_principalOpen_map
    {σ τ : Type*}
    (g : MvPolynomial σ Complex)
    (coord : τ -> PrincipalOpenCoordinate g) :
    Continuous (principalOpenMap g coord) := by
  rw [continuous_iff_isClosed]
  intro Z hZ
  letI : TopologicalSpace (τ -> Complex) := affineZariskiTopology τ
  have hRawClosed :
      IsClosed (toZariskiAffine ⁻¹' Z : Set (τ -> Complex)) :=
    hZ.preimage
      (WithTopology.continuous_toTopology (affineZariskiTopology τ))
  rcases (isClosed_affineZariski_iff τ
    (toZariskiAffine ⁻¹' Z)).mp hRawClosed with ⟨I, hI⟩
  let pull : MvPolynomial τ Complex -> MvPolynomial σ Complex :=
    principalOpenPullbackNumerator g coord
  let J : Ideal (MvPolynomial σ Complex) :=
    Ideal.span (pull '' (I : Set (MvPolynomial τ Complex)))
  have hAmbientClosed :
      IsClosed
        (zariskiLiftAffine (MvPolynomial.zeroLocus Complex J) :
          Set (ZariskiAffine σ)) :=
    zariskiLiftAffine_zeroLocus_isClosed σ J
  have hSubClosed :
      IsClosed
        ((fun u : PrincipalOpenDomain g => u.1) ⁻¹'
          zariskiLiftAffine (MvPolynomial.zeroLocus Complex J)) :=
    hAmbientClosed.preimage continuous_subtype_val
  suffices hPreimage :
      (principalOpenMap g coord) ⁻¹' Z =
        (fun u : PrincipalOpenDomain g => u.1) ⁻¹'
          zariskiLiftAffine (MvPolynomial.zeroLocus Complex J) by
    rw [hPreimage]
    exact hSubClosed
  ext u
  let x : σ -> Complex := ofZariskiAffine u.1
  have hx : MvPolynomial.eval x g ≠ 0 := u.2
  have hTarget :
      (principalOpenMap g coord u ∈ Z) ↔
        (fun i => PrincipalOpenCoordinate.value (coord i) x) ∈
          MvPolynomial.zeroLocus Complex I := by
    change
      toZariskiAffine
          (fun i => PrincipalOpenCoordinate.value (coord i) x) ∈ Z ↔ _
    rw [← hI]
    rfl
  change
    (principalOpenMap g coord u ∈ Z) ↔
      x ∈ MvPolynomial.zeroLocus Complex J
  rw [hTarget]
  constructor
  · intro hPoint
    rw [MvPolynomial.zeroLocus_span]
    intro N hN
    rcases hN with ⟨P, hPI, rfl⟩
    have hEvalTarget :
        MvPolynomial.eval
            (fun i => PrincipalOpenCoordinate.value (coord i) x) P = 0 := by
      simpa [MvPolynomial.aeval_eq_eval] using hPoint P hPI
    have hSpec := principalOpenPullback_spec g coord P x hx
    rw [hEvalTarget] at hSpec
    have hDiv :
        MvPolynomial.eval x
              (principalOpenPullbackNumerator g coord P) /
            MvPolynomial.eval x g ^
              principalOpenPullbackPower g coord P = 0 :=
      hSpec.symm
    rcases (div_eq_zero_iff.mp hDiv) with hNumerator | hDenominator
    · simpa [pull] using hNumerator
    · exact ((pow_ne_zero _ hx) hDenominator).elim
  · intro hPoint P hPI
    have hGenerator : pull P ∈ J := by
      apply Ideal.subset_span
      exact ⟨P, hPI, rfl⟩
    have hNumerator : MvPolynomial.eval x (pull P) = 0 := by
      simpa [MvPolynomial.aeval_eq_eval] using hPoint (pull P) hGenerator
    rw [MvPolynomial.aeval_eq_eval]
    rw [principalOpenPullback_spec g coord P x hx]
    simp [pull, hNumerator]

/-- Scalar special case of `continuous_principalOpen_map`. -/
theorem continuous_principalOpen_coordinate
    {σ : Type*}
    (g : MvPolynomial σ Complex)
    (c : PrincipalOpenCoordinate g) :
    Continuous
      (fun u : PrincipalOpenDomain g =>
        toZariskiAffine
          (fun _ : Fin 1 => PrincipalOpenCoordinate.value c
            (ofZariskiAffine u.1))) := by
  change Continuous (principalOpenMap g (fun _ : Fin 1 => c))
  exact continuous_principalOpen_map g (fun _ : Fin 1 => c)

/-! ## The critical map as a principal-open regular map -/

/-- The denominator polynomial `s` on affine two-space. -/
def criticalDenominator : MvPolynomial (Fin 2) Complex :=
  MvPolynomial.X 1

/-- Embed a univariate polynomial as a polynomial in the first parameter `p`. -/
def polynomialAtP (h : Complex[X]) : MvPolynomial (Fin 2) Complex :=
  h.eval₂ MvPolynomial.C (MvPolynomial.X 0)

@[simp]
theorem eval_polynomialAtP (h : Complex[X]) (x : C2) :
    MvPolynomial.eval x (polynomialAtP h) = h.eval (x 0) := by
  induction h using Polynomial.induction_on' with
  | add p q hp hq =>
      rw [show polynomialAtP (p + q) =
          polynomialAtP p + polynomialAtP q by
        simp [polynomialAtP]]
      rw [map_add, hp, hq, Polynomial.eval_add]
  | monomial n a =>
      simp [polynomialAtP]

/-- The three principal-open coordinate expressions of the critical map. -/
def criticalCoordinates (h : Complex[X]) :
    Fin 3 -> PrincipalOpenCoordinate criticalDenominator
  | 0 =>
      { numerator := MvPolynomial.X 0
        denominatorPower := 0 }
  | 1 =>
      { numerator :=
          MvPolynomial.X 0 ^ 6 * polynomialAtP h * MvPolynomial.X 1 ^ 5 +
            MvPolynomial.C 3 * MvPolynomial.X 0 * MvPolynomial.X 1 ^ 2 + 1
        denominatorPower := 1 }
  | 2 =>
      { numerator :=
          MvPolynomial.X 1 - MvPolynomial.X 0 * MvPolynomial.X 1 ^ 3 -
            MvPolynomial.C (2 / 3 : Complex) * MvPolynomial.X 0 ^ 6 *
              polynomialAtP h * MvPolynomial.X 1 ^ 6
        denominatorPower := 0 }

@[simp]
theorem eval_criticalDenominator (x : C2) :
    MvPolynomial.eval x criticalDenominator = x 1 := by
  simp [criticalDenominator]

/-- The generic principal-open coordinate map evaluates to the explicit
critical target formula. -/
theorem criticalCoordinates_value
    (h : Complex[X]) (x : C2) (hs : x 1 ≠ 0) :
    (fun i => PrincipalOpenCoordinate.value (criticalCoordinates h i) x) =
      criticalTarget h (x 0) (x 1) := by
  funext i
  fin_cases i
  · simp [criticalCoordinates, PrincipalOpenCoordinate.value,
      criticalDenominator]
  · simp [criticalCoordinates, PrincipalOpenCoordinate.value,
      criticalDenominator, criticalTarget, criticalQ]
    field_simp [hs]
  · simp [criticalCoordinates, PrincipalOpenCoordinate.value,
      criticalDenominator, criticalTarget, criticalR]

/-- The structured critical domain is definitionally the principal open of
`criticalDenominator`, up to the transparent coordinate definitions. -/
theorem criticalDomainZSet_eq_principalOpen :
    criticalDomainZSet = principalOpenSet criticalDenominator := by
  ext u
  simp [criticalDomainZSet, criticalDomainSet, principalOpenSet,
    criticalDenominator, ofZariskiC2]

/-- The finite-multiple-root parametrization as a map of Zariski spaces. -/
noncomputable def criticalTargetZ
    (h : Complex[X]) : CriticalDomain -> ZariskiC3 :=
  fun u => toZariskiC3
    (criticalTarget h (criticalP u) (criticalS u))

/-- The Zariski critical parametrization is continuous. -/
theorem criticalTargetZ_continuous
    (h : Complex[X]) :
    Continuous (criticalTargetZ h) := by
  let e : CriticalDomain -> PrincipalOpenDomain criticalDenominator :=
    fun u => ⟨u.1, by
      change
        MvPolynomial.eval (ofZariskiAffine u.1) criticalDenominator ≠ 0
      simpa [eval_criticalDenominator, criticalS, ofZariskiC2] using
        criticalS_ne_zero u⟩
  have he : Continuous e := by
    exact continuous_subtype_val.subtype_mk _
  have hRegular :
      Continuous (principalOpenMap criticalDenominator (criticalCoordinates h)) :=
    continuous_principalOpen_map criticalDenominator (criticalCoordinates h)
  have hComp := hRegular.comp he
  apply hComp.congr
  intro u
  have hValue :=
    criticalCoordinates_value h (ofZariskiC2 u.1) (criticalS_ne_zero u)
  exact congrArg toZariskiC3 hValue

/-- The range of the structured Zariski map is exactly the lifted old critical
image. -/
theorem range_criticalTargetZ
    (h : Complex[X]) :
    Set.range (criticalTargetZ h) =
      zariskiLift (criticalImage h) := by
  ext b
  constructor
  · rintro ⟨u, rfl⟩
    change criticalTarget h (criticalP u) (criticalS u) ∈ criticalImage h
    exact (mem_criticalImage_iff_exists_domain h _).2 ⟨u, rfl⟩
  · intro hb
    change ofZariskiC3 b ∈ criticalImage h at hb
    rcases (mem_criticalImage_iff_exists_domain h _).1 hb with ⟨u, hu⟩
    refine ⟨u, ?_⟩
    have hWrapped := congrArg toZariskiC3 hu.symm
    simpa [criticalTargetZ] using hWrapped

/-- The finite-multiple-root image is irreducible for every deformation
polynomial `h`. -/
theorem criticalImage_isIrreducible
    (h : Complex[X]) :
    IsIrreducible (zariskiLift (criticalImage h)) := by
  have hImage := criticalDomain_isIrreducible.image
    (criticalTargetZ h) (criticalTargetZ_continuous h).continuousOn
  rw [Set.image_univ, range_criticalTargetZ] at hImage
  exact hImage

/-! ## Pair-specific endpoints -/

/-- Unconditional irreducibility of the CEX-004 finite component. -/
theorem cex004_finiteComponentIrreducible :
    FiniteComponentIrreducible004 :=
  cex004_finiteComponentIrreducible_of_criticalImage
    (criticalImage_isIrreducible eta004)

/-- Unconditional irreducibility of the CEX-006 finite component. -/
theorem cex006_finiteComponentIrreducible :
    FiniteComponentIrreducible006 :=
  cex006_finiteComponentIrreducible_of_criticalImage
    (criticalImage_isIrreducible eta006)

end

end DegreeSixKeller
