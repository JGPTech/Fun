import DegreeSixKeller.PairSpecificNonproperness
import Mathlib.Algebra.MvPolynomial.Monad
import Mathlib.Analysis.Normed.Group.Bounded
import Mathlib.Topology.Algebra.MvPolynomial

/-!
# Polynomial automorphisms and the left-right bridge

This module closes the deliberately deferred bridge from polynomial source
and target automorphisms to `AlgebraicEscapeLeftRightEquivalent`.

A `PolynomialAutomorphism` contains polynomial coordinate triples for a map
and its inverse, together with the two exact inverse identities.  From those
data we derive, rather than assume:

* the underlying equivalence of `C3`;
* its Euclidean homeomorphism;
* its affine-Zariski homeomorphism; and
* preservation of escaping sequences.

Consequently polynomial left-right equivalence implies the algebraic-escape
equivalence already ruled out for CEX-004 and CEX-006.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Filter Ideal Set Topology

noncomputable section

/-- Evaluation of a triple of multivariate polynomials as a self-map of
complex affine three-space. -/
def polynomialCoordinateMap
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex) : C3 -> C3 :=
  fun x i => MvPolynomial.eval x (P i)

/-- Substitution into a polynomial agrees with evaluation after applying its
polynomial coordinate map. -/
@[simp]
theorem eval_bind₁_polynomialCoordinateMap
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex)
    (p : MvPolynomial (Fin 3) Complex) (x : C3) :
    MvPolynomial.eval x (MvPolynomial.bind₁ P p) =
      MvPolynomial.eval (polynomialCoordinateMap P x) p := by
  change MvPolynomial.eval x (MvPolynomial.bind₁ P p) =
    MvPolynomial.eval (fun i => MvPolynomial.eval x (P i)) p
  simpa only [MvPolynomial.aeval_eq_eval] using
    (MvPolynomial.aeval_bind₁ x P p)

/-- A polynomial coordinate map is continuous for the Euclidean topology. -/
theorem continuous_polynomialCoordinateMap
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex) :
    Continuous (polynomialCoordinateMap P) := by
  apply continuous_pi
  intro i
  exact (P i).continuous_eval

/-- The inverse image of an affine zero locus under a polynomial coordinate
map is again an affine zero locus, obtained by mapping the ideal along
polynomial substitution. -/
theorem preimage_zeroLocus_polynomialCoordinateMap
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex)
    (I : Ideal (MvPolynomial (Fin 3) Complex)) :
    polynomialCoordinateMap P ⁻¹'
        MvPolynomial.zeroLocus Complex I =
      MvPolynomial.zeroLocus Complex
        (I.map (MvPolynomial.bind₁ P).toRingHom) := by
  ext x
  constructor
  · intro hx q hq
    have hMap :
        I.map (MvPolynomial.bind₁ P).toRingHom ≤
          RingHom.ker (MvPolynomial.aeval x).toRingHom := by
      rw [Ideal.map_le_iff_le_comap]
      intro p hp
      change MvPolynomial.aeval x (MvPolynomial.bind₁ P p) = 0
      rw [MvPolynomial.aeval_bind₁]
      exact hx p hp
    exact hMap hq
  · intro hx p hp
    have hSub := hx (MvPolynomial.bind₁ P p)
      (Ideal.mem_map_of_mem (MvPolynomial.bind₁ P).toRingHom hp)
    simpa [polynomialCoordinateMap, MvPolynomial.aeval_eq_eval] using hSub

/-- A polynomial coordinate map is continuous for the genuine affine-Zariski
topology on the underlying affine space. -/
theorem continuous_polynomialCoordinateMap_affineZariski
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex) :
    @Continuous C3 C3 (affineZariskiTopology (Fin 3))
      (affineZariskiTopology (Fin 3)) (polynomialCoordinateMap P) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  rw [continuous_iff_isClosed]
  intro Z hZ
  rcases (isClosed_affineZariski_iff (Fin 3) Z).mp hZ with ⟨I, rfl⟩
  rw [preimage_zeroLocus_polynomialCoordinateMap]
  exact (isClosed_affineZariski_iff (Fin 3) _).mpr ⟨_, rfl⟩

/-- A polynomial self-map of affine space, transported to the type carrying
the Zariski topology. -/
def zariskiPolynomialCoordinateMap
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex) : ZariskiC3 -> ZariskiC3 :=
  fun x => toZariskiC3 (polynomialCoordinateMap P (ofZariskiC3 x))

/-- Zariski continuity of a polynomial coordinate map. -/
theorem continuous_zariskiPolynomialCoordinateMap
    (P : Fin 3 -> MvPolynomial (Fin 3) Complex) :
    Continuous (zariskiPolynomialCoordinateMap P) := by
  letI : TopologicalSpace C3 := affineZariskiTopology (Fin 3)
  exact (WithTopology.continuous_toTopology (affineZariskiTopology (Fin 3))).comp
    ((continuous_polynomialCoordinateMap_affineZariski P).comp
      (WithTopology.continuous_ofTopology
        (affineZariskiTopology (Fin 3))))

/-- A polynomial automorphism of complex affine three-space, represented by
polynomial coordinate triples in both directions and exact inverse laws. -/
structure PolynomialAutomorphism where
  forward : Fin 3 -> MvPolynomial (Fin 3) Complex
  inverse : Fin 3 -> MvPolynomial (Fin 3) Complex
  left_inv : Function.LeftInverse
    (polynomialCoordinateMap inverse) (polynomialCoordinateMap forward)
  right_inv : Function.RightInverse
    (polynomialCoordinateMap inverse) (polynomialCoordinateMap forward)

namespace PolynomialAutomorphism

/-- The underlying point-set equivalence of a polynomial automorphism. -/
def toEquiv (A : PolynomialAutomorphism) : C3 ≃ C3 where
  toFun := polynomialCoordinateMap A.forward
  invFun := polynomialCoordinateMap A.inverse
  left_inv := A.left_inv
  right_inv := A.right_inv

instance : CoeFun PolynomialAutomorphism (fun _ => C3 -> C3) :=
  ⟨fun A => A.toEquiv⟩

@[simp]
theorem coe_toEquiv (A : PolynomialAutomorphism) :
    (A.toEquiv : C3 -> C3) = A := rfl

/-- The Euclidean homeomorphism underlying a polynomial automorphism. -/
def toEuclideanHomeomorph (A : PolynomialAutomorphism) : C3 ≃ₜ C3 where
  toEquiv := A.toEquiv
  continuous_toFun := continuous_polynomialCoordinateMap A.forward
  continuous_invFun := continuous_polynomialCoordinateMap A.inverse

/-- The underlying equivalence on affine space carrying the Zariski topology. -/
def toZariskiEquiv (A : PolynomialAutomorphism) : ZariskiC3 ≃ ZariskiC3 where
  toFun := zariskiPolynomialCoordinateMap A.forward
  invFun := zariskiPolynomialCoordinateMap A.inverse
  left_inv := by
    intro x
    change toZariskiC3
      (polynomialCoordinateMap A.inverse
        (polynomialCoordinateMap A.forward (ofZariskiC3 x))) = x
    rw [A.left_inv (ofZariskiC3 x), toZariskiC3_ofZariskiC3]
  right_inv := by
    intro x
    change toZariskiC3
      (polynomialCoordinateMap A.forward
        (polynomialCoordinateMap A.inverse (ofZariskiC3 x))) = x
    rw [A.right_inv (ofZariskiC3 x), toZariskiC3_ofZariskiC3]

/-- The affine-Zariski homeomorphism underlying a polynomial automorphism. -/
def toZariskiHomeomorph (A : PolynomialAutomorphism) :
    ZariskiC3 ≃ₜ ZariskiC3 where
  toEquiv := A.toZariskiEquiv
  continuous_toFun := continuous_zariskiPolynomialCoordinateMap A.forward
  continuous_invFun := continuous_zariskiPolynomialCoordinateMap A.inverse

/-- Compatibility of the Euclidean and Zariski realizations of the same
polynomial automorphism. -/
theorem zariskiCompatible (A : PolynomialAutomorphism) :
    ZariskiCompatible A.toEuclideanHomeomorph A.toZariskiHomeomorph := by
  intro x
  rfl

end PolynomialAutomorphism

/-- Every homeomorphism of `C3` carries norm-escaping sequences to
norm-escaping sequences.  This is a properness fact and is independent of
polynomiality. -/
theorem escapes_homeomorph (R : C3 ≃ₜ C3) (u : Nat -> C3) :
    Escapes (fun n => R (u n)) ↔ Escapes u := by
  constructor
  · intro h
    rw [Escapes, tendsto_norm_atTop_iff_cobounded,
      Metric.cobounded_eq_cocompact] at h ⊢
    have hBack := R.symm.isClosedEmbedding.tendsto_cocompact.comp h
    have hFunction : (R.symm ∘ fun n => R (u n)) = u := by
      funext n
      exact R.symm_apply_apply (u n)
    rw [hFunction] at hBack
    exact hBack
  · intro h
    rw [Escapes, tendsto_norm_atTop_iff_cobounded,
      Metric.cobounded_eq_cocompact] at h ⊢
    exact R.isClosedEmbedding.tendsto_cocompact.comp h

/-- Left-right equivalence using genuine polynomial automorphisms on source
and target. -/
def PolynomialLeftRightEquivalent (F G : C3 -> C3) : Prop :=
  ∃ L R : PolynomialAutomorphism,
    G = fun x => L (F (R x))

/-- Polynomial left-right equivalence supplies every datum in the established
algebraic-escape equivalence interface. -/
theorem algebraicEscapeLeftRightEquivalent_of_polynomialLeftRightEquivalent
    {F G : C3 -> C3}
    (h : PolynomialLeftRightEquivalent F G) :
    AlgebraicEscapeLeftRightEquivalent F G := by
  rcases h with ⟨L, R, hMap⟩
  exact ⟨L.toEuclideanHomeomorph, R.toEuclideanHomeomorph,
    L.toZariskiHomeomorph, L.zariskiCompatible,
    escapes_homeomorph R.toEuclideanHomeomorph, hMap⟩

/-- CEX-004 and CEX-006 are not left-right equivalent by polynomial
automorphisms. -/
theorem cex004_cex006_not_polynomialLeftRightEquivalent :
    ¬ PolynomialLeftRightEquivalent F004 F006 := by
  intro h
  exact cex004_cex006_not_algebraicEscapeEquivalent
    (algebraicEscapeLeftRightEquivalent_of_polynomialLeftRightEquivalent h)

end

end DegreeSixKeller
