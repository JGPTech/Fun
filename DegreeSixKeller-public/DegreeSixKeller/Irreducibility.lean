import DegreeSixKeller.ReducedNonproperness
import DegreeSixKeller.AffineZariski
import Mathlib.Topology.Homeomorph.Lemmas

/-!
# Honest interface to the irreducible-component frontier

The geometric proof that the reduced nonproperness sets have component counts
three and two is not hidden here as an axiom.  Instead, this module defines the
exact invariance property a future irreducible-component implementation must
satisfy and proves the transport bridge from Section C of the human proof.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Set Topology

/-- Invariance required of a component-count implementation. -/
def ComponentCountInvariant (componentCount : Set C3 -> Nat) : Prop :=
  ∀ (L : C3 ≃ₜ C3) (S : Set C3),
    componentCount (L '' S) = componentCount S

/--
Left-right equivalence at the exact topological boundary needed here.
Polynomial automorphisms enter later by proving that they produce these data.
-/
def EscapeLeftRightEquivalent (F G : C3 -> C3) : Prop :=
  ∃ L R : C3 ≃ₜ C3,
    (∀ u : Nat -> C3,
      Escapes (fun n => R (u n)) ↔ Escapes u) ∧
    G = fun x => L (F (R x))

theorem componentCount_eq_of_escapeLeftRightEquivalent
    (componentCount : Set C3 -> Nat)
    (hInv : ComponentCountInvariant componentCount)
    {F G : C3 -> C3}
    (hEq : EscapeLeftRightEquivalent F G) :
    componentCount (NonpropernessSet G) =
      componentCount (NonpropernessSet F) := by
  rcases hEq with ⟨L, R, hR, hmap⟩
  calc
    componentCount (NonpropernessSet G) =
        componentCount (NonpropernessSet (fun x => L (F (R x)))) := by
          rw [hmap]
    _ = componentCount (L '' NonpropernessSet F) := by
          rw [nonproperness_leftRight F L R hR]
    _ = componentCount (NonpropernessSet F) := hInv L (NonpropernessSet F)

/-- A homeomorphism induces a bijection on genuine irreducible components. -/
theorem irreducibleComponentNcard_eq_of_homeomorph
    {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y]
    (e : X ≃ₜ Y) :
    (irreducibleComponents X).ncard =
      (irreducibleComponents Y).ncard := by
  let componentEquiv :
      irreducibleComponents Y ≃o irreducibleComponents X :=
    irreducibleComponentsEquivOfIsPreirreducibleFiber
      e e.continuous e.isOpenMap
      (fun y => by
        apply Set.Subsingleton.isPreirreducible
        intro a ha b hb
        simp only [Set.mem_preimage, Set.mem_singleton_iff] at ha hb
        exact e.injective (ha.trans hb.symm))
      e.surjective
  exact Set.ncard_congr' componentEquiv.symm.toEquiv

/-- Compatibility between a Euclidean target homeomorphism and the same
underlying map regarded as a Zariski homeomorphism. -/
def ZariskiCompatible (L : C3 ≃ₜ C3)
    (Lz : ZariskiC3 ≃ₜ ZariskiC3) : Prop :=
  ∀ x : C3, ofZariskiC3 (Lz (toZariskiC3 x)) = L x

/-- The genuine algebraic component count is invariant under a target map
that is a homeomorphism for both the Euclidean and affine-Zariski topologies. -/
theorem algebraicComponentCount_image
    (L : C3 ≃ₜ C3) (Lz : ZariskiC3 ≃ₜ ZariskiC3)
    (hCompatible : ZariskiCompatible L Lz)
    (S : Set C3) :
    algebraicComponentCount (L '' S) = algebraicComponentCount S := by
  have hIff (x : ZariskiC3) :
      x ∈ zariskiLift S ↔ Lz x ∈ zariskiLift (L '' S) := by
    constructor
    · intro hx
      refine ⟨ofZariskiC3 x, hx, ?_⟩
      exact (hCompatible (ofZariskiC3 x)).symm
    · rintro ⟨a, ha, hLa⟩
      have hLx : ofZariskiC3 (Lz x) = L (ofZariskiC3 x) := by
        simpa using hCompatible (ofZariskiC3 x)
      have haEq : a = ofZariskiC3 x :=
        L.injective (hLa.trans hLx)
      change ofZariskiC3 x ∈ S
      simpa [haEq] using ha
  let eS : ZariskiSubspace S ≃ₜ ZariskiSubspace (L '' S) :=
    Lz.subtype hIff
  rw [algebraicComponentCount, algebraicComponentCount]
  exact (irreducibleComponentNcard_eq_of_homeomorph eS).symm

/-- Left-right equivalence with the target algebraicity data needed by the
reduced algebraic component invariant made explicit. -/
def AlgebraicEscapeLeftRightEquivalent (F G : C3 -> C3) : Prop :=
  ∃ (L R : C3 ≃ₜ C3) (Lz : ZariskiC3 ≃ₜ ZariskiC3),
    ZariskiCompatible L Lz ∧
    (∀ u : Nat -> C3,
      Escapes (fun n => R (u n)) ↔ Escapes u) ∧
    G = fun x => L (F (R x))

theorem algebraicComponentCount_eq_of_algebraicEscapeLeftRightEquivalent
    {F G : C3 -> C3}
    (hEq : AlgebraicEscapeLeftRightEquivalent F G) :
    algebraicComponentCount (NonpropernessSet G) =
      algebraicComponentCount (NonpropernessSet F) := by
  rcases hEq with ⟨L, R, Lz, hCompatible, hR, hMap⟩
  calc
    algebraicComponentCount (NonpropernessSet G) =
        algebraicComponentCount
          (NonpropernessSet (fun x => L (F (R x)))) := by rw [hMap]
    _ = algebraicComponentCount (L '' NonpropernessSet F) := by
      rw [nonproperness_leftRight F L R hR]
    _ = algebraicComponentCount (NonpropernessSet F) :=
      algebraicComponentCount_image L Lz hCompatible _

end DegreeSixKeller
