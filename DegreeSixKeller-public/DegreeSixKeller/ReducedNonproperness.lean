import DegreeSixKeller.FetaCore
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.Homeomorph.Defs

/-!
# Sequence definition of nonproperness and its transport

This module formalizes the Euclidean sequence definition used in the human
proof and proves its transport under a target homeomorphism and a source
homeomorphism that explicitly preserves escaping sequences.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Filter Set Topology

/-- A sequence escapes every norm-bounded subset of `C^3`. -/
def Escapes (u : Nat -> C3) : Prop :=
  Tendsto (fun n => norm (u n)) atTop atTop

/-- Euclidean nonproperness set, defined by escaping source sequences. -/
def NonpropernessSet (F : C3 -> C3) : Set C3 :=
  {b | ∃ u : Nat -> C3,
    Escapes u ∧ Tendsto (fun n => F (u n)) atTop (nhds b)}

/--
Transport of nonproperness under left-right homeomorphisms, with the only
source-side properness requirement exposed as `hR`.
-/
theorem nonproperness_leftRight
    (F : C3 -> C3) (L R : C3 ≃ₜ C3)
    (hR : ∀ u : Nat -> C3,
      Escapes (fun n => R (u n)) ↔ Escapes u) :
    NonpropernessSet (fun x => L (F (R x))) =
      L '' NonpropernessSet F := by
  ext b
  constructor
  · rintro ⟨u, huEsc, huLim⟩
    let v : Nat -> C3 := fun n => R (u n)
    have hvEsc : Escapes v := (hR u).2 huEsc
    have hvLim : Tendsto (fun n => F (v n)) atTop (nhds (L.symm b)) := by
      have h := L.symm.continuous.continuousAt.tendsto.comp huLim
      change Tendsto (fun n => L.symm (L (F (R (u n))))) atTop
        (nhds (L.symm b)) at h
      simpa only [v, L.symm_apply_apply] using h
    exact ⟨L.symm b, ⟨v, hvEsc, hvLim⟩, L.apply_symm_apply b⟩
  · rintro ⟨b0, ⟨v, hvEsc, hvLim⟩, rfl⟩
    let u : Nat -> C3 := fun n => R.symm (v n)
    have hcomp : Escapes (fun n => R (u n)) := by
      simpa [u] using hvEsc
    have huEsc : Escapes u := (hR u).1 hcomp
    refine ⟨u, huEsc, ?_⟩
    have h := L.continuous.continuousAt.tendsto.comp hvLim
    change Tendsto (fun n => L (F (v n))) atTop (nhds (L b0)) at h
    simpa only [u, R.apply_symm_apply] using h

end DegreeSixKeller
