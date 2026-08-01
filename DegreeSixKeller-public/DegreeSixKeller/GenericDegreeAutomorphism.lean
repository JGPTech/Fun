import DegreeSixKeller.GenericDegree
import DegreeSixKeller.PolynomialAutomorphism

/-!
# Generic degree and polynomial automorphisms

A polynomial automorphism generates the full source function field: each
source coordinate is obtained by substituting the forward coordinates into
the corresponding inverse coordinate polynomial.  Its generic degree is
therefore one.  Combined with the degree-six computation for the Keller
family, this rules out polynomial inverses for every nontrivial family member
and, in particular, for CEX-004 and CEX-006.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open IntermediateField MvPolynomial Set

noncomputable section

namespace GenericDegreeAutomorphism

/-- Predicate asserting that a coordinate triple is the forward triple of a
genuine polynomial automorphism. -/
def IsPolynomialAutomorphism (P : PolynomialMap3) : Prop :=
  ∃ A : PolynomialAutomorphism, A.forward = P

/-- Substitution of the forward coordinates into an inverse coordinate
recovers the corresponding source variable. -/
theorem PolynomialAutomorphism.bind_forward_inverse
    (A : PolynomialAutomorphism) (i : Fin 3) :
    MvPolynomial.bind₁ A.forward (A.inverse i) = MvPolynomial.X i := by
  apply MvPolynomial.funext
  intro x
  rw [eval_bind₁_polynomialCoordinateMap]
  have hi := congrFun (A.left_inv x) i
  simpa [polynomialCoordinateMap] using hi

private theorem sourcePolynomial_bind_mem_coordinateFunctionField
    (P : PolynomialMap3) (f : MvPolynomial (Fin 3) Complex) :
    sourcePolynomialInclusion (MvPolynomial.bind₁ P f) ∈
      coordinateFunctionField P := by
  induction f using MvPolynomial.induction_on with
  | C c =>
      simp
  | add f g hf hg =>
      simp only [map_add]
      exact (coordinateFunctionField P).add_mem hf hg
  | mul_X f i hf =>
      simp only [map_mul, MvPolynomial.bind₁_X_right]
      apply (coordinateFunctionField P).mul_mem hf
      change coordinateRationalFunction P i ∈ coordinateFunctionField P
      exact IntermediateField.subset_adjoin Complex _ ⟨i, rfl⟩

/-- The coordinate functions of a polynomial automorphism generate the full
source rational function field. -/
theorem PolynomialAutomorphism.coordinateFunctionField_forward_eq_top
    (A : PolynomialAutomorphism) :
    coordinateFunctionField A.forward = ⊤ := by
  apply top_unique
  rw [← sourceCoordinateFunctionField_eq_top]
  apply IntermediateField.adjoin_le_iff.mpr
  rintro _ ⟨i, rfl⟩
  change sourcePolynomialInclusion (MvPolynomial.X i) ∈
    coordinateFunctionField A.forward
  rw [← GenericDegreeAutomorphism.PolynomialAutomorphism.bind_forward_inverse A i]
  exact sourcePolynomial_bind_mem_coordinateFunctionField A.forward (A.inverse i)

/-- A polynomial automorphism has generic degree one. -/
theorem PolynomialAutomorphism.genericDegree_forward_eq_one
    (A : PolynomialAutomorphism) :
    genericDegree A.forward = 1 := by
  rw [genericDegree,
    GenericDegreeAutomorphism.PolynomialAutomorphism.coordinateFunctionField_forward_eq_top A]
  exact IntermediateField.finrank_top

/-- Any polynomial map admitting a polynomial inverse has generic degree
one. -/
theorem genericDegree_eq_one_of_isPolynomialAutomorphism
    {P : PolynomialMap3} (hP : IsPolynomialAutomorphism P) :
    genericDegree P = 1 := by
  rcases hP with ⟨A, rfl⟩
  exact
    GenericDegreeAutomorphism.PolynomialAutomorphism.genericDegree_forward_eq_one A

/-- Every nontrivial member of the degree-six family fails to be a
polynomial automorphism. -/
theorem Fh_notPolynomialAutomorphism
    (h : Polynomial Complex) (hh : h ≠ 0) :
    ¬ IsPolynomialAutomorphism (FhPolynomial h) := by
  intro hauto
  have hone := genericDegree_eq_one_of_isPolynomialAutomorphism hauto
  rw [Fh_genericDegree_six h hh] at hone
  omega

theorem F004_notPolynomialAutomorphism :
    ¬ IsPolynomialAutomorphism F004Polynomial := by
  simpa [F004Polynomial] using
    Fh_notPolynomialAutomorphism eta004 eta004_ne_zero

theorem F006_notPolynomialAutomorphism :
    ¬ IsPolynomialAutomorphism F006Polynomial := by
  simpa [F006Polynomial] using
    Fh_notPolynomialAutomorphism eta006 eta006_ne_zero

end GenericDegreeAutomorphism

export GenericDegreeAutomorphism
  (IsPolynomialAutomorphism genericDegree_eq_one_of_isPolynomialAutomorphism
    Fh_notPolynomialAutomorphism F004_notPolynomialAutomorphism
    F006_notPolynomialAutomorphism)

end

end DegreeSixKeller
