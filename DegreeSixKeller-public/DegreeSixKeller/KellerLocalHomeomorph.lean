import DegreeSixKeller.JacobianKeller
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.InverseFunctionTheorem.FDeriv
import Mathlib.LinearAlgebra.Matrix.ToLinearEquiv
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Topology.IsLocalHomeomorph

/-!
# Keller polynomial maps are local homeomorphisms

The Fréchet derivative of a polynomial coordinate map is the evaluated formal
Jacobian.  For a Keller map its determinant is everywhere nonzero, so the
inverse function theorem gives a local homeomorphism at every source point.
-/

set_option autoImplicit false

namespace DegreeSixKeller

open Function MvPolynomial Set
open scoped BigOperators

noncomputable section

/-- The formal Jacobian evaluated at a source point. -/
def evaluatedFormalJacobian (P : PolynomialMap3) (x : C3) :
    Matrix (Fin 3) (Fin 3) Complex :=
  (MvPolynomial.eval x).mapMatrix (formalJacobian P)

/-- Evaluation commutes with the determinant of the formal Jacobian. -/
theorem evaluatedFormalJacobian_det (P : PolynomialMap3) (x : C3) :
    (evaluatedFormalJacobian P x).det =
      MvPolynomial.eval x (jacobianDet P) := by
  exact ((MvPolynomial.eval x).map_det (formalJacobian P)).symm

/-- The scalar-valued differential of a multivariate polynomial, written in
the coordinate basis. -/
def polynomialGradientCLM
    (f : MvPolynomial (Fin 3) Complex) (x : C3) : C3 →L[Complex] Complex :=
  ∑ j : Fin 3,
    MvPolynomial.eval x (MvPolynomial.pderiv j f) •
      (ContinuousLinearMap.proj j : C3 →L[Complex] Complex)

/-- Polynomial evaluation has strict derivative given by its evaluated formal
partial derivatives. -/
theorem hasStrictFDerivAt_eval
    (f : MvPolynomial (Fin 3) Complex) (x : C3) :
    HasStrictFDerivAt (fun u : C3 => MvPolynomial.eval u f)
      (polynomialGradientCLM f x) x := by
  induction f using MvPolynomial.induction_on with
  | C c =>
      have hclm : polynomialGradientCLM (MvPolynomial.C c) x = 0 := by
        ext v
        simp [polynomialGradientCLM]
      rw [hclm]
      simpa using (hasStrictFDerivAt_const (E := C3) (𝕜 := Complex) c x)
  | add f g hf hg =>
      have hclm : polynomialGradientCLM (f + g) x =
          polynomialGradientCLM f x + polynomialGradientCLM g x := by
        unfold polynomialGradientCLM
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro j _hj
        rw [map_add, map_add]
        exact add_smul
          (MvPolynomial.eval x (MvPolynomial.pderiv j f))
          (MvPolynomial.eval x (MvPolynomial.pderiv j g))
          (ContinuousLinearMap.proj j : C3 →L[Complex] Complex)
      have hfun : (fun u : C3 => MvPolynomial.eval u (f + g)) =
          (fun u : C3 => MvPolynomial.eval u f) +
            (fun u : C3 => MvPolynomial.eval u g) := by
        funext u
        exact map_add (MvPolynomial.eval u) f g
      rw [hfun, hclm]
      exact hf.add hg
  | mul_X f i hf =>
      have hi : HasStrictFDerivAt (fun u : C3 => u i)
          (ContinuousLinearMap.proj i : C3 →L[Complex] Complex) x :=
        hasStrictFDerivAt_apply (𝕜 := Complex) i x
      have hclm : polynomialGradientCLM (f * MvPolynomial.X i) x =
          MvPolynomial.eval x f •
              (ContinuousLinearMap.proj i : C3 →L[Complex] Complex) +
            x i • polynomialGradientCLM f x := by
        fin_cases i <;>
          ext v <;>
          simp [polynomialGradientCLM, MvPolynomial.pderiv_X,
            Fin.sum_univ_succ] <;>
          ring
      have hfun :
          (fun u : C3 => MvPolynomial.eval u (f * MvPolynomial.X i)) =
            (fun u : C3 => MvPolynomial.eval u f) * (fun u : C3 => u i) := by
        funext u
        simp
      rw [hfun, hclm]
      exact hf.mul hi

/-- The continuous linear map represented by the evaluated formal Jacobian. -/
def evaluatedFormalJacobianCLM (P : PolynomialMap3) (x : C3) :
    C3 →L[Complex] C3 :=
  LinearMap.toContinuousLinearMap (Matrix.toLin' (evaluatedFormalJacobian P x))

theorem evaluatedFormalJacobianCLM_apply
    (P : PolynomialMap3) (x v : C3) :
    evaluatedFormalJacobianCLM P x v =
      (evaluatedFormalJacobian P x).mulVec v := by
  rfl

/-- The analytic derivative of a polynomial coordinate map is its evaluated
formal Jacobian. -/
theorem hasStrictFDerivAt_polynomialMap_eval
    (P : PolynomialMap3) (x : C3) :
    HasStrictFDerivAt (PolynomialMap3.eval P)
      (evaluatedFormalJacobianCLM P x) x := by
  rw [show evaluatedFormalJacobianCLM P x =
      ContinuousLinearMap.pi (fun i => polynomialGradientCLM (P i) x) by
    ext v i
    simp [evaluatedFormalJacobianCLM, polynomialGradientCLM,
      evaluatedFormalJacobian, formalJacobian, Matrix.mulVec,
      dotProduct]]
  exact hasStrictFDerivAt_pi.mpr (fun i => hasStrictFDerivAt_eval (P i) x)

/-- The determinant of the analytic derivative is the pointwise evaluation
of the formal Jacobian determinant. -/
theorem evaluatedFormalJacobianCLM_det (P : PolynomialMap3) (x : C3) :
    (evaluatedFormalJacobianCLM P x).det =
      MvPolynomial.eval x (jacobianDet P) := by
  rw [evaluatedFormalJacobianCLM, LinearMap.det_toContinuousLinearMap,
    LinearMap.det_toLin']
  exact evaluatedFormalJacobian_det P x

/-- A Keller certificate makes the analytic derivative nonsingular at every
point. -/
theorem evaluatedFormalJacobianCLM_det_ne_zero
    {P : PolynomialMap3} (hP : IsKeller P) (x : C3) :
    (evaluatedFormalJacobianCLM P x).det ≠ 0 := by
  rcases hP with ⟨c, hc, hdet⟩
  rw [evaluatedFormalJacobianCLM_det, hdet, MvPolynomial.eval_C]
  exact hc

/-- The derivative equivalence canonically extracted from a Keller
certificate at a point. -/
def kellerDerivativeEquiv
    {P : PolynomialMap3} (hP : IsKeller P) (x : C3) : C3 ≃L[Complex] C3 :=
  (evaluatedFormalJacobianCLM P x).toContinuousLinearEquivOfDetNeZero
    (evaluatedFormalJacobianCLM_det_ne_zero hP x)

/-- The inverse function theorem gives a local chart around every point of a
Keller polynomial map. -/
theorem isLocalHomeomorph_polynomialMap_eval
    {P : PolynomialMap3} (hP : IsKeller P) :
    IsLocalHomeomorph (PolynomialMap3.eval P) := by
  intro x
  let e := kellerDerivativeEquiv hP x
  have hstrict : HasStrictFDerivAt (PolynomialMap3.eval P)
      (e : C3 →L[Complex] C3) x := by
    simpa [e, kellerDerivativeEquiv] using
      hasStrictFDerivAt_polynomialMap_eval P x
  exact ⟨hstrict.toOpenPartialHomeomorph (PolynomialMap3.eval P),
    hstrict.mem_toOpenPartialHomeomorph_source, rfl⟩

end

end DegreeSixKeller
