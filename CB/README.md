# tetranomboid_modZ
**From chalkboard babble to computational model.**  
We take a viral blackboard “proof” full of ⊗, ≥, and `mod ℤ`—and give it real semantics:
- encode manifolds as feature tensors,
- compare them with a mixed topology+geometry score,
- then quotient the integer topology **mod q** to expose a fractional geometric residue.

> here, I fixed this for you. ❤️

---

## TL;DR
We compare simple 2-D manifolds—sphere \(S^2\), torus \(T^2\), and a constant-negative curvature patch \(H^2_a\)—via:
- **Geometric fingerprint** \( \phi(M) = [\langle K\rangle,\ \sigma_K,\ A,\ D] \)
- **Topology vector** \( \tau(M) = (\chi,\ b_0,\ b_1,\ b_2) \)
- **Score**
  \[
  S(M_i,M_j)=\|W_t(\tau_i-\tau_j)\|_1+\|W_g(\phi_i-\phi_j)\|_2
  \]
- **Residue** \(S_{\text{mod}}=\mathrm{frac}_q\big(\|W_g(\phi_i-\phi_j)\|_2\big)\) with \(q=1\Rightarrow \text{mod }\mathbb Z\).

---

## Quick start
```bash
python tetranomboid_modZ.py --seed 0 --plot
