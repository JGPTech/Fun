# ETHICS.md — EchoKey Unified Reversible Coin (TSP & Chemistry)

**One‑liner:** Small‑N, measurement‑free **reversible** core (Bennett compute→use→uncompute) + explicit **inhomogeneous coined walk**. Demonstrated twice: **TSP** (exact ΔL) and **Chemistry** (exact ΔE).

> **Expectation (low‑key):** This is a **truth machine**, not a speed machine. No scaling claims. No hype. Clear guardrails.

---

## 1) Scope

* **Core:** ROM‑style, idempotent XOR loads ΔΦ into a work register; copy **sign only** to a flag; controlled marker rotation; **uncompute** back to |0…0⟩.
* **Pipelines:** identical for both domains; only INT changes: **INT = −ΔΦ** (ΔL for TSP, ΔE for Chem).
* **Instances:** tiny, explicit, audited. Padded states remain isolated by construction.

## 2) What we claim / don’t claim

**We claim:**

* Logical reversibility of the core; unitarity end‑to‑end; ancillas/work returned to zero; U†U ≈ I (FP tolerance).
* Exact local deltas (ΔL, ΔE) and transparent resource/metric logging.

**We do **not** claim:**

* Any polynomial scaling or quantum advantage.
* That logical reversibility eliminates physical energy cost.
* Hidden classical oracles inside U.
* People‑routing or chemistry synthesis capabilities.

## 3) Principles (short)

* **Landauer:** erasure has physical cost.
* **Bennett:** classical logic can be made **logically** reversible with compute→use→uncompute.
* **Feynman:** ideal quantum evolution is unitary/time‑reversible.
  We respect the logical side; we do not assert hardware energy wins.

## 4) Societal impact by track

### Reversible computing

* **Benefit:** concrete, auditable exemplar of reversibility.
* **Risk:** greenwashing (“reversible = free energy”).
* **Mitigation:** explicit energy appendix; limits slide; publish negative/neutral results.
* **Metrics:** ancilla‑zero rate, Pr\[Δ=0^b] (drop→return), U†U fidelity.

### Optimization (TSP)

* **Benefit:** emissions‑proxy routing demo; constraint hooks (no‑go, fairness caps).
* **Risk:** dual‑use for coercive logistics or surveillance.
* **Mitigation:** repo **USE\_POLICY.md** forbids people‑routing; constraints in code; synthetic data only.
* **Metrics:** optimality gap, symmetry‑aware hit rate (t or t\_rev), constraint satisfaction.

### Chemistry

* **Benefit:** clear reversible predicate on ΔE in a benign sandbox relevant to sustainability themes.
* **Risk:** pathway/synthesis misuse; over‑interpretation as scalable chemistry prediction.
* **Mitigation:** benign molecules only; no synthesis paths; scope disclaimers; property error bars vs baseline.
* **Metrics:** MAE/RMSE vs baseline; resource counts; U†U fidelity.

## 5) Risk register (concise)

| Risk                       | Harm                 | Likelihood | Mitigation                                              | Residual |
| -------------------------- | -------------------- | ---------: | ------------------------------------------------------- | -------- |
| “Reversible = free energy” | Public misperception |        Med | Energy appendix; explicit physical vs logical           | Low      |
| Dual‑use logistics         | Rights harms         |        Med | Forbid people‑routing; constraints; synthetic data      | Low‑Med  |
| Chemistry misuse           | Safety               |    Low‑Med | Benign sandbox only; scope note                         | Low      |
| Overclaiming quantum       | Misallocation        |        Med | Small‑N truth‑machine framing; show negatives           | Low      |
| Compute footprint          | CO₂e                 |        Med | Minimal runs; report wall‑time/energy; note uncertainty | Low      |

## 6) Use policy (excerpt)

**Allowed:** pedagogy; reversible‑logic demos; toy routing on synthetic instances; benign chemistry sandboxes.
**Prohibited:** people‑routing/surveillance; weapons/toxin design; any human‑centric deployment without consent + independent review.
**Review triggers:** moving to real‑world data; coupling to decision systems; attempting chemical synthesis.

## 7) Transparency & logging

We print or export:

* Reversible proofs: ancilla‑zero rate; Pr\[Δ=0^b] (compute vs sandwich); U†U fidelity; padded mass.
* Resources: qubits, gate counts, depth.
* Domain metrics: gaps, constraints, property errors.
* Repro tags: seeds, CLI flags, commit hash, host specs.

## 8) Environmental footprint (method)

Estimate energy from wall‑time and average device power, report **CO₂e** with stated grid factor:

```
E_est ≈ P_CPU^avg·t + P_GPU^avg·t
CO2e = E_est · (1 kWh / 3.6e6 J) · g_grid
```

This is a rough bound; we disclose uncertainty and do not normalize to misleading “per‑op” figures.

## 9) Human subjects, data, export

* **Human data:** none used. Any future real use would need consent + independent ethics review.
* **Export/dual‑use:** no synthesis/design tooling; users remain responsible for compliance with applicable laws.

## 10) Communication ethics (video/README)

* 20‑sec one‑liner; 60‑sec reversible‑core animation; 90‑sec TSP with constraints; 90‑sec chem with error bars; 45‑sec ethics & energy; 30‑sec limitations. No advantage claims.

## 11) Submission checklist (ethics)

* README expectations up front.
* **ETHICS.md** (this file) + **USE\_POLICY.md** in repo root.
* Logs: ancilla‑zero, Pr\[Δ=0^b], U†U; TSP gaps; Chem errors.
* Energy appendix: wall‑time, E\_est, g\_grid reference.
* Clear note that instances are tiny and synthetic.

## 12) References (indicative)

* R. Landauer (1961) *Irreversibility and Heat Generation in the Computing Process*.
* C. H. Bennett (1973) *Logical Reversibility of Computation*.
* R. P. Feynman (1982) *Simulating Physics with Computers*.
