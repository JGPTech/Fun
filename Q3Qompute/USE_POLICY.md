# USE\_POLICY.md — EchoKey Unified Reversible Coin (TSP & Chemistry)

**Purpose.** Keep usage safe, honest, and reproducible. This project is a **small‑N, truth‑machine** demonstration of logical **reversibility** and explicit unitary accounting. It is **not** a product, not an energy‑saver claim, and not a human‑decision tool.

---

## 1) Scope

This policy applies to all code, documents, figures, and example data in this repository (the “Work”). The Work includes:

* Reversible predicate (“Bennett sandwich”) circuits and tests
* Inhomogeneous coin + shift (“coined walk”) demos for **TSP** and **Chemistry**
* Readme, ethics materials, and any generated logs/metrics

**Definitions (short):**

* **Small‑N truth machine:** deliberately tiny instances used to demonstrate correctness/traceability.
* **Reversible predicate:** compute→use→uncompute, no mid‑circuit measurement; work/ancillas reset to |0…0⟩.
* **Synthetic data:** data created for demos; no human subjects.

---

## 2) Allowed Uses

You may, under the license (CC0‑1.0):

* Use for **pedagogy**, research on reversible logic, or unit testing of unitaries
* Run **toy routing** on synthetic TSP instances; adjust features/weights
* Run **benign chemistry sandboxes** (e.g., basic property probes with ΔE tables)
* Audit, benchmark, or fork the Work with clear attribution (appreciated, not required)
* Extend with additional **synthetic** or **benign** examples, keeping the same guardrails

---

## 3) Prohibited Uses

You must **not** use the Work to:

* **Optimize/route people** or operate surveillance, profiling, or coercive logistics
* Deploy **human‑impacting decisions** without **informed consent** and **independent ethics review**
* Design, optimize, or evaluate **hazardous chemical/biological agents**, weapons, or dual‑use synthesis
* Make or market **misleading claims** (e.g., “reversible = free energy”, implied quantum advantage, or scaling claims)
* Hide classical oracles inside unitary claims, or falsify/omit metrics to suggest performance that isn’t there
* Train, tune, or test on **personal data** or sensitive information without a lawful basis and consent
* Violate applicable **law**, **sanctions**, or **export‑control** regulations

---

## 4) Review‑Required Uses (get a green light first)

Before proceeding, obtain an independent ethics/security review if you intend to:

* Use **real‑world operational data** (routes, customers, patients, workers, etc.)
* Couple outputs to **any decision system** (dispatch, eligibility, triage, resource assignment)
* Move beyond **benign chemistry** (any synthesis path, activity prediction, toxicity‑relevant property)
* Make **hardware energy** claims (provide methods, measurements, and uncertainty)

> If in doubt: treat as prohibited until reviewed.

---

## 5) Transparency, Reproducibility, and Claims

* Log and preserve: **seeds**, **CLI flags**, **commit hash**, **host specs**, and **metrics** (fidelity, ancilla‑zero, Δ register probes, gate/depth/qubits, domain metrics)
* Clearly state **limits**: “small‑N truth machine,” no scaling/advantage claims
* Do not normalize energy to misleading **per‑op** numbers; disclose grid factors and uncertainty

---

## 6) Data & Privacy

* Use **synthetic** or properly licensed public data for demos
* No ingestion of **PII** or sensitive data without consent and a lawful basis
* Minimize retention; document any real‑data handling and delete when no longer needed

---

## 7) Environmental Reporting (optional but encouraged)

If you publish runs, include wall‑time and a rough energy estimate:

```
E_est ≈ P_CPU^avg · t + P_GPU^avg · t
CO2e = E_est · (1 kWh / 3.6e6 J) · g_grid
```

State the **grid factor** and uncertainty. Do **not** imply hardware‑level reversibility or energy recovery.

---

## 8) Compliance & Export Control

You are responsible for complying with applicable **laws**, **sanctions**, and **export‑control** restrictions in your jurisdiction.

---

## 9) Reporting Concerns

If you believe this Work is being used in violation of this policy or in a manner that creates material risk, open a public issue (preferred for transparency) or contact the maintainers privately if disclosure could increase harm. Provide:

* Repo fork/commit, description of use, data types, and claimed outputs
* Why you believe it violates this policy

The maintainers may respond by warning, documenting risks, or taking steps to reduce harm (e.g., clarifications, takedown of specific artifacts, or coordinated disclosure).

---

## 10) License & Warranty

* **License:** CC0‑1.0 (Public Domain). No rights reserved.
* **No warranty:** The Work is provided “as is,” without warranties of any kind.

---

## 11) Versioning

This policy may evolve to reflect new risks or clarifications.

* Version: **v1.0.0**
* Date: **2025‑09‑13**

**Quick checklist for new uses**

* [ ] Synthetic/benign data only
* [ ] No people‑routing or human decisions
* [ ] No hazardous chemistry
* [ ] No scaling/advantage/energy hype
* [ ] Seeds + metrics logged; limits stated
* [ ] Export control/law checked
