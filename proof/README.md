# Universal Proof Selection Network

A reusable proof-construction methodology for routing mathematical claims through a structured tool-selection process.

The core idea is simple:

> A proof is not guessed.  
> A proof is routed.

This project packages a proof-building workflow for AI-assisted mathematics. Instead of jumping directly to a familiar proof method, the workflow first identifies the theorem's logical shape, the mathematical material involved, the proof tools activated by that structure, the obligations created by those tools, and the closure checks needed before the proof can be considered complete.

The included `Example/` directory demonstrates the method on a concrete recursive probability problem.

---

## What this is

The **Universal Proof Selection Network** is a prompt-level methodology for constructing, auditing, and repairing mathematical proofs.

It is designed to help an AI assistant, researcher, student, or proof-writer answer questions like:

- What kind of theorem am I trying to prove?
- What proof tools does the theorem's structure activate?
- What obligations does the chosen proof method create?
- What closure checks must the proof pass?
- If the proof fails, what repair route should be taken?

The method treats proof construction as a network:

```text
Claim intake
→ Claim-shape detection
→ Mathematical-material detection
→ Tool activation
→ Obligation generation
→ Proof construction
→ Closure audit
→ Repair loop, if needed
→ Final proof
```

---

## Repository layout

```text
PROOF/
├── README.md
├── Example/
│   ├── README.MD
│   ├── prompt.md
│   ├── out/
│   │   ├── horse_state_graph_summary.json
│   │   ├── horse_state_graph.graphml
│   │   ├── horse_state_graph.html
│   │   ├── horse_state_graph.png
│   │   ├── probabilities.csv
│   │   ├── proof_tool_network_summary.json
│   │   ├── proof_tool_network.graphml
│   │   ├── proof_tool_network.html
│   │   ├── proof_tool_network.png
│   │   └── verification_report.json
│   ├── proof/
│   │   ├── horse_probability_proof.tex
│   │   └── proof.pdf
│   └── src/
│       ├── horse_networks.py
│       └── horse_probability_lab.py
```

---

## The reusable methodology

The reusable methodology is contained in `Example/prompt.md`.

That prompt can be pasted into an AI assistant as a high-detail instruction block for proof construction. It asks the assistant to:

1. Intake the exact theorem.
2. Detect the logical claim shape.
3. Detect the mathematical material.
4. Activate appropriate proof tools.
5. Generate proof obligations.
6. Construct the proof.
7. Audit the proof.
8. Repair failures.
9. Present the final proof cleanly.

The method is general. The horse probability example is only a case study.

---

## Core proof-routing categories

The methodology routes claims through categories such as:

### Claim shapes

- Universal claim
- Existential claim
- Implication
- Biconditional
- Uniqueness
- Equality of sets or structures
- Impossibility
- Algorithm/process correctness

### Mathematical material

- Natural numbers
- Recursive processes
- Finite sets
- Graphs/networks
- Algorithms
- Probability
- Algebraic structures
- Number theory
- Analysis and metric spaces
- Optimization
- Dynamical systems
- Combinatorics
- Linear algebra
- Topology

### Proof tools

- Direct proof
- Contrapositive
- Contradiction
- Cases
- Induction
- Strong induction
- Minimal counterexample
- Construction
- Uniqueness
- Double inclusion
- Bijection
- Double counting
- Invariant
- Monovariant
- Extremal principle
- Well-founded descent
- Fixed point
- Compactness
- Diagonalization
- Epsilon-delta
- Element chase
- Algebraic expansion
- Normal form
- Loop invariant
- Bellman / dynamic programming

### Closure checks

- Exact claim proved
- Assumptions respected
- Quantifiers handled
- Domains respected
- Cases exhausted
- Edge cases handled
- No circularity
- Both directions proved when needed
- Existence and uniqueness separated when needed
- Empirical claims separated from theorem claims

---

## Example case study

The included example applies the methodology to a recursive horse-combination probability problem.

It contains:

- a formal LaTeX proof
- an exact Python verifier
- Monte Carlo sanity checks
- finite-window scaling analysis
- a state-transition graph
- a proof-tool selection graph

See `Example/README.MD` for details.

---

## Why this exists

AI-generated proofs can sound locally fluent while hiding structural problems:

- skipped obligations
- weak induction
- hidden assumptions
- quantifier errors
- empirical overclaims
- circular reasoning
- missing edge cases

This methodology gives the assistant a scaffold for proof construction and proof repair.

The goal is not to make proof automatic.

The goal is to make proof attempts more inspectable, auditable, and structurally honest.

---

## Suggested use

Paste the prompt from `Example/prompt.md` into an AI assistant, then provide a theorem or proof attempt.

Ask it to follow the Universal Proof Selection Network protocol.

Example request:

```text
Use the Universal Proof Selection Network protocol to prove the following theorem:

[insert theorem]
```

For proof repair:

```text
Use the Universal Proof Selection Network protocol to audit and repair this proof:

[insert proof]
```

---

## License

This repository is dedicated to the public domain under **CC0 1.0 Universal**.

You may copy, modify, distribute, use, teach, embed, remix, and adapt this methodology for any purpose, with or without attribution.

Attribution is appreciated but not required.

No warranty is provided. Use at your own discretion.
