# Example: Horse Probability Proof Network

This directory contains a worked example of the **Universal Proof Selection Network** methodology.

The example starts from a recursive probability problem involving tiered horse combinations and turns it into:

1. a formal mathematical proof,
2. an exact computational verifier,
3. a state-transition graph,
4. a proof-tool selection graph.

The horse problem is not the main artifact. It is the case study.

The reusable artifact is the proof-construction methodology demonstrated by the case.

---

## Problem summary

A state is a tuple:

```text
(t1, t2, t3, t4)
```

where `t1`, `t2`, `t3`, and `t4` are the counts of tier-1 through tier-4 horses.

The objective is to maximize the probability of eventually obtaining at least one tier-4 horse.

Legal actions combine two horses of the same tier:

- combine two tier-1 horses,
- combine two tier-2 horses,
- combine two tier-3 horses.

The transition probabilities are:

```text
upgrade   = 1/5
same      = 1/2
downgrade = 3/10
```

For tier 1, downgrade is absorbed into the tier-1 outcome because there is no lower tier.

---

## Formal proof result

The proof establishes that the recursive value function computes the optimal success probability.

The central theorem is:

```text
For every finite horse state s,
P_impl(s) = V_opt(s).
```

That is, the recursive implementation returns the optimal probability of eventually reaching a state with `t4 >= 1`.

The proof uses:

- finite state modeling,
- terminal success and failure states,
- total horse count as a well-founded rank,
- strong induction,
- Bellman optimality,
- a tier-weight potential for the 8-horse resource bound.

The formal proof is located at:

```text
proof/horse_probability_proof.tex
proof/proof.pdf
```

---

## Key mathematical observations

### 1. Total horse count is a rank

Every legal combination consumes two horses and returns one horse.

Therefore the total horse count

```text
rho(s) = t1 + t2 + t3 + t4
```

decreases by exactly one after every nonterminal transition.

This proves:

- no directed cycles in the nonterminal transition graph,
- finite termination,
- well-founded recursion,
- validity of a strong-induction proof.

### 2. Bellman recurrence gives optimality

At each active state, the optimal value satisfies:

```text
V(s) = max over legal actions a of E[V(s')]
```

The recursive algorithm implements this recurrence directly.

### 3. Eight tier-1 horses is the minimum possible resource level

Define the tier-weight potential:

```text
omega(s) = t1 + 2*t2 + 4*t3 + 8*t4
```

This quantity never increases under legal transitions.

Therefore fewer than eight tier-1 horses cannot produce a tier-4 horse.

Starting with exactly eight tier-1 horses, success requires seven perfect upgrades:

```text
8 T1 → 4 T2 → 2 T3 → 1 T4
```

So:

```text
f(8) = (1/5)^7
```

### 4. Scaling is empirical, not a global theorem

The script includes finite-window log-log scaling analysis.

Any fitted expression like:

```text
P(X) proportional to X^alpha
```

must be interpreted as a finite-range empirical fit, not a global theorem, because probabilities are bounded above by 1.

---

## Files

```text
Example/
├── README.MD
├── prompt.md
├── out/
│   ├── horse_state_graph_summary.json
│   ├── horse_state_graph.graphml
│   ├── horse_state_graph.html
│   ├── horse_state_graph.png
│   ├── probabilities.csv
│   ├── proof_tool_network_summary.json
│   ├── proof_tool_network.graphml
│   ├── proof_tool_network.html
│   ├── proof_tool_network.png
│   └── verification_report.json
├── proof/
│   ├── horse_probability_proof.tex
│   └── proof.pdf
└── src/
    ├── horse_networks.py
    └── horse_probability_lab.py
```

---

## Script 1: probability lab

```text
src/horse_probability_lab.py
```

This script contains:

- exact recursive solver using `fractions.Fraction`,
- bottom-up dynamic programming verifier,
- Monte Carlo sanity checker,
- finite-window scaling fit,
- sampled threshold report,
- CSV and JSON exports.

Run:

```bash
python src/horse_probability_lab.py --max-x 80 --verify-max-horses 28 --trials 5000 --out out
```

Fast smoke test:

```bash
python src/horse_probability_lab.py --max-x 30 --verify-max-horses 16 --trials 1000 --out out
```

Expected checks include:

```text
f(8) exact     : 1/78125
expected       : 1/78125
passed         : True
```

and a bottom-up verification pass.

---

## Script 2: network builder

```text
src/horse_networks.py
```

This script builds two graphs:

### Horse state-transition graph

Nodes are horse states:

```text
(t1,t2,t3,t4)
```

Node colors:

```text
blue  = active state
green = success state
red   = terminal failure
```

Edges are legal stochastic transitions.

### Proof-tool selection graph

This graph shows how the proof methodology routes the theorem:

```text
Claim
→ Mathematical material
→ Proof tool
→ Lemma / theorem output
→ Closure check
→ Artifact
```

Node colors:

```text
gold   = claim
blue   = mathematical material
purple = proof tool
teal   = lemma / theorem output
red    = warning / audit
white  = closure check
gray   = artifact / output
```

Run:

```bash
python src/horse_networks.py --max-horses 12 --out out
```

Presentation mode is the default.

For denser diagnostic output:

```bash
python src/horse_networks.py --max-horses 12 --out out --lab-mode
```

To show edge labels:

```bash
python src/horse_networks.py --max-horses 12 --out out --show-edge-labels
```

---

## Output artifacts

The `out/` directory contains generated artifacts:

```text
probabilities.csv
verification_report.json
horse_state_graph.html
horse_state_graph.png
horse_state_graph.graphml
horse_state_graph_summary.json
proof_tool_network.html
proof_tool_network.png
proof_tool_network.graphml
proof_tool_network_summary.json
```

The HTML files are interactive when opened in a browser.

The PNG files are static presentation exports.

The GraphML files can be loaded into graph tools such as Gephi, Cytoscape, or yEd.

---

## Reusable lesson from the example

The proof was not guessed.

It was routed:

```text
Algorithm correctness claim
→ finite recursive stochastic decision process
→ total-horse rank
→ well-founded recursion
→ strong induction
→ Bellman recurrence
→ correctness theorem
→ closure audit
```

The same methodology can be reused on unrelated proof problems by replacing the problem-specific state model with the new theorem's claim shape and mathematical material.

---

## License

This example and associated methodology are dedicated to the public domain under **CC0 1.0 Universal**.

You may copy, modify, distribute, use, teach, embed, remix, and adapt this work for any purpose, with or without attribution.

Attribution is appreciated but not required.

No warranty is provided. Use at your own discretion.
