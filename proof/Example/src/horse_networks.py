#!/usr/bin/env python3
"""
horse_networks.py
=================

Builds two network artifacts for the horse-combination project:

1. A state-transition graph for the horse-combination process
2. A proof-tool selection network for the formal proof strategy

Outputs:
    - GraphML files for both graphs
    - JSON summaries
    - Interactive HTML visualizations when Plotly is available
    - Static PNG visualizations when Matplotlib is available

Suggested usage:
    python src/horse_networks.py --max-horses 12 --out out

Presentation mode is now the default: wider spacing, larger nodes, and hidden
edge labels for cleaner screenshots.

More examples:
    python src/horse_networks.py --max-horses 8 --out out
    python src/horse_networks.py --max-horses 16 --out out --no-html
    python src/horse_networks.py --max-horses 12 --out out --state-no-values
    python src/horse_networks.py --max-horses 12 --out out --lab-mode
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Optional imports for rendering
# ---------------------------------------------------------------------------

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    go = None
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional import from horse_probability_lab.py
# ---------------------------------------------------------------------------

try:
    from horse_probability_lab import exact_value, best_action
    LAB_AVAILABLE = True
except Exception:
    exact_value = None
    best_action = None
    LAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Exact probabilities
# ---------------------------------------------------------------------------

P_UPGRADE = Fraction(1, 5)
P_SAME = Fraction(1, 2)
P_DOWNGRADE = Fraction(3, 10)
P_T1_STAY = P_SAME + P_DOWNGRADE  # 4/5


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class HorseState:
    t1: int
    t2: int
    t3: int
    t4: int

    def total(self) -> int:
        return self.t1 + self.t2 + self.t3 + self.t4

    def tier_weight(self) -> int:
        return self.t1 + 2 * self.t2 + 4 * self.t3 + 8 * self.t4

    def is_success(self) -> bool:
        return self.t4 >= 1

    def legal_actions(self) -> Tuple[str, ...]:
        actions: List[str] = []
        if self.t1 >= 2:
            actions.append("combine_T1")
        if self.t2 >= 2:
            actions.append("combine_T2")
        if self.t3 >= 2:
            actions.append("combine_T3")
        return tuple(actions)

    def is_failure(self) -> bool:
        return (not self.is_success()) and len(self.legal_actions()) == 0

    def is_active(self) -> bool:
        return (not self.is_success()) and len(self.legal_actions()) > 0

    def key(self) -> str:
        return f"{self.t1},{self.t2},{self.t3},{self.t4}"

    def label(self) -> str:
        return f"({self.t1},{self.t2},{self.t3},{self.t4})"


Transition = Tuple[Fraction, HorseState, str]


def transitions(state: HorseState, action: str) -> Tuple[Transition, ...]:
    if action == "combine_T1":
        if state.t1 < 2:
            raise ValueError(f"Illegal action {action} at {state}")
        upgrade = HorseState(state.t1 - 2, state.t2 + 1, state.t3, state.t4)
        stay = HorseState(state.t1 - 1, state.t2, state.t3, state.t4)
        return (
            (P_UPGRADE, upgrade, "upgrade_to_T2"),
            (P_T1_STAY, stay, "same_or_downgrade_to_T1"),
        )

    if action == "combine_T2":
        if state.t2 < 2:
            raise ValueError(f"Illegal action {action} at {state}")
        upgrade = HorseState(state.t1, state.t2 - 2, state.t3 + 1, state.t4)
        same = HorseState(state.t1, state.t2 - 1, state.t3, state.t4)
        downgrade = HorseState(state.t1 + 1, state.t2 - 2, state.t3, state.t4)
        return (
            (P_UPGRADE, upgrade, "upgrade_to_T3"),
            (P_SAME, same, "same_to_T2"),
            (P_DOWNGRADE, downgrade, "downgrade_to_T1"),
        )

    if action == "combine_T3":
        if state.t3 < 2:
            raise ValueError(f"Illegal action {action} at {state}")
        upgrade = HorseState(state.t1, state.t2, state.t3 - 2, state.t4 + 1)
        same = HorseState(state.t1, state.t2, state.t3 - 1, state.t4)
        downgrade = HorseState(state.t1, state.t2 + 1, state.t3 - 2, state.t4)
        return (
            (P_UPGRADE, upgrade, "upgrade_to_T4"),
            (P_SAME, same, "same_to_T3"),
            (P_DOWNGRADE, downgrade, "downgrade_to_T2"),
        )

    raise ValueError(f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fraction_to_str(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def state_kind(state: HorseState) -> str:
    if state.is_success():
        return "success"
    if state.is_failure():
        return "failure"
    return "active"


STATE_COLORS = {
    "success": "#2ecc71",   # green
    "failure": "#e74c3c",   # red
    "active": "#4da3ff",    # blue
}

PROOF_KIND_COLORS = {
    "claim": "#f1c40f",       # gold
    "material": "#4da3ff",    # blue
    "tool": "#8e44ad",        # purple
    "lemma": "#1abc9c",       # teal
    "warning": "#e74c3c",     # red
    "check": "#ecf0f1",       # white-ish
    "artifact": "#95a5a6",    # gray
}


STATE_KIND_LABELS = {
    "active": "Active state",
    "success": "Success state",
    "failure": "Terminal failure",
}

PROOF_KIND_LABELS = {
    "claim": "Claim",
    "material": "Mathematical material",
    "tool": "Proof tool",
    "lemma": "Lemma / theorem output",
    "warning": "Warning / audit",
    "check": "Closure check",
    "artifact": "Artifact / output",
}


def legend_items_for_graph(G: nx.Graph) -> List[Tuple[str, str]]:
    """
    Return legend entries as (label, color), based on node kind/color.

    The state graph uses active/success/failure kinds.
    The proof-tool graph uses claim/material/tool/lemma/warning/check/artifact kinds.
    """
    kinds_present = {
        str(data.get("kind", ""))
        for _node, data in G.nodes(data=True)
        if data.get("kind") is not None
    }

    # Stable ordering for cleaner visual output.
    state_order = ["active", "success", "failure"]
    proof_order = ["claim", "material", "tool", "lemma", "warning", "check", "artifact"]

    if any(kind in STATE_KIND_LABELS for kind in kinds_present):
        ordered = [kind for kind in state_order if kind in kinds_present]
        return [(STATE_KIND_LABELS[kind], STATE_COLORS[kind]) for kind in ordered]

    ordered = [kind for kind in proof_order if kind in kinds_present]
    return [(PROOF_KIND_LABELS[kind], PROOF_KIND_COLORS[kind]) for kind in ordered]


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_reachable_state_graph(initial_x: int, include_values: bool = True) -> nx.DiGraph:
    """
    Build the reachable state-transition graph starting from (initial_x, 0, 0, 0).

    If horse_probability_lab.py is importable and include_values=True, the graph
    is annotated with exact optimal value and best action.
    """
    G = nx.DiGraph()
    start = HorseState(initial_x, 0, 0, 0)

    queue: deque[HorseState] = deque([start])
    seen = {start}

    while queue:
        state = queue.popleft()
        kind = state_kind(state)

        node_attrs = {
            "label": state.label(),
            "t1": state.t1,
            "t2": state.t2,
            "t3": state.t3,
            "t4": state.t4,
            "total": state.total(),
            "tier_weight": state.tier_weight(),
            "kind": kind,
            "color": STATE_COLORS[kind],
            "legal_actions": ",".join(state.legal_actions()),
        }

        if include_values and LAB_AVAILABLE:
            try:
                node_attrs["value_exact"] = fraction_to_str(exact_value(state))
                node_attrs["value_float"] = float(exact_value(state))
                node_attrs["best_action"] = best_action(state) or ""
            except Exception:
                node_attrs["value_exact"] = ""
                node_attrs["value_float"] = ""
                node_attrs["best_action"] = ""

        G.add_node(state.key(), **node_attrs)

        if state.is_success() or state.is_failure():
            continue

        for action in state.legal_actions():
            for prob, next_state, outcome in transitions(state, action):
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append(next_state)

                G.add_edge(
                    state.key(),
                    next_state.key(),
                    action=action,
                    outcome=outcome,
                    probability=fraction_to_str(prob),
                    probability_float=float(prob),
                    label=f"{action}\n{outcome}\n{fraction_to_str(prob)}",
                )

    return G


def build_proof_tool_graph() -> nx.DiGraph:
    """
    Build the proof-tool selection network reflecting the theorem architecture.
    """
    G = nx.DiGraph()

    nodes = [
        # layer 0: central target / artifact roots
        ("Algorithm Correctness Claim", "claim", 0, "For every finite state s, P_impl(s)=V_opt(s)."),
        ("State-Transition Artifact", "artifact", 0, "Horse-state network visual."),
        ("Proof-Tool Artifact", "artifact", 0, "Proof-routing network visual."),

        # layer 1: claim shape and mathematical material
        ("Universal Claim", "claim", 1, "Theorem quantified over all states."),
        ("Finite State Process", "material", 1, "Finite-horizon state process."),
        ("Decision Process", "material", 1, "Action choice at each active state."),
        ("Resource Bound", "material", 1, "Tier-weight potential and minimal 8-horse theorem."),
        ("Scaling Analysis", "material", 1, "Finite-window empirical fit."),

        # layer 2: structural observations
        ("Arbitrary State", "tool", 2, "Proof starts from an arbitrary state s."),
        ("Total Horse Rank", "lemma", 2, "rho(s)=t1+t2+t3+t4."),
        ("Tier-Weight Potential", "lemma", 2, "omega(s)=t1+2t2+4t3+8t4."),
        ("Optimal Substructure", "lemma", 2, "Best first action + optimal continuation."),
        ("Probability Boundedness", "warning", 2, "Probabilities satisfy 0 <= P(X) <= 1."),

        # layer 3: proof tools
        ("Strong Induction", "tool", 3, "Induction on total horse count."),
        ("Bellman Recurrence", "tool", 3, "V(s)=max_a sum K(s'|s,a)V(s')."),
        ("Potential Function Argument", "tool", 3, "Use omega(s) to prove minimal-resource bounds."),
        ("Empirical-Only Warning", "warning", 3, "Power-law fit is finite-window, not global."),

        # layer 4: theorem/lemma outputs
        ("Termination Lemma", "lemma", 4, "Every legal move decreases rank by exactly one."),
        ("No Directed Cycles", "lemma", 4, "State-transition DAG under rank decrease."),
        ("Recursive Correctness Theorem", "lemma", 4, "Recursive implementation equals optimal probability."),
        ("Minimal 8-Horses Lemma", "lemma", 4, "f(X)=0 for X<8 and f(8)=(1/5)^7."),
        ("Finite-Window Scaling Note", "warning", 4, "Positive power-law cannot hold globally."),

        # layer 5: closure / quality checks
        ("Closure Check", "check", 5, "Exact claim proved; assumptions preserved."),
        ("Terminal Base Cases", "check", 5, "Success => 1, terminal failure => 0."),
        ("Empirical/Theorem Separation", "check", 5, "Observed scaling kept separate from theorem."),

        # layer 6: outputs
        ("LaTeX Proof", "artifact", 6, "Formal proof document."),
        ("horse_probability_lab.py", "artifact", 6, "Exact verifier and Monte Carlo lab."),
        ("horse_networks.py", "artifact", 6, "This visual network builder."),
    ]

    for name, kind, layer, desc in nodes:
        G.add_node(
            name,
            label=name,
            kind=kind,
            layer=layer,
            description=desc,
            color=PROOF_KIND_COLORS[kind],
        )

    edges = [
        ("Algorithm Correctness Claim", "Universal Claim", "claim_shape"),
        ("Algorithm Correctness Claim", "Finite State Process", "material"),
        ("Algorithm Correctness Claim", "Decision Process", "material"),
        ("Algorithm Correctness Claim", "Resource Bound", "supporting_lemma"),
        ("Algorithm Correctness Claim", "Scaling Analysis", "audit"),

        ("Universal Claim", "Arbitrary State", "selects"),
        ("Finite State Process", "Total Horse Rank", "reveals"),
        ("Finite State Process", "Termination Lemma", "supports"),
        ("Decision Process", "Optimal Substructure", "reveals"),
        ("Resource Bound", "Tier-Weight Potential", "reveals"),
        ("Scaling Analysis", "Probability Boundedness", "audit"),

        ("Arbitrary State", "Strong Induction", "tool"),
        ("Total Horse Rank", "Strong Induction", "tool"),
        ("Total Horse Rank", "Termination Lemma", "supports"),
        ("Termination Lemma", "No Directed Cycles", "implies"),
        ("Optimal Substructure", "Bellman Recurrence", "tool"),
        ("Tier-Weight Potential", "Potential Function Argument", "tool"),
        ("Probability Boundedness", "Empirical-Only Warning", "forces"),

        ("Strong Induction", "Recursive Correctness Theorem", "proves"),
        ("Bellman Recurrence", "Recursive Correctness Theorem", "proves"),
        ("Potential Function Argument", "Minimal 8-Horses Lemma", "proves"),
        ("Empirical-Only Warning", "Finite-Window Scaling Note", "qualifies"),

        ("Terminal Base Cases", "Recursive Correctness Theorem", "required_for"),
        ("Recursive Correctness Theorem", "Closure Check", "passes_through"),
        ("Minimal 8-Horses Lemma", "Closure Check", "passes_through"),
        ("Finite-Window Scaling Note", "Empirical/Theorem Separation", "requires"),
        ("Closure Check", "LaTeX Proof", "exports_to"),
        ("Closure Check", "horse_probability_lab.py", "verified_by"),
        ("Empirical/Theorem Separation", "LaTeX Proof", "exports_to"),
        ("State-Transition Artifact", "horse_networks.py", "implemented_by"),
        ("Proof-Tool Artifact", "horse_networks.py", "implemented_by"),
        ("horse_networks.py", "LaTeX Proof", "supports"),
        ("horse_probability_lab.py", "LaTeX Proof", "supports"),
    ]

    for u, v, relation in edges:
        G.add_edge(u, v, relation=relation, label=relation)

    return G


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def layered_positions_from_attribute(
    G: nx.Graph,
    layer_attr: str,
    *,
    left_to_right: bool = True,
    within_layer_spacing: float = 1.25,
    layer_spacing: float = 2.5,
) -> Dict[str, Tuple[float, float]]:
    """
    Manual layered layout so layers are stable and deterministic.

    Nodes sharing the same layer_attr are stacked vertically.
    """
    groups: Dict[int, List[str]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        layer = int(data[layer_attr])
        groups[layer].append(node)

    positions: Dict[str, Tuple[float, float]] = {}
    sorted_layers = sorted(groups)

    for layer in sorted_layers:
        nodes = sorted(groups[layer])
        n = len(nodes)
        x = layer * layer_spacing if left_to_right else -layer * layer_spacing
        center_offset = (n - 1) / 2.0

        for i, node in enumerate(nodes):
            y = (center_offset - i) * within_layer_spacing
            positions[node] = (x, y)

    return positions


def state_graph_positions(
    G: nx.Graph,
    *,
    layer_spacing: float = 3.25,
    within_layer_spacing: float = 1.55,
) -> Dict[str, Tuple[float, float]]:
    """
    Layer by total horse count, descending left-to-right so the root appears
    on the left and terminals drift right as total count decreases.

    Presentation defaults intentionally use wider spacing than the raw lab map.
    """
    groups: Dict[int, List[str]] = defaultdict(list)
    max_total = 0

    for node, data in G.nodes(data=True):
        total = int(data["total"])
        groups[total].append(node)
        max_total = max(max_total, total)

    positions: Dict[str, Tuple[float, float]] = {}

    for total in sorted(groups.keys(), reverse=True):
        nodes = sorted(
            groups[total],
            key=lambda n: (
                int(G.nodes[n]["t4"]),
                int(G.nodes[n]["t3"]),
                int(G.nodes[n]["t2"]),
                int(G.nodes[n]["t1"]),
            ),
            reverse=True,
        )
        x = (max_total - total) * layer_spacing
        n = len(nodes)
        center_offset = (n - 1) / 2.0

        for i, node in enumerate(nodes):
            y = (center_offset - i) * within_layer_spacing
            positions[node] = (x, y)

    return positions


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def edge_midpoint(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float]:
    return ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)


def render_plotly_html(
    G: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    title: str,
    html_path: Path,
    *,
    node_label_mode: str = "label",
    show_edge_labels: bool = False,
    presentation_mode: bool = True,
) -> bool:
    if not PLOTLY_AVAILABLE:
        return False

    # edges
    edge_x: List[float] = []
    edge_y: List[float] = []
    edge_label_x: List[float] = []
    edge_label_y: List[float] = []
    edge_label_text: List[str] = []

    for u, v, data in G.edges(data=True):
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

        if show_edge_labels:
            mx, my = edge_midpoint((x0, y0), (x1, y1))
            label = data.get("label") or data.get("relation") or ""
            edge_label_x.append(mx)
            edge_label_y.append(my)
            edge_label_text.append(label)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="#8899aa"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # nodes
    node_x: List[float] = []
    node_y: List[float] = []
    node_color: List[str] = []
    node_size: List[float] = []
    node_text: List[str] = []
    node_labels: List[str] = []

    for node, data in G.nodes(data=True):
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(data.get("color", "#4da3ff"))

        if presentation_mode:
            if data.get("kind") == "success":
                node_size.append(34)
            elif data.get("kind") == "failure":
                node_size.append(30)
            else:
                node_size.append(32)
        else:
            if data.get("kind") == "success":
                node_size.append(30)
            elif data.get("kind") == "failure":
                node_size.append(24)
            else:
                node_size.append(26)

        hover_lines = [f"<b>{data.get('label', node)}</b>"]
        for k, v in data.items():
            if k in {"label", "color"}:
                continue
            hover_lines.append(f"{k}: {v}")
        node_text.append("<br>".join(hover_lines))
        node_labels.append(str(data.get(node_label_mode, data.get("label", node))))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(size=12 if presentation_mode else 10, color="#1f2d3a"),
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1.0, color="#1f2d3a"),
            opacity=0.95,
        ),
        showlegend=False,
    )

    traces = [edge_trace, node_trace]

    if show_edge_labels and edge_label_text:
        traces.append(
            go.Scatter(
                x=edge_label_x,
                y=edge_label_y,
                mode="text",
                text=edge_label_text,
                textfont=dict(size=10, color="#34495e"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Add a clean node-color legend without duplicating the actual node traces.
    for legend_label, legend_color in legend_items_for_graph(G):
        traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=14,
                    color=legend_color,
                    line=dict(width=1.0, color="#1f2d3a"),
                ),
                name=legend_label,
                hoverinfo="none",
                showlegend=True,
            )
        )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            title_x=0.5,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=60, b=20),
            hovermode="closest",
            showlegend=True,
            legend=dict(
                title=dict(text="Node colors"),
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#d0d7de",
                borderwidth=1,
            ),
        ),
    )

    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    return True


def render_matplotlib_png(
    G: nx.Graph,
    positions: Dict[str, Tuple[float, float]],
    title: str,
    png_path: Path,
    *,
    show_labels: bool = True,
    presentation_mode: bool = True,
) -> bool:
    if not MATPLOTLIB_AVAILABLE:
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 11) if presentation_mode else (16, 10))
    node_colors = [data.get("color", "#4da3ff") for _, data in G.nodes(data=True)]

    nx.draw_networkx_edges(
        G,
        positions,
        alpha=0.6,
        edge_color="#90a4ae",
        arrows=True,
        arrowsize=12,
        width=1.2,
    )

    nx.draw_networkx_nodes(
        G,
        positions,
        node_color=node_colors,
        node_size=1150 if presentation_mode else 850,
        edgecolors="#1f2d3a",
        linewidths=1.0,
    )

    if show_labels:
        labels = {node: data.get("label", node) for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G,
            positions,
            labels=labels,
            font_size=9 if presentation_mode else 8,
            font_color="#111111",
        )

    legend_items = legend_items_for_graph(G)
    if legend_items:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=legend_label,
                markerfacecolor=legend_color,
                markeredgecolor="#1f2d3a",
                markersize=10,
            )
            for legend_label, legend_color in legend_items
        ]
        plt.legend(
            handles=handles,
            title="Node colors",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=True,
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_state_graph(G: nx.DiGraph, initial_x: int) -> dict:
    kind_counts = defaultdict(int)
    action_counts = defaultdict(int)

    for _node, data in G.nodes(data=True):
        kind_counts[data["kind"]] += 1

    for _u, _v, data in G.edges(data=True):
        action_counts[data["action"]] += 1

    summary = {
        "initial_x": initial_x,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "kind_counts": dict(sorted(kind_counts.items())),
        "edge_action_counts": dict(sorted(action_counts.items())),
        "lab_annotations_available": LAB_AVAILABLE,
    }
    return summary


def summarize_proof_graph(G: nx.DiGraph) -> dict:
    kind_counts = defaultdict(int)
    layer_counts = defaultdict(int)

    for _node, data in G.nodes(data=True):
        kind_counts[data["kind"]] += 1
        layer_counts[int(data["layer"])] += 1

    return {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "kind_counts": dict(sorted(kind_counts.items())),
        "layer_counts": dict(sorted(layer_counts.items())),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build horse-state and proof-tool network artifacts."
    )
    parser.add_argument(
        "--max-horses",
        type=int,
        default=12,
        help="Initial number of T1 horses for the reachable state graph. Default: 12.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out",
        help="Output directory. Default: out.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML exports even if Plotly is available.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG exports even if Matplotlib is available.",
    )
    parser.add_argument(
        "--state-no-values",
        action="store_true",
        help="Do not annotate the state graph with exact values / best actions.",
    )
    parser.add_argument(
        "--lab-mode",
        action="store_true",
        help="Use denser development layout with visible proof-edge labels.",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="Show edge labels in HTML exports. Presentation default hides them.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.max_horses < 0:
        raise SystemExit("--max-horses must be nonnegative")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Horse Networks")
    print("==============")
    print(f"initial T1 horses : {args.max_horses}")
    print(f"out dir           : {out_dir}")
    print(f"plotly available  : {PLOTLY_AVAILABLE}")
    print(f"matplotlib avail  : {MATPLOTLIB_AVAILABLE}")
    presentation_mode = not args.lab_mode
    show_edge_labels = args.show_edge_labels or args.lab_mode

    print(f"lab annotations   : {LAB_AVAILABLE and not args.state_no_values}")
    print(f"presentation mode : {presentation_mode}")
    print(f"edge labels       : {show_edge_labels}")
    print()

    # Build graphs
    state_graph = build_reachable_state_graph(
        args.max_horses,
        include_values=(not args.state_no_values),
    )
    proof_graph = build_proof_tool_graph()

    if presentation_mode:
        state_positions = state_graph_positions(
            state_graph,
            layer_spacing=3.25,
            within_layer_spacing=1.55,
        )
        proof_positions = layered_positions_from_attribute(
            proof_graph,
            "layer",
            layer_spacing=3.35,
            within_layer_spacing=1.65,
        )
    else:
        state_positions = state_graph_positions(
            state_graph,
            layer_spacing=2.5,
            within_layer_spacing=1.2,
        )
        proof_positions = layered_positions_from_attribute(
            proof_graph,
            "layer",
            layer_spacing=2.5,
            within_layer_spacing=1.25,
        )

    # Export GraphML
    state_graphml = out_dir / "horse_state_graph.graphml"
    proof_graphml = out_dir / "proof_tool_network.graphml"
    nx.write_graphml(state_graph, state_graphml)
    nx.write_graphml(proof_graph, proof_graphml)

    # Export summaries
    state_summary = summarize_state_graph(state_graph, args.max_horses)
    proof_summary = summarize_proof_graph(proof_graph)

    write_json(state_summary, out_dir / "horse_state_graph_summary.json")
    write_json(proof_summary, out_dir / "proof_tool_network_summary.json")

    # Render HTML
    html_results = {"state_graph_html": False, "proof_graph_html": False}
    if not args.no_html:
        html_results["state_graph_html"] = render_plotly_html(
            state_graph,
            state_positions,
            f"Horse State-Transition Graph (start: {args.max_horses} T1 horses)",
            out_dir / "horse_state_graph.html",
            node_label_mode="label",
            show_edge_labels=show_edge_labels and not presentation_mode,
            presentation_mode=presentation_mode,
        )
        html_results["proof_graph_html"] = render_plotly_html(
            proof_graph,
            proof_positions,
            "Proof-Tool Selection Network",
            out_dir / "proof_tool_network.html",
            node_label_mode="label",
            show_edge_labels=show_edge_labels,
            presentation_mode=presentation_mode,
        )

    # Render PNG
    png_results = {"state_graph_png": False, "proof_graph_png": False}
    if not args.no_png:
        png_results["state_graph_png"] = render_matplotlib_png(
            state_graph,
            state_positions,
            f"Horse State-Transition Graph (start: {args.max_horses} T1 horses)",
            out_dir / "horse_state_graph.png",
            show_labels=True,
            presentation_mode=presentation_mode,
        )
        png_results["proof_graph_png"] = render_matplotlib_png(
            proof_graph,
            proof_positions,
            "Proof-Tool Selection Network",
            out_dir / "proof_tool_network.png",
            show_labels=True,
            presentation_mode=presentation_mode,
        )

    print("State graph")
    print("-----------")
    print(f"nodes              : {state_summary['node_count']}")
    print(f"edges              : {state_summary['edge_count']}")
    print(f"kind counts        : {state_summary['kind_counts']}")
    print()

    print("Proof-tool graph")
    print("----------------")
    print(f"nodes              : {proof_summary['node_count']}")
    print(f"edges              : {proof_summary['edge_count']}")
    print(f"kind counts        : {proof_summary['kind_counts']}")
    print()

    print("Exports")
    print("-------")
    print(f"wrote {state_graphml}")
    print(f"wrote {proof_graphml}")
    print(f"wrote {out_dir / 'horse_state_graph_summary.json'}")
    print(f"wrote {out_dir / 'proof_tool_network_summary.json'}")

    if html_results["state_graph_html"]:
        print(f"wrote {out_dir / 'horse_state_graph.html'}")
    if html_results["proof_graph_html"]:
        print(f"wrote {out_dir / 'proof_tool_network.html'}")
    if (not args.no_html) and not any(html_results.values()):
        print("html exports skipped (Plotly unavailable)")

    if png_results["state_graph_png"]:
        print(f"wrote {out_dir / 'horse_state_graph.png'}")
    if png_results["proof_graph_png"]:
        print(f"wrote {out_dir / 'proof_tool_network.png'}")
    if (not args.no_png) and not any(png_results.values()):
        print("png exports skipped (Matplotlib unavailable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
