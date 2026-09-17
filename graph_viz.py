"""
graph_viz.py
============
Plotly visualisations for the Knowledge Graph Virtual Lab.

Two live figures, both rebuilt from the current state on every rerun:

* :func:`schema_figure`   -- the *design* view: one marker per node label, one
  labelled arrow per relationship type.
* :func:`instance_figure` -- the *data* view: one marker per imported node, one
  arrow per imported relationship, with label / type filters.

Colour rules follow the project's data-viz guidance: a fixed categorical order
(never cycled), identity carried by a legend **and** a direct label on every
marker (so colour is never the only cue), recessive grey edges, and a separate
set of steps selected for the dark surface rather than an automatic flip.
The figures use transparent backgrounds so they sit correctly on the native
Streamlit light and dark themes without any custom CSS.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

# Fixed categorical order -- slot N always means the same hue.
# Slot 1 is the lab's navy so the diagrams open on the interface colour; the
# remaining slots are the validated colour-blind-safe order, unchanged.
CATEGORICAL_LIGHT = [
    "#1B365D", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
CATEGORICAL_DARK = [
    "#3987e5", "#d95926", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
]

INK_LIGHT, INK_DARK = "#10233F", "#ffffff"
MUTED_LIGHT, MUTED_DARK = "#5E6C82", "#c3c2b7"
EDGE_LIGHT, EDGE_DARK = "#9FAFC4", "#6f6f68"
GRID_LIGHT, GRID_DARK = "#E3E9F2", "#3a3a37"
SURFACE_LIGHT, SURFACE_DARK = "#FFFFFF", "#1a1a19"


class Theme:
    """The handful of colours a figure needs, for one Streamlit theme."""

    def __init__(self, dark: bool) -> None:
        self.dark = dark
        self.series = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
        self.ink = INK_DARK if dark else INK_LIGHT
        self.muted = MUTED_DARK if dark else MUTED_LIGHT
        self.edge = EDGE_DARK if dark else EDGE_LIGHT
        self.grid = GRID_DARK if dark else GRID_LIGHT
        self.surface = SURFACE_DARK if dark else SURFACE_LIGHT

    def color_for(self, index: int) -> str:
        """Assign hues in fixed order; past the palette, fall back to muted grey."""
        if index < len(self.series):
            return self.series[index]
        return self.muted


# ======================================================================
#  Layout helpers (no networkx dependency -- plain numpy)
# ======================================================================
def _circle_layout(count: int, radius: float = 1.0) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 2))
    if count == 1:
        return np.zeros((1, 2))
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False) - math.pi / 2
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _spring_layout(
    count: int,
    edges: Sequence[Tuple[int, int]],
    initial: Optional[np.ndarray] = None,
    iterations: int = 220,
    seed: int = 7,
) -> np.ndarray:
    """A small deterministic Fruchterman-Reingold layout."""
    if count == 0:
        return np.zeros((0, 2))
    if count == 1:
        return np.zeros((1, 2))

    rng = np.random.default_rng(seed)
    pos = _circle_layout(count) if initial is None else np.array(initial, dtype=float)
    pos = pos + rng.normal(0.0, 0.02, pos.shape)  # break perfect symmetry

    area = 1.0
    k = math.sqrt(area / count)
    temperature = 0.12
    cooling = temperature / (iterations + 1)

    edge_array = np.array(edges, dtype=int) if len(edges) else np.zeros((0, 2), dtype=int)

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distance, np.inf)
        distance = np.clip(distance, 0.005, None)

        # Repulsion between every pair of nodes.
        repulse = (k * k) / distance
        displacement = np.einsum("ijk,ij->ik", delta / distance[:, :, None], repulse)

        # Attraction along the edges.
        if len(edge_array):
            starts, ends = edge_array[:, 0], edge_array[:, 1]
            diff = pos[starts] - pos[ends]
            length = np.clip(np.linalg.norm(diff, axis=1), 0.005, None)
            force = (length * length) / k
            pull = (diff / length[:, None]) * force[:, None]
            np.add.at(displacement, starts, -pull)
            np.add.at(displacement, ends, pull)

        length = np.clip(np.linalg.norm(displacement, axis=1), 1e-9, None)
        step = np.minimum(length, temperature)
        pos = pos + (displacement / length[:, None]) * step[:, None]
        temperature = max(temperature - cooling, 1e-4)

    # Normalise into a tidy square.
    span = pos.max(axis=0) - pos.min(axis=0)
    span[span == 0] = 1.0
    pos = (pos - pos.min(axis=0)) / span * 2.0 - 1.0
    return pos


def _grouped_layout(labels: Sequence[str], edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Start each label's nodes in its own cluster, then relax with springs.

    Clustering the seed positions keeps nodes of the same label close together,
    which makes the imported graph far easier for a student to read.
    """
    unique = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    centres = _circle_layout(len(unique), radius=1.0)
    per_label: Dict[str, List[int]] = {label: [] for label in unique}
    for index, label in enumerate(labels):
        per_label[label].append(index)

    initial = np.zeros((len(labels), 2))
    for group_index, label in enumerate(unique):
        members = per_label[label]
        ring = _circle_layout(len(members), radius=0.28)
        for offset, node_index in enumerate(members):
            initial[node_index] = centres[group_index] + ring[offset]
    return _spring_layout(len(labels), edges, initial=initial, iterations=160)


# ======================================================================
#  Shared drawing helpers
# ======================================================================
def _arrow_annotation(
    x0: float, y0: float, x1: float, y1: float, color: str, shrink: float = 0.09
) -> Dict[str, Any]:
    """An edge drawn as an arrow that stops short of the target marker."""
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    return {
        "x": x1 - ux * shrink,
        "y": y1 - uy * shrink,
        "ax": x0 + ux * shrink,
        "ay": y0 + uy * shrink,
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "showarrow": True,
        "arrowhead": 2,
        "arrowsize": 1.1,
        "arrowwidth": 1.6,
        "arrowcolor": color,
        "standoff": 2,
        "startstandoff": 2,
    }


def _empty_figure(theme: Theme, message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14, "color": theme.muted},
    )
    _style_axes(figure, theme, height=320)
    return figure


def _style_axes(figure: go.Figure, theme: Theme, height: int) -> None:
    figure.update_xaxes(visible=False, showgrid=False, zeroline=False)
    # No equal-aspect lock: the diagram then fills the width of the page instead
    # of sitting in a square with dead space either side. Markers keep their
    # pixel size, so only the spacing between them changes.
    figure.update_yaxes(visible=False, showgrid=False, zeroline=False)
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 34, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme.ink},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": theme.ink, "size": 11},
        },
        hoverlabel={"font": {"size": 12}},
    )


# ======================================================================
#  1. Schema figure  (node labels + relationship types)
# ======================================================================
def schema_layout(
    schema: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Tuple[int, int]], List[Dict[str, Any]]]:
    """Positions for the schema diagram.

    Shared by the Plotly figure and the PDF report so the student sees the same
    picture on screen and in their submission.
    """
    nodes = [n for n in schema.get("nodes", []) if n.get("label")]
    labels = [n["label"] for n in nodes]
    index_of = {label: i for i, label in enumerate(labels)}

    edges: List[Tuple[int, int]] = []
    valid_rels: List[Dict[str, Any]] = []
    for rel in schema.get("relationships", []):
        source, target = rel.get("source"), rel.get("target")
        if source in index_of and target in index_of and rel.get("type"):
            edges.append((index_of[source], index_of[target]))
            valid_rels.append(rel)

    positions = (
        _circle_layout(len(labels), radius=1.0)
        if len(labels) <= 7
        else _spring_layout(len(labels), edges, iterations=200)
    )
    return nodes, positions, edges, valid_rels


def schema_figure(schema: Dict[str, Any], dark: bool = False) -> go.Figure:
    """Draw the *designed* schema. Redrawn whenever the student edits it."""
    theme = Theme(dark)
    nodes, positions, edges, valid_rels = schema_layout(schema)
    if not nodes:
        return _empty_figure(
            theme, "No node labels yet - add one in the Schema Designer to see the diagram."
        )

    figure = go.Figure()

    # Edges: recessive grey arrows with the relationship type written on them.
    annotations: List[Dict[str, Any]] = []
    mid_x, mid_y, mid_text = [], [], []
    seen_pairs: Dict[Tuple[int, int], int] = {}
    for (start, end), rel in zip(edges, valid_rels):
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        if start == end:
            # Self-relationship (e.g. User FOLLOWS User): draw a short loop label.
            mid_x.append(x0)
            mid_y.append(y0 + 0.22)
            mid_text.append(rel["type"] + " (self)")
            continue
        # Fan out parallel relationships so their labels do not overlap.
        pair = (min(start, end), max(start, end))
        rank = seen_pairs.get(pair, 0)
        seen_pairs[pair] = rank + 1
        offset = 0.0 if rank == 0 else (0.09 * rank * (1 if rank % 2 else -1))
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        ox, oy = nx * offset, ny * offset
        annotations.append(
            _arrow_annotation(x0 + ox, y0 + oy, x1 + ox, y1 + oy, theme.edge, shrink=0.16)
        )
        mid_x.append((x0 + x1) / 2 + ox)
        mid_y.append((y0 + y1) / 2 + oy)
        mid_text.append(rel["type"])

    if mid_x:
        figure.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="text",
                text=mid_text,
                textposition="middle center",
                textfont={"size": 10, "color": theme.muted},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # One trace per label so the legend names every label.
    for index, node in enumerate(nodes):
        x, y = positions[index]
        props = ", ".join(node.get("properties", [])) or "(no properties yet)"
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={
                    "size": 40,
                    "color": theme.color_for(index),
                    "line": {"width": 2, "color": theme.surface},
                },
                text=[node["label"]],
                textposition="bottom center",
                textfont={"size": 12, "color": theme.ink},
                name=node["label"],
                hovertemplate=(
                    "<b>%s</b><br>key: %s<br>properties: %s<extra></extra>"
                    % (node["label"], node.get("key") or "-", props)
                ),
            )
        )

    # No in-figure title: the page heading names the diagram, and a title here
    # would sit on top of the horizontal legend.
    figure.update_layout(annotations=annotations)
    _style_axes(figure, theme, height=520)
    return figure


# ======================================================================
#  2. Instance figure  (the imported / queried graph)
# ======================================================================
def instance_figure(
    snapshot: Dict[str, Any],
    dark: bool = False,
    labels_filter: Optional[Sequence[str]] = None,
    types_filter: Optional[Sequence[str]] = None,
    show_relationships: bool = True,
    show_edge_labels: bool = False,
    show_captions: bool = True,
    max_nodes: int = 120,
) -> Tuple[go.Figure, Dict[str, int]]:
    """Draw the imported graph. Returns ``(figure, counts_shown)``."""
    theme = Theme(dark)
    all_nodes = snapshot.get("nodes", [])
    all_rels = snapshot.get("relationships", [])

    if not all_nodes:
        return (
            _empty_figure(theme, "No graph data yet - import the dataset first."),
            {"nodes": 0, "relationships": 0},
        )

    if labels_filter:
        nodes = [n for n in all_nodes if n["label"] in labels_filter]
    else:
        nodes = list(all_nodes)
    truncated = len(nodes) > max_nodes
    nodes = nodes[:max_nodes]

    index_of = {n["nid"]: i for i, n in enumerate(nodes)}
    rels = []
    if show_relationships:
        for rel in all_rels:
            if types_filter and rel["type"] not in types_filter:
                continue
            if rel["source"] in index_of and rel["target"] in index_of:
                rels.append(rel)

    if not nodes:
        return (
            _empty_figure(theme, "No nodes match the current filter."),
            {"nodes": 0, "relationships": 0},
        )

    node_labels = [n["label"] for n in nodes]
    edges = [(index_of[r["source"]], index_of[r["target"]]) for r in rels]
    positions = _grouped_layout(node_labels, edges)

    figure = go.Figure()

    # Edges first so markers sit on top.
    annotations: List[Dict[str, Any]] = []
    mid_x, mid_y, mid_text = [], [], []
    draw_arrows = len(rels) <= 150  # keep the annotation count sane
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for rel, (start, end) in zip(rels, edges):
        x0, y0 = positions[start]
        x1, y1 = positions[end]
        if draw_arrows:
            annotations.append(_arrow_annotation(x0, y0, x1, y1, theme.edge))
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        if show_edge_labels:
            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            mid_text.append(rel["type"])

    if edge_x:
        figure.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"width": 1.4, "color": theme.edge},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    if mid_x:
        figure.add_trace(
            go.Scatter(
                x=mid_x,
                y=mid_y,
                mode="text",
                text=mid_text,
                textfont={"size": 9, "color": theme.muted},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # One trace per label -> legend entries plus stable colour assignment.
    ordered_labels: List[str] = []
    for label in node_labels:
        if label not in ordered_labels:
            ordered_labels.append(label)

    for slot, label in enumerate(ordered_labels):
        xs, ys, texts, hovers = [], [], [], []
        for index, node in enumerate(nodes):
            if node["label"] != label:
                continue
            xs.append(positions[index][0])
            ys.append(positions[index][1])
            texts.append(node["caption"])
            details = "<br>".join(
                "%s: %s" % (k, v) for k, v in list(node["properties"].items())[:8]
            )
            hovers.append("<b>:%s</b><br>%s<extra></extra>" % (label, details))
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text" if show_captions else "markers",
                marker={
                    "size": 20,
                    "color": theme.color_for(slot),
                    "line": {"width": 2, "color": theme.surface},
                },
                text=texts,
                textposition="top center",
                textfont={"size": 10, "color": theme.ink},
                name=label,
                hovertemplate=hovers,
            )
        )

    figure.update_layout(annotations=annotations)
    _style_axes(figure, theme, height=560)
    return figure, {
        "nodes": len(nodes),
        "relationships": len(rels),
        "truncated": truncated,
    }


# ======================================================================
#  3. Distribution bar chart (used in Observations)
# ======================================================================
def distribution_figure(
    counts: Dict[str, int], title: str, dark: bool = False
) -> go.Figure:
    """A single-series bar chart -- the title names the series, so no legend."""
    theme = Theme(dark)
    if not counts:
        return _empty_figure(theme, "Nothing to summarise yet.")
    names = list(counts.keys())
    values = [counts[name] for name in names]
    figure = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker={"color": theme.series[0], "cornerradius": 4},
            text=values,
            textposition="outside",
            textfont={"color": theme.ink, "size": 11},
            # Let the value label draw past the plot area instead of being
            # clipped on the tallest bar.
            cliponaxis=False,
            hovertemplate="%{x}: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        title={"text": title, "font": {"size": 14}},
        height=300,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": theme.ink},
        showlegend=False,
        bargap=0.35,
    )
    figure.update_xaxes(showgrid=False, tickfont={"color": theme.muted, "size": 11})
    figure.update_yaxes(
        showgrid=True,
        gridcolor=theme.grid,
        griddash="dot",
        zeroline=False,
        tickfont={"color": theme.muted, "size": 11},
        # Headroom so the outside value labels have somewhere to sit.
        range=[0, (max(values) if values else 1) * 1.18],
    )
    return figure
