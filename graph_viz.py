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

import json
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


# ======================================================================
#  4. Animated Flow & Physics Canvas (60 FPS Interactive HTML5 Engine)
# ======================================================================
def animated_graph_html(
    graph_data: Dict[str, Any],
    dark: bool = False,
    height: int = 520,
    mode: str = "schema",
    show_edge_labels: bool = True,
    show_captions: bool = True,
) -> str:
    """Generate a self-contained, interactive HTML5 Canvas visualization.

    Features:
    * 60 FPS animated glowing particles flowing continuously along relationship arrows.
    * Real-time spring physics (force-directed layout) with draggable, elastic nodes.
    * Interactive Path Tracer: click a node to highlight outgoing traversals and see node HUD.
    * Smooth zoom, pan, physics lock/unlock, and speed controls.
    * 100% offline-ready with zero external dependencies.
    """
    theme = Theme(dark)

    if mode == "schema":
        raw_nodes = [n for n in graph_data.get("nodes", []) if n.get("label")]
        labels = [n["label"] for n in raw_nodes]
        color_map = {label: theme.color_for(i) for i, label in enumerate(labels)}

        nodes_json = []
        for n in raw_nodes:
            nodes_json.append({
                "id": n["label"],
                "label": n["label"],
                "caption": n["label"],
                "key": n.get("key", ""),
                "props": ", ".join(n.get("properties", [])) or "(none)",
                "color": color_map[n["label"]],
                "radius": 28,
            })

        node_ids = {n["id"] for n in nodes_json}
        edges_json = []
        for r in graph_data.get("relationships", []):
            src, tgt = r.get("source"), r.get("target")
            if src in node_ids and tgt in node_ids and r.get("type"):
                edges_json.append({
                    "source": src,
                    "target": tgt,
                    "type": r.get("type"),
                    "props": ", ".join(r.get("properties", [])) or "-",
                    "color": theme.edge,
                })

    else:  # instance mode
        raw_nodes = graph_data.get("nodes", [])
        raw_rels = graph_data.get("relationships", [])

        unique_labels = sorted({n["label"] for n in raw_nodes if "label" in n})
        color_map = {label: theme.color_for(i) for i, label in enumerate(unique_labels)}

        nodes_json = []
        for n in raw_nodes:
            props_str = ", ".join(f"{k}: {v}" for k, v in n.get("properties", {}).items()) or "(none)"
            caption = n.get("caption", n["label"]) if show_captions else n["label"]
            nodes_json.append({
                "id": str(n["nid"]),
                "label": n["label"],
                "caption": caption,
                "key": str(n.get("properties", {}).get("id", n["nid"])),
                "props": props_str,
                "color": color_map.get(n["label"], theme.color_for(0)),
                "radius": 19 if len(raw_nodes) > 25 else 23,
            })

        node_ids = {n["id"] for n in nodes_json}
        edges_json = []
        for r in raw_rels:
            src, tgt = str(r["source"]), str(r["target"])
            if src in node_ids and tgt in node_ids:
                props_str = ", ".join(f"{k}: {v}" for k, v in r.get("properties", {}).items()) or "-"
                edges_json.append({
                    "source": src,
                    "target": tgt,
                    "type": r.get("type", "") if show_edge_labels else "",
                    "props": props_str,
                    "color": theme.edge,
                })

    nodes_json_str = json.dumps(nodes_json)
    edges_json_str = json.dumps(edges_json)
    is_dark_str = "true" if dark else "false"
    height_str = str(height)

    bg_gradient = "linear-gradient(135deg, #0c1322 0%, #111827 100%)" if dark else "linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)"
    border_color = "#334155" if dark else "#e2e8f0"
    toolbar_bg = "rgba(15, 23, 42, 0.78)" if dark else "rgba(255, 255, 255, 0.88)"
    text_color = "#f1f5f9" if dark else "#0f172a"
    subtext_color = "#94a3b8" if dark else "#64748b"
    btn_hover = "rgba(255, 255, 255, 0.12)" if dark else "rgba(0, 0, 0, 0.06)"

    template = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { overflow: hidden; background: transparent; user-select: none; }
  #canvas-container {
    position: relative;
    width: 100%;
    height: __HEIGHT__px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid __BORDER_COLOR__;
    background: __BG_GRADIENT__;
  }
  canvas { display: block; width: 100%; height: 100%; }
  .toolbar {
    position: absolute;
    top: 10px;
    left: 12px;
    right: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    pointer-events: none;
    z-index: 5;
  }
  .btn-group {
    display: flex;
    gap: 6px;
    pointer-events: auto;
    background: __TOOLBAR_BG__;
    backdrop-filter: blur(8px);
    border: 1px solid __BORDER_COLOR__;
    padding: 4px 8px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
  .tool-btn {
    background: transparent;
    border: 1px solid transparent;
    color: __TEXT_COLOR__;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .tool-btn:hover {
    background: __BTN_HOVER__;
    border-color: __BORDER_COLOR__;
  }
  .hint-text {
    font-size: 11px;
    color: __SUBTEXT_COLOR__;
    background: __TOOLBAR_BG__;
    backdrop-filter: blur(8px);
    border: 1px solid __BORDER_COLOR__;
    padding: 4px 10px;
    border-radius: 8px;
    pointer-events: auto;
  }
  #hud {
    position: absolute;
    bottom: 12px;
    right: 12px;
    max-width: 320px;
    background: __TOOLBAR_BG__;
    backdrop-filter: blur(12px);
    border: 1px solid __BORDER_COLOR__;
    color: __TEXT_COLOR__;
    padding: 10px 14px;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    display: none;
    z-index: 10;
  }
</style>
</head>
<body>
<div id="canvas-container">
  <div class="toolbar">
    <div class="btn-group">
      <button class="tool-btn" id="btn-flow" title="Toggle particle flow">Pause Flow</button>
      <button class="tool-btn" id="btn-speed" title="Toggle pulse speed">Speed: 1x</button>
      <button class="tool-btn" id="btn-phys" title="Lock/Unlock spring physics">Physics: Unlocked</button>
      <button class="tool-btn" id="btn-reset" title="Reset view">Center</button>
      <button class="tool-btn" id="btn-zin" title="Zoom In">+</button>
      <button class="tool-btn" id="btn-zout" title="Zoom Out">-</button>
    </div>
    <div class="hint-text">Click a node to trace paths | Drag to move | Scroll to zoom</div>
  </div>

  <div id="hud"></div>
  <canvas id="kgCanvas"></canvas>
</div>

<script>
(function() {
  const rawNodes = __NODES_DATA__;
  const rawEdges = __EDGES_DATA__;
  const isDark = __IS_DARK__;
  const targetHeight = __HEIGHT__;

  const canvas = document.getElementById('kgCanvas');
  const ctx = canvas.getContext('2d');
  const hud = document.getElementById('hud');

  let width = 800;
  let height = targetHeight;
  let dpr = window.devicePixelRatio || 1;

  function resize() {
    dpr = window.devicePixelRatio || 1;
    width = canvas.parentElement.clientWidth || 800;
    height = targetHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
  }
  resize();
  window.addEventListener('resize', resize);

  const N = rawNodes.length;
  const circleR = Math.min(width, height) * 0.32;
  const nodes = rawNodes.map((d, i) => {
    const angle = (i / Math.max(1, N)) * 2 * Math.PI - Math.PI / 2;
    return {
      ...d,
      x: width / 2 + circleR * Math.cos(angle) + (Math.random() - 0.5) * 20,
      y: height / 2 + circleR * Math.sin(angle) + (Math.random() - 0.5) * 20,
      vx: 0,
      vy: 0,
      r: d.radius || (N > 25 ? 19 : 24)
    };
  });

  const nodeMap = {};
  nodes.forEach(n => nodeMap[n.id] = n);

  const edges = rawEdges.map(e => ({
    ...e,
    sourceNode: nodeMap[e.source],
    targetNode: nodeMap[e.target],
    particles: [
      { progress: Math.random() },
      { progress: (Math.random() + 0.5) % 1.0 }
    ]
  })).filter(e => e.sourceNode && e.targetNode);

  let physicsRunning = true;
  let flowRunning = true;
  let speedMultiplier = 1.0;
  let selectedNode = null;
  let hoveredNode = null;
  let draggedNode = null;
  let pan = { x: 0, y: 0 };
  let zoom = 1.0;
  let isPanning = false;
  let startMouse = { x: 0, y: 0 };

  const bgColor = isDark ? "#0c1322" : "#f8fafc";
  const bgDotColor = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.05)";
  const edgeColor = isDark ? "#475569" : "#cbd5e1";
  const edgeHighlightColor = isDark ? "#38bdf8" : "#0284c7";
  const textColor = isDark ? "#f1f5f9" : "#0f172a";
  const subtextColor = isDark ? "#94a3b8" : "#64748b";
  const pillBg = isDark ? "rgba(15, 23, 42, 0.90)" : "rgba(255, 255, 255, 0.90)";
  const particleGlow = isDark ? "#38bdf8" : "#0284c7";
  const selectHaloColor = isDark ? "#f59e0b" : "#d97706";

  function toWorld(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    return {
      x: (mx - pan.x) / zoom,
      y: (my - pan.y) / zoom
    };
  }

  function findNode(wx, wy) {
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      const dist = Math.hypot(n.x - wx, n.y - wy);
      if (dist <= n.r + 5) return n;
    }
    return null;
  }

  canvas.addEventListener('mousedown', e => {
    const w = toWorld(e.clientX, e.clientY);
    const hit = findNode(w.x, w.y);
    if (hit) {
      draggedNode = hit;
      selectedNode = hit;
      updateHUD();
    } else {
      isPanning = true;
      startMouse = { x: e.clientX - pan.x, y: e.clientY - pan.y };
      selectedNode = null;
      updateHUD();
    }
  });

  window.addEventListener('mousemove', e => {
    const w = toWorld(e.clientX, e.clientY);
    hoveredNode = findNode(w.x, w.y);
    canvas.style.cursor = hoveredNode ? 'pointer' : (isPanning ? 'grabbing' : 'default');

    if (draggedNode) {
      draggedNode.x = w.x;
      draggedNode.y = w.y;
      draggedNode.vx = 0;
      draggedNode.vy = 0;
    } else if (isPanning) {
      pan.x = e.clientX - startMouse.x;
      pan.y = e.clientY - startMouse.y;
    }
  });

  window.addEventListener('mouseup', () => {
    draggedNode = null;
    isPanning = false;
  });

  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    const wBefore = toWorld(e.clientX, e.clientY);
    zoom = Math.max(0.3, Math.min(3.5, zoom * zoomFactor));
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    pan.x = mx - wBefore.x * zoom;
    pan.y = my - wBefore.y * zoom;
  });

  document.getElementById('btn-flow').onclick = () => {
    flowRunning = !flowRunning;
    document.getElementById('btn-flow').innerText = flowRunning ? 'Pause Flow' : 'Play Flow';
  };
  document.getElementById('btn-speed').onclick = () => {
    if (speedMultiplier === 1.0) speedMultiplier = 2.0;
    else if (speedMultiplier === 2.0) speedMultiplier = 3.0;
    else speedMultiplier = 1.0;
    document.getElementById('btn-speed').innerText = 'Speed: ' + speedMultiplier + 'x';
  };
  document.getElementById('btn-phys').onclick = () => {
    physicsRunning = !physicsRunning;
    document.getElementById('btn-phys').innerText = physicsRunning ? 'Physics: Unlocked' : 'Physics: Locked';
  };
  document.getElementById('btn-reset').onclick = () => {
    pan = { x: 0, y: 0 };
    zoom = 1.0;
    selectedNode = null;
    updateHUD();
  };
  document.getElementById('btn-zin').onclick = () => { zoom = Math.min(3.5, zoom * 1.2); };
  document.getElementById('btn-zout').onclick = () => { zoom = Math.max(0.3, zoom / 1.2); };

  function updateHUD() {
    if (!selectedNode) {
      hud.style.display = 'none';
      return;
    }
    hud.style.display = 'block';
    const outgoing = edges.filter(e => e.sourceNode === selectedNode);
    const incoming = edges.filter(e => e.targetNode === selectedNode);

    let outList = outgoing.map(e => `--[${e.type}]--> <b>${e.targetNode.caption || e.targetNode.label}</b>`).join('<br>');
    if (!outList) outList = '<span style="opacity:0.6;">(none)</span>';

    let inList = incoming.map(e => `<--[${e.type}]-- <b>${e.sourceNode.caption || e.sourceNode.label}</b>`).join('<br>');
    if (!inList) inList = '<span style="opacity:0.6;">(none)</span>';

    hud.innerHTML = `
      <div style="font-weight:700;font-size:13px;margin-bottom:4px;color:${isDark ? '#38bdf8' : '#0284c7'};display:flex;justify-content:space-between;align-items:center;">
        <span>${selectedNode.label}: ${selectedNode.caption || selectedNode.id}</span>
        <span style="cursor:pointer;opacity:0.7;font-size:16px;padding-left:10px;" id="hud-close">&times;</span>
      </div>
      <div style="font-size:11px;opacity:0.85;margin-bottom:6px;"><b>Properties:</b> ${selectedNode.props || '-'}</div>
      <div style="font-size:11px;border-top:1px solid ${isDark ? '#334155' : '#e2e8f0'};padding-top:5px;margin-top:5px;">
        <div style="color:${isDark ? '#34d399' : '#059669'};font-weight:600;">Outgoing Traversal (${outgoing.length}):</div>
        <div style="margin-left:4px;line-height:1.4;">${outList}</div>
      </div>
      <div style="font-size:11px;border-top:1px solid ${isDark ? '#334155' : '#e2e8f0'};padding-top:5px;margin-top:5px;">
        <div style="color:${isDark ? '#a78bfa' : '#7c3aed'};font-weight:600;">Incoming Traversal (${incoming.length}):</div>
        <div style="margin-left:4px;line-height:1.4;">${inList}</div>
      </div>
    `;
    const closeBtn = document.getElementById('hud-close');
    if (closeBtn) closeBtn.onclick = () => { selectedNode = null; updateHUD(); };
  }

  function updatePhysics() {
    if (!physicsRunning && !draggedNode) return;
    const repStrength = Math.min(22000, 45000 / Math.max(1, Math.sqrt(nodes.length)));
    const springLen = Math.min(200, Math.max(90, Math.min(width, height) / Math.max(2, Math.sqrt(nodes.length))));
    const springK = 0.035;
    const centerGravity = 0.012;

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const n1 = nodes[i];
        const n2 = nodes[j];
        let dx = n2.x - n1.x;
        let dy = n2.y - n1.y;
        let dist = Math.hypot(dx, dy) || 1;
        if (dist < 450) {
          const force = repStrength / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          n1.vx -= fx;
          n1.vy -= fy;
          n2.vx += fx;
          n2.vy += fy;
        }
      }
    }

    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      let dx = e.targetNode.x - e.sourceNode.x;
      let dy = e.targetNode.y - e.sourceNode.y;
      let dist = Math.hypot(dx, dy) || 1;
      const delta = dist - springLen;
      const force = delta * springK;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      e.sourceNode.vx += fx;
      e.sourceNode.vy += fy;
      e.targetNode.vx -= fx;
      e.targetNode.vy -= fy;
    }

    const cx = width / 2;
    const cy = height / 2;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      if (n === draggedNode) continue;
      n.vx += (cx - n.x) * centerGravity;
      n.vy += (cy - n.y) * centerGravity;
      n.vx *= 0.87;
      n.vy *= 0.87;

      const v = Math.hypot(n.vx, n.vy);
      if (v > 10) {
        n.vx = (n.vx / v) * 10;
        n.vy = (n.vy / v) * 10;
      }
      n.x += n.vx;
      n.y += n.vy;
    }
  }

  function render() {
    updatePhysics();

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const dotSpacing = 36;
    ctx.fillStyle = bgDotColor;
    const startX = -pan.x / zoom - 50;
    const startY = -pan.y / zoom - 50;
    const endX = (width - pan.x) / zoom + 50;
    const endY = (height - pan.y) / zoom + 50;
    for (let gx = Math.floor(startX / dotSpacing) * dotSpacing; gx < endX; gx += dotSpacing) {
      for (let gy = Math.floor(startY / dotSpacing) * dotSpacing; gy < endY; gy += dotSpacing) {
        ctx.beginPath();
        ctx.arc(gx, gy, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    edges.forEach(e => {
      const s = e.sourceNode;
      const t = e.targetNode;
      let dx = t.x - s.x;
      let dy = t.y - s.y;
      let dist = Math.hypot(dx, dy) || 1;
      let ux = dx / dist;
      let uy = dy / dist;

      const isConnectedToSelected = selectedNode && (s === selectedNode || t === selectedNode);
      const isOutgoingFromSelected = selectedNode && (s === selectedNode);

      const x1 = s.x + ux * s.r;
      const y1 = s.y + uy * s.r;
      const x2 = t.x - ux * (t.r + 6);
      const y2 = t.y - uy * (t.r + 6);

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = isConnectedToSelected ? edgeHighlightColor : edgeColor;
      ctx.lineWidth = isConnectedToSelected ? 2.8 : 1.6;
      ctx.stroke();

      const arrowSize = isConnectedToSelected ? 9 : 7;
      const arrowX = t.x - ux * (t.r + 2);
      const arrowY = t.y - uy * (t.r + 2);
      ctx.beginPath();
      ctx.moveTo(arrowX, arrowY);
      ctx.lineTo(arrowX - ux * arrowSize - uy * (arrowSize * 0.55), arrowY - uy * arrowSize + ux * (arrowSize * 0.55));
      ctx.lineTo(arrowX - ux * arrowSize + uy * (arrowSize * 0.55), arrowY - uy * arrowSize - ux * (arrowSize * 0.55));
      ctx.closePath();
      ctx.fillStyle = isConnectedToSelected ? edgeHighlightColor : edgeColor;
      ctx.fill();

      if (e.type) {
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        const txtWidth = ctx.measureText(e.type).width;

        ctx.fillStyle = pillBg;
        ctx.beginPath();
        ctx.roundRect(mx - txtWidth / 2 - 5, my - 7, txtWidth + 10, 14, 4);
        ctx.fill();
        ctx.strokeStyle = isConnectedToSelected ? edgeHighlightColor : (isDark ? "#334155" : "#e2e8f0");
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = isConnectedToSelected ? edgeHighlightColor : subtextColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(e.type, mx, my);
      }

      if (flowRunning) {
        const baseSpeed = 0.0055 * speedMultiplier;
        const currentSpeed = isOutgoingFromSelected ? baseSpeed * 2.2 : baseSpeed;
        e.particles.forEach(p => {
          p.progress = (p.progress + currentSpeed) % 1.0;
          const px = x1 + (x2 - x1) * p.progress;
          const py = y1 + (y2 - y1) * p.progress;

          const tailLen = isOutgoingFromSelected ? 15 : 9;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px - ux * tailLen, py - uy * tailLen);
          ctx.strokeStyle = isOutgoingFromSelected ? selectHaloColor : particleGlow;
          ctx.lineWidth = isOutgoingFromSelected ? 3.4 : 2.2;
          ctx.lineCap = 'round';
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(px, py, isOutgoingFromSelected ? 4.2 : 3.0, 0, Math.PI * 2);
          ctx.fillStyle = isOutgoingFromSelected ? "#fef08a" : "#ffffff";
          ctx.fill();

          const rad = isOutgoingFromSelected ? 11 : 6;
          const glow = ctx.createRadialGradient(px, py, 1, px, py, rad);
          glow.addColorStop(0, isOutgoingFromSelected ? "rgba(245, 158, 11, 0.7)" : "rgba(56, 189, 248, 0.6)");
          glow.addColorStop(1, "rgba(56, 189, 248, 0)");
          ctx.beginPath();
          ctx.arc(px, py, rad, 0, Math.PI * 2);
          ctx.fillStyle = glow;
          ctx.fill();
        });
      }
    });

    nodes.forEach(n => {
      const isSelected = (n === selectedNode);
      const isHovered = (n === hoveredNode);

      if (isSelected || isHovered) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 7, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? (isDark ? "rgba(245, 158, 11, 0.25)" : "rgba(217, 119, 6, 0.2)") : (isDark ? "rgba(56, 189, 248, 0.2)" : "rgba(2, 132, 199, 0.15)");
        ctx.fill();
        ctx.strokeStyle = isSelected ? selectHaloColor : edgeHighlightColor;
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = n.color || "#3b82f6";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.4;
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const dispText = (n.caption || n.label || "").substring(0, 14);
      ctx.fillText(dispText, n.x, n.y);

      if (n.caption && n.label && n.caption !== n.label) {
        ctx.fillStyle = subtextColor;
        ctx.font = '9px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.fillText(n.label, n.x, n.y + n.r + 12);
      }
    });

    ctx.restore();
    ctx.restore();

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
})();
</script>
</body>
</html>
"""

    return (
        template
        .replace("__NODES_DATA__", nodes_json_str)
        .replace("__EDGES_DATA__", edges_json_str)
        .replace("__IS_DARK__", is_dark_str)
        .replace("__HEIGHT__", height_str)
        .replace("__BG_GRADIENT__", bg_gradient)
        .replace("__BORDER_COLOR__", border_color)
        .replace("__TOOLBAR_BG__", toolbar_bg)
        .replace("__TEXT_COLOR__", text_color)
        .replace("__SUBTEXT_COLOR__", subtext_color)
        .replace("__BTN_HOVER__", btn_hover)
    )

