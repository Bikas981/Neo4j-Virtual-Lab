"""
schema_tools.py
===============
The engineering behind the laboratory's middle steps:

* :func:`validate_schema`  -- checks the schema the student designed
* :func:`dataset_from_uploads` -- turns two uploaded CSVs into a dataset
* :func:`validate_dataset` -- checks the data against that schema
* :func:`generate_cypher`  -- writes the import script from schema + data
* :func:`build_local_graph` -- runs that same script on the in-memory engine

The import statements produced here are executed by the in-memory simulation
engine. They are an internal implementation detail and are not shown in the
laboratory interface.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from simulation_engine import InMemoryGraph

NODE_CSV_REQUIRED = ["label", "id"]
REL_CSV_REQUIRED = ["source_id", "type", "target_id"]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ======================================================================
#  Schema validation
# ======================================================================
def validate_schema(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for the designed schema.

    Errors block the import; warnings are teaching hints.
    """
    errors: List[str] = []
    warnings: List[str] = []

    nodes = schema.get("nodes", [])
    rels = schema.get("relationships", [])

    if not nodes:
        errors.append("The schema has no node labels yet. Add at least one node label.")

    seen_labels: List[str] = []
    for index, node in enumerate(nodes, start=1):
        label = (node.get("label") or "").strip()
        if not label:
            errors.append("Node %d: the node label cannot be empty." % index)
            continue
        if not _IDENTIFIER.match(label):
            errors.append(
                "Node label '%s' is not valid. Use letters, digits and underscores, "
                "starting with a letter (for example Student)." % label
            )
        if label in seen_labels:
            errors.append("Duplicate node label detected: '%s' is defined more than once." % label)
        seen_labels.append(label)

        if not label[:1].isupper():
            warnings.append(
                "Node label '%s' does not start with a capital letter. The "
                "convention is CamelCase, for example Student." % label
            )

        properties = [p.strip() for p in node.get("properties", []) if p and p.strip()]
        if not properties:
            warnings.append("Node '%s' has no properties. A node with no data is rarely useful." % label)
        if len(set(properties)) != len(properties):
            errors.append("Node '%s' lists the same property more than once." % label)
        for prop in properties:
            if not _IDENTIFIER.match(prop):
                errors.append(
                    "Property '%s' on node '%s' is not a valid name. Use snake_case, "
                    "for example student_id." % (prop, label)
                )

        key = (node.get("key") or "").strip()
        if not key:
            errors.append(
                "Node '%s' has no unique ID property. Every entity needs a key so "
                "that the import can recognise an entity that already exists and "
                "avoid creating a duplicate." % label
            )
        elif key not in properties:
            errors.append(
                "Node '%s' declares '%s' as its unique ID, but that property is not in "
                "its property list." % (label, key)
            )

    defined = set(seen_labels)
    if not rels:
        warnings.append(
            "The schema has no relationships yet. Without relationships this is a set "
            "of tables, not a knowledge graph."
        )

    seen_triples: List[Tuple[str, str, str]] = []
    for index, rel in enumerate(rels, start=1):
        rtype = (rel.get("type") or "").strip()
        source = (rel.get("source") or "").strip()
        target = (rel.get("target") or "").strip()

        if not rtype:
            errors.append("Relationship %d: the relationship type cannot be empty." % index)
        elif not _IDENTIFIER.match(rtype):
            errors.append(
                "Relationship type '%s' is not valid. Use UPPER_SNAKE_CASE, for example "
                "ENROLLED_IN." % rtype
            )
        elif rtype != rtype.upper():
            warnings.append(
                "Relationship type '%s' is not in UPPER_SNAKE_CASE. The convention "
                "is ENROLLED_IN rather than %s." % (rtype, rtype)
            )

        label_for = rtype or "relationship %d" % index
        if not source:
            errors.append("Relationship %s has no source node label." % label_for)
        elif source not in defined:
            errors.append(
                "Your relationship %s starts at %s, but no %s node has been defined."
                % (label_for, source, source)
            )
        if not target:
            errors.append("Relationship %s has no target node label." % label_for)
        elif target not in defined:
            errors.append(
                "Your relationship %s refers to %s, but no %s node has been defined."
                % (label_for, target, target)
            )

        properties = [p.strip() for p in rel.get("properties", []) if p and p.strip()]
        for prop in properties:
            if not _IDENTIFIER.match(prop):
                errors.append(
                    "Property '%s' on relationship %s is not a valid name." % (prop, label_for)
                )

        triple = (rtype, source, target)
        if rtype and source and target:
            if triple in seen_triples:
                errors.append(
                    "Duplicate relationship: %s from %s to %s is defined more than once."
                    % triple
                )
            seen_triples.append(triple)

    # A label that nothing connects to is legal but usually a design mistake.
    connected = {r.get("source") for r in rels} | {r.get("target") for r in rels}
    for label in seen_labels:
        if rels and label not in connected:
            warnings.append(
                "Node label '%s' takes part in no relationship. Isolated nodes cannot be "
                "reached by traversal." % label
            )

    return errors, warnings


def schema_lookup(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """label -> node definition, for quick lookups."""
    return {n["label"]: n for n in schema.get("nodes", []) if n.get("label")}


def relationship_triples(schema: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    return [
        (r.get("type", ""), r.get("source", ""), r.get("target", ""))
        for r in schema.get("relationships", [])
        if r.get("type")
    ]


# ======================================================================
#  Dataset loading from CSV
# ======================================================================
def _coerce(value: Any) -> Any:
    """Turn CSV text into int / float / bool where that is clearly intended."""
    if value is None:
        return None
    # A blank cell arrives from pandas as float NaN, which is not a property.
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (int, float, bool)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none", "null"):
        return None
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d*\.\d+", text):
        try:
            number = float(text)
            # "4.0" came from a spreadsheet column that pandas floated; the
            # underlying value is a whole number, so keep it as one.
            return int(number) if number.is_integer() else number
        except ValueError:
            return text
    return text


def dataset_from_uploads(
    nodes_df: pd.DataFrame, rels_df: Optional[pd.DataFrame], source: str = "Uploaded CSV"
) -> Tuple[Dict[str, Any], List[str]]:
    """Convert uploaded dataframes into the lab's dataset structure.

    Returns ``(dataset, structural_errors)``.  Structural errors are the ones
    that make the file unusable (missing required columns); everything else is
    reported later by :func:`validate_dataset`.
    """
    errors: List[str] = []
    dataset: Dict[str, Any] = {"source": source, "nodes": [], "edges": []}

    if nodes_df is None or nodes_df.empty:
        errors.append("The nodes CSV is empty. It needs at least one data row.")
        return dataset, errors

    columns = {str(c).strip().lower(): str(c) for c in nodes_df.columns}
    missing = [c for c in NODE_CSV_REQUIRED if c not in columns]
    if missing:
        errors.append(
            "The nodes CSV is missing the required column(s): %s. Expected columns are "
            "label, id and then one column per property." % ", ".join(missing)
        )
        return dataset, errors

    label_col, id_col = columns["label"], columns["id"]
    property_cols = [c for c in nodes_df.columns if c not in (label_col, id_col)]

    for position, row in nodes_df.iterrows():
        label = str(row[label_col]).strip() if pd.notna(row[label_col]) else ""
        raw_id = row[id_col]
        node_id = "" if pd.isna(raw_id) else str(raw_id).strip()
        properties: Dict[str, Any] = {}
        for column in property_cols:
            value = _coerce(row[column])
            if value is not None:
                properties[str(column).strip()] = value
        dataset["nodes"].append(
            {
                "label": label,
                "id": node_id,
                "properties": properties,
                "row": int(position) + 2,  # +2 = header row plus 1-based counting
            }
        )

    if rels_df is not None and not rels_df.empty:
        rel_columns = {str(c).strip().lower(): str(c) for c in rels_df.columns}
        rel_missing = [c for c in REL_CSV_REQUIRED if c not in rel_columns]
        if rel_missing:
            errors.append(
                "The relationships CSV is missing the required column(s): %s. Expected "
                "columns are source_id, type, target_id (plus optional properties)."
                % ", ".join(rel_missing)
            )
            return dataset, errors
        src_col = rel_columns["source_id"]
        type_col = rel_columns["type"]
        tgt_col = rel_columns["target_id"]
        src_label_col = rel_columns.get("source_label")
        tgt_label_col = rel_columns.get("target_label")
        skip = {src_col, type_col, tgt_col, src_label_col, tgt_label_col} - {None}
        property_cols = [c for c in rels_df.columns if c not in skip]

        for position, row in rels_df.iterrows():
            properties = {}
            for column in property_cols:
                value = _coerce(row[column])
                if value is not None:
                    properties[str(column).strip()] = value
            dataset["edges"].append(
                {
                    "type": str(row[type_col]).strip() if pd.notna(row[type_col]) else "",
                    "source": "" if pd.isna(row[src_col]) else str(row[src_col]).strip(),
                    "target": "" if pd.isna(row[tgt_col]) else str(row[tgt_col]).strip(),
                    "source_label": (
                        str(row[src_label_col]).strip()
                        if src_label_col and pd.notna(row[src_label_col])
                        else None
                    ),
                    "target_label": (
                        str(row[tgt_label_col]).strip()
                        if tgt_label_col and pd.notna(row[tgt_label_col])
                        else None
                    ),
                    "properties": properties,
                    "row": int(position) + 2,
                }
            )

    return dataset, errors


# ======================================================================
#  Dataset validation
# ======================================================================
def validate_dataset(dataset: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Check the dataset against the schema. Never raises; always reports."""
    report: Dict[str, Any] = {
        "ok": False,
        "errors": [],
        "warnings": [],
        "checks": [],
        "node_counts": {},
        "edge_counts": {},
    }
    nodes = dataset.get("nodes", [])
    edges = dataset.get("edges", [])

    if not nodes:
        report["errors"].append("The dataset contains no nodes, so there is nothing to import.")
        return report

    known_labels = schema_lookup(schema)
    triples = set(relationship_triples(schema))

    # --- nodes -------------------------------------------------------
    seen: Dict[Tuple[str, str], int] = {}
    label_of_id: Dict[str, str] = {}
    counts: Dict[str, int] = {}

    for node in nodes:
        row = node.get("row", "?")
        label, node_id = node.get("label", ""), str(node.get("id", ""))
        if not label:
            report["errors"].append("Row %s: the label cell is empty." % row)
            continue
        if not node_id:
            report["errors"].append("Row %s: the id cell is empty for a %s node." % (row, label))
            continue
        key = (label, node_id)
        if key in seen:
            report["errors"].append(
                "Duplicate id: %s appears more than once as a %s (rows %s and %s)."
                % (node_id, label, seen[key], row)
            )
        else:
            seen[key] = row
        if node_id in label_of_id and label_of_id[node_id] != label:
            report["warnings"].append(
                "The id %s is used by both %s and %s. Ids that are unique across the "
                "whole graph are easier to work with." % (node_id, label_of_id[node_id], label)
            )
        label_of_id[node_id] = label
        counts[label] = counts.get(label, 0) + 1

        definition = known_labels.get(label)
        if definition is None:
            report["warnings"].append(
                "The dataset contains a '%s' node, but '%s' is not defined in your "
                "schema. Add the label in the Schema Designer." % (label, label)
            )
        else:
            key_property = definition.get("key")
            if key_property and key_property not in node.get("properties", {}):
                report["errors"].append(
                    "Row %s: a %s node is missing its key property '%s'."
                    % (row, label, key_property)
                )
            declared = set(definition.get("properties", []))
            extra = set(node.get("properties", {})) - declared
            if extra:
                report["warnings"].append(
                    "Row %s: the %s node has properties not in the schema: %s."
                    % (row, label, ", ".join(sorted(extra)))
                )

    # --- relationships ----------------------------------------------
    edge_counts: Dict[str, int] = {}
    for edge in edges:
        row = edge.get("row", "?")
        rtype = edge.get("type", "")
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if not rtype:
            report["errors"].append("Row %s: the relationship type is empty." % row)
            continue
        if not source or not target:
            report["errors"].append(
                "Row %s: relationship %s needs both a source_id and a target_id." % (row, rtype)
            )
            continue
        if source not in label_of_id:
            report["errors"].append(
                "Row %s: relationship %s points from '%s', but no node with that id "
                "exists in the dataset." % (row, rtype, source)
            )
            continue
        if target not in label_of_id:
            report["errors"].append(
                "Row %s: relationship %s points to '%s', but no node with that id "
                "exists in the dataset." % (row, rtype, target)
            )
            continue
        edge_counts[rtype] = edge_counts.get(rtype, 0) + 1
        triple = (rtype, label_of_id[source], label_of_id[target])
        if triples and triple not in triples:
            report["warnings"].append(
                "The dataset has %s from %s to %s, which your schema does not declare."
                % triple
            )

    report["node_counts"] = dict(sorted(counts.items()))
    report["edge_counts"] = dict(sorted(edge_counts.items()))
    for label, count in report["node_counts"].items():
        report["checks"].append("%d %s node(s) detected" % (count, label))
    total_edges = sum(report["edge_counts"].values())
    if total_edges:
        report["checks"].append("%d relationship(s) detected" % total_edges)
    report["ok"] = not report["errors"]
    return report


# ======================================================================
#  Cypher generation
# ======================================================================
def cypher_value(value: Any) -> str:
    """Render a Python value as a Cypher literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return "'%s'" % text


def constraint_statements(schema: Dict[str, Any]) -> List[str]:
    """Uniqueness constraints -- the proper way to stop duplicate entities."""
    statements = []
    for node in schema.get("nodes", []):
        label, key = node.get("label"), node.get("key")
        if label and key:
            statements.append(
                "CREATE CONSTRAINT %s_%s_unique IF NOT EXISTS\n"
                "FOR (n:%s) REQUIRE n.%s IS UNIQUE;" % (label.lower(), key, label, key)
            )
    return statements


def generate_cypher(
    schema: Dict[str, Any],
    dataset: Dict[str, Any],
    use_merge: bool = True,
    include_constraints: bool = False,
    limit: Optional[int] = None,
) -> str:
    """Write the import script for the current schema + dataset."""
    lookup = schema_lookup(schema)
    label_of_id = {str(n["id"]): n["label"] for n in dataset.get("nodes", []) if n.get("id")}

    lines: List[str] = [
        "// ====================================================",
        "// Knowledge graph import script",
        "// Domain: %s" % schema.get("domain", "custom"),
        "// Generated by the Virtual Lab from your schema and dataset",
        "// ====================================================",
        "",
    ]

    if include_constraints:
        lines.append("// --- 1. Uniqueness constraints (prevent duplicate entities) ---")
        statements = constraint_statements(schema)
        lines.extend(statements if statements else ["// (no key properties defined)"])
        lines.append("")

    lines.append("// --- %s. Nodes ---" % ("2" if include_constraints else "1"))
    nodes = dataset.get("nodes", [])
    if limit:
        nodes = nodes[:limit]
    for node in nodes:
        label = node.get("label")
        if not label:
            continue
        properties = dict(node.get("properties", {}))
        definition = lookup.get(label, {})
        key = definition.get("key")
        if use_merge and key and key in properties:
            key_value = properties.pop(key)
            statement = "MERGE (n:%s {%s: %s})" % (label, key, cypher_value(key_value))
            if properties:
                assignments = ", ".join(
                    "n.%s = %s" % (k, cypher_value(v)) for k, v in properties.items()
                )
                statement += "\nSET %s" % assignments
            lines.append(statement + ";")
        else:
            assignments = ", ".join(
                "%s: %s" % (k, cypher_value(v)) for k, v in properties.items()
            )
            keyword = "MERGE" if use_merge else "CREATE"
            lines.append("%s (n:%s {%s});" % (keyword, label, assignments))
    lines.append("")

    lines.append("// --- %s. Relationships ---" % ("3" if include_constraints else "2"))
    edges = dataset.get("edges", [])
    if limit:
        edges = edges[:limit]
    wrote_edge = False
    for edge in edges:
        rtype = edge.get("type")
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if not rtype or not source or not target:
            continue
        source_label = edge.get("source_label") or label_of_id.get(source)
        target_label = edge.get("target_label") or label_of_id.get(target)
        if not source_label or not target_label:
            continue
        source_key = lookup.get(source_label, {}).get("key") or "id"
        target_key = lookup.get(target_label, {}).get("key") or "id"
        statement = (
            "MATCH (a:%s {%s: %s}), (b:%s {%s: %s})\n%s (a)-[r:%s]->(b)"
            % (
                source_label,
                source_key,
                cypher_value(source),
                target_label,
                target_key,
                cypher_value(target),
                "MERGE" if use_merge else "CREATE",
                rtype,
            )
        )
        properties = edge.get("properties") or {}
        if properties:
            assignments = ", ".join(
                "r.%s = %s" % (k, cypher_value(v)) for k, v in properties.items()
            )
            statement += "\nSET %s" % assignments
        lines.append(statement + ";")
        wrote_edge = True
    if not wrote_edge:
        lines.append("// (no relationships in the dataset)")

    return "\n".join(lines) + "\n"


def sample_statements(schema: Dict[str, Any], dataset: Dict[str, Any]) -> Dict[str, str]:
    """Three short teaching examples generated from the live schema."""
    nodes = dataset.get("nodes", [])
    lookup = schema_lookup(schema)
    examples: Dict[str, str] = {}

    if nodes:
        first = nodes[0]
        label = first["label"]
        properties = dict(first.get("properties", {}))
        key = lookup.get(label, {}).get("key")
        inline = ",\n    ".join("%s: %s" % (k, cypher_value(v)) for k, v in properties.items())
        examples["CREATE"] = "CREATE (n:%s {\n    %s\n});" % (label, inline)
        if key and key in properties:
            rest = {k: v for k, v in properties.items() if k != key}
            sets = ", ".join("n.%s = %s" % (k, cypher_value(v)) for k, v in rest.items())
            examples["MERGE"] = "MERGE (n:%s {%s: %s})%s;" % (
                label,
                key,
                cypher_value(properties[key]),
                "\nSET %s" % sets if sets else "",
            )

    edges = dataset.get("edges", [])
    label_of_id = {str(n["id"]): n["label"] for n in nodes if n.get("id")}
    if edges:
        edge = edges[0]
        source_label = edge.get("source_label") or label_of_id.get(str(edge["source"]), "")
        target_label = edge.get("target_label") or label_of_id.get(str(edge["target"]), "")
        source_key = lookup.get(source_label, {}).get("key") or "id"
        target_key = lookup.get(target_label, {}).get("key") or "id"
        examples["RELATIONSHIP"] = (
            "MATCH (a:%s), (b:%s)\nWHERE a.%s = %s\n  AND b.%s = %s\nMERGE (a)-[:%s]->(b);"
            % (
                source_label,
                target_label,
                source_key,
                cypher_value(edge["source"]),
                target_key,
                cypher_value(edge["target"]),
                edge["type"],
            )
        )
    return examples


# ======================================================================
#  Local import
# ======================================================================
def build_local_graph(
    script: str, graph: Optional[InMemoryGraph] = None
) -> Tuple[InMemoryGraph, Dict[str, int], List[str]]:
    """Run the generated script on the in-memory engine (Local Simulation Mode)."""
    graph = graph or InMemoryGraph()
    totals, errors = execute_script(graph, script)
    return graph, totals, errors


def dataset_summary(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Counts used by the observations panel and the report."""
    labels: Dict[str, int] = {}
    for node in dataset.get("nodes", []):
        label = node.get("label") or "(unlabelled)"
        labels[label] = labels.get(label, 0) + 1
    types: Dict[str, int] = {}
    for edge in dataset.get("edges", []):
        rtype = edge.get("type") or "(untyped)"
        types[rtype] = types.get(rtype, 0) + 1
    return {
        "source": dataset.get("source", "unknown"),
        "node_total": len(dataset.get("nodes", [])),
        "edge_total": len(dataset.get("edges", [])),
        "label_counts": dict(sorted(labels.items())),
        "type_counts": dict(sorted(types.items())),
    }
