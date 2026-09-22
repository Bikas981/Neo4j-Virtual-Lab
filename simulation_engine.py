"""
simulation_engine.py
====================
Local, in-memory property-graph engine for the Knowledge Graph Virtual Lab.

This module is what stores and queries the knowledge graph of the laboratory:

* :class:`InMemoryGraph` -- stores nodes / relationships / properties in Python.
* :func:`execute_cypher` -- interprets the query language used internally.

Supported Cypher subset
-----------------------
``MATCH`` / ``OPTIONAL MATCH``, ``WHERE``, ``CREATE``, ``MERGE``,
``ON CREATE SET`` / ``ON MATCH SET``, ``SET``, ``DELETE``, ``DETACH DELETE``,
``WITH``, ``RETURN`` (with ``DISTINCT`` and aggregates), ``ORDER BY``,
``SKIP``, ``LIMIT``.  Variable-length patterns such as ``-[:ENROLLED_IN*1..3]->``
are supported so students can see multi-hop traversal.

This is deliberately a *teaching* interpreter: it covers the queries used in
this experiment, not the whole Cypher language.  Anything it cannot parse is
reported back as a friendly message instead of a traceback.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

SIMULATION_MODE = "Local Simulation"
NEO4J_MODE = "Neo4j"


# ======================================================================
#  Graph data model
# ======================================================================
@dataclass
class SimNode:
    """A property-graph node held in memory."""

    nid: int
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.labels[0] if self.labels else ""

    def caption(self) -> str:
        """Short human-readable caption used by the visualisations."""
        for key in ("name", "title", "full_name"):
            if self.properties.get(key):
                return str(self.properties[key])
        for key, value in self.properties.items():
            if key.endswith("_id") or key == "id":
                return str(value)
        return "%s#%d" % (self.label, self.nid)

    def key_value(self) -> Any:
        for key, value in self.properties.items():
            if key.endswith("_id") or key == "id":
                return value
        return self.nid

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"_labels": list(self.labels)}
        out.update(self.properties)
        return out


@dataclass
class SimRel:
    """A directed, typed relationship held in memory."""

    rid: int
    rtype: str
    start: int
    end: int
    properties: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"_type": self.rtype}
        out.update(self.properties)
        return out


class CypherError(Exception):
    """Raised for anything the teaching interpreter cannot handle."""


class InMemoryGraph:
    """A tiny property graph: nodes, typed relationships and properties."""

    def __init__(self) -> None:
        self.nodes: Dict[int, SimNode] = {}
        self.rels: Dict[int, SimRel] = {}
        self._next_nid = 0
        self._next_rid = 0

    # -- lifecycle ----------------------------------------------------
    def clear(self) -> None:
        self.nodes.clear()
        self.rels.clear()
        self._next_nid = 0
        self._next_rid = 0

    def is_empty(self) -> bool:
        return not self.nodes and not self.rels

    # -- writes -------------------------------------------------------
    def create_node(self, labels: Sequence[str], properties: Dict[str, Any]) -> SimNode:
        node = SimNode(self._next_nid, [l for l in labels if l], dict(properties))
        self.nodes[node.nid] = node
        self._next_nid += 1
        return node

    def create_rel(
        self, start: int, end: int, rtype: str, properties: Dict[str, Any]
    ) -> SimRel:
        rel = SimRel(self._next_rid, rtype, start, end, dict(properties))
        self.rels[rel.rid] = rel
        self._next_rid += 1
        return rel

    def merge_node(
        self, labels: Sequence[str], properties: Dict[str, Any]
    ) -> Tuple[SimNode, bool]:
        """Return ``(node, created)`` -- matches on the given labels+properties."""
        found = self.find_nodes(labels, properties)
        if found:
            return found[0], False
        return self.create_node(labels, properties), True

    def merge_rel(
        self, start: int, end: int, rtype: str, properties: Dict[str, Any]
    ) -> Tuple[SimRel, bool]:
        for rel in self.rels.values():
            if (
                rel.start == start
                and rel.end == end
                and rel.rtype == rtype
                and all(rel.properties.get(k) == v for k, v in properties.items())
            ):
                return rel, False
        return self.create_rel(start, end, rtype, properties), True

    def delete_node(self, nid: int, detach: bool = False) -> int:
        """Delete a node; returns the number of relationships removed with it."""
        attached = [r.rid for r in self.rels.values() if r.start == nid or r.end == nid]
        if attached and not detach:
            raise CypherError(
                "Cannot delete a node that still has relationships. Use "
                "DETACH DELETE to remove the node together with its relationships."
            )
        for rid in attached:
            self.rels.pop(rid, None)
        self.nodes.pop(nid, None)
        return len(attached)

    def delete_rel(self, rid: int) -> None:
        self.rels.pop(rid, None)

    # -- reads --------------------------------------------------------
    def find_nodes(
        self,
        labels: Optional[Sequence[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[SimNode]:
        wanted = [l for l in (labels or []) if l]
        props = properties or {}
        out: List[SimNode] = []
        for node in self.nodes.values():
            if wanted and not all(l in node.labels for l in wanted):
                continue
            if props and not all(
                node.properties.get(k) == v for k, v in props.items()
            ):
                continue
            out.append(node)
        return out

    def rels_of(self, nid: int, direction: str, types: Sequence[str]) -> List[SimRel]:
        """``direction`` is one of ``out``, ``in`` or ``both``."""
        wanted = [t for t in (types or []) if t]
        out: List[SimRel] = []
        for rel in self.rels.values():
            if wanted and rel.rtype not in wanted:
                continue
            if direction in ("out", "both") and rel.start == nid:
                out.append(rel)
            elif direction in ("in", "both") and rel.end == nid:
                out.append(rel)
        return out

    def labels(self) -> List[str]:
        seen: List[str] = []
        for node in self.nodes.values():
            for label in node.labels:
                if label not in seen:
                    seen.append(label)
        return sorted(seen)

    def rel_types(self) -> List[str]:
        return sorted({r.rtype for r in self.rels.values()})

    def label_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.label] = counts.get(node.label, 0) + 1
        return dict(sorted(counts.items()))

    def rel_type_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rel in self.rels.values():
            counts[rel.rtype] = counts.get(rel.rtype, 0) + 1
        return dict(sorted(counts.items()))

    def stats(self) -> Dict[str, Any]:
        node_count = len(self.nodes)
        rel_count = len(self.rels)
        return {
            "node_count": node_count,
            "relationship_count": rel_count,
            "label_count": len(self.labels()),
            "relationship_type_count": len(self.rel_types()),
            "avg_rels_per_node": round(rel_count / node_count, 2) if node_count else 0.0,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Plain-Python snapshot used by the visualisation helpers."""
        return {
            "nodes": [
                {
                    "nid": n.nid,
                    "label": n.label,
                    "caption": n.caption(),
                    "properties": dict(n.properties),
                }
                for n in self.nodes.values()
            ],
            "relationships": [
                {
                    "rid": r.rid,
                    "type": r.rtype,
                    "source": r.start,
                    "target": r.end,
                    "properties": dict(r.properties),
                }
                for r in self.rels.values()
            ],
        }

    def nodes_dataframe(self) -> pd.DataFrame:
        rows = []
        for node in self.nodes.values():
            row: Dict[str, Any] = {"id": node.nid, "label": node.label}
            row.update(node.properties)
            rows.append(row)
        return pd.DataFrame(rows)

    def rels_dataframe(self) -> pd.DataFrame:
        rows = []
        for rel in self.rels.values():
            start = self.nodes.get(rel.start)
            end = self.nodes.get(rel.end)
            row: Dict[str, Any] = {
                "source": start.caption() if start else rel.start,
                "source_label": start.label if start else "",
                "type": rel.rtype,
                "target": end.caption() if end else rel.end,
                "target_label": end.label if end else "",
            }
            row.update(rel.properties)
            rows.append(row)
        return pd.DataFrame(rows)


# ======================================================================
#  Query result container
# ======================================================================
@dataclass
class QueryResult:
    """Uniform result object returned by both execution backends."""

    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    execution_ms: float = 0.0
    mode: str = SIMULATION_MODE
    query: str = ""
    notice: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def record_count(self) -> int:
        return len(self.rows)

    def summary_text(self) -> str:
        parts = ["%s: %d" % (k.replace("_", " "), v) for k, v in self.summary.items() if v]
        return ", ".join(parts)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.columns:
            return pd.DataFrame()
        display = [[value_to_display(v) for v in row] for row in self.rows]
        frame = pd.DataFrame(display, columns=self.columns)
        # A column holding e.g. both numbers and the text "null" cannot be
        # converted to an Arrow table, so give those columns one type.
        for column in frame.columns:
            if frame[column].dtype == object:
                kinds = {type(v) for v in frame[column]}
                if len(kinds) > 1:
                    frame[column] = frame[column].astype(str)
        return frame


def value_to_display(value: Any) -> Any:
    """Render nodes / relationships / lists as readable table cells."""
    if isinstance(value, SimNode):
        props = ", ".join("%s: %s" % (k, v) for k, v in value.properties.items())
        return "(:%s {%s})" % (":".join(value.labels), props)
    if isinstance(value, SimRel):
        if value.properties:
            props = ", ".join("%s: %s" % (k, v) for k, v in value.properties.items())
            return "[:%s {%s}]" % (value.rtype, props)
        return "[:%s]" % value.rtype
    if isinstance(value, (list, tuple)):
        return ", ".join(str(value_to_display(v)) for v in value)
    if isinstance(value, dict):
        return ", ".join("%s: %s" % (k, v) for k, v in value.items())
    if value is None:
        return "null"
    return value


# ======================================================================
#  Expression tokenizer + evaluator
# ======================================================================
_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
      | (?P<number>\d+\.\d+|\d+)
      | (?P<op><>|!=|<=|>=|=~|\+=|[=<>+\-*/%,.:()\[\]{}|$])
      | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
    )
    """,
    re.VERBOSE,
)

_KEYWORD_OPS = {
    "AND", "OR", "XOR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE",
    "CONTAINS", "STARTS", "ENDS", "WITH", "AS", "DISTINCT", "DESC", "ASC",
}

_AGGREGATES = {"count", "collect", "sum", "avg", "min", "max"}


@dataclass
class Token:
    kind: str
    text: str


def tokenize(text: str) -> List[Token]:
    """Split an expression into tokens; raises on unknown characters."""
    tokens: List[Token] = []
    pos = 0
    length = len(text)
    while pos < length:
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(text, pos)
        if not match or match.end() == pos:
            raise CypherError("Could not understand the characters near %r." % text[pos : pos + 20])
        pos = match.end()
        if match.group("string") is not None:
            tokens.append(Token("string", match.group("string")))
        elif match.group("number") is not None:
            tokens.append(Token("number", match.group("number")))
        elif match.group("op") is not None:
            tokens.append(Token("op", match.group("op")))
        else:
            tokens.append(Token("word", match.group("word")))
    return tokens


class _Parser:
    """Recursive-descent parser/evaluator for Cypher expressions."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- token helpers ------------------------------------------------
    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek_word(self) -> str:
        token = self.peek()
        return token.text.upper() if token and token.kind == "word" else ""

    def next(self) -> Token:
        if self.pos >= len(self.tokens):
            raise CypherError("The expression ended unexpectedly.")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def accept_op(self, *ops: str) -> Optional[str]:
        token = self.peek()
        if token and token.kind == "op" and token.text in ops:
            self.pos += 1
            return token.text
        return None

    def accept_word(self, *words: str) -> Optional[str]:
        token = self.peek()
        if token and token.kind == "word" and token.text.upper() in words:
            self.pos += 1
            return token.text.upper()
        return None

    def expect_op(self, op: str) -> None:
        if not self.accept_op(op):
            raise CypherError("Expected %r in the expression." % op)

    def at_end(self) -> bool:
        return self.pos >= len(self.tokens)

    # -- grammar ------------------------------------------------------
    def parse_expression(self, ctx: "EvalContext") -> Any:
        return self.parse_or(ctx)

    def parse_or(self, ctx: "EvalContext") -> Any:
        left = self.parse_and(ctx)
        while True:
            if self.accept_word("OR"):
                right = self.parse_and(ctx)
                left = bool(left) or bool(right)
            elif self.accept_word("XOR"):
                right = self.parse_and(ctx)
                left = bool(left) != bool(right)
            else:
                return left

    def parse_and(self, ctx: "EvalContext") -> Any:
        left = self.parse_not(ctx)
        while self.accept_word("AND"):
            right = self.parse_not(ctx)
            left = bool(left) and bool(right)
        return left

    def parse_not(self, ctx: "EvalContext") -> Any:
        if self.accept_word("NOT"):
            return not bool(self.parse_not(ctx))
        return self.parse_comparison(ctx)

    def parse_comparison(self, ctx: "EvalContext") -> Any:
        left = self.parse_additive(ctx)
        while True:
            op = self.accept_op("=", "<>", "!=", "<", "<=", ">", ">=", "=~")
            if op:
                right = self.parse_additive(ctx)
                left = _compare(op, left, right)
                continue
            word = self.peek_word()
            if word == "IS":
                self.next()
                if self.accept_word("NOT"):
                    self.accept_word("NULL")
                    left = left is not None
                else:
                    self.accept_word("NULL")
                    left = left is None
                continue
            if word == "IN":
                self.next()
                right = self.parse_additive(ctx)
                left = bool(right) and left in right if isinstance(right, (list, tuple)) else False
                continue
            if word == "CONTAINS":
                self.next()
                right = self.parse_additive(ctx)
                left = str(right) in str(left) if left is not None else False
                continue
            if word == "STARTS":
                self.next()
                self.accept_word("WITH")
                right = self.parse_additive(ctx)
                left = str(left).startswith(str(right)) if left is not None else False
                continue
            if word == "ENDS":
                self.next()
                self.accept_word("WITH")
                right = self.parse_additive(ctx)
                left = str(left).endswith(str(right)) if left is not None else False
                continue
            return left

    def parse_additive(self, ctx: "EvalContext") -> Any:
        left = self.parse_multiplicative(ctx)
        while True:
            op = self.accept_op("+", "-")
            if not op:
                return left
            right = self.parse_multiplicative(ctx)
            left = _arith(op, left, right)

    def parse_multiplicative(self, ctx: "EvalContext") -> Any:
        left = self.parse_unary(ctx)
        while True:
            op = self.accept_op("*", "/", "%")
            if not op:
                return left
            right = self.parse_unary(ctx)
            left = _arith(op, left, right)

    def parse_unary(self, ctx: "EvalContext") -> Any:
        if self.accept_op("-"):
            value = self.parse_unary(ctx)
            return -value if isinstance(value, (int, float)) else None
        if self.accept_op("+"):
            return self.parse_unary(ctx)
        return self.parse_primary(ctx)

    def parse_primary(self, ctx: "EvalContext") -> Any:
        token = self.next()
        if token.kind == "string":
            return _unquote(token.text)
        if token.kind == "number":
            return float(token.text) if "." in token.text else int(token.text)
        if token.kind == "op":
            if token.text == "(":
                value = self.parse_expression(ctx)
                self.expect_op(")")
                return value
            if token.text == "[":
                items: List[Any] = []
                if not self.accept_op("]"):
                    while True:
                        items.append(self.parse_expression(ctx))
                        if self.accept_op(","):
                            continue
                        self.expect_op("]")
                        break
                return items
            if token.text == "{":
                mapping: Dict[str, Any] = {}
                if not self.accept_op("}"):
                    while True:
                        key = self.next().text
                        self.expect_op(":")
                        mapping[key] = self.parse_expression(ctx)
                        if self.accept_op(","):
                            continue
                        self.expect_op("}")
                        break
                return mapping
            if token.text == "$":
                name = self.next().text
                return ctx.parameters.get(name)
            if token.text == "*":
                return "*"
            raise CypherError("Unexpected symbol %r in the expression." % token.text)

        word = token.text
        upper = word.upper()
        if upper == "TRUE":
            return True
        if upper == "FALSE":
            return False
        if upper == "NULL":
            return None

        # function call?
        if self.peek() and self.peek().kind == "op" and self.peek().text == "(":
            self.next()  # consume '('
            args: List[Any] = []
            distinct = False
            if not self.accept_op(")"):
                if self.accept_word("DISTINCT"):
                    distinct = True
                while True:
                    args.append(self.parse_expression(ctx))
                    if self.accept_op(","):
                        continue
                    self.expect_op(")")
                    break
            value = _call_function(word, args, distinct)
            return self._parse_trailer(value, ctx)

        value = ctx.lookup(word)
        return self._parse_trailer(value, ctx)

    def _parse_trailer(self, value: Any, ctx: "EvalContext") -> Any:
        """Handle property access such as ``s.name`` (possibly chained)."""
        while self.peek() and self.peek().kind == "op" and self.peek().text == ".":
            self.next()
            prop = self.next().text
            value = _get_property(value, prop)
        return value


class EvalContext:
    """Variable bindings + parameters available to an expression."""

    def __init__(
        self,
        bindings: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        strict: bool = False,
    ) -> None:
        self.bindings = bindings
        self.parameters = parameters or {}
        self.strict = strict

    def lookup(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.strict:
            raise CypherError(
                "Variable '%s' is not defined. Did you bind it in a MATCH or CREATE clause?" % name
            )
        return None


def _unquote(text: str) -> str:
    body = text[1:-1]
    return body.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def _get_property(value: Any, prop: str) -> Any:
    if isinstance(value, SimNode) or isinstance(value, SimRel):
        return value.properties.get(prop)
    if isinstance(value, dict):
        return value.get(prop)
    return None


def _compare(op: str, left: Any, right: Any) -> Any:
    if op == "=":
        return left == right
    if op in ("<>", "!="):
        return left != right
    if op == "=~":
        try:
            return bool(re.fullmatch(str(right), str(left)))
        except re.error:
            return False
    if left is None or right is None:
        return False
    try:
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
    except TypeError:
        # Comparing e.g. a number with a string -- compare as text instead.
        left, right = str(left), str(right)
        return _compare(op, left, right)
    return False


def _arith(op: str, left: Any, right: Any) -> Any:
    if op == "+" and (isinstance(left, str) or isinstance(right, str)):
        return "%s%s" % ("" if left is None else left, "" if right is None else right)
    if left is None or right is None:
        return None
    try:
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right if right else None
        if op == "%":
            return left % right if right else None
    except TypeError:
        return None
    return None


def _call_function(name: str, args: List[Any], distinct: bool) -> Any:
    """Scalar functions.  Aggregates are handled separately in projection."""
    lname = name.lower()
    if lname in _AGGREGATES:
        # Reached only when an aggregate is used outside a projection.
        raise CypherError(
            "%s() can only be used in a RETURN or WITH clause." % name
        )
    if lname == "tolower":
        return str(args[0]).lower() if args and args[0] is not None else None
    if lname == "toupper":
        return str(args[0]).upper() if args and args[0] is not None else None
    if lname == "trim":
        return str(args[0]).strip() if args and args[0] is not None else None
    if lname in ("tointeger", "toint"):
        try:
            return int(float(args[0]))
        except (TypeError, ValueError):
            return None
    if lname == "tofloat":
        try:
            return float(args[0])
        except (TypeError, ValueError):
            return None
    if lname == "tostring":
        return None if not args or args[0] is None else str(args[0])
    if lname == "id":
        value = args[0] if args else None
        if isinstance(value, SimNode):
            return value.nid
        if isinstance(value, SimRel):
            return value.rid
        return None
    if lname == "labels":
        value = args[0] if args else None
        return list(value.labels) if isinstance(value, SimNode) else []
    if lname == "type":
        value = args[0] if args else None
        return value.rtype if isinstance(value, SimRel) else None
    if lname == "keys":
        value = args[0] if args else None
        if isinstance(value, (SimNode, SimRel)):
            return list(value.properties.keys())
        return []
    if lname == "properties":
        value = args[0] if args else None
        if isinstance(value, (SimNode, SimRel)):
            return dict(value.properties)
        return {}
    if lname == "size" or lname == "length":
        value = args[0] if args else None
        try:
            return len(value)
        except TypeError:
            return None
    if lname == "coalesce":
        for arg in args:
            if arg is not None:
                return arg
        return None
    if lname == "round":
        try:
            return round(float(args[0]), int(args[1]) if len(args) > 1 else 0)
        except (TypeError, ValueError, IndexError):
            return None
    raise CypherError(
        "The function %s() is not supported by the local simulation engine." % name
    )


def eval_expression(
    text: str, bindings: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None
) -> Any:
    parser = _Parser(tokenize(text))
    value = parser.parse_expression(EvalContext(bindings, parameters))
    if not parser.at_end():
        raise CypherError("Could not parse the whole expression: %r" % text.strip())
    return value


# ======================================================================
#  Clause splitting
# ======================================================================
_CLAUSES = [
    "OPTIONAL MATCH",
    "DETACH DELETE",
    "ON CREATE SET",
    "ON MATCH SET",
    "ORDER BY",
    "MATCH",
    "WHERE",
    "CREATE",
    "MERGE",
    "SET",
    "REMOVE",
    "DELETE",
    "RETURN",
    "SKIP",
    "LIMIT",
    "WITH",
    "UNWIND",
]
# Longest first so that e.g. DETACH DELETE wins over DELETE.
_CLAUSES_SORTED = sorted(_CLAUSES, key=len, reverse=True)
_WORD_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


def strip_comments(query: str) -> str:
    lines = []
    for line in query.splitlines():
        idx = line.find("//")
        if idx >= 0:
            # Keep '//' that lives inside a quoted string.
            before = line[:idx]
            if before.count("'") % 2 == 0 and before.count('"') % 2 == 0:
                line = before
        lines.append(line)
    return "\n".join(lines)


def split_clauses(query: str) -> List[Tuple[str, str]]:
    """Split a query into ``(CLAUSE, body)`` pairs, ignoring quotes/brackets."""
    clauses: List[Tuple[str, str]] = []
    current_kw: Optional[str] = None
    start = 0
    pos = 0
    depth = 0
    quote: Optional[str] = None
    upper = query.upper()
    length = len(query)

    while pos < length:
        char = query[pos]
        if quote:
            if char == "\\":
                pos += 2
                continue
            if char == quote:
                quote = None
            pos += 1
            continue
        if char in "'\"":
            quote = char
            pos += 1
            continue
        if char in "([{":
            depth += 1
            pos += 1
            continue
        if char in ")]}":
            depth -= 1
            pos += 1
            continue
        if depth == 0:
            matched = None
            for keyword in _CLAUSES_SORTED:
                end = pos + len(keyword)
                if upper.startswith(keyword, pos):
                    # A keyword only starts a clause when it stands alone: not
                    # inside a parameter ($limit) or a property (n.limit).
                    previous = query[pos - 1] if pos else ""
                    before_ok = pos == 0 or (
                        previous not in _WORD_CHARS and previous not in "$."
                    )
                    after_ok = end >= length or query[end] not in _WORD_CHARS
                    if before_ok and after_ok:
                        matched = keyword
                        break
            # "STARTS WITH" / "ENDS WITH" are operators, not a WITH clause.
            if matched == "WITH":
                preceding = query[:pos].rstrip().upper()
                if preceding.endswith("STARTS") or preceding.endswith("ENDS"):
                    matched = None
                    pos += len("WITH")
                    continue
            if matched:
                body = query[start:pos].strip()
                if current_kw is None:
                    if body:
                        raise CypherError(
                            "The query should start with a clause such as MATCH, "
                            "CREATE or MERGE (found %r)." % body[:30]
                        )
                else:
                    clauses.append((current_kw, body))
                current_kw = matched
                pos += len(matched)
                start = pos
                continue
        pos += 1

    if quote:
        raise CypherError("A quoted string was never closed -- check your ' or \" marks.")
    if depth != 0:
        raise CypherError("Unbalanced brackets -- check your ( ), [ ] or { } pairs.")
    if current_kw is None:
        raise CypherError(
            "No Cypher clause was recognised. Start with MATCH, CREATE or MERGE."
        )
    clauses.append((current_kw, query[start:].strip()))
    return clauses


def split_top_level(text: str, separator: str = ",") -> List[str]:
    """Split on a separator that is not inside quotes or brackets."""
    parts: List[str] = []
    depth = 0
    quote: Optional[str] = None
    buf: List[str] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if quote:
            buf.append(char)
            if char == "\\" and pos + 1 < len(text):
                buf.append(text[pos + 1])
                pos += 2
                continue
            if char == quote:
                quote = None
            pos += 1
            continue
        if char in "'\"":
            quote = char
            buf.append(char)
            pos += 1
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == separator and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            pos += 1
            continue
        buf.append(char)
        pos += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


# ======================================================================
#  Pattern parsing
# ======================================================================
@dataclass
class NodePattern:
    var: str = ""
    labels: List[str] = field(default_factory=list)
    props: str = ""


@dataclass
class RelPattern:
    var: str = ""
    types: List[str] = field(default_factory=list)
    props: str = ""
    direction: str = "out"  # out | in | both
    min_hops: int = 1
    max_hops: int = 1
    var_length: bool = False


def _match_bracket(text: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    quote: Optional[str] = None
    pos = start
    while pos < len(text):
        char = text[pos]
        if quote:
            if char == "\\":
                pos += 2
                continue
            if char == quote:
                quote = None
            pos += 1
            continue
        if char in "'\"":
            quote = char
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise CypherError("Unbalanced %s%s in the pattern." % (open_ch, close_ch))


def _parse_node_inner(inner: str) -> NodePattern:
    props = ""
    brace = inner.find("{")
    if brace >= 0:
        end = _match_bracket(inner, brace, "{", "}")
        props = inner[brace : end + 1]
        inner = inner[:brace] + inner[end + 1 :]
    inner = inner.strip()
    parts = [p.strip() for p in inner.split(":")]
    var = parts[0] if parts else ""
    labels = [p for p in parts[1:] if p]
    if var and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var):
        raise CypherError("%r is not a valid variable name in a node pattern." % var)
    return NodePattern(var=var, labels=labels, props=props)


def _parse_rel_inner(inner: str, direction: str) -> RelPattern:
    props = ""
    brace = inner.find("{")
    if brace >= 0:
        end = _match_bracket(inner, brace, "{", "}")
        props = inner[brace : end + 1]
        inner = inner[:brace] + inner[end + 1 :]
    inner = inner.strip()
    min_hops, max_hops, var_length = 1, 1, False
    star = inner.find("*")
    if star >= 0:
        spec = inner[star + 1 :].strip()
        inner = inner[:star].strip()
        var_length = True
        if not spec:
            min_hops, max_hops = 1, 5
        elif ".." in spec:
            low, _, high = spec.partition("..")
            min_hops = int(low) if low.strip().isdigit() else 1
            max_hops = int(high) if high.strip().isdigit() else 5
        elif spec.isdigit():
            min_hops = max_hops = int(spec)
        if max_hops > 8:
            max_hops = 8  # keep the teaching engine responsive
    parts = [p.strip() for p in inner.split(":")]
    var = parts[0] if parts else ""
    types: List[str] = []
    for part in parts[1:]:
        types.extend([t.strip() for t in part.split("|") if t.strip()])
    return RelPattern(
        var=var,
        types=types,
        props=props,
        direction=direction,
        min_hops=min_hops,
        max_hops=max_hops,
        var_length=var_length,
    )


def parse_path(pattern: str) -> List[Any]:
    """Parse ``(a)-[:R]->(b)`` into ``[NodePattern, RelPattern, NodePattern]``."""
    pattern = pattern.strip()
    # A named path such as "p = (a)-->(b)" -- drop the name, keep the path.
    named = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.*)$", pattern, re.DOTALL)
    if named:
        pattern = named.group(1).strip()
    if not pattern:
        raise CypherError("An empty pattern was found -- expected something like (n:Label).")

    elements: List[Any] = []
    pos = 0
    expect_node = True
    while pos < len(pattern):
        if pattern[pos].isspace():
            pos += 1
            continue
        if expect_node:
            if pattern[pos] != "(":
                raise CypherError(
                    "Expected a node pattern like (n:Label) near %r." % pattern[pos : pos + 20]
                )
            end = _match_bracket(pattern, pos, "(", ")")
            elements.append(_parse_node_inner(pattern[pos + 1 : end]))
            pos = end + 1
            expect_node = False
            continue

        # relationship
        left_arrow = False
        if pattern[pos] == "<":
            left_arrow = True
            pos += 1
        if pos >= len(pattern) or pattern[pos] != "-":
            raise CypherError(
                "Expected a relationship such as -[:TYPE]-> near %r." % pattern[pos : pos + 20]
            )
        pos += 1
        inner = ""
        if pos < len(pattern) and pattern[pos] == "[":
            end = _match_bracket(pattern, pos, "[", "]")
            inner = pattern[pos + 1 : end]
            pos = end + 1
        if pos >= len(pattern) or pattern[pos] != "-":
            raise CypherError("A relationship pattern is incomplete -- expected '-' or '->'.")
        pos += 1
        right_arrow = False
        if pos < len(pattern) and pattern[pos] == ">":
            right_arrow = True
            pos += 1
        if left_arrow and right_arrow:
            raise CypherError("A relationship cannot point in both directions.")
        direction = "in" if left_arrow else ("out" if right_arrow else "both")
        elements.append(_parse_rel_inner(inner, direction))
        expect_node = True

    if expect_node:
        raise CypherError("The pattern ends with a relationship -- it needs a node after it.")
    return elements


def _props_dict(
    props_text: str, bindings: Dict[str, Any], parameters: Dict[str, Any]
) -> Dict[str, Any]:
    if not props_text:
        return {}
    value = eval_expression(props_text, bindings, parameters)
    if not isinstance(value, dict):
        raise CypherError("Expected a property map such as {name: 'Aditi'}.")
    return value


# ======================================================================
#  Pattern matching
# ======================================================================
_PRIVATE = "__used_rels"


def _node_matches(
    node: SimNode, pat: NodePattern, binding: Dict[str, Any], params: Dict[str, Any]
) -> bool:
    if pat.labels and not all(l in node.labels for l in pat.labels):
        return False
    props = _props_dict(pat.props, binding, params)
    if props and not all(node.properties.get(k) == v for k, v in props.items()):
        return False
    if pat.var and pat.var in binding:
        bound = binding[pat.var]
        if not isinstance(bound, SimNode) or bound.nid != node.nid:
            return False
    return True


def _node_candidates(
    graph: InMemoryGraph, pat: NodePattern, binding: Dict[str, Any], params: Dict[str, Any]
) -> List[SimNode]:
    if pat.var and pat.var in binding:
        bound = binding[pat.var]
        if isinstance(bound, SimNode) and _node_matches(bound, pat, binding, params):
            return [bound]
        return []
    props = _props_dict(pat.props, binding, params)
    return graph.find_nodes(pat.labels, props)


def _rel_steps(
    graph: InMemoryGraph,
    node: SimNode,
    pat: RelPattern,
    binding: Dict[str, Any],
    params: Dict[str, Any],
) -> List[Tuple[Any, SimNode]]:
    """Return the ``(rel_or_rel_list, end_node)`` options leaving ``node``."""
    props = _props_dict(pat.props, binding, params)
    used = binding.get(_PRIVATE, frozenset())

    def one_hop(from_nid: int, blocked) -> List[Tuple[SimRel, SimNode]]:
        options: List[Tuple[SimRel, SimNode]] = []
        for rel in graph.rels_of(from_nid, pat.direction, pat.types):
            if rel.rid in blocked:
                continue
            if props and not all(rel.properties.get(k) == v for k, v in props.items()):
                continue
            other_id = rel.end if rel.start == from_nid else rel.start
            other = graph.nodes.get(other_id)
            if other is not None:
                options.append((rel, other))
        return options

    if not pat.var_length:
        return [(rel, target) for rel, target in one_hop(node.nid, used)]

    # Variable-length traversal, explored breadth-first with a hop limit.
    results: List[Tuple[Any, SimNode]] = []
    if pat.min_hops == 0:
        results.append(([], node))
    frontier: List[Tuple[List[SimRel], SimNode]] = [([], node)]
    for depth in range(1, pat.max_hops + 1):
        next_frontier: List[Tuple[List[SimRel], SimNode]] = []
        for path, current in frontier:
            blocked = used | frozenset(r.rid for r in path)
            for rel, target in one_hop(current.nid, blocked):
                new_path = path + [rel]
                next_frontier.append((new_path, target))
                if depth >= pat.min_hops:
                    results.append((new_path, target))
        frontier = next_frontier
        if not frontier:
            break
    return results


def _expand(
    graph: InMemoryGraph,
    elements: List[Any],
    idx: int,
    binding: Dict[str, Any],
    params: Dict[str, Any],
    out: List[Dict[str, Any]],
    forced: Optional[SimNode] = None,
) -> None:
    pat = elements[idx]
    if forced is not None:
        candidates = [forced] if _node_matches(forced, pat, binding, params) else []
    else:
        candidates = _node_candidates(graph, pat, binding, params)

    for node in candidates:
        current = dict(binding)
        if pat.var:
            current[pat.var] = node
        if idx + 1 >= len(elements):
            out.append(current)
            continue
        rel_pat = elements[idx + 1]
        for rel_value, target in _rel_steps(graph, node, rel_pat, current, params):
            nxt = dict(current)
            if rel_pat.var:
                nxt[rel_pat.var] = rel_value
            used = nxt.get(_PRIVATE, frozenset())
            new_rels = rel_value if isinstance(rel_value, list) else [rel_value]
            nxt[_PRIVATE] = used | frozenset(r.rid for r in new_rels)
            _expand(graph, elements, idx + 2, nxt, params, out, forced=target)


def match_pattern(
    graph: InMemoryGraph,
    pattern: str,
    bindings: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    elements = parse_path(pattern)
    out: List[Dict[str, Any]] = []
    for binding in bindings:
        _expand(graph, elements, 0, binding, params, out)
    return out


# ======================================================================
#  Projection (RETURN / WITH)
# ======================================================================
_AGG_RE = re.compile(r"^(count|collect|sum|avg|min|max)\s*\(", re.IGNORECASE)


def _split_alias(item: str) -> Tuple[str, str]:
    """Split an ``expr AS alias`` item into its two halves."""
    depth = 0
    quote: Optional[str] = None
    pos = 0
    while pos < len(item):
        char = item[pos]
        if quote:
            if char == "\\":
                pos += 2
                continue
            if char == quote:
                quote = None
            pos += 1
            continue
        if char in "'\"":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and item[pos : pos + 2].upper() == "AS":
            previous = item[pos - 1] if pos else ""
            before_ok = pos == 0 or (
                previous not in _WORD_CHARS and previous not in "$."
            )
            after = pos + 2
            after_ok = after >= len(item) or item[after] not in _WORD_CHARS
            if before_ok and after_ok:
                return item[:pos].strip(), item[after:].strip()
        pos += 1
    return item.strip(), ""


def _is_aggregate(expr: str) -> bool:
    text = expr.strip()
    if not _AGG_RE.match(text):
        return False
    # Confirm the call spans the whole expression, so "count(n) + 1" is rejected.
    open_idx = text.index("(")
    try:
        close_idx = _match_bracket(text, open_idx, "(", ")")
    except CypherError:
        return False
    return close_idx == len(text) - 1


def _group_key(value: Any) -> Any:
    if isinstance(value, SimNode):
        return ("node", value.nid)
    if isinstance(value, SimRel):
        return ("rel", value.rid)
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_group_key(v) for v in value))
    if isinstance(value, dict):
        return ("map", tuple(sorted((k, str(_group_key(v))) for k, v in value.items())))
    return value


def _eval_aggregate(expr: str, group: List[Dict[str, Any]], params: Dict[str, Any]) -> Any:
    text = expr.strip()
    open_idx = text.index("(")
    name = text[:open_idx].strip().lower()
    inner = text[open_idx + 1 : _match_bracket(text, open_idx, "(", ")")].strip()
    distinct = False
    if inner.upper().startswith("DISTINCT "):
        distinct = True
        inner = inner[len("DISTINCT ") :].strip()

    if inner == "*":
        values: List[Any] = [1 for _ in group]
    else:
        values = [eval_expression(inner, b, params) for b in group]

    if distinct:
        seen: List[Any] = []
        unique: List[Any] = []
        for value in values:
            key = _group_key(value)
            if key not in seen:
                seen.append(key)
                unique.append(value)
        values = unique

    present = [v for v in values if v is not None]
    if name == "count":
        return len(values) if inner == "*" else len(present)
    if name == "collect":
        return present
    numeric = [v for v in present if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if name == "sum":
        return sum(numeric)
    if name == "avg":
        return round(sum(numeric) / len(numeric), 3) if numeric else None
    if name == "min":
        return min(present) if present else None
    if name == "max":
        return max(present) if present else None
    return None


def project(
    body: str, bindings: List[Dict[str, Any]], params: Dict[str, Any]
) -> Tuple[List[str], List[List[Any]], List[Dict[str, Any]]]:
    """Evaluate a RETURN / WITH body.

    Returns ``(columns, rows, row_bindings)``.  ``row_bindings`` carries the
    aliases forward so ORDER BY -- and clauses after WITH -- can use them.
    """
    text = body.strip()
    distinct = False
    if text.upper().startswith("DISTINCT "):
        distinct = True
        text = text[len("DISTINCT ") :].strip()
    if not text:
        raise CypherError("RETURN needs at least one expression, for example RETURN n.")

    if text == "*":
        names: List[str] = []
        for binding in bindings:
            for key in binding:
                if not key.startswith("__") and key not in names:
                    names.append(key)
        items = [(name, name) for name in names]
        if not items:
            raise CypherError("RETURN * found no variables to return.")
    else:
        items = []
        for raw in split_top_level(text):
            expr, alias = _split_alias(raw)
            if not expr:
                raise CypherError("An empty item was found in the RETURN list.")
            items.append((expr, alias or expr))

    columns = [alias for _, alias in items]
    aggregates = [_is_aggregate(expr) for expr, _ in items]

    rows: List[List[Any]] = []
    row_bindings: List[Dict[str, Any]] = []

    if not any(aggregates):
        for binding in bindings:
            row = [eval_expression(expr, binding, params) for expr, _ in items]
            rows.append(row)
            merged = dict(binding)
            for (expr, alias), value in zip(items, row):
                merged[alias] = value
            row_bindings.append(merged)
    else:
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        order: List[Any] = []
        for binding in bindings:
            key = tuple(
                _group_key(eval_expression(expr, binding, params))
                for (expr, _), is_agg in zip(items, aggregates)
                if not is_agg
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(binding)
        if not bindings:
            # Aggregating an empty result still produces one row (count = 0).
            order, groups = [()], {(): []}
        for key in order:
            group = groups[key]
            sample = group[0] if group else {}
            row = []
            for (expr, _), is_agg in zip(items, aggregates):
                if is_agg:
                    row.append(_eval_aggregate(expr, group, params))
                else:
                    row.append(eval_expression(expr, sample, params) if group else None)
            rows.append(row)
            merged = dict(sample)
            for (expr, alias), value in zip(items, row):
                merged[alias] = value
            row_bindings.append(merged)

    if distinct:
        seen: List[Any] = []
        kept_rows: List[List[Any]] = []
        kept_bindings: List[Dict[str, Any]] = []
        for row, binding in zip(rows, row_bindings):
            key = tuple(_group_key(v) for v in row)
            if key in seen:
                continue
            seen.append(key)
            kept_rows.append(row)
            kept_bindings.append(binding)
        rows, row_bindings = kept_rows, kept_bindings

    return columns, rows, row_bindings


# ======================================================================
#  Clause execution
# ======================================================================
class _Executor:
    """Runs the clause list of a single Cypher statement against the graph."""

    def __init__(self, graph: InMemoryGraph, params: Dict[str, Any]) -> None:
        self.graph = graph
        self.params = params
        self.bindings: List[Dict[str, Any]] = [{}]
        self.created_flags: List[bool] = [False]
        self.summary: Dict[str, int] = {
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_set": 0,
            "nodes_deleted": 0,
            "relationships_deleted": 0,
        }
        self.columns: List[str] = []
        self.rows: List[List[Any]] = []
        self.row_bindings: List[Dict[str, Any]] = []
        self.returned = False

    # -- helpers ------------------------------------------------------
    def _clean(self, binding: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in binding.items() if not k.startswith("__")}

    # -- clauses ------------------------------------------------------
    def do_match(self, body: str, optional: bool) -> None:
        patterns = split_top_level(body)
        if not patterns:
            raise CypherError("MATCH needs a pattern such as (s:Student).")
        current = self.bindings
        for pattern in patterns:
            if optional:
                # OPTIONAL MATCH keeps every incoming row: rows with no match
                # simply carry null for the new variables.
                kept: List[Dict[str, Any]] = []
                for binding in current:
                    found = match_pattern(self.graph, pattern, [binding], self.params)
                    kept.extend(found if found else [dict(binding)])
                current = kept
            else:
                current = match_pattern(self.graph, pattern, current, self.params)
        self.bindings = current
        self.created_flags = [False] * len(current)

    def do_where(self, body: str) -> None:
        if not body.strip():
            raise CypherError("WHERE needs a condition, for example WHERE s.semester = 4.")
        kept: List[Dict[str, Any]] = []
        for binding in self.bindings:
            if bool(eval_expression(body, binding, self.params)):
                kept.append(binding)
        self.bindings = kept
        self.created_flags = [False] * len(kept)

    def do_unwind(self, body: str) -> None:
        expr, alias = _split_alias(body)
        if not alias:
            raise CypherError("UNWIND needs an alias, for example UNWIND [1,2,3] AS x.")
        out: List[Dict[str, Any]] = []
        for binding in self.bindings:
            value = eval_expression(expr, binding, self.params)
            if not isinstance(value, (list, tuple)):
                value = [value]
            for item in value:
                new_binding = dict(binding)
                new_binding[alias] = item
                out.append(new_binding)
        self.bindings = out
        self.created_flags = [False] * len(out)

    def _build_pattern(
        self, pattern: str, binding: Dict[str, Any], merge: bool
    ) -> Tuple[Dict[str, Any], bool]:
        """Create (or MERGE) every element of a pattern; returns (binding, created)."""
        elements = parse_path(pattern)
        binding = dict(binding)
        any_created = False
        previous_node: Optional[SimNode] = None

        idx = 0
        while idx < len(elements):
            node_pat: NodePattern = elements[idx]
            props = _props_dict(node_pat.props, binding, self.params)
            if node_pat.var and node_pat.var in binding:
                node = binding[node_pat.var]
                if not isinstance(node, SimNode):
                    raise CypherError(
                        "'%s' is already bound to something that is not a node." % node_pat.var
                    )
            else:
                if not node_pat.labels:
                    raise CypherError(
                        "A new node needs a label, for example (s:Student {student_id: 'S001'})."
                    )
                if merge:
                    node, created = self.graph.merge_node(node_pat.labels, props)
                else:
                    node, created = self.graph.create_node(node_pat.labels, props), True
                if created:
                    self.summary["nodes_created"] += 1
                    any_created = True
                if node_pat.var:
                    binding[node_pat.var] = node

            if previous_node is not None:
                rel_pat: RelPattern = elements[idx - 1]
                if rel_pat.direction == "both":
                    raise CypherError(
                        "A relationship being created must have a direction, "
                        "for example -[:ENROLLED_IN]->."
                    )
                if len(rel_pat.types) != 1:
                    raise CypherError(
                        "A relationship being created needs exactly one type, "
                        "for example -[:ENROLLED_IN]->."
                    )
                if rel_pat.var_length:
                    raise CypherError("Variable-length patterns cannot be created.")
                rel_props = _props_dict(rel_pat.props, binding, self.params)
                start, end = previous_node, node
                if rel_pat.direction == "in":
                    start, end = node, previous_node
                if merge:
                    rel, created = self.graph.merge_rel(
                        start.nid, end.nid, rel_pat.types[0], rel_props
                    )
                else:
                    rel = self.graph.create_rel(
                        start.nid, end.nid, rel_pat.types[0], rel_props
                    )
                    created = True
                if created:
                    self.summary["relationships_created"] += 1
                    any_created = True
                if rel_pat.var:
                    binding[rel_pat.var] = rel

            previous_node = node
            idx += 2

        return binding, any_created

    def do_create(self, body: str) -> None:
        patterns = split_top_level(body)
        if not patterns:
            raise CypherError("CREATE needs a pattern such as (s:Student {name: 'Aditi'}).")
        out: List[Dict[str, Any]] = []
        for binding in self.bindings:
            current = binding
            for pattern in patterns:
                current, _ = self._build_pattern(pattern, current, merge=False)
            out.append(current)
        self.bindings = out
        self.created_flags = [True] * len(out)

    def do_merge(self, body: str) -> None:
        patterns = split_top_level(body)
        if len(patterns) != 1:
            raise CypherError("MERGE takes exactly one pattern at a time.")
        pattern = patterns[0]
        out: List[Dict[str, Any]] = []
        flags: List[bool] = []
        for binding in self.bindings:
            matched = match_pattern(self.graph, pattern, [binding], self.params)
            if matched:
                out.extend(matched)
                flags.extend([False] * len(matched))
            else:
                new_binding, _ = self._build_pattern(pattern, binding, merge=True)
                out.append(new_binding)
                flags.append(True)
        self.bindings = out
        self.created_flags = flags

    def do_set(self, body: str, only_created: Optional[bool] = None) -> None:
        items = split_top_level(body)
        if not items:
            raise CypherError("SET needs an assignment, for example SET s.name = 'Aditi'.")
        for binding, created in zip(self.bindings, self.created_flags):
            if only_created is True and not created:
                continue
            if only_created is False and created:
                continue
            for item in items:
                self._apply_set(item, binding)

    def _apply_set(self, item: str, binding: Dict[str, Any]) -> None:
        if "+=" in item:
            target_text, _, value_text = item.partition("+=")
            merge_map = True
        elif "=" in item:
            target_text, _, value_text = item.partition("=")
            merge_map = False
        else:
            raise CypherError("Could not understand the SET item %r." % item.strip())

        target_text = target_text.strip()
        value = eval_expression(value_text, binding, self.params)

        if "." in target_text:
            var, _, prop = target_text.partition(".")
            entity = binding.get(var.strip())
            if not isinstance(entity, (SimNode, SimRel)):
                raise CypherError(
                    "SET could not find the node or relationship '%s'." % var.strip()
                )
            entity.properties[prop.strip()] = value
            self.summary["properties_set"] += 1
            return

        entity = binding.get(target_text)
        if not isinstance(entity, (SimNode, SimRel)):
            raise CypherError("SET could not find the node or relationship '%s'." % target_text)
        if not isinstance(value, dict):
            raise CypherError("SET %s = ... expects a property map such as {name: 'Aditi'}." % target_text)
        if not merge_map:
            entity.properties.clear()
        entity.properties.update(value)
        self.summary["properties_set"] += len(value)

    def do_delete(self, body: str, detach: bool) -> None:
        items = split_top_level(body)
        if not items:
            raise CypherError("DELETE needs a variable, for example DELETE n.")
        deleted_nodes: set = set()
        deleted_rels: set = set()
        for binding in self.bindings:
            for item in items:
                value = eval_expression(item, binding, self.params)
                targets = value if isinstance(value, list) else [value]
                for target in targets:
                    if isinstance(target, SimRel):
                        if target.rid in self.graph.rels:
                            self.graph.delete_rel(target.rid)
                        deleted_rels.add(target.rid)
                    elif isinstance(target, SimNode):
                        if target.nid not in self.graph.nodes:
                            continue
                        attached = [
                            r.rid
                            for r in self.graph.rels.values()
                            if r.start == target.nid or r.end == target.nid
                        ]
                        # Raises a friendly error when DELETE is used instead
                        # of DETACH DELETE on a connected node.
                        self.graph.delete_node(target.nid, detach=detach)
                        deleted_nodes.add(target.nid)
                        deleted_rels.update(attached)
        self.summary["nodes_deleted"] += len(deleted_nodes)
        self.summary["relationships_deleted"] += len(deleted_rels)

    def do_with(self, body: str) -> None:
        columns, rows, row_bindings = project(body, self.bindings, self.params)
        self.bindings = row_bindings
        self.created_flags = [False] * len(row_bindings)

    def do_return(self, body: str) -> None:
        self.columns, self.rows, self.row_bindings = project(
            body, self.bindings, self.params
        )
        self.returned = True

    def do_order_by(self, body: str) -> None:
        items = split_top_level(body)
        if not items:
            raise CypherError("ORDER BY needs an expression, for example ORDER BY s.name.")
        target_rows = self.rows if self.returned else None
        bindings = self.row_bindings if self.returned else self.bindings

        # Sort last key first so earlier keys take precedence.
        indexed = list(range(len(bindings)))
        for item in reversed(items):
            expr = item.strip()
            descending = False
            if re.search(r"\bDESC(ENDING)?$", expr, re.IGNORECASE):
                descending = True
                expr = re.sub(r"\bDESC(ENDING)?$", "", expr, flags=re.IGNORECASE).strip()
            else:
                expr = re.sub(r"\bASC(ENDING)?$", "", expr, flags=re.IGNORECASE).strip()

            def sort_key(index: int, expression: str = expr) -> Tuple[int, str, float]:
                value = eval_expression(expression, bindings[index], self.params)
                if value is None:
                    return (2, "", 0.0)
                if isinstance(value, bool):
                    return (1, "", float(value))
                if isinstance(value, (int, float)):
                    return (0, "", float(value))
                return (1, str(value).lower(), 0.0)

            indexed.sort(key=sort_key, reverse=descending)

        if self.returned and target_rows is not None:
            self.rows = [target_rows[i] for i in indexed]
            self.row_bindings = [bindings[i] for i in indexed]
        else:
            self.bindings = [bindings[i] for i in indexed]

    def _int_arg(self, body: str, clause: str) -> int:
        value = eval_expression(body, {}, self.params)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CypherError("%s expects a whole number, for example %s 10." % (clause, clause))
        return int(value)

    def do_skip(self, body: str) -> None:
        count = self._int_arg(body, "SKIP")
        if self.returned:
            self.rows = self.rows[count:]
            self.row_bindings = self.row_bindings[count:]
        else:
            self.bindings = self.bindings[count:]

    def do_limit(self, body: str) -> None:
        count = self._int_arg(body, "LIMIT")
        if self.returned:
            self.rows = self.rows[:count]
            self.row_bindings = self.row_bindings[:count]
        else:
            self.bindings = self.bindings[:count]

    # -- driver -------------------------------------------------------
    def run(self, clauses: List[Tuple[str, str]]) -> None:
        for keyword, body in clauses:
            if keyword == "MATCH":
                self.do_match(body, optional=False)
            elif keyword == "OPTIONAL MATCH":
                self.do_match(body, optional=True)
            elif keyword == "WHERE":
                self.do_where(body)
            elif keyword == "UNWIND":
                self.do_unwind(body)
            elif keyword == "CREATE":
                self.do_create(body)
            elif keyword == "MERGE":
                self.do_merge(body)
            elif keyword == "ON CREATE SET":
                self.do_set(body, only_created=True)
            elif keyword == "ON MATCH SET":
                self.do_set(body, only_created=False)
            elif keyword == "SET":
                self.do_set(body)
            elif keyword == "DELETE":
                self.do_delete(body, detach=False)
            elif keyword == "DETACH DELETE":
                self.do_delete(body, detach=True)
            elif keyword == "WITH":
                self.do_with(body)
            elif keyword == "RETURN":
                self.do_return(body)
            elif keyword == "ORDER BY":
                self.do_order_by(body)
            elif keyword == "SKIP":
                self.do_skip(body)
            elif keyword == "LIMIT":
                self.do_limit(body)
            else:
                raise CypherError(
                    "The clause %s is not supported by the local simulation engine." % keyword
                )


_SCHEMA_COMMAND_RE = re.compile(
    r"^\s*(CREATE|DROP)\s+(CONSTRAINT|INDEX|TEXT\s+INDEX|RANGE\s+INDEX|POINT\s+INDEX)\b"
    r"|^\s*SHOW\s+(CONSTRAINTS?|INDEXES?)\b",
    re.IGNORECASE,
)


def split_statements(script: str) -> List[str]:
    """Split a Cypher script into individual statements on top-level ';'."""
    cleaned = strip_comments(script)
    return [s.strip() for s in split_top_level(cleaned, ";") if s.strip()]


def execute_cypher(
    graph: InMemoryGraph, query: str, parameters: Optional[Dict[str, Any]] = None
) -> QueryResult:
    """Execute one Cypher statement against the in-memory graph."""
    started = time.perf_counter()
    result = QueryResult(mode=SIMULATION_MODE, query=query)
    text = strip_comments(query or "").strip().rstrip(";").strip()
    if not text:
        result.error = "The query is empty. Type a Cypher statement such as MATCH (n) RETURN n."
        return result

    # Constraints and indexes are a real-database feature. Rather than failing
    # with a parse error, say plainly what the local engine does instead.
    if _SCHEMA_COMMAND_RE.match(text):
        result.notice = (
            "Constraints and indexes are enforced by a real Neo4j server. The local "
            "simulation engine skipped this statement and relies on MERGE to keep "
            "entities unique. Connect to Neo4j to run it for real."
        )
        result.summary = {"schema_commands_skipped": 1}
        result.execution_ms = (time.perf_counter() - started) * 1000
        return result

    try:
        executor = _Executor(graph, parameters or {})
        executor.run(split_clauses(text))
        result.columns = executor.columns
        result.rows = executor.rows
        result.summary = {k: v for k, v in executor.summary.items() if v}
    except CypherError as exc:
        result.error = str(exc)
    except RecursionError:
        result.error = (
            "The pattern is too deep for the local simulation engine. "
            "Try a smaller variable-length range such as *1..3."
        )
    except Exception as exc:  # pragma: no cover - defensive, keeps the UI alive
        result.error = "%s: %s" % (type(exc).__name__, exc)
    result.execution_ms = (time.perf_counter() - started) * 1000
    return result


def execute_script(
    graph: InMemoryGraph, script: str, parameters: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, int], List[str]]:
    """Run a multi-statement Cypher script; returns (totals, error messages)."""
    totals: Dict[str, int] = {
        "nodes_created": 0,
        "relationships_created": 0,
        "properties_set": 0,
        "nodes_deleted": 0,
        "relationships_deleted": 0,
        "statements": 0,
    }
    errors: List[str] = []
    for statement in split_statements(script):
        result = execute_cypher(graph, statement, parameters)
        totals["statements"] += 1
        if result.ok:
            for key, value in result.summary.items():
                totals[key] = totals.get(key, 0) + value
        else:
            errors.append("%s  ->  %s" % (statement.splitlines()[0][:70], result.error))
    return totals, errors
