"""
report_generator.py
===================
Builds the downloadable PDF laboratory report for Experiment 9.

Uses fpdf2 with the built-in Helvetica font, so no font files have to ship with
the project.  That font is Latin-1 only, so every string passes through
:func:`_safe` first -- a report must never fail because a student typed an
em dash.

The schema diagram is drawn directly with fpdf2's vector primitives (no image
export dependency), using the same layout the on-screen Plotly figure uses.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fpdf import FPDF

try:  # positioning enums moved around between fpdf2 releases
    from fpdf.enums import XPos, YPos

    _HAS_ENUMS = True
except Exception:  # pragma: no cover
    _HAS_ENUMS = False

EXPERIMENT_TITLE = "Design a Knowledge Graph Schema and Import Data"
EXPERIMENT_NUMBER = "Experiment 9"

# Replacements applied before the Latin-1 encode.
_REPLACEMENTS = {
    "→": "->", "←": "<-", "↔": "<->", "⇒": "=>",
    "–": "-", "—": "-", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "•": "-",
    "✓": "[OK]", "✔": "[OK]", "✗": "[X]", "✘": "[X]",
    "×": "x", "≥": ">=", "≤": "<=", "≠": "!=",
    " ": " ", "−": "-",
}

# Colours (RGB) used by the report.
# Matched to the application's white + navy identity (.streamlit/config.toml).
INK = (16, 35, 63)
MUTED = (94, 108, 130)
ACCENT = (27, 54, 93)
RULE = (216, 224, 236)
BOX = (243, 246, 251)
PALETTE = [
    (27, 54, 93), (235, 104, 52), (27, 175, 122), (237, 161, 0),
    (232, 123, 164), (0, 131, 0), (74, 58, 167), (227, 73, 72),
]


def _safe(text: Any) -> str:
    """Make any value printable with the Latin-1 core fonts."""
    if text is None:
        return ""
    value = str(text)
    for source, target in _REPLACEMENTS.items():
        value = value.replace(source, target)
    return value.encode("latin-1", "replace").decode("latin-1")


class LabReport(FPDF):
    """An A4 report with a running header and footer."""

    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(15, 15, 15)
        self.set_title("Virtual Laboratory Report")

    # -- chrome -------------------------------------------------------
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self._line_text("Virtual Laboratory Report - %s" % EXPERIMENT_TITLE, align="R")
        self.set_draw_color(*RULE)
        self.line(15, 20, 195, 20)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self._line_text("Page %d" % self.page_no(), align="C")

    def _line_text(self, text: str, align: str = "L") -> None:
        if _HAS_ENUMS:
            self.cell(0, 6, _safe(text), align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:  # pragma: no cover
            self.cell(0, 6, _safe(text), ln=1, align=align)

    # -- building blocks ----------------------------------------------
    def title_block(self) -> None:
        self.set_fill_color(*BOX)
        self.rect(15, 15, 180, 34, style="F")
        self.set_xy(15, 20)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*INK)
        self._line_text("Virtual Laboratory Report", align="C")
        self.set_x(15)
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*ACCENT)
        self._line_text("%s: %s" % (EXPERIMENT_NUMBER, EXPERIMENT_TITLE), align="C")
        self.set_x(15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*MUTED)
        self._line_text(
            "Knowledge Graph Schema Design, Data Import and Cypher Querying with Neo4j",
            align="C",
        )
        self.set_y(56)

    def h1(self, text: str) -> None:
        if self.get_y() > 250:
            self.add_page()
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*ACCENT)
        self._line_text(text)
        self.set_draw_color(*RULE)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(2)
        self.set_text_color(*INK)

    def h2(self, text: str) -> None:
        self.ln(1)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*INK)
        self._line_text(text)

    def body(self, text: str, size: int = 9.5) -> None:
        self.set_font("Helvetica", "", size)
        self.set_text_color(*INK)
        self.multi_cell(0, 4.8, _safe(text))
        self.ln(1)

    def bullets(self, items: Sequence[str], numbered: bool = False) -> None:
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*INK)
        for index, item in enumerate(items, start=1):
            marker = "%d." % index if numbered else "-"
            self.set_x(18)
            self.multi_cell(0, 4.8, _safe("%s %s" % (marker, item)))
        self.ln(1)

    def key_values(self, pairs: Sequence[Tuple[str, Any]]) -> None:
        self.set_font("Helvetica", "", 9.5)
        for key, value in pairs:
            self.set_x(15)
            self.set_font("Helvetica", "B", 9.5)
            self.cell(52, 5.5, _safe(key), border=0)
            self.set_font("Helvetica", "", 9.5)
            if _HAS_ENUMS:
                self.cell(0, 5.5, _safe(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:  # pragma: no cover
                self.cell(0, 5.5, _safe(value), ln=1)
        self.ln(1)

    def table(
        self, headers: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[float]
    ) -> None:
        """A simple fixed-width table that wraps long cells."""
        if not rows:
            self.body("(nothing recorded)")
            return
        line_height = 5.0
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*BOX)
        self.set_text_color(*INK)
        self.set_x(15)
        for header, width in zip(headers, widths):
            self.cell(width, line_height + 1, _safe(header), border=1, align="C", fill=True)
        self.ln(line_height + 1)

        self.set_font("Helvetica", "", 8)
        for row in rows:
            # Work out how many lines the tallest cell needs.
            heights = []
            for value, width in zip(row, widths):
                text = _safe(value)
                lines = max(1, math.ceil(self.get_string_width(text) / max(width - 3, 4)))
                heights.append(lines)
            height = max(heights) * line_height
            if self.get_y() + height > self.h - 20:
                self.add_page()
                self.set_font("Helvetica", "B", 8.5)
                self.set_x(15)
                for header, width in zip(headers, widths):
                    self.cell(width, line_height + 1, _safe(header), border=1, align="C", fill=True)
                self.ln(line_height + 1)
                self.set_font("Helvetica", "", 8)
            start_x, start_y = 15, self.get_y()
            x = start_x
            for value, width in zip(row, widths):
                self.rect(x, start_y, width, height)
                self.set_xy(x + 1.5, start_y + 0.8)
                self.multi_cell(width - 3, line_height - 0.6, _safe(value), border=0)
                x += width
            self.set_xy(start_x, start_y + height)
        self.ln(2)

    def code_block(self, code: str, max_lines: int = 18) -> None:
        lines = _safe(code).splitlines()
        clipped = lines[:max_lines]
        height = len(clipped) * 4.0 + 3
        if self.get_y() + height > self.h - 22:
            self.add_page()
        self.set_fill_color(245, 245, 242)
        self.set_draw_color(*RULE)
        self.rect(15, self.get_y(), 180, height, style="DF")
        self.set_font("Courier", "", 8)
        self.set_text_color(*INK)
        y = self.get_y() + 1.5
        for line in clipped:
            self.set_xy(17, y)
            self.cell(176, 4, line[:105])
            y += 4
        self.set_y(self.get_y() + height + 1)
        if len(lines) > max_lines:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*MUTED)
            self._line_text("... %d more line(s) omitted" % (len(lines) - max_lines))
            self.set_text_color(*INK)

    # -- schema diagram -----------------------------------------------
    def schema_diagram(self, schema: Dict[str, Any]) -> bool:
        """Draw the designed schema with vector primitives. True if drawn."""
        try:
            from graph_viz import schema_layout
        except Exception:  # pragma: no cover
            return False

        nodes, positions, edges, rels = schema_layout(schema)
        if not nodes:
            return False

        box_height = 92.0
        if self.get_y() + box_height > self.h - 22:
            self.add_page()
        top = self.get_y() + 2
        centre_x, centre_y = 105.0, top + box_height / 2
        scale = min(72.0, box_height / 2 - 12)

        def place(index: int) -> Tuple[float, float]:
            x, y = positions[index]
            return centre_x + float(x) * scale, centre_y + float(y) * (scale * 0.52)

        # Relationships first, so the boxes sit on top of the lines.
        self.set_draw_color(150, 150, 150)
        self.set_line_width(0.3)
        drawn: Dict[Tuple[int, int], int] = {}
        for (start, end), rel in zip(edges, rels):
            if start == end:
                continue
            x0, y0 = place(start)
            x1, y1 = place(end)
            pair = (min(start, end), max(start, end))
            rank = drawn.get(pair, 0)
            drawn[pair] = rank + 1
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            offset = 0.0 if rank == 0 else 4.5 * rank * (1 if rank % 2 else -1)
            ox, oy = -uy * offset, ux * offset
            sx, sy = x0 + ux * 11 + ox, y0 + uy * 6 + oy
            ex, ey = x1 - ux * 11 + ox, y1 - uy * 6 + oy
            self.line(sx, sy, ex, ey)
            # Arrow head: two short strokes.
            angle = math.atan2(ey - sy, ex - sx)
            for sign in (1, -1):
                a = angle + sign * 0.45
                self.line(ex, ey, ex - 2.4 * math.cos(a), ey - 2.4 * math.sin(a))
            self.set_font("Helvetica", "", 6.5)
            self.set_text_color(*MUTED)
            label = _safe(rel.get("type", ""))
            width = self.get_string_width(label) + 2
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            self.set_fill_color(255, 255, 255)
            self.rect(mx - width / 2, my - 2, width, 4, style="F")
            self.set_xy(mx - width / 2, my - 2)
            self.cell(width, 4, label, align="C")

        # Node label boxes.
        for index, node in enumerate(nodes):
            x, y = place(index)
            colour = PALETTE[index % len(PALETTE)]
            label = _safe(node.get("label", ""))
            self.set_font("Helvetica", "B", 8)
            width = max(self.get_string_width(label) + 6, 20)
            self.set_fill_color(*colour)
            self.set_draw_color(255, 255, 255)
            self.set_line_width(0.5)
            self.rect(x - width / 2, y - 4.5, width, 9, style="DF")
            self.set_text_color(255, 255, 255)
            self.set_xy(x - width / 2, y - 4.5)
            self.cell(width, 9, label, align="C")

        self.set_line_width(0.2)
        self.set_text_color(*INK)
        self.set_y(top + box_height)
        return True


# ======================================================================
#  Report assembly
# ======================================================================
LEARNING_OBJECTIVES = [
    "Explain the concept of a knowledge graph.",
    "Identify entities and relationships from structured data.",
    "Design appropriate Neo4j node labels.",
    "Define meaningful relationship types.",
    "Assign properties to nodes and relationships.",
    "Construct a domain-specific graph schema.",
    "Import structured data into Neo4j.",
    "Write basic Cypher queries.",
    "Traverse relationships in a graph.",
    "Analyze the resulting knowledge graph.",
]

PROCEDURE_STEPS = [
    "Study the theory of knowledge graphs and Neo4j.",
    "Identify entities from the provided dataset.",
    "Identify relationships between entities.",
    "Define node labels.",
    "Define node properties.",
    "Define relationship types.",
    "Design the graph schema.",
    "Review the schema using the visual schema designer.",
    "Import the structured data.",
    "Execute Cypher queries.",
    "Analyze the graph visualization.",
    "Record observations.",
    "Complete the quiz.",
    "Generate the PDF lab report.",
]

THEORY_SUMMARY = (
    "A knowledge graph stores information as entities (nodes) joined by named, "
    "directed relationships, with properties held on both. Neo4j implements the "
    "property graph model: a node carries one or more labels that classify it "
    "(for example :Student), a relationship carries exactly one type that names "
    "the connection (for example ENROLLED_IN), and both carry key-value "
    "properties. Because each node stores its own relationships, traversing a "
    "connection is a local step rather than a join across tables, which is why "
    "graph databases suit highly connected data.\n\n"
    "Schema design for a knowledge graph means choosing the node labels that "
    "represent real entities, choosing relationship types that read as verbs "
    "between them, deciding which facts are properties of a node and which "
    "belong to a relationship, and giving every entity a unique identifier so "
    "that repeated imports do not create duplicates. Data is loaded with Cypher: "
    "CREATE always inserts, MERGE matches first and only creates when nothing "
    "matched, MATCH finds existing patterns, SET updates properties, and "
    "DETACH DELETE removes a node together with its relationships. LOAD CSV "
    "streams rows from a structured file so that each row can be turned into "
    "nodes and relationships by those same clauses."
)


def _schema_tables(schema: Dict[str, Any]) -> Tuple[List[List[str]], List[List[str]]]:
    node_rows = [
        [
            node.get("label", ""),
            node.get("key", ""),
            ", ".join(node.get("properties", [])),
        ]
        for node in schema.get("nodes", [])
    ]
    rel_rows = [
        [
            rel.get("type", ""),
            rel.get("source", ""),
            rel.get("target", ""),
            ", ".join(rel.get("properties", [])) or "-",
        ]
        for rel in schema.get("relationships", [])
    ]
    return node_rows, rel_rows


def build_report(context: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """Render the report. Returns ``(pdf_bytes, error_message)``.

    Any failure is returned as a message so the Streamlit page can show it
    instead of crashing.
    """
    try:
        pdf = _render(context)
        output = pdf.output()
        return bytes(output), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, "%s: %s" % (type(exc).__name__, exc)


def _render(context: Dict[str, Any]) -> LabReport:
    student = context.get("student", {})
    schema = context.get("schema", {})
    observations = context.get("observations", {})
    quiz = context.get("quiz")
    trials = context.get("trials", [])
    queries = context.get("queries", [])
    dataset = context.get("dataset_summary", {})

    pdf = LabReport()
    pdf.add_page()
    pdf.title_block()

    # 1. Student information -------------------------------------------------
    pdf.h1("1. Student Information")
    pdf.key_values(
        [
            ("Student Name", student.get("name") or "-"),
            ("Student ID / Roll No.", student.get("roll") or "-"),
            ("Department", student.get("department") or "-"),
            ("Semester", student.get("semester") or "-"),
            ("Experiment Date", student.get("date") or date.today().isoformat()),
            ("Experiment", "%s - %s" % (EXPERIMENT_NUMBER, EXPERIMENT_TITLE)),
            ("Domain Modelled", schema.get("domain") or context.get("domain") or "-"),
            ("Execution Mode", context.get("mode", "Local Simulation")),
        ]
    )

    # 2. Aim and objectives --------------------------------------------------
    pdf.h1("2. Aim and Learning Objectives")
    pdf.body(
        "Aim: to design a domain-specific knowledge graph schema, import structured "
        "entity-relationship data into it, and query the resulting graph using Cypher."
    )
    pdf.h2("By the end of the experiment the student should be able to:")
    pdf.bullets(LEARNING_OBJECTIVES, numbered=True)

    # 3. Theory --------------------------------------------------------------
    pdf.h1("3. Theory Summary")
    pdf.body(THEORY_SUMMARY)

    # 4. Procedure -----------------------------------------------------------
    pdf.h1("4. Experimental Procedure")
    pdf.bullets(PROCEDURE_STEPS, numbered=True)

    # 5. Designed schema -----------------------------------------------------
    pdf.add_page()
    pdf.h1("5. Designed Knowledge Graph Schema")
    node_rows, rel_rows = _schema_tables(schema)
    pdf.h2("5.1 Node labels, unique identifiers and properties")
    pdf.table(
        ["Node label", "Unique ID property", "Properties"],
        node_rows,
        [38, 40, 102],
    )
    pdf.h2("5.2 Relationship types")
    pdf.table(
        ["Relationship type", "Source label", "Target label", "Properties"],
        rel_rows,
        [48, 42, 42, 48],
    )
    pdf.h2("5.3 Schema diagram")
    if not pdf.schema_diagram(schema):
        pdf.body("(The schema has no node labels, so no diagram could be drawn.)")

    # 6. Dataset -------------------------------------------------------------
    pdf.h1("6. Imported Dataset Summary")
    if dataset.get("node_total"):
        pdf.key_values(
            [
                ("Data source", dataset.get("source", "-")),
                ("Nodes in dataset", dataset.get("node_total", 0)),
                ("Relationships in dataset", dataset.get("edge_total", 0)),
                ("Import status", context.get("import_status", "Not imported")),
            ]
        )
        pdf.table(
            ["Node label", "Count"],
            [[k, v] for k, v in dataset.get("label_counts", {}).items()],
            [90, 90],
        )
        pdf.table(
            ["Relationship type", "Count"],
            [[k, v] for k, v in dataset.get("type_counts", {}).items()],
            [90, 90],
        )
    else:
        pdf.body("No dataset was loaded during this session.")

    # 7. Trials --------------------------------------------------------------
    pdf.add_page()
    pdf.h1("7. Recorded Experimental Trials")
    if trials:
        headers = ["#", "Domain", "Labels", "Rel types", "Nodes", "Rels", "Records", "Status"]
        rows = [
            [
                trial.get("Trial", ""),
                trial.get("Domain", ""),
                trial.get("Node Types", ""),
                trial.get("Relationship Types", ""),
                trial.get("Nodes", ""),
                trial.get("Relationships", ""),
                trial.get("Query Result Count", ""),
                trial.get("Import Status", ""),
            ]
            for trial in trials
        ]
        pdf.table(headers, rows, [10, 32, 18, 22, 18, 18, 22, 40])
        pdf.h2("Cypher query recorded with each trial")
        for trial in trials:
            query = str(trial.get("Cypher Query", "")).strip()
            if query:
                pdf.body("Trial %s (%s):" % (trial.get("Trial", "?"), trial.get("Timestamp", "")))
                pdf.code_block(query, max_lines=6)
    else:
        pdf.body("No trials were recorded in the Experimental Data Logbook.")

    # 8. Queries -------------------------------------------------------------
    pdf.h1("8. Cypher Queries Executed")
    if queries:
        for index, item in enumerate(queries[-10:], start=1):
            status = "SUCCESS" if item.get("ok") else "FAILED"
            pdf.h2(
                "Query %d - %s (%s, %d record(s), %.1f ms)"
                % (
                    index,
                    status,
                    item.get("mode", "-"),
                    item.get("records", 0),
                    float(item.get("ms", 0.0)),
                )
            )
            pdf.code_block(item.get("query", ""), max_lines=8)
            if not item.get("ok") and item.get("error"):
                pdf.body("Error reported: %s" % item["error"])
    else:
        pdf.body("No Cypher queries were executed from the query playground.")

    pdf.h1("9. Query Results Summary")
    pdf.key_values(
        [
            ("Queries executed", observations.get("queries_total", 0)),
            ("Successful queries", observations.get("queries_ok", 0)),
            ("Failed queries", observations.get("queries_failed", 0)),
            ("Total records returned", observations.get("records_returned", 0)),
        ]
    )

    # 10. Observations -------------------------------------------------------
    pdf.h1("10. Observations")
    pdf.key_values(
        [
            ("Node labels in graph", observations.get("label_count", 0)),
            ("Relationship types in graph", observations.get("relationship_type_count", 0)),
            ("Nodes in graph", observations.get("node_count", 0)),
            ("Relationships in graph", observations.get("relationship_count", 0)),
            ("Average relationships per node", observations.get("avg_rels_per_node", 0)),
        ]
    )
    notes = observations.get("notes", [])
    if notes:
        pdf.bullets(notes)

    # 11. Quiz ---------------------------------------------------------------
    pdf.h1("11. Quiz Score")
    if quiz:
        pdf.key_values(
            [
                ("Score", "%d / %d" % (quiz.get("correct", 0), quiz.get("total", 0))),
                ("Percentage", "%.1f %%" % float(quiz.get("percentage", 0.0))),
                ("Result", quiz.get("verdict", "-")),
            ]
        )
        by_level = quiz.get("by_level", {})
        if by_level:
            pdf.table(
                ["Difficulty", "Correct", "Total"],
                [[level, data["correct"], data["total"]] for level, data in by_level.items()],
                [60, 60, 60],
            )
    else:
        pdf.body("The quiz was not attempted in this session.")

    # 12. Conclusion ---------------------------------------------------------
    pdf.h1("12. Student Conclusion")
    conclusion = (context.get("conclusion") or "").strip()
    pdf.body(conclusion if conclusion else "(No conclusion was written by the student.)")

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*MUTED)
    pdf._line_text("Signature of Student: ______________________        Date: ______________")
    pdf.ln(2)
    pdf._line_text("Signature of Faculty: ______________________        Grade: _____________")
    return pdf
