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
    "\u2192": "->", "\u2190": "<-", "\u2194": "<->", "\u21d2": "=>",
    "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
    "\u2713": "[OK]", "\u2714": "[OK]", "\u2717": "[X]", "\u2718": "[X]",
    "\u00d7": "x", "\u2265": ">=", "\u2264": "<=", "\u2260": "!=",
    "\u00a0": " ", "\u2212": "-",
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
            "Knowledge Graph Schema Design, Data Import and Graph Analysis",
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
    "Design appropriate node labels.",
    "Define meaningful relationship types.",
    "Assign properties to nodes and relationships.",
    "Construct a domain-specific graph schema.",
    "Validate a graph schema before importing data into it.",
    "Import structured entity-relationship data into a graph.",
    "Query and analyze a knowledge graph.",
    "Interpret the results of graph queries.",
]

PROCEDURE_STEPS = [
    "Study the theory of knowledge graphs and graph databases.",
    "Identify entities from the provided dataset.",
    "Identify relationships between entities.",
    "Define node labels.",
    "Define node properties.",
    "Define relationship types.",
    "Design the graph schema.",
    "Review the schema using the visual schema designer.",
    "Validate the schema and correct any reported problems.",
    "Import the structured data into the graph.",
    "Execute graph queries.",
    "Analyze the graph visualization.",
    "Record observations.",
    "Complete the quiz.",
    "Generate the PDF lab report.",
]

THEORY_SUMMARY = (
    "A knowledge graph stores information as entities (nodes) joined by named, "
    "directed relationships, with properties held on both. This is the property "
    "graph model: a node carries one or more labels that classify it (for example "
    "Student), a relationship carries exactly one type that names the connection "
    "(for example ENROLLED_IN), and both carry key-value properties. Because each "
    "node stores its own relationships, following a connection is a local step "
    "rather than a join across tables, which is why this model suits highly "
    "connected data.\n\n"
    "Schema design for a knowledge graph means choosing the node labels that "
    "represent real entities, choosing relationship types that read as verbs "
    "between them, deciding which facts are properties of a node and which belong "
    "to a relationship, and giving every entity a unique identifier so that "
    "repeated imports do not create duplicates. Structured data maps directly onto "
    "this model: a table becomes a label, a row becomes a node, a column becomes a "
    "property, and a foreign key or junction table becomes a relationship. The "
    "import runs in two passes, creating the entities first and then the "
    "relationships between them, and each entity is matched on its unique "
    "identifier so that importing the same data twice updates it instead of "
    "duplicating it. The resulting graph is then analyzed by following "
    "relationships from one entity to another."
)

REFERENCES = [
    "I. Robinson, J. Webber and E. Eifrem, Graph Databases, 2nd edition, "
    "O'Reilly Media, 2015.",
    "A. Hogan, E. Blomqvist, M. Cochez et al., \"Knowledge Graphs\", ACM Computing "
    "Surveys, vol. 54, no. 4, article 71, 2021.",
    "R. Angles and C. Gutierrez, \"Survey of Graph Database Models\", ACM Computing "
    "Surveys, vol. 40, no. 1, 2008.",
    "A. Silberschatz, H. F. Korth and S. Sudarshan, Database System Concepts, 7th "
    "edition, McGraw-Hill, 2019.",
]


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


def _result_statement(context: Dict[str, Any]) -> str:
    """Describe what the student actually completed -- nothing more."""
    schema = context.get("schema", {})
    observations = context.get("observations", {})
    dataset = context.get("dataset_summary", {})
    labels = [n.get("label") for n in schema.get("nodes", []) if n.get("label")]
    rel_types = sorted({r.get("type") for r in schema.get("relationships", []) if r.get("type")})
    nodes = observations.get("node_count", 0)
    rels = observations.get("relationship_count", 0)

    parts = [
        "A knowledge graph schema for the %s domain was designed with %d node "
        "label(s) (%s) and %d relationship type(s) (%s)."
        % (
            context.get("domain", "selected"),
            len(labels),
            ", ".join(labels) or "none",
            len(rel_types),
            ", ".join(rel_types) or "none",
        )
    ]
    if nodes:
        parts.append(
            "Structured entity-relationship data from '%s' was imported into the "
            "simulation graph, producing %d node(s) and %d relationship(s)."
            % (dataset.get("source", "the dataset"), nodes, rels)
        )
    else:
        parts.append("No data was imported into the graph during this session.")
    if observations.get("queries_total"):
        parts.append(
            "%d graph quer(y/ies) were executed: %d succeeded and %d failed, "
            "returning %d record(s) in total."
            % (
                observations.get("queries_total", 0),
                observations.get("queries_ok", 0),
                observations.get("queries_failed", 0),
                observations.get("records_returned", 0),
            )
        )
    else:
        parts.append("No graph queries were executed during this session.")
    return " ".join(parts)


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

    # Student information ----------------------------------------------------
    pdf.key_values(
        [
            ("Student Name", student.get("name") or "-"),
            ("Student ID / Roll No.", student.get("roll") or "-"),
            ("Department", student.get("department") or "-"),
            ("Semester", student.get("semester") or "-"),
            ("Experiment Date", student.get("date") or date.today().isoformat()),
        ]
    )

    # 1. Experiment title ----------------------------------------------------
    pdf.h1("1. Experiment Title")
    pdf.key_values(
        [
            ("Experiment No.", EXPERIMENT_NUMBER.replace("Experiment ", "")),
            ("Title", EXPERIMENT_TITLE),
            ("Domain Modelled", schema.get("domain") or context.get("domain") or "-"),
            ("Execution Mode", context.get("mode", "Local Simulation")),
        ]
    )

    # 2. Aim -----------------------------------------------------------------
    pdf.h1("2. Aim")
    pdf.body(
        "To design a domain-specific knowledge graph schema using nodes, "
        "relationships and properties, to import structured entity-relationship "
        "data into the graph, and to query and analyze the resulting knowledge graph."
    )

    # 3. Objectives ----------------------------------------------------------
    pdf.h1("3. Objectives")
    pdf.body("By the end of the experiment the student should be able to:")
    pdf.bullets(LEARNING_OBJECTIVES, numbered=True)

    # 4. Theory --------------------------------------------------------------
    pdf.h1("4. Theory")
    pdf.body(THEORY_SUMMARY)

    # 5. Procedure -----------------------------------------------------------
    pdf.h1("5. Procedure")
    pdf.bullets(PROCEDURE_STEPS, numbered=True)

    # 6. Schema design -------------------------------------------------------
    pdf.add_page()
    pdf.h1("6. Schema Design")
    node_rows, rel_rows = _schema_tables(schema)
    pdf.h2("6.1 Node labels, unique identifiers and properties")
    pdf.table(["Node label", "Unique ID property", "Properties"], node_rows, [38, 40, 102])
    pdf.h2("6.2 Relationship types")
    pdf.table(
        ["Relationship type", "Source label", "Target label", "Properties"],
        rel_rows,
        [48, 42, 42, 48],
    )
    pdf.h2("6.3 Schema diagram")
    if not pdf.schema_diagram(schema):
        pdf.body("(The schema has no node labels, so no diagram could be drawn.)")

    # 7. Data used -----------------------------------------------------------
    pdf.h1("7. Data Used")
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

    # 8. Graph queries / analysis --------------------------------------------
    pdf.add_page()
    pdf.h1("8. Graph Queries / Analysis")
    if queries:
        rows = [
            [
                index,
                item.get("query", ""),
                "Yes" if item.get("ok") else "No",
                item.get("records", 0),
                "%.1f" % float(item.get("ms", 0.0)),
            ]
            for index, item in enumerate(queries[-15:], start=1)
        ]
        pdf.table(
            ["#", "Graph query", "Succeeded", "Results", "Time (ms)"],
            rows,
            [10, 98, 24, 24, 24],
        )
    else:
        pdf.body("No graph queries were executed during this session.")
    pdf.h2("8.1 Query results summary")
    pdf.key_values(
        [
            ("Queries executed", observations.get("queries_total", 0)),
            ("Successful queries", observations.get("queries_ok", 0)),
            ("Failed queries", observations.get("queries_failed", 0)),
            ("Total results returned", observations.get("records_returned", 0)),
        ]
    )

    # 9. Observations --------------------------------------------------------
    pdf.h1("9. Observations")
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
    pdf.h2("9.1 Recorded experimental trials")
    if trials:
        headers = ["#", "Domain", "Labels", "Rel types", "Nodes", "Rels", "Results", "Status"]
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
        pdf.h2("9.2 Graph query recorded with each trial")
        for trial in trials:
            query = str(trial.get("Graph Query", "")).strip()
            if query:
                pdf.body(
                    "Trial %s (%s): %s"
                    % (trial.get("Trial", "?"), trial.get("Timestamp", ""), query)
                )
    else:
        pdf.body("No trials were recorded in the Experimental Data Logbook.")

    # 10. Results ------------------------------------------------------------
    pdf.h1("10. Results")
    pdf.body(_result_statement(context))

    # 11. Quiz score ---------------------------------------------------------
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
    pdf.h1("12. Conclusion")
    conclusion = (context.get("conclusion") or "").strip()
    pdf.body(conclusion if conclusion else "(No conclusion was written by the student.)")

    # 13. References ---------------------------------------------------------
    pdf.h1("13. References")
    pdf.bullets(REFERENCES, numbered=True)

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*MUTED)
    pdf._line_text("Signature of Student: ______________________        Date: ______________")
    pdf.ln(2)
    pdf._line_text("Signature of Faculty: ______________________        Grade: _____________")
    return pdf


# ======================================================================
#  Certificate of Completion
# ======================================================================
GOLD = (180, 140, 50)


class LabCertificate(FPDF):
    """An official landscape A4 Certificate of Lab Completion."""

    def __init__(self) -> None:
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)
        self.set_margins(12, 12, 12)
        self.set_title("Virtual Laboratory Certificate of Completion")

    def draw_frames(self) -> None:
        """Draw an elegant double border with decorative corner accents."""
        # Outer border
        self.set_draw_color(*ACCENT)
        self.set_line_width(1.4)
        self.rect(8, 8, 281, 194)

        # Inner gold border
        self.set_draw_color(*GOLD)
        self.set_line_width(0.6)
        self.rect(11, 11, 275, 188)

        # Corner geometric accents
        self.set_fill_color(*ACCENT)
        for x in (11, 281):
            for y in (11, 194):
                self.rect(x, y, 5, 5, style="F")

    def _line_text(self, text: str, align: str = "C") -> None:
        if _HAS_ENUMS:
            self.cell(0, 6, _safe(text), align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:  # pragma: no cover
            self.cell(0, 6, _safe(text), ln=1, align=align)


def _render_certificate(context: Dict[str, Any]) -> LabCertificate:
    student = context.get("student", {})
    observations = context.get("observations", {})
    quiz = context.get("quiz")
    cert_id = context.get("cert_id") or "VLAB-KG-9-%s" % (
        (student.get("roll") or "EXP9")[:6].replace(" ", "").upper()
    )

    pdf = LabCertificate()
    pdf.add_page()
    pdf.draw_frames()

    # 1. Masthead
    pdf.set_y(17)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*MUTED)
    pdf._line_text("VIRTUAL LABORATORIES PROJECT · MINISTRY OF EDUCATION")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf._line_text("National Knowledge Graph Systems & Graph Databases Laboratory")

    # 2. Title
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 23)
    pdf.set_text_color(*ACCENT)
    pdf._line_text("CERTIFICATE OF LAB COMPLETION")

    # 3. Awarded to
    pdf.ln(1)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(*MUTED)
    pdf._line_text("This is to proudly certify that")

    # 4. Student Name
    pdf.ln(2)
    student_name = (student.get("name") or "Student").strip().title()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*INK)
    pdf._line_text(student_name)

    # Decorative underline
    name_w = min(160, max(80, pdf.get_string_width(_safe(student_name)) + 20))
    start_x = (297 - name_w) / 2
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.8)
    pdf.line(start_x, pdf.get_y(), start_x + name_w, pdf.get_y())
    pdf.ln(3)

    # 5. Student details
    roll = student.get("roll") or "Enrolled Student"
    dept = student.get("department") or "Computer Science & Engineering"
    sem = student.get("semester") or "Semester IV"
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf._line_text(f"Roll No: {roll}   |   Department: {dept}   |   {sem}")

    # 6. Citation text
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*INK)
    citation = (
        "has successfully demonstrated practical competency in designing a domain-specific "
        "Knowledge Graph schema, validating and importing structured entity-relationship data into "
        "the in-memory simulation engine, executing graph traversal queries, and completing all requirements for:"
    )
    pdf.set_x(30)
    pdf.multi_cell(237, 5.0, _safe(citation), align="C")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*ACCENT)
    pdf._line_text(f"{EXPERIMENT_NUMBER}: {EXPERIMENT_TITLE}")

    # 7. Summary metrics box
    pdf.ln(3)
    box_y = pdf.get_y()
    pdf.set_fill_color(*BOX)
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.4)
    pdf.rect(32, box_y, 233, 17, style="DF")

    domain_name = context.get("domain") or "University"
    nodes_cnt = observations.get("node_count", 0)
    rels_cnt = observations.get("relationship_count", 0)
    quiz_str = (
        f"{quiz['correct']}/{quiz['total']} ({quiz['percentage']:.0f}%)"
        if quiz
        else "Completed"
    )

    pdf.set_y(box_y + 2.5)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*MUTED)
    cols = [
        ("DOMAIN", domain_name),
        ("GRAPH NODES", str(nodes_cnt)),
        ("RELATIONSHIPS", str(rels_cnt)),
        ("ASSESSMENT SCORE", quiz_str),
    ]
    col_w = 233 / len(cols)
    for i, (label, val) in enumerate(cols):
        cx = 32 + i * col_w
        pdf.set_xy(cx, box_y + 2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(col_w, 4.5, _safe(label), align="C")
        pdf.set_xy(cx, box_y + 7.5)
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*ACCENT)
        pdf.cell(col_w, 6, _safe(val), align="C")

    # 8. Signatures and verification footer
    footer_y = 158

    # Left: Evaluation Engine
    pdf.set_xy(32, footer_y)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(*INK)
    pdf.cell(65, 5, _safe("Automated Simulation Engine"), align="C")
    pdf.set_xy(32, footer_y + 5)
    pdf.set_draw_color(*RULE)
    pdf.line(35, footer_y + 5, 95, footer_y + 5)
    pdf.set_xy(32, footer_y + 6)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(65, 4.5, _safe("Virtual Lab Evaluator"), align="C")

    # Center: Ornate Seal Badge
    seal_x, seal_y = 148.5, footer_y + 4
    pdf.set_draw_color(*GOLD)
    pdf.set_fill_color(253, 250, 242)
    pdf.set_line_width(0.8)
    pdf.ellipse(seal_x - 12, seal_y - 12, 24, 24, style="DF")
    pdf.set_draw_color(*GOLD)
    pdf.set_line_width(0.3)
    pdf.ellipse(seal_x - 10, seal_y - 10, 20, 20, style="D")
    pdf.set_xy(seal_x - 11, seal_y - 4)
    pdf.set_font("Helvetica", "B", 6.5)
    pdf.set_text_color(*GOLD)
    pdf.cell(22, 3.5, _safe("VERIFIED"), align="C")
    pdf.set_xy(seal_x - 11, seal_y)
    pdf.set_font("Helvetica", "B", 6)
    pdf.cell(22, 3.5, _safe("VLAB EVAL"), align="C")

    # Right: Course Coordinator
    pdf.set_xy(200, footer_y)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(*INK)
    pdf.cell(65, 5, _safe("Course Faculty Coordinator"), align="C")
    pdf.set_xy(200, footer_y + 5)
    pdf.set_draw_color(*RULE)
    pdf.line(202, footer_y + 5, 262, footer_y + 5)
    pdf.set_xy(200, footer_y + 6)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(65, 4.5, _safe("Department of CSE"), align="C")

    # Bottom line: Cert ID and Date
    issue_date = student.get("date") or date.today().isoformat()
    pdf.set_xy(15, 185)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*MUTED)
    cert_text = f"Certificate ID: {cert_id}   ·   Issued On: {issue_date}   ·   Verify at: virtual-labs.ac.in"
    pdf.cell(267, 5, _safe(cert_text), align="C")

    return pdf


def build_certificate(context: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """Render the Certificate of Lab Completion. Returns (pdf_bytes, error_message)."""
    try:
        pdf = _render_certificate(context)
        output = pdf.output()
        return bytes(output), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, "%s: %s" % (type(exc).__name__, exc)

