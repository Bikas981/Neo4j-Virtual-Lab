"""
app.py
======
Virtual Lab: Knowledge Graph Schema Design & Neo4j Data Import
(Experiment 9 -- Design a Knowledge Graph Schema and Import Data)

Run with:  streamlit run app.py

The laboratory works with or without a Neo4j server.  When no database is
reachable it switches to a Local Simulation engine that runs the *same*
generated Cypher in memory, so every stage of the experiment stays usable.

Only native Streamlit components are used -- no custom CSS -- so the app looks
correct in both the light and dark Streamlit themes.
"""

from __future__ import annotations

import io
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import domains
import graph_viz
import quiz_bank
import report_generator
import schema_tools
from neo4j_service import DRIVER_AVAILABLE, Neo4jService, env_defaults, env_password_present
from simulation_engine import (
    NEO4J_MODE,
    SIMULATION_MODE,
    InMemoryGraph,
    QueryResult,
    execute_cypher,
)

APP_TITLE = "Virtual Lab: Knowledge Graph Schema Design & Neo4j Data Import"
APP_SUBTITLE = "Design, construct, import, query and analyze a domain-specific knowledge graph"

# Note the separator: a label like "1. Theory" would be parsed as a Markdown
# ordered list and lose its number, so the number is joined with a middot.
SECTIONS = ["1 · Theory", "2 · Simulation", "3 · Quiz", "4 · Report Generation"]

PROGRESS_STEPS = [
    ("theory_done", "Theory"),
    ("schema_done", "Schema Design"),
    ("import_done", "Data Import"),
    ("query_done", "Query Execution"),
    ("quiz_done", "Quiz"),
    ("report_done", "Report"),
]

st.set_page_config(
    page_title="KG Virtual Lab",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ======================================================================
#  Session state
# ======================================================================
def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _tag_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Give every schema row a stable id so widget keys survive re-ordering."""
    for node in schema.get("nodes", []):
        node.setdefault("_id", _new_id())
    for rel in schema.get("relationships", []):
        rel.setdefault("_id", _new_id())
    return schema


def load_domain(domain_name: str) -> None:
    """Load a domain's schema + sample data and reset the working graph."""
    domain = domains.get_domain(domain_name)
    st.session_state.domain = domain["name"]
    st.session_state.schema = _tag_schema(domains.schema_from_domain(domain))
    st.session_state.dataset = domains.dataset_from_domain(domain)
    st.session_state.dataset_report = None
    st.session_state.generated_cypher = ""
    st.session_state.graph = InMemoryGraph()
    st.session_state.import_status = "Not imported"
    st.session_state.import_summary = {}
    st.session_state.import_done = False
    st.session_state.query_text = domain["queries"][0]["cypher"]
    st.session_state.query_nonce = st.session_state.get("query_nonce", 0) + 1


def init_state() -> None:
    defaults = {
        "page": SECTIONS[0],
        "theory_done": False,
        "schema_done": False,
        "import_done": False,
        "query_done": False,
        "quiz_done": False,
        "report_done": False,
        "query_history": [],
        "trials": [],
        "quiz_answers": {},
        "quiz_result": None,
        "quiz_attempts": 0,
        # Bumped on every retake so each attempt gets fresh widget keys
        # instead of deleting the old ones while they may still be rendered.
        "quiz_round": 0,
        "quiz_unanswered": 0,
        "query_nonce": 0,
        "last_result": None,
        "report_bytes": None,
        "report_error": None,
        "conclusion": "",
        "student": {
            "name": "",
            "roll": "",
            "department": "",
            "semester": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
        },
        "connection_message": "",
        "connection_ok": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "neo4j" not in st.session_state:
        st.session_state.neo4j = Neo4jService()
    if "schema" not in st.session_state:
        load_domain(domains.DEFAULT_DOMAIN)


def service() -> Neo4jService:
    return st.session_state.neo4j


def current_mode() -> str:
    return NEO4J_MODE if service().is_connected else SIMULATION_MODE


def is_dark_theme() -> bool:
    """Ask Streamlit which theme is active so the figures can match it."""
    try:
        theme = getattr(st.context, "theme", None)
        if theme is not None and getattr(theme, "type", None):
            return theme.type == "dark"
    except Exception:
        pass
    try:
        return str(st.get_option("theme.base")).lower() == "dark"
    except Exception:
        return False


def active_snapshot() -> Dict[str, Any]:
    """Graph data for the visualisations, from whichever backend is active."""
    if service().is_connected:
        return service().fetch_graph(limit=300)
    return st.session_state.graph.snapshot()


def active_stats() -> Dict[str, Any]:
    if service().is_connected:
        return service().stats()
    graph: InMemoryGraph = st.session_state.graph
    stats = graph.stats()
    stats["label_counts"] = graph.label_counts()
    stats["relationship_type_counts"] = graph.rel_type_counts()
    return stats


def run_cypher(query: str) -> QueryResult:
    """Execute against Neo4j when connected, otherwise the local engine."""
    if service().is_connected:
        return service().run_query(query)
    return execute_cypher(st.session_state.graph, query)


def remember_query(result: QueryResult) -> None:
    st.session_state.query_history.append(
        {
            "query": result.query,
            "ok": result.ok,
            "mode": result.mode,
            "records": result.record_count,
            "ms": round(result.execution_ms, 2),
            "error": result.error or "",
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )
    st.session_state.query_done = True


def observations() -> Dict[str, Any]:
    """Metrics + plain-language notes used by the UI and the PDF report."""
    stats = active_stats()
    history = st.session_state.query_history
    ok = sum(1 for item in history if item["ok"])
    failed = len(history) - ok
    records = sum(item["records"] for item in history if item["ok"])

    notes: List[str] = []
    if stats["node_count"] == 0:
        notes.append(
            "No data has been imported yet, so no graph measurements are available."
        )
    else:
        notes.append(
            "The current graph contains %d node label(s) and %d relationship type(s)."
            % (stats["label_count"], stats["relationship_type_count"])
        )
        notes.append(
            "The imported dataset contains %d node(s) and %d relationship(s)."
            % (stats["node_count"], stats["relationship_count"])
        )
        notes.append(
            "On average each node takes part in %.2f relationship(s), so the graph is "
            "%s."
            % (
                stats["avg_rels_per_node"],
                "densely connected" if stats["avg_rels_per_node"] >= 2 else "sparsely connected",
            )
        )
        if stats.get("label_counts"):
            biggest = max(stats["label_counts"].items(), key=lambda kv: kv[1])
            notes.append(
                "The most frequent node label is %s with %d node(s)." % (biggest[0], biggest[1])
            )
    if history:
        notes.append(
            "%d Cypher quer(y/ies) were executed in %s mode: %d succeeded and %d failed."
            % (len(history), current_mode(), ok, failed)
        )

    return {
        **stats,
        "queries_total": len(history),
        "queries_ok": ok,
        "queries_failed": failed,
        "records_returned": records,
        "notes": notes,
    }


# ======================================================================
#  Sidebar
# ======================================================================
def _bullet_block(heading: str, messages: List[str], limit: int = 25) -> str:
    """One message box with a bulleted list, instead of a stack of boxes."""
    shown = messages[:limit]
    body = "\n".join("- %s" % message for message in shown)
    if len(messages) > limit:
        body += "\n- ... and %d more" % (len(messages) - limit)
    return "**%s**\n\n%s" % (heading, body)


def _checklist(items: List[Tuple[str, bool]]) -> str:
    """A plain tick/circle checklist -- one markdown block, no emoji noise."""
    return "  \n".join(
        ("✓ &nbsp;%s" if done else "○ &nbsp;%s") % label for label, done in items
    )


def _goto_section(section: str) -> None:
    st.session_state.page = section


def render_topbar() -> None:
    """Masthead, section navigation and lab status -- all across the top.

    There is no sidebar: nothing is written to ``st.sidebar``, so Streamlit does
    not render one at all.
    """
    title_col, status_col = st.columns([3, 1], vertical_alignment="center")
    with title_col:
        st.title(APP_TITLE)
        st.caption("%s · Experiment 9" % APP_SUBTITLE)
    with status_col:
        if service().is_connected:
            st.success("**Mode:** Neo4j")
            st.caption("%s · database %s" % (service().uri, service().database))
        else:
            st.info("**Mode:** Local Simulation")
            st.caption("No database connected.")

    # Navigation bar: one full-width button per section, the active one filled.
    # (A segmented control would look the same but cannot be driven by
    # Streamlit's AppTest harness in this version, which would cost the whole
    # regression suite.)
    nav_columns = st.columns(len(SECTIONS), gap="small")
    for column, section in zip(nav_columns, SECTIONS):
        column.button(
            section,
            key="nav_%s" % section.split("·")[0].strip(),
            type="primary" if st.session_state.page == section else "secondary",
            width="stretch",
            on_click=_goto_section,
            args=(section,),
        )

    steps = [(label, bool(st.session_state.get(key))) for key, label in PROGRESS_STEPS]
    done = sum(1 for _, complete in steps if complete)
    with st.expander("Progress tracker — %d of %d stages complete" % (done, len(steps))):
        st.progress(done / len(steps))
        columns = st.columns(len(steps))
        for column, (label, complete) in zip(columns, steps):
            column.markdown(("✓ &nbsp;**%s**" if complete else "○ &nbsp;%s") % label)
            column.caption("Complete" if complete else "Pending")

    st.divider()


# ======================================================================
#  1. THEORY
# ======================================================================
def render_theory() -> None:
    st.header("Theory")
    st.caption("Background reading for the experiment. Each tab covers one topic.")

    tabs = st.tabs(
        [
            "Knowledge Graphs",
            "Neo4j Fundamentals",
            "Schema Design",
            "Design Principles",
            "Data Import",
            "Cypher Basics",
            "Objectives & Procedure",
        ]
    )

    with tabs[0]:
        st.subheader("Introduction to Knowledge Graphs")
        st.markdown(
            """
**What is a knowledge graph?**
A knowledge graph is a way of storing information as a *network*: real-world things
are stored as **entities**, and the meaningful connections between them are stored
as **relationships**. Unlike a spreadsheet, the connections are data in their own
right - they have a name, a direction, and they can carry their own facts.

**What is an entity?**
An entity is a distinct thing you want to describe: a student, a course, a patient,
a product, a movie. Entities are usually the *nouns* in the description of a domain.

**What is a node?**
A node is how a graph database stores one entity. In this lab the entity
"the student with id S001" becomes the node `(:Student {student_id: 'S001'})`.

**What is a relationship (an edge)?**
A relationship joins exactly two nodes, has a direction and a single **type** that
names the connection, for example `(:Student)-[:ENROLLED_IN]->(:Course)`. Reading
the pattern out loud should form a sentence: *a Student is enrolled in a Course.*

**What is a property?**
A property is a key-value fact stored on a node or on a relationship, such as
`name: 'Aditi'`, `credits: 4` or `grade: 'A'`.
            """
        )
        st.subheader("Relational database vs graph database")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Aspect": "Basic unit",
                        "Relational (SQL)": "Row in a table",
                        "Graph (Neo4j)": "Node with labels and properties",
                    },
                    {
                        "Aspect": "How things connect",
                        "Relational (SQL)": "Foreign keys + JOIN at query time",
                        "Graph (Neo4j)": "Stored relationships, traversed directly",
                    },
                    {
                        "Aspect": "Cost of one more hop",
                        "Relational (SQL)": "Another JOIN, cost grows with table size",
                        "Graph (Neo4j)": "A local step from the node you are on",
                    },
                    {
                        "Aspect": "Many-to-many",
                        "Relational (SQL)": "Extra junction table",
                        "Graph (Neo4j)": "Just another relationship",
                    },
                    {
                        "Aspect": "Schema",
                        "Relational (SQL)": "Fixed columns, declared up front",
                        "Graph (Neo4j)": "Flexible; labels and properties can differ per node",
                    },
                    {
                        "Aspect": "Best at",
                        "Relational (SQL)": "Aggregating large uniform tables",
                        "Graph (Neo4j)": "Following connections, paths and patterns",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.info(
            "**Why graph databases suit connected data:** each node physically stores "
            "its own relationships (*index-free adjacency*), so the cost of following a "
            "connection does not depend on how big the database is. A question like "
            "'which students share a course with Aditi?' is two hops in Cypher, but two "
            "joins over potentially huge tables in SQL."
        )

    with tabs[1]:
        st.subheader("Neo4j Fundamentals")
        st.markdown(
            """
**Neo4j** is a graph database that implements the **property graph model**. Its four
building blocks are:

| Concept | Meaning | Example |
|---|---|---|
| **Node** | One entity | `(s:Student)` |
| **Label** | A category for a node; a node may have several | `:Student`, `:Person` |
| **Relationship** | A directed connection between two nodes | `(s)-[:ENROLLED_IN]->(c)` |
| **Relationship type** | The single name a relationship carries | `ENROLLED_IN` |
| **Property** | A key-value fact on a node *or* a relationship | `name: 'Aditi'`, `grade: 'A'` |

**Cypher** is Neo4j's query language. It is *pattern based*: you draw the shape of
the data you want using ASCII art, and Neo4j finds every place that shape occurs.
            """
        )
        st.code(
            "//  node        relationship          node\n"
            "(s:Student)-[r:ENROLLED_IN]->(c:Course)\n"
            "//   ^              ^                  ^\n"
            "//  label      relationship type     label",
            language="cypher",
        )
        st.markdown(
            """
**The property graph model in one sentence:** *nodes carry labels and properties,
relationships carry one type, a direction and properties, and everything is stored
so that a node knows its own relationships.*
            """
        )
        st.caption(
            "A relationship always has a direction in storage, but you can ignore that "
            "direction when querying by writing `-[:ENROLLED_IN]-` without an arrow."
        )

    with tabs[2]:
        st.subheader("Knowledge Graph Schema: a worked example")
        st.markdown(
            "A schema answers three questions: **what are the entities**, "
            "**how are they connected**, and **what facts do we keep about each**. "
            "Here is the University schema used as the default in this lab."
        )
        st.markdown("**Node labels**")
        st.code(
            "Student\nCourse\nFaculty\nDepartment\nProject",
            language="text",
        )
        st.markdown("**Relationship types**")
        st.code(
            "ENROLLED_IN\nTEACHES\nBELONGS_TO\nWORKS_ON\nGUIDED_BY",
            language="text",
        )
        st.markdown("**Patterns**")
        st.code(
            "(Student)-[:ENROLLED_IN]->(Course)\n"
            "(Faculty)-[:TEACHES]->(Course)\n"
            "(Student)-[:BELONGS_TO]->(Department)\n"
            "(Student)-[:WORKS_ON]->(Project)\n"
            "(Faculty)-[:GUIDED_BY]->(Project)",
            language="cypher",
        )
        st.markdown("**Properties**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Node label": "Student", "Unique ID": "student_id", "Properties": "student_id, name, semester"},
                    {"Node label": "Course", "Unique ID": "course_id", "Properties": "course_id, name, credits"},
                    {"Node label": "Faculty", "Unique ID": "faculty_id", "Properties": "faculty_id, name, specialization"},
                    {"Node label": "Department", "Unique ID": "dept_id", "Properties": "dept_id, name"},
                    {"Node label": "Project", "Unique ID": "project_id", "Properties": "project_id, title, domain"},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Relationship properties**")
        st.code(
            "(Student)-[:ENROLLED_IN {grade: 'A'}]->(Course)\n"
            "(Student)-[:WORKS_ON {role: 'Developer'}]->(Project)",
            language="cypher",
        )
        st.warning(
            "**Direction reads as meaning.** This lab keeps `(Faculty)-[:GUIDED_BY]->(Project)` "
            "because that is the pattern given in the syllabus, but notice that it reads "
            "backwards: a project is guided by a faculty member, so `(Project)-[:GUIDED_BY]->(Faculty)` "
            "would be the more natural direction. Either works technically - you just have "
            "to match the direction you chose when you query. Try reversing it in the "
            "Schema Designer and watch the diagram update."
        )

    with tabs[3]:
        st.subheader("Schema Design Principles")
        st.markdown(
            """
1. **Choose meaningful node labels.** A label is a category of *thing*: `Student`,
   `Course`. Use CamelCase and the singular form. `StudentData` or `Table1` tell a
   reader nothing.
2. **Choose relationship types that read as verbs.** `ENROLLED_IN`, `TEACHES`,
   `WORKS_ON`. Convention is UPPER_SNAKE_CASE. Reading
   *source-type-target* should form a sentence.
3. **Identify entities before relationships.** Take the nouns in the dataset first;
   the verbs connecting them become relationships.
4. **Avoid unnecessary relationships.** If a connection can be derived by
   traversing two others, do not store it as well. A student's department can be
   reached through their course, so store it only if the student's own department
   really can differ.
5. **Select useful properties.** Keep the facts you will query or display. A fact
   that depends on *both* endpoints (a grade, a rating, a role) belongs on the
   relationship, not on either node.
6. **Give every entity a unique identifier.** `student_id`, `course_id`. This is
   what `MERGE` uses to decide whether an entity already exists.
7. **Avoid duplicate entities.** Import with `MERGE` on the key property and back
   it with a uniqueness constraint. `CREATE` run twice creates two nodes.
8. **Stay consistent.** One naming style, one direction convention, the same
   property name for the same fact everywhere in the graph.
            """
        )
        st.code(
            "// Enforce the key so duplicates become impossible\n"
            "CREATE CONSTRAINT student_id_unique IF NOT EXISTS\n"
            "FOR (s:Student) REQUIRE s.student_id IS UNIQUE;",
            language="cypher",
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Poor design**")
            st.code("(:Data {type: 'student', rel: 'course C101'})", language="cypher")
            st.caption("The meaning is hidden inside properties, so nothing can be traversed.")
        with col2:
            st.markdown("**Good design**")
            st.code("(:Student)-[:ENROLLED_IN]->(:Course)", language="cypher")
            st.caption("The label and the relationship type carry the meaning.")

    with tabs[4]:
        st.subheader("Data Import")
        st.markdown(
            """
**Structured entity-relationship data** usually arrives as tables. In this lab the
format is two CSV files:

* `nodes.csv` - columns `label`, `id`, then one column per property
* `relationships.csv` - columns `source_id`, `type`, `target_id`, then any
  relationship properties

The import then happens in two passes: **create the nodes first**, then **connect
them**. A relationship can only be created once both of its end nodes exist.
            """
        )
        st.markdown("**`CREATE` - always inserts**")
        st.code(
            "CREATE (s:Student {student_id: 'S001', name: 'Aditi', semester: 4});",
            language="cypher",
        )
        st.markdown("**`MERGE` - match first, create only if nothing matched**")
        st.code(
            "MERGE (s:Student {student_id: 'S001'})\nSET s.name = 'Aditi', s.semester = 4;",
            language="cypher",
        )
        st.markdown("**`MATCH` + `MERGE` - connect two existing nodes**")
        st.code(
            "MATCH (s:Student {student_id: 'S001'}),\n"
            "      (c:Course {course_id: 'C101'})\n"
            "MERGE (s)-[r:ENROLLED_IN]->(c)\n"
            "SET r.grade = 'A';",
            language="cypher",
        )
        st.markdown("**`LOAD CSV` - reading a file directly inside Neo4j**")
        st.code(
            "LOAD CSV WITH HEADERS FROM 'file:///students.csv' AS row\n"
            "MERGE (s:Student {student_id: row.student_id})\n"
            "SET s.name = row.name,\n"
            "    s.semester = toInteger(row.semester);",
            language="cypher",
        )
        st.info(
            "`LOAD CSV` reads the file from the database server's `import` folder, not "
            "from your own machine. Values arrive as **text**, which is why numbers need "
            "`toInteger()` or `toFloat()`. This lab generates plain MERGE statements "
            "instead, so that the same script also runs in the local simulation engine."
        )
        st.warning(
            "Use `MATCH` (not `MERGE`) for the endpoints when creating relationships. "
            "`MERGE` on a mistyped id silently creates an empty phantom node."
        )

    with tabs[5]:
        st.subheader("Cypher Basics")
        examples = [
            ("CREATE", "Insert new data.", "CREATE (s:Student {student_id: 'S001', name: 'Aditi', semester: 4});"),
            ("MATCH", "Find an existing pattern.", "MATCH (s:Student)\nRETURN s;"),
            ("RETURN", "Choose what comes back.", "MATCH (s:Student)\nRETURN s.name, s.semester;"),
            ("WHERE", "Filter the matched rows.", "MATCH (s:Student)\nWHERE s.semester = 4\nRETURN s.name;"),
            ("MERGE", "Match or create - the safe import clause.", "MERGE (s:Student {student_id: 'S001'})\nSET s.name = 'Aditi';"),
            ("SET", "Add or change a property.", "MATCH (s:Student {student_id: 'S001'})\nSET s.semester = 5;"),
            ("DELETE", "Remove a relationship (or a node with none).", "MATCH (s:Student)-[r:ENROLLED_IN]->(:Course)\nDELETE r;"),
            ("DETACH DELETE", "Remove a node together with its relationships.", "MATCH (s:Student {student_id: 'S001'})\nDETACH DELETE s;"),
            ("WITH", "Pass results from one part of a query to the next.", "MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)\nWITH c, count(s) AS enrolled\nWHERE enrolled > 1\nRETURN c.name, enrolled;"),
            ("ORDER BY", "Sort the rows.", "MATCH (s:Student)\nRETURN s.name, s.semester\nORDER BY s.semester DESC;"),
            ("LIMIT", "Keep only the first N rows.", "MATCH (s:Student)\nRETURN s.name\nORDER BY s.name\nLIMIT 3;"),
        ]
        for clause, description, code in examples:
            with st.expander("%s - %s" % (clause, description)):
                st.code(code, language="cypher")
        st.markdown("**Traversal: the reason the graph exists**")
        st.code(
            "// Students who share a course with Aditi\n"
            "MATCH (a:Student {name: 'Aditi'})-[:ENROLLED_IN]->(c:Course)<-[:ENROLLED_IN]-(b:Student)\n"
            "RETURN DISTINCT b.name AS classmate, c.name AS shared_course;",
            language="cypher",
        )

    with tabs[6]:
        st.subheader("Learning Objectives")
        st.markdown("By the end of the experiment, students should be able to:")
        for index, objective in enumerate(report_generator.LEARNING_OBJECTIVES, start=1):
            st.markdown("%d. %s" % (index, objective))

        st.subheader("Experimental Procedure")
        for index, step in enumerate(report_generator.PROCEDURE_STEPS, start=1):
            st.markdown("**Step %d:** %s" % (index, step))

        st.divider()
        if st.button("Mark theory as studied", type="primary"):
            st.session_state.theory_done = True
            st.success("Theory marked as complete. Move on to the Simulation section.")
        if st.session_state.theory_done:
            st.caption("Theory is marked complete in your progress tracker.")


# ======================================================================
#  2. SIMULATION
# ======================================================================
def render_simulation() -> None:
    st.header("Simulation")

    # The status row is reserved here but filled in at the end of the function.
    # An import or a query runs while a tab below is being drawn, so reading the
    # counts now would report the graph as it was *before* that action.
    status = st.container()

    tabs = st.tabs(
        [
            "Domain & Data",
            "Schema Designer",
            "Schema Diagram",
            "Data Import",
            "Neo4j Connection",
            "Query Console",
            "Graph View",
            "Logbook",
        ]
    )

    with tabs[0]:
        render_domain_tab()
    with tabs[1]:
        render_schema_designer()
    with tabs[2]:
        render_schema_graph()
    with tabs[3]:
        render_import_tab()
    with tabs[4]:
        render_connection_tab()
    with tabs[5]:
        render_query_tab()
    with tabs[6]:
        render_graph_tab()
    with tabs[7]:
        render_logbook_tab()

    # Now that every tab has run, report the graph as it actually stands.
    with status:
        # The execution mode lives in the top bar, so this row carries the
        # numbers that change as the student works instead of repeating it.
        mode = current_mode()
        stats = active_stats()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Domain", st.session_state.domain)
        col2.metric("Node labels", stats["label_count"])
        col3.metric("Nodes in graph", stats["node_count"])
        col4.metric("Relationships", stats["relationship_count"])
        if mode == SIMULATION_MODE:
            st.caption(
                "Running on the in-memory engine. Open the **Neo4j Connection** tab to "
                "run the same Cypher against a real server."
            )
        else:
            st.caption("Queries and imports run on the connected Neo4j database.")


# ------------------------------------------------- Tab: Domain & Data
def _on_domain_change() -> None:
    """Switching domain reloads its schema, data and examples."""
    chosen = st.session_state.get("domain_select")
    if chosen and chosen != st.session_state.domain:
        load_domain(chosen)


def render_domain_tab() -> None:
    st.subheader("Domain selection")
    st.caption(
        "Pick the domain you want to model. Changing the domain reloads its schema, "
        "its sample dataset and its example queries, and clears the working graph."
    )

    names = domains.domain_names()
    st.selectbox(
        "Domain",
        names,
        index=names.index(st.session_state.domain) if st.session_state.domain in names else 0,
        key="domain_select",
        on_change=_on_domain_change,
    )

    domain = domains.get_domain(st.session_state.domain)
    st.write(domain["description"])
    st.info("**Identifying entities:** " + domain["entity_hint"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Entities (node labels) in this domain**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Node label": node["label"],
                        "Unique ID": node["key"],
                        "Properties": ", ".join(node["properties"]),
                    }
                    for node in domain["nodes"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with col2:
        st.markdown("**Relationships in this domain**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Pattern": "(%s)-[:%s]->(%s)" % (rel["source"], rel["type"], rel["target"]),
                        "Properties": ", ".join(rel["properties"]) or "-",
                    }
                    for rel in domain["relationships"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.divider()
    st.subheader("Sample dataset")
    st.caption("Inspect the data before importing it. Every relationship below points at rows that really exist.")

    frames = domains.label_frames(domain)
    for label, frame in frames.items():
        with st.expander("%s - %d row(s)" % (label, len(frame))):
            st.dataframe(frame, hide_index=True, width="stretch")

    with st.expander("Relationships - %d row(s)" % len(domain["edges"]), expanded=True):
        st.dataframe(
            domains.edges_dataframe(domains.dataset_from_domain(domain)),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Every source_id and target_id above refers to a row in the tables "
            "above it - that is what the validator checks before an import."
        )

    st.divider()
    st.subheader("Example Cypher for this domain")
    for example in domain["queries"][:5]:
        with st.expander(example["title"]):
            st.code(example["cypher"], language="cypher")


# ---------------------------------------------- Tab: Schema Designer
# Structural edits run as widget callbacks: a callback fires *before* the script
# re-executes, so the page is drawn once from the new state and there is no need
# for st.rerun() (which would leave widgets for rows that no longer exist).
def _set_designer_message(slot: str, kind: str, text: str) -> None:
    st.session_state["designer_msg_%s" % slot] = (kind, text)


def _show_designer_message(slot: str) -> None:
    message = st.session_state.pop("designer_msg_%s" % slot, None)
    if not message:
        return
    kind, text = message
    {"error": st.error, "success": st.success, "warning": st.warning}.get(kind, st.info)(text)


def _remove_node(node_id: str) -> None:
    schema = st.session_state.schema
    schema["nodes"] = [n for n in schema["nodes"] if n["_id"] != node_id]


def _remove_relationship(rel_id: str) -> None:
    schema = st.session_state.schema
    schema["relationships"] = [r for r in schema["relationships"] if r["_id"] != rel_id]


def _add_node() -> None:
    schema = st.session_state.schema
    label = (st.session_state.get("add_node_label") or "").strip()
    properties = [
        p.strip() for p in (st.session_state.get("add_node_props") or "").split(",") if p.strip()
    ]
    key = (st.session_state.get("add_node_key") or "").strip()

    if not label:
        _set_designer_message("node", "error", "Node label cannot be empty.")
        return
    if any(n["label"] == label for n in schema["nodes"]):
        _set_designer_message(
            "node", "error", "Duplicate node label detected: '%s' already exists." % label
        )
        return
    if key and key not in properties:
        properties.insert(0, key)
    schema["nodes"].append(
        {"_id": _new_id(), "label": label, "key": key, "properties": properties}
    )
    _set_designer_message("node", "success", "Added the node label '%s'." % label)


def _add_relationship() -> None:
    schema = st.session_state.schema
    labels = [n["label"] for n in schema["nodes"] if n["label"]]
    rtype = (st.session_state.get("add_rel_type") or "").strip()
    source = st.session_state.get("add_rel_source")
    target = st.session_state.get("add_rel_target")

    if not rtype:
        _set_designer_message("rel", "error", "Relationship type cannot be empty.")
        return
    if not labels:
        _set_designer_message(
            "rel", "error", "Define at least one node label before adding a relationship."
        )
        return
    if source not in labels or target not in labels:
        _set_designer_message(
            "rel", "error", "Choose a source and a target from the defined node labels."
        )
        return
    schema["relationships"].append(
        {
            "_id": _new_id(),
            "type": rtype,
            "source": source,
            "target": target,
            "properties": [
                p.strip()
                for p in (st.session_state.get("add_rel_props") or "").split(",")
                if p.strip()
            ],
        }
    )
    _set_designer_message(
        "rel", "success", "Added (%s)-[:%s]->(%s)." % (source, rtype, target)
    )


def _reset_schema() -> None:
    domain = domains.get_domain(st.session_state.domain)
    st.session_state.schema = _tag_schema(domains.schema_from_domain(domain))


def _clear_schema() -> None:
    st.session_state.schema = {
        "domain": st.session_state.domain,
        "nodes": [],
        "relationships": [],
    }


def render_schema_designer() -> None:
    st.subheader("Schema designer")
    st.caption(
        "Define the node labels, their properties and unique identifiers, then the "
        "relationship types that connect them. The diagram and the generated Cypher "
        "follow whatever you design here."
    )

    schema = st.session_state.schema

    # ---- node labels ------------------------------------------------
    st.markdown("### Node labels")
    if not schema["nodes"]:
        st.warning("No node labels defined. Add one below to begin.")

    for node in list(schema["nodes"]):
        node_id = node["_id"]
        with st.expander(":%s" % (node["label"] or "(unnamed)"), expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                node["label"] = st.text_input(
                    "Node label", value=node["label"], key="nlabel_%s" % node_id
                ).strip()
            with col2:
                raw = st.text_input(
                    "Properties (comma separated)",
                    value=", ".join(node["properties"]),
                    key="nprops_%s" % node_id,
                )
            node["properties"] = [p.strip() for p in raw.split(",") if p.strip()]

            col3, col4 = st.columns([2, 1])
            with col3:
                options = node["properties"] or ["(define a property first)"]
                current = node.get("key")
                index = options.index(current) if current in options else 0
                node["key"] = st.selectbox(
                    "Unique ID property",
                    options,
                    index=index,
                    key="nkey_%s" % node_id,
                    help="MERGE uses this property to decide whether the entity already exists.",
                )
                if node["key"] not in node["properties"]:
                    node["key"] = ""
            with col4:
                st.write("")
                st.write("")
                st.button(
                    "Remove label",
                    key="ndel_%s" % node_id,
                    on_click=_remove_node,
                    args=(node_id,),
                )

    with st.form("add_node_form", clear_on_submit=True):
        st.markdown("**Add a node label**")
        col1, col2, col3 = st.columns([1, 2, 1])
        col1.text_input("Label", placeholder="Student", key="add_node_label")
        col2.text_input(
            "Properties (comma separated)",
            placeholder="student_id, name, semester",
            key="add_node_props",
        )
        col3.text_input("Unique ID property", placeholder="student_id", key="add_node_key")
        st.form_submit_button("Add node label", type="primary", on_click=_add_node)
    _show_designer_message("node")

    # ---- relationships ---------------------------------------------
    st.markdown("### Relationship types")
    labels = [n["label"] for n in schema["nodes"] if n["label"]]

    if not schema["relationships"]:
        st.warning("No relationships defined yet.")

    for rel in list(schema["relationships"]):
        rel_id = rel["_id"]
        title = "(%s)-[:%s]->(%s)" % (
            rel.get("source") or "?",
            rel.get("type") or "?",
            rel.get("target") or "?",
        )
        with st.expander(title, expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                rel["type"] = st.text_input(
                    "Relationship type", value=rel["type"], key="rtype_%s" % rel_id
                ).strip()
            # Keep a label the relationship already points at in the option
            # list even if it has been deleted, so the dangling reference is
            # preserved and reported instead of being silently rewired.
            options = list(labels)
            for referenced in (rel.get("source"), rel.get("target")):
                if referenced and referenced not in options:
                    options.append(referenced)
            if not options:
                options = ["(no node labels yet)"]
            with col2:
                index = options.index(rel["source"]) if rel.get("source") in options else 0
                rel["source"] = st.selectbox(
                    "Source node label", options, index=index, key="rsrc_%s" % rel_id
                )
            with col3:
                index = options.index(rel["target"]) if rel.get("target") in options else 0
                rel["target"] = st.selectbox(
                    "Target node label", options, index=index, key="rtgt_%s" % rel_id
                )
            col4, col5 = st.columns([3, 1])
            with col4:
                raw = st.text_input(
                    "Relationship properties (comma separated)",
                    value=", ".join(rel.get("properties", [])),
                    key="rprops_%s" % rel_id,
                )
                rel["properties"] = [p.strip() for p in raw.split(",") if p.strip()]
            with col5:
                st.write("")
                st.write("")
                st.button(
                    "Remove",
                    key="rdel_%s" % rel_id,
                    on_click=_remove_relationship,
                    args=(rel_id,),
                )

    with st.form("add_rel_form", clear_on_submit=True):
        st.markdown("**Add a relationship type**")
        col1, col2, col3, col4 = st.columns(4)
        col1.text_input("Type", placeholder="ENROLLED_IN", key="add_rel_type")
        col2.selectbox(
            "Source label", labels or ["(add a node label first)"], key="add_rel_source"
        )
        col3.selectbox(
            "Target label", labels or ["(add a node label first)"], key="add_rel_target"
        )
        col4.text_input("Properties", placeholder="grade", key="add_rel_props")
        st.form_submit_button("Add relationship", type="primary", on_click=_add_relationship)
    _show_designer_message("rel")

    # ---- validation -------------------------------------------------
    st.divider()
    st.markdown("#### Validation")
    errors, warnings = schema_tools.validate_schema(schema)
    if errors:
        st.session_state.schema_done = False
        st.error(_bullet_block("%d problem(s) to fix" % len(errors), errors))
    else:
        st.session_state.schema_done = True
        st.success(
            "Schema is valid: %d node label(s) and %d relationship(s)."
            % (len(schema["nodes"]), len(schema["relationships"]))
        )
    if warnings:
        st.warning(_bullet_block("%d design note(s)" % len(warnings), warnings))

    col1, col2 = st.columns(2)
    col1.button("Reset schema to the domain default", on_click=_reset_schema)
    col2.button("Clear the whole schema", on_click=_clear_schema)


# ----------------------------------------------- Tab: Schema Diagram
def render_schema_graph() -> None:
    st.subheader("Schema diagram")
    st.caption(
        "This diagram is generated from your schema on every change - edit the "
        "designer and it redraws."
    )
    schema = st.session_state.schema
    errors, _ = schema_tools.validate_schema(schema)
    if errors:
        st.warning(
            "The schema has %d problem(s); the diagram shows only the parts that are "
            "currently valid." % len(errors)
        )
    figure = graph_viz.schema_figure(schema, dark=is_dark_theme())
    st.plotly_chart(figure, use_container_width=True, key="schema_chart")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Node labels and properties**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Node label": n["label"],
                        "Unique ID": n.get("key") or "(none)",
                        "Properties": ", ".join(n["properties"]) or "(none)",
                    }
                    for n in schema["nodes"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with col2:
        st.markdown("**Relationship types**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Pattern": "(%s)-[:%s]->(%s)" % (r.get("source"), r.get("type"), r.get("target")),
                        "Properties": ", ".join(r.get("properties", [])) or "-",
                    }
                    for r in schema["relationships"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )


# -------------------------------------------------- Tab: Data Import
def _sample_csv_bytes(kind: str) -> bytes:
    domain = domains.get_domain(st.session_state.domain)
    dataset = domains.dataset_from_domain(domain)
    frame = (
        domains.nodes_dataframe(dataset) if kind == "nodes" else domains.edges_dataframe(dataset)
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def render_import_tab() -> None:
    st.subheader("Step 1 — Choose the data")

    source = st.radio(
        "Choose the data to import",
        ["Use the built-in sample dataset", "Upload CSV files"],
        key="data_source_choice",
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "Use the built-in sample dataset":
        domain = domains.get_domain(st.session_state.domain)
        col1, col2, col3 = st.columns(3)
        if col1.button("Load sample dataset", type="primary"):
            st.session_state.dataset = domains.dataset_from_domain(domain)
            st.session_state.dataset_report = None
            st.success("Loaded the built-in %s dataset." % domain["name"])
        col2.download_button(
            "Download nodes.csv template",
            data=_sample_csv_bytes("nodes"),
            file_name="%s_nodes.csv" % domain["name"].lower().replace(" ", "_"),
            mime="text/csv",
        )
        col3.download_button(
            "Download relationships.csv template",
            data=_sample_csv_bytes("relationships"),
            file_name="%s_relationships.csv" % domain["name"].lower().replace(" ", "_"),
            mime="text/csv",
        )
    else:
        st.caption(
            "`nodes.csv` needs the columns **label** and **id** plus one column per "
            "property. `relationships.csv` needs **source_id**, **type** and "
            "**target_id** plus any relationship properties."
        )
        col1, col2 = st.columns(2)
        nodes_file = col1.file_uploader("nodes.csv", type=["csv"], key="upload_nodes")
        rels_file = col2.file_uploader("relationships.csv (optional)", type=["csv"], key="upload_rels")
        if st.button("Load uploaded files", type="primary"):
            if nodes_file is None:
                st.error("Please choose a nodes CSV file first.")
            else:
                try:
                    nodes_df = pd.read_csv(nodes_file)
                    rels_df = pd.read_csv(rels_file) if rels_file is not None else None
                except Exception as exc:
                    st.error("The CSV file could not be read: %s" % exc)
                else:
                    dataset, structural = schema_tools.dataset_from_uploads(
                        nodes_df,
                        rels_df,
                        source="Uploaded CSV (%s)" % nodes_file.name,
                    )
                    if structural:
                        st.error(_bullet_block("The file cannot be used", structural))
                    else:
                        st.session_state.dataset = dataset
                        st.session_state.dataset_report = None
                        st.success(
                            "Loaded %d node row(s) and %d relationship row(s)."
                            % (len(dataset["nodes"]), len(dataset["edges"]))
                        )

    dataset = st.session_state.dataset
    summary = schema_tools.dataset_summary(dataset)
    st.caption("Current data source: **%s**" % summary["source"])

    with st.expander("Inspect the loaded data", expanded=False):
        st.markdown("**Nodes**")
        st.dataframe(domains.nodes_dataframe(dataset), hide_index=True, width="stretch")
        st.markdown("**Relationships**")
        st.dataframe(domains.edges_dataframe(dataset), hide_index=True, width="stretch")

    st.divider()
    st.subheader("Step 2 — Validate the data against your schema")
    st.caption(
        "Checks for missing columns, blank or duplicate ids, and relationships that "
        "point at entities which do not exist."
    )
    if st.button("Validate dataset"):
        st.session_state.dataset_report = schema_tools.validate_dataset(
            dataset, st.session_state.schema
        )

    report = st.session_state.dataset_report
    if report is not None:
        if report["ok"]:
            st.success(
                _bullet_block("Validation passed — ready to import", report["checks"])
            )
        else:
            if report["checks"]:
                st.info(_bullet_block("Detected in the file", report["checks"]))
            st.error(
                _bullet_block(
                    "Validation failed — %d problem(s) to fix before importing"
                    % len(report["errors"]),
                    report["errors"],
                )
            )
        if report["warnings"]:
            st.warning(
                _bullet_block("%d warning(s)" % len(report["warnings"]), report["warnings"])
            )

    # ---- Cypher generator ------------------------------------------
    st.divider()
    st.subheader("Step 3 — Generate the Cypher")
    col1, col2 = st.columns(2)
    use_merge = col1.checkbox(
        "Use MERGE instead of CREATE (recommended - prevents duplicates)", value=True
    )
    include_constraints = col2.checkbox(
        "Include uniqueness constraints (Neo4j only)", value=False
    )
    if include_constraints and not service().is_connected:
        st.caption(
            "Constraints are a Neo4j feature. The local simulation engine skips them "
            "and relies on MERGE to keep entities unique."
        )

    col1, col2, col3 = st.columns(3)
    if col1.button("Generate Cypher", type="primary"):
        st.session_state.generated_cypher = schema_tools.generate_cypher(
            st.session_state.schema,
            dataset,
            use_merge=use_merge,
            include_constraints=include_constraints,
        )
    if col2.button("Clear Cypher"):
        st.session_state.generated_cypher = ""
    if st.session_state.generated_cypher:
        col3.download_button(
            "Download .cypher",
            data=st.session_state.generated_cypher.encode("utf-8"),
            file_name="import_%s.cypher" % st.session_state.domain.lower().replace(" ", "_"),
            mime="text/plain",
        )

    if st.session_state.generated_cypher:
        st.caption(
            "Use the copy button in the top-right corner of the code block to copy the script."
        )
        st.code(st.session_state.generated_cypher, language="cypher")
    else:
        examples = schema_tools.sample_statements(st.session_state.schema, dataset)
        if examples:
            st.caption("Press **Generate Cypher** to build the full script. It will look like this:")
            for name, code in examples.items():
                st.code(code, language="cypher")

    # ---- run the import --------------------------------------------
    st.divider()
    st.subheader("Step 4 — Import into the graph")
    mode = current_mode()
    st.caption(
        "Target: %s"
        % (
            "Neo4j database '%s'" % service().database
            if mode == NEO4J_MODE
            else "Local Simulation (in-memory graph)"
        )
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Import into %s" % mode, type="primary", key="do_import"):
            _run_import(mode)
    with col2:
        if st.button("Clear the imported graph", key="clear_graph"):
            if mode == NEO4J_MODE:
                result = service().clear_database()
                if result.ok:
                    st.success("The Neo4j database was cleared.")
                else:
                    st.error(result.error)
            else:
                st.session_state.graph = InMemoryGraph()
                st.success("The local simulation graph was cleared.")
            st.session_state.import_status = "Cleared"
            st.session_state.import_done = False

    if st.session_state.import_summary:
        summary = st.session_state.import_summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes imported", summary.get("nodes_created", 0))
        col2.metric("Relationships imported", summary.get("relationships_created", 0))
        col3.metric("Statements executed", summary.get("statements", 0))
        st.caption("Status: %s" % st.session_state.import_status)


def _run_import(mode: str) -> None:
    """Validate, generate if needed, then execute the import script."""
    dataset = st.session_state.dataset
    schema = st.session_state.schema

    schema_errors, _ = schema_tools.validate_schema(schema)
    if schema_errors:
        st.error(_bullet_block("Fix the schema before importing", schema_errors))
        return

    report = schema_tools.validate_dataset(dataset, schema)
    st.session_state.dataset_report = report
    if not report["ok"]:
        st.error(
            _bullet_block(
                "Nothing was imported — the dataset did not pass validation",
                report["errors"],
            )
        )
        return

    script = st.session_state.generated_cypher
    if not script.strip():
        script = schema_tools.generate_cypher(schema, dataset, use_merge=True)
        st.session_state.generated_cypher = script

    started = time.perf_counter()
    if mode == NEO4J_MODE:
        totals, errors = service().run_script(script)
    else:
        graph, totals, errors = schema_tools.build_local_graph(script, st.session_state.graph)
        st.session_state.graph = graph
    elapsed = (time.perf_counter() - started) * 1000

    st.session_state.import_summary = totals
    if errors:
        st.session_state.import_status = "Imported with %d error(s) (%s)" % (len(errors), mode)
        st.warning("%d statement(s) failed. The rest were imported." % len(errors))
        with st.expander("Show the failed statements"):
            for message in errors[:20]:
                st.code(message, language="text")
    else:
        st.session_state.import_status = "Imported successfully (%s)" % mode
        st.success(
            "Import finished in %.0f ms - nodes imported: %d, relationships imported: %d."
            % (elapsed, totals.get("nodes_created", 0), totals.get("relationships_created", 0))
        )
        skipped = totals.get("schema_commands_skipped", 0)
        if skipped:
            st.info(
                "%d constraint statement(s) were skipped: constraints are enforced by a "
                "real Neo4j server, and the local engine relies on MERGE instead." % skipped
            )
        if totals.get("nodes_created", 0) == 0 and totals.get("relationships_created", 0) == 0:
            st.info(
                "Nothing new was created because MERGE found every entity already in the "
                "graph - that is exactly the duplicate prevention you are testing."
            )
    st.session_state.import_done = True


# --------------------------------------------- Tab: Neo4j Connection
def _form_credentials() -> Dict[str, str]:
    return {
        "uri": st.session_state.get("n4j_uri", ""),
        "username": st.session_state.get("n4j_user", ""),
        "password": st.session_state.get("n4j_pwd", ""),
        "database": st.session_state.get("n4j_db", ""),
    }


def _test_connection() -> None:
    credentials = _form_credentials()
    ok, message = service().test_connection(**credentials)
    st.session_state.connection_ok = ok
    st.session_state.connection_message = message


def _connect_neo4j() -> None:
    credentials = _form_credentials()
    ok, message = service().connect(**credentials)
    st.session_state.connection_ok = ok
    st.session_state.connection_message = message


def _disconnect_neo4j() -> None:
    service().disconnect()
    st.session_state.connection_ok = None
    st.session_state.connection_message = (
        "Disconnected. The lab is back in Local Simulation Mode."
    )


def render_connection_tab() -> None:
    st.subheader("Neo4j connection")

    if not DRIVER_AVAILABLE:
        st.warning(
            "The `neo4j` Python driver is not installed in this environment, so only "
            "Local Simulation Mode is available. Install it with "
            "`pip install -r requirements.txt`."
        )

    defaults = env_defaults()
    if env_password_present():
        st.caption(
            "A password was found in the `NEO4J_PASSWORD` environment variable and has "
            "been pre-filled."
        )

    with st.form("neo4j_form"):
        col1, col2 = st.columns(2)
        col1.text_input(
            "Neo4j URI", value=defaults["uri"], placeholder="bolt://localhost:7687", key="n4j_uri"
        )
        col2.text_input("Database", value=defaults["database"], placeholder="neo4j", key="n4j_db")
        col3, col4 = st.columns(2)
        col3.text_input("Username", value=defaults["username"], placeholder="neo4j", key="n4j_user")
        col4.text_input(
            "Password",
            value=defaults["password"],
            type="password",
            key="n4j_pwd",
            help="The password stays masked; it is never displayed back to you.",
        )
        col5, col6, col7 = st.columns(3)
        col5.form_submit_button("Test Connection", on_click=_test_connection)
        col6.form_submit_button("Connect", type="primary", on_click=_connect_neo4j)
        col7.form_submit_button("Disconnect", on_click=_disconnect_neo4j)

    if st.session_state.connection_message:
        if st.session_state.connection_ok:
            st.success(st.session_state.connection_message)
        elif st.session_state.connection_ok is False:
            st.error(st.session_state.connection_message)
            st.info(
                "**Neo4j is not available. The laboratory has switched to Local "
                "Simulation Mode.** Every step of the experiment still works; the "
                "Cypher just runs against the in-memory engine instead."
            )
        else:
            st.info(st.session_state.connection_message)

    st.divider()
    if service().is_connected:
        st.success("Connected to %s" % service().uri)
        stats = service().stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Server", service().server_info or "Neo4j")
        col2.metric("Nodes on server", stats["node_count"])
        col3.metric("Relationships on server", stats["relationship_count"])
    else:
        st.info("Not connected. Execution mode: **Local Simulation**.")

    with st.expander("Using environment variables instead of this form"):
        st.code(
            "# Windows PowerShell\n"
            '$env:NEO4J_URI = "bolt://localhost:7687"\n'
            '$env:NEO4J_USERNAME = "neo4j"\n'
            '$env:NEO4J_PASSWORD = "your-password"\n'
            '$env:NEO4J_DATABASE = "neo4j"\n\n'
            "# macOS / Linux\n"
            'export NEO4J_URI="bolt://localhost:7687"\n'
            'export NEO4J_USERNAME="neo4j"\n'
            'export NEO4J_PASSWORD="your-password"\n'
            'export NEO4J_DATABASE="neo4j"',
            language="bash",
        )
        st.caption("Set them before starting Streamlit; the form picks them up automatically.")


# ------------------------------------------------ Tab: Query Console
def _set_query(text: str) -> None:
    """Replace the editor contents.

    The editor key carries a version number: bumping it makes Streamlit build a
    fresh widget seeded from ``value=``. Writing to a live widget's own key does
    not update the box in the browser, even though the server value changes.
    """
    st.session_state.query_text = text
    st.session_state.query_nonce = st.session_state.get("query_nonce", 0) + 1


def _load_example_query() -> None:
    domain = domains.get_domain(st.session_state.domain)
    choice = st.session_state.get("example_query_choice")
    for example in domain["queries"]:
        if example["title"] == choice:
            _set_query(example["cypher"])
            return


def _clear_query() -> None:
    _set_query("")
    st.session_state.last_result = None


def render_query_tab() -> None:
    st.subheader("Cypher query console")
    mode = current_mode()
    if mode == SIMULATION_MODE:
        st.caption("Queries run on the in-memory engine (Local Simulation).")
        with st.expander("Which Cypher does the local engine support?"):
            st.markdown(
                "`MATCH`, `OPTIONAL MATCH`, `WHERE`, `CREATE`, `MERGE`, `SET`, "
                "`DELETE`, `DETACH DELETE`, `WITH`, `UNWIND`, `RETURN` (with "
                "`DISTINCT` and `count` / `collect` / `sum` / `avg` / `min` / `max`), "
                "`ORDER BY`, `SKIP`, `LIMIT`, and variable-length patterns such as "
                "`-[:ENROLLED_IN*1..3]->`.\n\n"
                "Anything outside this subset is reported as a message. Connect a real "
                "Neo4j server to run the full language."
            )
    else:
        st.caption("Queries run on the connected Neo4j database.")

    domain = domains.get_domain(st.session_state.domain)
    titles = [example["title"] for example in domain["queries"]]
    st.selectbox("Example queries for this domain", titles, key="example_query_choice")
    st.button("Load this example", on_click=_load_example_query)

    editor_key = "query_editor_%d" % st.session_state.get("query_nonce", 0)
    typed = st.text_area(
        "Cypher query", value=st.session_state.query_text, key=editor_key, height=160
    )
    # Keep the plain session key in step with whatever is in the box.
    st.session_state.query_text = typed

    col1, col2, col3 = st.columns([1, 1, 3])
    run = col1.button("Run query", type="primary")
    col2.button("Clear", on_click=_clear_query)

    if run:
        query = st.session_state.query_text
        if not query.strip():
            st.error("The query is empty. Type a Cypher statement first.")
        else:
            result = run_cypher(query)
            result.query = query
            st.session_state.last_result = result
            remember_query(result)

    result: Optional[QueryResult] = st.session_state.last_result
    if result is not None:
        st.divider()
        if result.ok:
            st.success(
                "Query executed successfully in %s mode - %d record(s) in %.1f ms."
                % (result.mode, result.record_count, result.execution_ms)
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Records", result.record_count)
            col2.metric("Execution time", "%.1f ms" % result.execution_ms)
            col3.metric("Mode", result.mode)
            if result.notice:
                st.info(result.notice)
            elif result.summary:
                st.info("Write summary - %s" % result.summary_text())
            frame = result.to_dataframe()
            if frame.empty:
                st.info(
                    "The query ran but returned no rows. Check the labels, property "
                    "values and relationship directions in your pattern."
                )
            else:
                st.dataframe(frame, width="stretch", hide_index=True)
                st.download_button(
                    "Download results CSV",
                    data=frame.to_csv(index=False).encode("utf-8"),
                    file_name="query_results.csv",
                    mime="text/csv",
                )
        else:
            st.error("Query failed: %s" % result.error)
            st.caption(
                "Nothing was changed in the graph. Fix the statement and run it again."
            )

    if st.session_state.query_history:
        with st.expander("Query history (%d)" % len(st.session_state.query_history)):
            st.dataframe(
                pd.DataFrame(st.session_state.query_history)[
                    ["time", "mode", "ok", "records", "ms", "query"]
                ],
                hide_index=True,
                width="stretch",
            )


# --------------------------------------------------- Tab: Graph View
def render_graph_tab() -> None:
    st.subheader("Graph visualization")
    snapshot = active_snapshot()
    node_count = len(snapshot["nodes"])

    if node_count == 0:
        st.info(
            "The graph is empty. Open the **Data Import** tab, validate the dataset "
            "and import it, then come back here."
        )
        st.plotly_chart(
            graph_viz.instance_figure({"nodes": [], "relationships": []}, dark=is_dark_theme())[0],
            use_container_width=True,
            key="empty_graph_chart",
        )
        return

    labels = sorted({node["label"] for node in snapshot["nodes"]})
    types = sorted({rel["type"] for rel in snapshot["relationships"]})

    col1, col2 = st.columns(2)
    label_filter = col1.multiselect("Filter by node label", labels, default=labels)
    type_filter = col2.multiselect("Filter by relationship type", types, default=types)

    col3, col4, col5 = st.columns(3)
    show_rels = col3.checkbox("Show relationships", value=True)
    show_edge_labels = col4.checkbox("Show relationship types on the arrows", value=False)
    show_captions = col5.checkbox("Show node captions", value=True)

    figure, shown = graph_viz.instance_figure(
        snapshot,
        dark=is_dark_theme(),
        labels_filter=label_filter,
        types_filter=type_filter,
        show_relationships=show_rels,
        show_edge_labels=show_edge_labels,
        show_captions=show_captions,
    )
    st.plotly_chart(figure, use_container_width=True, key="instance_chart")
    caption = "Showing %d of %d node(s) and %d of %d relationship(s) · source: %s" % (
        shown["nodes"],
        node_count,
        shown["relationships"],
        len(snapshot["relationships"]),
        current_mode(),
    )
    if shown.get("truncated"):
        caption += " · the view is capped for readability"
    st.caption(caption)

    with st.expander("Node table"):
        rows = []
        for node in snapshot["nodes"]:
            row = {"label": node["label"], "caption": node["caption"]}
            row.update(node["properties"])
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows).fillna("").astype(str), hide_index=True, width="stretch"
        )

    with st.expander("Relationship table"):
        caption_of = {n["nid"]: n["caption"] for n in snapshot["nodes"]}
        label_of = {n["nid"]: n["label"] for n in snapshot["nodes"]}
        rows = []
        for rel in snapshot["relationships"]:
            row = {
                "source": caption_of.get(rel["source"], rel["source"]),
                "source_label": label_of.get(rel["source"], ""),
                "type": rel["type"],
                "target": caption_of.get(rel["target"], rel["target"]),
                "target_label": label_of.get(rel["target"], ""),
            }
            row.update(rel["properties"])
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows).fillna("").astype(str), hide_index=True, width="stretch"
        )


# ------------------------------------------------------ Tab: Logbook
def _clear_trials() -> None:
    st.session_state.trials = []


def render_logbook_tab() -> None:
    st.subheader("Experimental data logbook")
    st.caption(
        "Record a trial after each import or query so the values appear in your lab report."
    )

    schema = st.session_state.schema
    stats = active_stats()
    last: Optional[QueryResult] = st.session_state.last_result

    col1, col2, col3 = st.columns(3)
    if col1.button("Record current trial", type="primary"):
        st.session_state.trials.append(
            {
                "Trial": len(st.session_state.trials) + 1,
                "Domain": st.session_state.domain,
                "Node Types": len([n for n in schema["nodes"] if n["label"]]),
                "Relationship Types": len({r["type"] for r in schema["relationships"] if r["type"]}),
                "Nodes": stats["node_count"],
                "Relationships": stats["relationship_count"],
                "Cypher Query": last.query if last is not None else "",
                "Query Result Count": last.record_count if last is not None and last.ok else 0,
                "Import Status": st.session_state.import_status,
                "Mode": current_mode(),
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        st.success("Trial %d recorded." % len(st.session_state.trials))
    col2.button("Clear trials", on_click=_clear_trials)
    if st.session_state.trials:
        frame = pd.DataFrame(st.session_state.trials)
        col3.download_button(
            "Download trials CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="kg_lab_trials.csv",
            mime="text/csv",
        )
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.info("No trials recorded yet.")

    st.divider()
    st.subheader("Observations")
    data = observations()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Node labels", data["label_count"])
    col2.metric("Relationship types", data["relationship_type_count"])
    col3.metric("Nodes", data["node_count"])
    col4.metric("Relationships", data["relationship_count"])
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg. relationships / node", data["avg_rels_per_node"])
    col6.metric("Queries executed", data["queries_total"])
    col7.metric("Successful", data["queries_ok"])
    col8.metric("Failed", data["queries_failed"])

    st.markdown("\n".join("- %s" % note for note in data["notes"]))

    if data.get("label_counts"):
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                graph_viz.distribution_figure(
                    data["label_counts"], "Nodes per label", dark=is_dark_theme()
                ),
                use_container_width=True,
                key="label_dist_chart",
            )
        with col2:
            st.plotly_chart(
                graph_viz.distribution_figure(
                    data.get("relationship_type_counts", {}),
                    "Relationships per type",
                    dark=is_dark_theme(),
                ),
                use_container_width=True,
                key="type_dist_chart",
            )


# ======================================================================
#  3. QUIZ
# ======================================================================
def render_quiz() -> None:
    st.header("Quiz")
    st.caption(
        "%d multiple-choice questions covering knowledge graphs, Neo4j, Cypher, "
        "schema design and data import." % len(quiz_bank.QUESTIONS)
    )

    result = st.session_state.quiz_result
    if result is not None:
        _render_quiz_result(result)
        return

    round_no = st.session_state.quiz_round
    with st.form("quiz_form_%d" % round_no, enter_to_submit=False):
        for level in quiz_bank.level_order():
            st.subheader("%s questions" % level)
            for question in quiz_bank.questions_by_level(level):
                st.markdown("**%s**" % question["question"])
                st.radio(
                    "Select an answer",
                    options=list(range(len(question["options"]))),
                    format_func=lambda index, q=question: q["options"][index],
                    index=None,
                    key="quiz_%d_%s" % (round_no, question["id"]),
                    label_visibility="collapsed",
                )
                st.divider()
        st.form_submit_button(
            "Submit Quiz", type="primary", on_click=_submit_quiz, args=(round_no,)
        )


def _submit_quiz(round_no: int) -> None:
    """Grade the attempt.

    Run as a form callback, which fires *before* the script re-executes, so the
    result view is what gets drawn next -- no st.rerun() needed.
    """
    answers = {
        question["id"]: st.session_state.get("quiz_%d_%s" % (round_no, question["id"]))
        for question in quiz_bank.QUESTIONS
    }
    st.session_state.quiz_answers = answers
    st.session_state.quiz_result = quiz_bank.grade(answers)
    st.session_state.quiz_unanswered = sum(1 for value in answers.values() if value is None)
    st.session_state.quiz_attempts += 1
    st.session_state.quiz_done = True


def _retake_quiz() -> None:
    """Start a new attempt.

    The round counter changes every widget key, so the previous answers are left
    behind instead of being deleted while their widgets may still be on screen.
    """
    st.session_state.quiz_round += 1
    st.session_state.quiz_result = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_unanswered = 0


def _render_quiz_result(result: Dict[str, Any]) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Score", "%d / %d" % (result["correct"], result["total"]))
    col2.metric("Percentage", "%.1f %%" % result["percentage"])
    col3.metric("Attempts", st.session_state.quiz_attempts)
    st.progress(result["percentage"] / 100.0)

    if result["percentage"] >= 60:
        st.success(result["verdict"])
    else:
        st.warning(result["verdict"])

    unanswered = st.session_state.get("quiz_unanswered", 0)
    if unanswered:
        st.info(
            "%d question(s) were left unanswered and counted as incorrect." % unanswered
        )

    st.dataframe(
        pd.DataFrame(
            [
                {"Difficulty": level, "Correct": data["correct"], "Total": data["total"]}
                for level, data in result["by_level"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Answer review")
    for index, detail in enumerate(result["details"], start=1):
        mark = "✓" if detail["is_correct"] else "✗"
        with st.expander(
            "%s  Q%d · %s — %s" % (mark, index, detail["level"], detail["question"])
        ):
            if detail["is_correct"]:
                st.success("Your answer: %s" % detail["chosen_text"])
            else:
                st.error("Your answer: %s" % detail["chosen_text"])
                st.success("Correct answer: %s" % detail["correct_text"])
            st.caption("**Why:** %s" % detail["explanation"])

    st.button("Retake Quiz", on_click=_retake_quiz)


# ======================================================================
#  4. REPORT
# ======================================================================
def render_report() -> None:
    st.header("Report Generation")
    st.caption("Fill in your details, then generate and download the PDF laboratory report.")

    student = st.session_state.student
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Student Name", value=student["name"])
        roll = col2.text_input("Student ID / Roll Number", value=student["roll"])
        col3, col4, col5 = st.columns(3)
        department = col3.text_input("Department", value=student["department"])
        semester = col4.text_input("Semester", value=student["semester"])
        exam_date = col5.date_input("Experiment Date")
        conclusion = st.text_area(
            "Student Conclusion",
            value=st.session_state.conclusion,
            height=140,
            placeholder=(
                "What did you design, what did you import, what did your queries show, "
                "and what did you learn about schema design?"
            ),
        )
        saved = st.form_submit_button("Save details", type="primary")

    if saved:
        st.session_state.student = {
            "name": name,
            "roll": roll,
            "department": department,
            "semester": semester,
            "date": exam_date.isoformat(),
        }
        st.session_state.conclusion = conclusion
        st.success("Details saved.")

    st.divider()
    st.subheader("Report preview")
    data = observations()
    quiz = st.session_state.quiz_result
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Node labels", data["label_count"])
    col2.metric("Nodes", data["node_count"])
    col3.metric("Relationships", data["relationship_count"])
    col4.metric("Quiz", "%d/%d" % (quiz["correct"], quiz["total"]) if quiz else "Not taken")

    checks = [
        ("Schema designed and valid", st.session_state.schema_done),
        ("Data imported", st.session_state.import_done),
        ("Queries executed", bool(st.session_state.query_history)),
        ("Trials recorded", bool(st.session_state.trials)),
        ("Quiz completed", quiz is not None),
        ("Student details filled in", bool(st.session_state.student["name"])),
    ]
    st.markdown(_checklist(checks))

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Generate PDF Report", type="primary"):
        context = {
            "student": st.session_state.student,
            "schema": st.session_state.schema,
            "domain": st.session_state.domain,
            "mode": current_mode(),
            "dataset_summary": schema_tools.dataset_summary(st.session_state.dataset),
            "import_status": st.session_state.import_status,
            "observations": data,
            "quiz": quiz,
            "trials": st.session_state.trials,
            "queries": st.session_state.query_history,
            "conclusion": st.session_state.conclusion,
        }
        pdf_bytes, error = report_generator.build_report(context)
        st.session_state.report_bytes = pdf_bytes
        st.session_state.report_error = error
        if error:
            st.error("The report could not be generated: %s" % error)
        else:
            st.session_state.report_done = True
            st.success("Report generated - %.1f KB. Use the download button." % (len(pdf_bytes) / 1024))

    if st.session_state.report_bytes:
        roll_text = (st.session_state.student["roll"] or "student").replace(" ", "_")
        col2.download_button(
            "Download PDF Report",
            data=st.session_state.report_bytes,
            file_name="KG_Lab_Report_%s.pdf" % roll_text,
            mime="application/pdf",
            type="primary",
        )


# ======================================================================
#  Main
# ======================================================================
def main() -> None:
    init_state()
    render_topbar()

    page = st.session_state.page
    try:
        if page == SECTIONS[0]:
            render_theory()
        elif page == SECTIONS[1]:
            render_simulation()
        elif page == SECTIONS[2]:
            render_quiz()
        else:
            render_report()
    except Exception as exc:  # last-resort guard: never lose the interface
        st.error(
            "Something went wrong while drawing this section: %s: %s"
            % (type(exc).__name__, exc)
        )
        st.caption(
            "Your schema, data and trials are safe in the session. Switch sections or "
            "reload the page to continue."
        )


main()
