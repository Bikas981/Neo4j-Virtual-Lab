"""
app.py
======
Virtual Lab: Knowledge Graph Schema Design & Data Import
(Experiment 9 -- Design a Knowledge Graph Schema and Import Data)

Run with:  streamlit run app.py

The laboratory runs entirely on its built-in Local Simulation engine, which
stores the knowledge graph in memory. No database installation or connection is
required, and the query language of the engine is an internal implementation
detail that the student never has to see.

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
import streamlit.components.v1 as components

import domains
import graph_viz
import quiz_bank
import report_generator
import schema_tools
from simulation_engine import (
    SIMULATION_MODE,
    InMemoryGraph,
    QueryResult,
    execute_cypher,
)

APP_TITLE = "Virtual Lab: Design a Knowledge Graph Schema and Import Data"
APP_SUBTITLE = "Design, construct, import, query and analyze a domain-specific knowledge graph"

SECTIONS = [
    "Purpose",
    "Theory",
    "Simulation",
    "Quiz",
    "Report Generation",
    "Certificate",
    "References",
]

PROGRESS_STEPS = [
    ("purpose_done", "Purpose"),
    ("theory_done", "Theory"),
    ("schema_done", "Schema Design"),
    ("import_done", "Data Import"),
    ("query_done", "Query Execution"),
    ("quiz_done", "Quiz"),
    ("report_done", "Report"),
    ("certificate_done", "Certificate"),
]

st.set_page_config(
    page_title="KG Virtual Lab",
    layout="wide",
    initial_sidebar_state="expanded",
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


def init_state() -> None:
    defaults = {
        "page": SECTIONS[0],
        "purpose_done": False,
        "theory_done": False,
        "schema_done": False,
        "import_done": False,
        "query_done": False,
        "quiz_done": False,
        "report_done": False,
        "certificate_done": False,
        "query_history": [],
        "trials": [],
        "quiz_answers": {},
        "quiz_result": None,
        "quiz_attempts": 0,
        # Bumped on every retake so each attempt gets fresh widget keys
        # instead of deleting the old ones while they may still be rendered.
        "quiz_round": 0,
        "quiz_unanswered": 0,
        "last_query_frame": None,
        "last_result": None,
        "report_bytes": None,
        "report_error": None,
        "certificate_bytes": None,
        "certificate_error": None,
        "conclusion": "",
        "student": {
            "name": "",
            "roll": "",
            "department": "",
            "semester": "",
            "date": datetime.now().strftime("%Y-%m-%d"),
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.get("page") not in SECTIONS:
        st.session_state.page = SECTIONS[0]
    if (
        "nav_selection" not in st.session_state
        or st.session_state.nav_selection not in SECTIONS
    ):
        st.session_state.nav_selection = st.session_state.page

    if "schema" not in st.session_state:
        load_domain(domains.DEFAULT_DOMAIN)


def current_mode() -> str:
    """The laboratory always executes on its built-in simulation engine."""
    return SIMULATION_MODE


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
    """Graph data for the visualisations, taken from the simulation engine."""
    return st.session_state.graph.snapshot()


def active_stats() -> Dict[str, Any]:
    graph: InMemoryGraph = st.session_state.graph
    stats = graph.stats()
    stats["label_counts"] = graph.label_counts()
    stats["relationship_type_counts"] = graph.rel_type_counts()
    return stats


def run_graph_query(query: str) -> QueryResult:
    """Run a query built by the interface against the simulation engine.

    The engine's query language is an internal implementation detail: the
    interface builds the statement and never shows it to the student.
    """
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
            "%d graph quer(y/ies) were executed in %s mode: %d succeeded and %d failed."
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
    """A plain checklist -- one markdown block, no emoji noise."""
    return "  \n".join(
        ("[x] %s" if done else "[ ] %s") % label for label, done in items
    )


def _goto_section(section: str) -> None:
    st.session_state.pending_nav = section
    st.session_state.page = section
    st.rerun()


def render_sidebar() -> None:
    """Left sidebar navigation with Lab Navigator heading and Progress Tracker."""
    with st.sidebar:
        st.subheader("Lab Navigator")

        if "pending_nav" in st.session_state:
            target = st.session_state.pop("pending_nav")
            st.session_state.page = target
            st.session_state.nav_selection = target
        elif (
            "nav_selection" not in st.session_state
            or st.session_state.nav_selection not in SECTIONS
        ):
            st.session_state.nav_selection = st.session_state.page

        def _handle_nav_change():
            st.session_state.page = st.session_state.nav_selection

        selected = st.radio(
            "Lab Navigation",
            SECTIONS,
            key="nav_selection",
            on_change=_handle_nav_change,
            label_visibility="collapsed",
        )
        st.session_state.page = selected

        st.divider()

        # Progress Tracker
        st.markdown("**Progress Tracker**")
        steps = [
            (key, label, bool(st.session_state.get(key)))
            for key, label in PROGRESS_STEPS
        ]
        done = sum(1 for _, _, complete in steps if complete)
        pct = done / len(steps)
        st.progress(pct)
        st.caption(f"{done} of {len(steps)} stages complete ({int(pct * 100)}%)")

        with st.expander("Milestones", expanded=False):
            for _, label, complete in steps:
                status_mark = "[x]" if complete else "[ ]"
                st.markdown(f"`{status_mark}` {label}")


def render_main_header() -> None:
    """Header across the top of the main content area."""
    title_col, status_col = st.columns([3, 1], vertical_alignment="center")
    with title_col:
        st.title(APP_TITLE)
        st.caption("%s · Experiment 9" % APP_SUBTITLE)
    with status_col:
        st.info("**Mode:** Local Simulation")
        st.caption("Built-in simulation engine.")
    st.divider()


# ======================================================================
#  1. PURPOSE
# ======================================================================
def render_purpose() -> None:
    st.header("Purpose & Learning Objectives")
    st.caption("Overview, aims, competencies, and experimental roadmap for Experiment 9.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "Property Graph", help="Labeled Property Graph (LPG) model")
    col2.metric("Engine", "In-Memory Simulation", help="Zero-install embedded local graph simulation")
    col3.metric("Key Competency", "Schema & Ingestion", help="Entity-relationship translation to graph topology")

    st.subheader("1. Experiment Aim & Context")
    st.markdown(
        """
**Aim:** To design a domain-specific Knowledge Graph schema using the **Labeled Property Graph (LPG)** model,
validate and import structured tabular entity-relationship data into an in-memory graph engine, execute traversal
queries, and analyze network topology and connectivity patterns.

**Why Knowledge Graphs?**  
Traditional relational databases organize information into rigid, flat tables joined dynamically at query time using foreign keys.
When data exhibits rich, multi-hop interconnectedness (e.g. academic networks, healthcare interactions, recommendation systems, or enterprise knowledge silos),
relational queries incur severe combinatorial join penalties.

In contrast, **Knowledge Graphs** store entities as **nodes** and connections as first-class **relationships** carrying their own types,
directions, and properties. Because relationships are indexed directly on the nodes they connect, traversing a relationship is a local pointer-chasing operation
independent of the total database size.
        """
    )

    st.subheader("2. Learning Objectives")
    st.markdown("Upon successful completion of this virtual laboratory experiment, students will be able to:")
    obj_cols = st.columns(2)
    for idx, obj in enumerate(report_generator.LEARNING_OBJECTIVES, start=1):
        target_col = obj_cols[0] if idx <= 5 else obj_cols[1]
        with target_col:
            st.markdown(f"**{idx}.** {obj}")

    st.subheader("3. Experimental Procedure Roadmap")
    st.markdown("Follow this systematic 5-phase procedure through the virtual lab workflow:")

    phases = [
        (
            "Phase I: Foundational Concepts",
            [
                "Study the theory of knowledge graphs, nodes, labels, and property graph representations.",
                "Understand the trade-offs between relational database tables and graph databases.",
            ],
        ),
        (
            "Phase II: Domain Selection & Schema Design",
            [
                "Select an academic or industry domain (University, Healthcare, E-Commerce, Movies, Library).",
                "Identify entities (nouns) and define distinct node labels with unique identifier keys.",
                "Identify relationships (verbs) and define directed types with appropriate properties.",
                "Use the visual Schema Designer and interactive Schema Diagram to validate model integrity.",
            ],
        ),
        (
            "Phase III: Data Validation & Ingestion",
            [
                "Inspect structured tabular datasets (nodes.csv and relationships.csv).",
                "Perform automated integrity validation (checking for missing headers, duplicate keys, orphaned relations).",
                "Execute a two-pass data import into the built-in in-memory simulation engine.",
            ],
        ),
        (
            "Phase IV: Graph Traversal & Analysis",
            [
                "Execute graph queries to retrieve entities, filter by properties, and traverse multi-hop connections.",
                "Analyze interactive network topology visualizations using graph filters.",
                "Record experimental trials and observations in the Experimental Data Logbook.",
            ],
        ),
        (
            "Phase V: Assessment, Reporting & Certification",
            [
                "Complete the 15-question comprehensive assessment quiz across Basic, Intermediate, and Advanced tiers.",
                "Generate and download the formal 13-section laboratory PDF report.",
                "Earn and download the verified landscape Certificate of Lab Completion.",
            ],
        ),
    ]

    for title, steps in phases:
        with st.expander(title, expanded=True):
            for step in steps:
                st.markdown(f"- {step}")

    st.subheader("4. Target Audience & Prerequisites")
    st.info(
        "**Prerequisites:** Basic knowledge of database concepts (tables, primary/foreign keys) and basic familiarity with CSV file formats. "
        "No prior graph database or query language experience is required."
    )

    st.divider()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Mark Purpose as Reviewed", type="primary", key="mark_purpose_done"):
            st.session_state.purpose_done = True
            st.success("Purpose marked as reviewed! You can now proceed to Theory.")
        if st.session_state.purpose_done:
            st.caption("[Done] Purpose is marked as reviewed.")
    with col_b:
        if st.button("Proceed to Theory", use_container_width=True, key="goto_theory_from_purpose"):
            _goto_section("Theory")
            st.rerun()


# ======================================================================
#  2. THEORY
# ======================================================================
def render_theory() -> None:
    st.header("Theory")
    st.caption("Background reading for the experiment. Each tab covers one topic.")

    tabs = st.tabs(
        [
            "Knowledge Graphs",
            "Graph Database Fundamentals",
            "Schema Design",
            "Design Principles",
            "Data Import",
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
A node is how a graph stores one entity. In this lab the entity "the student with
id S001" becomes a node with the label **Student** and the properties
`student_id = S001`, `name = Aditi`, `semester = 4`.

**What is a relationship (an edge)?**
A relationship joins exactly two nodes, has a direction and a single **type** that
names the connection. Reading it out loud should form a sentence:
*a Student is enrolled in a Course.*
            """
        )
        st.code("Student  --[ENROLLED_IN]-->  Course", language="text")
        st.markdown(
            """
**What is a property?**
A property is a key-value fact stored on a node or on a relationship, such as
`name = Aditi`, `credits = 4` or `grade = A`.
            """
        )
        st.subheader("Relational database vs graph database")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Aspect": "Basic unit",
                        "Relational (tables)": "Row in a table",
                        "Graph database": "Node with labels and properties",
                    },
                    {
                        "Aspect": "How things connect",
                        "Relational (tables)": "Foreign keys, joined at query time",
                        "Graph database": "Stored relationships, traversed directly",
                    },
                    {
                        "Aspect": "Cost of one more hop",
                        "Relational (tables)": "Another join; cost grows with table size",
                        "Graph database": "A local step from the node you are on",
                    },
                    {
                        "Aspect": "Many-to-many",
                        "Relational (tables)": "Extra junction table",
                        "Graph database": "Just another relationship",
                    },
                    {
                        "Aspect": "Schema",
                        "Relational (tables)": "Fixed columns, declared up front",
                        "Graph database": "Flexible; labels and properties can differ per node",
                    },
                    {
                        "Aspect": "Best at",
                        "Relational (tables)": "Aggregating large uniform tables",
                        "Graph database": "Following connections, paths and patterns",
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.info(
            "**Why graph storage suits connected data:** each node keeps its own "
            "relationships, so the cost of following a connection does not depend on "
            "how big the dataset is. A question such as 'which students share a course "
            "with Aditi?' is two hops in a graph, but two joins over potentially huge "
            "tables in a relational database."
        )

    with tabs[1]:
        st.subheader("Graph Database Fundamentals")
        st.markdown(
            """
A knowledge graph is usually stored using the **property graph model**. It has four
building blocks:

| Concept | Meaning | Example |
|---|---|---|
| **Node** | One entity | a particular student |
| **Label** | A category for a node; a node may have several | `Student`, `Course` |
| **Relationship** | A directed connection between exactly two nodes | Student → Course |
| **Relationship type** | The single name a relationship carries | `ENROLLED_IN` |
| **Property** | A key-value fact on a node *or* on a relationship | `name = Aditi`, `grade = A` |

The rules of the model are short:

1. A node may carry zero, one or several labels.
2. A relationship always has a direction, exactly one type, a start node and an
   end node.
3. Both nodes and relationships may carry properties.
4. The relationship itself is the connection; no separate join structure is needed.
            """
        )
        st.code(
            "  Student                ENROLLED_IN                Course\n"
            "  node + label     relationship + type + property   node + label\n"
            "  {student_id,           {grade: A}                 {course_id,\n"
            "   name, semester}                                   name, credits}",
            language="text",
        )
        st.markdown(
            """
**The property graph model in one sentence:** *nodes carry labels and properties,
relationships carry one type, a direction and properties, and every node knows its
own relationships.*

**Direction carries meaning.** A relationship is always stored with a direction,
so `Student → ENROLLED_IN → Course` and the reverse are different statements. When
a question does not care about direction, the connection can still be followed
either way.
            """
        )
        st.caption(
            "This laboratory stores the graph in its own in-memory simulation engine, "
            "so the model above can be designed, imported and explored without "
            "installing any database software."
        )

    with tabs[2]:
        st.subheader("Knowledge Graph Schema: a worked example")
        st.markdown(
            "A schema answers three questions: **what are the entities**, "
            "**how are they connected**, and **what facts do we keep about each**. "
            "Here is the University schema used as the default in this lab."
        )
        st.markdown("**Node labels**")
        st.code("Student\nCourse\nFaculty\nDepartment\nProject", language="text")
        st.markdown("**Relationship types**")
        st.code(
            "ENROLLED_IN\nTEACHES\nBELONGS_TO\nWORKS_ON\nGUIDED_BY", language="text"
        )
        st.markdown("**Patterns**")
        st.code(
            "Student  --[ENROLLED_IN]-->  Course\n"
            "Faculty  --[TEACHES]-->      Course\n"
            "Student  --[BELONGS_TO]-->   Department\n"
            "Student  --[WORKS_ON]-->     Project\n"
            "Faculty  --[GUIDED_BY]-->    Project",
            language="text",
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
            "Student  --[ENROLLED_IN {grade: A}]-->       Course\n"
            "Student  --[WORKS_ON {role: Developer}]-->   Project",
            language="text",
        )
        st.warning(
            "**Direction reads as meaning.** This lab keeps `Faculty --[GUIDED_BY]--> "
            "Project` because that is the pattern given in the syllabus, but notice "
            "that it reads backwards: a project is guided by a faculty member, so "
            "`Project --[GUIDED_BY]--> Faculty` would be the more natural direction. "
            "Either works, as long as questions follow the direction you chose. Try "
            "reversing it in the Schema Designer and watch the diagram update."
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
5. **Select useful properties.** Keep the facts you will use. A fact that depends on
   *both* endpoints (a grade, a rating, a role) belongs on the relationship, not on
   either node.
6. **Give every entity a unique identifier.** `student_id`, `course_id`. The import
   uses this property to decide whether an entity already exists.
7. **Avoid duplicate entities.** Import by matching on the unique identifier, so
   that importing the same data twice updates the entity instead of creating a
   second copy of it.
8. **Stay consistent.** One naming style, one direction convention, the same
   property name for the same fact everywhere in the graph.
            """
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Poor design**")
            st.code("Data {type: student, course: C101}", language="text")
            st.caption("The meaning is hidden inside properties, so nothing can be traversed.")
        with col2:
            st.markdown("**Good design**")
            st.code("Student  --[ENROLLED_IN]-->  Course", language="text")
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

Mapping tabular data onto a graph follows a direct correspondence:

| Structured (table) concept | Knowledge graph concept |
|---|---|
| Table (e.g. STUDENT) | Node label (`Student`) |
| One row of that table | One node |
| Column | Property |
| Primary key | Unique identifier property |
| Foreign key | Relationship |
| Junction table (e.g. ENROLMENT) | Relationship, with its extra columns as relationship properties |

The import happens in **two passes**: first every node is created, then the
relationships are created between nodes that already exist. A relationship row
whose `source_id` or `target_id` does not exist must be rejected, because the
relationship would have nothing to connect.
            """
        )
        st.code(
            "nodes.csv\n"
            "label,id,student_id,name,semester,course_id,credits\n"
            "Student,S001,S001,Aditi,4,,\n"
            "Course,C101,,Database Management Systems,,C101,4\n"
            "\n"
            "relationships.csv\n"
            "source_id,type,target_id,grade\n"
            "S001,ENROLLED_IN,C101,A",
            language="text",
        )
        st.info(
            "**Duplicate prevention.** Each entity is imported by matching on its "
            "unique identifier: if an entity with that identifier is already in the "
            "graph it is updated, otherwise it is created. Importing the same file "
            "twice therefore leaves one copy of every entity - which you can verify "
            "yourself in the Data Import tab."
        )
        st.warning(
            "Validate the data before importing it. Blank identifiers, duplicated "
            "identifiers, and relationships that point at entities which do not exist "
            "are the three faults that most often corrupt a graph."
        )

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Mark theory as studied", type="primary", key="btn_mark_theory"):
            st.session_state.theory_done = True
            st.success("Theory marked as complete! Proceed to the Simulation section.")
        if st.session_state.theory_done:
            st.caption("[Done] Theory is marked as studied.")
    with col2:
        if st.button("Proceed to Simulation", use_container_width=True, key="goto_sim_from_theory"):
            _goto_section("Simulation")
            st.rerun()


# ======================================================================
#  2. SIMULATION
# ======================================================================
def render_simulation() -> None:
    st.header("Simulation")

    tabs = st.tabs(
        [
            "Schema Studio",
            "Data Import & Graph View",
            "Logbook",
        ]
    )

    with tabs[0]:
        render_schema_studio_tab()
    with tabs[1]:
        render_data_import_and_graph_tab()
    with tabs[2]:
        render_logbook_tab()

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Mark Simulation as Completed", type="primary", key="btn_mark_sim"):
            st.session_state.schema_done = True
            st.session_state.import_done = True
            st.session_state.query_done = True
            st.success("Simulation marked as complete! Proceed to the Quiz section.")
        sim_done = (
            st.session_state.get("schema_done")
            and st.session_state.get("import_done")
        )
        if sim_done:
            st.caption("[Done] Simulation is marked as completed.")
    with col2:
        if st.button("Proceed to Quiz", use_container_width=True, key="goto_quiz_from_sim"):
            _goto_section("Quiz")


# ------------------------------------------------- Tab 1: Schema Studio
def _on_domain_change() -> None:
    """Switching domain reloads its schema, data and examples."""
    chosen = st.session_state.get("domain_select")
    if chosen and chosen != st.session_state.domain:
        load_domain(chosen)


def render_schema_studio_tab() -> None:
    st.subheader("Domain Selection & Schema Modeling")
    st.caption(
        "Select a target domain, define or customize node labels and relationship types, "
        "and observe the live schema diagram update in real time."
    )

    names = domains.domain_names()
    st.selectbox(
        "Selected Domain",
        names,
        index=names.index(st.session_state.domain) if st.session_state.domain in names else 0,
        key="domain_select",
        on_change=_on_domain_change,
    )
    domain = domains.get_domain(st.session_state.domain)
    st.markdown(f"**Domain Context:** {domain['description']}")
    st.info("**Entity Guidance:** " + domain["entity_hint"])

    stats = active_stats()
    schema_node_count = len(st.session_state.schema.get("nodes", []))
    schema_rel_count = len(st.session_state.schema.get("relationships", []))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Domain", st.session_state.domain)
    c2.metric("Schema Labels", schema_node_count, help="Node labels defined in this schema")
    c3.metric("Relationship Types", schema_rel_count, help="Relationship types defined in this schema")
    c4.metric(
        "Nodes in Active Graph",
        stats["node_count"],
        delta="Imported" if stats["node_count"] > 0 else "Pending Import",
        delta_color="normal" if stats["node_count"] > 0 else "off",
        help="Nodes created in the active graph engine. Requires data import.",
    )

    if stats["node_count"] == 0:
        st.caption(
            "Note: The graph database is currently empty (0 nodes). "
            "After designing your schema below, switch to the **Data Import & Graph View** tab to ingest data."
        )

    st.divider()

    # 2. NODE AND RELATIONSHIP SIDE BY SIDE (in the middle)
    st.subheader("Schema Design: Entities & Relationships")
    st.caption("Define node labels (entities) on the left and relationship types (connections) on the right.")

    col_nodes, col_rels = st.columns([1, 1], gap="large")
    with col_nodes:
        render_node_labels_section()
    with col_rels:
        render_relationship_types_section()

    # Schema Validation and actions right below the editor
    st.divider()
    st.markdown("#### Schema Validation & Actions")
    schema = st.session_state.schema
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
    col1.button("Reset schema to domain default", on_click=_reset_schema)
    col2.button("Clear the whole schema", on_click=_clear_schema)

    st.divider()

    # 3. GRAPH BELOW (Full-width Live Schema Diagram)
    col_hdr, col_view = st.columns([3, 1])
    with col_hdr:
        st.subheader("Live Schema Diagram")
        st.caption(
            "Interactive diagram generated from your active schema above. Updates automatically as you edit."
        )
    with col_view:
        schema_viz_mode = st.radio(
            "Schema Visualizer",
            ["Animated Flow", "Static Plotly"],
            horizontal=True,
            key="schema_viz_mode",
            label_visibility="collapsed",
        )
    if errors:
        st.warning(
            "The schema has %d problem(s); the diagram shows only the parts that are "
            "currently valid." % len(errors)
        )
    if schema_viz_mode == "Animated Flow":
        html_code = graph_viz.animated_graph_html(
            schema, dark=is_dark_theme(), height=520, mode="schema"
        )
        components.html(html_code, height=540)
    else:
        figure = graph_viz.schema_figure(schema, dark=is_dark_theme())
        st.plotly_chart(figure, use_container_width=True, key="schema_chart")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**Entities (Node Labels) Defined**")
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
    with col_t2:
        st.markdown("**Relationships Defined**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Pattern": "%s --[%s]--> %s" % (r.get("source"), r.get("type"), r.get("target")),
                        "Properties": ", ".join(r.get("properties", [])) or "-",
                    }
                    for r in schema["relationships"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )




def render_domain_tab() -> None:
    """Backward compatibility alias."""
    render_schema_studio_tab()



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


def render_node_labels_section() -> None:
    st.markdown("### Node Labels (Entities)")
    st.caption("Define the entities, their unique identifier key, and properties.")
    schema = st.session_state.schema

    if not schema["nodes"]:
        st.warning("No node labels defined. Add one below to begin.")

    for node in list(schema["nodes"]):
        node_id = node["_id"]
        with st.expander(":%s" % (node["label"] or "(unnamed)"), expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                node["label"] = st.text_input(
                    "Node label", value=node["label"], key="nlabel_%s" % node_id
                ).strip()
            with col2:
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

            raw = st.text_input(
                "Properties (comma separated)",
                value=", ".join(node["properties"]),
                key="nprops_%s" % node_id,
            )
            node["properties"] = [p.strip() for p in raw.split(",") if p.strip()]

            st.button(
                "Remove label",
                key="ndel_%s" % node_id,
                on_click=_remove_node,
                args=(node_id,),
            )

    with st.form("add_node_form", clear_on_submit=True):
        st.markdown("**Add a node label**")
        col1, col2 = st.columns(2)
        col1.text_input("Label", placeholder="Student", key="add_node_label")
        col2.text_input("Unique ID property", placeholder="student_id", key="add_node_key")
        st.text_input(
            "Properties (comma separated)",
            placeholder="student_id, name, semester",
            key="add_node_props",
        )
        st.form_submit_button("Add node label", type="primary", on_click=_add_node)
    _show_designer_message("node")


def render_relationship_types_section() -> None:
    st.markdown("### Relationship Types")
    st.caption("Connect entity types with typed, directed relationships.")
    schema = st.session_state.schema
    labels = [n["label"] for n in schema["nodes"] if n["label"]]

    if not schema["relationships"]:
        st.warning("No relationships defined yet.")

    for rel in list(schema["relationships"]):
        rel_id = rel["_id"]
        title = "%s --[%s]--> %s" % (
            rel.get("source") or "?",
            rel.get("type") or "?",
            rel.get("target") or "?",
        )
        with st.expander(title, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                rel["type"] = st.text_input(
                    "Relationship type", value=rel["type"], key="rtype_%s" % rel_id
                ).strip()
            with col2:
                raw = st.text_input(
                    "Properties (comma separated)",
                    value=", ".join(rel.get("properties", [])),
                    key="rprops_%s" % rel_id,
                )
                rel["properties"] = [p.strip() for p in raw.split(",") if p.strip()]

            options = list(labels)
            for referenced in (rel.get("source"), rel.get("target")):
                if referenced and referenced not in options:
                    options.append(referenced)
            if not options:
                options = ["(no node labels yet)"]

            col3, col4 = st.columns(2)
            with col3:
                index = options.index(rel["source"]) if rel.get("source") in options else 0
                rel["source"] = st.selectbox(
                    "Source node label", options, index=index, key="rsrc_%s" % rel_id
                )
            with col4:
                index = options.index(rel["target"]) if rel.get("target") in options else 0
                rel["target"] = st.selectbox(
                    "Target node label", options, index=index, key="rtgt_%s" % rel_id
                )

            st.button(
                "Remove relationship",
                key="rdel_%s" % rel_id,
                on_click=_remove_relationship,
                args=(rel_id,),
            )

    with st.form("add_rel_form", clear_on_submit=True):
        st.markdown("**Add a relationship type**")
        col1, col2 = st.columns(2)
        col1.text_input("Type", placeholder="ENROLLED_IN", key="add_rel_type")
        col2.text_input("Properties", placeholder="grade", key="add_rel_props")
        col3, col4 = st.columns(2)
        col3.selectbox(
            "Source label", labels or ["(add a node label first)"], key="add_rel_source"
        )
        col4.selectbox(
            "Target label", labels or ["(add a node label first)"], key="add_rel_target"
        )
        st.form_submit_button("Add relationship", type="primary", on_click=_add_relationship)
    _show_designer_message("rel")


def render_schema_designer_content() -> None:
    """Side-by-side node and relationship sections."""
    col_nodes, col_rels = st.columns([1, 1], gap="large")
    with col_nodes:
        render_node_labels_section()
    with col_rels:
        render_relationship_types_section()


def render_schema_designer() -> None:
    """Backward compatibility alias."""
    render_schema_designer_content()


# ----------------------------------------------- Tab 1 (Right): Schema Diagram
def render_schema_diagram_content() -> None:
    st.subheader("Live Schema Diagram")
    st.caption(
        "Interactive diagram generated from your active schema. Updates automatically as you edit."
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

    with st.expander("Defined Node Labels & Keys", expanded=False):
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
    with st.expander("Defined Relationship Types", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Pattern": "%s --[%s]--> %s" % (r.get("source"), r.get("type"), r.get("target")),
                        "Properties": ", ".join(r.get("properties", [])) or "-",
                    }
                    for r in schema["relationships"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def render_schema_graph() -> None:
    """Backward compatibility alias."""
    render_schema_diagram_content()


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

    # ---- run the import --------------------------------------------
    st.divider()
    st.subheader("Step 3 — Import into the graph")
    st.caption(
        "The data is written into the in-memory simulation graph of this laboratory."
    )
    prevent_duplicates = st.checkbox(
        "Prevent duplicate entities (match each entity on its unique identifier)",
        value=True,
        key="opt_prevent_duplicates",
        help=(
            "When enabled, importing the same data twice updates the existing "
            "entities instead of creating a second copy of each of them."
        ),
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Import into the graph", type="primary", key="do_import"):
            _run_import(prevent_duplicates)
    with col2:
        if st.button("Clear the imported graph", key="clear_graph"):
            st.session_state.graph = InMemoryGraph()
            st.success("The simulation graph was cleared.")
            st.session_state.import_status = "Cleared"
            st.session_state.import_done = False

    if st.session_state.import_summary:
        summary = st.session_state.import_summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes imported", summary.get("nodes_created", 0))
        col2.metric("Relationships imported", summary.get("relationships_created", 0))
        col3.metric("Properties set", summary.get("properties_set", 0))
        st.caption("Status: %s" % st.session_state.import_status)


def _run_import(prevent_duplicates: bool = True) -> None:
    """Validate the schema and data, then load the dataset into the graph.

    The loading statements are produced internally by schema_tools and executed
    by the simulation engine; they are never shown to the student.
    """
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

    script = schema_tools.generate_cypher(schema, dataset, use_merge=prevent_duplicates)
    st.session_state.generated_cypher = script

    started = time.perf_counter()
    graph, totals, errors = schema_tools.build_local_graph(script, st.session_state.graph)
    st.session_state.graph = graph
    elapsed = (time.perf_counter() - started) * 1000

    st.session_state.import_summary = totals
    if errors:
        st.session_state.import_status = "Imported with %d error(s)" % len(errors)
        st.warning(
            "%d record(s) could not be imported. The remaining data was imported."
            % len(errors)
        )
    else:
        st.session_state.import_status = "Imported successfully (Local Simulation)"
        st.success(
            "Import finished in %.0f ms - nodes imported: %d, relationships imported: %d."
            % (elapsed, totals.get("nodes_created", 0), totals.get("relationships_created", 0))
        )
        if totals.get("nodes_created", 0) == 0 and totals.get("relationships_created", 0) == 0:
            st.info(
                "Nothing new was created because every entity in the dataset was already "
                "present in the graph - that is exactly the duplicate prevention you are "
                "testing."
            )
    st.session_state.import_done = True


# -------------------------------------------------- Tab: Graph Query
# The student composes a question from the schema of the graph. The statement
# handed to the simulation engine is built here and is never displayed: the
# engine's query language is an internal implementation detail.
QUERY_KINDS = [
    "List all entities of a type",
    "Find entities by property value",
    "Show relationships of a type",
    "Show the connections of one entity",
    "Count entities of each type",
]

OPERATORS = {
    "is equal to": "=",
    "is not equal to": "<>",
    "is greater than": ">",
    "is less than": "<",
    "contains": "CONTAINS",
    "starts with": "STARTS WITH",
}


def _literal(value: str) -> str:
    """Quote a value for the engine, keeping numbers numeric."""
    text = (value or "").strip()
    try:
        return str(int(text))
    except ValueError:
        pass
    try:
        return str(float(text))
    except ValueError:
        pass
    return "'%s'" % text.replace("\\", "\\\\").replace("'", "\\'")


def _properties_of(label: str) -> List[str]:
    """Property names of a label, from the graph itself."""
    names: List[str] = []
    for node in st.session_state.graph.find_nodes([label]):
        for key in node.properties:
            if key not in names:
                names.append(key)
    return names


def _entity_rows(nodes: List[Any]) -> pd.DataFrame:
    rows = []
    for node in nodes:
        row = {"Entity type": node.label}
        row.update({k: v for k, v in node.properties.items()})
        rows.append(row)
    return pd.DataFrame(rows).fillna("").astype(str)


def _run_and_report(statement: str, description: str, formatter) -> None:
    """Execute a built statement, then present the rows in neutral columns."""
    result = run_graph_query(statement)
    result.query = description
    st.session_state.last_result = result
    remember_query(result)
    if not result.ok:
        st.error(
            "The query could not be completed. Import the dataset in the **Data "
            "Import** tab and make sure the entity type still exists in the graph."
        )
        return
    frame = formatter(result)
    st.session_state.last_query_frame = frame
    st.success(
        "Query executed successfully - %d result(s) in %.1f ms."
        % (len(frame), result.execution_ms)
    )
    if frame.empty:
        st.info(
            "No results matched. Check the entity type, the property and the value "
            "you selected."
        )
    else:
        st.markdown("**Results**")
        st.dataframe(frame, width="stretch", hide_index=True)
        st.download_button(
            "Download results CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="graph_query_results.csv",
            mime="text/csv",
        )


def render_query_tab() -> None:
    st.subheader("Graph Query")
    st.caption(
        "Ask questions of the knowledge graph you imported. Queries are executed by "
        "the built-in simulation engine."
    )

    graph: InMemoryGraph = st.session_state.graph
    labels = graph.labels()
    rel_types = graph.rel_types()
    if not labels:
        st.info(
            "The graph is empty. Open the **Data Import** tab, validate the dataset "
            "and import it, then return here."
        )
        return

    kind = st.selectbox("Question", QUERY_KINDS, key="query_kind")

    # ---- List all entities of a type --------------------------------
    if kind == QUERY_KINDS[0]:
        label = st.selectbox("Entity type", labels, key="q_list_label")
        if st.button("Execute Query", type="primary", key="q_list_run"):
            _run_and_report(
                "MATCH (n:%s) RETURN n" % label,
                "List all %s entities" % label,
                lambda r: _entity_rows([row[0] for row in r.rows]),
            )

    # ---- Find entities by property value -----------------------------
    elif kind == QUERY_KINDS[1]:
        col1, col2 = st.columns(2)
        label = col1.selectbox("Entity type", labels, key="q_filter_label")
        properties = _properties_of(label)
        prop = col2.selectbox(
            "Property", properties or ["(no properties)"], key="q_filter_prop"
        )
        col3, col4 = st.columns(2)
        operator = col3.selectbox("Condition", list(OPERATORS), key="q_filter_op")
        value = col4.text_input("Value", key="q_filter_value")
        if st.button("Execute Query", type="primary", key="q_filter_run"):
            if not properties:
                st.error("This entity type has no properties to filter on.")
            elif not value.strip():
                st.error("Enter a value to compare with.")
            else:
                statement = "MATCH (n:%s) WHERE n.%s %s %s RETURN n" % (
                    label,
                    prop,
                    OPERATORS[operator],
                    _literal(value),
                )
                _run_and_report(
                    statement,
                    "Find %s entities where %s %s %s" % (label, prop, operator, value),
                    lambda r: _entity_rows([row[0] for row in r.rows]),
                )

    # ---- Show relationships of a type --------------------------------
    elif kind == QUERY_KINDS[2]:
        if not rel_types:
            st.info("The graph has no relationships yet.")
            return
        rtype = st.selectbox("Relationship type", rel_types, key="q_rel_type")
        if st.button("Execute Query", type="primary", key="q_rel_run"):

            def relationship_rows(result: QueryResult) -> pd.DataFrame:
                rows = []
                for source, rel, target in result.rows:
                    row = {
                        "Source": source.caption(),
                        "Source type": source.label,
                        "Relationship": rel.rtype,
                        "Target": target.caption(),
                        "Target type": target.label,
                    }
                    row.update(rel.properties)
                    rows.append(row)
                return pd.DataFrame(rows).fillna("").astype(str)

            _run_and_report(
                "MATCH (a)-[r:%s]->(b) RETURN a, r, b" % rtype,
                "Show all %s relationships" % rtype,
                relationship_rows,
            )

    # ---- Show the connections of one entity ---------------------------
    elif kind == QUERY_KINDS[3]:
        col1, col2 = st.columns(2)
        label = col1.selectbox("Entity type", labels, key="q_conn_label")
        nodes = graph.find_nodes([label])
        captions = sorted({node.caption() for node in nodes})
        chosen = col2.selectbox(
            "Entity", captions or ["(no entities)"], key="q_conn_entity"
        )
        if st.button("Execute Query", type="primary", key="q_conn_run"):
            node = next((n for n in nodes if n.caption() == chosen), None)
            if node is None:
                st.error("Select an entity to inspect.")
            else:
                key_property = next(
                    (k for k in node.properties if k.endswith("_id") or k == "id"), None
                )
                if key_property:
                    pattern = "(a:%s {%s: %s})" % (
                        label,
                        key_property,
                        _literal(str(node.properties[key_property])),
                    )
                else:
                    pattern = "(a:%s)" % label
                statement = "MATCH %s-[r]-(b) RETURN r, b, a" % pattern

                def connection_rows(result: QueryResult) -> pd.DataFrame:
                    rows = []
                    for rel, other, origin in result.rows:
                        outgoing = rel.start == origin.nid
                        rows.append(
                            {
                                "Direction": "outgoing" if outgoing else "incoming",
                                "Relationship": rel.rtype,
                                "Connected entity": other.caption(),
                                "Entity type": other.label,
                            }
                        )
                    return pd.DataFrame(rows).fillna("").astype(str)

                _run_and_report(
                    statement,
                    "Show the connections of %s (%s)" % (chosen, label),
                    connection_rows,
                )

    # ---- Count entities of each type ----------------------------------
    else:
        if st.button("Execute Query", type="primary", key="q_count_run"):
            counts = []
            total_ms = 0.0
            failed = False
            for label in labels:
                result = run_graph_query("MATCH (n:%s) RETURN count(n) AS total" % label)
                total_ms += result.execution_ms
                if not result.ok:
                    failed = True
                    break
                counts.append({"Entity type": label, "Entities": result.rows[0][0]})
            summary = QueryResult(
                columns=["Entity type", "Entities"],
                rows=[[c["Entity type"], c["Entities"]] for c in counts],
                execution_ms=total_ms,
                mode=SIMULATION_MODE,
                query="Count the entities of each type",
                error="count failed" if failed else None,
            )
            st.session_state.last_result = summary
            remember_query(summary)
            if failed:
                st.error("The query could not be completed.")
            else:
                frame = pd.DataFrame(counts)
                st.session_state.last_query_frame = frame
                st.success(
                    "Query executed successfully - %d result(s) in %.1f ms."
                    % (len(frame), total_ms)
                )
                st.markdown("**Results**")
                st.dataframe(frame, width="stretch", hide_index=True)

    if st.session_state.query_history:
        with st.expander("Query history (%d)" % len(st.session_state.query_history)):
            st.dataframe(
                pd.DataFrame(st.session_state.query_history)[
                    ["time", "query", "ok", "records", "ms"]
                ].rename(
                    columns={
                        "time": "Time",
                        "query": "Query",
                        "ok": "Succeeded",
                        "records": "Results",
                        "ms": "Time (ms)",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


# --------------------------------------------------- Tab: Graph View
def render_graph_tab() -> None:
    col_hdr, col_view = st.columns([3, 1])
    with col_hdr:
        st.subheader("Graph visualization")
    with col_view:
        inst_viz_mode = st.radio(
            "Graph Visualizer",
            ["Animated Flow", "Static Plotly"],
            horizontal=True,
            key="inst_viz_mode",
            label_visibility="collapsed",
        )

    snapshot = active_snapshot()
    node_count = len(snapshot["nodes"])

    if node_count == 0:
        st.info(
            "The graph database is currently empty. Complete Step 3 above by clicking "
            "**'Import into the graph'** to populate and view your graph network."
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

    if inst_viz_mode == "Animated Flow":
        filtered_snapshot = {
            "nodes": [n for n in snapshot["nodes"] if n["label"] in label_filter],
            "relationships": [
                r for r in snapshot["relationships"]
                if (not type_filter or r["type"] in type_filter)
            ] if show_rels else [],
        }
        html_code = graph_viz.animated_graph_html(
            filtered_snapshot,
            dark=is_dark_theme(),
            height=540,
            mode="instance",
            show_edge_labels=show_edge_labels,
            show_captions=show_captions,
        )
        components.html(html_code, height=560)
    else:
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


# --------------------------------------------------- Tab 2: Data Import & Graph View
def render_data_import_and_graph_tab() -> None:
    st.subheader("Data Ingestion & Graph Creation")
    st.caption(
        "Load data, validate records against your schema, and import them into the graph. "
        "The live graph network appears immediately below upon import."
    )

    render_import_tab()

    st.divider()
    render_graph_tab()


# ------------------------------------------------------ Tab 3: Logbook
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
                "Graph Query": last.query if last is not None else "",
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
        "%d multiple-choice questions covering knowledge graphs, graph databases, "
        "schema design, data import and graph analysis." % len(quiz_bank.QUESTIONS)
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
        mark = "[Correct]" if detail["is_correct"] else "[Incorrect]"
        with st.expander(
            "%s  Q%d · %s — %s" % (mark, index, detail["level"], detail["question"])
        ):
            if detail["is_correct"]:
                st.success("Your answer: %s" % detail["chosen_text"])
            else:
                st.error("Your answer: %s" % detail["chosen_text"])
                st.success("Correct answer: %s" % detail["correct_text"])
            st.caption("**Why:** %s" % detail["explanation"])

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Retake Quiz", on_click=_retake_quiz, use_container_width=True)
    with col2:
        if st.button(
            "Proceed to Report Generation",
            type="primary",
            use_container_width=True,
            key="goto_report_from_quiz",
        ):
            _goto_section("Report Generation")


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

    st.divider()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.session_state.report_bytes:
            st.caption("[Done] PDF report generated and ready for submission.")
        else:
            st.caption("Generate your PDF laboratory report to record experimental results.")
    with col_b:
        if st.button(
            "Proceed to Certificate",
            type="primary" if st.session_state.report_bytes else "secondary",
            use_container_width=True,
            key="goto_cert_from_report",
        ):
            _goto_section("Certificate")


# ======================================================================
#  6. CERTIFICATE
# ======================================================================
def render_certificate() -> None:
    st.header("Certificate of Completion")
    st.caption("Official verified Certificate of Completion for Experiment 9.")

    student = st.session_state.student
    data = observations()
    quiz = st.session_state.quiz_result
    has_name = bool((student.get("name") or "").strip())
    has_imported = bool(st.session_state.import_done or data["node_count"] > 0)
    has_quiz = quiz is not None
    quiz_passed = has_quiz and (quiz.get("percentage", 0.0) >= 50.0)

    # Completion indicators
    st.subheader("Laboratory Completion Status")
    chk_cols = st.columns(4)
    chk_cols[0].metric(
        "Student Profile",
        "Complete" if has_name else "Missing",
        delta="OK" if has_name else "Action required",
    )
    chk_cols[1].metric(
        "Data Ingestion",
        f"{data['node_count']} nodes" if has_imported else "Pending",
        delta="OK" if has_imported else "Action required",
    )
    chk_cols[2].metric(
        "Assessment",
        f"{quiz['percentage']:.0f}%" if has_quiz else "Pending",
        delta="Passed" if quiz_passed else ("Pending" if not has_quiz else "Retake advised"),
    )
    chk_cols[3].metric(
        "Lab Report",
        "Generated" if st.session_state.report_done else "Pending",
        delta="OK" if st.session_state.report_done else "Recommended",
    )

    if not has_name:
        st.warning("Please provide your name and academic details to personalize your certificate:")
        with st.form("cert_student_form"):
            c1, c2 = st.columns(2)
            c_name = c1.text_input("Full Name", value=student.get("name", ""))
            c_roll = c2.text_input("Roll No / Student ID", value=student.get("roll", ""))
            c3, c4 = st.columns(2)
            c_dept = c3.text_input(
                "Department", value=student.get("department", "Computer Science & Engineering")
            )
            c_sem = c4.text_input("Semester", value=student.get("semester", "Semester IV"))
            if st.form_submit_button("Save Student Details", type="primary"):
                st.session_state.student["name"] = c_name
                st.session_state.student["roll"] = c_roll
                st.session_state.student["department"] = c_dept
                st.session_state.student["semester"] = c_sem
                st.success("Details saved! Refreshing certificate...")
                st.rerun()

    if not has_quiz:
        st.info("Tip: To validate your subject mastery, complete the quiz in the Quiz section.")
        if st.button("Go to Quiz", key="goto_quiz_from_cert"):
            _goto_section("Quiz")
            st.rerun()

    st.divider()

    # Visual Certificate Preview
    cert_id = f"VLAB-KG-9-{(student.get('roll') or 'EXP9')[:6].replace(' ', '').upper()}"
    student_display_name = (student.get("name") or "Enrolled Student").strip().title()

    st.subheader("Official Certificate Preview")

    # Render a stylized container simulating the certificate
    with st.container(border=True):
        quiz_summary_text = (
            f"{quiz['correct']}/{quiz['total']} ({quiz['percentage']:.0f}%)"
            if quiz
            else "Completed"
        )
        st.markdown(
            f"""
            <div style="text-align: center; padding: 25px; border: 3px double #1B365D; border-radius: 8px; background-color: #FAFCFF;">
                <p style="letter-spacing: 2px; font-size: 13px; color: #5E6C82; margin-bottom: 2px; text-transform: uppercase;">
                    Virtual Laboratories Project · Ministry of Education
                </p>
                <p style="font-size: 11px; color: #5E6C82; margin-top: 0px;">
                    National Knowledge Graph Systems & Graph Databases Laboratory
                </p>
                <h1 style="color: #1B365D; font-size: 28px; margin: 15px 0 5px 0; font-family: serif; letter-spacing: 1px;">
                    CERTIFICATE OF LAB COMPLETION
                </h1>
                <p style="font-style: italic; color: #5E6C82; font-size: 14px; margin: 0 0 10px 0;">
                    This is to proudly certify that
                </p>
                <h2 style="color: #10233F; font-size: 24px; margin: 5px 0 2px 0; text-decoration: underline; text-decoration-color: #B48C32;">
                    {student_display_name}
                </h2>
                <p style="color: #5E6C82; font-size: 13px; margin: 5px 0 15px 0;">
                    Roll No: <strong>{student.get('roll') or 'N/A'}</strong> &nbsp;|&nbsp; 
                    Department: <strong>{student.get('department') or 'CSE'}</strong> &nbsp;|&nbsp; 
                    <strong>{student.get('semester') or 'Semester IV'}</strong>
                </p>
                <p style="color: #10233F; font-size: 14px; max-width: 750px; margin: 0 auto 15px auto; line-height: 1.6;">
                    has successfully demonstrated practical competency in designing a domain-specific Knowledge Graph schema,
                    validating and importing structured entity-relationship data into the simulation engine,
                    executing graph traversal queries, and completing all requirements for:
                </p>
                <h3 style="color: #1B365D; font-size: 17px; margin: 10px 0 20px 0;">
                    Experiment 9: Knowledge Graph Schema Design & Data Import
                </h3>
                <div style="display: flex; justify-content: space-around; max-width: 600px; margin: 0 auto 20px auto; padding: 10px; background-color: #F3F6FB; border: 1px solid #D8E0EC; border-radius: 5px;">
                    <div><span style="font-size: 11px; color: #5E6C82; font-weight: bold;">DOMAIN</span><br><strong style="color: #1B365D;">{st.session_state.domain}</strong></div>
                    <div><span style="font-size: 11px; color: #5E6C82; font-weight: bold;">NODES</span><br><strong style="color: #1B365D;">{data['node_count']}</strong></div>
                    <div><span style="font-size: 11px; color: #5E6C82; font-weight: bold;">RELATIONSHIPS</span><br><strong style="color: #1B365D;">{data['relationship_count']}</strong></div>
                    <div><span style="font-size: 11px; color: #5E6C82; font-weight: bold;">SCORE</span><br><strong style="color: #1B365D;">{quiz_summary_text}</strong></div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 30px; padding: 0 40px;">
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 13px; font-style: italic;">Automated Simulation Engine</p>
                        <hr style="width: 140px; border: 0.5px solid #D8E0EC; margin: 4px auto;">
                        <p style="margin: 0; font-size: 11px; color: #5E6C82;">Virtual Lab Evaluator</p>
                    </div>
                    <div style="text-align: center; border: 2px dashed #B48C32; border-radius: 50%; width: 60px; height: 60px; line-height: 56px; color: #B48C32; font-weight: bold; font-size: 9px;">
                        VERIFIED
                    </div>
                    <div style="text-align: center;">
                        <p style="margin: 0; font-size: 13px; font-style: italic;">Course Faculty Coordinator</p>
                        <hr style="width: 140px; border: 0.5px solid #D8E0EC; margin: 4px auto;">
                        <p style="margin: 0; font-size: 11px; color: #5E6C82;">Department of CSE</p>
                    </div>
                </div>
                <p style="font-size: 10px; color: #5E6C82; margin-top: 25px; margin-bottom: 0;">
                    Certificate ID: <strong>{cert_id}</strong> &nbsp;·&nbsp; Issued On: <strong>{student.get('date') or datetime.now().strftime('%Y-%m-%d')}</strong> &nbsp;·&nbsp; Verify at: virtual-labs.ac.in
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # PDF Certificate Generation & Download Controls
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Generate Official Landscape PDF Certificate", type="primary", use_container_width=True):
            context = {
                "student": st.session_state.student,
                "domain": st.session_state.domain,
                "observations": data,
                "quiz": quiz,
                "cert_id": cert_id,
            }
            cert_bytes, cert_err = report_generator.build_certificate(context)
            st.session_state.certificate_bytes = cert_bytes
            st.session_state.certificate_error = cert_err
            if cert_err:
                st.error(f"Could not generate certificate: {cert_err}")
            else:
                st.session_state.certificate_done = True
                st.success("Official Certificate generated! Click download below.")

    with btn_col2:
        if st.session_state.certificate_bytes:
            filename = f"KG_Lab_Certificate_{(student.get('roll') or 'student').replace(' ', '_')}.pdf"
            st.download_button(
                "Download Certificate (PDF)",
                data=st.session_state.certificate_bytes,
                file_name=filename,
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.button("Download Certificate (PDF)", disabled=True, use_container_width=True)

    st.divider()
    ref_col1, ref_col2 = st.columns([3, 1])
    with ref_col1:
        st.caption("Looking to expand your knowledge graph skills? Explore academic books, formal GQL standards, and research papers.")
    with ref_col2:
        if st.button("Explore References", use_container_width=True, key="goto_refs_from_cert"):
            _goto_section("References")
            st.rerun()


# ======================================================================
#  7. REFERENCES
# ======================================================================
def render_references() -> None:
    st.header("References & Bibliography")
    st.caption("Curated academic textbooks, landmark survey papers, international standards, and software ecosystem.")

    st.subheader("1. Foundational Textbooks & Monographs")
    st.markdown(
        """
1. **Robinson, I., Webber, J., & Eifrem, E. (2015).**  
   *Graph Databases: New Opportunities for Connected Data* (2nd ed.). O'Reilly Media.  
   *Focus: Property graph modeling, traversal algorithms, and real-world connected data architectures.*

2. **Silberschatz, A., Korth, H. F., & Sudarshan, S. (2019).**  
   *Database System Concepts* (7th ed.). McGraw-Hill.  
   *Focus: Comparative database architectures, relational normal forms vs. semi-structured data models.*

3. **Kleppmann, M. (2017).**  
   *Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems*. O'Reilly Media.  
   *Focus: Chapter 2: Data Models and Query Languages (Document vs. Relational vs. Graph-like Data Models).*

4. **Allemang, D., & Hendler, J. (2011).**  
   *Semantic Web for the Working Ontologist: Effective Modeling in RDFS and OWL* (2nd ed.). Morgan Kaufmann.  
   *Focus: Formal knowledge representation, RDF triples, ontologies, and semantic inferencing.*
        """
    )

    st.divider()

    st.subheader("2. Landmark Academic Papers & Surveys")
    st.markdown(
        """
1. **Hogan, A., Blomqvist, E., Cochez, M., d'Amato, C., de Melo, G., Gutierrez, C., Labra Gayo, J. E., Sabrina, K., Neumaier, S., Polleres, A., Sabbir, M., & Zimmermann, A. (2021).**  
   *"Knowledge Graphs"*. **ACM Computing Surveys**, 54(4), Article 71, 1–37.  
   *Overview: Authoritative survey synthesizing graph representations, deductive/inductive knowledge, validation, and embeddings.*

2. **Angles, R., & Gutierrez, C. (2008).**  
   *"Survey of Graph Database Models"*. **ACM Computing Surveys**, 40(1), Article 1, 1–39.  
   *Overview: Foundational taxonomy contrasting hypergraphs, labeled property graphs, and semantic networks.*

3. **Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2021).**  
   *"A Survey on Knowledge Graphs: Representation, Acquisition, and Applications"*. **IEEE Transactions on Neural Networks and Learning Systems**, 33(2), 494–514.  
   *Overview: Knowledge graph construction, multi-hop reasoning, and neural graph embeddings.*

4. **Bonifati, A., Dumbrava, S., & Fletcher, G. (2020).**  
   *"Graph Query Languages: Status, Trends, and Challenges"*. **ACM SIGMOD Record**, 49(1), 5–16.  
   *Overview: Evolution of graph pattern matching, path queries, and standard query language design.*
        """
    )

    st.divider()

    st.subheader("3. Formal Standards & Specifications")
    standards_data = [
        {
            "Standard": "ISO/IEC 39075:2024 (GQL)",
            "Organization": "ISO / IEC JTC 1/SC 32",
            "Description": "The first formal international standard query language specifically for Property Graphs, supporting declarative graph pattern matching, graph types, and mutations.",
        },
        {
            "Standard": "openCypher Specification",
            "Organization": "openCypher Project",
            "Description": "Open-source specification of the Cypher property graph query language with formal EBNF grammar and TCK (Technology Compatibility Kit).",
        },
        {
            "Standard": "W3C RDF 1.1 & OWL 2",
            "Organization": "World Wide Web Consortium (W3C)",
            "Description": "Standards for semantic graph data interchange, uniform resource identifiers (URIs), triples (subject-predicate-object), and formal web ontology languages.",
        },
        {
            "Standard": "Property Graph Schema (PGS)",
            "Organization": "LDBC (Linked Data Benchmark Council)",
            "Description": "Formal schema definition language for property graphs, specifying vertex and edge types, property domains, and cardinality constraints.",
        },
    ]
    st.dataframe(pd.DataFrame(standards_data), hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("4. Software & Educational Ecosystem")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Python Graph Libraries**")
        st.markdown(
            """
- **NetworkX:** Python software for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
- **Plotly:** Interactive declarative visualization library used in this virtual lab for dynamic schema and graph network rendering.
- **Pandas:** Structured tabular manipulation library facilitating CSV validation and two-pass ingestion.
- **fpdf2:** Modern minimalist PDF engine used for building the 13-section lab report and landscape certificate.
            """
        )
    with col2:
        st.markdown("**Educational Portals & Tutorials**")
        st.markdown(
            """
- **Virtual Labs (Govt. of India):** An initiative under NMEICT providing remote-access to interactive simulation labs in science and engineering.
- **Neo4j GraphAcademy:** Interactive hands-on training courses on Graph Data Modeling, Cypher queries, and Knowledge Graph architectures.
- **Stanford CS224W (Machine Learning with Graphs):** Comprehensive university course on graph algorithms, graph neural networks (GNNs), and knowledge graphs.
            """
        )

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption("You have reached the end of the virtual laboratory curriculum. Review your report in Report Generation or download your Certificate!")
    with col2:
        if st.button("Return to Purpose / Overview", use_container_width=True, key="goto_purpose_from_refs"):
            _goto_section("Purpose")


# ======================================================================
#  Main
# ======================================================================
def main() -> None:
    init_state()
    render_sidebar()
    render_main_header()

    page = st.session_state.page
    try:
        if page == "Purpose":
            render_purpose()
        elif page == "Theory":
            render_theory()
        elif page == "Simulation":
            render_simulation()
        elif page == "Quiz":
            render_quiz()
        elif page == "Report Generation":
            render_report()
        elif page == "Certificate":
            render_certificate()
        elif page == "References":
            render_references()
        else:
            render_purpose()
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
