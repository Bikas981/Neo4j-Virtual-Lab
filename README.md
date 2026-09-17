# Virtual Lab: Knowledge Graph Schema Design & Neo4j Data Import

**Experiment 9 — Design a Knowledge Graph Schema and Import Data**

*Design, construct, import, query and analyze a domain-specific knowledge graph.*

An interactive Streamlit virtual laboratory where a student designs a knowledge
graph schema, validates and imports structured entity–relationship data, runs
Cypher queries, inspects the resulting graph, records trials, takes a quiz and
downloads a PDF lab report.

**The whole experiment runs without Neo4j.** If no database is reachable the lab
switches to a Local Simulation engine that executes the *same generated Cypher*
in memory. Neo4j is an optional real backend, never a requirement — and the lab
never claims to be connected when it is not.

---

## 1. Features

| Stage | What the student actually does |
|---|---|
| **Theory** | Seven tabs: knowledge graphs, Neo4j fundamentals, schema design, design principles, data import, Cypher basics, objectives & procedure |
| **Domain selection** | Five coherent domains — University (default), E-Commerce, Healthcare, Movie Recommendation, Library. Changing the domain reloads its schema, data and example queries |
| **Schema designer** | Add / edit / remove node labels, properties, unique ID properties, relationship types and relationship properties, with live validation |
| **Schema diagram** | A Plotly diagram rebuilt from the schema on every edit — never a static image |
| **Sample dataset** | Built-in data per domain, shown as tables before import |
| **CSV import** | Upload `nodes.csv` + `relationships.csv`, with validation for missing columns, blank cells, duplicate ids and relationships pointing at non-existent nodes |
| **Cypher generator** | Builds the import script from *your* schema and data — Generate / Copy / Download / Clear |
| **Neo4j connection** | URI, username, password, database; Test / Connect / Disconnect; environment-variable support |
| **Import** | Runs the generated script against Neo4j or the local engine and reports node/relationship counts; `MERGE` makes repeat imports idempotent |
| **Query console** | Editable Cypher editor with per-domain examples, result table, record count and execution time |
| **Graph visualization** | Interactive network view with filters by node label and relationship type |
| **Logbook** | Record trials, view the table, download trials CSV; data survives navigation |
| **Observations** | Auto-computed metrics and plain-language findings |
| **Quiz** | 15 MCQs (5 basic, 5 intermediate, 5 advanced) with instant feedback, explanations and retake |
| **Report** | A PDF lab report with student details, schema tables, a vector schema diagram, trials, queries, observations, quiz score and conclusion |

---

## 2. Technology stack

- **Python 3.9+** (developed and tested on 3.10)
- **Streamlit** — the laboratory interface. Native components only, no custom
  CSS: the white + navy identity is declared with Streamlit's own theme options
  in `.streamlit/config.toml`
- **Neo4j + Cypher** — optional real graph database backend, via the official
  `neo4j` Python driver
- **pandas** — datasets, CSV handling, result tables
- **numpy** — the graph layout maths
- **Plotly** — schema diagram, graph visualisation, distribution charts
- **fpdf2** — PDF report generation

---

## 3. Installation

```bash
# 1. Get the project
cd "KG_Virtual lab"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#    Windows (PowerShell / cmd)
.venv\Scripts\activate
#    macOS / Linux
source .venv/bin/activate

# 4. Install the dependencies
pip install -r requirements.txt

# 5. Run the laboratory
streamlit run app.py
```

Streamlit opens `http://localhost:8501` in your browser. **Nothing else is
required** — the lab starts in Local Simulation Mode.

---

## 4. Project structure

```
app.py                  Streamlit interface: Theory, Simulation, Quiz, Report
simulation_engine.py    In-memory property graph + a teaching subset of Cypher
neo4j_service.py        Defensive wrapper around the official Neo4j driver
schema_tools.py         Schema validation, CSV validation, Cypher generation
domains.py              The five domains: schemas, sample data, example queries
graph_viz.py            Plotly schema diagram, graph view and bar charts
quiz_bank.py            The 15 assessment questions and the grader
report_generator.py     The PDF lab report (fpdf2)
requirements.txt        Dependencies
sample_data/            Ready-made CSVs for every domain, plus a broken pair
                        for practising the validator
.streamlit/config.toml  The white + navy theme, chart colours and the hidden
                        Deploy button - all native Streamlit theme options
```

---

## 5. Execution modes

The active mode is always shown at the top right of the page and again in the
Simulation section.

### A. Local Simulation Mode (the default)

No database needed. The generated Cypher runs against `InMemoryGraph`, which
stores nodes, relationships and properties in Python. The built-in interpreter
supports:

`MATCH`, `OPTIONAL MATCH`, `WHERE`, `CREATE`, `MERGE`, `ON CREATE SET`,
`ON MATCH SET`, `SET`, `DELETE`, `DETACH DELETE`, `WITH`, `UNWIND`, `RETURN`
(with `DISTINCT` and `count` / `collect` / `sum` / `avg` / `min` / `max`),
`ORDER BY`, `SKIP`, `LIMIT`, and variable-length patterns such as
`-[:ENROLLED_IN*1..3]->`.

It is a *teaching* interpreter, not a complete Cypher implementation. Anything
it cannot parse comes back as a friendly message — it never crashes the app.

### B. Neo4j Connected Mode

Open **Simulation → Neo4j Connection**, fill in the details and press
**Connect**. The connection is verified with a real round trip to the server
before the lab reports success. Once connected, imports and queries run on the
live database and the graph view reads back from it.

| Field | Typical value |
|---|---|
| URI | `bolt://localhost:7687` (or `neo4j+s://<host>` for Aura) |
| Username | `neo4j` |
| Password | the password you set when the database was created |
| Database | `neo4j` |

**Environment variables** are read as the form's defaults:

```bash
# Windows PowerShell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "your-password"
$env:NEO4J_DATABASE = "neo4j"

# macOS / Linux
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export NEO4J_DATABASE="neo4j"
```

Set them **before** starting Streamlit. The password is masked in the form and
is never displayed back.

### Installing Neo4j (optional)

1. **Neo4j Desktop** — download from <https://neo4j.com/download/>, create a
   local DBMS, set a password, press **Start**. Bolt runs on `7687`.
2. **Docker**

   ```bash
   docker run --name neo4j-lab -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/testpassword neo4j:5
   ```

3. **Neo4j Aura** — free cloud instance; use the `neo4j+s://` URI it gives you.

If the connection fails, the lab tells you why (wrong password, no server,
bad URI) and stays in Local Simulation Mode.

---

## 6. Performing the experiment

There is no sidebar. The four sections are the buttons in the **navigation bar
across the top** (the active one is filled navy), and the stages of the
experiment are the tabs inside *Simulation*. The current execution mode sits at
the top right, and **Progress tracker** under the navigation bar expands to show
which stages are complete.

1. **1 · Theory** — read the seven tabs, then press *Mark theory as studied*.
2. **2 · Simulation → Domain & Data** — choose a domain (default: University)
   and inspect the entities, relationships and sample data.
3. **Schema Designer** — define node labels, their properties and unique ID
   property, then the relationship types. Validation runs as you edit.
4. **Schema Diagram** — check that the diagram matches what you intended.
5. **Data Import → Step 1** — use the sample dataset or upload your own CSVs.
6. **Data Import → Step 2** — press *Validate dataset* and read the results.
7. **Data Import → Step 3** — press *Generate Cypher* and read the script that
   your schema and data produced.
8. **Data Import → Step 4** — press *Import into …* and note the counts. Import
   a second time to see `MERGE` prevent duplicates.
9. **Query Console** — run the example queries, then write your own.
10. **Graph View** — filter by node label and relationship type; follow a path.
11. **Logbook** — press *Record current trial* after each experiment, and read
    the observations underneath.
12. **3 · Quiz** — answer all 15 questions and read the explanations.
13. **4 · Report Generation** — fill in your details and conclusion, generate
    the PDF and download it.

---

## 7. Importing your own CSV

Two files, both plain CSV with a header row.

**`nodes.csv`** — required columns `label` and `id`, then one column per
property (leave a cell blank where a property does not apply to that label):

```csv
label,id,student_id,name,semester,course_id,credits
Student,S001,S001,Aditi,4,,
Student,S002,S002,Rahul,4,,
Course,C101,,Database Management Systems,,C101,4
```

**`relationships.csv`** — required columns `source_id`, `type` and `target_id`.
`source_label` / `target_label` are optional (they are inferred from the nodes
file), and any further columns become relationship properties:

```csv
source_id,source_label,type,target_id,target_label,grade
S001,Student,ENROLLED_IN,C101,Course,A
S002,Student,ENROLLED_IN,C101,Course,B
```

Working examples for every domain are in `sample_data/`, and you can download
the current domain's templates from the Data Import tab.

The validator reports, rather than silently accepting:

- missing required columns
- blank `label` or `id` cells
- duplicate ids within a label
- ids reused across different labels (warning)
- relationships whose `source_id` or `target_id` does not exist
- empty relationship types
- labels, properties or relationship patterns that your schema does not declare

`sample_data/university_nodes_with_errors.csv` and its matching relationships
file contain each of these mistakes on purpose — load them to see the validator
work.

---

## 8. Generating the report

Go to **4. Report Generation**, fill in name, roll number, department, semester
and date, write a conclusion, press **Generate PDF Report** and then **Download
PDF Report**.

The PDF contains: student information, aim and learning objectives, a theory
summary, the experimental procedure, your schema as tables plus a drawn schema
diagram, the imported dataset summary, recorded trials with their queries, the
Cypher you executed, the query results summary, observations, the quiz score by
difficulty, and your conclusion — with signature lines at the end.

The diagram is drawn with vector primitives, so no image-export dependency is
needed and the report always generates.

---

## 9. Troubleshooting

| Symptom | What to do |
|---|---|
| `streamlit: command not found` | The virtual environment is not active, or run `python -m streamlit run app.py` |
| "Neo4j is not available…" | Expected without a server — everything still works in Local Simulation Mode |
| Neo4j refuses the password | Use the password set when the DBMS was created; reset it in Neo4j Desktop if needed |
| "No Neo4j server answered" | Start the DBMS and check the Bolt port (`7687` by default) |
| A query returns 0 rows | Check the labels, property values and the direction of the arrow in your pattern |
| "not supported by the local simulation engine" | That clause is outside the teaching subset — connect a real Neo4j to run it |
| Import created 0 nodes | `MERGE` found everything already there; clear the graph first to re-import |

---

## 10. Academic note

This laboratory covers the syllabus requirements for Experiment 9: designing
node labels, designing relationship types, defining node and relationship
properties, importing structured entity–relationship data and implementing a
domain-specific knowledge graph in Neo4j.
