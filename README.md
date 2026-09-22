# Virtual Lab: Knowledge Graph Schema Design & Data Import

**Experiment 9 — Design a Knowledge Graph Schema and Import Data**

*Design, construct, import, query and analyze a domain-specific knowledge graph.*

An interactive Streamlit virtual laboratory where a student designs a knowledge
graph schema, validates and imports structured entity–relationship data, queries
and analyzes the resulting graph, records trials, takes a quiz and downloads a
PDF lab report.

**Everything runs in the laboratory itself.** The knowledge graph is built and
queried by a built-in in-memory simulation engine, so no database software or
connection is required.

---

## 1. Features

| Stage | What the student actually does |
|---|---|
| **Theory** | Six tabs: knowledge graphs, graph database fundamentals, schema design, design principles, data import, objectives & procedure |
| **Domain selection** | Five coherent domains — University (default), E-Commerce, Healthcare, Movie Recommendation, Library. Changing the domain reloads its schema and data |
| **Schema designer** | Add / edit / remove node labels, properties, unique ID properties, relationship types and relationship properties, with live validation |
| **Schema diagram** | A Plotly diagram rebuilt from the schema on every edit — never a static image |
| **Sample dataset** | Built-in data per domain, shown as tables before import |
| **Data import** | Built-in data or uploaded `nodes.csv` + `relationships.csv`, validated for missing columns, blank cells, duplicate ids and relationships pointing at non-existent entities |
| **Duplicate prevention** | Each entity is matched on its unique identifier, so importing the same data twice updates it instead of duplicating it |
| **Graph Query** | Ask questions of the graph — list entities of a type, filter by property, show relationships of a type, show the connections of one entity, count entities per type — executed by the simulation engine |
| **Graph visualization** | Interactive network view with filters by node label and relationship type |
| **Logbook** | Record trials, view the table, download trials CSV; data survives navigation |
| **Observations** | Auto-computed metrics and plain-language findings |
| **Quiz** | 15 MCQs (5 basic, 5 intermediate, 5 advanced) with instant feedback and explanations |
| **Report** | A PDF lab report in 13 sections, containing the student's own experiment data |

---

## 2. Technology stack

- **Python 3.9+** (developed and tested on 3.10)
- **Streamlit** — the laboratory interface. Native components only, no custom
  CSS: the white + navy identity is declared with Streamlit's own theme options
  in `.streamlit/config.toml`
- **pandas** — datasets, CSV handling, result tables
- **numpy** — the graph layout maths
- **Plotly** — schema diagram, graph visualisation, distribution charts
- **fpdf2** — PDF report generation

The knowledge graph itself is stored and queried by `simulation_engine.py`, an
in-memory graph engine written for this laboratory. Its internal query language
is an implementation detail and is never shown in the interface.

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
simulation_engine.py    In-memory property graph and its query engine
schema_tools.py         Schema validation, CSV validation, import statements
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

## 5. How the graph is stored

The laboratory keeps the knowledge graph in memory for the duration of the
session, in the engine implemented by `simulation_engine.py`. The Simulation
section shows **Mode: Local Simulation** so it is always clear where the data
lives.

Everything in the experiment — schema validation, data import, duplicate
prevention, graph queries and the graph view — is executed by that engine. There
is no database to install, configure or connect to, and nothing leaves the
machine the laboratory runs on.

The graph is held for the session only: reloading the page starts from an empty
graph, and the dataset can be imported again in a few seconds.

---

## 6. Performing the experiment

There is no sidebar. The four sections are the buttons in the **navigation bar
across the top** (the active one is filled navy), and the stages of the
experiment are the tabs inside *Simulation*. The current execution mode sits at
the top right, and **Progress tracker** under the navigation bar expands to show
which stages are complete.

1. **1 · Theory** — read the six tabs, then press *Mark theory as studied*.
2. **2 · Simulation → Domain & Data** — choose a domain (default: University)
   and inspect the entities, relationships and sample data.
3. **Schema Designer** — define node labels, their properties and unique ID
   property, then the relationship types. Validation runs as you edit.
4. **Schema Diagram** — check that the diagram matches what you intended.
5. **Data Import → Step 1** — use the sample dataset or upload your own CSVs.
6. **Data Import → Step 2** — press *Validate dataset* and read the results.
7. **Data Import → Step 3** — press *Import into the graph* and note the counts.
   Import a second time and see that no duplicates are created.
8. **Graph Query** — choose a question, set its options and press *Execute
   Query*; try all five question types.
9. **Graph View** — filter by node label and relationship type; follow a path.
10. **Logbook** — press *Record current trial* after each experiment, and read
    the observations underneath.
11. **3 · Quiz** — answer all 15 questions and read the explanations.
12. **4 · Report Generation** — fill in your details and conclusion, generate
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

The PDF has thirteen sections: experiment title, aim, objectives, theory,
procedure, schema design (tables plus a drawn diagram), data used, graph queries
and analysis, observations (with the recorded trials), results, quiz score,
conclusion and references. It contains the student's own experiment data.

The diagram is drawn with vector primitives, so no image-export dependency is
needed and the report always generates.

---

## 9. Troubleshooting

| Symptom | What to do |
|---|---|
| `streamlit: command not found` | The virtual environment is not active, or run `python -m streamlit run app.py` |
| A query returns no results | Check the entity type, the property and the value you selected |
| The Graph Query tab says the graph is empty | Import the dataset in the **Data Import** tab first |
| Import created 0 nodes | Every entity was already in the graph; clear the graph first to re-import |
| The graph is empty after reloading the page | The graph is held for the session only — import the dataset again |

---

## 10. Academic note

This laboratory covers the syllabus requirements for Experiment 9: designing
node labels, designing relationship types, defining node and relationship
properties, importing structured entity–relationship data and implementing a
domain-specific knowledge graph.
