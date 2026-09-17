"""
quiz_bank.py
============
The assessment for Experiment 9.

Fifteen multiple-choice questions, five at each level (Basic, Intermediate,
Advanced), each with four options, one correct answer and an explanation that
teaches rather than just marks.
"""

from __future__ import annotations

from typing import Any, Dict, List

BASIC = "Basic"
INTERMEDIATE = "Intermediate"
ADVANCED = "Advanced"

QUESTIONS: List[Dict[str, Any]] = [
    # ---------------------------- Basic -----------------------------
    {
        "id": "q01",
        "level": BASIC,
        "topic": "Knowledge graph fundamentals",
        "question": "What is a knowledge graph?",
        "options": [
            "A network of entities connected by meaningful, typed relationships",
            "A bar chart that shows how much knowledge a system stores",
            "A table of rows and columns with a primary key",
            "A folder structure used to organise documents",
        ],
        "answer": 0,
        "explanation": (
            "A knowledge graph stores entities (nodes) and the named relationships "
            "between them. The relationships carry meaning and are stored as first-class "
            "data, which is what makes connected questions easy to answer."
        ),
    },
    {
        "id": "q02",
        "level": BASIC,
        "topic": "Nodes and labels",
        "question": "In Neo4j, what is a label used for?",
        "options": [
            "To group nodes into a category, such as :Student or :Course",
            "To store the value of a property",
            "To give a relationship a direction",
            "To rename the database",
        ],
        "answer": 0,
        "explanation": (
            "A label classifies a node. Writing (s:Student) says 'this node is a "
            "Student'. A node may have more than one label, and labels are what you "
            "match on in queries such as MATCH (s:Student)."
        ),
    },
    {
        "id": "q03",
        "level": BASIC,
        "topic": "Properties",
        "question": "Which of these is a property in the pattern (s:Student {student_id: 'S001', name: 'Aditi'})?",
        "options": [
            "name: 'Aditi'",
            "Student",
            "s",
            "MATCH",
        ],
        "answer": 0,
        "explanation": (
            "Properties are the key-value pairs inside the braces. Here student_id and "
            "name are properties, Student is the label, and s is just the variable that "
            "lets the rest of the query refer to this node."
        ),
    },
    {
        "id": "q04",
        "level": BASIC,
        "topic": "Cypher basics",
        "question": "Which Cypher clause reads existing data from the graph?",
        "options": ["MATCH", "CREATE", "DELETE", "SET"],
        "answer": 0,
        "explanation": (
            "MATCH finds patterns that already exist. CREATE adds new data, SET changes "
            "properties, and DELETE removes data."
        ),
    },
    {
        "id": "q05",
        "level": BASIC,
        "topic": "Relationships",
        "question": "What does the pattern (s:Student)-[:ENROLLED_IN]->(c:Course) express?",
        "options": [
            "A directed relationship of type ENROLLED_IN from a Student to a Course",
            "A Course that contains a Student as a property",
            "Two unrelated nodes returned side by side",
            "A join table between students and courses",
        ],
        "answer": 0,
        "explanation": (
            "The arrow shows direction and ENROLLED_IN is the relationship type. The "
            "relationship is stored on disk, so traversing it does not need a join."
        ),
    },
    # ------------------------- Intermediate -------------------------
    {
        "id": "q06",
        "level": INTERMEDIATE,
        "topic": "CREATE vs MERGE",
        "question": "You run the same CREATE (s:Student {student_id: 'S001'}) statement twice. What happens?",
        "options": [
            "Two separate Student nodes exist, because CREATE never checks for duplicates",
            "The second statement is ignored because the id already exists",
            "The second statement updates the first node",
            "Neo4j raises an error automatically",
        ],
        "answer": 0,
        "explanation": (
            "CREATE always creates. Without a uniqueness constraint nothing stops the "
            "duplicate, which is exactly why imports use MERGE on the key property "
            "instead."
        ),
    },
    {
        "id": "q07",
        "level": INTERMEDIATE,
        "topic": "Duplicate prevention",
        "question": "Which statement imports a student without creating a duplicate if it is run again?",
        "options": [
            "MERGE (s:Student {student_id: 'S001'}) SET s.name = 'Aditi'",
            "CREATE (s:Student {student_id: 'S001', name: 'Aditi'})",
            "MATCH (s:Student {student_id: 'S001'}) RETURN s",
            "DELETE (s:Student {student_id: 'S001'})",
        ],
        "answer": 0,
        "explanation": (
            "MERGE matches the pattern first and only creates it when nothing matches. "
            "Merging on the key property alone and then using SET for the remaining "
            "properties is the standard idempotent import pattern."
        ),
    },
    {
        "id": "q08",
        "level": INTERMEDIATE,
        "topic": "WHERE",
        "question": "What does MATCH (s:Student) WHERE s.semester = 4 RETURN s.name return?",
        "options": [
            "The names of the students whose semester property equals 4",
            "All students, with the semester column set to 4",
            "The fourth student in the database",
            "An error, because WHERE cannot follow MATCH",
        ],
        "answer": 0,
        "explanation": (
            "WHERE filters the rows produced by MATCH. The same filter can also be "
            "written inline as MATCH (s:Student {semester: 4})."
        ),
    },
    {
        "id": "q09",
        "level": INTERMEDIATE,
        "topic": "LOAD CSV",
        "question": "What is LOAD CSV used for in Neo4j?",
        "options": [
            "Reading rows from a CSV file so they can be turned into nodes and relationships",
            "Exporting the graph to a spreadsheet",
            "Creating a backup of the database",
            "Defining node labels automatically from column names",
        ],
        "answer": 0,
        "explanation": (
            "LOAD CSV streams rows into a query. Each row is a map, and you still have "
            "to write the MERGE/CREATE statements that decide which columns become "
            "labels, properties and relationships."
        ),
    },
    {
        "id": "q10",
        "level": INTERMEDIATE,
        "topic": "Deleting data",
        "question": "Why does DELETE fail on a node that still has relationships?",
        "options": [
            "A relationship cannot exist without both of its end nodes, so DETACH DELETE is required",
            "DELETE only works on relationships, never on nodes",
            "The node is locked by another transaction",
            "Neo4j needs the node to be empty of properties first",
        ],
        "answer": 0,
        "explanation": (
            "Deleting the node would leave dangling relationships, so Neo4j refuses. "
            "DETACH DELETE removes the node together with every relationship attached "
            "to it."
        ),
    },
    # --------------------------- Advanced ---------------------------
    {
        "id": "q11",
        "level": ADVANCED,
        "topic": "Graph vs relational",
        "question": "Why does a graph database usually beat a relational database for deeply connected queries?",
        "options": [
            "Relationships are stored with the nodes, so traversal does not need repeated joins",
            "Graph databases keep the whole dataset in RAM at all times",
            "Graph databases do not enforce any schema, so queries are shorter",
            "SQL cannot express relationships at all",
        ],
        "answer": 0,
        "explanation": (
            "This is index-free adjacency: each node knows its own relationships, so the "
            "cost of a hop is local. In SQL the same question needs a join per hop, and "
            "the cost grows with the size of the tables."
        ),
    },
    {
        "id": "q12",
        "level": ADVANCED,
        "topic": "Schema design",
        "question": "A dataset records the grade a student earned in a course. Where does 'grade' belong?",
        "options": [
            "As a property of the ENROLLED_IN relationship",
            "As a property of the Student node",
            "As a property of the Course node",
            "As a separate Grade node connected to nothing",
        ],
        "answer": 0,
        "explanation": (
            "The grade depends on the student *and* the course together, so it belongs "
            "to the relationship that joins them. Putting it on either node would be "
            "wrong as soon as the student takes a second course."
        ),
    },
    {
        "id": "q13",
        "level": ADVANCED,
        "topic": "Graph traversal",
        "question": "What does MATCH (a:Student)-[:ENROLLED_IN]->(c:Course)<-[:ENROLLED_IN]-(b:Student) RETURN a.name, b.name find?",
        "options": [
            "Pairs of students who share at least one course",
            "Students who are enrolled in no courses",
            "Courses with exactly two students",
            "Students who teach a course",
        ],
        "answer": 0,
        "explanation": (
            "The pattern hops out to a course and back in, so a and b are classmates. "
            "Adding WHERE a.student_id < b.student_id removes the mirrored duplicates."
        ),
    },
    {
        "id": "q14",
        "level": ADVANCED,
        "topic": "Data import integrity",
        "question": "Before importing relationships from a CSV, which check matters most?",
        "options": [
            "That every source and target id already exists as a node",
            "That the file is sorted alphabetically",
            "That the file has fewer than 1000 rows",
            "That the relationship types are lowercase",
        ],
        "answer": 0,
        "explanation": (
            "A relationship row that points at a missing id cannot be created. If you "
            "use MERGE on the endpoints instead of MATCH, a typo silently creates an "
            "empty phantom node - which is worse than an error."
        ),
    },
    {
        "id": "q15",
        "level": ADVANCED,
        "topic": "Uniqueness and constraints",
        "question": "What is the main effect of CREATE CONSTRAINT FOR (s:Student) REQUIRE s.student_id IS UNIQUE?",
        "options": [
            "It rejects a second Student with the same student_id and speeds up lookups on it",
            "It automatically deletes duplicate students that already exist",
            "It makes student_id mandatory on every node in the database",
            "It creates the Student label if it does not exist",
        ],
        "answer": 0,
        "explanation": (
            "A uniqueness constraint enforces the key and is backed by an index, so "
            "MERGE on student_id becomes both safe and fast. Existing duplicates must "
            "be cleaned up before the constraint can be created."
        ),
    },
]


def questions_by_level(level: str) -> List[Dict[str, Any]]:
    return [q for q in QUESTIONS if q["level"] == level]


def level_order() -> List[str]:
    return [BASIC, INTERMEDIATE, ADVANCED]


def grade(answers: Dict[str, int]) -> Dict[str, Any]:
    """Score a submission. ``answers`` maps question id -> chosen option index."""
    details = []
    correct = 0
    for question in QUESTIONS:
        chosen = answers.get(question["id"])
        is_correct = chosen is not None and chosen == question["answer"]
        correct += 1 if is_correct else 0
        details.append(
            {
                "id": question["id"],
                "level": question["level"],
                "topic": question["topic"],
                "question": question["question"],
                "chosen": chosen,
                "chosen_text": question["options"][chosen] if chosen is not None else "Not answered",
                "correct_index": question["answer"],
                "correct_text": question["options"][question["answer"]],
                "is_correct": is_correct,
                "explanation": question["explanation"],
            }
        )
    total = len(QUESTIONS)
    percentage = round(correct / total * 100, 1) if total else 0.0
    by_level = {}
    for level in level_order():
        level_items = [d for d in details if d["level"] == level]
        by_level[level] = {
            "correct": sum(1 for d in level_items if d["is_correct"]),
            "total": len(level_items),
        }
    return {
        "correct": correct,
        "total": total,
        "percentage": percentage,
        "details": details,
        "by_level": by_level,
        "verdict": _verdict(percentage),
    }


def _verdict(percentage: float) -> str:
    if percentage >= 80:
        return "Excellent - the concepts are clear."
    if percentage >= 60:
        return "Good - revise the questions you missed."
    if percentage >= 40:
        return "Fair - re-read the Theory section before the viva."
    return "Needs work - go back through Theory and repeat the simulation."
