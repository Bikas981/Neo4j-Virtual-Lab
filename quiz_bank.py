"""
quiz_bank.py
============
The assessment for Experiment 9.

Fifteen multiple-choice questions, five at each level (Basic, Intermediate,
Advanced), each with four options, one correct answer and an explanation that
teaches rather than just marks.

The questions cover knowledge graph concepts, graph database fundamentals,
schema design, data import and graph analysis. They are written in general
terms, without reference to any particular database product or query language.
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
            "A chart that shows how much data a system stores",
            "A network of entities connected by meaningful, named relationships",
            "A table of rows and columns with a primary key",
            "A folder structure used to organise documents",
        ],
        "answer": 1,
        "explanation": (
            "A knowledge graph stores entities as nodes and the connections between "
            "them as named, directed relationships. The connections are data in their "
            "own right, which is what makes connected questions easy to answer."
        ),
    },
    {
        "id": "q02",
        "level": BASIC,
        "topic": "Nodes and labels",
        "question": "What is a node label used for?",
        "options": [
            "To store the value of a property",
            "To give a relationship its direction",
            "To rename the dataset",
            "To group nodes into a category, such as Student or Course",
        ],
        "answer": 3,
        "explanation": (
            "A label classifies a node: it says what the entity is. Labels are what "
            "you select on when you ask for 'all Student entities', and one node may "
            "carry more than one label."
        ),
    },
    {
        "id": "q03",
        "level": BASIC,
        "topic": "Properties",
        "question": (
            "A Student entity is stored with student_id = S001 and name = Aditi. "
            "Which of these is a property?"
        ),
        "options": [
            "name = Aditi",
            "Student",
            "The arrow that connects two entities",
            "The graph itself",
        ],
        "answer": 0,
        "explanation": (
            "Properties are the key-value facts stored on an entity: student_id and "
            "name are properties. Student is the label that classifies the entity, "
            "not a property of it."
        ),
    },
    {
        "id": "q04",
        "level": BASIC,
        "topic": "Relationships",
        "question": "What does the pattern  Student --[ENROLLED_IN]--> Course  express?",
        "options": [
            "A Course that stores a Student as one of its properties",
            "Two unrelated entities shown side by side",
            "A directed relationship of type ENROLLED_IN from a Student to a Course",
            "A junction table linking students and courses",
        ],
        "answer": 2,
        "explanation": (
            "ENROLLED_IN is the relationship type and the arrow gives its direction. "
            "The relationship is stored with the entities, so following it needs no "
            "join between tables."
        ),
    },
    {
        "id": "q05",
        "level": BASIC,
        "topic": "Graph structure",
        "question": "Which statement about a relationship in a knowledge graph is correct?",
        "options": [
            "It may connect any number of entities at once",
            "It connects exactly two entities and carries one type and a direction",
            "It can never store properties of its own",
            "It is automatically created between entities that share a property value",
        ],
        "answer": 1,
        "explanation": (
            "A relationship always joins exactly two entities, carries a single type "
            "such as TEACHES, and has a direction. It may also carry its own "
            "properties, such as a grade or a role."
        ),
    },
    # ------------------------- Intermediate -------------------------
    {
        "id": "q06",
        "level": INTERMEDIATE,
        "topic": "Duplicate prevention",
        "question": (
            "The same dataset is imported twice, and the import always inserts new "
            "entities without checking what already exists. What is the result?"
        ),
        "options": [
            "The second import is ignored automatically",
            "The second import updates the existing entities",
            "An error stops the second import",
            "Every entity exists twice, because nothing checked for an existing copy",
        ],
        "answer": 3,
        "explanation": (
            "An import that always inserts creates a second copy of every entity. "
            "This is why an import should match each entity on its unique identifier "
            "and update it when it already exists."
        ),
    },
    {
        "id": "q07",
        "level": INTERMEDIATE,
        "topic": "Unique identifiers",
        "question": "Why does every entity type need a unique identifier property?",
        "options": [
            "It distinguishes entities that look alike and lets an import recognise existing ones",
            "It makes the graph diagram easier to draw",
            "It is required before any property can be added",
            "It determines the direction of the relationships",
        ],
        "answer": 0,
        "explanation": (
            "Two students may share a name, so the identifier is what tells them "
            "apart. The import compares this property to decide whether an entity is "
            "already present in the graph."
        ),
    },
    {
        "id": "q08",
        "level": INTERMEDIATE,
        "topic": "Schema design",
        "question": (
            "A dataset records the grade a student earned in a course. Where does "
            "'grade' belong?"
        ),
        "options": [
            "As a property of the Student entity",
            "As a property of the Course entity",
            "As a property of the ENROLLED_IN relationship",
            "As a separate Grade entity connected to nothing",
        ],
        "answer": 2,
        "explanation": (
            "The grade depends on the student and the course together, so it belongs "
            "to the relationship that joins them. On either entity alone it would "
            "break as soon as the student takes a second course."
        ),
    },
    {
        "id": "q09",
        "level": INTERMEDIATE,
        "topic": "Data import",
        "question": "Why are entities imported before relationships?",
        "options": [
            "Because relationships take longer to create",
            "Because a relationship can only be created between entities that already exist",
            "Because entities must be sorted alphabetically first",
            "Because relationships cannot carry properties",
        ],
        "answer": 1,
        "explanation": (
            "A relationship connects two existing entities. If the entities have not "
            "been created yet, there is nothing for the relationship to join, which "
            "is why the import runs in two passes."
        ),
    },
    {
        "id": "q10",
        "level": INTERMEDIATE,
        "topic": "Data validation",
        "question": (
            "A relationship record refers to a source identifier that appears in no "
            "entity record. What should the import do?"
        ),
        "options": [
            "Create an empty entity with that identifier and continue",
            "Import the relationship without a source",
            "Reject the record and report it, because the relationship has nothing to connect",
            "Ignore the record silently",
        ],
        "answer": 2,
        "explanation": (
            "The record is invalid and must be reported. Creating an empty entity "
            "would silently add a meaningless node to the graph, which is harder to "
            "detect later than an error message."
        ),
    },
    # --------------------------- Advanced ---------------------------
    {
        "id": "q11",
        "level": ADVANCED,
        "topic": "Graph vs relational",
        "question": (
            "Why does a graph database usually outperform a relational database on "
            "deeply connected queries?"
        ),
        "options": [
            "It keeps the entire dataset in memory at all times",
            "It enforces no schema, so queries are shorter",
            "Relational databases cannot express relationships",
            "Each entity stores its own relationships, so a hop is a local step rather than a join",
        ],
        "answer": 3,
        "explanation": (
            "Because each node keeps references to its own relationships, the cost of "
            "one hop does not grow with the size of the dataset. The same question in "
            "a relational database needs one join per hop over whole tables."
        ),
    },
    {
        "id": "q12",
        "level": ADVANCED,
        "topic": "Graph traversal",
        "question": (
            "Starting at a student, you follow ENROLLED_IN to a course and then "
            "follow ENROLLED_IN backwards to other students. What have you found?"
        ),
        "options": [
            "The students who share a course with the first student",
            "The students who are enrolled in no courses",
            "The courses that have exactly two students",
            "The faculty members who teach that course",
        ],
        "answer": 0,
        "explanation": (
            "This two-hop traversal goes out to a shared course and back to other "
            "students, which identifies classmates. Answering the same question with "
            "tables would require two joins."
        ),
    },
    {
        "id": "q13",
        "level": ADVANCED,
        "topic": "Many-to-many data",
        "question": "How is a many-to-many association represented in a knowledge graph?",
        "options": [
            "With a junction entity that must be created for every pair",
            "Simply as many relationships of the same type between the entities",
            "By duplicating one of the entities for each association",
            "By storing a list of identifiers as a property",
        ],
        "answer": 1,
        "explanation": (
            "A student may have many ENROLLED_IN relationships and a course may "
            "receive many, so the association needs no extra structure. A relational "
            "schema would need a junction table for the same data."
        ),
    },
    {
        "id": "q14",
        "level": ADVANCED,
        "topic": "Graph integrity",
        "question": (
            "Why can an entity that still has relationships not simply be deleted?"
        ),
        "options": [
            "Because entities can never be removed once imported",
            "Because its properties must be cleared first",
            "Because the relationships would be left pointing at something that no longer exists",
            "Because only relationships may be deleted, never entities",
        ],
        "answer": 2,
        "explanation": (
            "A relationship needs both of its end entities. Removing one of them would "
            "leave a dangling connection, so the relationships must be removed "
            "together with the entity."
        ),
    },
    {
        "id": "q15",
        "level": ADVANCED,
        "topic": "Schema design",
        "question": (
            "A design stores everything as one entity type with properties such as "
            "type = 'student' and course = 'C101'. What is the main problem?"
        ),
        "options": [
            "It uses more disk space than necessary",
            "It cannot store numbers as property values",
            "The meaning is hidden inside properties, so connections cannot be traversed",
            "It prevents properties from being added later",
        ],
        "answer": 2,
        "explanation": (
            "Labels and relationship types are what carry meaning in a graph. When "
            "the connection is only a text value inside a property, nothing can be "
            "followed, and the graph loses the advantage it was chosen for."
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
