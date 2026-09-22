"""
domains.py
==========
Domain catalogue for the Knowledge Graph Virtual Lab.

Every domain ships with a *coherent* package:

* ``nodes``          -- the node labels, their key property and their properties
* ``relationships``  -- the relationship types with source/target labels
* ``data``           -- the sample entities for each label
* ``edges``          -- the sample relationships (every endpoint really exists)
* ``queries``        -- reference queries kept with each domain (not displayed;
                       the interface builds its own queries in the Graph Query tab)

The sample data is validated at import time by :func:`validate_domain_data`, so
a domain can never hand the laboratory an edge that points at a missing node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

DEFAULT_DOMAIN = "University"


# ======================================================================
#  1. University  (the default domain used by the experiment)
# ======================================================================
UNIVERSITY: Dict[str, Any] = {
    "name": "University",
    "description": (
        "An academic knowledge graph connecting students, the courses they take, "
        "the faculty who teach them, the departments they belong to and the "
        "projects they work on."
    ),
    "entity_hint": (
        "Look for the *nouns* in the dataset: student, course, faculty, department, "
        "project. Each noun becomes a node label."
    ),
    "nodes": [
        {"label": "Student", "key": "student_id", "properties": ["student_id", "name", "semester"]},
        {"label": "Course", "key": "course_id", "properties": ["course_id", "name", "credits"]},
        {"label": "Faculty", "key": "faculty_id", "properties": ["faculty_id", "name", "specialization"]},
        {"label": "Department", "key": "dept_id", "properties": ["dept_id", "name"]},
        {"label": "Project", "key": "project_id", "properties": ["project_id", "title", "domain"]},
    ],
    "relationships": [
        {"type": "ENROLLED_IN", "source": "Student", "target": "Course", "properties": ["grade"]},
        {"type": "TEACHES", "source": "Faculty", "target": "Course", "properties": []},
        {"type": "BELONGS_TO", "source": "Student", "target": "Department", "properties": []},
        {"type": "BELONGS_TO", "source": "Faculty", "target": "Department", "properties": []},
        {"type": "BELONGS_TO", "source": "Course", "target": "Department", "properties": []},
        {"type": "WORKS_ON", "source": "Student", "target": "Project", "properties": ["role"]},
        {"type": "GUIDED_BY", "source": "Faculty", "target": "Project", "properties": []},
    ],
    "data": {
        "Student": [
            {"student_id": "S001", "name": "Aditi", "semester": 4},
            {"student_id": "S002", "name": "Rahul", "semester": 4},
            {"student_id": "S003", "name": "Priya", "semester": 6},
            {"student_id": "S004", "name": "Arjun", "semester": 6},
            {"student_id": "S005", "name": "Neha", "semester": 4},
        ],
        "Course": [
            {"course_id": "C101", "name": "Database Management Systems", "credits": 4},
            {"course_id": "C102", "name": "Computer Networks", "credits": 3},
            {"course_id": "C103", "name": "Artificial Intelligence", "credits": 4},
        ],
        "Faculty": [
            {"faculty_id": "F001", "name": "Dr. Sharma", "specialization": "Databases"},
            {"faculty_id": "F002", "name": "Dr. Mehta", "specialization": "Computer Networks"},
            {"faculty_id": "F003", "name": "Dr. Rao", "specialization": "Artificial Intelligence"},
        ],
        "Department": [
            {"dept_id": "D01", "name": "Computer Engineering"},
            {"dept_id": "D02", "name": "Information Technology"},
        ],
        "Project": [
            {"project_id": "P001", "title": "Smart Campus", "domain": "IoT"},
            {"project_id": "P002", "title": "AI Chatbot", "domain": "NLP"},
            {"project_id": "P003", "title": "Library Management", "domain": "Web"},
        ],
    },
    "edges": [
        {"type": "ENROLLED_IN", "source": "S001", "target": "C101", "properties": {"grade": "A"}},
        {"type": "ENROLLED_IN", "source": "S001", "target": "C103", "properties": {"grade": "B"}},
        {"type": "ENROLLED_IN", "source": "S002", "target": "C101", "properties": {"grade": "B"}},
        {"type": "ENROLLED_IN", "source": "S002", "target": "C102", "properties": {"grade": "A"}},
        {"type": "ENROLLED_IN", "source": "S003", "target": "C102", "properties": {"grade": "A"}},
        {"type": "ENROLLED_IN", "source": "S003", "target": "C103", "properties": {"grade": "B"}},
        {"type": "ENROLLED_IN", "source": "S004", "target": "C103", "properties": {"grade": "C"}},
        {"type": "ENROLLED_IN", "source": "S005", "target": "C101", "properties": {"grade": "A"}},
        {"type": "TEACHES", "source": "F001", "target": "C101", "properties": {}},
        {"type": "TEACHES", "source": "F002", "target": "C102", "properties": {}},
        {"type": "TEACHES", "source": "F003", "target": "C103", "properties": {}},
        {"type": "BELONGS_TO", "source": "S001", "target": "D01", "properties": {}},
        {"type": "BELONGS_TO", "source": "S002", "target": "D01", "properties": {}},
        {"type": "BELONGS_TO", "source": "S003", "target": "D02", "properties": {}},
        {"type": "BELONGS_TO", "source": "S004", "target": "D02", "properties": {}},
        {"type": "BELONGS_TO", "source": "S005", "target": "D01", "properties": {}},
        {"type": "BELONGS_TO", "source": "F001", "target": "D01", "properties": {}},
        {"type": "BELONGS_TO", "source": "F002", "target": "D02", "properties": {}},
        {"type": "BELONGS_TO", "source": "F003", "target": "D01", "properties": {}},
        {"type": "BELONGS_TO", "source": "C101", "target": "D01", "properties": {}},
        {"type": "BELONGS_TO", "source": "C102", "target": "D02", "properties": {}},
        {"type": "BELONGS_TO", "source": "C103", "target": "D01", "properties": {}},
        {"type": "WORKS_ON", "source": "S001", "target": "P001", "properties": {"role": "Developer"}},
        {"type": "WORKS_ON", "source": "S002", "target": "P001", "properties": {"role": "Tester"}},
        {"type": "WORKS_ON", "source": "S003", "target": "P002", "properties": {"role": "Developer"}},
        {"type": "WORKS_ON", "source": "S004", "target": "P002", "properties": {"role": "Designer"}},
        {"type": "WORKS_ON", "source": "S005", "target": "P003", "properties": {"role": "Developer"}},
        {"type": "GUIDED_BY", "source": "F002", "target": "P001", "properties": {}},
        {"type": "GUIDED_BY", "source": "F003", "target": "P002", "properties": {}},
        {"type": "GUIDED_BY", "source": "F001", "target": "P003", "properties": {}},
    ],
    "queries": [
        {"title": "List every student", "cypher": "MATCH (s:Student)\nRETURN s;"},
        {
            "title": "Students and the courses they are enrolled in",
            "cypher": "MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)\nRETURN s.name, c.name;",
        },
        {
            "title": "Which faculty teaches which course",
            "cypher": "MATCH (f:Faculty)-[:TEACHES]->(c:Course)\nRETURN f.name, c.name;",
        },
        {
            "title": "Students and their projects",
            "cypher": "MATCH (s:Student)-[:WORKS_ON]->(p:Project)\nRETURN s.name, p.title;",
        },
        {
            "title": "Students ordered by semester",
            "cypher": "MATCH (s:Student)\nRETURN s.name, s.semester\nORDER BY s.semester DESC;",
        },
        {
            "title": "Filter with WHERE (4th semester students)",
            "cypher": "MATCH (s:Student)\nWHERE s.semester = 4\nRETURN s.name, s.semester;",
        },
        {
            "title": "Count the students on each course",
            "cypher": (
                "MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)\n"
                "RETURN c.name AS course, count(s) AS students\n"
                "ORDER BY students DESC;"
            ),
        },
        {
            "title": "Two-hop traversal: classmates of Aditi",
            "cypher": (
                "MATCH (s:Student {name: 'Aditi'})-[:ENROLLED_IN]->(c:Course)<-[:ENROLLED_IN]-(mate:Student)\n"
                "RETURN DISTINCT mate.name AS classmate, c.name AS shared_course;"
            ),
        },
        {
            "title": "Who guides the project each student works on",
            "cypher": (
                "MATCH (s:Student)-[:WORKS_ON]->(p:Project)<-[:GUIDED_BY]-(f:Faculty)\n"
                "RETURN s.name AS student, p.title AS project, f.name AS guide;"
            ),
        },
    ],
}


# ======================================================================
#  2. E-Commerce
# ======================================================================
ECOMMERCE: Dict[str, Any] = {
    "name": "E-Commerce",
    "description": (
        "An online-store knowledge graph linking customers, the orders they place, "
        "the products in those orders, product categories and sellers."
    ),
    "entity_hint": (
        "Customers, orders, products, categories and sellers are the entities. "
        "An order is an entity in its own right, not a property of a customer."
    ),
    "nodes": [
        {"label": "Customer", "key": "customer_id", "properties": ["customer_id", "name", "city"]},
        {"label": "Order", "key": "order_id", "properties": ["order_id", "order_date", "amount"]},
        {"label": "Product", "key": "product_id", "properties": ["product_id", "name", "price"]},
        {"label": "Category", "key": "category_id", "properties": ["category_id", "name"]},
        {"label": "Seller", "key": "seller_id", "properties": ["seller_id", "name", "rating"]},
    ],
    "relationships": [
        {"type": "PLACED", "source": "Customer", "target": "Order", "properties": []},
        {"type": "CONTAINS", "source": "Order", "target": "Product", "properties": ["quantity"]},
        {"type": "BELONGS_TO", "source": "Product", "target": "Category", "properties": []},
        {"type": "SOLD_BY", "source": "Product", "target": "Seller", "properties": []},
        {"type": "RATED", "source": "Customer", "target": "Product", "properties": ["stars"]},
    ],
    "data": {
        "Customer": [
            {"customer_id": "CU01", "name": "Aditi", "city": "Mumbai"},
            {"customer_id": "CU02", "name": "Rahul", "city": "Pune"},
            {"customer_id": "CU03", "name": "Priya", "city": "Delhi"},
            {"customer_id": "CU04", "name": "Arjun", "city": "Mumbai"},
        ],
        "Order": [
            {"order_id": "O1001", "order_date": "2026-01-12", "amount": 54990},
            {"order_id": "O1002", "order_date": "2026-01-15", "amount": 2499},
            {"order_id": "O1003", "order_date": "2026-02-02", "amount": 18990},
            {"order_id": "O1004", "order_date": "2026-02-09", "amount": 1299},
        ],
        "Product": [
            {"product_id": "P01", "name": "Laptop 14 inch", "price": 54990},
            {"product_id": "P02", "name": "Wireless Mouse", "price": 1299},
            {"product_id": "P03", "name": "Mechanical Keyboard", "price": 2499},
            {"product_id": "P04", "name": "Smartphone", "price": 18990},
            {"product_id": "P05", "name": "Running Shoes", "price": 3499},
        ],
        "Category": [
            {"category_id": "CAT1", "name": "Electronics"},
            {"category_id": "CAT2", "name": "Accessories"},
            {"category_id": "CAT3", "name": "Footwear"},
        ],
        "Seller": [
            {"seller_id": "SE1", "name": "TechWorld", "rating": 4.5},
            {"seller_id": "SE2", "name": "GadgetHub", "rating": 4.1},
            {"seller_id": "SE3", "name": "SportsKart", "rating": 3.9},
        ],
    },
    "edges": [
        {"type": "PLACED", "source": "CU01", "target": "O1001", "properties": {}},
        {"type": "PLACED", "source": "CU02", "target": "O1002", "properties": {}},
        {"type": "PLACED", "source": "CU03", "target": "O1003", "properties": {}},
        {"type": "PLACED", "source": "CU01", "target": "O1004", "properties": {}},
        {"type": "CONTAINS", "source": "O1001", "target": "P01", "properties": {"quantity": 1}},
        {"type": "CONTAINS", "source": "O1002", "target": "P03", "properties": {"quantity": 1}},
        {"type": "CONTAINS", "source": "O1003", "target": "P04", "properties": {"quantity": 1}},
        {"type": "CONTAINS", "source": "O1004", "target": "P02", "properties": {"quantity": 1}},
        {"type": "BELONGS_TO", "source": "P01", "target": "CAT1", "properties": {}},
        {"type": "BELONGS_TO", "source": "P02", "target": "CAT2", "properties": {}},
        {"type": "BELONGS_TO", "source": "P03", "target": "CAT2", "properties": {}},
        {"type": "BELONGS_TO", "source": "P04", "target": "CAT1", "properties": {}},
        {"type": "BELONGS_TO", "source": "P05", "target": "CAT3", "properties": {}},
        {"type": "SOLD_BY", "source": "P01", "target": "SE1", "properties": {}},
        {"type": "SOLD_BY", "source": "P02", "target": "SE2", "properties": {}},
        {"type": "SOLD_BY", "source": "P03", "target": "SE2", "properties": {}},
        {"type": "SOLD_BY", "source": "P04", "target": "SE1", "properties": {}},
        {"type": "SOLD_BY", "source": "P05", "target": "SE3", "properties": {}},
        {"type": "RATED", "source": "CU01", "target": "P01", "properties": {"stars": 5}},
        {"type": "RATED", "source": "CU02", "target": "P03", "properties": {"stars": 4}},
        {"type": "RATED", "source": "CU03", "target": "P04", "properties": {"stars": 4}},
        {"type": "RATED", "source": "CU04", "target": "P05", "properties": {"stars": 3}},
    ],
    "queries": [
        {"title": "List every customer", "cypher": "MATCH (c:Customer)\nRETURN c;"},
        {
            "title": "Customers and the orders they placed",
            "cypher": "MATCH (c:Customer)-[:PLACED]->(o:Order)\nRETURN c.name, o.order_id, o.amount;",
        },
        {
            "title": "What is inside each order",
            "cypher": "MATCH (o:Order)-[:CONTAINS]->(p:Product)\nRETURN o.order_id, p.name, p.price;",
        },
        {
            "title": "Products in the Electronics category",
            "cypher": (
                "MATCH (p:Product)-[:BELONGS_TO]->(cat:Category)\n"
                "WHERE cat.name = 'Electronics'\nRETURN p.name, p.price;"
            ),
        },
        {
            "title": "Products ordered by price",
            "cypher": "MATCH (p:Product)\nRETURN p.name, p.price\nORDER BY p.price DESC;",
        },
        {
            "title": "Recommendation hop: what customers bought from the same seller",
            "cypher": (
                "MATCH (c:Customer)-[:PLACED]->(:Order)-[:CONTAINS]->(:Product)-[:SOLD_BY]->(s:Seller)\n"
                "RETURN c.name AS customer, s.name AS seller;"
            ),
        },
        {
            "title": "Average rating a product received",
            "cypher": (
                "MATCH (:Customer)-[r:RATED]->(p:Product)\n"
                "RETURN p.name AS product, avg(r.stars) AS average_stars;"
            ),
        },
    ],
}


# ======================================================================
#  3. Healthcare
# ======================================================================
HEALTHCARE: Dict[str, Any] = {
    "name": "Healthcare",
    "description": (
        "A clinical knowledge graph connecting patients, their diagnoses, the "
        "doctors treating them, the medicines prescribed and the hospitals involved."
    ),
    "entity_hint": (
        "Patients, doctors, diseases, medicines and hospitals are entities. "
        "A diagnosis is a *relationship* between a patient and a disease, not a node."
    ),
    "nodes": [
        {"label": "Patient", "key": "patient_id", "properties": ["patient_id", "name", "age"]},
        {"label": "Doctor", "key": "doctor_id", "properties": ["doctor_id", "name", "specialization"]},
        {"label": "Disease", "key": "disease_id", "properties": ["disease_id", "name", "category"]},
        {"label": "Medicine", "key": "medicine_id", "properties": ["medicine_id", "name", "dosage"]},
        {"label": "Hospital", "key": "hospital_id", "properties": ["hospital_id", "name", "city"]},
    ],
    "relationships": [
        {"type": "DIAGNOSED_WITH", "source": "Patient", "target": "Disease", "properties": ["diagnosed_on"]},
        {"type": "TREATED_BY", "source": "Patient", "target": "Doctor", "properties": []},
        {"type": "PRESCRIBED", "source": "Doctor", "target": "Medicine", "properties": []},
        {"type": "TREATS", "source": "Medicine", "target": "Disease", "properties": []},
        {"type": "WORKS_AT", "source": "Doctor", "target": "Hospital", "properties": []},
    ],
    "data": {
        "Patient": [
            {"patient_id": "PT01", "name": "Aditi", "age": 34},
            {"patient_id": "PT02", "name": "Rahul", "age": 51},
            {"patient_id": "PT03", "name": "Priya", "age": 28},
            {"patient_id": "PT04", "name": "Arjun", "age": 62},
        ],
        "Doctor": [
            {"doctor_id": "DR01", "name": "Dr. Sharma", "specialization": "Cardiology"},
            {"doctor_id": "DR02", "name": "Dr. Mehta", "specialization": "Endocrinology"},
            {"doctor_id": "DR03", "name": "Dr. Rao", "specialization": "General Medicine"},
        ],
        "Disease": [
            {"disease_id": "DI01", "name": "Hypertension", "category": "Cardiovascular"},
            {"disease_id": "DI02", "name": "Type 2 Diabetes", "category": "Metabolic"},
            {"disease_id": "DI03", "name": "Asthma", "category": "Respiratory"},
        ],
        "Medicine": [
            {"medicine_id": "ME01", "name": "Amlodipine", "dosage": "5 mg"},
            {"medicine_id": "ME02", "name": "Metformin", "dosage": "500 mg"},
            {"medicine_id": "ME03", "name": "Salbutamol", "dosage": "100 mcg"},
        ],
        "Hospital": [
            {"hospital_id": "HO01", "name": "City General Hospital", "city": "Mumbai"},
            {"hospital_id": "HO02", "name": "Sunrise Medical Centre", "city": "Pune"},
        ],
    },
    "edges": [
        {"type": "DIAGNOSED_WITH", "source": "PT01", "target": "DI01", "properties": {"diagnosed_on": "2026-01-05"}},
        {"type": "DIAGNOSED_WITH", "source": "PT02", "target": "DI02", "properties": {"diagnosed_on": "2025-11-20"}},
        {"type": "DIAGNOSED_WITH", "source": "PT03", "target": "DI03", "properties": {"diagnosed_on": "2026-02-11"}},
        {"type": "DIAGNOSED_WITH", "source": "PT04", "target": "DI01", "properties": {"diagnosed_on": "2025-09-30"}},
        {"type": "TREATED_BY", "source": "PT01", "target": "DR01", "properties": {}},
        {"type": "TREATED_BY", "source": "PT02", "target": "DR02", "properties": {}},
        {"type": "TREATED_BY", "source": "PT03", "target": "DR03", "properties": {}},
        {"type": "TREATED_BY", "source": "PT04", "target": "DR01", "properties": {}},
        {"type": "PRESCRIBED", "source": "DR01", "target": "ME01", "properties": {}},
        {"type": "PRESCRIBED", "source": "DR02", "target": "ME02", "properties": {}},
        {"type": "PRESCRIBED", "source": "DR03", "target": "ME03", "properties": {}},
        {"type": "TREATS", "source": "ME01", "target": "DI01", "properties": {}},
        {"type": "TREATS", "source": "ME02", "target": "DI02", "properties": {}},
        {"type": "TREATS", "source": "ME03", "target": "DI03", "properties": {}},
        {"type": "WORKS_AT", "source": "DR01", "target": "HO01", "properties": {}},
        {"type": "WORKS_AT", "source": "DR02", "target": "HO01", "properties": {}},
        {"type": "WORKS_AT", "source": "DR03", "target": "HO02", "properties": {}},
    ],
    "queries": [
        {"title": "List every patient", "cypher": "MATCH (p:Patient)\nRETURN p;"},
        {
            "title": "Patients and their diagnoses",
            "cypher": "MATCH (p:Patient)-[:DIAGNOSED_WITH]->(d:Disease)\nRETURN p.name, d.name;",
        },
        {
            "title": "Which doctor treats which patient",
            "cypher": "MATCH (p:Patient)-[:TREATED_BY]->(d:Doctor)\nRETURN p.name, d.name, d.specialization;",
        },
        {
            "title": "Patients older than 50",
            "cypher": "MATCH (p:Patient)\nWHERE p.age > 50\nRETURN p.name, p.age\nORDER BY p.age DESC;",
        },
        {
            "title": "Medicine that treats each patient's disease",
            "cypher": (
                "MATCH (p:Patient)-[:DIAGNOSED_WITH]->(d:Disease)<-[:TREATS]-(m:Medicine)\n"
                "RETURN p.name AS patient, d.name AS disease, m.name AS medicine;"
            ),
        },
        {
            "title": "How many patients each hospital serves",
            "cypher": (
                "MATCH (p:Patient)-[:TREATED_BY]->(:Doctor)-[:WORKS_AT]->(h:Hospital)\n"
                "RETURN h.name AS hospital, count(p) AS patients;"
            ),
        },
    ],
}


# ======================================================================
#  4. Movie Recommendation
# ======================================================================
MOVIES: Dict[str, Any] = {
    "name": "Movie Recommendation",
    "description": (
        "A recommendation knowledge graph linking users, the movies they rate, "
        "the actors and directors behind those movies, and the movie genres."
    ),
    "entity_hint": (
        "Users, movies, actors, directors and genres are entities. The rating a "
        "user gives is a property *of the relationship*, not of the movie."
    ),
    "nodes": [
        {"label": "User", "key": "user_id", "properties": ["user_id", "name", "age"]},
        {"label": "Movie", "key": "movie_id", "properties": ["movie_id", "title", "year"]},
        {"label": "Actor", "key": "actor_id", "properties": ["actor_id", "name"]},
        {"label": "Director", "key": "director_id", "properties": ["director_id", "name"]},
        {"label": "Genre", "key": "genre_id", "properties": ["genre_id", "name"]},
    ],
    "relationships": [
        {"type": "RATED", "source": "User", "target": "Movie", "properties": ["rating"]},
        {"type": "ACTED_IN", "source": "Actor", "target": "Movie", "properties": ["role"]},
        {"type": "DIRECTED", "source": "Director", "target": "Movie", "properties": []},
        {"type": "IN_GENRE", "source": "Movie", "target": "Genre", "properties": []},
        {"type": "FOLLOWS", "source": "User", "target": "User", "properties": []},
    ],
    "data": {
        "User": [
            {"user_id": "U01", "name": "Aditi", "age": 22},
            {"user_id": "U02", "name": "Rahul", "age": 25},
            {"user_id": "U03", "name": "Priya", "age": 21},
            {"user_id": "U04", "name": "Arjun", "age": 30},
        ],
        "Movie": [
            {"movie_id": "M01", "title": "The Deep Field", "year": 2019},
            {"movie_id": "M02", "title": "Monsoon Lines", "year": 2021},
            {"movie_id": "M03", "title": "Circuit City", "year": 2023},
            {"movie_id": "M04", "title": "Paper Boats", "year": 2020},
        ],
        "Actor": [
            {"actor_id": "A01", "name": "Kiran Desai"},
            {"actor_id": "A02", "name": "Meera Iyer"},
            {"actor_id": "A03", "name": "Vikram Nair"},
        ],
        "Director": [
            {"director_id": "D01", "name": "Anand Kulkarni"},
            {"director_id": "D02", "name": "Sara Pillai"},
        ],
        "Genre": [
            {"genre_id": "G01", "name": "Science Fiction"},
            {"genre_id": "G02", "name": "Drama"},
            {"genre_id": "G03", "name": "Thriller"},
        ],
    },
    "edges": [
        {"type": "RATED", "source": "U01", "target": "M01", "properties": {"rating": 5}},
        {"type": "RATED", "source": "U01", "target": "M03", "properties": {"rating": 4}},
        {"type": "RATED", "source": "U02", "target": "M01", "properties": {"rating": 4}},
        {"type": "RATED", "source": "U02", "target": "M02", "properties": {"rating": 3}},
        {"type": "RATED", "source": "U03", "target": "M04", "properties": {"rating": 5}},
        {"type": "RATED", "source": "U04", "target": "M03", "properties": {"rating": 2}},
        {"type": "ACTED_IN", "source": "A01", "target": "M01", "properties": {"role": "Lead"}},
        {"type": "ACTED_IN", "source": "A01", "target": "M03", "properties": {"role": "Lead"}},
        {"type": "ACTED_IN", "source": "A02", "target": "M02", "properties": {"role": "Lead"}},
        {"type": "ACTED_IN", "source": "A02", "target": "M04", "properties": {"role": "Support"}},
        {"type": "ACTED_IN", "source": "A03", "target": "M03", "properties": {"role": "Support"}},
        {"type": "DIRECTED", "source": "D01", "target": "M01", "properties": {}},
        {"type": "DIRECTED", "source": "D01", "target": "M04", "properties": {}},
        {"type": "DIRECTED", "source": "D02", "target": "M02", "properties": {}},
        {"type": "DIRECTED", "source": "D02", "target": "M03", "properties": {}},
        {"type": "IN_GENRE", "source": "M01", "target": "G01", "properties": {}},
        {"type": "IN_GENRE", "source": "M02", "target": "G02", "properties": {}},
        {"type": "IN_GENRE", "source": "M03", "target": "G03", "properties": {}},
        {"type": "IN_GENRE", "source": "M04", "target": "G02", "properties": {}},
        {"type": "FOLLOWS", "source": "U01", "target": "U02", "properties": {}},
        {"type": "FOLLOWS", "source": "U03", "target": "U01", "properties": {}},
        {"type": "FOLLOWS", "source": "U04", "target": "U02", "properties": {}},
    ],
    "queries": [
        {"title": "List every movie", "cypher": "MATCH (m:Movie)\nRETURN m;"},
        {
            "title": "Who rated what, and how highly",
            "cypher": "MATCH (u:User)-[r:RATED]->(m:Movie)\nRETURN u.name, m.title, r.rating;",
        },
        {
            "title": "Cast of each movie",
            "cypher": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)\nRETURN m.title, a.name;",
        },
        {
            "title": "Highly rated movies only",
            "cypher": (
                "MATCH (u:User)-[r:RATED]->(m:Movie)\nWHERE r.rating >= 4\n"
                "RETURN m.title, r.rating\nORDER BY r.rating DESC;"
            ),
        },
        {
            "title": "Recommendation: movies liked by people you follow",
            "cypher": (
                "MATCH (u:User)-[:FOLLOWS]->(friend:User)-[r:RATED]->(m:Movie)\n"
                "WHERE r.rating >= 4\n"
                "RETURN u.name AS user, m.title AS recommended, friend.name AS because_of;"
            ),
        },
        {
            "title": "Average rating per movie",
            "cypher": "MATCH (:User)-[r:RATED]->(m:Movie)\nRETURN m.title, avg(r.rating) AS avg_rating;",
        },
    ],
}


# ======================================================================
#  5. Library
# ======================================================================
LIBRARY: Dict[str, Any] = {
    "name": "Library",
    "description": (
        "A library knowledge graph linking members, the books they borrow, the "
        "authors who wrote them, the publishers and the genres."
    ),
    "entity_hint": (
        "Members, books, authors, publishers and genres are entities. Borrowing "
        "is a relationship, and the issue date is a property of that relationship."
    ),
    "nodes": [
        {"label": "Member", "key": "member_id", "properties": ["member_id", "name", "membership_type"]},
        {"label": "Book", "key": "book_id", "properties": ["book_id", "title", "isbn"]},
        {"label": "Author", "key": "author_id", "properties": ["author_id", "name", "country"]},
        {"label": "Publisher", "key": "publisher_id", "properties": ["publisher_id", "name"]},
        {"label": "Genre", "key": "genre_id", "properties": ["genre_id", "name"]},
    ],
    "relationships": [
        {"type": "BORROWED", "source": "Member", "target": "Book", "properties": ["issue_date"]},
        {"type": "WROTE", "source": "Author", "target": "Book", "properties": []},
        {"type": "PUBLISHED_BY", "source": "Book", "target": "Publisher", "properties": []},
        {"type": "IN_GENRE", "source": "Book", "target": "Genre", "properties": []},
        {"type": "RESERVED", "source": "Member", "target": "Book", "properties": []},
    ],
    "data": {
        "Member": [
            {"member_id": "MB01", "name": "Aditi", "membership_type": "Student"},
            {"member_id": "MB02", "name": "Rahul", "membership_type": "Student"},
            {"member_id": "MB03", "name": "Priya", "membership_type": "Faculty"},
            {"member_id": "MB04", "name": "Arjun", "membership_type": "Student"},
        ],
        "Book": [
            {"book_id": "BK01", "title": "Database System Concepts", "isbn": "978-0078022159"},
            {"book_id": "BK02", "title": "Computer Networks", "isbn": "978-0132126953"},
            {"book_id": "BK03", "title": "Artificial Intelligence: A Modern Approach", "isbn": "978-0134610993"},
            {"book_id": "BK04", "title": "Graph Databases", "isbn": "978-1491930892"},
        ],
        "Author": [
            {"author_id": "AU01", "name": "Abraham Silberschatz", "country": "USA"},
            {"author_id": "AU02", "name": "Andrew S. Tanenbaum", "country": "Netherlands"},
            {"author_id": "AU03", "name": "Stuart Russell", "country": "UK"},
            {"author_id": "AU04", "name": "Ian Robinson", "country": "UK"},
        ],
        "Publisher": [
            {"publisher_id": "PB01", "name": "McGraw-Hill"},
            {"publisher_id": "PB02", "name": "Pearson"},
            {"publisher_id": "PB03", "name": "O'Reilly Media"},
        ],
        "Genre": [
            {"genre_id": "GN01", "name": "Databases"},
            {"genre_id": "GN02", "name": "Networking"},
            {"genre_id": "GN03", "name": "Artificial Intelligence"},
        ],
    },
    "edges": [
        {"type": "BORROWED", "source": "MB01", "target": "BK01", "properties": {"issue_date": "2026-01-08"}},
        {"type": "BORROWED", "source": "MB01", "target": "BK04", "properties": {"issue_date": "2026-01-20"}},
        {"type": "BORROWED", "source": "MB02", "target": "BK02", "properties": {"issue_date": "2026-02-02"}},
        {"type": "BORROWED", "source": "MB03", "target": "BK03", "properties": {"issue_date": "2026-02-10"}},
        {"type": "BORROWED", "source": "MB04", "target": "BK01", "properties": {"issue_date": "2026-02-14"}},
        {"type": "RESERVED", "source": "MB02", "target": "BK04", "properties": {}},
        {"type": "RESERVED", "source": "MB04", "target": "BK03", "properties": {}},
        {"type": "WROTE", "source": "AU01", "target": "BK01", "properties": {}},
        {"type": "WROTE", "source": "AU02", "target": "BK02", "properties": {}},
        {"type": "WROTE", "source": "AU03", "target": "BK03", "properties": {}},
        {"type": "WROTE", "source": "AU04", "target": "BK04", "properties": {}},
        {"type": "PUBLISHED_BY", "source": "BK01", "target": "PB01", "properties": {}},
        {"type": "PUBLISHED_BY", "source": "BK02", "target": "PB02", "properties": {}},
        {"type": "PUBLISHED_BY", "source": "BK03", "target": "PB02", "properties": {}},
        {"type": "PUBLISHED_BY", "source": "BK04", "target": "PB03", "properties": {}},
        {"type": "IN_GENRE", "source": "BK01", "target": "GN01", "properties": {}},
        {"type": "IN_GENRE", "source": "BK02", "target": "GN02", "properties": {}},
        {"type": "IN_GENRE", "source": "BK03", "target": "GN03", "properties": {}},
        {"type": "IN_GENRE", "source": "BK04", "target": "GN01", "properties": {}},
    ],
    "queries": [
        {"title": "List every book", "cypher": "MATCH (b:Book)\nRETURN b;"},
        {
            "title": "Who borrowed which book",
            "cypher": "MATCH (m:Member)-[r:BORROWED]->(b:Book)\nRETURN m.name, b.title, r.issue_date;",
        },
        {
            "title": "Author of each book",
            "cypher": "MATCH (a:Author)-[:WROTE]->(b:Book)\nRETURN b.title, a.name, a.country;",
        },
        {
            "title": "Books in the Databases genre",
            "cypher": (
                "MATCH (b:Book)-[:IN_GENRE]->(g:Genre)\nWHERE g.name = 'Databases'\nRETURN b.title, b.isbn;"
            ),
        },
        {
            "title": "How many times each book was borrowed",
            "cypher": (
                "MATCH (:Member)-[:BORROWED]->(b:Book)\n"
                "RETURN b.title AS book, count(*) AS times_borrowed\nORDER BY times_borrowed DESC;"
            ),
        },
        {
            "title": "Readers who borrowed the same book",
            "cypher": (
                "MATCH (m1:Member)-[:BORROWED]->(b:Book)<-[:BORROWED]-(m2:Member)\n"
                "RETURN DISTINCT m1.name, m2.name, b.title;"
            ),
        },
    ],
}


DOMAINS: Dict[str, Dict[str, Any]] = {
    UNIVERSITY["name"]: UNIVERSITY,
    ECOMMERCE["name"]: ECOMMERCE,
    HEALTHCARE["name"]: HEALTHCARE,
    MOVIES["name"]: MOVIES,
    LIBRARY["name"]: LIBRARY,
}


# ======================================================================
#  Helpers
# ======================================================================
def domain_names() -> List[str]:
    return list(DOMAINS.keys())


def get_domain(name: str) -> Dict[str, Any]:
    """Return a domain by name, falling back to the default domain."""
    return DOMAINS.get(name, DOMAINS[DEFAULT_DOMAIN])


def key_property(domain: Dict[str, Any], label: str) -> Optional[str]:
    for node in domain["nodes"]:
        if node["label"] == label:
            return node["key"]
    return None


def label_of_id(domain: Dict[str, Any], entity_id: Any) -> Optional[str]:
    """Find which label owns a given key value in the domain's sample data."""
    for label, rows in domain["data"].items():
        key = key_property(domain, label)
        for row in rows:
            if key and str(row.get(key)) == str(entity_id):
                return label
    return None


def schema_from_domain(domain: Dict[str, Any]) -> Dict[str, Any]:
    """Build the editable schema used by the Schema Designer."""
    return {
        "domain": domain["name"],
        "nodes": [
            {
                "label": node["label"],
                "key": node["key"],
                "properties": list(node["properties"]),
            }
            for node in domain["nodes"]
        ],
        "relationships": [
            {
                "type": rel["type"],
                "source": rel["source"],
                "target": rel["target"],
                "properties": list(rel["properties"]),
            }
            for rel in domain["relationships"]
        ],
    }


def dataset_from_domain(domain: Dict[str, Any]) -> Dict[str, Any]:
    """Build the working dataset (nodes + edges) used by the import step."""
    nodes: List[Dict[str, Any]] = []
    for label, rows in domain["data"].items():
        key = key_property(domain, label)
        for row in rows:
            nodes.append(
                {"label": label, "id": row.get(key), "properties": dict(row)}
            )
    edges: List[Dict[str, Any]] = []
    for edge in domain["edges"]:
        edges.append(
            {
                "type": edge["type"],
                "source": edge["source"],
                "source_label": label_of_id(domain, edge["source"]),
                "target": edge["target"],
                "target_label": label_of_id(domain, edge["target"]),
                "properties": dict(edge.get("properties") or {}),
            }
        )
    return {"source": "Built-in sample dataset (%s)" % domain["name"], "nodes": nodes, "edges": edges}


def nodes_dataframe(dataset: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the dataset nodes into one table (blank cells where a property
    does not apply to that label)."""
    rows = []
    for node in dataset["nodes"]:
        row: Dict[str, Any] = {"label": node["label"], "id": node["id"]}
        row.update(node["properties"])
        rows.append(row)
    # dtype=object keeps whole numbers as ints (pandas would otherwise turn a
    # column with gaps into floats and write "4.0"); astype(str) then gives a
    # single, Arrow-friendly column type for display without changing the text.
    return pd.DataFrame(rows, dtype=object).fillna("").astype(str)


def edges_dataframe(dataset: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for edge in dataset["edges"]:
        row: Dict[str, Any] = {
            "source_id": edge["source"],
            "source_label": edge.get("source_label") or "",
            "type": edge["type"],
            "target_id": edge["target"],
            "target_label": edge.get("target_label") or "",
        }
        row.update(edge.get("properties") or {})
        rows.append(row)
    return pd.DataFrame(rows, dtype=object).fillna("").astype(str)


def label_frames(domain: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """One tidy dataframe per node label, for inspection before import."""
    return {label: pd.DataFrame(rows) for label, rows in domain["data"].items()}


def validate_domain_data(domain: Dict[str, Any]) -> List[str]:
    """Self-check a domain definition; returns a list of problems (empty = fine)."""
    problems: List[str] = []
    ids: Dict[str, str] = {}
    for label, rows in domain["data"].items():
        key = key_property(domain, label)
        if key is None:
            problems.append("%s: no key property declared" % label)
            continue
        declared = set()
        for node in domain["nodes"]:
            if node["label"] == label:
                declared = set(node["properties"])
        for row in rows:
            if key not in row:
                problems.append("%s: a row is missing the key %s" % (label, key))
                continue
            value = str(row[key])
            if value in ids:
                problems.append("duplicate id %s (%s and %s)" % (value, ids[value], label))
            ids[value] = label
            extra = set(row) - declared
            if extra:
                problems.append("%s: undeclared properties %s" % (label, sorted(extra)))
    declared_types = {
        (r["type"], r["source"], r["target"]) for r in domain["relationships"]
    }
    for edge in domain["edges"]:
        src, tgt = str(edge["source"]), str(edge["target"])
        if src not in ids:
            problems.append("edge %s: source %s does not exist" % (edge["type"], src))
        if tgt not in ids:
            problems.append("edge %s: target %s does not exist" % (edge["type"], tgt))
        if src in ids and tgt in ids:
            triple = (edge["type"], ids[src], ids[tgt])
            if triple not in declared_types:
                problems.append("edge %s is not declared in the schema" % str(triple))
    return problems
