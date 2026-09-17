"""
neo4j_service.py
================
Thin, defensive wrapper around the official Neo4j Python driver.

Design rules for this laboratory:

* A connection is only ever reported as *connected* after a real round trip to
  the server -- the lab never pretends a database is available.
* Every failure comes back as ``(False, friendly_message)`` instead of an
  exception, so the Streamlit interface can fall back to Local Simulation Mode.
* Results are converted into the same :class:`~simulation_engine.QueryResult`
  used by the local engine, so the rest of the app does not care which backend
  produced them.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from simulation_engine import NEO4J_MODE, QueryResult, SimNode, SimRel, split_statements

try:  # The driver is optional -- the lab still runs in simulation mode without it.
    from neo4j import GraphDatabase

    DRIVER_AVAILABLE = True
    DRIVER_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on the environment
    GraphDatabase = None  # type: ignore[assignment]
    DRIVER_AVAILABLE = False
    DRIVER_IMPORT_ERROR = str(exc)


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_DATABASE = "neo4j"


def env_defaults() -> Dict[str, str]:
    """Connection defaults taken from environment variables, if present."""
    return {
        "uri": os.environ.get("NEO4J_URI", DEFAULT_URI),
        "username": os.environ.get("NEO4J_USERNAME", DEFAULT_USER),
        "password": os.environ.get("NEO4J_PASSWORD", ""),
        "database": os.environ.get("NEO4J_DATABASE", DEFAULT_DATABASE),
    }


def env_password_present() -> bool:
    return bool(os.environ.get("NEO4J_PASSWORD"))


def _friendly_error(exc: Exception) -> str:
    """Translate driver exceptions into something a student can act on."""
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()
    if "authentication" in lowered or "unauthorized" in lowered:
        return (
            "Neo4j refused the username or password. Check the credentials you "
            "set when the database was first started."
        )
    if "serviceunavailable" in name.lower() or "unable to connect" in lowered or "connection refused" in lowered:
        return (
            "No Neo4j server answered at that address. Make sure Neo4j Desktop or "
            "the Neo4j service is running and that the Bolt port is correct."
        )
    if "configuration" in name.lower() or "scheme" in lowered:
        return (
            "The URI looks wrong. Use a scheme such as bolt://localhost:7687 or "
            "neo4j+s://<your-aura-host>."
        )
    if "database" in lowered and "not exist" in lowered:
        return "That database name does not exist on the server."
    if "syntax" in lowered:
        return "Neo4j reported a Cypher syntax error: %s" % text.split("\n")[0]
    return "%s: %s" % (name, text.split("\n")[0])


def _entity_id(entity: Any) -> Any:
    """Neo4j 5 prefers element_id; Neo4j 4 only has the numeric id."""
    element_id = getattr(entity, "element_id", None)
    if element_id is not None:
        return element_id
    try:
        return entity.id
    except Exception:  # pragma: no cover
        return id(entity)


def _convert(value: Any) -> Any:
    """Convert driver types into the lab's own node/relationship objects."""
    try:
        from neo4j.graph import Node, Relationship, Path
    except Exception:  # pragma: no cover
        return value

    if isinstance(value, Node):
        return SimNode(_entity_id(value), list(value.labels), dict(value))
    if isinstance(value, Relationship):
        return SimRel(
            _entity_id(value),
            value.type,
            _entity_id(value.start_node) if value.start_node is not None else None,
            _entity_id(value.end_node) if value.end_node is not None else None,
            dict(value),
        )
    if isinstance(value, Path):
        return [_convert(rel) for rel in value.relationships]
    if isinstance(value, (list, tuple)):
        return [_convert(v) for v in value]
    if isinstance(value, dict):
        return {k: _convert(v) for k, v in value.items()}
    return value


class Neo4jService:
    """Holds (at most) one live driver and exposes the operations the lab needs."""

    def __init__(self) -> None:
        self._driver: Any = None
        self.uri: str = ""
        self.username: str = ""
        self.database: str = DEFAULT_DATABASE
        self.server_info: str = ""
        self.last_error: str = ""

    # -- connection ---------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def connect(
        self, uri: str, username: str, password: str, database: str
    ) -> Tuple[bool, str]:
        """Open a driver and verify it with a real query. Never raises."""
        self.disconnect()
        if not DRIVER_AVAILABLE:
            self.last_error = (
                "The neo4j Python driver is not installed. Run "
                "'pip install -r requirements.txt' to add it."
            )
            return False, self.last_error
        if not uri.strip():
            self.last_error = "Please enter a Neo4j URI, for example bolt://localhost:7687."
            return False, self.last_error

        try:
            driver = GraphDatabase.driver(uri.strip(), auth=(username, password))
            # A real round trip -- this is what makes the status trustworthy.
            with driver.session(database=database or DEFAULT_DATABASE) as session:
                record = session.run("RETURN 1 AS ok").single()
                if not record or record["ok"] != 1:
                    driver.close()
                    self.last_error = "The server answered, but the test query failed."
                    return False, self.last_error
        except Exception as exc:
            self.last_error = _friendly_error(exc)
            return False, self.last_error

        self._driver = driver
        self.uri = uri.strip()
        self.username = username
        self.database = database or DEFAULT_DATABASE
        self.server_info = self._read_server_info()
        self.last_error = ""
        return True, "Connected to Neo4j at %s (database: %s)." % (self.uri, self.database)

    def test_connection(
        self, uri: str, username: str, password: str, database: str
    ) -> Tuple[bool, str]:
        """Check credentials without changing the current connection."""
        if not DRIVER_AVAILABLE:
            return False, (
                "The neo4j Python driver is not installed. Run "
                "'pip install -r requirements.txt' to add it."
            )
        driver = None
        try:
            driver = GraphDatabase.driver(uri.strip(), auth=(username, password))
            with driver.session(database=database or DEFAULT_DATABASE) as session:
                session.run("RETURN 1").single()
            return True, "Neo4j answered successfully at %s." % uri.strip()
        except Exception as exc:
            return False, _friendly_error(exc)
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass

    def disconnect(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
        self._driver = None
        self.server_info = ""

    def _read_server_info(self) -> str:
        result = self.run_query(
            "CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS version"
        )
        if result.ok and result.rows:
            return "%s %s" % (result.rows[0][0], result.rows[0][1])
        return "Neo4j"

    # -- queries ------------------------------------------------------
    def run_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute one statement on the server and return a uniform result."""
        outcome = QueryResult(mode=NEO4J_MODE, query=query)
        started = time.perf_counter()
        text = (query or "").strip().rstrip(";").strip()
        if not text:
            outcome.error = "The query is empty. Type a Cypher statement such as MATCH (n) RETURN n."
            return outcome
        if not self.is_connected:
            outcome.error = "Not connected to Neo4j."
            return outcome
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(text, parameters or {})
                outcome.columns = list(result.keys())
                outcome.rows = [[_convert(v) for v in record.values()] for record in result]
                counters = result.consume().counters
                summary = {
                    "nodes_created": counters.nodes_created,
                    "relationships_created": counters.relationships_created,
                    "properties_set": counters.properties_set,
                    "nodes_deleted": counters.nodes_deleted,
                    "relationships_deleted": counters.relationships_deleted,
                    "labels_added": counters.labels_added,
                }
                outcome.summary = {k: v for k, v in summary.items() if v}
        except Exception as exc:
            outcome.error = _friendly_error(exc)
        outcome.execution_ms = (time.perf_counter() - started) * 1000
        return outcome

    def run_script(
        self, script: str, parameters: Optional[Dict[str, Any]] = None
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
            result = self.run_query(statement, parameters)
            totals["statements"] += 1
            if result.ok:
                for key, value in result.summary.items():
                    totals[key] = totals.get(key, 0) + value
            else:
                errors.append("%s  ->  %s" % (statement.splitlines()[0][:70], result.error))
        return totals, errors

    # -- introspection ------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Node/relationship counts straight from the server."""
        empty = {
            "node_count": 0,
            "relationship_count": 0,
            "label_count": 0,
            "relationship_type_count": 0,
            "avg_rels_per_node": 0.0,
            "label_counts": {},
            "relationship_type_counts": {},
        }
        if not self.is_connected:
            return empty
        nodes = self.run_query("MATCH (n) RETURN count(n) AS c")
        rels = self.run_query("MATCH ()-[r]->() RETURN count(r) AS c")
        labels = self.run_query(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label"
        )
        types = self.run_query(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS c ORDER BY type"
        )
        if not (nodes.ok and rels.ok):
            return empty
        node_count = nodes.rows[0][0] if nodes.rows else 0
        rel_count = rels.rows[0][0] if rels.rows else 0
        label_counts = {row[0]: row[1] for row in labels.rows} if labels.ok else {}
        type_counts = {row[0]: row[1] for row in types.rows} if types.ok else {}
        return {
            "node_count": node_count,
            "relationship_count": rel_count,
            "label_count": len(label_counts),
            "relationship_type_count": len(type_counts),
            "avg_rels_per_node": round(rel_count / node_count, 2) if node_count else 0.0,
            "label_counts": label_counts,
            "relationship_type_counts": type_counts,
        }

    def _supports_element_id(self) -> bool:
        """Cache whether the server understands elementId() (Neo4j 5+)."""
        if getattr(self, "_element_id_ok", None) is None:
            probe = self.run_query("MATCH (n) RETURN elementId(n) LIMIT 1")
            self._element_id_ok = probe.ok
        return bool(self._element_id_ok)

    def fetch_graph(self, limit: int = 300) -> Dict[str, Any]:
        """Pull a bounded slice of the stored graph for visualisation."""
        snapshot: Dict[str, Any] = {"nodes": [], "relationships": []}
        if not self.is_connected:
            return snapshot
        # elementId() is the Neo4j 5 function; id() is the Neo4j 4 fallback.
        id_fn = "elementId" if self._supports_element_id() else "id"
        nodes = self.run_query(
            "MATCH (n) RETURN %s(n) AS nid, labels(n) AS labels, properties(n) AS props "
            "LIMIT $limit" % id_fn,
            {"limit": limit},
        )
        if nodes.ok:
            for nid, labels, props in nodes.rows:
                node = SimNode(nid, list(labels or []), dict(props or {}))
                snapshot["nodes"].append(
                    {
                        "nid": nid,
                        "label": node.label,
                        "caption": node.caption(),
                        "properties": dict(props or {}),
                    }
                )
        known = {n["nid"] for n in snapshot["nodes"]}
        rels = self.run_query(
            "MATCH (a)-[r]->(b) RETURN %s(r) AS rid, %s(a) AS s, %s(b) AS t, "
            "type(r) AS type, properties(r) AS props LIMIT $limit" % (id_fn, id_fn, id_fn),
            {"limit": limit * 3},
        )
        if rels.ok:
            for rid, source, target, rtype, props in rels.rows:
                if source in known and target in known:
                    snapshot["relationships"].append(
                        {
                            "rid": rid,
                            "type": rtype,
                            "source": source,
                            "target": target,
                            "properties": dict(props or {}),
                        }
                    )
        return snapshot

    def clear_database(self) -> QueryResult:
        """Remove every node and relationship (used by the 'reset' control)."""
        return self.run_query("MATCH (n) DETACH DELETE n")
