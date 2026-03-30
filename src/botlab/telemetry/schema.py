from __future__ import annotations

import sqlite3


CREATE_CYCLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL UNIQUE,
    event_ts REAL NOT NULL,
    expected_spawn_ts REAL,
    actual_spawn_ts REAL,
    drift_s REAL,
    reason TEXT NOT NULL DEFAULT '',
    reaction_ms REAL,
    verification_ms REAL,
    result TEXT,
    final_state TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""

CREATE_STATE_TRANSITIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER,
    event_ts REAL NOT NULL,
    state_enter TEXT,
    state_exit TEXT,
    reason TEXT NOT NULL DEFAULT '',
    final_state TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""

CREATE_ATTEMPTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER,
    event_ts REAL NOT NULL,
    state TEXT NOT NULL,
    expected_spawn_ts REAL,
    actual_spawn_ts REAL,
    reaction_ms REAL,
    verification_ms REAL,
    result TEXT,
    reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""

CREATE_CYCLES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_cycles_cycle_id
ON cycles (cycle_id);
"""

CREATE_STATE_TRANSITIONS_CYCLE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_state_transitions_cycle_id
ON state_transitions (cycle_id);
"""

CREATE_STATE_TRANSITIONS_EVENT_TS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_state_transitions_event_ts
ON state_transitions (event_ts);
"""

CREATE_ATTEMPTS_CYCLE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_attempts_cycle_id
ON attempts (cycle_id);
"""

CREATE_ATTEMPTS_EVENT_TS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_attempts_event_ts
ON attempts (event_ts);
"""


def initialize_schema(connection: sqlite3.Connection) -> None:
    """
    Tworzy pełny minimalny schemat telemetry dla MVP.

    Tabele:
    - cycles
    - state_transitions
    - attempts
    """
    statements = [
        CREATE_CYCLES_TABLE_SQL,
        CREATE_STATE_TRANSITIONS_TABLE_SQL,
        CREATE_ATTEMPTS_TABLE_SQL,
        CREATE_CYCLES_INDEX_SQL,
        CREATE_STATE_TRANSITIONS_CYCLE_INDEX_SQL,
        CREATE_STATE_TRANSITIONS_EVENT_TS_INDEX_SQL,
        CREATE_ATTEMPTS_CYCLE_INDEX_SQL,
        CREATE_ATTEMPTS_EVENT_TS_INDEX_SQL,
    ]

    with connection:
        for statement in statements:
            connection.execute(statement)
