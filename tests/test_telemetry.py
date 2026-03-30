from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from botlab.config import TelemetryConfig
from botlab.telemetry.logger import configure_telemetry_logger, log_telemetry_record
from botlab.telemetry.storage import SQLiteTelemetryStorage
from botlab.types import BotState, TelemetryRecord


def test_configure_telemetry_logger_creates_log_file_and_writes_record(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "botlab.log"
    sqlite_path = tmp_path / "data" / "telemetry.sqlite3"

    telemetry_config = TelemetryConfig(
        sqlite_path=sqlite_path,
        log_path=log_path,
        log_level="INFO",
    )

    logger = configure_telemetry_logger(
        telemetry_config=telemetry_config,
        logger_name="botlab.test.telemetry.logger",
        enable_console=False,
    )

    record = TelemetryRecord(
        cycle_id=1,
        event_ts=100.5,
        state=BotState.READY_WINDOW,
        expected_spawn_ts=100.0,
        actual_spawn_ts=100.1,
        drift_s=0.1,
        state_enter=BotState.PREPARE_WINDOW,
        state_exit=BotState.READY_WINDOW,
        reason="ready_window_opened",
        reaction_ms=None,
        verification_ms=None,
        result="pending",
        final_state=BotState.READY_WINDOW,
        metadata={"source": "test"},
    )

    log_telemetry_record(logger, record, level=logging.INFO)

    assert log_path.exists() is True

    content = log_path.read_text(encoding="utf-8")
    assert '"cycle_id": 1' in content
    assert '"state": "READY_WINDOW"' in content
    assert '"reason": "ready_window_opened"' in content


def test_storage_initialize_creates_schema_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry" / "botlab.sqlite3"
    storage = SQLiteTelemetryStorage(db_path)

    storage.initialize()

    assert db_path.exists() is True

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name ASC
            """
        ).fetchall()

    table_names = {row[0] for row in rows}

    assert "attempts" in table_names
    assert "cycles" in table_names
    assert "state_transitions" in table_names


def test_storage_can_record_and_fetch_cycle_state_transition_and_attempt(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry" / "botlab.sqlite3"
    storage = SQLiteTelemetryStorage(db_path)
    storage.initialize()

    cycle_record = TelemetryRecord(
        cycle_id=10,
        event_ts=145.0,
        state=BotState.WAIT_NEXT_CYCLE,
        expected_spawn_ts=145.0,
        actual_spawn_ts=145.2,
        drift_s=0.2,
        state_enter=None,
        state_exit=None,
        reason="cycle_finished",
        reaction_ms=42.0,
        verification_ms=110.0,
        result="success",
        final_state=BotState.COMBAT,
        metadata={"note": "cycle-summary"},
    )

    transition_record = TelemetryRecord(
        cycle_id=10,
        event_ts=144.0,
        state=BotState.READY_WINDOW,
        expected_spawn_ts=145.0,
        actual_spawn_ts=None,
        drift_s=None,
        state_enter=BotState.PREPARE_WINDOW,
        state_exit=BotState.READY_WINDOW,
        reason="entered_ready_window",
        reaction_ms=None,
        verification_ms=None,
        result=None,
        final_state=BotState.READY_WINDOW,
        metadata={"transition_index": 2},
    )

    attempt_record = TelemetryRecord(
        cycle_id=10,
        event_ts=145.1,
        state=BotState.ATTEMPT,
        expected_spawn_ts=145.0,
        actual_spawn_ts=145.2,
        drift_s=0.2,
        state_enter=BotState.READY_WINDOW,
        state_exit=BotState.ATTEMPT,
        reason="signal_detected",
        reaction_ms=38.0,
        verification_ms=85.0,
        result="attempt_sent",
        final_state=BotState.VERIFY,
        metadata={"attempt_index": 1},
    )

    storage.record_cycle(cycle_record)
    storage.record_state_transition(transition_record)
    storage.record_attempt(attempt_record)

    cycles = storage.fetch_cycles()
    transitions = storage.fetch_state_transitions()
    attempts = storage.fetch_attempts()

    assert len(cycles) == 1
    assert len(transitions) == 1
    assert len(attempts) == 1

    assert cycles[0]["cycle_id"] == 10
    assert cycles[0]["result"] == "success"
    assert cycles[0]["final_state"] == "COMBAT"
    assert cycles[0]["metadata"]["note"] == "cycle-summary"

    assert transitions[0]["cycle_id"] == 10
    assert transitions[0]["state_enter"] == "PREPARE_WINDOW"
    assert transitions[0]["state_exit"] == "READY_WINDOW"
    assert transitions[0]["reason"] == "entered_ready_window"

    assert attempts[0]["cycle_id"] == 10
    assert attempts[0]["state"] == "ATTEMPT"
    assert attempts[0]["reaction_ms"] == 38.0
    assert attempts[0]["result"] == "attempt_sent"


def test_record_cycle_uses_upsert_for_same_cycle_id(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry" / "botlab.sqlite3"
    storage = SQLiteTelemetryStorage(db_path)
    storage.initialize()

    first_record = TelemetryRecord(
        cycle_id=3,
        event_ts=45.0,
        state=BotState.WAIT_NEXT_CYCLE,
        expected_spawn_ts=45.0,
        actual_spawn_ts=None,
        drift_s=None,
        state_enter=None,
        state_exit=None,
        reason="cycle_created",
        reaction_ms=None,
        verification_ms=None,
        result="pending",
        final_state=BotState.WAIT_NEXT_CYCLE,
        metadata={"version": 1},
    )

    second_record = TelemetryRecord(
        cycle_id=3,
        event_ts=46.0,
        state=BotState.VERIFY,
        expected_spawn_ts=45.0,
        actual_spawn_ts=45.2,
        drift_s=0.2,
        state_enter=None,
        state_exit=None,
        reason="cycle_finalized",
        reaction_ms=30.0,
        verification_ms=75.0,
        result="success",
        final_state=BotState.COMBAT,
        metadata={"version": 2},
    )

    storage.record_cycle(first_record)
    storage.record_cycle(second_record)

    cycles = storage.fetch_cycles()

    assert len(cycles) == 1
    assert cycles[0]["cycle_id"] == 3
    assert cycles[0]["event_ts"] == 46.0
    assert cycles[0]["actual_spawn_ts"] == 45.2
    assert cycles[0]["result"] == "success"
    assert cycles[0]["final_state"] == "COMBAT"
    assert cycles[0]["metadata"]["version"] == 2


def test_record_cycle_requires_cycle_id(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry" / "botlab.sqlite3"
    storage = SQLiteTelemetryStorage(db_path)
    storage.initialize()

    record = TelemetryRecord(
        cycle_id=None,
        event_ts=10.0,
        state=BotState.IDLE,
        expected_spawn_ts=None,
        actual_spawn_ts=None,
        drift_s=None,
        state_enter=None,
        state_exit=None,
        reason="invalid-cycle-summary",
        reaction_ms=None,
        verification_ms=None,
        result=None,
        final_state=BotState.IDLE,
        metadata={},
    )

    with pytest.raises(ValueError, match="cycle_id"):
        storage.record_cycle(record)


def test_count_rows_returns_current_row_count(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry" / "botlab.sqlite3"
    storage = SQLiteTelemetryStorage(db_path)
    storage.initialize()

    transition_record = TelemetryRecord(
        cycle_id=1,
        event_ts=11.0,
        state=BotState.PREPARE_WINDOW,
        expected_spawn_ts=15.0,
        actual_spawn_ts=None,
        drift_s=None,
        state_enter=BotState.IDLE,
        state_exit=BotState.PREPARE_WINDOW,
        reason="prepare_window_opened",
        reaction_ms=None,
        verification_ms=None,
        result=None,
        final_state=BotState.PREPARE_WINDOW,
        metadata={},
    )

    attempt_record = TelemetryRecord(
        cycle_id=1,
        event_ts=15.1,
        state=BotState.ATTEMPT,
        expected_spawn_ts=15.0,
        actual_spawn_ts=15.05,
        drift_s=0.05,
        state_enter=BotState.READY_WINDOW,
        state_exit=BotState.ATTEMPT,
        reason="attempt_started",
        reaction_ms=25.0,
        verification_ms=None,
        result="sent",
        final_state=BotState.VERIFY,
        metadata={},
    )

    storage.record_state_transition(transition_record)
    storage.record_attempt(attempt_record)

    assert storage.count_rows("cycles") == 0
    assert storage.count_rows("state_transitions") == 1
    assert storage.count_rows("attempts") == 1
