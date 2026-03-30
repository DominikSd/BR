from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from botlab.config import TelemetryConfig
from botlab.telemetry.schema import initialize_schema
from botlab.types import BotState, TelemetryRecord


class SQLiteTelemetryStorage:
    """
    Minimalna warstwa zapisu telemetry do SQLite.

    Odpowiada za:
    - utworzenie katalogu bazy,
    - inicjalizację schematu,
    - zapis rekordów do tabel:
        * cycles
        * state_transitions
        * attempts
    - odczyt rekordów do testów i prostych inspekcji.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path).expanduser().resolve()

    @classmethod
    def from_config(cls, telemetry_config: TelemetryConfig) -> "SQLiteTelemetryStorage":
        return cls(telemetry_config.sqlite_path)

    def initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            initialize_schema(connection)

    def record_cycle(self, record: TelemetryRecord) -> None:
        """
        Zapisuje lub aktualizuje rekord podsumowania cyklu.

        W tej tabeli przechowujemy jeden rekord na cycle_id.
        To ułatwia późniejsze finalizowanie cyklu po przejściu przez FSM.
        """
        self._require_cycle_id(record)

        sql = """
        INSERT INTO cycles (
            cycle_id,
            event_ts,
            expected_spawn_ts,
            actual_spawn_ts,
            drift_s,
            reason,
            reaction_ms,
            verification_ms,
            result,
            final_state,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cycle_id) DO UPDATE SET
            event_ts = excluded.event_ts,
            expected_spawn_ts = excluded.expected_spawn_ts,
            actual_spawn_ts = excluded.actual_spawn_ts,
            drift_s = excluded.drift_s,
            reason = excluded.reason,
            reaction_ms = excluded.reaction_ms,
            verification_ms = excluded.verification_ms,
            result = excluded.result,
            final_state = excluded.final_state,
            metadata_json = excluded.metadata_json
        """

        params = (
            record.cycle_id,
            record.event_ts,
            record.expected_spawn_ts,
            record.actual_spawn_ts,
            record.drift_s,
            record.reason,
            record.reaction_ms,
            record.verification_ms,
            record.result,
            self._state_to_str(record.final_state),
            self._dump_metadata(record.metadata),
        )

        with self._connect() as connection:
            connection.execute(sql, params)

    def record_state_transition(self, record: TelemetryRecord) -> None:
        """
        Zapisuje pojedyncze przejście stanu.
        """
        sql = """
        INSERT INTO state_transitions (
            cycle_id,
            event_ts,
            state_enter,
            state_exit,
            reason,
            final_state,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            record.cycle_id,
            record.event_ts,
            self._state_to_str(record.state_enter),
            self._state_to_str(record.state_exit),
            record.reason,
            self._state_to_str(record.final_state),
            self._dump_metadata(record.metadata),
        )

        with self._connect() as connection:
            connection.execute(sql, params)

    def record_attempt(self, record: TelemetryRecord) -> None:
        """
        Zapisuje próbę reakcji / weryfikacji.
        """
        sql = """
        INSERT INTO attempts (
            cycle_id,
            event_ts,
            state,
            expected_spawn_ts,
            actual_spawn_ts,
            reaction_ms,
            verification_ms,
            result,
            reason,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            record.cycle_id,
            record.event_ts,
            self._state_to_str(record.state),
            record.expected_spawn_ts,
            record.actual_spawn_ts,
            record.reaction_ms,
            record.verification_ms,
            record.result,
            record.reason,
            self._dump_metadata(record.metadata),
        )

        with self._connect() as connection:
            connection.execute(sql, params)

    def fetch_cycles(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
            id,
            cycle_id,
            event_ts,
            expected_spawn_ts,
            actual_spawn_ts,
            drift_s,
            reason,
            reaction_ms,
            verification_ms,
            result,
            final_state,
            metadata_json
        FROM cycles
        ORDER BY cycle_id ASC
        """
        return self._fetch_all(sql)

    def fetch_state_transitions(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
            id,
            cycle_id,
            event_ts,
            state_enter,
            state_exit,
            reason,
            final_state,
            metadata_json
        FROM state_transitions
        ORDER BY id ASC
        """
        return self._fetch_all(sql)

    def fetch_attempts(self) -> list[dict[str, Any]]:
        sql = """
        SELECT
            id,
            cycle_id,
            event_ts,
            state,
            expected_spawn_ts,
            actual_spawn_ts,
            reaction_ms,
            verification_ms,
            result,
            reason,
            metadata_json
        FROM attempts
        ORDER BY id ASC
        """
        return self._fetch_all(sql)

    def count_rows(self, table_name: str) -> int:
        allowed_tables = {"cycles", "state_transitions", "attempts"}
        if table_name not in allowed_tables:
            raise ValueError(f"Nieobsługiwana tabela: {table_name}")

        sql = f"SELECT COUNT(*) AS row_count FROM {table_name}"

        with self._connect() as connection:
            row = connection.execute(sql).fetchone()

        if row is None:
            return 0

        return int(row["row_count"])

    def _fetch_all(self, sql: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(sql).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(self._row_to_dict(row))
        return result

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        raw = dict(row)

        if "metadata_json" in raw:
            raw["metadata"] = json.loads(raw.pop("metadata_json"))

        return raw

    def _dump_metadata(self, metadata: dict[str, Any]) -> str:
        return json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    def _state_to_str(self, state: BotState | None) -> str | None:
        if state is None:
            return None
        return state.value

    def _require_cycle_id(self, record: TelemetryRecord) -> None:
        if record.cycle_id is None:
            raise ValueError("record.cycle_id nie może być None dla tabeli cycles.")
