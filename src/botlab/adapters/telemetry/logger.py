from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from botlab.config import TelemetryConfig
from botlab.types import TelemetryRecord


def configure_telemetry_logger(
    telemetry_config: TelemetryConfig,
    logger_name: str = "botlab",
    enable_console: bool = True,
) -> logging.Logger:
    """
    Tworzy i konfiguruje logger projektu.

    Właściwości:
    - zapis do pliku logów,
    - opcjonalny zapis na stdout/stderr,
    - brak duplikacji handlerów przy wielokrotnym wywołaniu,
    - stabilne formatowanie.

    Parametry:
    - telemetry_config: konfiguracja telemetry
    - logger_name: nazwa loggera
    - enable_console: czy dodać StreamHandler

    Zwraca:
    - skonfigurowany obiekt logging.Logger
    """
    log_path = telemetry_config.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(_resolve_log_level(telemetry_config.log_level))
    logger.propagate = False

    _reset_handlers(logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(_resolve_log_level(telemetry_config.log_level))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if enable_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(_resolve_log_level(telemetry_config.log_level))
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def log_telemetry_record(
    logger: logging.Logger,
    record: TelemetryRecord,
    level: int = logging.INFO,
) -> None:
    """
    Zapisuje TelemetryRecord jako jedną linię logu w formacie JSON.

    To jest prosty punkt wspólny, z którego później będą mogły korzystać
    scheduler, FSM i symulacja.
    """
    payload = telemetry_record_to_dict(record)
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))


def telemetry_record_to_dict(record: TelemetryRecord) -> dict[str, Any]:
    """
    Konwertuje TelemetryRecord na słownik gotowy do serializacji/logowania.
    """
    return {
        "cycle_id": record.cycle_id,
        "event_ts": record.event_ts,
        "state": record.state.value,
        "expected_spawn_ts": record.expected_spawn_ts,
        "actual_spawn_ts": record.actual_spawn_ts,
        "drift_s": record.drift_s,
        "state_enter": record.state_enter.value if record.state_enter is not None else None,
        "state_exit": record.state_exit.value if record.state_exit is not None else None,
        "reason": record.reason,
        "reaction_ms": record.reaction_ms,
        "verification_ms": record.verification_ms,
        "result": record.result,
        "final_state": record.final_state.value if record.final_state is not None else None,
        "metadata": record.metadata,
    }


def _resolve_log_level(log_level: str) -> int:
    normalized = log_level.upper()

    if normalized == "DEBUG":
        return logging.DEBUG
    if normalized == "INFO":
        return logging.INFO
    if normalized == "WARNING":
        return logging.WARNING
    if normalized == "ERROR":
        return logging.ERROR
    if normalized == "CRITICAL":
        return logging.CRITICAL

    raise ValueError(f"Nieobsługiwany poziom logowania: {log_level}")


def _reset_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
