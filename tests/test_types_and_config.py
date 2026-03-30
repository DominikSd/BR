from __future__ import annotations

from pathlib import Path

import pytest

from botlab.config import ConfigError, load_config, load_default_config
from botlab.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from botlab.types import (
    BotState,
    CombatSnapshot,
    CyclePrediction,
    Decision,
    Observation,
    TelemetryRecord,
)


def test_load_default_config_returns_settings() -> None:
    settings = load_default_config()

    assert settings.source_path == DEFAULT_CONFIG_PATH.resolve()
    assert settings.app.name == "botlab"
    assert settings.app.mode == "simulation"

    assert settings.cycle.interval_s == 45.0
    assert settings.cycle.prepare_before_s == 5.0
    assert settings.cycle.ready_before_s == 1.0
    assert settings.cycle.ready_after_s == 1.0
    assert settings.cycle.verify_timeout_s == 0.5
    assert settings.cycle.recover_timeout_s == 2.0

    assert settings.combat.low_hp_threshold == 0.35
    assert settings.combat.rest_start_threshold == 0.50
    assert settings.combat.rest_stop_threshold == 0.90

    assert settings.telemetry.sqlite_path == (PROJECT_ROOT / "data/telemetry/botlab.sqlite3").resolve()
    assert settings.telemetry.log_path == (PROJECT_ROOT / "logs/botlab.log").resolve()
    assert settings.telemetry.log_level == "INFO"

    assert settings.vision.enabled is False


def test_cycle_prediction_window_methods() -> None:
    prediction = CyclePrediction(
        cycle_id=1,
        predicted_spawn_ts=100.0,
        interval_s=45.0,
        prepare_window_start_ts=95.0,
        ready_window_start_ts=99.0,
        ready_window_end_ts=101.0,
        based_on_observation_count=3,
    )

    assert prediction.is_in_prepare_window(96.0) is True
    assert prediction.is_in_prepare_window(99.0) is False

    assert prediction.is_in_ready_window(99.0) is True
    assert prediction.is_in_ready_window(100.0) is True
    assert prediction.is_in_ready_window(101.0) is True
    assert prediction.is_in_ready_window(101.1) is False


def test_domain_models_can_be_instantiated() -> None:
    observation = Observation(
        cycle_id=7,
        observed_at_ts=123.456,
        signal_detected=True,
        actual_spawn_ts=123.400,
        source="simulation",
        confidence=0.98,
        metadata={"frame": 17},
    )

    decision = Decision(
        cycle_id=7,
        state=BotState.READY_WINDOW,
        next_state=BotState.ATTEMPT,
        action="attempt_interaction",
        reason="signal_detected_in_ready_window",
        decided_at_ts=123.500,
        metadata={"confidence": 0.98},
    )

    snapshot = CombatSnapshot(
        hp_ratio=0.72,
        turn_index=2,
        enemy_count=1,
        strategy="default",
        in_combat=True,
        combat_started_ts=124.0,
        combat_finished_ts=None,
        metadata={"target": "dummy"},
    )

    telemetry = TelemetryRecord(
        cycle_id=7,
        event_ts=123.600,
        state=BotState.VERIFY,
        expected_spawn_ts=123.400,
        actual_spawn_ts=123.410,
        drift_s=0.010,
        state_enter=BotState.ATTEMPT,
        state_exit=BotState.VERIFY,
        reason="attempt_completed",
        reaction_ms=45.0,
        verification_ms=120.0,
        result="pending",
        final_state=BotState.VERIFY,
        metadata={"attempt_id": 1},
    )

    assert observation.signal_detected is True
    assert decision.next_state is BotState.ATTEMPT
    assert snapshot.in_combat is True
    assert telemetry.state is BotState.VERIFY


def test_missing_required_section_raises_config_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 5.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Brak wymaganych sekcji konfiguracji"):
        load_config(invalid_config)


def test_invalid_cycle_values_raise_config_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid_cycle.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 50.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "combat:",
                "  low_hp_threshold: 0.35",
                "  rest_start_threshold: 0.50",
                "  rest_stop_threshold: 0.90",
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="prepare_before_s"):
        load_config(invalid_config)


def test_invalid_log_level_raises_config_error(tmp_path: Path) -> None:
    invalid_config = tmp_path / "invalid_log_level.yaml"
    invalid_config.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab"',
                '  mode: "simulation"',
                "cycle:",
                "  interval_s: 45.0",
                "  prepare_before_s: 5.0",
                "  ready_before_s: 1.0",
                "  ready_after_s: 1.0",
                "  verify_timeout_s: 0.5",
                "  recover_timeout_s: 2.0",
                "combat:",
                "  low_hp_threshold: 0.35",
                "  rest_start_threshold: 0.50",
                "  rest_stop_threshold: 0.90",
                "telemetry:",
                '  sqlite_path: "data/telemetry/test.sqlite3"',
                '  log_path: "logs/test.log"',
                '  log_level: "TRACE"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Nieprawidłowy poziom logowania"):
        load_config(invalid_config)
