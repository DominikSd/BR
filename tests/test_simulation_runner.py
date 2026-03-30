from __future__ import annotations

from pathlib import Path

from botlab.config import Settings, TelemetryConfig, load_default_config
from botlab.simulation.runner import SimulationRunner
from botlab.simulation.spawner import CycleScenario, SimulatedSpawner
from botlab.types import BotState


def _build_settings(tmp_path: Path) -> Settings:
    base = load_default_config()

    telemetry = TelemetryConfig(
        sqlite_path=(tmp_path / "data" / "telemetry" / "botlab.sqlite3").resolve(),
        log_path=(tmp_path / "logs" / "botlab.log").resolve(),
        log_level="INFO",
    )

    return Settings(
        app=base.app,
        cycle=base.cycle,
        combat=base.combat,
        telemetry=telemetry,
        vision=base.vision,
        source_path=base.source_path,
    )


def test_runner_can_simulate_ten_cycles_and_persist_telemetry(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_final_hp_ratio=0.40,
                note="c1-rest",
            ),
            2: CycleScenario(
                has_event=True,
                drift_s=0.2,
                verify_result="failure",
                note="c2-failure",
            ),
            3: CycleScenario(
                has_event=False,
                drift_s=0.0,
                verify_result="success",
                note="c3-no-event",
            ),
            4: CycleScenario(
                has_event=True,
                drift_s=1.2,
                verify_result="success",
                note="c4-late",
            ),
            5: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="timeout",
                note="c5-timeout",
            ),
            6: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_final_hp_ratio=0.85,
                note="c6-no-rest",
            ),
            7: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_final_hp_ratio=0.35,
                note="c7-rest",
            ),
            8: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="failure",
                note="c8-failure",
            ),
            9: CycleScenario(
                has_event=True,
                drift_s=-0.2,
                verify_result="success",
                combat_final_hp_ratio=0.95,
                note="c9-no-rest",
            ),
            10: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_final_hp_ratio=0.45,
                note="c10-rest",
            ),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(10)

    assert report.total_cycles == 10
    assert report.count_result("success") == 5
    assert report.count_result("failure") == 2
    assert report.count_result("no_event") == 1
    assert report.count_result("late_event_missed") == 1
    assert report.count_result("verify_timeout") == 1

    assert runner.storage.count_rows("cycles") == 10
    assert runner.storage.count_rows("attempts") == 8
    assert runner.storage.count_rows("state_transitions") >= 28

    assert report.log_path.exists() is True
    assert report.sqlite_path.exists() is True

    log_content = report.log_path.read_text(encoding="utf-8")
    assert '"result": "success"' in log_content
    assert '"result": "late_event_missed"' in log_content
    assert '"result": "verify_timeout"' in log_content

    cycles = runner.storage.fetch_cycles()
    assert len(cycles) == 10
    assert any(item["result"] == "success" for item in cycles)
    assert any(item["result"] == "failure" for item in cycles)
    assert any(item["result"] == "no_event" for item in cycles)
    assert any(item["result"] == "late_event_missed" for item in cycles)
    assert any(item["result"] == "verify_timeout" for item in cycles)

    success_cycles = [item for item in cycles if item["result"] == "success"]
    assert len(success_cycles) == 5
    assert all(item["final_state"] == "WAIT_NEXT_CYCLE" for item in success_cycles)


def test_runner_distinguishes_missing_event_from_late_event(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(has_event=False, note="missing"),
            2: CycleScenario(has_event=True, drift_s=1.5, verify_result="success", note="late"),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(2)

    assert report.cycle_results[0].result == "no_event"
    assert report.cycle_results[1].result == "late_event_missed"
    assert runner.storage.count_rows("attempts") == 0

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["result"] == "no_event"
    assert cycles[1]["result"] == "late_event_missed"


def test_runner_records_verify_timeout_and_recover_path(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(has_event=True, drift_s=0.0, verify_result="timeout", note="timeout"),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(1)

    assert report.total_cycles == 1
    assert report.cycle_results[0].result == "verify_timeout"
    assert report.cycle_results[0].observation_used is True
    assert report.cycle_results[0].final_state is BotState.WAIT_NEXT_CYCLE

    attempts = runner.storage.fetch_attempts()
    assert len(attempts) == 1
    assert attempts[0]["result"] == "verify_timeout"

    transitions = runner.storage.fetch_state_transitions()
    assert any(item["state_exit"] == "RECOVER" for item in transitions)
    assert any(
        item["state_exit"] == "WAIT_NEXT_CYCLE" and item["reason"] == "recover_timeout_elapsed"
        for item in transitions
    )

    cycles = runner.storage.fetch_cycles()
    assert len(cycles) == 1
    assert cycles[0]["result"] == "verify_timeout"
    assert cycles[0]["final_state"] == "WAIT_NEXT_CYCLE"


def test_runner_executes_combat_then_rest_then_wait_next_cycle(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_turns=3,
                combat_final_hp_ratio=0.40,
                note="combat-rest",
            ),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(1)

    assert report.total_cycles == 1
    assert report.cycle_results[0].result == "success"
    assert report.cycle_results[0].final_state is BotState.WAIT_NEXT_CYCLE

    transitions = runner.storage.fetch_state_transitions()

    assert any(
        item["state_enter"] == "VERIFY" and item["state_exit"] == "COMBAT"
        for item in transitions
    )
    assert any(
        item["state_enter"] == "COMBAT" and item["state_exit"] == "REST"
        and item["reason"] == "combat_finished_low_hp"
        for item in transitions
    )
    assert any(
        item["state_enter"] == "REST" and item["state_exit"] == "WAIT_NEXT_CYCLE"
        and item["reason"] == "rest_completed_hp_restored"
        for item in transitions
    )

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["result"] == "success"
    assert cycles[0]["final_state"] == "WAIT_NEXT_CYCLE"


def test_runner_executes_combat_without_rest_when_hp_is_high_enough(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_turns=2,
                combat_final_hp_ratio=0.95,
                note="combat-no-rest",
            ),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(1)

    assert report.total_cycles == 1
    assert report.cycle_results[0].result == "success"
    assert report.cycle_results[0].final_state is BotState.WAIT_NEXT_CYCLE

    transitions = runner.storage.fetch_state_transitions()

    assert any(
        item["state_enter"] == "VERIFY" and item["state_exit"] == "COMBAT"
        for item in transitions
    )
    assert any(
        item["state_enter"] == "COMBAT" and item["state_exit"] == "WAIT_NEXT_CYCLE"
        and item["reason"] == "combat_finished_no_rest_needed"
        for item in transitions
    )
    assert not any(item["state_exit"] == "REST" for item in transitions)


def test_runner_handles_force_battle_error(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_turns=3,
                combat_final_hp_ratio=0.40,
                force_battle_error=True,
                note="forced-battle-error",
            ),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(1)

    assert report.total_cycles == 1
    assert report.cycle_results[0].result == "execution_error"
    assert report.cycle_results[0].final_state is BotState.WAIT_NEXT_CYCLE

    cycles = runner.storage.fetch_cycles()
    assert len(cycles) == 1
    assert cycles[0]["result"] == "execution_error"
    assert cycles[0]["final_state"] == "WAIT_NEXT_CYCLE"
    assert cycles[0]["metadata"]["exception_type"] == "RuntimeError"

    transitions = runner.storage.fetch_state_transitions()
    assert any(item["state_exit"] == "RECOVER" for item in transitions)
    assert any(
        item["state_exit"] == "WAIT_NEXT_CYCLE"
        and item["reason"] == "execution_error_RuntimeError_reset_complete"
        for item in transitions
    )


def test_runner_recovers_from_rest_exception_and_resets_to_wait_next_cycle(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_turns=3,
                combat_final_hp_ratio=0.40,
                force_rest_error=True,
                note="forced-rest-error",
            ),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(1)

    assert report.total_cycles == 1
    assert report.cycle_results[0].result == "execution_error"
    assert report.cycle_results[0].final_state is BotState.WAIT_NEXT_CYCLE

    cycles = runner.storage.fetch_cycles()
    assert len(cycles) == 1
    assert cycles[0]["result"] == "execution_error"
    assert cycles[0]["final_state"] == "WAIT_NEXT_CYCLE"
    assert cycles[0]["metadata"]["exception_type"] == "RuntimeError"

    transitions = runner.storage.fetch_state_transitions()
    assert any(item["state_exit"] == "REST" for item in transitions)
    assert any(item["state_exit"] == "RECOVER" for item in transitions)
    assert any(
        item["state_exit"] == "WAIT_NEXT_CYCLE"
        and item["reason"] == "execution_error_RuntimeError_reset_complete"
        for item in transitions
    )
