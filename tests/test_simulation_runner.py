from __future__ import annotations

from pathlib import Path

from botlab.config import Settings, TelemetryConfig, load_default_config
from botlab.adapters.simulation.runner import SimulationRunner
from botlab.adapters.simulation.spawner import CycleScenario, SimulatedGroupState, SimulatedSpawner
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
    assert cycles[0]["metadata"]["combat_completed"] is True
    assert cycles[0]["metadata"]["combat_turn_count"] == 3
    assert cycles[0]["metadata"]["combat_final_hp_ratio"] == 0.40
    assert cycles[0]["metadata"]["combat_finished_with_rest"] is True
    assert cycles[0]["metadata"]["rest_tick_count"] >= 1
    assert cycles[0]["metadata"]["combat_plan_name"] == "basic_1_space"
    assert cycles[0]["metadata"]["combat_plan_source"] == "default_catalog_plan"
    assert cycles[0]["metadata"]["combat_profile_name"] is None


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

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["metadata"]["combat_completed"] is True
    assert cycles[0]["metadata"]["combat_finished_with_rest"] is False
    assert cycles[0]["metadata"]["rest_tick_count"] == 0


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


def test_runner_reports_observation_preparation_and_nearest_free_target(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(
                        group_id="busy-near",
                        position_xy=(1.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(
                        group_id="free-far",
                        position_xy=(5.0, 0.0),
                    ),
                    SimulatedGroupState(
                        group_id="free-near",
                        position_xy=(2.0, 0.0),
                    ),
                ),
                note="pick-nearest-free",
            ),
            2: CycleScenario(
                has_event=True,
                spawn_zone_visible=False,
                bot_position_xy=(10.0, 10.0),
                groups=(
                    SimulatedGroupState(
                        group_id="hidden-group",
                        position_xy=(11.0, 10.0),
                    ),
                ),
                note="spawn-zone-hidden",
            ),
        }
    )

    runner = SimulationRunner.from_settings(
        settings,
        spawner=spawner,
        initial_anchor_spawn_ts=100.0,
        enable_console=False,
    )

    report = runner.run_cycles(2)

    assert [item.cycle_id for item in report.observation_preparations] == [1, 2]
    assert report.observation_preparations[0].ready_for_observation is True
    assert report.observation_preparations[1].ready_for_observation is False

    assert [item.cycle_id for item in report.target_resolutions] == [1, 2]
    assert report.target_resolutions[0].selected_target_id == "free-near"
    assert report.target_resolutions[0].decision.reason == "selected_initial_target"
    assert report.target_resolutions[1].selected_target_id is None
    assert report.target_resolutions[1].decision.reason == "observation_not_ready"
    assert [item.cycle_id for item in report.approach_results] == [1, 2]
    assert report.approach_results[0].target_id == "free-near"
    assert report.approach_results[0].arrived is True
    assert report.approach_results[0].travel_s == 0.375
    assert report.approach_results[1].target_id is None
    assert report.approach_results[1].reason == "observation_not_ready"
    assert [item.cycle_id for item in report.interaction_results] == [1, 2]
    assert report.interaction_results[0].target_id == "free-near"
    assert report.interaction_results[0].ready is True
    assert report.interaction_results[1].target_id is None
    assert report.interaction_results[1].ready is False
    assert report.cycle_results[0].result == "success"
    assert report.cycle_results[1].result == "no_event"

    attempts = runner.storage.fetch_attempts()
    assert len(attempts) == 1
    assert attempts[0]["metadata"]["initial_target_id"] == "free-near"
    assert attempts[0]["metadata"]["selected_target_id"] == "free-near"
    assert attempts[0]["metadata"]["target_decision_reason"] == "selected_initial_target"
    assert attempts[0]["metadata"]["approach_travel_s"] == 0.375
    assert attempts[0]["metadata"]["retarget_count"] == 0
    assert attempts[0]["metadata"]["target_loss_count"] == 0
    assert attempts[0]["metadata"]["interaction_ready"] is True
    assert attempts[0]["metadata"]["combat_plan_name"] == "basic_1_space"
    assert attempts[0]["metadata"]["combat_plan_source"] == "default_catalog_plan"
    assert attempts[0]["metadata"]["combat_profile_name"] is None
    assert attempts[0]["metadata"]["combat_strategy"] == "default"


def test_runner_records_combat_profile_and_named_plan_in_telemetry(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_turns=3,
                combat_profile_name="fast_farmer",
                note="combat-profile-telemetry",
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

    assert report.cycle_results[0].result == "success"

    attempts = runner.storage.fetch_attempts()
    assert attempts[0]["metadata"]["combat_profile_name"] == "fast_farmer"
    assert attempts[0]["metadata"]["combat_plan_name"] == "spam_1_space"
    assert attempts[0]["metadata"]["combat_plan_source"] == "combat_profile"
    assert attempts[0]["metadata"]["combat_input_sequence"] == ["1", "1", "space"]

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["metadata"]["combat_profile_name"] == "fast_farmer"
    assert cycles[0]["metadata"]["combat_plan_name"] == "spam_1_space"
    assert cycles[0]["metadata"]["combat_plan_source"] == "combat_profile"
    assert cycles[0]["metadata"]["combat_turn_count"] == 3


def test_runner_reports_no_target_available_when_visible_groups_are_not_targetable(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(
                        group_id="busy-near",
                        position_xy=(1.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(
                        group_id="blocked-far",
                        position_xy=(4.0, 0.0),
                        reachable=False,
                    ),
                ),
                note="visible-no-free-target",
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
    assert report.target_resolutions[0].selected_target_id is None
    assert report.target_resolutions[0].decision.reason == "no_target_available"
    assert report.approach_results[0].target_id is None
    assert report.approach_results[0].reason == "no_target_selected"
    assert report.interaction_results[0].target_id is None
    assert report.interaction_results[0].ready is False
    assert report.cycle_results[0].result == "no_target_available"
    assert report.cycle_results[0].observation_used is True

    attempts = runner.storage.fetch_attempts()
    assert attempts == []

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["result"] == "no_target_available"
    assert cycles[0]["metadata"]["initial_target_id"] is None
    assert cycles[0]["metadata"]["selected_target_id"] is None
    assert cycles[0]["metadata"]["target_decision_reason"] == "no_target_available"
    assert cycles[0]["metadata"]["retarget_count"] == 0
    assert cycles[0]["metadata"]["target_loss_count"] == 0
    assert cycles[0]["metadata"]["interaction_ready"] is False


def test_runner_loses_cycle_before_targeting_when_observation_position_is_reached_too_late(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                observation_start_position_xy=(-40.0, 0.0),
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(group_id="would-be-target", position_xy=(2.0, 0.0)),
                ),
                note="late-to-observation-position",
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

    assert report.observation_preparations[0].ready_for_observation is False
    assert report.observation_preparations[0].metadata["ready_reason"] == (
        "arrived_after_ready_window_start"
    )
    assert report.target_resolutions[0].decision.reason == "observation_not_ready"
    assert report.approach_results[0].reason == "observation_not_ready"
    assert report.interaction_results[0].reason == "observation_not_ready"
    assert report.cycle_results[0].result == "no_event"
    assert report.cycle_results[0].observation_used is False

    attempts = runner.storage.fetch_attempts()
    assert attempts == []

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["result"] == "no_event"
    assert cycles[0]["metadata"]["observation_ready_reason"] == "arrived_after_ready_window_start"


def test_runner_retargets_immediately_when_target_becomes_unavailable_during_approach(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(group_id="current", position_xy=(2.0, 0.0)),
                    SimulatedGroupState(group_id="replacement", position_xy=(6.0, 0.0)),
                ),
                approach_revalidation_delay_s=0.3,
                approach_bot_position_xy=(1.0, 0.0),
                approach_groups=(
                    SimulatedGroupState(
                        group_id="current",
                        position_xy=(2.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(group_id="replacement", position_xy=(4.0, 0.0)),
                ),
                note="retarget-during-approach",
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

    assert report.target_resolutions[0].selected_target_id == "current"
    assert report.approach_results[0].initial_target_id == "current"
    assert report.approach_results[0].target_id == "replacement"
    assert report.approach_results[0].retargeted is True
    assert report.approach_results[0].metadata["revalidation_reason"] == (
        "current_target_invalid_retargeted"
    )
    assert report.interaction_results[0].target_id == "replacement"
    assert report.interaction_results[0].ready is True
    assert report.cycle_results[0].result == "success"

    attempts = runner.storage.fetch_attempts()
    assert len(attempts) == 1
    assert attempts[0]["metadata"]["initial_target_id"] == "current"
    assert attempts[0]["metadata"]["selected_target_id"] == "replacement"
    assert attempts[0]["metadata"]["approach_reason"] == "target_reached_in_simulation"
    assert attempts[0]["metadata"]["retargeted_during_approach"] is True
    assert attempts[0]["metadata"]["retarget_count"] == 1
    assert attempts[0]["metadata"]["target_loss_count"] == 1


def test_runner_stops_immediately_when_target_is_lost_during_approach_without_replacement(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(SimulatedGroupState(group_id="current", position_xy=(2.0, 0.0)),),
                approach_revalidation_delay_s=0.3,
                approach_bot_position_xy=(1.0, 0.0),
                approach_groups=(
                    SimulatedGroupState(
                        group_id="current",
                        position_xy=(2.0, 0.0),
                        reachable=False,
                    ),
                ),
                note="lost-during-approach",
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

    assert report.target_resolutions[0].selected_target_id == "current"
    assert report.approach_results[0].target_id is None
    assert report.approach_results[0].initial_target_id == "current"
    assert report.approach_results[0].reason == "target_lost_during_approach_no_replacement"
    assert report.interaction_results[0].target_id is None
    assert report.interaction_results[0].ready is False
    assert report.cycle_results[0].result == "no_target_available"

    attempts = runner.storage.fetch_attempts()
    assert attempts == []

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["result"] == "no_target_available"
    assert cycles[0]["metadata"]["selected_target_id"] is None
    assert cycles[0]["metadata"]["approach_reason"] == "target_lost_during_approach_no_replacement"
    assert cycles[0]["metadata"]["initial_target_id"] == "current"
    assert cycles[0]["metadata"]["retarget_count"] == 0
    assert cycles[0]["metadata"]["target_loss_count"] == 1


def test_runner_retargets_again_if_target_is_lost_right_before_interaction(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(group_id="current", position_xy=(2.0, 0.0)),
                    SimulatedGroupState(group_id="replacement", position_xy=(6.0, 0.0)),
                ),
                interaction_revalidation_delay_s=0.6,
                interaction_bot_position_xy=(2.0, 0.0),
                interaction_groups=(
                    SimulatedGroupState(
                        group_id="current",
                        position_xy=(2.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(group_id="replacement", position_xy=(3.0, 0.0)),
                ),
                note="retarget-before-interaction",
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

    assert report.target_resolutions[0].selected_target_id == "replacement"
    assert report.approach_results[0].target_id == "replacement"
    assert report.approach_results[0].initial_target_id == "replacement"
    assert report.interaction_results[0].target_id == "replacement"
    assert report.interaction_results[0].retargeted is True
    assert report.interaction_results[0].metadata["revalidation_reason"] == (
        "current_target_still_valid"
    )
    assert report.cycle_results[0].result == "success"

    attempts = runner.storage.fetch_attempts()
    assert attempts[0]["metadata"]["initial_target_id"] == "current"
    assert attempts[0]["metadata"]["selected_target_id"] == "replacement"
    assert attempts[0]["metadata"]["interaction_reason"] == "interaction_ready"
    assert attempts[0]["metadata"]["retargeted_before_interaction"] is True
    assert attempts[0]["metadata"]["retarget_count"] == 1
    assert attempts[0]["metadata"]["target_loss_count"] == 1


def test_runner_stops_when_target_is_lost_right_before_interaction_without_replacement(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)

    spawner = SimulatedSpawner(
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(SimulatedGroupState(group_id="current", position_xy=(2.0, 0.0)),),
                interaction_revalidation_delay_s=0.6,
                interaction_bot_position_xy=(2.0, 0.0),
                interaction_groups=(
                    SimulatedGroupState(
                        group_id="current",
                        position_xy=(2.0, 0.0),
                        reachable=False,
                    ),
                ),
                note="lost-before-interaction",
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

    assert report.interaction_results[0].target_id is None
    assert report.interaction_results[0].ready is False
    assert report.interaction_results[0].reason == "target_lost_before_interaction_no_replacement"
    assert report.cycle_results[0].result == "no_target_available"

    attempts = runner.storage.fetch_attempts()
    assert attempts == []

    cycles = runner.storage.fetch_cycles()
    assert cycles[0]["metadata"]["interaction_reason"] == "target_lost_before_interaction_no_replacement"
    assert cycles[0]["metadata"]["initial_target_id"] == "current"
    assert cycles[0]["metadata"]["retarget_count"] == 0
    assert cycles[0]["metadata"]["target_loss_count"] == 1
