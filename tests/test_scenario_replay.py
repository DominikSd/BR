from __future__ import annotations

from pathlib import Path

from botlab.adapters.simulation.replay import (
    ScenarioReplayRunner,
    list_scenario_replay_presets,
    load_scenario_replay,
)
from botlab.config import Settings, TelemetryConfig, load_default_config


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


def test_list_scenario_replay_presets_contains_builtin_replays() -> None:
    presets = list_scenario_replay_presets()
    preset_names = {preset.name for preset in presets}

    assert "baseline_mixed_cycle" in preset_names
    assert "demo_farming_cycle" in preset_names
    assert "demo_farming_session" in preset_names
    assert "demo_observation_miss" in preset_names
    assert "demo_observation_reposition" in preset_names
    assert "demo_farming_showcase" in preset_names
    assert "retarget_path" in preset_names


def test_scenario_replay_runner_can_execute_builtin_baseline_preset(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="baseline_mixed_cycle",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 6
    assert report.count_result("success") == 1
    assert report.count_result("failure") == 1
    assert report.count_result("no_event") == 1
    assert report.count_result("late_event_missed") == 1
    assert report.count_result("verify_timeout") == 1
    assert report.count_result("no_target_available") == 1


def test_load_scenario_replay_from_yaml_and_run_it(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    replay_path = tmp_path / "scenario.yaml"
    replay_path.write_text(
        "\n".join(
            [
                'name: "custom_replay"',
                'description: "Single success path with named combat plan"',
                "total_cycles: 1",
                "initial_anchor_spawn_ts: 140.0",
                "initial_anchor_cycle_id: 3",
                "default_scenario:",
                '  combat_plan_name: "basic_1_space"',
                "overrides:",
                "  4:",
                "    has_event: true",
                "    spawn_zone_visible: true",
                "    bot_position_xy: [0.0, 0.0]",
                "    groups:",
                '      - group_id: "near"',
                "        position_xy: [1.0, 0.0]",
                '      - group_id: "busy"',
                "        position_xy: [0.5, 0.0]",
                "        engaged_by_other: true",
                "    note: custom-success",
                '    combat_plan_name: "spam_1_space"',
            ]
        ),
        encoding="utf-8",
    )

    replay = load_scenario_replay(replay_path)

    assert replay.name == "custom_replay"
    assert replay.total_cycles == 1
    assert replay.initial_anchor_spawn_ts == 140.0
    assert replay.initial_anchor_cycle_id == 3
    assert replay.default_scenario.combat_plan_name == "basic_1_space"
    assert replay.overrides[4].combat_plan_name == "spam_1_space"
    assert replay.overrides[4].groups[0].group_id == "near"

    replay_runner = ScenarioReplayRunner.from_file(
        settings,
        replay_path=replay_path,
        enable_console=False,
    )
    report = replay_runner.run()

    assert report.total_cycles == 1
    assert report.cycle_results[0].cycle_id == 4
    assert report.cycle_results[0].result == "success"
    assert report.target_resolutions[0].selected_target_id == "near"


def test_load_scenario_replay_with_round_based_combat_plan(tmp_path: Path) -> None:
    replay_path = tmp_path / "round-replay.yaml"
    replay_path.write_text(
        "\n".join(
            [
                'name: "round_replay"',
                "total_cycles: 1",
                "overrides:",
                "  1:",
                "    has_event: true",
                "    combat_plan_rounds:",
                '      - ["1", "space"]',
                '      - ["2"]',
            ]
        ),
        encoding="utf-8",
    )

    replay = load_scenario_replay(replay_path)

    assert replay.name == "round_replay"
    assert replay.overrides[1].combat_plan_rounds == (("1", "space"), ("2",))


def test_load_scenario_replay_with_combat_profile_name(tmp_path: Path) -> None:
    replay_path = tmp_path / "profile-replay.yaml"
    replay_path.write_text(
        "\n".join(
            [
                'name: "profile_replay"',
                "total_cycles: 1",
                "overrides:",
                "  1:",
                "    has_event: true",
                '    combat_profile_name: "fast_farmer"',
            ]
        ),
        encoding="utf-8",
    )

    replay = load_scenario_replay(replay_path)

    assert replay.name == "profile_replay"
    assert replay.overrides[1].combat_profile_name == "fast_farmer"


def test_scenario_replay_runner_can_execute_demo_farming_cycle_preset(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="demo_farming_cycle",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 2
    assert report.count_result("success") == 2

    assert report.target_resolutions[0].selected_target_id == "front-free"
    assert report.approach_results[0].initial_target_id == "front-free"
    assert report.approach_results[0].target_id == "fallback-safe"
    assert report.approach_results[0].retargeted is True
    assert report.interaction_results[0].target_id == "fallback-safe"
    assert report.interaction_results[0].ready is True

    assert report.target_resolutions[1].selected_target_id == "clean-near"
    assert report.approach_results[1].target_id == "clean-near"
    assert report.approach_results[1].retargeted is False
    assert report.interaction_results[1].target_id == "clean-near"
    assert report.interaction_results[1].ready is True

    cycle_records = {record["cycle_id"]: record for record in report.cycle_records}
    assert cycle_records[1]["metadata"]["combat_finished_with_rest"] is True
    assert cycle_records[2]["metadata"]["combat_finished_with_rest"] is False


def test_scenario_replay_runner_can_execute_demo_farming_showcase_preset(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="demo_farming_showcase",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 2
    assert report.count_result("success") == 2
    assert report.cycle_results[0].predicted_spawn_ts == 145.0
    assert report.cycle_results[0].actual_spawn_ts == 145.15
    assert report.observation_preparations[0].starting_position_xy == (-6.0, 0.0)
    assert report.observation_preparations[0].observation_position_xy == (0.0, 0.0)
    assert report.observation_preparations[0].travel_s == 1.5
    assert report.observation_preparations[0].wait_for_spawn_s == 3.5
    assert report.approach_results[0].retargeted is True
    assert report.cycle_results[1].predicted_spawn_ts == 190.3
    assert round(report.cycle_results[1].actual_spawn_ts or 0.0, 3) == 190.2


def test_scenario_replay_runner_can_execute_demo_observation_miss_preset(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="demo_observation_miss",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 1
    assert report.count_result("no_event") == 1
    assert report.observation_preparations[0].ready_for_observation is False
    assert report.observation_preparations[0].metadata["ready_reason"] == (
        "arrived_after_ready_window_start"
    )
    assert report.target_resolutions[0].decision.reason == "observation_not_ready"


def test_scenario_replay_runner_can_execute_demo_observation_reposition_preset(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="demo_observation_reposition",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 2
    assert report.count_result("no_event") == 1
    assert report.count_result("success") == 1
    assert report.observation_preparations[0].metadata["ready_reason"] == (
        "arrived_after_ready_window_start"
    )
    assert report.observation_preparations[1].metadata["start_position_source"] == (
        "carryover_from_previous_missed_cycle"
    )
    assert report.observation_preparations[1].travel_s == 0.0
    assert report.target_resolutions[1].selected_target_id == "clean-near"


def test_scenario_replay_runner_can_execute_demo_farming_session_preset(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="demo_farming_session",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 2
    assert report.count_result("success") == 2

    assert report.target_resolutions[0].selected_target_id == "front-free"
    assert report.approach_results[0].target_id == "fallback-safe"
    assert report.approach_results[0].retargeted is True
    assert report.interaction_results[0].target_id == "fallback-safe"
    assert report.interaction_results[0].ready is True

    cycle_records = {record["cycle_id"]: record for record in report.cycle_records}
    assert cycle_records[1]["metadata"]["reward_started_ts"] is not None
    assert cycle_records[1]["metadata"]["reward_completed_ts"] is not None
    assert cycle_records[1]["metadata"]["combat_final_condition_ratio"] == 0.38
    assert cycle_records[1]["metadata"]["combat_finished_with_rest"] is True
    assert cycle_records[1]["metadata"]["rest_final_condition_ratio"] >= 0.9

    assert report.target_resolutions[1].selected_target_id is not None
    assert report.target_resolutions[1].world_snapshot.groups
    assert all(
        group.metadata.get("mob_variant") in {"mob_a", "mob_b"}
        for group in report.target_resolutions[1].world_snapshot.groups
    )
