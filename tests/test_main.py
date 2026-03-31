from __future__ import annotations

import json
from pathlib import Path

from botlab.main import main


def test_main_runs_with_temp_config_and_creates_outputs(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--cycles",
            "3",
            "--anchor-spawn-ts",
            "100.0",
            "--anchor-cycle-id",
            "0",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "app.name=botlab-test" in captured.out
    assert "total_cycles=3" in captured.out
    assert "success=" in captured.out
    assert "no_target_available=" in captured.out
    assert "sqlite_path=" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True


def test_main_lists_scenario_presets(capsys) -> None:
    exit_code = main(["--list-scenario-presets"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "baseline_mixed_cycle:" in captured.out
    assert "demo_farming_cycle:" in captured.out
    assert "demo_observation_miss:" in captured.out
    assert "demo_observation_reposition:" in captured.out
    assert "demo_farming_showcase:" in captured.out
    assert "retarget_path:" in captured.out


def test_main_lists_combat_plans(capsys) -> None:
    exit_code = main(["--list-combat-plans"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "basic_1_space" in captured.out
    assert "spam_1_space" in captured.out


def test_main_lists_combat_profiles(capsys) -> None:
    exit_code = main(["--list-combat-profiles"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "basic_farmer" in captured.out
    assert "fast_farmer" in captured.out


def test_main_runs_with_named_scenario_preset(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--scenario-preset",
            "retarget_path",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "scenario_replay=retarget_path" in captured.out
    assert "total_cycles=2" in captured.out
    assert "success=2" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True


def test_main_runs_with_named_combat_plan_in_standard_simulation(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--cycles",
            "1",
            "--combat-plan",
            "spam_1_space",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "total_cycles=1" in captured.out
    assert "success=1" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True


def test_main_uses_default_combat_profile_from_config_when_cli_override_is_missing(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                '  default_profile_name: "fast_farmer"',
                "telemetry:",
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--cycles",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "total_cycles=1" in captured.out
    assert "success=1" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True


def test_main_runs_with_combat_profile_in_standard_simulation(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--cycles",
            "1",
            "--combat-profile",
            "fast_farmer",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "total_cycles=1" in captured.out
    assert "success=1" in captured.out
    assert "combat_plan_summary=spam_1_space" in captured.out
    assert "combat_profile_summary=fast_farmer" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True


def test_main_can_export_report_json(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"
    report_json_path = tmp_path / "exports" / "report.json"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                '  default_profile_name: "fast_farmer"',
                "telemetry:",
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--cycles",
            "1",
            "--export-report-json",
            str(report_json_path),
        ]
    )

    capsys.readouterr()

    assert exit_code == 0
    assert report_json_path.exists() is True

    payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert payload["results"]["success"] == 1
    assert payload["combat_profile_summaries"][0]["key"] == "fast_farmer"
    assert payload["combat_plan_summaries"][0]["key"] == "spam_1_space"


def test_main_prints_readable_demo_cycle_trace_for_demo_farming_preset(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--scenario-preset",
            "demo_farming_cycle",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "scenario_replay=demo_farming_cycle" in captured.out
    assert "cycle_trace=1 phase=targeting ts=145.000 selected=front-free" in captured.out
    assert "cycle_trace=1 phase=targeting rejected=taken-near reason=engaged_by_other" in captured.out
    assert "cycle_trace=1 phase=targeting rejected=blocked-mid reason=unreachable" in captured.out
    assert "cycle_trace=1 phase=approach ts=145.800 retarget_from=front-free retarget_to=fallback-safe" in captured.out
    assert "cycle_trace=1 phase=combat ts=145.800 started_target=fallback-safe" in captured.out
    assert "cycle_trace=1 phase=rest result=completed" in captured.out
    assert "cycle_trace=1 phase=cycle predicted_spawn_ts=145.000 actual_spawn_ts=145.000 result=success" in captured.out
    assert "cycle_trace=2 phase=targeting ts=190.000 selected=clean-near" in captured.out
    assert "cycle_trace=2 phase=cycle predicted_spawn_ts=190.000 actual_spawn_ts=190.000 result=success" in captured.out


def test_main_prints_showcase_trace_with_timestamps_for_demo_farming_showcase(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--scenario-preset",
            "demo_farming_showcase",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "scenario_replay=demo_farming_showcase" in captured.out
    assert "cycle_trace=1 phase=staging source=scenario_override start_xy=(-6.0, 0.0) observation_xy=(0.0, 0.0) travel_s=1.500 arrived_ts=141.500" in captured.out
    assert "cycle_trace=1 phase=wait wait_for_spawn_s=3.500 observation_xy=(0.0, 0.0)" in captured.out
    assert "cycle_trace=1 phase=targeting ts=145.150 selected=front-free" in captured.out
    assert "cycle_trace=1 phase=approach_step target=front-free step=1 arrived_ts=145.850 remaining_distance=1.050" in captured.out
    assert "cycle_trace=1 phase=approach_revalidate target=front-free step=2 reason=current_target_invalid_retargeted" in captured.out
    assert "cycle_trace=1 phase=approach ts=146.112 retarget_from=front-free retarget_to=fallback-safe" in captured.out
    assert "cycle_trace=1 phase=combat ts=146.112 started_target=fallback-safe" in captured.out
    assert "cycle_trace=2 phase=staging source=scenario_override start_xy=(-3.0, 1.0) observation_xy=(0.0, 0.0) travel_s=0.791 arrived_ts=186.091" in captured.out
    assert "cycle_trace=2 phase=cycle predicted_spawn_ts=190.300 actual_spawn_ts=190.200 result=success" in captured.out


def test_main_prints_observation_miss_trace_before_targeting(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--scenario-preset",
            "demo_observation_miss",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "scenario_replay=demo_observation_miss" in captured.out
    assert "cycle_trace=1 phase=staging source=scenario_override start_xy=(-40.0, 0.0) observation_xy=(0.0, 0.0) travel_s=10.000 arrived_ts=150.000" in captured.out
    assert "cycle_trace=1 phase=staging_missed reason=arrived_after_ready_window_start arrived_ts=150.000 ready_window_start_ts=144.000" in captured.out
    assert "phase=targeting" not in captured.out
    assert "cycle_trace=1 phase=cycle predicted_spawn_ts=145.000 actual_spawn_ts=145.000 result=no_event" in captured.out


def test_main_prints_reposition_recovery_trace_for_next_cycle(
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "config.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "botlab.sqlite3"
    log_path = tmp_path / "logs" / "botlab.log"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-test"',
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
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: false",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--scenario-preset",
            "demo_observation_reposition",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "scenario_replay=demo_observation_reposition" in captured.out
    assert "cycle_trace=1 phase=staging_missed reason=arrived_after_ready_window_start" in captured.out
    assert "cycle_trace=2 phase=staging source=carryover_from_previous_missed_cycle start_xy=(0.0, 0.0) observation_xy=(0.0, 0.0) travel_s=0.000" in captured.out
    assert "cycle_trace=2 phase=targeting ts=190.000 selected=clean-near" in captured.out
    assert "cycle_trace=2 phase=cycle predicted_spawn_ts=190.000 actual_spawn_ts=190.000 result=success" in captured.out
