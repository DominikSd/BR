from __future__ import annotations

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
