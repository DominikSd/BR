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
    assert "sqlite_path=" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True
