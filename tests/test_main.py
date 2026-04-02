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
    assert "demo_farming_session:" in captured.out
    assert "demo_observation_miss:" in captured.out
    assert "demo_observation_reposition:" in captured.out
    assert "demo_farming_showcase:" in captured.out
    assert "retarget_path:" in captured.out


def test_main_runs_live_dry_run_from_config(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "live.yaml"
    sqlite_path = tmp_path / "data" / "telemetry" / "live.sqlite3"
    log_path = tmp_path / "logs" / "live.log"
    debug_directory = tmp_path / "debug"

    config_path.write_text(
        "\n".join(
            [
                "app:",
                '  name: "botlab-live-test"',
                '  mode: "live"',
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
                '  default_profile_name: "basic_farmer"',
                "telemetry:",
                f'  sqlite_path: "{sqlite_path.as_posix()}"',
                f'  log_path: "{log_path.as_posix()}"',
                '  log_level: "INFO"',
                "vision:",
                "  enabled: true",
                "live:",
                "  dry_run: true",
                "  foreground_only: true",
                '  window_title: "Game Window"',
                "  capture_region: [0, 0, 1280, 720]",
                "  spawn_roi: [320, 140, 640, 320]",
                "  hp_bar_roi: [40, 40, 220, 18]",
                "  condition_bar_roi: [40, 68, 220, 18]",
                "  combat_indicator_roi: [560, 620, 160, 60]",
                "  reward_roi: [500, 120, 260, 120]",
                f'  debug_directory: "{debug_directory.as_posix()}"',
                "  save_frames: true",
                "  save_overlays: true",
                "  stall_timeout_s: 1.0",
                '  dry_run_profile: "single_spot_mvp"',
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--cycles",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "app.mode=live" in captured.out
    assert "total_cycles=2" in captured.out
    assert "approach_failed=" in captured.out
    assert sqlite_path.exists() is True
    assert log_path.exists() is True
    assert (debug_directory / "cycle_001" / "observation_frame.json").exists() is True
    assert (debug_directory / "cycle_001" / "observation_perception.json").exists() is True
    assert (debug_directory / "perception_session_summary.json").exists() is True


def test_main_can_run_perception_analysis_for_single_frame(tmp_path: Path, capsys) -> None:
    frame_path = tmp_path / "fixture.json"
    output_dir = tmp_path / "perception-output"
    frame_path.write_text(
        json.dumps(
            {
                "width": 1280,
                "height": 720,
                "captured_at_ts": 145.0,
                "source": "fixture",
                "metadata": {
                    "reference_point_xy": [640, 360],
                    "perception_profile": {
                        "detection_duration_s": 0.010,
                        "selection_duration_s": 0.005,
                        "action_ready_duration_s": 0.001,
                    },
                    "template_hits": [
                        {
                            "label": "mob_a",
                            "x": 600,
                            "y": 280,
                            "width": 40,
                            "height": 52,
                            "confidence": 0.94,
                            "rotation_deg": 0,
                            "target_id": "free-near",
                        },
                        {
                            "label": "mob_b",
                            "x": 740,
                            "y": 300,
                            "width": 42,
                            "height": 54,
                            "confidence": 0.90,
                            "rotation_deg": 90,
                            "target_id": "free-far",
                        },
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--analyze-frame",
            str(frame_path),
            "--perception-output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "perception_mode=analysis" in captured.out
    assert "frame_count=1" in captured.out
    assert "selected_target=free-near" in captured.out
    assert "perception_latency_summary=total_reaction_latency_ms" in captured.out
    assert (output_dir / "fixture_perception.json").exists() is True
    assert (output_dir / "fixture_perception_overlay.svg").exists() is True
    assert (output_dir / "perception_session_summary.json").exists() is True


def test_main_can_run_perception_batch_analysis(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "perception-batch-output"

    exit_code = main(
        [
            "--config",
            "config/live_dry_run.yaml",
            "--analyze-batch-dir",
            "tests/fixtures/live/perception",
            "--perception-output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "perception_mode=analysis" in captured.out
    assert "frame_count=2" in captured.out
    assert "perception_latency_summary=detection_latency_ms" in captured.out
    assert "perception_latency_summary=candidate_hits" in captured.out
    assert (output_dir / "batch_frame_a_perception.json").exists() is True
    assert (output_dir / "batch_frame_b_perception.json").exists() is True
    assert (output_dir / "perception_session_summary.json").exists() is True
    assert "real_scene_regression=" not in captured.out


def test_main_can_run_perception_benchmark_split(monkeypatch, capsys) -> None:
    from botlab.adapters.live.perception import (
        AccuracySummary,
        BenchmarkSummary,
        BenchmarkFrameReport,
        LatencyAggregate,
        NumericAggregate,
        PerceptionSessionSummary,
    )

    def aggregate(name: str, value: float) -> LatencyAggregate:
        return LatencyAggregate.from_values(name, (value,))

    def numeric(name: str, value: float) -> NumericAggregate:
        return NumericAggregate.from_values(name, (value,))

    def stub_run_perception_analysis(
        *,
        settings,
        analyze_frame,
        analyze_batch_dir,
        benchmark_split,
        output_directory,
        strict_pixel_only,
    ):
        assert settings.app.mode == "live"
        assert benchmark_split == "regression"
        assert strict_pixel_only is True
        return (
            PerceptionSessionSummary(
                frame_results=(),
                candidate_hits=aggregate("candidate_hits", 4.0),
                merged_hits=aggregate("merged_hits", 2.0),
                free_targets=aggregate("free_targets", 1.0),
                out_of_zone_rejections=aggregate("out_of_zone_rejections", 0.0),
                detection_latency=aggregate("detection_latency_ms", 11.0),
                selection_latency=aggregate("selection_latency_ms", 3.0),
                total_reaction_latency=aggregate("total_reaction_latency_ms", 16.0),
                strict_pixel_only=True,
                accuracy_summary=AccuracySummary(
                    evaluated_frame_count=1,
                    behavior_match_count=1,
                    selected_target_match_count=1,
                    occupied_contract_match_count=1,
                ),
                benchmark_summary=BenchmarkSummary(
                    evaluated_frame_count=1,
                    strict_pixel_only=True,
                    pixel_frame_count=1,
                    fallback_frame_count=0,
                    target_true_positive_count=1,
                    target_false_positive_count=0,
                    target_false_negative_count=0,
                    target_recall=1.0,
                    target_precision=1.0,
                    occupied_classification_accuracy=1.0,
                    selected_target_accuracy=1.0,
                    selected_target_in_zone_accuracy=1.0,
                    out_of_zone_rejection_count=numeric("out_of_zone_rejection_count", 0.0),
                    false_positive_reduction_after_zone_filtering=numeric(
                        "false_positive_reduction_after_zone_filtering",
                        0.0,
                    ),
                    candidate_count=numeric("candidate_count", 4.0),
                    merged_count=numeric("merged_count", 2.0),
                    false_positive_count=numeric("false_positive_count", 0.0),
                    false_negative_count=numeric("false_negative_count", 0.0),
                    frame_reports=(
                        BenchmarkFrameReport(
                            frame_source="regression__frame",
                            pipeline_mode="pixel",
                            ground_truth_target_count=1,
                            predicted_target_count=1,
                            in_zone_target_count=1,
                            out_of_zone_target_count=0,
                            true_positive_count=1,
                            false_positive_count=0,
                            false_negative_count=0,
                            target_recall=1.0,
                            target_precision=1.0,
                            occupied_classification_accuracy=1.0,
                            selected_target_correct=True,
                            selected_target_in_zone=True,
                            expected_selected_candidate_id="target-a",
                            matched_ground_truth_count=1,
                            occupied_correct_count=1,
                        ),
                    ),
                ),
            ),
            Path("data/perception_benchmark"),
        )

    monkeypatch.setattr("botlab.main._run_perception_analysis", stub_run_perception_analysis)

    exit_code = main(
        [
            "--config",
            "config/live_dry_run.yaml",
            "--benchmark-split",
            "regression",
            "--strict-pixel-benchmark",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "perception_mode=analysis" in captured.out
    assert "perception_benchmark_summary=evaluated_frames=1 strict_pixel_only=True" in captured.out


def test_main_can_route_to_live_preview(monkeypatch, capsys) -> None:
    calls = {"count": 0}

    class PreviewStub:
        def __init__(self, *, settings, enable_console):
            self.settings = settings
            self.enable_console = enable_console

        def run(self):
            calls["count"] += 1
            print(f"preview_mode=live source_mode={self.settings.app.mode}")
            return 0

    monkeypatch.setattr("botlab.main.LiveVisionPreview", PreviewStub)

    exit_code = main(
        [
            "--config",
            "config/live_dry_run.yaml",
            "--live-preview",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls["count"] == 1
    assert "preview_mode=live" in captured.out


def test_main_can_route_to_live_engage_observe(monkeypatch, capsys) -> None:
    calls = {"count": 0}

    class ObserveStub:
        def __init__(self, *, settings, enable_console):
            self.settings = settings
            self.enable_console = enable_console

        def run(self):
            calls["count"] += 1
            print(f"engage_observe_mode=live source_mode={self.settings.app.mode}")
            return 0

    monkeypatch.setattr("botlab.main.LiveEngageObserve", ObserveStub)

    exit_code = main(
        [
            "--config",
            "config/live_dry_run.yaml",
            "--live-engage-observe",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls["count"] == 1
    assert "engage_observe_mode=live" in captured.out


def test_main_can_run_live_engage_mvp(monkeypatch, capsys) -> None:
    from botlab.adapters.live.models import LiveEngageOutcome, LiveEngageResult
    from botlab.adapters.live.engage import LiveEngageRunReport, LiveEngageSessionSummary
    from botlab.adapters.live.perception import LatencyAggregate

    def build_aggregate(name: str, value: float) -> LatencyAggregate:
        return LatencyAggregate.from_values(name, (value,))

    def stub_run_live_engage_mvp(*, settings, attempts, anchor_spawn_ts, anchor_cycle_id, enable_console):
        assert settings.app.mode == "live"
        assert attempts == 2
        return LiveEngageRunReport(
            results=(
                LiveEngageResult(
                    cycle_id=1,
                    outcome=LiveEngageOutcome.ENGAGED,
                    reason="entered_combat_detected",
                    selected_target_id="front-free",
                    final_target_id="front-free",
                    click_screen_xy=(620, 300),
                    started_at_ts=100.0,
                    completed_at_ts=100.2,
                    detection_latency_ms=10.0,
                    selection_latency_ms=3.0,
                    total_reaction_latency_ms=18.0,
                    verification_latency_ms=120.0,
                ),
            ),
            summary=LiveEngageSessionSummary(
                total_attempts=1,
                engaged_count=1,
                target_stolen_count=0,
                misclick_count=0,
                approach_stalled_count=0,
                approach_timeout_count=0,
                no_target_available_count=0,
                occupied_rejection_count=0,
                out_of_zone_rejection_count=0,
                detection_latency=build_aggregate("engage_detection_latency_ms", 10.0),
                selection_latency=build_aggregate("engage_selection_latency_ms", 3.0),
                total_reaction_latency=build_aggregate("engage_total_reaction_latency_ms", 18.0),
                verification_latency=build_aggregate("engage_verification_latency_ms", 120.0),
            ),
            log_path=Path("logs/live.log"),
            sqlite_path=Path("data/live.sqlite3"),
        )

    monkeypatch.setattr("botlab.main._run_live_engage_mvp", stub_run_live_engage_mvp)

    exit_code = main(
        [
            "--config",
            "config/live_dry_run.yaml",
            "--live-engage-mvp",
            "--cycles",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "engage_mode=mvp" in captured.out
    assert "engaged=1" in captured.out
    assert "engage_attempt=1 outcome=engaged" in captured.out


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


def test_main_prints_full_farming_session_trace(
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
            "demo_farming_session",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "scenario_replay=demo_farming_session" in captured.out
    assert "cycle_trace=1 phase=targeting ts=145.000 selected=front-free" in captured.out
    assert "cycle_trace=1 phase=targeting rejected=occupied-near reason=engaged_by_other" in captured.out
    assert "cycle_trace=1 phase=approach_revalidate target=front-free" in captured.out
    assert "cycle_trace=1 phase=approach ts=145." in captured.out
    assert "retarget_from=front-free retarget_to=fallback-safe" in captured.out
    assert "cycle_trace=1 phase=reward reward_started_ts=" in captured.out
    assert "cycle_trace=1 phase=rest result=completed" in captured.out
    assert "final_condition_ratio=" in captured.out
    assert "cycle_trace=2 phase=targeting ts=190.000 selected=" in captured.out
    assert "cycle_trace=2 phase=cycle predicted_spawn_ts=190.000 actual_spawn_ts=190.000 result=success" in captured.out
