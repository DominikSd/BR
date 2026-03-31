from __future__ import annotations

from pathlib import Path

from botlab.adapters.live import (
    LiveRunner,
    StallDetector,
    filter_occupied_targets,
    select_nearest_target,
    should_start_rest,
)
from botlab.adapters.live.models import LiveTargetDetection
from botlab.config import Settings, TelemetryConfig, load_config, load_default_config


def _build_live_settings(tmp_path: Path) -> Settings:
    base = load_default_config()
    telemetry = TelemetryConfig(
        sqlite_path=(tmp_path / "data" / "telemetry" / "live.sqlite3").resolve(),
        log_path=(tmp_path / "logs" / "live.log").resolve(),
        log_level="INFO",
    )
    return Settings(
        app=base.app,
        cycle=base.cycle,
        combat=base.combat,
        telemetry=telemetry,
        vision=base.vision,
        source_path=base.source_path,
        live=load_config("config/live_dry_run.yaml").live,
    )


def test_filter_occupied_targets_excludes_busy_groups() -> None:
    targets = (
        LiveTargetDetection("busy", 100, 100, 1.0, occupied=True),
        LiveTargetDetection("free", 120, 120, 2.0, occupied=False),
    )

    free_targets = filter_occupied_targets(targets)

    assert [target.target_id for target in free_targets] == ["free"]


def test_select_nearest_target_prefers_smallest_distance() -> None:
    targets = (
        LiveTargetDetection("far", 100, 100, 4.0, occupied=False),
        LiveTargetDetection("near", 100, 100, 1.5, occupied=False),
        LiveTargetDetection("mid", 100, 100, 2.0, occupied=False),
    )

    selected = select_nearest_target(targets)

    assert selected is not None
    assert selected.target_id == "near"


def test_should_start_rest_when_hp_or_condition_is_below_threshold() -> None:
    combat_config = load_default_config().combat

    assert should_start_rest(hp_ratio=0.49, condition_ratio=0.95, combat_config=combat_config) is True
    assert should_start_rest(hp_ratio=0.95, condition_ratio=0.49, combat_config=combat_config) is True
    assert should_start_rest(hp_ratio=0.95, condition_ratio=0.95, combat_config=combat_config) is False


def test_stall_detector_marks_stall_after_one_second_without_progress() -> None:
    detector = StallDetector(timeout_s=1.0)

    assert detector.is_stalled(last_progress_ts=10.0, now_ts=11.0, entered_combat=False) is True
    assert detector.is_stalled(last_progress_ts=10.0, now_ts=10.8, entered_combat=False) is False
    assert detector.is_stalled(last_progress_ts=10.0, now_ts=12.0, entered_combat=True) is False


def test_live_runner_dry_run_executes_minimal_vertical_slice(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    runner = LiveRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=100.0,
        initial_anchor_cycle_id=0,
        enable_console=False,
    )

    report = runner.run_cycles(2)

    assert report.total_cycles == 2
    assert report.cycle_results[0].result == "success"
    assert report.cycle_results[1].result == "success"
    assert report.target_resolutions[0].selected_target_id == "front-free"
    assert report.approach_results[0].target_id == "fallback-safe"
    assert report.approach_results[0].retargeted is True
    assert report.cycle_results[1].final_state.value == "WAIT_NEXT_CYCLE"

    debug_root = settings.live.debug_directory
    assert (debug_root / "cycle_001" / "observation_frame.json").exists() is True
    assert (debug_root / "cycle_001" / "observation_overlay.svg").exists() is True
