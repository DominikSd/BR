from __future__ import annotations

import json
from pathlib import Path

import pytest

from botlab.adapters.live import (
    LiveRunner,
    PerceptionAnalysisRunner,
    StallDetector,
    filter_occupied_targets,
    merge_template_hits,
    select_nearest_target,
    should_start_rest,
)
from botlab.adapters.live.models import LiveTargetDetection
from botlab.adapters.live.perception import PerceptionFrameLoader, TemplateHit, TemplatePackLoader
from botlab.adapters.live.vision import extract_named_roi
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


def test_merge_template_hits_deduplicates_multiple_hits_for_same_object() -> None:
    hits = (
        TemplateHit("mob_a", 100, 100, 40, 50, 0.95, rotation_deg=0, target_id="same"),
        TemplateHit("mob_a", 104, 103, 40, 50, 0.88, rotation_deg=90, target_id="same"),
        TemplateHit("mob_b", 260, 240, 42, 52, 0.91, rotation_deg=180, target_id="other"),
    )

    merged = merge_template_hits(hits, merge_distance_px=28)

    assert len(merged) == 2
    assert merged[0].target_id == "same"
    assert merged[0].metadata["raw_hit_count"] == 2
    assert merged[1].target_id == "other"


def test_perception_analysis_computes_reaction_latency_metrics(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 1280,
                "height": 720,
                "captured_at_ts": 145.0,
                "source": "fixture-frame",
                "metadata": {
                    "reference_point_xy": [640, 360],
                    "perception_profile": {
                        "detection_duration_s": 0.012,
                        "selection_duration_s": 0.004,
                        "action_ready_duration_s": 0.002,
                    },
                    "template_hits": [
                        {
                            "label": "mob_b",
                            "x": 600,
                            "y": 280,
                            "width": 42,
                            "height": 54,
                            "confidence": 0.93,
                            "rotation_deg": 90,
                            "target_id": "front-free",
                        },
                        {
                            "label": "occupied_swords",
                            "x": 500,
                            "y": 220,
                            "width": 30,
                            "height": 20,
                            "confidence": 0.96,
                            "target_id": "occupied-near",
                        },
                        {
                            "label": "mob_a",
                            "x": 496,
                            "y": 245,
                            "width": 40,
                            "height": 52,
                            "confidence": 0.95,
                            "rotation_deg": 0,
                            "target_id": "occupied-near",
                        },
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "perception",
    )

    summary = runner.analyze_frame_path(frame_path)

    assert len(summary.frame_results) == 1
    result = summary.frame_results[0]
    assert result.selected_target_id == "front-free"
    assert len(result.occupied_detections) == 1
    assert result.timings.detection_latency_ms == pytest.approx(12.0)
    assert result.timings.selection_latency_ms == pytest.approx(4.0)
    assert result.timings.total_reaction_latency_ms == pytest.approx(18.0)
    assert summary.total_reaction_latency.p95_ms == pytest.approx(18.0)
    assert (tmp_path / "perception" / "frame_perception.json").exists() is True
    assert (tmp_path / "perception" / "frame_perception_overlay.svg").exists() is True
    assert (tmp_path / "perception" / "perception_results.jsonl").exists() is True


def test_perception_batch_analysis_aggregates_session_metrics_from_fixtures(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    fixture_directory = Path("tests/fixtures/live/perception").resolve()
    output_directory = tmp_path / "perception-batch"
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )

    summary = runner.analyze_directory(fixture_directory)

    assert len(summary.frame_results) == 2
    assert summary.frame_results[0].selected_target_id == "free-near-a"
    assert summary.frame_results[1].selected_target_id == "free-mid-b"
    assert summary.detection_latency.count == 2
    assert summary.detection_latency.min_ms == pytest.approx(8.0)
    assert summary.detection_latency.max_ms == pytest.approx(16.0)
    assert summary.detection_latency.avg_ms == pytest.approx(12.0)
    assert summary.detection_latency.p50_ms == pytest.approx(8.0)
    assert summary.detection_latency.p95_ms == pytest.approx(16.0)
    assert summary.total_reaction_latency.max_ms == pytest.approx(23.0)
    assert (output_directory / "batch_frame_a_perception.json").exists() is True
    assert (output_directory / "batch_frame_b_perception.json").exists() is True
    assert (output_directory / "perception_session_summary.json").exists() is True


def test_template_pack_loader_reads_real_template_directories(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    loader = TemplatePackLoader(settings.live)

    template_pack = loader.load()

    assert len(template_pack.mob_variants) >= 4
    assert len(template_pack.occupied_variants) >= 2
    assert {variant.label for variant in template_pack.mob_variants} >= {"mob_a", "mob_b"}
    assert {variant.rotation_deg for variant in template_pack.mob_variants} >= {0, 90, 180, 270}


def test_frame_loader_and_roi_extraction_use_sidecar_override(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    image_path = tmp_path / "frame.png"
    sidecar_path = tmp_path / "frame.json"
    from PIL import Image

    Image.new("RGB", (640, 360), color=(255, 255, 255)).save(image_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 123.0,
                "source": "roi-override-test",
                "metadata": {
                    "spawn_roi": [120, 80, 200, 120],
                    "reference_point_xy": [222, 111]
                }
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8",
    )

    loader = PerceptionFrameLoader()
    frame = loader.load_frame(image_path)
    roi = extract_named_roi(frame, roi_name="spawn_roi", live_config=settings.live)

    assert frame.source == "roi-override-test"
    assert frame.metadata["spawn_roi"] == [120, 80, 200, 120]
    assert roi["x"] == 120
    assert roi["y"] == 80
    assert roi["width"] == 200
    assert roi["height"] == 120


def test_perception_pipeline_can_analyze_real_screenshot_assets(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    output_directory = tmp_path / "perception-real"
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )

    free_summary = runner.analyze_frame_path(settings.live.sample_frames_directory / "Screenshot_3.png")
    occupied_summary = runner.analyze_frame_path(settings.live.sample_frames_directory / "Zajete1.png")

    free_result = free_summary.frame_results[0]
    occupied_result = occupied_summary.frame_results[0]

    assert len(free_result.detections) >= 1
    assert free_result.selected_target_id is not None
    assert len(free_result.free_detections) >= 1
    assert len(occupied_result.occupied_detections) >= 1
    assert occupied_result.selected_target_id is None
    assert (output_directory / "Screenshot_3_perception.json").exists() is True
    assert (output_directory / "Zajete1_perception.json").exists() is True


def test_live_spot_scene_sidecar_is_loaded_for_real_sample_frames() -> None:
    settings = load_config("config/live_dry_run.yaml")
    loader = PerceptionFrameLoader()

    frame = loader.load_frame(settings.live.sample_frames_directory / "live_spot_scene_1.png")
    roi = extract_named_roi(frame, roi_name="spawn_roi", live_config=settings.live)

    assert frame.source == "live_spot_scene_1"
    assert frame.metadata["reference_point_xy"] == [1380, 700]
    assert frame.metadata["template_match_stride_px"] == 12
    assert roi["x"] == 720
    assert roi["y"] == 220
    assert roi["width"] == 1280
    assert roi["height"] == 720


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
