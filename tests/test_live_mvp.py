from __future__ import annotations

from dataclasses import replace
import json
import logging
from pathlib import Path

import pytest

from botlab.adapters.live import (
    LiveEngageSessionSummary,
    LiveEngageObserve,
    LiveRunner,
    LiveVisionPreviewRenderer,
    PerceptionAnalysisRunner,
    SceneProfileLoader,
    StallDetector,
    build_live_preview_state,
    classify_engage_outcome,
    filter_occupied_targets,
    merge_template_hits,
    point_in_polygon,
    select_nearest_target,
    should_start_rest,
)
from botlab.adapters.live.capture import CaptureRegion, _calculate_window_relative_capture_bbox
from botlab.adapters.live.input import LiveInputDriver
from botlab.adapters.live.models import (
    LiveEngageOutcome,
    LiveFrame,
    LiveResourceSnapshot,
    LiveSessionState,
    LiveStateSnapshot,
    LiveTargetDetection,
)
from botlab.adapters.live.perception import (
    BenchmarkDatasetLoader,
    PerceptionFrameLoader,
    ReactionLatency,
    TemplateHit,
    TemplatePackLoader,
    compare_perception_summary_payloads,
)
from botlab.adapters.live.runner import LiveRestProvider
from botlab.adapters.live.vision import LiveResourceProvider, SimpleStateDetector, extract_named_roi
from botlab.application.dto import (
    TargetApproachResult,
    TargetEngagementResult,
    TargetInteractionResult,
    TargetResolution,
)
from botlab.config import Settings, TelemetryConfig, load_config, load_default_config
from botlab.domain.targeting import (
    RetargetDecision,
    TargetCandidate,
    TargetValidationResult,
    TargetValidationStatus,
)
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot
from botlab.types import Observation


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


def _build_scene_aware_live_settings(tmp_path: Path) -> Settings:
    base = load_default_config()
    telemetry = TelemetryConfig(
        sqlite_path=(tmp_path / "data" / "telemetry" / "live_scene.sqlite3").resolve(),
        log_path=(tmp_path / "logs" / "live_scene.log").resolve(),
        log_level="INFO",
    )
    return Settings(
        app=base.app,
        cycle=base.cycle,
        combat=base.combat,
        telemetry=telemetry,
        vision=base.vision,
        source_path=base.source_path,
        live=replace(
            load_config("config/live_real_mvp.yaml").live,
            marker_color_mode="red",
            marker_min_green=120,
            marker_green_blue_delta=25,
            marker_red_green_balance_delta=80,
            marker_min_blob_pixels=6,
            marker_max_blob_pixels=180,
            marker_min_width_px=3,
            marker_max_width_px=36,
            marker_min_height_px=3,
            marker_max_height_px=36,
            marker_min_fill_density=0.0,
            marker_max_fill_density=1.0,
            marker_dark_core_max_rgb=110,
            marker_min_dark_core_ratio=0.0,
            marker_confidence_threshold=0.55,
        ),
    )


def _build_live_settings_with_profile(tmp_path: Path, dry_run_profile: str) -> Settings:
    settings = _build_live_settings(tmp_path)
    return replace(
        settings,
        live=replace(settings.live, dry_run_profile=dry_run_profile),
    )


def _draw_bar_fill(image, roi: tuple[int, int, int, int], ratio: float, fill_rgb: tuple[int, int, int]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    x, y, width, height = roi
    draw.rectangle((x, y, x + width - 1, y + height - 1), fill=(20, 20, 20))
    fill_width = max(0, min(width, int(round(width * ratio))))
    if fill_width > 0:
        draw.rectangle((x, y, x + fill_width - 1, y + height - 1), fill=fill_rgb)


def _draw_roi_fill(image, roi: tuple[int, int, int, int], fill_rgb: tuple[int, int, int]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    x, y, width, height = roi
    draw.rectangle((x, y, x + width - 1, y + height - 1), fill=fill_rgb)


def _build_state_resource_frame(
    settings: Settings,
    *,
    hp_ratio: float,
    condition_ratio: float,
    in_combat: bool = False,
    reward_visible: bool = False,
    captured_at_ts: float = 100.0,
    source: str = "resource-frame",
) -> LiveFrame:
    from PIL import Image

    _, _, width, height = settings.live.capture_region
    image = Image.new("RGB", (width, height), color=(15, 15, 15))
    _draw_bar_fill(
        image,
        settings.live.hp_bar_roi,
        hp_ratio,
        (
            max(settings.live.hp_bar_min_red + 30, 200),
            30,
            30,
        ),
    )
    _draw_bar_fill(
        image,
        settings.live.condition_bar_roi,
        condition_ratio,
        (
            30,
            max(settings.live.condition_bar_min_green + 40, 180),
            30,
        ),
    )
    if in_combat:
        _draw_roi_fill(
            image,
            settings.live.combat_indicator_roi,
            (
                max(settings.live.combat_indicator_min_red + 30, 200),
                20,
                20,
            ),
        )
    if reward_visible:
        _draw_roi_fill(
            image,
            settings.live.reward_roi,
            (
                max(settings.live.reward_min_red + 20, 220),
                max(settings.live.reward_min_green + 20, 180),
                max(0, settings.live.reward_max_blue - 20),
            ),
        )
    return LiveFrame(
        width=width,
        height=height,
        captured_at_ts=captured_at_ts,
        source=source,
        metadata={},
        image=image,
    )


def _build_resource_frame(
    settings: Settings,
    *,
    hp_ratio: float,
    condition_ratio: float,
    in_combat: bool = False,
    reward_visible: bool = False,
    captured_at_ts: float = 100.0,
    source: str = "resource-frame",
) -> LiveFrame:
    return _build_state_resource_frame(
        settings,
        hp_ratio=hp_ratio,
        condition_ratio=condition_ratio,
        in_combat=in_combat,
        reward_visible=reward_visible,
        captured_at_ts=captured_at_ts,
        source=source,
    )


class _FakeRestRuntime:
    def __init__(self, settings: Settings, frames: list[LiveFrame]) -> None:
        self.settings = settings
        self._frames = list(frames)
        self._session_state = LiveSessionState()
        self.updated_resources: tuple[float, float] | None = None

    @property
    def session_state(self) -> LiveSessionState:
        return self._session_state

    def capture_frame(self, *, cycle_id: int, phase: str, default_ts: float) -> LiveFrame:
        assert cycle_id >= 0
        assert phase.startswith("rest_sample_")
        assert default_ts >= 0.0
        if not self._frames:
            raise AssertionError("Brak kolejnych klatek dla fake runtime.")
        return self._frames.pop(0)

    def update_resources(self, *, hp_ratio: float, condition_ratio: float) -> None:
        self.updated_resources = (hp_ratio, condition_ratio)
        self._session_state.hp_ratio = hp_ratio
        self._session_state.condition_ratio = condition_ratio


def _write_simple_marker_first_templates(root: Path) -> tuple[Path, Path]:
    from PIL import Image, ImageDraw

    mobs_root = root / "mobs"
    occupied_root = root / "occupied"
    (mobs_root / "mob_a").mkdir(parents=True, exist_ok=True)
    (mobs_root / "mob_b").mkdir(parents=True, exist_ok=True)
    occupied_root.mkdir(parents=True, exist_ok=True)

    mob_a = Image.new("RGB", (18, 24), color=(35, 35, 35))
    draw_a = ImageDraw.Draw(mob_a)
    draw_a.rectangle((5, 2, 12, 7), fill=(170, 195, 225))
    draw_a.rectangle((4, 8, 13, 22), fill=(120, 80, 40))
    mob_a.save(mobs_root / "mob_a" / "base.png")

    mob_b = Image.new("RGB", (18, 24), color=(35, 35, 35))
    draw_b = ImageDraw.Draw(mob_b)
    draw_b.rectangle((5, 2, 12, 7), fill=(110, 180, 210))
    draw_b.rectangle((4, 8, 13, 22), fill=(70, 110, 150))
    mob_b.save(mobs_root / "mob_b" / "base.png")

    swords = Image.new("RGB", (12, 12), color=(0, 0, 0))
    draw_s = ImageDraw.Draw(swords)
    draw_s.line((2, 2, 9, 9), fill=(60, 220, 60), width=2)
    draw_s.line((9, 2, 2, 9), fill=(60, 220, 60), width=2)
    swords.save(occupied_root / "crossed_swords.png")

    return mobs_root, occupied_root


def _build_marker_first_settings(tmp_path: Path) -> Settings:
    settings = _build_live_settings(tmp_path)
    templates_root = tmp_path / "templates"
    mobs_root, occupied_root = _write_simple_marker_first_templates(templates_root)
    live = replace(
        settings.live,
        spawn_roi=(0, 0, 320, 240),
        mobs_template_directory=mobs_root,
        occupied_template_directory=occupied_root,
        template_rotations_deg=(0,),
        template_match_stride_px=1,
        perception_confidence_threshold=0.80,
        confirmation_confidence_threshold=0.80,
        occupied_confidence_threshold=0.55,
        merge_distance_px=16,
        marker_min_red=160,
        marker_red_green_delta=40,
        marker_red_blue_delta=30,
        marker_min_blob_pixels=4,
        marker_max_blob_pixels=80,
        marker_min_width_px=3,
        marker_max_width_px=18,
        marker_min_height_px=3,
        marker_max_height_px=18,
        marker_confidence_threshold=0.45,
        swords_min_green=120,
        swords_green_red_delta=20,
        swords_green_blue_delta=10,
        swords_min_blob_pixels=6,
        swords_max_blob_pixels=80,
        occupied_local_roi_width_px=40,
        occupied_local_roi_height_px=32,
        occupied_local_roi_offset_y_px=-4,
        confirmation_roi_width_px=60,
        confirmation_roi_height_px=72,
        confirmation_roi_offset_y_px=0,
    )
    return replace(settings, live=live)


def _write_benchmark_split_manifest(
    split_directory: Path,
    *,
    split_name: str,
    frame_names: tuple[str, ...],
) -> None:
    split_directory.mkdir(parents=True, exist_ok=True)
    split_directory.joinpath("frames.json").write_text(
        json.dumps(
            {
                "split": split_name,
                "frames": [
                    {
                        "frame_name": frame_name,
                        "frame_path": f"../raw/{frame_name}.png",
                    }
                    for frame_name in frame_names
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_marker_first_benchmark_dataset(tmp_path: Path) -> Settings:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    dataset_root = tmp_path / "dataset"
    raw_directory = dataset_root / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    mob_a_template = Image.open(settings.live.mobs_template_directory / "mob_a" / "base.png").convert("RGB")
    mob_b_template = Image.open(settings.live.mobs_template_directory / "mob_b" / "base.png").convert("RGB")
    swords_template = Image.open(settings.live.occupied_template_directory / "crossed_swords.png").convert("RGB")

    free_frame = Image.new("RGB", (320, 240), color=(40, 40, 40))
    free_draw = ImageDraw.Draw(free_frame)
    free_frame.paste(mob_a_template, (146, 104))
    free_draw.polygon([(155, 92), (159, 86), (163, 92), (159, 98)], fill=(225, 55, 55))
    free_frame.save(raw_directory / "benchmark_free.png")
    (raw_directory / "benchmark_free.json").write_text(
        json.dumps(
            {
                "captured_at_ts": 10.0,
                "source": "benchmark_free",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                    "ground_truth": {
                        "candidates": [
                            {
                                "candidate_id": "free_primary",
                                "screen_xy": [159, 116],
                                "max_error_px": 32,
                                "occupied": False,
                                "selected": True,
                            }
                        ]
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    occupied_frame = Image.new("RGB", (320, 240), color=(40, 40, 40))
    occupied_draw = ImageDraw.Draw(occupied_frame)
    occupied_frame.paste(mob_b_template, (58, 84))
    occupied_draw.polygon([(67, 72), (71, 66), (75, 72), (71, 78)], fill=(225, 55, 55))
    occupied_frame.paste(swords_template, (64, 72))
    occupied_frame.save(raw_directory / "benchmark_occupied.png")
    (raw_directory / "benchmark_occupied.json").write_text(
        json.dumps(
            {
                "captured_at_ts": 11.0,
                "source": "benchmark_occupied",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                    "ground_truth": {
                        "candidates": [
                            {
                                "candidate_id": "busy_primary",
                                "screen_xy": [71, 96],
                                "max_error_px": 32,
                                "occupied": True,
                                "selected": False,
                            }
                        ]
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_benchmark_split_manifest(
        dataset_root / "regression",
        split_name="regression",
        frame_names=("benchmark_free", "benchmark_occupied"),
    )
    _write_benchmark_split_manifest(
        dataset_root / "tuning",
        split_name="tuning",
        frame_names=("benchmark_free",),
    )
    _write_benchmark_split_manifest(
        dataset_root / "holdout",
        split_name="holdout",
        frame_names=("benchmark_occupied",),
    )
    _write_benchmark_split_manifest(
        dataset_root / "hard_cases",
        split_name="hard_cases",
        frame_names=("benchmark_occupied",),
    )

    return replace(
        settings,
        live=replace(
            settings.live,
            sample_frames_directory=raw_directory,
            benchmark_dataset_directory=dataset_root,
        ),
    )


def _euclidean_distance(
    *,
    left_xy: tuple[float, float],
    right_xy: tuple[float, float],
) -> float:
    return ((left_xy[0] - right_xy[0]) ** 2 + (left_xy[1] - right_xy[1]) ** 2) ** 0.5


def _find_closest_detection(
    *,
    detections: tuple[LiveTargetDetection, ...] | list[LiveTargetDetection],
    expected_xy: tuple[float, float],
) -> tuple[LiveTargetDetection | None, float]:
    closest_target: LiveTargetDetection | None = None
    closest_distance = float("inf")
    for detection in detections:
        distance = _euclidean_distance(
            left_xy=(float(detection.screen_x), float(detection.screen_y)),
            right_xy=expected_xy,
        )
        if distance < closest_distance:
            closest_distance = distance
            closest_target = detection
    return closest_target, closest_distance


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


def test_point_in_polygon_accepts_points_inside_and_on_edge() -> None:
    polygon = (
        (10.0, 10.0),
        (110.0, 10.0),
        (110.0, 110.0),
        (10.0, 110.0),
    )

    assert point_in_polygon(point_xy=(50.0, 50.0), polygon=polygon) is True
    assert point_in_polygon(point_xy=(10.0, 50.0), polygon=polygon) is True
    assert point_in_polygon(point_xy=(150.0, 50.0), polygon=polygon) is False


def test_window_relative_capture_bbox_is_computed_from_window_bbox() -> None:
    capture_bbox = _calculate_window_relative_capture_bbox(
        (100, 200, 1380, 920),
        CaptureRegion(left=20, top=30, width=640, height=320),
    )

    assert capture_bbox == (120, 230, 760, 550)


def test_should_start_rest_when_hp_or_condition_is_below_threshold() -> None:
    combat_config = load_default_config().combat

    assert should_start_rest(hp_ratio=0.49, condition_ratio=0.95, combat_config=combat_config) is True
    assert should_start_rest(hp_ratio=0.95, condition_ratio=0.49, combat_config=combat_config) is True
    assert should_start_rest(hp_ratio=0.95, condition_ratio=0.95, combat_config=combat_config) is False


def test_live_resource_provider_reads_hp_and_condition_from_pixels(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    frame = _build_state_resource_frame(
        settings,
        hp_ratio=0.74,
        condition_ratio=0.58,
        captured_at_ts=101.0,
    )

    snapshot = LiveResourceProvider(settings.live).read_resources(frame)

    assert snapshot.metadata["source"] == "pixel"
    assert snapshot.hp_ratio == pytest.approx(0.74, abs=0.03)
    assert snapshot.condition_ratio == pytest.approx(0.58, abs=0.03)
    assert snapshot.metadata["hp"]["roi"]["name"] == "hp_bar_roi"
    assert snapshot.metadata["condition"]["roi"]["name"] == "condition_bar_roi"


def test_simple_state_detector_reads_combat_and_reward_from_pixels(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    frame = _build_state_resource_frame(
        settings,
        hp_ratio=0.80,
        condition_ratio=0.80,
        in_combat=True,
        reward_visible=True,
        captured_at_ts=102.0,
    )

    snapshot = SimpleStateDetector(settings.live).detect_state(frame)

    assert snapshot.in_combat is True
    assert snapshot.reward_visible is True
    assert snapshot.metadata["source"] == "pixel"
    assert snapshot.metadata["combat_indicator"]["active_ratio"] >= settings.live.combat_indicator_min_ratio
    assert snapshot.metadata["reward_visibility"]["active_ratio"] >= settings.live.reward_min_ratio


def test_live_resource_provider_aggregates_multi_frame_reads_with_median(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    provider = LiveResourceProvider(settings.live)
    samples = (
        LiveResourceSnapshot(
            hp_ratio=0.60,
            condition_ratio=0.40,
            metadata={
                "source": "pixel",
                "hp": {"fill_ratio": 0.60, "matching_pixels": 40, "total_pixels": 100},
                "condition": {"fill_ratio": 0.40, "matching_pixels": 32, "total_pixels": 100},
            },
        ),
        LiveResourceSnapshot(
            hp_ratio=0.63,
            condition_ratio=0.45,
            metadata={
                "source": "pixel",
                "hp": {"fill_ratio": 0.63, "matching_pixels": 42, "total_pixels": 100},
                "condition": {"fill_ratio": 0.45, "matching_pixels": 35, "total_pixels": 100},
            },
        ),
        LiveResourceSnapshot(
            hp_ratio=0.61,
            condition_ratio=0.43,
            metadata={
                "source": "pixel",
                "hp": {"fill_ratio": 0.61, "matching_pixels": 41, "total_pixels": 100},
                "condition": {"fill_ratio": 0.43, "matching_pixels": 34, "total_pixels": 100},
            },
        ),
    )

    aggregate = provider.aggregate_resource_reads(samples)

    assert aggregate.sample_count == 3
    assert aggregate.hp_ratio == pytest.approx(0.61)
    assert aggregate.condition_ratio == pytest.approx(0.43)
    assert aggregate.hp_spread == pytest.approx(0.03)
    assert aggregate.condition_spread == pytest.approx(0.05)
    assert aggregate.confidence > 0.45
    assert "resource_confidence_low" in aggregate.warnings


def test_live_resource_provider_marks_unstable_reads_with_warnings(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    provider = LiveResourceProvider(settings.live)
    samples = (
        LiveResourceSnapshot(
            hp_ratio=0.20,
            condition_ratio=0.25,
            metadata={
                "source": "pixel",
                "hp": {"fill_ratio": 0.20, "matching_pixels": 8, "total_pixels": 100},
                "condition": {"fill_ratio": 0.25, "matching_pixels": 10, "total_pixels": 100},
            },
        ),
        LiveResourceSnapshot(
            hp_ratio=0.78,
            condition_ratio=0.82,
            metadata={
                "source": "pixel",
                "hp": {"fill_ratio": 0.78, "matching_pixels": 45, "total_pixels": 100},
                "condition": {"fill_ratio": 0.82, "matching_pixels": 48, "total_pixels": 100},
            },
        ),
        LiveResourceSnapshot(
            hp_ratio=0.34,
            condition_ratio=0.36,
            metadata={
                "source": "metadata_fallback",
                "hp": {"fill_ratio": 0.34, "matching_pixels": 0, "total_pixels": 100},
                "condition": {"fill_ratio": 0.36, "matching_pixels": 0, "total_pixels": 100},
            },
        ),
    )

    aggregate = provider.aggregate_resource_reads(
        samples,
        previous_snapshot=LiveResourceSnapshot(
            hp_ratio=0.70,
            condition_ratio=0.75,
            metadata={"source": "previous"},
        ),
    )

    assert aggregate.confidence < settings.live.rest_resource_min_confidence
    assert "resource_metadata_fallback_used" in aggregate.warnings
    assert "hp_read_spread_high" in aggregate.warnings
    assert "condition_read_spread_high" in aggregate.warnings
    assert "resource_confidence_low" in aggregate.warnings
    assert "hp_dropped_during_rest" in aggregate.warnings
    assert "condition_dropped_during_rest" in aggregate.warnings


def test_live_rest_provider_stops_when_threshold_is_stably_reached(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            rest_resource_sample_count=3,
            rest_resource_max_ticks=4,
            rest_resource_min_confidence=0.60,
            rest_resource_growth_min_delta=0.01,
        ),
    )
    frames = [
        _build_state_resource_frame(settings, hp_ratio=0.58, condition_ratio=0.55, captured_at_ts=100.00, source="rest-1-1"),
        _build_state_resource_frame(settings, hp_ratio=0.60, condition_ratio=0.56, captured_at_ts=100.10, source="rest-1-2"),
        _build_state_resource_frame(settings, hp_ratio=0.59, condition_ratio=0.57, captured_at_ts=100.20, source="rest-1-3"),
        _build_state_resource_frame(settings, hp_ratio=0.93, condition_ratio=0.92, captured_at_ts=100.50, source="rest-2-1"),
        _build_state_resource_frame(settings, hp_ratio=0.94, condition_ratio=0.93, captured_at_ts=100.60, source="rest-2-2"),
        _build_state_resource_frame(settings, hp_ratio=0.95, condition_ratio=0.94, captured_at_ts=100.70, source="rest-2-3"),
    ]
    runtime = _FakeRestRuntime(settings, frames)
    provider = LiveRestProvider(
        runtime,
        input_driver=LiveInputDriver(
            logger=logging.getLogger("botlab.test.rest"),
            dry_run=True,
            enable_real_input=False,
            enable_real_clicks=False,
            enable_real_keys=False,
        ),
        resource_provider=LiveResourceProvider(settings.live),
    )

    timeline = provider.apply_rest(
        1,
        rest_started_ts=100.0,
        starting_hp_ratio=0.55,
        starting_condition_ratio=0.50,
        observation=Observation(
            cycle_id=1,
            observed_at_ts=99.0,
            signal_detected=True,
            actual_spawn_ts=99.0,
            source="test",
            confidence=1.0,
            metadata={},
        ),
    )

    assert len(timeline.snapshots) == 2
    assert timeline.snapshots[-1].snapshot.metadata["rest_decision"] == "rest_stop_threshold_reached"
    assert timeline.metadata["rest_stop_threshold_reached_count"] == 1
    assert timeline.metadata["rest_stalled_or_uncertain_count"] == 0
    assert timeline.metadata["rest_stabilized_tick_index"] == 2
    assert timeline.metadata["resource_final_hp_ratio"] == pytest.approx(0.94, abs=0.03)
    assert timeline.metadata["resource_final_condition_ratio"] == pytest.approx(0.93, abs=0.03)
    assert runtime.updated_resources is not None
    assert runtime.updated_resources[0] >= settings.combat.rest_stop_threshold


def test_live_rest_provider_warns_when_read_is_unstable(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            rest_resource_sample_count=3,
            rest_resource_max_ticks=1,
            rest_resource_min_confidence=0.70,
            rest_resource_warning_spread_threshold=0.08,
        ),
    )
    frames = [
        _build_state_resource_frame(settings, hp_ratio=0.20, condition_ratio=0.20, captured_at_ts=100.00, source="rest-u-1"),
        _build_state_resource_frame(settings, hp_ratio=0.80, condition_ratio=0.82, captured_at_ts=100.10, source="rest-u-2"),
        _build_state_resource_frame(settings, hp_ratio=0.35, condition_ratio=0.30, captured_at_ts=100.20, source="rest-u-3"),
    ]
    runtime = _FakeRestRuntime(settings, frames)
    provider = LiveRestProvider(
        runtime,
        input_driver=LiveInputDriver(
            logger=logging.getLogger("botlab.test.rest"),
            dry_run=True,
            enable_real_input=False,
            enable_real_clicks=False,
            enable_real_keys=False,
        ),
        resource_provider=LiveResourceProvider(settings.live),
    )

    timeline = provider.apply_rest(
        1,
        rest_started_ts=100.0,
        starting_hp_ratio=0.25,
        starting_condition_ratio=0.25,
        observation=Observation(
            cycle_id=1,
            observed_at_ts=99.0,
            signal_detected=True,
            actual_spawn_ts=99.0,
            source="test",
            confidence=1.0,
            metadata={},
        ),
    )

    assert len(timeline.snapshots) == 1
    snapshot_metadata = timeline.snapshots[0].snapshot.metadata
    assert snapshot_metadata["rest_decision"] == "rest_stalled_or_uncertain"
    assert snapshot_metadata["resource_warning_count"] >= 1
    assert "resource_confidence_low" in snapshot_metadata["resource_warnings"]
    assert timeline.metadata["rest_stalled_or_uncertain_count"] == 1
    assert timeline.metadata["resource_warning_count"] >= 1


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


def test_scene_profile_loader_reads_single_spot_profile() -> None:
    settings = load_config("config/live_real_mvp.yaml")
    loader = SceneProfileLoader(settings.live.scene_profile_path)

    profile = loader.load()

    assert profile is not None
    assert profile.scene_name == "single_spot_scene"
    assert len(profile.spawn_zone_polygon) >= 4
    assert profile.reference_frame_path is not None
    assert profile.reference_frame_path.name == "live_spot_scene_1.png"


def test_scene_profile_calibration_scales_polygon_and_reference_point(tmp_path: Path) -> None:
    from PIL import Image

    reference_path = tmp_path / "reference_scene.png"
    Image.new("RGB", (200, 100), color=(0, 0, 0)).save(reference_path)
    profile_path = tmp_path / "scene_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "scene_name": "scaled_scene",
                "reference_frame_path": str(reference_path),
                "reference_point_xy": [100, 50],
                "spawn_zone_polygon": [
                    [20, 10],
                    [180, 10],
                    [180, 90],
                    [20, 90],
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    profile = SceneProfileLoader(profile_path).load()

    assert profile is not None
    calibrated = profile.calibrate(frame_width=100, frame_height=50, offset_xy=(10, 4))

    assert calibrated.reference_frame_size == (200, 100)
    assert calibrated.frame_size == (100, 50)
    assert calibrated.scale_x == pytest.approx(0.5)
    assert calibrated.scale_y == pytest.approx(0.5)
    assert calibrated.reference_point_xy == pytest.approx((60.0, 29.0))
    assert calibrated.spawn_zone_polygon[0] == pytest.approx((20.0, 9.0))
    assert calibrated.spawn_zone_polygon[2] == pytest.approx((100.0, 49.0))


def test_scene_profile_calibration_can_follow_frame_center_anchor(tmp_path: Path) -> None:
    from PIL import Image

    reference_path = tmp_path / "reference_scene.png"
    Image.new("RGB", (200, 100), color=(0, 0, 0)).save(reference_path)
    profile_path = tmp_path / "scene_profile_anchor.json"
    profile_path.write_text(
        json.dumps(
            {
                "scene_name": "anchored_scene",
                "reference_frame_path": str(reference_path),
                "reference_point_xy": [60, 40],
                "spawn_zone_polygon": [
                    [20, 20],
                    [120, 20],
                    [120, 80],
                    [20, 80],
                ],
                "sub_rois": {
                    "spawn_focus_roi": [10, 15, 80, 40]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    profile = SceneProfileLoader(profile_path).load()

    assert profile is not None
    calibrated = profile.calibrate(
        frame_width=200,
        frame_height=100,
        offset_xy=(0, 0),
        anchor_target_xy=(100.0, 50.0),
        anchor_mode="frame_center",
    )

    assert calibrated.reference_point_xy == pytest.approx((100.0, 50.0))
    assert calibrated.anchor_mode == "frame_center"
    assert calibrated.anchor_target_xy == pytest.approx((100.0, 50.0))
    assert calibrated.anchor_adjustment_xy == pytest.approx((40.0, 10.0))
    assert calibrated.spawn_zone_polygon[0] == pytest.approx((60.0, 30.0))
    assert calibrated.sub_rois["spawn_focus_roi"] == (50, 25, 80, 40)


def test_scene_profile_spawn_focus_roi_overrides_config_spawn_roi(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    scene_profile_path = tmp_path / "scene_profile.json"
    scene_profile_path.write_text(
        json.dumps(
            {
                "scene_name": "spawn_focus_scene",
                "reference_frame_path": None,
                "reference_point_xy": [100, 100],
                "spawn_zone_polygon": [
                    [10, 10],
                    [210, 10],
                    [210, 210],
                    [10, 210],
                ],
                "sub_rois": {
                    "spawn_focus_roi": [30, 40, 120, 90]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings = replace(settings, live=replace(settings.live, scene_profile_path=scene_profile_path))
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "scene-focus-output",
    )
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 240,
                "height": 220,
                "captured_at_ts": 1.0,
                "source": "scene-focus",
                "metadata": {
                    "template_hits": [],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert result.roi["x"] == 30
    assert result.roi["y"] == 40
    assert result.roi["width"] == 120
    assert result.roi["height"] == 90
    assert result.roi["source"] == "scene_profile"


def test_scene_calibration_is_applied_before_zone_filtering(tmp_path: Path) -> None:
    from PIL import Image

    settings = _build_live_settings(tmp_path)
    reference_path = tmp_path / "reference_scene.png"
    Image.new("RGB", (200, 100), color=(0, 0, 0)).save(reference_path)
    scene_profile_path = tmp_path / "scene_profile.json"
    scene_profile_path.write_text(
        json.dumps(
            {
                "scene_name": "scaled_scene",
                "reference_frame_path": str(reference_path),
                "reference_point_xy": [100, 90],
                "spawn_zone_polygon": [
                    [20, 10],
                    [180, 10],
                    [180, 90],
                    [20, 90],
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings = replace(settings, live=replace(settings.live, scene_profile_path=scene_profile_path))
    frame_path = tmp_path / "scaled_scene_frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 100,
                "height": 50,
                "captured_at_ts": 12.0,
                "source": "scene-calibration-fixture",
                "metadata": {
                    "spawn_roi": [0, 0, 100, 50],
                    "template_hits": [
                        {
                            "label": "mob_a",
                            "x": 38,
                            "y": 16,
                            "width": 16,
                            "height": 16,
                            "confidence": 0.95,
                            "target_id": "inside-calibrated",
                        },
                        {
                            "label": "mob_b",
                            "x": -4,
                            "y": 16,
                            "width": 16,
                            "height": 16,
                            "confidence": 0.94,
                            "target_id": "outside-calibrated",
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
        output_directory=tmp_path / "perception-scene-calibration",
    )

    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert result.scene_name == "scaled_scene"
    assert result.scene_calibration["frame_size"] == [100, 50]
    assert result.selected_target_id == "inside-calibrated"
    assert [target.target_id for target in result.out_of_zone_detections] == ["outside-calibrated"]


def test_scene_aware_perception_rejects_out_of_zone_targets_and_selects_nearest_in_zone(
    tmp_path: Path,
) -> None:
    settings = _build_live_settings(tmp_path)
    scene_profile_path = tmp_path / "scene_profile.json"
    scene_profile_path.write_text(
        json.dumps(
            {
                "scene_name": "test_scene",
                "reference_frame_path": None,
                "reference_point_xy": [100, 180],
                "spawn_zone_polygon": [
                    [40, 40],
                    [160, 40],
                    [160, 160],
                    [40, 160]
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings = replace(settings, live=replace(settings.live, scene_profile_path=scene_profile_path))
    frame_path = tmp_path / "scene_frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 240,
                "height": 220,
                "captured_at_ts": 12.0,
                "source": "scene-aware-fixture",
                "metadata": {
                    "reference_point_xy": [100, 180],
                    "spawn_roi": [0, 0, 240, 220],
                    "template_hits": [
                        {
                            "label": "mob_a",
                            "x": 70,
                            "y": 70,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.94,
                            "target_id": "inside-free"
                        },
                        {
                            "label": "mob_b",
                            "x": 178,
                            "y": 72,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.92,
                            "target_id": "outside-free"
                        },
                        {
                            "label": "mob_a",
                            "x": 92,
                            "y": 132,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.93,
                            "target_id": "inside-busy"
                        },
                        {
                            "label": "occupied_swords",
                            "x": 96,
                            "y": 112,
                            "width": 24,
                            "height": 18,
                            "confidence": 0.96,
                            "target_id": "inside-busy"
                        }
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
        output_directory=tmp_path / "scene-aware-perception",
    )

    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert result.scene_name == "test_scene"
    assert len(result.in_zone_detections) == 2
    assert len(result.out_of_zone_detections) == 1
    assert result.out_of_zone_detections[0].target_id == "outside-free"
    assert result.out_of_zone_detections[0].reachable is False
    assert result.selected_target_id == "inside-free"
    assert len(result.selectable_detections) == 1
    assert summary.out_of_zone_rejections.avg_ms == pytest.approx(1.0)


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
    assert summary.accuracy_summary is None
    assert (output_directory / "batch_frame_a_perception.json").exists() is True
    assert (output_directory / "batch_frame_b_perception.json").exists() is True
    assert (output_directory / "perception_session_summary.json").exists() is True
    assert summary.real_scene_regression_entries() == ()


def test_perception_session_summary_can_emit_real_scene_regression_entries() -> None:
    settings = load_config("config/live_dry_run.yaml")
    analyzer = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=Path("data") / "unused-perception-summary-test",
    )._analyzer
    result = analyzer.analyze_frame(
        LiveFrame(
            width=1280,
            height=720,
            captured_at_ts=0.0,
            source="live_spot_scene_1",
            metadata={
                "reference_point_xy": [640, 360],
                "template_hits": [
                    {
                        "label": "mob_a",
                        "x": 600,
                        "y": 260,
                        "width": 40,
                        "height": 52,
                        "confidence": 0.93,
                        "target_id": "free-near",
                    },
                    {
                        "label": "mob_b",
                        "x": 720,
                        "y": 280,
                        "width": 42,
                        "height": 54,
                        "confidence": 0.91,
                        "target_id": "busy-mid",
                    },
                    {
                        "label": "occupied_swords",
                        "x": 722,
                        "y": 244,
                        "width": 28,
                        "height": 18,
                        "confidence": 0.95,
                        "target_id": "busy-mid",
                    },
                ],
            },
            image=None,
        ),
        cycle_id=1,
        phase="live_spot_scene_1",
    )
    result = replace(
        result,
        timings=ReactionLatency(
            frame_captured_ts=0.0,
            detection_started_ts=0.0,
            detection_finished_ts=0.012,
            target_selected_ts=0.016,
            action_ready_ts=0.018,
        ),
    )
    summary = analyzer.summarize_session((result,))

    entries = summary.real_scene_regression_entries()

    assert len(entries) == 1
    assert entries[0]["frame_source"] == "live_spot_scene_1"
    assert entries[0]["target_count"] == 2
    assert entries[0]["free_target_count"] == 1
    assert entries[0]["occupied_target_count"] == 1
    assert entries[0]["selected_target_id"] == "free-near"
    assert entries[0]["selected_target_xy"] == [620, 286]
    assert entries[0]["occupied_target_xy"] == [[741, 307]]
    assert "real_scene_regression" in summary.to_dict()


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


def test_perception_frame_loader_loads_nested_directory_structure(tmp_path: Path) -> None:
    nested_directory = tmp_path / "nested" / "hard"
    nested_directory.mkdir(parents=True)
    frame_path = nested_directory / "frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 640,
                "height": 360,
                "captured_at_ts": 1.0,
                "source": "nested-frame",
                "metadata": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loader = PerceptionFrameLoader()
    entries = loader.load_directory(tmp_path)

    assert len(entries) == 1
    assert entries[0][0] == "nested__hard__frame"
    assert entries[0][1].source == "nested-frame"


def test_benchmark_dataset_loader_reads_split_manifest(tmp_path: Path) -> None:
    settings = _write_marker_first_benchmark_dataset(tmp_path)
    loader = BenchmarkDatasetLoader(
        dataset_root=settings.live.benchmark_dataset_directory,
        frame_loader=PerceptionFrameLoader(),
    )

    entries = loader.load_split("regression")

    assert len(entries) == 2
    assert entries[0][0] == "regression__benchmark_free"
    assert entries[1][0] == "regression__benchmark_occupied"
    assert entries[0][1].metadata["dataset_split"] == "regression"


def test_ground_truth_is_loaded_from_sidecar_for_benchmark_frames(tmp_path: Path) -> None:
    settings = _write_marker_first_benchmark_dataset(tmp_path)
    frame = PerceptionFrameLoader().load_frame(
        settings.live.sample_frames_directory / "benchmark_free.png"
    )

    ground_truth = frame.metadata.get("ground_truth")

    assert isinstance(ground_truth, dict)
    assert len(ground_truth["candidates"]) == 1
    assert ground_truth["candidates"][0]["candidate_id"] == "free_primary"


def test_strict_pixel_based_mode_rejects_json_only_frame_specs(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    frame_path = tmp_path / "json_only_frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 320,
                "height": 240,
                "captured_at_ts": 1.0,
                "source": "json-only",
                "metadata": {
                    "template_hits": [
                        {
                            "label": "mob_a",
                            "x": 10,
                            "y": 10,
                            "width": 20,
                            "height": 20,
                            "confidence": 0.9,
                        }
                    ]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "strict-json-only",
        strict_pixel_only=True,
    )

    with pytest.raises(RuntimeError, match="Strict pixel-based benchmark wymaga prawdziwego obrazu"):
        runner.analyze_frame_path(frame_path)


def test_strict_benchmark_split_reports_selected_target_and_occupied_accuracy(tmp_path: Path) -> None:
    settings = _write_marker_first_benchmark_dataset(tmp_path)
    output_directory = tmp_path / "benchmark-output"
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
        strict_pixel_only=True,
    )

    summary = runner.analyze_benchmark_split("regression")

    assert summary.strict_pixel_only is True
    assert summary.benchmark_summary is not None
    assert summary.benchmark_summary.evaluated_frame_count == 2
    assert summary.benchmark_summary.strict_pixel_only is True
    assert summary.benchmark_summary.fallback_frame_count == 0
    assert summary.benchmark_summary.selected_target_accuracy == pytest.approx(1.0)
    assert summary.benchmark_summary.occupied_classification_accuracy == pytest.approx(1.0)
    assert summary.benchmark_summary.target_recall == pytest.approx(1.0)
    assert summary.benchmark_summary.target_precision == pytest.approx(1.0)
    assert (output_directory / "regression__benchmark_free_perception.json").exists() is True
    assert (output_directory / "regression__benchmark_occupied_perception.json").exists() is True
    payload = json.loads((output_directory / "perception_session_summary.json").read_text(encoding="utf-8"))
    assert payload["benchmark_summary"]["strict_pixel_only"] is True
    assert payload["benchmark_summary"]["target_recall"] == pytest.approx(1.0)


def test_benchmark_summary_aggregates_false_positive_and_false_negative_counts(tmp_path: Path) -> None:
    settings = _write_marker_first_benchmark_dataset(tmp_path)
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "benchmark-aggregate-output",
        strict_pixel_only=True,
    )

    summary = runner.analyze_benchmark_split("regression")

    assert summary.benchmark_summary is not None
    assert summary.benchmark_summary.false_positive_count.count == 2
    assert summary.benchmark_summary.false_negative_count.count == 2
    assert summary.benchmark_summary.false_positive_count.max_value == pytest.approx(0.0)
    assert summary.benchmark_summary.false_negative_count.max_value == pytest.approx(0.0)


def test_perception_result_contains_detection_diagnostics(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    frame_path = tmp_path / "diagnostics_frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 320,
                "height": 240,
                "captured_at_ts": 30.0,
                "source": "diagnostics-frame",
                "metadata": {
                    "reference_point_xy": [160, 210],
                    "template_hits": [
                        {
                            "label": "mob_a",
                            "x": 120,
                            "y": 80,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.90,
                            "target_id": "inside-free",
                            "metadata": {"raw_hit_count": 2},
                        },
                        {
                            "label": "mob_b",
                            "x": 220,
                            "y": 80,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.20,
                            "target_id": "low-confidence",
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
        output_directory=tmp_path / "diagnostics-output",
    )

    summary = runner.analyze_frame_path(frame_path)
    diagnostics = summary.frame_results[0].diagnostics

    assert diagnostics["raw_hit_summary"]["mob_a"] == 1
    assert diagnostics["raw_hit_summary"]["mob_b"] == 1
    assert len(diagnostics["low_confidence_hits"]) == 1
    assert diagnostics["low_confidence_hits"][0]["rejection_reason"] == "low_confidence"
    assert diagnostics["selection_reason"] == "nearest_free_in_zone_reachable_target"
    assert len(diagnostics["summary_lines"]) >= 3


def test_benchmark_summary_contains_worst_frames_ranked_by_errors(tmp_path: Path) -> None:
    settings = _write_marker_first_benchmark_dataset(tmp_path)
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "worst-frame-output",
        strict_pixel_only=True,
    )

    summary = runner.analyze_benchmark_split("regression")

    assert len(summary.worst_frames) >= 1
    assert "frame_source" in summary.worst_frames[0]
    assert "false_positive_count" in summary.worst_frames[0]
    persisted = json.loads(
        (tmp_path / "worst-frame-output" / "perception_session_summary.json").read_text(encoding="utf-8")
    )
    assert isinstance(persisted["worst_frames"], list)


def test_compare_perception_summary_payloads_reports_accuracy_and_latency_deltas() -> None:
    baseline_payload = {
        "benchmark_summary": {
            "target_recall": 0.70,
            "target_precision": 0.75,
            "occupied_classification_accuracy": 0.80,
            "selected_target_accuracy": 0.60,
            "target_false_positive_count": 6,
            "target_false_negative_count": 5,
        },
        "detection_latency": {"avg_ms": 20.0},
        "selection_latency": {"avg_ms": 5.0},
        "total_reaction_latency": {"avg_ms": 28.0},
    }
    candidate_payload = {
        "benchmark_summary": {
            "target_recall": 0.82,
            "target_precision": 0.79,
            "occupied_classification_accuracy": 0.90,
            "selected_target_accuracy": 0.75,
            "target_false_positive_count": 4,
            "target_false_negative_count": 3,
        },
        "detection_latency": {"avg_ms": 24.0},
        "selection_latency": {"avg_ms": 4.0},
        "total_reaction_latency": {"avg_ms": 31.0},
    }

    comparison = compare_perception_summary_payloads(baseline_payload, candidate_payload)

    assert comparison["target_recall_delta"] == pytest.approx(0.12)
    assert comparison["target_precision_delta"] == pytest.approx(0.04)
    assert comparison["selected_target_accuracy_delta"] == pytest.approx(0.15)
    assert comparison["false_positive_total_delta"] == pytest.approx(-2.0)
    assert comparison["detection_latency_avg_ms_delta"] == pytest.approx(4.0)


def test_benchmark_summary_tracks_out_of_zone_rejections_for_scene_aware_frames(
    tmp_path: Path,
) -> None:
    settings = _build_live_settings(tmp_path)
    scene_profile_path = tmp_path / "scene_profile.json"
    scene_profile_path.write_text(
        json.dumps(
            {
                "scene_name": "benchmark_scene",
                "reference_frame_path": None,
                "reference_point_xy": [100, 180],
                "spawn_zone_polygon": [
                    [40, 40],
                    [160, 40],
                    [160, 160],
                    [40, 160]
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings = replace(settings, live=replace(settings.live, scene_profile_path=scene_profile_path))
    frame_path = tmp_path / "benchmark_scene_frame.json"
    frame_path.write_text(
        json.dumps(
            {
                "width": 240,
                "height": 220,
                "captured_at_ts": 14.0,
                "source": "benchmark_scene_frame",
                "metadata": {
                    "reference_point_xy": [100, 180],
                    "spawn_roi": [0, 0, 240, 220],
                    "ground_truth": {
                        "candidates": [
                            {
                                "candidate_id": "inside-free",
                                "screen_xy": [90, 95],
                                "max_error_px": 40,
                                "occupied": False,
                                "selected": True
                            }
                        ]
                    },
                    "template_hits": [
                        {
                            "label": "mob_a",
                            "x": 70,
                            "y": 70,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.94,
                            "target_id": "inside-free"
                        },
                        {
                            "label": "mob_b",
                            "x": 180,
                            "y": 72,
                            "width": 40,
                            "height": 50,
                            "confidence": 0.91,
                            "target_id": "outside-free"
                        }
                    ]
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "scene-aware-benchmark",
    )

    summary = runner.analyze_frame_path(frame_path)

    assert summary.benchmark_summary is not None
    assert summary.benchmark_summary.out_of_zone_rejection_count.avg_value == pytest.approx(1.0)
    assert summary.benchmark_summary.selected_target_in_zone_accuracy == pytest.approx(1.0)
    assert summary.benchmark_summary.false_positive_reduction_after_zone_filtering.avg_value == pytest.approx(1.0)


def test_marker_first_perception_detects_occupied_and_selects_nearest_free_target(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    output_directory = tmp_path / "perception-marker-first"
    frame_path = tmp_path / "marker_frame.png"
    sidecar_path = tmp_path / "marker_frame.json"

    frame_image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(frame_image)

    mob_a_template = Image.open(settings.live.mobs_template_directory / "mob_a" / "base.png").convert("RGB")
    mob_b_template = Image.open(settings.live.mobs_template_directory / "mob_b" / "base.png").convert("RGB")
    swords_template = Image.open(settings.live.occupied_template_directory / "crossed_swords.png").convert("RGB")

    frame_image.paste(mob_b_template, (58, 82))
    draw.polygon([(67, 72), (71, 66), (75, 72), (71, 78)], fill=(225, 55, 55))
    frame_image.paste(swords_template, (65, 70))

    frame_image.paste(mob_a_template, (150, 102))
    draw.polygon([(159, 90), (163, 84), (167, 90), (163, 96)], fill=(225, 55, 55))

    frame_image.paste(mob_b_template, (250, 74))
    draw.polygon([(259, 62), (263, 56), (267, 62), (263, 68)], fill=(225, 55, 55))

    frame_image.save(frame_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 100.0,
                "source": "marker-first-test",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                    "perception_profile": {
                        "detection_duration_s": 0.010,
                        "selection_duration_s": 0.003,
                        "action_ready_duration_s": 0.002,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )
    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert len(result.detections) == 3
    assert len(result.occupied_detections) == 1
    assert len(result.free_detections) == 2
    assert result.selected_target is not None
    assert result.selected_target.occupied is False
    assert result.selected_target.mob_variant == "mob_a"
    assert abs(result.selected_target.screen_x - 159) <= 8
    assert abs(result.selected_target.screen_y - 114) <= 10
    assert result.timings.detection_latency_ms == pytest.approx(10.0)
    assert result.timings.selection_latency_ms == pytest.approx(3.0)
    assert result.timings.total_reaction_latency_ms == pytest.approx(15.0)
    assert "marker_bbox" in result.selected_target.metadata
    assert "confirmation_roi" in result.selected_target.metadata
    assert "occupied_roi" in result.selected_target.metadata
    assert "occupied_green_ratio" in result.selected_target.metadata
    assert "occupied_template_match_enabled" in result.selected_target.metadata
    assert "confirmation_foreground_score" in result.selected_target.metadata
    assert (output_directory / "marker_frame_perception.json").exists() is True


def test_marker_first_perception_can_detect_yellow_marker_for_live_profile(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            marker_color_mode="yellow",
            marker_min_red=170,
            marker_min_green=145,
            marker_red_blue_delta=25,
            marker_green_blue_delta=40,
            marker_red_green_balance_delta=70,
            marker_confidence_threshold=0.40,
        ),
    )
    output_directory = tmp_path / "perception-yellow-marker"
    frame_path = tmp_path / "yellow_marker_frame.png"
    sidecar_path = tmp_path / "yellow_marker_frame.json"

    frame_image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(frame_image)
    mob_a_template = Image.open(settings.live.mobs_template_directory / "mob_a" / "base.png").convert("RGB")

    frame_image.paste(mob_a_template, (150, 102))
    draw.polygon([(159, 90), (163, 84), (167, 90), (163, 96)], fill=(230, 205, 32))

    frame_image.save(frame_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 101.0,
                "source": "yellow-marker-test",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )
    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert result.candidate_hit_count >= 1
    assert any(hit.label == "yellow_marker" for hit in result.raw_hits)
    assert len(result.detections) >= 1


def test_marker_first_perception_rejects_large_yellow_blob_without_dark_core(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            marker_color_mode="yellow",
            marker_min_red=170,
            marker_min_green=145,
            marker_red_blue_delta=25,
            marker_green_blue_delta=40,
            marker_red_green_balance_delta=70,
            marker_min_blob_pixels=10,
            marker_max_blob_pixels=72,
            marker_min_width_px=5,
            marker_max_width_px=18,
            marker_min_height_px=6,
            marker_max_height_px=20,
            marker_min_fill_density=0.18,
            marker_max_fill_density=0.78,
            marker_dark_core_max_rgb=125,
            marker_min_dark_core_ratio=0.08,
            marker_confidence_threshold=0.40,
        ),
    )
    output_directory = tmp_path / "perception-yellow-blob"
    frame_path = tmp_path / "yellow_blob_frame.png"
    sidecar_path = tmp_path / "yellow_blob_frame.json"

    frame_image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(frame_image)
    draw.ellipse((150, 90, 178, 118), fill=(240, 190, 25))
    frame_image.save(frame_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 102.0,
                "source": "yellow-blob-test",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )
    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert all(hit.label != "yellow_marker" for hit in result.raw_hits)
    assert len(result.detections) == 0


def test_marker_first_perception_rejects_player_like_candidate_with_green_name_veto(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            marker_color_mode="yellow",
            marker_min_red=170,
            marker_min_green=135,
            marker_red_blue_delta=25,
            marker_green_blue_delta=22,
            marker_red_green_balance_delta=70,
            player_veto_enabled=True,
            player_veto_roi_width_px=170,
            player_veto_roi_height_px=52,
            player_veto_roi_offset_y_px=-38,
            player_veto_green_min_green=120,
            player_veto_green_red_delta=15,
            player_veto_green_blue_delta=10,
            player_veto_min_pixels=18,
            player_veto_min_width_px=26,
            player_veto_max_height_px=18,
        ),
    )
    output_directory = tmp_path / "perception-player-veto"
    frame_path = tmp_path / "player_veto_frame.png"
    sidecar_path = tmp_path / "player_veto_frame.json"

    frame_image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(frame_image)
    mob_a_template = Image.open(settings.live.mobs_template_directory / "mob_a" / "base.png").convert("RGB")

    frame_image.paste(mob_a_template, (150, 104))
    draw.polygon([(159, 92), (163, 86), (167, 92), (163, 98)], fill=(230, 205, 32))
    draw.rectangle((120, 82, 205, 94), fill=(20, 210, 40))

    frame_image.save(frame_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 103.0,
                "source": "player-veto-test",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )
    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert len(result.detections) == 0
    vetoes = result.diagnostics.get("player_veto_rejections", [])
    assert isinstance(vetoes, list)
    assert len(vetoes) >= 1
    assert vetoes[0]["rejection_reason"] == "player_veto_green_name"


def test_marker_first_perception_rejects_non_icy_candidate_with_marker_present(
    tmp_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            marker_color_mode="yellow",
            confirmation_confidence_threshold=0.35,
            ice_mob_signature_enabled=True,
            ice_mob_min_blue=135,
            ice_mob_min_green=105,
            ice_mob_min_brightness=118,
            ice_mob_blue_red_tolerance=16,
            ice_mob_min_pixels=42,
            ice_mob_min_ratio=0.10,
        ),
    )
    output_directory = tmp_path / "perception-ice-signature"
    frame_path = tmp_path / "ice_signature_frame.png"
    sidecar_path = tmp_path / "ice_signature_frame.json"

    frame_image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(frame_image)
    draw.rectangle((150, 104, 167, 127), fill=(120, 78, 48))
    draw.rectangle((154, 106, 162, 112), fill=(168, 138, 98))
    draw.polygon([(159, 92), (163, 86), (167, 92), (163, 98)], fill=(230, 205, 32))

    frame_image.save(frame_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 104.0,
                "source": "ice-signature-test",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )
    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert len(result.detections) == 0
    signature_rejections = result.diagnostics.get("mob_signature_rejections", [])
    assert isinstance(signature_rejections, list)
    assert len(signature_rejections) >= 1
    assert signature_rejections[0]["rejection_reason"] == "mob_signature_not_icy"


def test_live_preview_state_and_renderer_prepare_debug_overlay(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    settings = _build_marker_first_settings(tmp_path)
    output_directory = tmp_path / "perception-preview"
    frame_path = tmp_path / "preview_frame.png"
    sidecar_path = tmp_path / "preview_frame.json"

    frame_image = Image.new("RGB", (320, 240), color=(40, 40, 40))
    draw = ImageDraw.Draw(frame_image)
    mob_a_template = Image.open(settings.live.mobs_template_directory / "mob_a" / "base.png").convert("RGB")
    frame_image.paste(mob_a_template, (150, 102))
    draw.polygon([(159, 90), (163, 84), (167, 90), (163, 96)], fill=(225, 55, 55))
    frame_image.save(frame_path)
    sidecar_path.write_text(
        json.dumps(
            {
                "captured_at_ts": 100.0,
                "source": "preview-render-test",
                "metadata": {
                    "spawn_roi": [0, 0, 320, 240],
                    "reference_point_xy": [160, 210],
                    "perception_profile": {
                        "detection_duration_s": 0.010,
                        "selection_duration_s": 0.003,
                        "action_ready_duration_s": 0.002,
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )
    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]
    frame = PerceptionFrameLoader().load_frame(frame_path)
    state = build_live_preview_state(frame=frame, result=result)
    renderer = LiveVisionPreviewRenderer(max_width_px=200, max_height_px=160)
    image = renderer.render(frame=frame, result=result, state=state)

    assert state.candidate_count == len(result.detections)
    assert state.free_target_count == len(result.selectable_detections)
    assert state.selected_target_id == result.selected_target_id
    assert any("reaction=" in line for line in state.headline_lines)
    assert image.size[0] <= 200
    assert image.size[1] <= 160


def test_live_engage_observe_can_render_next_attempt(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    observe = LiveEngageObserve(settings=settings, enable_console=False)

    state, image = observe.render_next_attempt()

    assert state.selected_target_id is not None
    assert any("engage=" in line for line in state.headline_lines)
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_live_engage_observe_forces_safe_dry_run_even_for_real_live_settings(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    settings = replace(settings, live=replace(settings.live, dry_run=False))

    observe = LiveEngageObserve(settings=settings, enable_console=False)
    state, _image = observe.render_next_attempt()

    assert observe.safe_dry_run_enabled is True
    assert any("input_mode=dry_run_safe" in line for line in state.headline_lines)


def test_simple_state_detector_can_detect_combat_indicator_from_pixels(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    settings = _build_live_settings(tmp_path)
    image = Image.new("RGB", (1280, 720), color=(10, 10, 10))
    draw = ImageDraw.Draw(image)
    roi = settings.live.combat_indicator_roi
    draw.rectangle(
        (
            roi[0] + 10,
            roi[1] + 10,
            roi[0] + 70,
            roi[1] + 28,
        ),
        fill=(230, 40, 40),
    )
    frame = LiveFrame(
        width=1280,
        height=720,
        captured_at_ts=1.0,
        source="combat-indicator-pixels",
        metadata={},
        image=image,
    )

    state = SimpleStateDetector(settings.live).detect_state(frame)

    assert state.in_combat is True
    assert state.reward_visible is False


def test_live_resource_provider_reads_hp_and_condition_from_pixels(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    settings = _build_live_settings(tmp_path)
    image = Image.new("RGB", (1280, 720), color=(10, 10, 10))
    draw = ImageDraw.Draw(image)
    hp_roi = settings.live.hp_bar_roi
    condition_roi = settings.live.condition_bar_roi
    draw.rectangle(
        (
            hp_roi[0],
            hp_roi[1],
            hp_roi[0] + int(hp_roi[2] * 0.50),
            hp_roi[1] + hp_roi[3] - 1,
        ),
        fill=(220, 30, 30),
    )
    draw.rectangle(
        (
            condition_roi[0],
            condition_roi[1],
            condition_roi[0] + int(condition_roi[2] * 0.75),
            condition_roi[1] + condition_roi[3] - 1,
        ),
        fill=(40, 200, 40),
    )
    frame = LiveFrame(
        width=1280,
        height=720,
        captured_at_ts=1.0,
        source="resource-bars",
        metadata={
            "hp_ratio": 1.0,
            "condition_ratio": 1.0,
        },
        image=image,
    )

    resources = LiveResourceProvider(settings.live).read_resources(frame)

    assert resources.metadata["source"] == "pixel"
    assert resources.hp_ratio == pytest.approx(0.5, rel=0.05)
    assert resources.condition_ratio == pytest.approx(0.75, rel=0.05)


def test_simple_state_detector_can_detect_reward_from_pixels(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    settings = _build_live_settings(tmp_path)
    image = Image.new("RGB", (1280, 720), color=(10, 10, 10))
    draw = ImageDraw.Draw(image)
    roi = settings.live.reward_roi
    draw.rectangle(
        (
            roi[0] + 12,
            roi[1] + 12,
            roi[0] + 92,
            roi[1] + 54,
        ),
        fill=(230, 180, 60),
    )
    frame = LiveFrame(
        width=1280,
        height=720,
        captured_at_ts=1.0,
        source="reward-pixels",
        metadata={
            "reward_visible": False,
        },
        image=image,
    )

    state = SimpleStateDetector(settings.live).detect_state(frame)

    assert state.reward_visible is True
    assert state.metadata["source"] == "pixel"


def test_live_resource_provider_aggregates_multi_frame_reads_with_median_and_confidence(
    tmp_path: Path,
) -> None:
    settings = _build_live_settings(tmp_path)
    provider = LiveResourceProvider(settings.live)
    samples = (
        provider.read_resources(_build_resource_frame(settings, hp_ratio=0.50, condition_ratio=0.60)),
        provider.read_resources(_build_resource_frame(settings, hp_ratio=0.55, condition_ratio=0.62)),
        provider.read_resources(_build_resource_frame(settings, hp_ratio=0.90, condition_ratio=0.95)),
    )

    aggregate = provider.aggregate_resource_reads(samples)

    assert aggregate.sample_count == 3
    assert aggregate.hp_ratio == pytest.approx(0.55, rel=0.05)
    assert aggregate.condition_ratio == pytest.approx(0.62, rel=0.05)
    assert aggregate.confidence > 0.0


def test_live_resource_provider_marks_unstable_reads_with_warning(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    provider = LiveResourceProvider(
        replace(settings.live, rest_resource_warning_spread_threshold=0.05, rest_resource_min_confidence=0.8)
    )
    samples = (
        provider.read_resources(_build_resource_frame(settings, hp_ratio=0.20, condition_ratio=0.30)),
        provider.read_resources(_build_resource_frame(settings, hp_ratio=0.85, condition_ratio=0.88)),
        provider.read_resources(_build_resource_frame(settings, hp_ratio=0.25, condition_ratio=0.32)),
    )

    aggregate = provider.aggregate_resource_reads(samples)

    assert "hp_read_spread_high" in aggregate.warnings
    assert "condition_read_spread_high" in aggregate.warnings
    assert "resource_confidence_low" in aggregate.warnings


def test_live_rest_provider_uses_stable_multi_frame_read_to_stop_rest(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    live_settings = replace(
        settings.live,
        rest_resource_sample_count=3,
        rest_resource_sample_interval_s=0.01,
        rest_resource_max_ticks=4,
        rest_resource_min_confidence=0.50,
    )
    settings = replace(settings, live=live_settings)
    frames = [
        _build_resource_frame(settings, hp_ratio=0.55, condition_ratio=0.60),
        _build_resource_frame(settings, hp_ratio=0.57, condition_ratio=0.61),
        _build_resource_frame(settings, hp_ratio=0.56, condition_ratio=0.62),
        _build_resource_frame(settings, hp_ratio=0.93, condition_ratio=0.91),
        _build_resource_frame(settings, hp_ratio=0.94, condition_ratio=0.92),
        _build_resource_frame(settings, hp_ratio=0.95, condition_ratio=0.93),
    ]

    class FakeRuntime:
        def __init__(self, settings_obj: Settings, frame_list: list[LiveFrame]) -> None:
            self.settings = settings_obj
            self._frames = list(frame_list)
            self.updated_resources: tuple[float, float] | None = None

        def capture_frame(self, *, cycle_id: int, phase: str, default_ts: float) -> LiveFrame:
            assert self._frames, f"Brak przygotowanej klatki dla phase={phase}"
            return self._frames.pop(0)

        def update_resources(self, *, hp_ratio: float, condition_ratio: float) -> None:
            self.updated_resources = (hp_ratio, condition_ratio)

    import logging

    runtime = FakeRuntime(settings, frames)
    provider = LiveRestProvider(
        runtime,
        input_driver=LiveInputDriver(logger=logging.getLogger("botlab.live.test.rest"), dry_run=True),
        resource_provider=LiveResourceProvider(settings.live),
    )

    timeline = provider.apply_rest(
        cycle_id=1,
        rest_started_ts=100.0,
        starting_hp_ratio=0.40,
        starting_condition_ratio=0.45,
        observation=Observation(cycle_id=1, observed_at_ts=100.0, signal_detected=True),
    )

    assert len(timeline.snapshots) == 2
    assert timeline.snapshots[-1].snapshot.metadata["rest_decision"] == "rest_stop_threshold_reached"
    assert timeline.metadata["rest_stop_threshold_reached_count"] == 1
    assert timeline.metadata["rest_stalled_or_uncertain_count"] == 0
    assert runtime.updated_resources is not None
    assert runtime.updated_resources[0] >= 0.9
    assert runtime.updated_resources[1] >= 0.9


def test_live_rest_provider_warns_when_rest_read_is_uncertain(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    live_settings = replace(
        settings.live,
        rest_resource_sample_count=3,
        rest_resource_sample_interval_s=0.01,
        rest_resource_max_ticks=2,
        rest_resource_min_confidence=0.80,
        rest_resource_warning_spread_threshold=0.04,
    )
    settings = replace(settings, live=live_settings)
    frames = [
        _build_resource_frame(settings, hp_ratio=0.40, condition_ratio=0.40),
        _build_resource_frame(settings, hp_ratio=0.85, condition_ratio=0.88),
        _build_resource_frame(settings, hp_ratio=0.35, condition_ratio=0.42),
        _build_resource_frame(settings, hp_ratio=0.41, condition_ratio=0.41),
        _build_resource_frame(settings, hp_ratio=0.84, condition_ratio=0.87),
        _build_resource_frame(settings, hp_ratio=0.36, condition_ratio=0.43),
    ]

    class FakeRuntime:
        def __init__(self, settings_obj: Settings, frame_list: list[LiveFrame]) -> None:
            self.settings = settings_obj
            self._frames = list(frame_list)
            self.updated_resources: tuple[float, float] | None = None

        def capture_frame(self, *, cycle_id: int, phase: str, default_ts: float) -> LiveFrame:
            return self._frames.pop(0)

        def update_resources(self, *, hp_ratio: float, condition_ratio: float) -> None:
            self.updated_resources = (hp_ratio, condition_ratio)

    import logging

    runtime = FakeRuntime(settings, frames)
    provider = LiveRestProvider(
        runtime,
        input_driver=LiveInputDriver(logger=logging.getLogger("botlab.live.test.rest.uncertain"), dry_run=True),
        resource_provider=LiveResourceProvider(settings.live),
    )

    timeline = provider.apply_rest(
        cycle_id=2,
        rest_started_ts=200.0,
        starting_hp_ratio=0.30,
        starting_condition_ratio=0.30,
        observation=Observation(cycle_id=2, observed_at_ts=200.0, signal_detected=True),
    )

    assert timeline.metadata["rest_stop_threshold_reached_count"] == 0
    assert timeline.metadata["rest_stalled_or_uncertain_count"] >= 1
    assert timeline.metadata["resource_warning_count"] >= 1
    assert "resource_confidence_low" in timeline.snapshots[0].snapshot.metadata["resource_warnings"]


def test_live_input_driver_real_path_plumbing_can_be_mocked(monkeypatch) -> None:
    import logging

    logger = logging.getLogger("botlab.live.test.input")
    driver = LiveInputDriver(
        logger=logger,
        dry_run=False,
        enable_real_input=True,
        enable_real_clicks=True,
        screen_offset_xy=(100, 200),
    )
    calls: list[tuple[int, int]] = []

    def fake_perform_right_click(*, absolute_x: int, absolute_y: int) -> str:
        calls.append((absolute_x, absolute_y))
        return "real_click_sent"

    monkeypatch.setattr(driver, "_perform_right_click", fake_perform_right_click)
    driver.right_click_target(
        LiveTargetDetection(
            target_id="front-free",
            screen_x=620,
            screen_y=300,
            distance=1.5,
        )
    )

    assert calls == [(720, 500)]
    assert driver.events[-1].payload["execution_status"] == "real_click_sent"
    assert driver.events[-1].payload["absolute_x"] == 720
    assert driver.events[-1].payload["absolute_y"] == 500


def test_live_input_driver_uses_capture_bbox_from_window_guard_for_click_offset(monkeypatch) -> None:
    import logging

    logger = logging.getLogger("botlab.live.test.input.guard.offset")
    driver = LiveInputDriver(
        logger=logger,
        dry_run=False,
        enable_real_input=True,
        enable_real_clicks=True,
        screen_offset_xy=(100, 200),
        real_input_guard=lambda: (
            True,
            "window_guard_ok",
            {"capture_bbox": [1000, 300, 2280, 1020]},
        ),
    )
    calls: list[tuple[int, int]] = []

    def fake_perform_right_click(*, absolute_x: int, absolute_y: int) -> str:
        calls.append((absolute_x, absolute_y))
        return "real_click_sent"

    monkeypatch.setattr(driver, "_perform_right_click", fake_perform_right_click)
    driver.right_click_target(
        LiveTargetDetection(
            target_id="front-free",
            screen_x=620,
            screen_y=300,
            distance=1.5,
        )
    )

    assert calls == [(1620, 600)]
    assert driver.events[-1].payload["absolute_x"] == 1620
    assert driver.events[-1].payload["absolute_y"] == 600


def test_live_input_driver_real_key_path_can_be_mocked(monkeypatch) -> None:
    import logging

    logger = logging.getLogger("botlab.live.test.keys")
    driver = LiveInputDriver(
        logger=logger,
        dry_run=False,
        enable_real_input=True,
        enable_real_keys=True,
    )
    pressed_keys: list[str] = []

    def fake_perform_key_press(*, key: str) -> str:
        pressed_keys.append(key)
        return "real_key_sent"

    monkeypatch.setattr(driver, "_perform_key_press", fake_perform_key_press)

    driver.press_key("r")
    driver.press_sequence(("1", "space"))

    assert pressed_keys == ["r", "1", "space"]
    assert driver.events[0].payload["execution_status"] == "real_key_sent"
    assert driver.events[1].payload["execution_statuses"] == ["real_key_sent", "real_key_sent"]


def test_live_input_driver_blocks_real_input_when_window_guard_rejects() -> None:
    import logging

    driver = LiveInputDriver(
        logger=logging.getLogger("botlab.live.test.window_guard"),
        dry_run=False,
        enable_real_input=True,
        enable_real_clicks=True,
        enable_real_keys=True,
        real_input_guard=lambda: (
            False,
            "foreground_window_mismatch",
            {"foreground_matches": False, "configured_window_title": "Game Window"},
        ),
    )

    driver.right_click_target(
        LiveTargetDetection(
            target_id="guarded-target",
            screen_x=320,
            screen_y=240,
            distance=1.0,
        )
    )
    driver.press_key("r")

    assert driver.events[0].payload["execution_status"] == "window_guard_blocked"
    assert driver.events[0].payload["guard_reason"] == "foreground_window_mismatch"
    assert driver.events[1].payload["execution_status"] == "window_guard_blocked"


def test_live_input_driver_does_not_send_real_keys_when_disabled() -> None:
    import logging

    logger = logging.getLogger("botlab.live.test.keys.disabled")
    driver = LiveInputDriver(
        logger=logger,
        dry_run=False,
        enable_real_keys=False,
    )

    driver.press_key("r")

    assert driver.events[-1].payload["execution_status"] == "real_input_disabled"


def test_live_input_driver_blocks_real_actions_when_master_flag_is_disabled() -> None:
    import logging

    logger = logging.getLogger("botlab.live.test.real_input_gate")
    driver = LiveInputDriver(
        logger=logger,
        dry_run=False,
        enable_real_input=False,
        enable_real_clicks=True,
        enable_real_keys=True,
        screen_offset_xy=(50, 60),
    )

    driver.right_click_target(
        LiveTargetDetection(
            target_id="front-free",
            screen_x=100,
            screen_y=120,
            distance=1.5,
        )
    )
    driver.press_key("r")
    driver.press_sequence(("1", "space"))

    assert driver.events[0].payload["execution_status"] == "real_input_disabled"
    assert driver.events[1].payload["execution_status"] == "real_input_disabled"
    assert driver.events[2].payload["execution_statuses"] == [
        "real_input_disabled",
        "real_input_disabled",
    ]


def test_marker_first_pipeline_can_smoke_test_real_sample_frame_if_present(tmp_path: Path) -> None:
    settings = _build_scene_aware_live_settings(tmp_path)
    frame_path = settings.live.sample_frames_directory / "live_spot_scene_1.png"
    if not frame_path.exists():
        pytest.skip("Brak lokalnej klatki real sample frame do smoke testu.")

    output_directory = tmp_path / "perception-real-smoke"
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=output_directory,
    )

    summary = runner.analyze_frame_path(frame_path)
    result = summary.frame_results[0]

    assert result.candidate_hit_count >= 1
    assert result.timings.total_reaction_latency_ms >= 0.0
    assert (output_directory / "live_spot_scene_1_perception.json").exists() is True


def test_live_spot_scene_sidecar_is_loaded_for_real_sample_frames() -> None:
    settings = load_config("config/live_real_mvp.yaml")
    loader = PerceptionFrameLoader()

    frame = loader.load_frame(settings.live.sample_frames_directory / "live_spot_scene_1.png")
    roi = extract_named_roi(frame, roi_name="spawn_roi", live_config=settings.live)

    assert frame.source == "live_spot_scene_1"
    assert frame.metadata["reference_point_xy"] == [760, 285]
    assert frame.metadata["template_match_stride_px"] == 12
    assert roi["x"] == 80
    assert roi["y"] == 300
    assert roi["width"] == 1360
    assert roi["height"] == 640


def test_live_spot_scenes_match_expected_perception_contracts(tmp_path: Path) -> None:
    settings = _build_scene_aware_live_settings(tmp_path)
    loader = PerceptionFrameLoader()
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=tmp_path / "perception-expected",
    )
    frame_names = (
        "live_spot_scene_1",
        "live_spot_scene_2",
        "live_spot_scene_3",
    )

    for frame_name in frame_names:
        frame_path = settings.live.sample_frames_directory / f"{frame_name}.png"
        if not frame_path.exists():
            pytest.skip(f"Brak klatki referencyjnej {frame_name}.png")
        frame = loader.load_frame(frame_path)
        expected = frame.metadata.get("expected_perception")
        assert isinstance(expected, dict)

        summary = runner.analyze_frame_path(frame_path)
        result = summary.frame_results[0]
        assert summary.accuracy_summary is not None
        assert result.scene_name == "single_spot_scene"
        free_count = len(result.selectable_detections)
        occupied_count = len(result.occupied_detections)

        min_target_count = expected.get("min_target_count")
        if isinstance(min_target_count, int):
            assert len(result.detections) >= min_target_count, frame_name

        max_target_count = expected.get("max_target_count")
        if isinstance(max_target_count, int):
            assert len(result.detections) <= max_target_count, frame_name

        min_free_target_count = expected.get("min_free_target_count")
        if isinstance(min_free_target_count, int):
            assert free_count >= min_free_target_count, frame_name

        max_free_target_count = expected.get("max_free_target_count")
        if isinstance(max_free_target_count, int):
            assert free_count <= max_free_target_count, frame_name

        min_occupied_target_count = expected.get("min_occupied_target_count")
        if isinstance(min_occupied_target_count, int):
            assert occupied_count >= min_occupied_target_count, frame_name

        max_occupied_target_count = expected.get("max_occupied_target_count")
        if isinstance(max_occupied_target_count, int):
            assert occupied_count <= max_occupied_target_count, frame_name

        selected_target_required = expected.get("selected_target_required")
        if selected_target_required is True:
            assert result.selected_target is not None, frame_name
        elif selected_target_required is False:
            assert result.selected_target is None, frame_name

        selected_target_must_be_free = expected.get("selected_target_must_be_free")
        if selected_target_must_be_free is True:
            assert result.selected_target is not None, frame_name
            assert result.selected_target.occupied is False, frame_name

        selected_target_screen_xy = expected.get("selected_target_screen_xy")
        selected_target_max_error_px = expected.get("selected_target_max_error_px", 48)
        if (
            isinstance(selected_target_screen_xy, list)
            and len(selected_target_screen_xy) == 2
            and all(isinstance(item, (int, float)) for item in selected_target_screen_xy)
        ):
            assert result.selected_target is not None, frame_name
            delta = _euclidean_distance(
                left_xy=(float(result.selected_target.screen_x), float(result.selected_target.screen_y)),
                right_xy=(float(selected_target_screen_xy[0]), float(selected_target_screen_xy[1])),
            )
            assert delta <= float(selected_target_max_error_px), frame_name

        occupied_target_screen_xy = expected.get("occupied_target_screen_xy")
        occupied_target_max_error_px = expected.get("occupied_target_max_error_px", 56)
        if isinstance(occupied_target_screen_xy, list):
            actual_occupied_points = [
                (float(target.screen_x), float(target.screen_y))
                for target in result.occupied_detections
            ]
            if len(occupied_target_screen_xy) == 0:
                assert len(actual_occupied_points) == 0, frame_name
            else:
                for expected_xy in occupied_target_screen_xy:
                    assert isinstance(expected_xy, list), frame_name
                    assert len(expected_xy) == 2, frame_name
                    assert any(
                        _euclidean_distance(
                            left_xy=(float(expected_xy[0]), float(expected_xy[1])),
                            right_xy=actual_xy,
                        ) <= float(occupied_target_max_error_px)
                        for actual_xy in actual_occupied_points
                    ), frame_name

        expected_candidates = expected.get("expected_candidates")
        if isinstance(expected_candidates, list):
            for candidate_expectation in expected_candidates:
                assert isinstance(candidate_expectation, dict), frame_name
                raw_screen_xy = candidate_expectation.get("screen_xy")
                assert isinstance(raw_screen_xy, list), frame_name
                assert len(raw_screen_xy) == 2, frame_name
                assert all(isinstance(item, (int, float)) for item in raw_screen_xy), frame_name
                max_error_px = float(candidate_expectation.get("max_error_px", 48))

                matched_target, matched_distance = _find_closest_detection(
                    detections=result.detections,
                    expected_xy=(float(raw_screen_xy[0]), float(raw_screen_xy[1])),
                )
                assert matched_target is not None, frame_name
                assert matched_distance <= max_error_px, frame_name

                expected_occupied = candidate_expectation.get("occupied")
                if isinstance(expected_occupied, bool):
                    assert matched_target.occupied is expected_occupied, frame_name

                expected_selected = candidate_expectation.get("selected")
                if isinstance(expected_selected, bool):
                    if expected_selected:
                        assert result.selected_target is not None, frame_name
                        assert matched_target.target_id == result.selected_target.target_id, frame_name
                    elif result.selected_target is not None:
                        assert matched_target.target_id != result.selected_target.target_id, frame_name


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


def test_live_runner_cycle_report_contains_rest_resource_metadata(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    runner = LiveRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=100.0,
        initial_anchor_cycle_id=0,
        enable_console=False,
    )

    report = runner.run_cycles(1)

    assert len(report.cycle_records) == 1
    metadata = report.cycle_records[0]["metadata"]
    assert metadata["resource_sample_count"] >= 1
    assert "resource_confidence" in metadata
    assert "resource_warning_count" in metadata


def test_live_engage_session_summary_aggregates_outcomes_and_latencies() -> None:
    summary = LiveEngageSessionSummary.from_results(
        (
            LiveEngageOutcomeResultFactory.build(
                cycle_id=1,
                outcome=LiveEngageOutcome.ENGAGED,
                detection_latency_ms=10.0,
                selection_latency_ms=3.0,
                total_reaction_latency_ms=18.0,
                verification_latency_ms=120.0,
            ),
            LiveEngageOutcomeResultFactory.build(
                cycle_id=2,
                outcome=LiveEngageOutcome.TARGET_STOLEN,
                detection_latency_ms=12.0,
                selection_latency_ms=4.0,
                total_reaction_latency_ms=20.0,
                verification_latency_ms=160.0,
            ),
        )
    )

    assert summary.total_attempts == 2
    assert summary.engaged_count == 1
    assert summary.target_stolen_count == 1
    assert summary.total_reaction_latency.avg_ms == pytest.approx(19.0)
    assert summary.verification_latency.max_ms == pytest.approx(160.0)


def test_classify_engage_outcome_marks_target_stolen_when_verify_target_is_occupied() -> None:
    world = _build_test_world_snapshot()
    decision = RetargetDecision(
        current_target_id=None,
        selected_target=TargetCandidate(
            group_id="target-a",
            score=1.0,
            reason="nearest_free_target",
            reachable=True,
            engaged_by_other=False,
            distance=1.5,
        ),
        validation=TargetValidationResult(
            group_id="target-a",
            status=TargetValidationStatus.VALID,
            reason="target_still_available",
            can_continue=True,
        ),
        changed=True,
        reason="selected_new_target",
    )
    engagement = TargetEngagementResult(
        cycle_id=1,
        target_resolution=TargetResolution(
            cycle_id=1,
            current_target_id=None,
            selected_target_id="target-a",
            world_snapshot=world,
            decision=decision,
        ),
        approach_result=TargetApproachResult(
            cycle_id=1,
            target_id="target-a",
            started_at_ts=100.0,
            completed_at_ts=100.4,
            travel_s=0.4,
            arrived=True,
            reason="target_reached_in_live_adapter",
            initial_target_id="target-a",
            retargeted=False,
        ),
        interaction_result=TargetInteractionResult(
            cycle_id=1,
            target_id="target-a",
            ready=True,
            observed_at_ts=100.45,
            reason="interaction_ready",
            initial_target_id="target-a",
            retargeted=False,
        ),
    )
    verify_perception = PerceptionAnalysisRunner(
        live_config=load_config("config/live_dry_run.yaml").live,
        output_directory=Path("data") / "unused-engage-classification",
    )._analyzer.analyze_frame(
        LiveFrame(
            width=1280,
            height=720,
            captured_at_ts=100.7,
            source="engage-verify",
            metadata={
                "reference_point_xy": [640, 360],
                "template_hits": [
                    {
                        "label": "occupied_swords",
                        "x": 618,
                        "y": 252,
                        "width": 30,
                        "height": 18,
                        "confidence": 0.95,
                        "target_id": "target-a",
                    },
                    {
                        "label": "mob_a",
                        "x": 612,
                        "y": 278,
                        "width": 40,
                        "height": 52,
                        "confidence": 0.92,
                        "target_id": "target-a",
                    },
                ],
            },
        ),
        cycle_id=1,
        phase="engage_verify",
    )

    outcome, reason, metadata = classify_engage_outcome(
        engagement=engagement,
        verify_state=LiveStateSnapshot(in_combat=False, reward_visible=False, rest_available=True),
        verify_frame_metadata={},
        verify_perception=verify_perception,
        click_point_xy=(632, 304),
        target_match_max_distance_px=72,
    )

    assert outcome is LiveEngageOutcome.TARGET_STOLEN
    assert reason == "target_became_occupied_after_click"
    assert metadata["verify_target_id"] == "target-a"


def test_live_runner_dry_run_engage_attempts_produce_artifacts_and_records(tmp_path: Path) -> None:
    settings = _build_live_settings(tmp_path)
    runner = LiveRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=100.0,
        initial_anchor_cycle_id=0,
        enable_console=False,
    )

    report = runner.run_engage_attempts(1)

    assert report.summary.total_attempts == 1
    assert report.results[0].outcome is LiveEngageOutcome.ENGAGED
    assert report.results[0].artifact_paths["engage_result_json"].exists() is True
    assert report.results[0].artifact_paths["engage_overlay_svg"].exists() is True
    assert (settings.live.debug_directory / "engage" / "engage_results.jsonl").exists() is True
    assert (settings.live.debug_directory / "engage" / "engage_session_summary.json").exists() is True
    assert runner.storage.count_rows("attempts") == 1

    summary_payload = json.loads(
        (settings.live.debug_directory / "engage" / "engage_session_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_payload["right_click_action_count"] >= 1
    assert summary_payload["key_sequence_action_count"] == 0
    assert summary_payload["calibration_warning_count"] == 0


def test_live_runner_engage_quality_gate_rejects_unstable_target_before_click(
    tmp_path: Path,
) -> None:
    settings = _build_live_settings(tmp_path)
    settings = replace(
        settings,
        live=replace(
            settings.live,
            engage_min_seen_frames=2,
        ),
    )
    runner = LiveRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=100.0,
        initial_anchor_cycle_id=0,
        enable_console=False,
    )

    report = runner.run_engage_attempts(1)
    result = report.results[0]
    payload = json.loads(
        result.artifact_paths["engage_result_json"].read_text(encoding="utf-8")
    )

    assert result.outcome is LiveEngageOutcome.NO_TARGET_AVAILABLE
    assert result.reason == "engage_quality_gate_not_stable"
    assert payload["metadata"]["engage_quality_gate_rejected"] is True
    assert payload["metadata"]["target_seen_frames"] == 1
    assert all(event["action"] != "right_click_target" for event in payload["input_events"])


@pytest.mark.parametrize(
    ("dry_run_profile", "expected_outcome"),
    (
        ("engage_target_stolen", LiveEngageOutcome.TARGET_STOLEN),
        ("engage_target_stolen_noisy", LiveEngageOutcome.TARGET_STOLEN),
        ("engage_misclick", LiveEngageOutcome.MISCLICK),
        ("engage_misclick_partial", LiveEngageOutcome.MISCLICK),
        ("engage_approach_stalled", LiveEngageOutcome.APPROACH_STALLED),
        ("engage_timeout", LiveEngageOutcome.APPROACH_TIMEOUT),
    ),
)
def test_live_runner_dry_run_engage_profiles_cover_controlled_outcomes(
    tmp_path: Path,
    dry_run_profile: str,
    expected_outcome: LiveEngageOutcome,
) -> None:
    settings = _build_live_settings_with_profile(tmp_path, dry_run_profile)
    runner = LiveRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=100.0,
        initial_anchor_cycle_id=0,
        enable_console=False,
    )

    report = runner.run_engage_attempts(1)

    assert report.summary.total_attempts == 1
    assert report.results[0].outcome is expected_outcome
    assert report.results[0].artifact_paths["engage_result_json"].exists() is True
    engage_result_payload = json.loads(
        report.results[0].artifact_paths["engage_result_json"].read_text(encoding="utf-8")
    )
    assert engage_result_payload["outcome"] == expected_outcome.value


class LiveEngageOutcomeResultFactory:
    @staticmethod
    def build(
        *,
        cycle_id: int,
        outcome: LiveEngageOutcome,
        detection_latency_ms: float,
        selection_latency_ms: float,
        total_reaction_latency_ms: float,
        verification_latency_ms: float,
    ):
        from botlab.adapters.live.models import LiveEngageResult

        return LiveEngageResult(
            cycle_id=cycle_id,
            outcome=outcome,
            reason=outcome.value,
            selected_target_id="target-a",
            final_target_id="target-a",
            click_screen_xy=(620, 300),
            started_at_ts=100.0,
            completed_at_ts=100.0 + (total_reaction_latency_ms / 1000.0),
            detection_latency_ms=detection_latency_ms,
            selection_latency_ms=selection_latency_ms,
            total_reaction_latency_ms=total_reaction_latency_ms,
            verification_latency_ms=verification_latency_ms,
            metadata={},
        )


def _build_test_world_snapshot() -> WorldSnapshot:
    return WorldSnapshot(
        observed_at_ts=100.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(
            GroupSnapshot(
                group_id="target-a",
                position=Position(x=620.0, y=304.0),
                distance=1.5,
                alive_count=1,
                engaged_by_other=False,
                reachable=True,
                threat_score=0.0,
                metadata={"mob_variant": "mob_a"},
            ),
        ),
        in_combat=False,
        current_target_id=None,
        spawn_zone_visible=True,
    )
