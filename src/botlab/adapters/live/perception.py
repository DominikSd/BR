from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from botlab.adapters.live.models import LiveFrame, LiveTargetDetection
from botlab.adapters.live.scene import CalibratedSceneProfile, SceneProfile, SceneProfileLoader
from botlab.adapters.live.vision import (
    extract_named_roi,
    filter_occupied_targets,
    select_nearest_target,
)
from botlab.config import LiveConfig
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot

try:
    from PIL import Image, ImageChops, ImageOps, ImageStat
except Exception:  # pragma: no cover - optional dependency path
    Image = None
    ImageChops = None
    ImageOps = None
    ImageStat = None


Clock = Callable[[], float]


@dataclass(slots=True, frozen=True)
class TemplateHit:
    label: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    rotation_deg: int = 0
    target_id: str | None = None
    source: str = "metadata"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def center_xy(self) -> tuple[float, float]:
        return (self.x + (self.width / 2.0), self.y + (self.height / 2.0))


@dataclass(slots=True, frozen=True)
class TemplateVariant:
    label: str
    variant_name: str
    rotation_deg: int
    image: Any
    source_path: Path
    mask_image: Any | None = None

    @property
    def width(self) -> int:
        return int(self.image.size[0])

    @property
    def height(self) -> int:
        return int(self.image.size[1])


@dataclass(slots=True, frozen=True)
class TemplatePack:
    marker_variants: tuple[TemplateVariant, ...]
    mob_upper_variants: tuple[TemplateVariant, ...]
    mob_fallback_variants: tuple[TemplateVariant, ...]
    occupied_variants: tuple[TemplateVariant, ...]

    @property
    def mob_variants(self) -> tuple[TemplateVariant, ...]:
        return tuple((*self.mob_upper_variants, *self.mob_fallback_variants))


@dataclass(slots=True)
class TrackedDetectionState:
    track_id: str
    screen_x: float
    screen_y: float
    seen_frames: int
    occupied_seen_frames: int
    missed_frames: int


@dataclass(slots=True, frozen=True)
class ReactionLatency:
    frame_captured_ts: float
    detection_started_ts: float
    detection_finished_ts: float
    target_selected_ts: float
    action_ready_ts: float

    @property
    def detection_latency_ms(self) -> float:
        return max(0.0, (self.detection_finished_ts - self.detection_started_ts) * 1000.0)

    @property
    def selection_latency_ms(self) -> float:
        return max(0.0, (self.target_selected_ts - self.detection_finished_ts) * 1000.0)

    @property
    def total_reaction_latency_ms(self) -> float:
        return max(0.0, (self.action_ready_ts - self.frame_captured_ts) * 1000.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "frame_captured_ts": self.frame_captured_ts,
            "detection_started_ts": self.detection_started_ts,
            "detection_finished_ts": self.detection_finished_ts,
            "target_selected_ts": self.target_selected_ts,
            "action_ready_ts": self.action_ready_ts,
            "detection_latency_ms": self.detection_latency_ms,
            "selection_latency_ms": self.selection_latency_ms,
            "total_reaction_latency_ms": self.total_reaction_latency_ms,
        }


@dataclass(slots=True, frozen=True)
class LatencyAggregate:
    name: str
    count: int
    min_ms: float | None
    avg_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    max_ms: float | None

    @classmethod
    def from_values(cls, name: str, values: Iterable[float]) -> "LatencyAggregate":
        sorted_values = sorted(float(value) for value in values)
        if not sorted_values:
            return cls(name=name, count=0, min_ms=None, avg_ms=None, p50_ms=None, p95_ms=None, max_ms=None)
        return cls(
            name=name,
            count=len(sorted_values),
            min_ms=sorted_values[0],
            avg_ms=sum(sorted_values) / len(sorted_values),
            p50_ms=_percentile(sorted_values, 50),
            p95_ms=_percentile(sorted_values, 95),
            max_ms=sorted_values[-1],
        )

    def to_dict(self) -> dict[str, float | int | None | str]:
        return {
            "name": self.name,
            "count": self.count,
            "min_ms": self.min_ms,
            "avg_ms": self.avg_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
        }


@dataclass(slots=True, frozen=True)
class NumericAggregate:
    name: str
    count: int
    min_value: float | None
    avg_value: float | None
    p50_value: float | None
    p95_value: float | None
    max_value: float | None

    @classmethod
    def from_values(cls, name: str, values: Iterable[float]) -> "NumericAggregate":
        sorted_values = sorted(float(value) for value in values)
        if not sorted_values:
            return cls(
                name=name,
                count=0,
                min_value=None,
                avg_value=None,
                p50_value=None,
                p95_value=None,
                max_value=None,
            )
        return cls(
            name=name,
            count=len(sorted_values),
            min_value=sorted_values[0],
            avg_value=sum(sorted_values) / len(sorted_values),
            p50_value=_percentile(sorted_values, 50),
            p95_value=_percentile(sorted_values, 95),
            max_value=sorted_values[-1],
        )

    def to_dict(self) -> dict[str, float | int | None | str]:
        return {
            "name": self.name,
            "count": self.count,
            "min_value": self.min_value,
            "avg_value": self.avg_value,
            "p50_value": self.p50_value,
            "p95_value": self.p95_value,
            "max_value": self.max_value,
        }


@dataclass(slots=True, frozen=True)
class GroundTruthCandidate:
    candidate_id: str
    screen_xy: tuple[float, float]
    max_error_px: float
    occupied: bool
    selected: bool
    bbox: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "screen_xy": [self.screen_xy[0], self.screen_xy[1]],
            "max_error_px": self.max_error_px,
            "occupied": self.occupied,
            "selected": self.selected,
            "bbox": None if self.bbox is None else list(self.bbox),
        }


@dataclass(slots=True, frozen=True)
class BenchmarkFrameReport:
    frame_source: str
    pipeline_mode: str
    ground_truth_target_count: int
    predicted_target_count: int
    in_zone_target_count: int
    out_of_zone_target_count: int
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    player_false_positive_count: int
    wrong_mob_false_positive_count: int
    ui_or_environment_false_positive_count: int
    target_recall: float
    target_precision: float
    occupied_classification_accuracy: float | None
    selected_target_correct: bool
    selected_target_in_zone: bool
    expected_selected_candidate_id: str | None
    matched_ground_truth_count: int
    occupied_correct_count: int
    false_positive_target_ids: tuple[str, ...] = ()
    false_negative_candidate_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_source": self.frame_source,
            "pipeline_mode": self.pipeline_mode,
            "ground_truth_target_count": self.ground_truth_target_count,
            "predicted_target_count": self.predicted_target_count,
            "in_zone_target_count": self.in_zone_target_count,
            "out_of_zone_target_count": self.out_of_zone_target_count,
            "true_positive_count": self.true_positive_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "player_false_positive_count": self.player_false_positive_count,
            "wrong_mob_false_positive_count": self.wrong_mob_false_positive_count,
            "ui_or_environment_false_positive_count": self.ui_or_environment_false_positive_count,
            "target_recall": self.target_recall,
            "target_precision": self.target_precision,
            "occupied_classification_accuracy": self.occupied_classification_accuracy,
            "selected_target_correct": self.selected_target_correct,
            "selected_target_in_zone": self.selected_target_in_zone,
            "expected_selected_candidate_id": self.expected_selected_candidate_id,
            "matched_ground_truth_count": self.matched_ground_truth_count,
            "occupied_correct_count": self.occupied_correct_count,
            "false_positive_target_ids": list(self.false_positive_target_ids),
            "false_negative_candidate_ids": list(self.false_negative_candidate_ids),
        }


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    evaluated_frame_count: int
    strict_pixel_only: bool
    pixel_frame_count: int
    fallback_frame_count: int
    target_true_positive_count: int
    target_false_positive_count: int
    target_false_negative_count: int
    target_recall: float
    target_precision: float
    occupied_classification_accuracy: float | None
    selected_target_accuracy: float | None
    selected_target_in_zone_accuracy: float | None
    out_of_zone_rejection_count: NumericAggregate
    false_positive_reduction_after_zone_filtering: NumericAggregate
    candidate_count: NumericAggregate
    merged_count: NumericAggregate
    false_positive_count: NumericAggregate
    false_negative_count: NumericAggregate
    player_false_positive_count: NumericAggregate
    wrong_mob_false_positive_count: NumericAggregate
    ui_or_environment_false_positive_count: NumericAggregate
    frame_reports: tuple[BenchmarkFrameReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_frame_count": self.evaluated_frame_count,
            "strict_pixel_only": self.strict_pixel_only,
            "pixel_frame_count": self.pixel_frame_count,
            "fallback_frame_count": self.fallback_frame_count,
            "target_true_positive_count": self.target_true_positive_count,
            "target_false_positive_count": self.target_false_positive_count,
            "target_false_negative_count": self.target_false_negative_count,
            "target_recall": self.target_recall,
            "target_precision": self.target_precision,
            "occupied_classification_accuracy": self.occupied_classification_accuracy,
            "selected_target_accuracy": self.selected_target_accuracy,
            "selected_target_in_zone_accuracy": self.selected_target_in_zone_accuracy,
            "out_of_zone_rejection_count": self.out_of_zone_rejection_count.to_dict(),
            "false_positive_reduction_after_zone_filtering": self.false_positive_reduction_after_zone_filtering.to_dict(),
            "candidate_count": self.candidate_count.to_dict(),
            "merged_count": self.merged_count.to_dict(),
            "false_positive_count": self.false_positive_count.to_dict(),
            "false_negative_count": self.false_negative_count.to_dict(),
            "player_false_positive_count": self.player_false_positive_count.to_dict(),
            "wrong_mob_false_positive_count": self.wrong_mob_false_positive_count.to_dict(),
            "ui_or_environment_false_positive_count": self.ui_or_environment_false_positive_count.to_dict(),
            "frame_reports": [report.to_dict() for report in self.frame_reports],
        }


@dataclass(slots=True, frozen=True)
class PerceptionFrameResult:
    cycle_id: int
    phase: str
    frame_source: str
    frame_width: int
    frame_height: int
    reference_point_xy: tuple[float, float]
    roi: dict[str, Any]
    raw_hits: tuple[TemplateHit, ...]
    detections: tuple[LiveTargetDetection, ...]
    selected_target_id: str | None
    timings: ReactionLatency
    scene_name: str | None = None
    scene_zone_polygon: tuple[tuple[float, float], ...] = ()
    scene_calibration: dict[str, Any] = field(default_factory=dict)
    pipeline_mode: str = "pixel"
    expectations: dict[str, Any] = field(default_factory=dict)
    ground_truth: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def selected_target(self) -> LiveTargetDetection | None:
        if self.selected_target_id is None:
            return None
        for detection in self.detections:
            if detection.target_id == self.selected_target_id:
                return detection
        return None

    @property
    def free_detections(self) -> tuple[LiveTargetDetection, ...]:
        return filter_occupied_targets(self.detections)

    @property
    def occupied_detections(self) -> tuple[LiveTargetDetection, ...]:
        return tuple(detection for detection in self.detections if detection.occupied)

    @property
    def in_zone_detections(self) -> tuple[LiveTargetDetection, ...]:
        return tuple(
            detection
            for detection in self.detections
            if bool(detection.metadata.get("in_scene_zone", True))
        )

    @property
    def out_of_zone_detections(self) -> tuple[LiveTargetDetection, ...]:
        return tuple(
            detection
            for detection in self.detections
            if not bool(detection.metadata.get("in_scene_zone", True))
        )

    @property
    def selectable_detections(self) -> tuple[LiveTargetDetection, ...]:
        return tuple(
            detection
            for detection in self.in_zone_detections
            if not detection.occupied and detection.reachable
        )

    @property
    def candidate_hit_count(self) -> int:
        return len(self.raw_hits)

    @property
    def merged_hit_count(self) -> int:
        return len(self.detections)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "phase": self.phase,
            "frame_source": self.frame_source,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "reference_point_xy": list(self.reference_point_xy),
            "roi": self.roi,
            "scene_name": self.scene_name,
            "scene_zone_polygon": [
                [point_x, point_y] for point_x, point_y in self.scene_zone_polygon
            ],
            "scene_calibration": self.scene_calibration,
            "raw_hits": [
                {
                    "label": hit.label,
                    "x": hit.x,
                    "y": hit.y,
                    "width": hit.width,
                    "height": hit.height,
                    "confidence": hit.confidence,
                    "rotation_deg": hit.rotation_deg,
                    "target_id": hit.target_id,
                    "source": hit.source,
                    "metadata": hit.metadata,
                }
                for hit in self.raw_hits
            ],
            "detections": [
                {
                    "target_id": detection.target_id,
                    "screen_x": detection.screen_x,
                    "screen_y": detection.screen_y,
                    "distance": detection.distance,
                    "occupied": detection.occupied,
                    "mob_variant": detection.mob_variant,
                    "reachable": detection.reachable,
                    "confidence": detection.confidence,
                    "bbox": list(detection.bbox) if detection.bbox is not None else None,
                    "orientation_deg": detection.orientation_deg,
                    "metadata": detection.metadata,
                }
                for detection in self.detections
            ],
            "selected_target_id": self.selected_target_id,
            "candidate_hit_count": self.candidate_hit_count,
            "merged_hit_count": self.merged_hit_count,
            "free_target_count": len(self.free_detections),
            "in_zone_target_count": len(self.in_zone_detections),
            "out_of_zone_target_count": len(self.out_of_zone_detections),
            "selectable_target_count": len(self.selectable_detections),
            "pipeline_mode": self.pipeline_mode,
            "expectations": self.expectations,
            "ground_truth": self.ground_truth,
            "diagnostics": self.diagnostics,
            "timings": self.timings.to_dict(),
            "artifact_paths": {key: str(path) for key, path in self.artifact_paths.items()},
        }


@dataclass(slots=True, frozen=True)
class AccuracySummary:
    evaluated_frame_count: int
    behavior_match_count: int
    selected_target_match_count: int
    occupied_contract_match_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "evaluated_frame_count": self.evaluated_frame_count,
            "behavior_match_count": self.behavior_match_count,
            "selected_target_match_count": self.selected_target_match_count,
            "occupied_contract_match_count": self.occupied_contract_match_count,
        }


@dataclass(slots=True, frozen=True)
class PerceptionSessionSummary:
    frame_results: tuple[PerceptionFrameResult, ...]
    candidate_hits: LatencyAggregate
    merged_hits: LatencyAggregate
    free_targets: LatencyAggregate
    out_of_zone_rejections: LatencyAggregate
    detection_latency: LatencyAggregate
    selection_latency: LatencyAggregate
    total_reaction_latency: LatencyAggregate
    strict_pixel_only: bool = False
    accuracy_summary: AccuracySummary | None = None
    benchmark_summary: BenchmarkSummary | None = None
    tuning_parameters: dict[str, Any] = field(default_factory=dict)
    worst_frames: tuple[dict[str, Any], ...] = ()

    def real_scene_regression_entries(self) -> tuple[dict[str, Any], ...]:
        entries: list[dict[str, Any]] = []
        for result in self.frame_results:
            if not str(result.frame_source).startswith("live_spot_scene_"):
                continue
            selected_target = result.selected_target
            entries.append(
                {
                    "frame_source": result.frame_source,
                    "target_count": len(result.detections),
                    "free_target_count": len(result.selectable_detections),
                    "occupied_target_count": len(result.occupied_detections),
                    "in_zone_target_count": len(result.in_zone_detections),
                    "out_of_zone_target_count": len(result.out_of_zone_detections),
                    "selected_target_id": result.selected_target_id,
                    "selected_target_in_zone": None
                    if selected_target is None
                    else bool(selected_target.metadata.get("in_scene_zone", True)),
                    "selected_target_xy": None
                    if selected_target is None
                    else [selected_target.screen_x, selected_target.screen_y],
                    "occupied_target_xy": [
                        [target.screen_x, target.screen_y] for target in result.occupied_detections
                    ],
                    "detection_latency_ms": result.timings.detection_latency_ms,
                    "selection_latency_ms": result.timings.selection_latency_ms,
                    "total_reaction_latency_ms": result.timings.total_reaction_latency_ms,
                }
            )
        return tuple(entries)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "frame_count": len(self.frame_results),
            "candidate_hits": self.candidate_hits.to_dict(),
            "merged_hits": self.merged_hits.to_dict(),
            "free_targets": self.free_targets.to_dict(),
            "out_of_zone_rejections": self.out_of_zone_rejections.to_dict(),
            "detection_latency": self.detection_latency.to_dict(),
            "selection_latency": self.selection_latency.to_dict(),
            "total_reaction_latency": self.total_reaction_latency.to_dict(),
            "strict_pixel_only": self.strict_pixel_only,
            "tuning_parameters": self.tuning_parameters,
            "worst_frames": list(self.worst_frames),
            "frames": [result.to_dict() for result in self.frame_results],
        }
        if self.accuracy_summary is not None:
            payload["accuracy_summary"] = self.accuracy_summary.to_dict()
        if self.benchmark_summary is not None:
            payload["benchmark_summary"] = self.benchmark_summary.to_dict()
        real_scene_regression = self.real_scene_regression_entries()
        if real_scene_regression:
            payload["real_scene_regression"] = list(real_scene_regression)
        return payload


class TemplatePackLoader:
    def __init__(self, live_config: LiveConfig) -> None:
        self._live_config = live_config
        self._cached_pack: TemplatePack | None = None

    def load(self) -> TemplatePack:
        if self._cached_pack is not None:
            return self._cached_pack
        if Image is None or ImageOps is None:
            raise RuntimeError("Pixel-based perception wymaga Pillow.")
        marker_variants: list[TemplateVariant] = []
        markers_root = self._live_config.mobs_template_directory.parent / "markers"
        marker_label = (
            "yellow_marker"
            if self._live_config.marker_color_mode == "yellow"
            else "red_marker"
        )
        if markers_root.exists():
            for template_path in sorted(markers_root.glob("*.png")):
                base_image = Image.open(template_path).convert("RGB")
                marker_variants.extend(
                    _build_template_variants(
                        label=marker_label,
                        template_path=template_path,
                        image=base_image,
                        rotations_deg=(0,),
                    )
                )
        upper_variants: list[TemplateVariant] = []
        fallback_variants: list[TemplateVariant] = []
        mobs_root = self._live_config.mobs_template_directory
        if mobs_root.exists():
            for label_directory in sorted(path for path in mobs_root.iterdir() if path.is_dir()):
                label = label_directory.name
                template_paths = sorted(
                    path
                    for path in label_directory.glob("*.png")
                    if _is_active_mob_template_path(path)
                )
                upper_paths = [
                    path for path in template_paths if "upper" in path.stem.lower()
                ]
                body_paths = [
                    path for path in template_paths if path not in upper_paths
                ]
                for template_path in upper_paths:
                    base_image = Image.open(template_path).convert("RGB")
                    upper_variants.extend(
                        _build_template_variants(
                            label=label,
                            template_path=template_path,
                            image=base_image,
                            rotations_deg=self._live_config.template_rotations_deg,
                        )
                    )
                for template_path in body_paths:
                    base_image = Image.open(template_path).convert("RGB")
                    fallback_variants.extend(
                        _build_template_variants(
                            label=label,
                            template_path=template_path,
                            image=base_image,
                            rotations_deg=self._live_config.template_rotations_deg,
                        )
                    )
        occupied_variants: list[TemplateVariant] = []
        occupied_root = self._live_config.occupied_template_directory
        if occupied_root.exists():
            for template_path in sorted(occupied_root.glob("*.png")):
                base_image = Image.open(template_path).convert("RGB")
                occupied_variants.extend(
                    _build_template_variants(
                        label="occupied_swords",
                        template_path=template_path,
                        image=base_image,
                        rotations_deg=(0,),
                    )
                )
        self._cached_pack = TemplatePack(
            marker_variants=tuple(marker_variants),
            mob_upper_variants=tuple(upper_variants),
            mob_fallback_variants=tuple(fallback_variants),
            occupied_variants=tuple(occupied_variants),
        )
        return self._cached_pack


class PerceptionAnalyzer:
    def __init__(
        self,
        live_config: LiveConfig,
        *,
        clock: Clock | None = None,
        strict_pixel_only: bool = False,
    ) -> None:
        self._live_config = live_config
        self._clock = clock or time.perf_counter
        self._template_pack_loader = TemplatePackLoader(live_config)
        self._scene_profile_loader = SceneProfileLoader(live_config.scene_profile_path)
        self._track_states: dict[str, TrackedDetectionState] = {}
        self._track_sequence = 0
        self._strict_pixel_only = strict_pixel_only
        self._zero_seed_streak = 0
        self._last_selected_target_id: str | None = None
        self._last_selected_screen_xy: tuple[float, float] | None = None

    def analyze_frame(
        self,
        frame: LiveFrame,
        *,
        cycle_id: int,
        phase: str,
    ) -> PerceptionFrameResult:
        scene_profile = self._scene_profile_loader.load()
        calibrated_scene_profile = self._calibrate_scene_profile(
            scene_profile=scene_profile,
            frame=frame,
        )
        roi = self._resolve_spawn_roi(
            frame=frame,
            calibrated_scene_profile=calibrated_scene_profile,
        )
        reference_point_xy = _resolve_reference_point(
            frame,
            scene_profile=calibrated_scene_profile,
        )

        detection_started_ts = frame.captured_at_ts
        detection_perf_started = self._clock()
        pipeline_mode = "pixel"
        player_veto_rejections: tuple[dict[str, Any], ...] = ()
        mob_signature_rejections: tuple[dict[str, Any], ...] = ()
        seed_diagnostics: dict[str, Any] = {}
        if frame.image is not None:
            (
                roi_hits,
                detections,
                player_veto_rejections,
                mob_signature_rejections,
                seed_diagnostics,
            ) = self._run_marker_first_pipeline(
                frame=frame,
                roi=roi,
                reference_point_xy=reference_point_xy,
                phase=phase,
            )
        else:
            if self._strict_pixel_only:
                raise RuntimeError(
                    f"Strict pixel-based benchmark wymaga prawdziwego obrazu. Frame '{frame.source}' nie ma raster image."
                )
            raw_hits = self._load_metadata_template_hits(frame)
            roi_hits = self._filter_hits_to_roi(raw_hits=raw_hits, roi=roi)
            detections = self._build_detections_from_template_hits(
                hits=roi_hits,
                reference_point_xy=reference_point_xy,
            )
            pipeline_mode = "metadata_fallback"
        detections, candidate_tracks = self._smooth_detections(detections)
        detections = self._apply_scene_zone_filter(
            detections=detections,
            scene_profile=calibrated_scene_profile,
        )
        candidate_tracks = self._apply_scene_zone_filter(
            detections=candidate_tracks,
            scene_profile=calibrated_scene_profile,
        )
        detection_finished_ts = detection_started_ts + self._resolve_duration_s(
            frame=frame,
            phase_key="detection_duration_s",
            measured_duration_s=max(0.0, self._clock() - detection_perf_started),
        )

        selection_perf_started = self._clock()
        selectable_detections = tuple(
            detection
            for detection in filter_occupied_targets(detections)
            if bool(detection.metadata.get("in_scene_zone", True)) and detection.reachable
        )
        selected_target, selection_reason = self._select_target_with_stability_guard(
            detections=selectable_detections,
            frame_source=frame.source,
        )
        selection_finished_ts = detection_finished_ts + self._resolve_duration_s(
            frame=frame,
            phase_key="selection_duration_s",
            measured_duration_s=max(0.0, self._clock() - selection_perf_started),
        )
        action_ready_ts = selection_finished_ts + self._resolve_duration_s(
            frame=frame,
            phase_key="action_ready_duration_s",
            measured_duration_s=0.0,
        )
        diagnostics = self._build_detection_diagnostics(
            raw_hits=roi_hits,
            detections=detections,
            candidate_tracks=candidate_tracks,
            player_veto_rejections=player_veto_rejections,
            mob_signature_rejections=mob_signature_rejections,
            seed_diagnostics=seed_diagnostics,
            selected_target=selected_target,
            selection_reason=selection_reason,
        )

        return PerceptionFrameResult(
            cycle_id=cycle_id,
            phase=phase,
            frame_source=frame.source,
            frame_width=frame.width,
            frame_height=frame.height,
            reference_point_xy=reference_point_xy,
            roi=roi,
            raw_hits=roi_hits,
            detections=detections,
            selected_target_id=None if selected_target is None else selected_target.target_id,
            timings=ReactionLatency(
                frame_captured_ts=frame.captured_at_ts,
                detection_started_ts=detection_started_ts,
                detection_finished_ts=detection_finished_ts,
                target_selected_ts=selection_finished_ts,
                action_ready_ts=action_ready_ts,
            ),
            scene_name=None if calibrated_scene_profile is None else calibrated_scene_profile.scene_name,
            scene_zone_polygon=()
            if calibrated_scene_profile is None or not self._live_config.scene_zone_overlay_visible
            else calibrated_scene_profile.spawn_zone_polygon,
            scene_calibration={}
            if calibrated_scene_profile is None
            else calibrated_scene_profile.to_dict(),
            pipeline_mode=pipeline_mode,
            expectations=dict(frame.metadata.get("expected_perception", {})),
            ground_truth=dict(frame.metadata.get("ground_truth", {})),
            diagnostics=diagnostics,
        )

    def summarize_session(
        self,
        frame_results: Iterable[PerceptionFrameResult],
    ) -> PerceptionSessionSummary:
        results = tuple(frame_results)
        return PerceptionSessionSummary(
            frame_results=results,
            candidate_hits=LatencyAggregate.from_values(
                "candidate_hits",
                (result.candidate_hit_count for result in results),
            ),
            merged_hits=LatencyAggregate.from_values(
                "merged_hits",
                (result.merged_hit_count for result in results),
            ),
            free_targets=LatencyAggregate.from_values(
                "free_targets",
                (len(result.free_detections) for result in results),
            ),
            out_of_zone_rejections=LatencyAggregate.from_values(
                "out_of_zone_rejections",
                (len(result.out_of_zone_detections) for result in results),
            ),
            detection_latency=LatencyAggregate.from_values(
                "detection_latency_ms",
                (result.timings.detection_latency_ms for result in results),
            ),
            selection_latency=LatencyAggregate.from_values(
                "selection_latency_ms",
                (result.timings.selection_latency_ms for result in results),
            ),
            total_reaction_latency=LatencyAggregate.from_values(
                "total_reaction_latency_ms",
                (result.timings.total_reaction_latency_ms for result in results),
            ),
            strict_pixel_only=self._strict_pixel_only,
            accuracy_summary=_build_accuracy_summary(results),
            benchmark_summary=_build_benchmark_summary(
                results,
                strict_pixel_only=self._strict_pixel_only,
            ),
            tuning_parameters=_build_detection_tuning_parameters(self._live_config),
            worst_frames=_build_worst_frame_entries(results),
        )

    def _load_metadata_template_hits(self, frame: LiveFrame) -> tuple[TemplateHit, ...]:
        raw_hits = frame.metadata.get("template_hits")
        if isinstance(raw_hits, list):
            parsed_hits: list[TemplateHit] = []
            for raw_hit in raw_hits:
                if not isinstance(raw_hit, dict):
                    continue
                parsed_hits.append(
                    TemplateHit(
                        label=str(raw_hit.get("label", "unknown")),
                        x=int(raw_hit.get("x", 0)),
                        y=int(raw_hit.get("y", 0)),
                        width=int(raw_hit.get("width", 0)),
                        height=int(raw_hit.get("height", 0)),
                        confidence=float(raw_hit.get("confidence", 0.0)),
                        rotation_deg=int(raw_hit.get("rotation_deg", 0)),
                        target_id=_optional_str(raw_hit.get("target_id")),
                        source=str(raw_hit.get("source", "metadata")),
                        metadata=dict(raw_hit.get("metadata", {})),
                    )
                )
            return tuple(parsed_hits)
        return _synthesize_hits_from_targets(frame)

    def _apply_scene_zone_filter(
        self,
        *,
        detections: tuple[LiveTargetDetection, ...],
        scene_profile: CalibratedSceneProfile | None,
    ) -> tuple[LiveTargetDetection, ...]:
        if scene_profile is None:
            return detections
        filtered_detections: list[LiveTargetDetection] = []
        for detection in detections:
            point_xy = (float(detection.screen_x), float(detection.screen_y))
            in_scene_zone = scene_profile.contains_point(point_xy)
            filtered_detections.append(
                replace(
                    detection,
                    reachable=detection.reachable and in_scene_zone,
                    metadata={
                        **detection.metadata,
                        "scene_name": scene_profile.scene_name,
                        "in_scene_zone": in_scene_zone,
                        "zone_rejection_reason": None if in_scene_zone else "outside_scene_zone",
                    },
                )
            )
        return tuple(filtered_detections)

    def _calibrate_scene_profile(
        self,
        *,
        scene_profile: SceneProfile | None,
        frame: LiveFrame,
    ) -> CalibratedSceneProfile | None:
        if scene_profile is None:
            return None
        anchor_target_xy = self._resolve_scene_anchor_target(frame=frame)
        return scene_profile.calibrate(
            frame_width=frame.width,
            frame_height=frame.height,
            offset_xy=self._live_config.scene_calibration_offset_xy,
            anchor_target_xy=anchor_target_xy,
            anchor_mode=self._live_config.scene_reference_anchor_mode,
        )

    def _resolve_scene_anchor_target(self, *, frame: LiveFrame) -> tuple[float, float] | None:
        anchor_mode = self._live_config.scene_reference_anchor_mode
        if anchor_mode == "static":
            return None
        if not str(frame.source).startswith("foreground_window_capture"):
            return None
        if anchor_mode == "frame_center":
            return (frame.width / 2.0, frame.height / 2.0)
        if anchor_mode == "custom":
            return (
                float(self._live_config.scene_reference_anchor_xy[0]),
                float(self._live_config.scene_reference_anchor_xy[1]),
            )
        return None

    def _run_marker_first_pipeline(
        self,
        *,
        frame: LiveFrame,
        roi: dict[str, Any],
        reference_point_xy: tuple[float, float],
        phase: str,
    ) -> tuple[
        tuple[TemplateHit, ...],
        tuple[LiveTargetDetection, ...],
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
        dict[str, Any],
    ]:
        if Image is None or frame.image is None:
            return (), (), (), (), {}
        image = frame.image.convert("RGB")
        template_pack = self._template_pack_loader.load()
        roi_box = (
            int(roi["x"]),
            int(roi["y"]),
            int(roi["x"]) + int(roi["width"]),
            int(roi["y"]) + int(roi["height"]),
        )
        roi_image = image.crop(roi_box)
        stride_px = self._resolve_match_stride_px(frame)

        marker_hits, marker_seed_diagnostics = _detect_marker_hits_with_diagnostics(
            roi_image=roi_image,
            roi_offset_xy=(roi_box[0], roi_box[1]),
            live_config=self._live_config,
        )
        filtered_template_marker_hits: tuple[TemplateHit, ...] = ()
        use_template_marker_fallback = (
            not marker_hits
            and str(frame.source).startswith("foreground_window_capture")
        )
        if use_template_marker_fallback:
            template_marker_hits = _match_template_variants(
                roi_image=roi_image,
                roi_offset_xy=(roi_box[0], roi_box[1]),
                variants=template_pack.marker_variants,
                confidence_threshold=max(0.45, self._live_config.marker_confidence_threshold - 0.03),
                stride_px=max(1, stride_px),
            )
            filtered_template_marker_hits = tuple(
                hit
                for hit in template_marker_hits
                if _marker_template_hit_is_valid(
                    image=image,
                    hit=hit,
                    live_config=self._live_config,
                )
            )
        rescue_seed_hits: tuple[TemplateHit, ...] = ()
        should_run_upper_rescue = self._should_run_upper_rescue(
            frame_source=frame.source,
            phase=phase,
            marker_hits=marker_hits,
            template_marker_hits=filtered_template_marker_hits,
        )
        if should_run_upper_rescue:
            rescue_seed_hits = _seed_from_upper_rescue_scan(
                image=image,
                roi_box=roi_box,
                template_pack=template_pack,
                live_config=self._live_config,
                stride_px=stride_px,
            )
        all_marker_hits = tuple((*marker_hits, *filtered_template_marker_hits, *rescue_seed_hits))
        merged_marker_hits = merge_template_hits(
            all_marker_hits,
            merge_distance_px=self._live_config.merge_distance_px,
        )
        limited_marker_hits = _limit_seed_hits_for_confirmation(
            hits=merged_marker_hits,
            reference_point_xy=reference_point_xy,
            max_seed_hits=self._live_config.max_seed_hits_for_confirmation,
        )

        raw_hits: list[TemplateHit] = list(all_marker_hits)
        seed_mode = "marker"
        if not marker_hits and filtered_template_marker_hits:
            seed_mode = "marker_template_fallback"
        elif not marker_hits and not filtered_template_marker_hits and rescue_seed_hits:
            seed_mode = "upper_rescue"
        elif not all_marker_hits:
            seed_mode = "none"
        seed_diagnostics = {
            **marker_seed_diagnostics,
            "marker_hit_count": len(marker_hits),
            "marker_template_hit_count": len(filtered_template_marker_hits),
            "rescue_seed_hit_count": len(rescue_seed_hits),
            "merged_seed_hit_count": len(merged_marker_hits),
            "limited_seed_hit_count": len(limited_marker_hits),
            "seed_limit_applied": len(limited_marker_hits) < len(merged_marker_hits),
            "seed_mode": seed_mode,
            "zero_seed_streak": self._zero_seed_streak,
            "upper_rescue_attempted": should_run_upper_rescue,
        }
        detections: list[LiveTargetDetection] = []
        player_veto_rejections: list[dict[str, Any]] = []
        mob_signature_rejections: list[dict[str, Any]] = []
        local_stride_px = max(1, self._live_config.confirmation_template_stride_px)
        skip_fallback_confirmation = (
            phase == "preview"
            and self._live_config.preview_fast_mode
            and self._live_config.preview_skip_fallback_confirmation
        )
        for index, marker_hit in enumerate(limited_marker_hits, start=1):
            confirmation_roi_box = _build_local_roi_box(
                anchor_x=marker_hit.x + (marker_hit.width / 2.0),
                anchor_y=marker_hit.y + marker_hit.height,
                width=self._live_config.confirmation_roi_width_px,
                height=self._live_config.confirmation_roi_height_px,
                offset_y=self._live_config.confirmation_roi_offset_y_px,
                frame_width=frame.width,
                frame_height=frame.height,
            )
            upper_confirmation_hits: tuple[TemplateHit, ...] = ()
            confirmation_search_mode = "sliding"
            if self._live_config.confirmation_anchor_search_enabled:
                upper_confirmation_hits = _match_anchor_aligned_variants(
                    image=image,
                    marker_hit=marker_hit,
                    box=confirmation_roi_box,
                    variants=template_pack.mob_upper_variants,
                    confidence_threshold=self._live_config.confirmation_confidence_threshold,
                    stride_px=local_stride_px,
                )
                if upper_confirmation_hits:
                    confirmation_search_mode = "anchor"
                elif not self._live_config.confirmation_anchor_only:
                    upper_confirmation_hits = _match_local_variants(
                        image=image,
                        box=confirmation_roi_box,
                        variants=template_pack.mob_upper_variants,
                        confidence_threshold=self._live_config.confirmation_confidence_threshold,
                        stride_px=local_stride_px,
                    )
                    confirmation_search_mode = "anchor_then_sliding"
                else:
                    confirmation_search_mode = "anchor_only"
            else:
                upper_confirmation_hits = _match_local_variants(
                    image=image,
                    box=confirmation_roi_box,
                    variants=template_pack.mob_upper_variants,
                    confidence_threshold=self._live_config.confirmation_confidence_threshold,
                    stride_px=local_stride_px,
                )
            raw_hits.extend(upper_confirmation_hits)
            best_upper_confirmation = _select_best_confirmation_hit(
                marker_hit=marker_hit,
                confirmation_hits=upper_confirmation_hits,
                image=image,
                live_config=self._live_config,
            )
            fallback_confirmation_hits: tuple[TemplateHit, ...] = ()
            best_fallback_confirmation: TemplateHit | None = None
            if (
                best_upper_confirmation is None
                and template_pack.mob_fallback_variants
                and not skip_fallback_confirmation
                and self._live_config.enable_fallback_confirmation
            ):
                fallback_confirmation_hits = _match_local_variants(
                    image=image,
                    box=confirmation_roi_box,
                    variants=template_pack.mob_fallback_variants,
                    confidence_threshold=self._live_config.confirmation_confidence_threshold,
                    stride_px=local_stride_px,
                )
                raw_hits.extend(fallback_confirmation_hits)
                best_fallback_confirmation = _select_best_confirmation_hit(
                    marker_hit=marker_hit,
                    confirmation_hits=fallback_confirmation_hits,
                    image=image,
                    live_config=self._live_config,
                )

            best_confirmation = best_upper_confirmation or best_fallback_confirmation
            if best_confirmation is None:
                continue
            confirmation_stage = "upper" if best_upper_confirmation is not None else "fallback"
            upper_score = 0.0 if best_upper_confirmation is None else float(best_upper_confirmation.confidence)
            fallback_score = 0.0 if best_fallback_confirmation is None else float(best_fallback_confirmation.confidence)

            ice_signature = _detect_ice_mob_signature(
                image=image,
                bbox=best_confirmation.bbox,
                live_config=self._live_config,
            )
            if self._live_config.ice_mob_signature_enabled and not bool(ice_signature.get("accepted", False)):
                mob_signature_rejections.append(
                    {
                        "target_id": f"mob-signature-{index:03d}",
                        "marker_bbox": list(marker_hit.bbox),
                        "confirmation_bbox": list(best_confirmation.bbox),
                        "mob_variant": best_confirmation.label,
                        "confirmation_confidence": best_confirmation.confidence,
                        "rejection_reason": "mob_signature_not_icy",
                        "ice_pixel_count": int(ice_signature.get("ice_pixel_count", 0)),
                        "ice_pixel_ratio": float(ice_signature.get("ice_pixel_ratio", 0.0)),
                        "dark_ratio": float(ice_signature.get("dark_ratio", 0.0)),
                        "brown_ratio": float(ice_signature.get("brown_ratio", 0.0)),
                        "marker_score": marker_hit.confidence,
                        "upper_template_score": upper_score,
                        "fallback_template_score": fallback_score,
                        "ice_score": float(ice_signature.get("score", 0.0)),
                        "confirmation_stage": confirmation_stage,
                    }
                )
                continue

            player_veto_roi_box = _build_local_roi_box(
                anchor_x=best_confirmation.center_xy[0],
                anchor_y=best_confirmation.bbox[1],
                width=self._live_config.player_veto_roi_width_px,
                height=self._live_config.player_veto_roi_height_px,
                offset_y=self._live_config.player_veto_roi_offset_y_px,
                frame_width=frame.width,
                frame_height=frame.height,
            )
            player_veto = _detect_player_veto(
                image=image,
                box=player_veto_roi_box,
                live_config=self._live_config,
            )
            if bool(player_veto.get("triggered", False)):
                player_veto_rejections.append(
                    {
                        "target_id": f"player-veto-{index:03d}",
                        "marker_bbox": list(marker_hit.bbox),
                        "confirmation_bbox": list(best_confirmation.bbox),
                        "player_veto_roi": list(player_veto_roi_box),
                        "mob_variant": best_confirmation.label,
                        "confirmation_confidence": best_confirmation.confidence,
                        "rejection_reason": "player_veto_green_name",
                        "green_pixel_count": int(player_veto.get("green_pixel_count", 0)),
                        "green_pixel_ratio": float(player_veto.get("green_pixel_ratio", 0.0)),
                        "green_bbox": player_veto.get("green_bbox"),
                        "marker_score": marker_hit.confidence,
                        "upper_template_score": upper_score,
                        "fallback_template_score": fallback_score,
                        "ice_score": float(ice_signature.get("score", 0.0)),
                        "player_veto_score": float(player_veto.get("score", 0.0)),
                        "confirmation_stage": confirmation_stage,
                    }
                )
                continue

            occupied_roi_box = _build_local_roi_box(
                anchor_x=best_confirmation.center_xy[0],
                anchor_y=best_confirmation.bbox[1],
                width=self._live_config.occupied_local_roi_width_px,
                height=self._live_config.occupied_local_roi_height_px,
                offset_y=self._live_config.occupied_local_roi_offset_y_px,
                frame_width=frame.width,
                frame_height=frame.height,
            )
            occupied_color_hits = _detect_green_swords_hits(
                image=image,
                box=occupied_roi_box,
                live_config=self._live_config,
            )
            occupied_green_ratio = _estimate_green_pixel_ratio(
                image=image,
                box=occupied_roi_box,
                live_config=self._live_config,
            )
            should_run_occupied_template_match = bool(occupied_color_hits) or (
                occupied_green_ratio >= self._live_config.occupied_template_match_min_green_ratio
            )
            occupied_hits = ()
            if should_run_occupied_template_match:
                occupied_hits = _match_local_variants(
                    image=image,
                    box=occupied_roi_box,
                    variants=template_pack.occupied_variants,
                    confidence_threshold=self._live_config.occupied_confidence_threshold,
                    stride_px=local_stride_px,
                )
            all_occupied_hits = tuple((*occupied_hits, *occupied_color_hits))
            raw_hits.extend(all_occupied_hits)
            occupied_candidate, occupied_confidence = classify_occupied(
                mob_hit=best_confirmation,
                occupied_hits=all_occupied_hits,
                minimum_confidence=max(
                    0.35,
                    self._live_config.occupied_confidence_threshold * 0.75,
                ),
            )
            center_x, center_y = best_confirmation.center_xy
            distance = math.dist(reference_point_xy, (center_x, center_y))
            detection_confidence = (
                marker_hit.confidence * 0.45
                + best_confirmation.confidence * 0.55
            )
            detections.append(
                LiveTargetDetection(
                    target_id=f"{best_confirmation.label}-marker-{index:03d}",
                    screen_x=int(round(center_x)),
                    screen_y=int(round(center_y)),
                    distance=distance,
                    occupied=False,
                    mob_variant=best_confirmation.label,
                    reachable=True,
                    confidence=max(marker_hit.confidence, detection_confidence),
                    bbox=best_confirmation.bbox,
                    orientation_deg=best_confirmation.rotation_deg,
                    metadata={
                        "detection_pipeline": "marker_first",
                        "detection_state": "watcher",
                        "marker_bbox": list(marker_hit.bbox),
                        "marker_confidence": marker_hit.confidence,
                        "marker_pixel_count": int(marker_hit.metadata.get("pixel_count", 0)),
                        "marker_score": marker_hit.confidence,
                        "marker_source": marker_hit.source,
                        "upper_template_score": upper_score,
                        "fallback_template_score": fallback_score,
                        "confirmation_stage": confirmation_stage,
                        "confirmation_search_mode": confirmation_search_mode,
                        "confirmation_selected_score": best_confirmation.confidence,
                        "occupied_confidence": occupied_confidence,
                        "occupied_candidate": occupied_candidate,
                        "occupied_score": occupied_confidence,
                        "occupied_green_ratio": occupied_green_ratio,
                        "occupied_template_match_enabled": should_run_occupied_template_match,
                        "occupied_roi": list(occupied_roi_box),
                        "confirmation_roi": list(confirmation_roi_box),
                        "confirmation_confidence": best_confirmation.confidence,
                        "confirmation_alignment_score": float(
                            best_confirmation.metadata.get("alignment_score", 0.0)
                        ),
                        "confirmation_foreground_score": float(
                            best_confirmation.metadata.get("foreground_score", 0.0)
                        ),
                        "confirmation_horizontal_gap_px": float(
                            best_confirmation.metadata.get("horizontal_gap_px", 0.0)
                        ),
                        "confirmation_vertical_gap_px": float(
                            best_confirmation.metadata.get("vertical_gap_px", 0.0)
                        ),
                        "player_veto_roi": list(player_veto_roi_box),
                        "player_veto_triggered": False,
                        "player_veto_score": float(player_veto.get("score", 0.0)),
                        "confirmation_template": best_confirmation.metadata.get("template_path"),
                        "variant_name": best_confirmation.metadata.get("variant_name"),
                        "raw_hit_count": int(best_confirmation.metadata.get("raw_hit_count", 1)),
                        "ice_signature_pixel_count": int(ice_signature.get("ice_pixel_count", 0)),
                        "ice_signature_ratio": float(ice_signature.get("ice_pixel_ratio", 0.0)),
                        "ice_signature_dark_ratio": float(ice_signature.get("dark_ratio", 0.0)),
                        "ice_signature_brown_ratio": float(ice_signature.get("brown_ratio", 0.0)),
                        "ice_score": float(ice_signature.get("score", 0.0)),
                        "ice_signature_passed": bool(ice_signature.get("accepted", False)),
                    },
                )
            )
        ordered_detections = tuple(sorted(detections, key=lambda item: (item.distance, item.target_id)))
        return (
            tuple(raw_hits),
            ordered_detections,
            tuple(player_veto_rejections),
            tuple(mob_signature_rejections),
            seed_diagnostics,
        )

    def _should_run_upper_rescue(
        self,
        *,
        frame_source: str,
        phase: str,
        marker_hits: tuple[TemplateHit, ...],
        template_marker_hits: tuple[TemplateHit, ...],
    ) -> bool:
        if marker_hits or template_marker_hits:
            self._zero_seed_streak = 0
            return False
        if not str(frame_source).startswith("foreground_window_capture"):
            self._zero_seed_streak = 0
            return False
        self._zero_seed_streak += 1
        if phase == "preview":
            return self._zero_seed_streak >= 2 and (self._zero_seed_streak % 3 == 0)
        return True

    def _resolve_match_stride_px(self, frame: LiveFrame) -> int:
        override = frame.metadata.get("template_match_stride_px")
        if isinstance(override, int) and override > 0:
            return override
        return self._live_config.template_match_stride_px

    def _filter_hits_to_roi(
        self,
        *,
        raw_hits: tuple[TemplateHit, ...],
        roi: dict[str, Any],
    ) -> tuple[TemplateHit, ...]:
        roi_x = int(roi["x"])
        roi_y = int(roi["y"])
        roi_right = roi_x + int(roi["width"])
        roi_bottom = roi_y + int(roi["height"])
        filtered_hits: list[TemplateHit] = []
        for hit in raw_hits:
            center_x, center_y = hit.center_xy
            if roi_x <= center_x <= roi_right and roi_y <= center_y <= roi_bottom:
                filtered_hits.append(hit)
        return tuple(filtered_hits)

    def _build_detections_from_template_hits(
        self,
        *,
        hits: tuple[TemplateHit, ...],
        reference_point_xy: tuple[float, float],
    ) -> tuple[LiveTargetDetection, ...]:
        mob_hits = tuple(
            hit
            for hit in hits
            if hit.label in {"mob_a", "mob_b"}
            and hit.confidence >= self._live_config.perception_confidence_threshold
        )
        occupied_hits = tuple(
            hit
            for hit in hits
            if hit.label == "occupied_swords"
            and hit.confidence >= self._live_config.occupied_confidence_threshold
        )
        merged_hits = merge_template_hits(
            mob_hits,
            merge_distance_px=self._live_config.merge_distance_px,
        )

        detections: list[LiveTargetDetection] = []
        for index, merged_hit in enumerate(merged_hits, start=1):
            center_x, center_y = merged_hit.center_xy
            occupied, occupied_confidence = classify_occupied(
                mob_hit=merged_hit,
                occupied_hits=occupied_hits,
                minimum_confidence=max(
                    0.35,
                    self._live_config.occupied_confidence_threshold * 0.75,
                ),
            )
            distance = math.dist(reference_point_xy, (center_x, center_y))
            target_id = merged_hit.target_id or f"{merged_hit.label}-{index:03d}"
            detections.append(
                LiveTargetDetection(
                    target_id=target_id,
                    screen_x=int(round(center_x)),
                    screen_y=int(round(center_y)),
                    distance=distance,
                    occupied=occupied,
                    mob_variant=merged_hit.label,
                    reachable=not bool(merged_hit.metadata.get("reachable") is False),
                    confidence=merged_hit.confidence,
                    bbox=merged_hit.bbox,
                    orientation_deg=merged_hit.rotation_deg,
                    metadata={
                        "detection_pipeline": "metadata_fallback",
                        "detection_state": "watcher",
                        "marker_score": float(merged_hit.confidence),
                        "upper_template_score": float(merged_hit.confidence),
                        "fallback_template_score": 0.0,
                        "ice_score": float(merged_hit.metadata.get("ice_score", 0.0)),
                        "player_veto_score": float(merged_hit.metadata.get("player_veto_score", 0.0)),
                        "occupied_candidate": occupied,
                        "occupied_confidence": occupied_confidence,
                        "occupied_score": occupied_confidence,
                        "raw_hit_count": int(merged_hit.metadata.get("raw_hit_count", 1)),
                        **merged_hit.metadata,
                    },
                )
            )
        return tuple(sorted(detections, key=lambda item: (item.distance, item.target_id)))

    def _smooth_detections(
        self,
        detections: tuple[LiveTargetDetection, ...],
    ) -> tuple[tuple[LiveTargetDetection, ...], tuple[LiveTargetDetection, ...]]:
        if not detections and not self._track_states:
            return (), ()
        matched_track_ids: set[str] = set()
        actionable: list[LiveTargetDetection] = []
        candidates: list[LiveTargetDetection] = []
        for detection in detections:
            track_id = self._match_track_id(detection, matched_track_ids)
            if track_id is None:
                self._track_sequence += 1
                track_id = f"track-{self._track_sequence:04d}"
                seen_frames = max(1, int(detection.metadata.get("seed_seen_frames", 1)))
                if bool(detection.metadata.get("occupied_candidate", False)):
                    occupied_seen_frames = max(
                        1,
                        int(detection.metadata.get("seed_occupied_seen_frames", 1)),
                    )
                else:
                    occupied_seen_frames = 0
            else:
                previous = self._track_states[track_id]
                seen_frames = previous.seen_frames + 1
                occupied_seen_frames = (
                    previous.occupied_seen_frames + 1
                    if bool(detection.metadata.get("occupied_candidate", False))
                    else 0
                )
            self._track_states[track_id] = TrackedDetectionState(
                track_id=track_id,
                screen_x=float(detection.screen_x),
                screen_y=float(detection.screen_y),
                seen_frames=seen_frames,
                occupied_seen_frames=occupied_seen_frames,
                missed_frames=0,
            )
            matched_track_ids.add(track_id)
            actionable_now = seen_frames >= self._live_config.candidate_confirmation_frames
            smoothed_occupied = bool(detection.metadata.get("occupied_candidate", False)) and (
                occupied_seen_frames >= self._live_config.occupied_confirmation_frames
            )
            tracked_detection = replace(
                detection,
                occupied=smoothed_occupied if actionable_now else False,
                metadata={
                    **detection.metadata,
                    "track_id": track_id,
                    "track_seen_frames": seen_frames,
                    "seen_frames": seen_frames,
                    "occupied_seen_frames": occupied_seen_frames,
                    "stable_candidate": actionable_now,
                    "actionable": actionable_now,
                    "detection_state": "actionable" if actionable_now else "candidate",
                    "rejection_reason": None if actionable_now else "candidate_not_actionable_yet",
                },
            )
            candidates.append(tracked_detection)
            if actionable_now:
                actionable.append(tracked_detection)

        stale_track_ids: list[str] = []
        for track_id, state in self._track_states.items():
            if track_id in matched_track_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames >= self._live_config.candidate_loss_frames:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            del self._track_states[track_id]
        return (
            tuple(sorted(actionable, key=lambda item: (item.distance, item.target_id))),
            tuple(sorted(candidates, key=lambda item: (item.distance, item.target_id))),
        )

    def _match_track_id(
        self,
        detection: LiveTargetDetection,
        matched_track_ids: set[str],
    ) -> str | None:
        best_track_id: str | None = None
        best_distance: float | None = None
        max_distance = float(self._live_config.merge_distance_px * 1.5)
        for track_id, state in self._track_states.items():
            if track_id in matched_track_ids:
                continue
            distance = math.dist(
                (float(detection.screen_x), float(detection.screen_y)),
                (state.screen_x, state.screen_y),
            )
            if distance > max_distance:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        return best_track_id

    def _select_target_with_stability_guard(
        self,
        *,
        detections: tuple[LiveTargetDetection, ...],
        frame_source: str,
    ) -> tuple[LiveTargetDetection | None, str | None]:
        best_target = select_nearest_target(detections)
        if best_target is None:
            self._last_selected_target_id = None
            self._last_selected_screen_xy = None
            return None, None

        selected_target = best_target
        selection_reason = "nearest_free_in_zone_reachable_target"
        previous_target_id = self._last_selected_target_id
        previous_screen_xy = self._last_selected_screen_xy
        guarded_target = self._apply_target_stability_guard(
            detections=detections,
            best_target=best_target,
            frame_source=frame_source,
        )
        if guarded_target is not None and guarded_target.target_id != best_target.target_id:
            selected_target = guarded_target
            selection_reason = "stability_guard_retained_previous_target"
        self._last_selected_target_id = selected_target.target_id
        self._last_selected_screen_xy = (
            float(selected_target.screen_x),
            float(selected_target.screen_y),
        )
        if previous_target_id is not None or previous_screen_xy is not None:
            selected_target = replace(
                selected_target,
                metadata={
                    **selected_target.metadata,
                    "selection_stability_guard_applied": selection_reason.startswith("stability_guard"),
                    "selection_previous_target_id": previous_target_id,
                    "selection_previous_screen_xy": None
                    if previous_screen_xy is None
                    else [previous_screen_xy[0], previous_screen_xy[1]],
                    "selection_reason": selection_reason,
                },
            )
        return selected_target, selection_reason

    def _apply_target_stability_guard(
        self,
        *,
        detections: tuple[LiveTargetDetection, ...],
        best_target: LiveTargetDetection,
        frame_source: str,
    ) -> LiveTargetDetection | None:
        if not self._live_config.target_stability_enabled:
            return best_target
        if self._last_selected_target_id is None and self._last_selected_screen_xy is None:
            return best_target
        if not str(frame_source).startswith("foreground_window_capture") and not str(frame_source).startswith(
            "window_content_capture_preview_bypass"
        ):
            return best_target

        previous_target = None
        if self._last_selected_target_id is not None:
            previous_target = next(
                (detection for detection in detections if detection.target_id == self._last_selected_target_id),
                None,
            )
        if previous_target is None and self._last_selected_screen_xy is not None and detections:
            previous_target = min(
                detections,
                key=lambda detection: math.dist(
                    (float(detection.screen_x), float(detection.screen_y)),
                    self._last_selected_screen_xy if self._last_selected_screen_xy is not None else (0.0, 0.0),
                ),
            )
            if self._last_selected_screen_xy is not None:
                previous_distance = math.dist(
                    (float(previous_target.screen_x), float(previous_target.screen_y)),
                    self._last_selected_screen_xy,
                )
                if previous_distance > float(self._live_config.target_stability_center_distance_px):
                    previous_target = None
        if previous_target is None:
            return best_target
        if previous_target.target_id == best_target.target_id:
            return best_target
        if (
            best_target.distance
            + float(self._live_config.target_stability_switch_distance_gain_px)
            >= previous_target.distance
            and best_target.confidence
            <= previous_target.confidence + float(self._live_config.target_stability_confidence_margin)
        ):
            return previous_target
        return best_target

    def _resolve_duration_s(
        self,
        *,
        frame: LiveFrame,
        phase_key: str,
        measured_duration_s: float,
    ) -> float:
        perception_profile = frame.metadata.get("perception_profile", {})
        if isinstance(perception_profile, dict):
            override_value = perception_profile.get(phase_key)
            if isinstance(override_value, (int, float)) and float(override_value) >= 0.0:
                return float(override_value)
        return measured_duration_s

    def _resolve_spawn_roi(
        self,
        *,
        frame: LiveFrame,
        calibrated_scene_profile: CalibratedSceneProfile | None,
    ) -> dict[str, Any]:
        if calibrated_scene_profile is not None:
            for roi_name in ("spawn_focus_roi", "spawn_roi"):
                roi_value = calibrated_scene_profile.sub_rois.get(roi_name)
                if roi_value is None:
                    continue
                return {
                    "name": roi_name,
                    "x": int(roi_value[0]),
                    "y": int(roi_value[1]),
                    "width": int(roi_value[2]),
                    "height": int(roi_value[3]),
                    "frame_width": frame.width,
                    "frame_height": frame.height,
                    "source": "scene_profile",
                }
        roi = extract_named_roi(frame, roi_name="spawn_roi", live_config=self._live_config)
        return {
            **roi,
            "source": "config",
        }

    def _build_detection_diagnostics(
        self,
        *,
        raw_hits: tuple[TemplateHit, ...],
        detections: tuple[LiveTargetDetection, ...],
        candidate_tracks: tuple[LiveTargetDetection, ...],
        player_veto_rejections: tuple[dict[str, Any], ...],
        mob_signature_rejections: tuple[dict[str, Any], ...],
        seed_diagnostics: dict[str, Any],
        selected_target: LiveTargetDetection | None,
        selection_reason: str | None,
    ) -> dict[str, Any]:
        low_confidence_hits: list[dict[str, Any]] = []
        duplicate_hits: list[dict[str, Any]] = []
        for hit in raw_hits:
            threshold = None
            if hit.label in {"mob_a", "mob_b"}:
                threshold = self._live_config.confirmation_confidence_threshold
            elif hit.label == "occupied_swords":
                threshold = self._live_config.occupied_confidence_threshold
            if threshold is not None and hit.confidence < threshold:
                low_confidence_hits.append(_template_hit_to_dict(hit, rejection_reason="low_confidence"))

        for detection in detections:
            raw_hit_count = int(detection.metadata.get("raw_hit_count", 1))
            if raw_hit_count > 1:
                duplicate_hits.append(
                    {
                        "target_id": detection.target_id,
                        "raw_hit_count": raw_hit_count,
                        "bbox": None if detection.bbox is None else list(detection.bbox),
                    }
                )

        occupied_rejections = [
            _detection_to_diagnostic_entry(detection, rejection_reason="occupied")
            for detection in detections
            if detection.occupied
        ]
        out_of_zone_rejections = [
            _detection_to_diagnostic_entry(detection, rejection_reason="out_of_zone")
            for detection in detections
            if not bool(detection.metadata.get("in_scene_zone", True))
        ]
        candidate_entries = [
            {
                "target_id": detection.target_id,
                "screen_xy": [detection.screen_x, detection.screen_y],
                "bbox": None if detection.bbox is None else list(detection.bbox),
                "confidence": detection.confidence,
                "distance": detection.distance,
                "mob_variant": detection.mob_variant,
                "seen_frames": int(detection.metadata.get("seen_frames", 0)),
                "track_seen_frames": int(detection.metadata.get("track_seen_frames", 0)),
                "detection_state": detection.metadata.get("detection_state", "candidate"),
                "marker_score": float(detection.metadata.get("marker_score", 0.0)),
                "upper_template_score": float(detection.metadata.get("upper_template_score", 0.0)),
                "fallback_template_score": float(detection.metadata.get("fallback_template_score", 0.0)),
                "ice_score": float(detection.metadata.get("ice_score", 0.0)),
                "player_veto_score": float(detection.metadata.get("player_veto_score", 0.0)),
                "occupied_score": float(detection.metadata.get("occupied_score", 0.0)),
                "rejection_reason": "candidate_not_actionable_yet",
            }
            for detection in candidate_tracks
            if not bool(detection.metadata.get("actionable", False))
        ]
        actionable_entries: list[dict[str, Any]] = []
        for detection in detections:
            reasons: list[str] = []
            if detection.occupied:
                reasons.append("occupied")
            if not bool(detection.metadata.get("in_scene_zone", True)):
                reasons.append("out_of_zone")
            if not detection.reachable:
                reasons.append("unreachable")
            if selected_target is not None and detection.target_id == selected_target.target_id:
                reasons.append("selected_nearest_free_in_zone")
            elif not reasons:
                reasons.append("candidate_not_selected_farther_than_best")
            actionable_entries.append(
                {
                    "target_id": detection.target_id,
                    "screen_xy": [detection.screen_x, detection.screen_y],
                    "confidence": detection.confidence,
                    "distance": detection.distance,
                    "occupied": detection.occupied,
                    "in_scene_zone": bool(detection.metadata.get("in_scene_zone", True)),
                    "reachable": detection.reachable,
                    "detection_state": detection.metadata.get("detection_state", "actionable"),
                    "marker_score": float(detection.metadata.get("marker_score", 0.0)),
                    "upper_template_score": float(detection.metadata.get("upper_template_score", 0.0)),
                    "fallback_template_score": float(detection.metadata.get("fallback_template_score", 0.0)),
                    "ice_score": float(detection.metadata.get("ice_score", 0.0)),
                    "player_veto_score": float(detection.metadata.get("player_veto_score", 0.0)),
                    "occupied_score": float(detection.metadata.get("occupied_score", 0.0)),
                    "track_seen_frames": int(detection.metadata.get("track_seen_frames", 0)),
                    "reasons": reasons,
                }
            )

        summary_lines = [
            "thresholds="
            f"mob>={self._live_config.confirmation_confidence_threshold:.2f} "
            f"occupied>={self._live_config.occupied_confidence_threshold:.2f} "
            f"merge_px={self._live_config.merge_distance_px} stride={self._live_config.template_match_stride_px}",
            "seed="
            f"mode={seed_diagnostics.get('seed_mode', 'unknown')} "
            f"marker={seed_diagnostics.get('marker_hit_count', 0)} "
            f"marker_template={seed_diagnostics.get('marker_template_hit_count', 0)} "
            f"upper_rescue={seed_diagnostics.get('rescue_seed_hit_count', 0)}",
            "raw="
            f"{len(raw_hits)} low_conf={len(low_confidence_hits)} "
            f"merged={len(detections)} candidates={len(candidate_entries)} "
            f"veto={len(player_veto_rejections)} not_icy={len(mob_signature_rejections)}",
            "rejects="
            f"occupied={len(occupied_rejections)} out_of_zone={len(out_of_zone_rejections)} "
            f"selected={None if selected_target is None else selected_target.target_id}",
        ]
        ladder_diagnostics = {
            "seed_stage_count": int(seed_diagnostics.get("merged_seed_hit_count", 0)),
            "marker_hit_count": int(seed_diagnostics.get("marker_hit_count", 0)),
            "marker_template_fallback_count": int(seed_diagnostics.get("marker_template_hit_count", 0)),
            "upper_rescue_count": int(seed_diagnostics.get("rescue_seed_hit_count", 0)),
            "confirmation_pass_count": len(candidate_tracks),
            "ice_signature_rejection_count": len(mob_signature_rejections),
            "player_veto_rejection_count": len(player_veto_rejections),
            "out_of_zone_rejection_count": len(out_of_zone_rejections),
            "final_detection_count": len(detections),
        }
        return {
            "tuning_parameters": _build_detection_tuning_parameters(self._live_config),
            "ladder_diagnostics": ladder_diagnostics,
            "raw_hit_summary": _count_hits_by_label(raw_hits),
            "low_confidence_hits": low_confidence_hits,
            "duplicate_merges": duplicate_hits,
            "occupied_rejections": occupied_rejections,
            "out_of_zone_rejections": out_of_zone_rejections,
            "unstable_rejections": candidate_entries,
            "candidate_tracks": candidate_entries,
            "player_veto_rejections": list(player_veto_rejections),
            "mob_signature_rejections": list(mob_signature_rejections),
            "seed_diagnostics": seed_diagnostics,
            "final_candidates": actionable_entries,
            "selection_reason": selection_reason,
            "summary_lines": summary_lines,
        }


class PerceptionArtifactWriter:
    def __init__(self, output_directory: Path) -> None:
        self._output_directory = output_directory

    def write_cycle_result(
        self,
        *,
        cycle_id: int,
        phase: str,
        frame: LiveFrame,
        result: PerceptionFrameResult,
    ) -> dict[str, Path]:
        cycle_directory = self._output_directory / f"cycle_{cycle_id:03d}"
        cycle_directory.mkdir(parents=True, exist_ok=True)
        return self._write_named_result(
            directory=cycle_directory,
            stem=phase,
            frame=frame,
            result=result,
        )

    def write_batch_result(
        self,
        *,
        frame_name: str,
        frame: LiveFrame,
        result: PerceptionFrameResult,
    ) -> dict[str, Path]:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        return self._write_named_result(
            directory=self._output_directory,
            stem=frame_name,
            frame=frame,
            result=result,
        )

    def append_jsonl(
        self,
        *,
        record_name: str,
        result: PerceptionFrameResult,
    ) -> Path:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        jsonl_path = self._output_directory / "perception_results.jsonl"
        payload = {
            "record_name": record_name,
            **result.to_dict(),
        }
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return jsonl_path

    def write_session_summary(self, summary: PerceptionSessionSummary) -> Path:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        summary_path = self._output_directory / "perception_session_summary.json"
        summary_path.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary_path

    def _write_named_result(
        self,
        *,
        directory: Path,
        stem: str,
        frame: LiveFrame,
        result: PerceptionFrameResult,
    ) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        artifact_paths: dict[str, Path] = {}
        if frame.image is not None:
            input_path = directory / f"{stem}_input.png"
            frame.image.save(input_path)
            artifact_paths["input_image"] = input_path
        analysis_json_path = directory / f"{stem}_perception.json"
        analysis_json_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_paths["perception_json"] = analysis_json_path
        overlay_svg_path = directory / f"{stem}_perception_overlay.svg"
        overlay_svg_path.write_text(
            self._build_overlay_svg(frame=frame, result=result),
            encoding="utf-8",
        )
        artifact_paths["perception_overlay_svg"] = overlay_svg_path
        crop_directory = self._write_local_crops(directory=directory, stem=stem, frame=frame, result=result)
        if crop_directory is not None:
            artifact_paths["local_crop_directory"] = crop_directory
        return artifact_paths

    def _write_local_crops(
        self,
        *,
        directory: Path,
        stem: str,
        frame: LiveFrame,
        result: PerceptionFrameResult,
    ) -> Path | None:
        if frame.image is None or Image is None:
            return None
        crop_directory = directory / f"{stem}_crops"
        crop_directory.mkdir(parents=True, exist_ok=True)

        def _save_crop(box: Any, crop_stem: str) -> None:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return
            left, top, width, height = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            if width <= 0 or height <= 0:
                return
            crop = frame.image.crop((left, top, left + width, top + height))
            crop.save(crop_directory / f"{crop_stem}.png")

        for detection in result.detections:
            detection_key = detection.target_id.replace("/", "_")
            _save_crop(detection.metadata.get("marker_bbox"), f"{detection_key}_marker")
            _save_crop(detection.metadata.get("confirmation_roi"), f"{detection_key}_confirmation_roi")
            _save_crop(detection.metadata.get("player_veto_roi"), f"{detection_key}_player_veto_roi")
            _save_crop(detection.metadata.get("occupied_roi"), f"{detection_key}_occupied_roi")

        for entry in result.diagnostics.get("candidate_tracks", []):
            target_id = str(entry.get("target_id", "candidate")).replace("/", "_")
            _save_crop(entry.get("bbox"), f"{target_id}_candidate_bbox")

        for entry in result.diagnostics.get("player_veto_rejections", []):
            target_id = str(entry.get("target_id", "player_veto")).replace("/", "_")
            _save_crop(entry.get("marker_bbox"), f"{target_id}_marker")
            _save_crop(entry.get("confirmation_bbox"), f"{target_id}_confirmation_bbox")
            _save_crop(entry.get("player_veto_roi"), f"{target_id}_player_veto_roi")

        for entry in result.diagnostics.get("mob_signature_rejections", []):
            target_id = str(entry.get("target_id", "mob_signature")).replace("/", "_")
            _save_crop(entry.get("marker_bbox"), f"{target_id}_marker")
            _save_crop(entry.get("confirmation_bbox"), f"{target_id}_confirmation_bbox")

        return crop_directory

    def _build_overlay_svg(
        self,
        *,
        frame: LiveFrame,
        result: PerceptionFrameResult,
    ) -> str:
        selected_target_id = result.selected_target_id
        reference_x, reference_y = result.reference_point_xy
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{frame.width}" height="{frame.height}" viewBox="0 0 {frame.width} {frame.height}">',
            '<rect width="100%" height="100%" fill="#111827" />',
        ]
        roi = result.roi
        lines.append(
            f'<rect x="{roi["x"]}" y="{roi["y"]}" width="{roi["width"]}" height="{roi["height"]}" fill="none" stroke="#60a5fa" stroke-width="2" />'
        )
        lines.append(
            f'<text x="{roi["x"] + 4}" y="{roi["y"] + 18}" fill="#93c5fd" font-size="14">spawn_roi</text>'
        )
        lines.append(
            f'<line x1="{reference_x - 10}" y1="{reference_y}" x2="{reference_x + 10}" y2="{reference_y}" stroke="#fde047" stroke-width="2" />'
        )
        lines.append(
            f'<line x1="{reference_x}" y1="{reference_y - 10}" x2="{reference_x}" y2="{reference_y + 10}" stroke="#fde047" stroke-width="2" />'
        )
        window_guard = frame.metadata.get("window_guard")
        if isinstance(window_guard, dict):
            lines.append(
                '<rect x="8" y="8" width="660" height="78" fill="#0f172a" fill-opacity="0.80" stroke="#334155" stroke-width="1" />'
            )
            lines.append(
                f'<text x="16" y="28" fill="#f8fafc" font-size="14">window={window_guard.get("configured_window_title")} matched={window_guard.get("matched_window_title")} foreground={window_guard.get("foreground_window_title")}</text>'
            )
            lines.append(
                f'<text x="16" y="46" fill="#f8fafc" font-size="14">foreground_matches={window_guard.get("foreground_matches")} reliable={window_guard.get("reliable")} capture_bbox={window_guard.get("capture_bbox")}</text>'
            )
            lines.append(
                f'<text x="16" y="64" fill="#f8fafc" font-size="14">block_reason={window_guard.get("block_reason")} warning={window_guard.get("warning")}</text>'
            )
        if result.scene_zone_polygon:
            polygon_points = " ".join(
                f"{point_x},{point_y}" for point_x, point_y in result.scene_zone_polygon
            )
            lines.append(
                f'<polygon points="{polygon_points}" fill="#22c55e" fill-opacity="0.08" stroke="#22c55e" stroke-width="3" />'
            )
            lines.append(
                f'<text x="{int(result.scene_zone_polygon[0][0])}" y="{max(18, int(result.scene_zone_polygon[0][1]) - 8)}" fill="#86efac" font-size="14">{result.scene_name or "scene_zone"}</text>'
            )
        for hit in result.raw_hits:
            hit_color = "#9ca3af"
            if hit.label in {"red_marker", "yellow_marker"}:
                hit_color = "#facc15" if hit.label == "yellow_marker" else "#ef4444"
            elif hit.label == "occupied_swords":
                hit_color = "#f97316"
            elif hit.label in {"mob_a", "mob_b"}:
                hit_color = "#38bdf8"
            lines.append(
                f'<rect x="{hit.x}" y="{hit.y}" width="{hit.width}" height="{hit.height}" fill="none" stroke="{hit_color}" stroke-dasharray="4 4" stroke-width="1" />'
            )
            lines.append(
                f'<text x="{hit.x}" y="{max(12, hit.y - 2)}" fill="{hit_color}" font-size="11">{hit.label} {hit.confidence:.2f}</text>'
            )
        for detection in result.detections:
            bbox = detection.bbox or (
                max(0, detection.screen_x - 18),
                max(0, detection.screen_y - 24),
                36,
                48,
            )
            selected = detection.target_id == selected_target_id
            in_scene_zone = bool(detection.metadata.get("in_scene_zone", True))
            detection_state = str(detection.metadata.get("detection_state", "actionable"))
            color = "#ef4444" if detection.occupied else "#22c55e"
            if not in_scene_zone:
                color = "#9ca3af"
            elif detection_state == "candidate":
                color = "#facc15"
            stroke_width = 4 if selected else 2
            label = "actionable_occupied" if detection.occupied else "actionable_free"
            if detection_state == "candidate":
                label = "candidate"
            if not in_scene_zone:
                label = f"{label}/out_of_zone"
            marker_bbox = detection.metadata.get("marker_bbox")
            occupied_roi = detection.metadata.get("occupied_roi")
            confirmation_roi = detection.metadata.get("confirmation_roi")
            if isinstance(marker_bbox, list) and len(marker_bbox) == 4:
                lines.append(
                    f'<rect x="{marker_bbox[0]}" y="{marker_bbox[1]}" width="{marker_bbox[2]}" height="{marker_bbox[3]}" fill="none" stroke="#facc15" stroke-width="2" />'
                )
            if isinstance(occupied_roi, list) and len(occupied_roi) == 4:
                lines.append(
                    f'<rect x="{occupied_roi[0]}" y="{occupied_roi[1]}" width="{occupied_roi[2]}" height="{occupied_roi[3]}" fill="none" stroke="#f97316" stroke-dasharray="3 3" stroke-width="1" />'
                )
            if isinstance(confirmation_roi, list) and len(confirmation_roi) == 4:
                lines.append(
                    f'<rect x="{confirmation_roi[0]}" y="{confirmation_roi[1]}" width="{confirmation_roi[2]}" height="{confirmation_roi[3]}" fill="none" stroke="#38bdf8" stroke-dasharray="3 3" stroke-width="1" />'
                )
            lines.append(
                f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}" fill="none" stroke="{color}" stroke-width="{stroke_width}" />'
            )
            lines.append(
                f'<circle cx="{detection.screen_x}" cy="{detection.screen_y}" r="4" fill="{color}" />'
            )
            lines.append(
                f'<text x="{bbox[0]}" y="{max(14, bbox[1] - 6)}" fill="#f9fafb" font-size="14">{detection.target_id} {label} conf={detection.confidence:.2f} dist={detection.distance:.1f} m={float(detection.metadata.get("marker_score", 0.0)):.2f} u={float(detection.metadata.get("upper_template_score", 0.0)):.2f} f={float(detection.metadata.get("fallback_template_score", 0.0)):.2f} ice={float(detection.metadata.get("ice_score", 0.0)):.2f} occ={float(detection.metadata.get("occupied_score", 0.0)):.2f} seen={int(detection.metadata.get("track_seen_frames", 0))}</text>'
            )
            if selected:
                lines.append(
                    f'<text x="{bbox[0]}" y="{bbox[1] + bbox[3] + 18}" fill="#fde047" font-size="14">selected</text>'
                )
        candidate_tracks = result.diagnostics.get("candidate_tracks", [])
        if isinstance(candidate_tracks, list):
            for entry in candidate_tracks:
                bbox = entry.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                lines.append(
                    f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}" fill="none" stroke="#facc15" stroke-dasharray="5 4" stroke-width="2" />'
                )
                lines.append(
                    f'<text x="{bbox[0]}" y="{max(14, bbox[1] - 4)}" fill="#fef08a" font-size="12">{entry.get("target_id")} candidate seen={entry.get("track_seen_frames")} m={float(entry.get("marker_score", 0.0)):.2f} u={float(entry.get("upper_template_score", 0.0)):.2f} f={float(entry.get("fallback_template_score", 0.0)):.2f} ice={float(entry.get("ice_score", 0.0)):.2f}</text>'
                )
        player_veto_rejections = result.diagnostics.get("player_veto_rejections", [])
        if isinstance(player_veto_rejections, list):
            for entry in player_veto_rejections:
                marker_bbox = entry.get("marker_bbox")
                confirmation_bbox = entry.get("confirmation_bbox")
                veto_roi = entry.get("player_veto_roi")
                green_bbox = entry.get("green_bbox")
                if isinstance(marker_bbox, list) and len(marker_bbox) == 4:
                    lines.append(
                        f'<rect x="{marker_bbox[0]}" y="{marker_bbox[1]}" width="{marker_bbox[2]}" height="{marker_bbox[3]}" fill="none" stroke="#eab308" stroke-width="2" />'
                    )
                if isinstance(confirmation_bbox, list) and len(confirmation_bbox) == 4:
                    lines.append(
                        f'<rect x="{confirmation_bbox[0]}" y="{confirmation_bbox[1]}" width="{confirmation_bbox[2]}" height="{confirmation_bbox[3]}" fill="none" stroke="#f43f5e" stroke-width="2" stroke-dasharray="4 3" />'
                    )
                    lines.append(
                        f'<text x="{confirmation_bbox[0]}" y="{max(14, confirmation_bbox[1] - 4)}" fill="#fecdd3" font-size="12">{entry.get("target_id")} player_veto green_px={entry.get("green_pixel_count")}</text>'
                    )
                if isinstance(veto_roi, list) and len(veto_roi) == 4:
                    lines.append(
                        f'<rect x="{veto_roi[0]}" y="{veto_roi[1]}" width="{veto_roi[2]}" height="{veto_roi[3]}" fill="none" stroke="#22c55e" stroke-width="1" stroke-dasharray="3 3" />'
                    )
                if isinstance(green_bbox, list) and len(green_bbox) == 4:
                    lines.append(
                        f'<rect x="{green_bbox[0]}" y="{green_bbox[1]}" width="{green_bbox[2]}" height="{green_bbox[3]}" fill="none" stroke="#4ade80" stroke-width="2" />'
                    )
        mob_signature_rejections = result.diagnostics.get("mob_signature_rejections", [])
        if isinstance(mob_signature_rejections, list):
            for entry in mob_signature_rejections:
                marker_bbox = entry.get("marker_bbox")
                confirmation_bbox = entry.get("confirmation_bbox")
                if isinstance(marker_bbox, list) and len(marker_bbox) == 4:
                    lines.append(
                        f'<rect x="{marker_bbox[0]}" y="{marker_bbox[1]}" width="{marker_bbox[2]}" height="{marker_bbox[3]}" fill="none" stroke="#eab308" stroke-width="2" />'
                    )
                if isinstance(confirmation_bbox, list) and len(confirmation_bbox) == 4:
                    lines.append(
                        f'<rect x="{confirmation_bbox[0]}" y="{confirmation_bbox[1]}" width="{confirmation_bbox[2]}" height="{confirmation_bbox[3]}" fill="none" stroke="#fb923c" stroke-width="2" stroke-dasharray="4 3" />'
                    )
                    lines.append(
                        f'<text x="{confirmation_bbox[0]}" y="{max(14, confirmation_bbox[1] - 4)}" fill="#fdba74" font-size="12">{entry.get("target_id")} not_icy ratio={float(entry.get("ice_pixel_ratio", 0.0)):.2f} dark={float(entry.get("dark_ratio", 0.0)):.2f} brown={float(entry.get("brown_ratio", 0.0)):.2f}</text>'
                    )
        summary_lines = result.diagnostics.get("summary_lines", [])
        if isinstance(summary_lines, list) and summary_lines:
            base_y = max(110, frame.height - (len(summary_lines) * 18) - 12)
            lines.append(
                f'<rect x="8" y="{base_y - 18}" width="760" height="{(len(summary_lines) * 18) + 24}" fill="#020617" fill-opacity="0.80" stroke="#334155" stroke-width="1" />'
            )
            for index, summary_line in enumerate(summary_lines, start=1):
                lines.append(
                    f'<text x="16" y="{base_y + (index * 16)}" fill="#f8fafc" font-size="13">{summary_line}</text>'
                )
        lines.append("</svg>")
        return "\n".join(lines)


class PerceptionFrameLoader:
    def load_frame(self, frame_path: str | Path) -> LiveFrame:
        path = Path(frame_path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            return self._load_frame_spec(path)
        return self._load_image_frame(path)

    def load_directory(self, directory: str | Path) -> tuple[tuple[str, LiveFrame], ...]:
        root = Path(directory).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Katalog z klatkami nie istnieje: {root}")
        loaded_frames: list[tuple[str, LiveFrame]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".png", ".jpg", ".jpeg"}:
                continue
            if path.suffix.lower() == ".json":
                sibling_images = (
                    path.with_suffix(".png"),
                    path.with_suffix(".jpg"),
                    path.with_suffix(".jpeg"),
                )
                if any(candidate.exists() for candidate in sibling_images):
                    continue
            if path.name == "frames.json":
                continue
            if path.name.endswith("_perception.json"):
                continue
            if path.name == "perception_session_summary.json":
                continue
            relative_stem = path.relative_to(root).with_suffix("")
            frame_name = "__".join(relative_stem.parts)
            loaded_frames.append((frame_name, self.load_frame(path)))
        return tuple(loaded_frames)

    def _load_frame_spec(self, path: Path) -> LiveFrame:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Pole 'metadata' w {path} musi byc mapa.")
        return LiveFrame(
            width=int(payload.get("width", 1280)),
            height=int(payload.get("height", 720)),
            captured_at_ts=float(payload.get("captured_at_ts", 0.0)),
            source=str(payload.get("source", path.stem)),
            metadata=metadata,
            image=None,
        )

    def _load_image_frame(self, path: Path) -> LiveFrame:
        width = 1280
        height = 720
        image = None
        if Image is not None:
            image = Image.open(path)
            width, height = image.size
        sidecar_path = path.with_suffix(".json")
        metadata: dict[str, Any] = {"frame_path": str(path)}
        captured_at_ts = 0.0
        source = path.name
        if sidecar_path.exists():
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            metadata = {"frame_path": str(path), **dict(payload.get("metadata", {}))}
            captured_at_ts = float(payload.get("captured_at_ts", 0.0))
            source = str(payload.get("source", path.name))
        return LiveFrame(
            width=width,
            height=height,
            captured_at_ts=captured_at_ts,
            source=source,
            metadata=metadata,
            image=image,
        )


class BenchmarkDatasetLoader:
    def __init__(self, *, dataset_root: Path, frame_loader: PerceptionFrameLoader) -> None:
        self._dataset_root = dataset_root
        self._frame_loader = frame_loader

    def load_split(self, split_name: str) -> tuple[tuple[str, LiveFrame], ...]:
        split_directory = (self._dataset_root / split_name).resolve()
        manifest_path = split_directory / "frames.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Brak manifestu benchmark split '{split_name}': {manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_frames = payload.get("frames", [])
        if not isinstance(raw_frames, list):
            raise ValueError(f"Pole 'frames' w {manifest_path} musi byc lista.")
        loaded_frames: list[tuple[str, LiveFrame]] = []
        for index, raw_entry in enumerate(raw_frames, start=1):
            if not isinstance(raw_entry, dict):
                raise ValueError(
                    f"Nieprawidlowy wpis frames[{index - 1}] w {manifest_path}: oczekiwano mapy."
                )
            raw_frame_path = raw_entry.get("frame_path")
            if not isinstance(raw_frame_path, str) or not raw_frame_path.strip():
                raise ValueError(
                    f"Pole 'frame_path' w frames[{index - 1}] w {manifest_path} musi byc niepustym napisem."
                )
            frame_path = (split_directory / raw_frame_path).resolve()
            frame = self._frame_loader.load_frame(frame_path)
            metadata_overlay = raw_entry.get("metadata_overlay", {})
            if metadata_overlay and not isinstance(metadata_overlay, dict):
                raise ValueError(
                    f"Pole 'metadata_overlay' w frames[{index - 1}] w {manifest_path} musi byc mapa."
                )
            merged_metadata = {
                **frame.metadata,
                **dict(metadata_overlay),
                "dataset_split": split_name,
            }
            entry_name = raw_entry.get("frame_name")
            if not isinstance(entry_name, str) or not entry_name.strip():
                entry_name = Path(raw_frame_path).stem
            loaded_frames.append(
                (
                    f"{split_name}__{entry_name}",
                    replace(frame, metadata=merged_metadata),
                )
            )
        return tuple(loaded_frames)


class PerceptionAnalysisRunner:
    def __init__(
        self,
        *,
        live_config: LiveConfig,
        output_directory: Path,
        clock: Clock | None = None,
        strict_pixel_only: bool = False,
    ) -> None:
        self._analyzer = PerceptionAnalyzer(
            live_config,
            clock=clock,
            strict_pixel_only=strict_pixel_only,
        )
        self._loader = PerceptionFrameLoader()
        self._artifact_writer = PerceptionArtifactWriter(output_directory)
        self._dataset_loader = BenchmarkDatasetLoader(
            dataset_root=live_config.benchmark_dataset_directory,
            frame_loader=self._loader,
        )

    def analyze_frame_path(self, frame_path: str | Path) -> PerceptionSessionSummary:
        frame = self._loader.load_frame(frame_path)
        frame_name = Path(frame_path).stem
        return self._analyze_entries(((frame_name, frame),))

    def analyze_directory(self, directory: str | Path) -> PerceptionSessionSummary:
        frame_entries = self._loader.load_directory(directory)
        return self._analyze_entries(frame_entries)

    def analyze_benchmark_split(self, split_name: str) -> PerceptionSessionSummary:
        frame_entries = self._dataset_loader.load_split(split_name)
        return self._analyze_entries(frame_entries)

    def _analyze_entries(
        self,
        frame_entries: Iterable[tuple[str, LiveFrame]],
    ) -> PerceptionSessionSummary:
        persisted_results: list[PerceptionFrameResult] = []
        for index, (frame_name, frame) in enumerate(frame_entries, start=1):
            result = self._analyzer.analyze_frame(frame, cycle_id=index, phase=frame_name)
            artifact_paths = self._artifact_writer.write_batch_result(
                frame_name=frame_name,
                frame=frame,
                result=result,
            )
            persisted_result = replace(result, artifact_paths=artifact_paths)
            self._artifact_writer.append_jsonl(record_name=frame_name, result=persisted_result)
            persisted_results.append(persisted_result)
        summary = self._analyzer.summarize_session(persisted_results)
        self._artifact_writer.write_session_summary(summary)
        return summary


def merge_template_hits(
    hits: tuple[TemplateHit, ...],
    *,
    merge_distance_px: int,
) -> tuple[TemplateHit, ...]:
    merged_hits: list[TemplateHit] = []
    for hit in sorted(hits, key=lambda item: item.confidence, reverse=True):
        merged_index = _find_merge_index(merged_hits, hit=hit, merge_distance_px=merge_distance_px)
        if merged_index is None:
            merged_hits.append(
                TemplateHit(
                    label=hit.label,
                    x=hit.x,
                    y=hit.y,
                    width=hit.width,
                    height=hit.height,
                    confidence=hit.confidence,
                    rotation_deg=hit.rotation_deg,
                    target_id=hit.target_id,
                    source=hit.source,
                    metadata={**hit.metadata, "raw_hit_count": 1},
                )
            )
            continue

        previous = merged_hits[merged_index]
        merged_hits[merged_index] = _merge_two_hits(previous, hit)
    return tuple(merged_hits)


def classify_occupied(
    *,
    mob_hit: TemplateHit,
    occupied_hits: tuple[TemplateHit, ...],
    minimum_confidence: float = 0.0,
) -> tuple[bool, float]:
    best_confidence = 0.0
    mob_center_x, mob_center_y = mob_hit.center_xy
    bbox_x, bbox_y, bbox_width, bbox_height = mob_hit.bbox
    left_bound = bbox_x - 12
    right_bound = bbox_x + bbox_width + 12
    top_bound = bbox_y - 48
    bottom_bound = bbox_y + min(48, bbox_height)
    for occupied_hit in occupied_hits:
        swords_x, swords_y = occupied_hit.center_xy
        if left_bound <= swords_x <= right_bound and top_bound <= swords_y <= bottom_bound:
            if abs(swords_x - mob_center_x) <= max(24.0, bbox_width * 0.75):
                if swords_y <= (mob_center_y + 6.0):
                    best_confidence = max(best_confidence, occupied_hit.confidence)
                    continue
        if left_bound <= swords_x <= right_bound:
            if swords_y < mob_center_y:
                vertical_gap = mob_center_y - swords_y
                if vertical_gap <= max(140.0, bbox_height * 2.0):
                    best_confidence = max(best_confidence, occupied_hit.confidence * 0.92)
    return (best_confidence >= minimum_confidence, best_confidence)


def build_world_snapshot_from_perception(
    *,
    cycle_id: int,
    frame: LiveFrame,
    perception_result: PerceptionFrameResult,
    current_target_id: str | None,
    phase: str,
) -> WorldSnapshot:
    groups = tuple(
        GroupSnapshot(
            group_id=detection.target_id,
            position=Position(x=float(detection.screen_x), y=float(detection.screen_y)),
            distance=detection.distance,
            alive_count=1,
            engaged_by_other=detection.occupied,
            reachable=detection.reachable,
            threat_score=0.0,
            metadata={
                "mob_variant": detection.mob_variant,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "orientation_deg": detection.orientation_deg,
                **detection.metadata,
            },
        )
        for detection in perception_result.in_zone_detections
    )
    return WorldSnapshot(
        observed_at_ts=frame.captured_at_ts,
        bot_position=Position(x=0.0, y=0.0),
        groups=groups,
        in_combat=False,
        current_target_id=current_target_id,
        spawn_zone_visible=True,
        metadata={
            "cycle_id": cycle_id,
            "phase": phase,
            "target_count": len(groups),
            "free_target_count": len(perception_result.selectable_detections),
            "occupied_target_count": len(perception_result.occupied_detections),
            "in_zone_target_count": len(perception_result.in_zone_detections),
            "out_of_zone_target_count": len(perception_result.out_of_zone_detections),
            "selected_target_id": perception_result.selected_target_id,
            "detection_latency_ms": perception_result.timings.detection_latency_ms,
            "selection_latency_ms": perception_result.timings.selection_latency_ms,
            "total_reaction_latency_ms": perception_result.timings.total_reaction_latency_ms,
        },
    )


def _detect_marker_hits(
    *,
    roi_image: Any,
    roi_offset_xy: tuple[int, int],
    live_config: LiveConfig,
) -> tuple[TemplateHit, ...]:
    hits, _ = _detect_marker_hits_with_diagnostics(
        roi_image=roi_image,
        roi_offset_xy=roi_offset_xy,
        live_config=live_config,
    )
    return hits


def _detect_marker_hits_with_diagnostics(
    *,
    roi_image: Any,
    roi_offset_xy: tuple[int, int],
    live_config: LiveConfig,
) -> tuple[tuple[TemplateHit, ...], dict[str, int]]:
    if Image is None:
        return (), {}
    rgb_image = roi_image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()
    marker_mask = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            if _is_marker_pixel(
                red=red,
                green=green,
                blue=blue,
                live_config=live_config,
            ):
                marker_mask[(y * width) + x] = 1

    hits: list[TemplateHit] = []
    visited = bytearray(width * height)
    marker_label = "yellow_marker" if live_config.marker_color_mode == "yellow" else "red_marker"
    diagnostics = {
        "color_component_count": 0,
        "size_pass_count": 0,
        "density_pass_count": 0,
        "dark_core_pass_count": 0,
        "confidence_pass_count": 0,
    }
    for y in range(height):
        for x in range(width):
            index = (y * width) + x
            if marker_mask[index] == 0 or visited[index] == 1:
                continue
            diagnostics["color_component_count"] += 1
            queue: list[tuple[int, int]] = [(x, y)]
            visited[index] = 1
            component_pixels: list[tuple[int, int]] = []
            min_x = x
            max_x = x
            min_y = y
            max_y = y
            marker_strength_sum = 0.0
            while queue:
                current_x, current_y = queue.pop()
                component_pixels.append((current_x, current_y))
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                red, green, blue = pixels[current_x, current_y]
                marker_strength_sum += _compute_marker_pixel_strength(
                    red=red,
                    green=green,
                    blue=blue,
                    live_config=live_config,
                )
                for neighbor_x, neighbor_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                    (current_x - 1, current_y - 1),
                    (current_x + 1, current_y - 1),
                    (current_x - 1, current_y + 1),
                    (current_x + 1, current_y + 1),
                ):
                    if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                        continue
                    neighbor_index = (neighbor_y * width) + neighbor_x
                    if marker_mask[neighbor_index] == 0 or visited[neighbor_index] == 1:
                        continue
                    visited[neighbor_index] = 1
                    queue.append((neighbor_x, neighbor_y))

            pixel_count = len(component_pixels)
            component_width = max_x - min_x + 1
            component_height = max_y - min_y + 1
            if pixel_count < live_config.marker_min_blob_pixels:
                continue
            if pixel_count > live_config.marker_max_blob_pixels:
                continue
            if component_width < live_config.marker_min_width_px:
                continue
            if component_width > live_config.marker_max_width_px:
                continue
            if component_height < live_config.marker_min_height_px:
                continue
            if component_height > live_config.marker_max_height_px:
                continue
            diagnostics["size_pass_count"] += 1
            bbox_area = max(1, component_width * component_height)
            fill_density = float(pixel_count) / float(bbox_area)
            dark_core_ratio = _compute_marker_dark_core_ratio(
                pixels=pixels,
                min_x=min_x,
                min_y=min_y,
                max_x=max_x,
                max_y=max_y,
                dark_core_max_rgb=live_config.marker_dark_core_max_rgb,
            )
            if fill_density < live_config.marker_min_fill_density:
                continue
            if fill_density > live_config.marker_max_fill_density:
                continue
            diagnostics["density_pass_count"] += 1
            if dark_core_ratio < live_config.marker_min_dark_core_ratio:
                continue
            diagnostics["dark_core_pass_count"] += 1
            confidence = min(1.0, marker_strength_sum / max(1.0, float(pixel_count)))
            if confidence < live_config.marker_confidence_threshold:
                continue
            diagnostics["confidence_pass_count"] += 1
            hits.append(
                TemplateHit(
                    label=marker_label,
                    x=roi_offset_xy[0] + min_x,
                    y=roi_offset_xy[1] + min_y,
                    width=component_width,
                    height=component_height,
                    confidence=confidence,
                    rotation_deg=0,
                    target_id=None,
                    source="pixel_marker",
                    metadata={
                        "pixel_count": pixel_count,
                        "mean_marker_strength": confidence,
                        "fill_density": fill_density,
                        "dark_core_ratio": dark_core_ratio,
                    },
                )
            )
    return tuple(hits), diagnostics


def _seed_from_upper_rescue_scan(
    *,
    image: Any,
    roi_box: tuple[int, int, int, int],
    template_pack: TemplatePack,
    live_config: LiveConfig,
    stride_px: int,
) -> tuple[TemplateHit, ...]:
    rescue_threshold = float(live_config.rescue_upper_scan_confidence_threshold)
    rescue_stride_px = int(max(1, live_config.rescue_upper_scan_stride_px))
    pseudo_marker_size_px = int(max(4, live_config.rescue_pseudo_marker_size_px))
    pseudo_marker_offset_y_px = int(max(4, live_config.rescue_pseudo_marker_offset_y_px))

    upper_hits = _match_local_variants(
        image=image,
        box=roi_box,
        variants=template_pack.mob_upper_variants,
        confidence_threshold=rescue_threshold,
        stride_px=max(rescue_stride_px, stride_px),
    )

    accepted: list[TemplateHit] = []
    for hit in upper_hits:
        ice_signature = _detect_ice_mob_signature(
            image=image,
            bbox=hit.bbox,
            live_config=live_config,
        )
        if not bool(ice_signature.get("accepted", False)):
            continue

        center_x, _ = hit.center_xy
        pseudo_x = max(0, int(round(center_x - (pseudo_marker_size_px / 2.0))))
        pseudo_y = max(0, int(hit.bbox[1]) - pseudo_marker_offset_y_px)
        accepted.append(
            TemplateHit(
                label="upper_rescue_seed",
                x=pseudo_x,
                y=pseudo_y,
                width=pseudo_marker_size_px,
                height=pseudo_marker_size_px,
                confidence=min(0.99, max(0.55, float(hit.confidence))),
                rotation_deg=hit.rotation_deg,
                target_id=hit.target_id,
                source="upper_rescue",
                metadata={
                    "seed_mode": "upper_rescue",
                    "rescue_confirmation_bbox": list(hit.bbox),
                    "rescue_template_label": hit.label,
                    "rescue_template_score": float(hit.confidence),
                    "ice_score": float(ice_signature.get("score", 0.0)),
                    "ice_pixel_count": int(ice_signature.get("ice_pixel_count", 0)),
                    "ice_pixel_ratio": float(ice_signature.get("ice_pixel_ratio", 0.0)),
                },
            )
        )

    return merge_template_hits(
        tuple(accepted),
        merge_distance_px=max(int(live_config.merge_distance_px), 36),
    )


def _marker_template_hit_is_valid(
    *,
    image: Any,
    hit: TemplateHit,
    live_config: LiveConfig,
) -> bool:
    if Image is None:
        return False
    left, top, width, height = hit.bbox
    if width <= 0 or height <= 0:
        return False
    crop = image.crop((left, top, left + width, top + height)).convert("RGB")
    crop_width, crop_height = crop.size
    pixels = crop.load()
    marker_pixel_count = 0
    marker_strength_sum = 0.0
    relaxed_min_red = max(120, live_config.marker_min_red - 60)
    relaxed_min_green = max(105, live_config.marker_min_green - 45)
    relaxed_red_blue_delta = max(8, live_config.marker_red_blue_delta - 18)
    relaxed_green_blue_delta = max(6, live_config.marker_green_blue_delta - 16)
    relaxed_balance_delta = max(90, live_config.marker_red_green_balance_delta + 25)
    for y in range(crop_height):
        for x in range(crop_width):
            red, green, blue = pixels[x, y]
            if live_config.marker_color_mode == "yellow":
                is_marker_pixel = (
                    red >= relaxed_min_red
                    and green >= relaxed_min_green
                    and (red - blue) >= relaxed_red_blue_delta
                    and (green - blue) >= relaxed_green_blue_delta
                    and abs(red - green) <= relaxed_balance_delta
                )
            else:
                is_marker_pixel = _is_marker_pixel(
                    red=red,
                    green=green,
                    blue=blue,
                    live_config=live_config,
                )
            if is_marker_pixel:
                marker_pixel_count += 1
                marker_strength_sum += _compute_marker_pixel_strength(
                    red=red,
                    green=green,
                    blue=blue,
                    live_config=live_config,
                )
    bbox_area = max(1, crop_width * crop_height)
    fill_density = float(marker_pixel_count) / float(bbox_area)
    dark_core_ratio = _compute_marker_dark_core_ratio(
        pixels=pixels,
        min_x=0,
        min_y=0,
        max_x=max(0, crop_width - 1),
        max_y=max(0, crop_height - 1),
        dark_core_max_rgb=live_config.marker_dark_core_max_rgb,
    )
    minimum_pixels = max(3, live_config.marker_min_blob_pixels // 2)
    minimum_density = min(live_config.marker_min_fill_density, 0.04)
    minimum_core_ratio = min(live_config.marker_min_dark_core_ratio, 0.01)
    if marker_pixel_count < minimum_pixels:
        return False
    if fill_density < minimum_density:
        return False
    if dark_core_ratio < minimum_core_ratio:
        return False
    expanded_margin_x = max(3, crop_width // 2)
    expanded_margin_y = max(3, crop_height // 2)
    expanded_left = max(0, left - expanded_margin_x)
    expanded_top = max(0, top - expanded_margin_y)
    expanded_right = min(image.size[0], left + width + expanded_margin_x)
    expanded_bottom = min(image.size[1], top + height + expanded_margin_y)
    expanded_crop = image.crop((expanded_left, expanded_top, expanded_right, expanded_bottom)).convert("RGB")
    expanded_width, expanded_height = expanded_crop.size
    expanded_pixels = expanded_crop.load()
    expanded_marker_pixel_count = 0
    expanded_min_x: int | None = None
    expanded_min_y: int | None = None
    expanded_max_x: int | None = None
    expanded_max_y: int | None = None
    for y in range(expanded_height):
        for x in range(expanded_width):
            red, green, blue = expanded_pixels[x, y]
            if live_config.marker_color_mode == "yellow":
                is_marker_pixel = (
                    red >= relaxed_min_red
                    and green >= relaxed_min_green
                    and (red - blue) >= relaxed_red_blue_delta
                    and (green - blue) >= relaxed_green_blue_delta
                    and abs(red - green) <= relaxed_balance_delta
                )
            else:
                is_marker_pixel = _is_marker_pixel(
                    red=red,
                    green=green,
                    blue=blue,
                    live_config=live_config,
                )
            if not is_marker_pixel:
                continue
            expanded_marker_pixel_count += 1
            expanded_min_x = x if expanded_min_x is None else min(expanded_min_x, x)
            expanded_min_y = y if expanded_min_y is None else min(expanded_min_y, y)
            expanded_max_x = x if expanded_max_x is None else max(expanded_max_x, x)
            expanded_max_y = y if expanded_max_y is None else max(expanded_max_y, y)
    if expanded_marker_pixel_count > int(max(1, live_config.marker_max_blob_pixels) * 1.5):
        return False
    if (
        expanded_min_x is not None
        and expanded_min_y is not None
        and expanded_max_x is not None
        and expanded_max_y is not None
    ):
        expanded_component_width = expanded_max_x - expanded_min_x + 1
        expanded_component_height = expanded_max_y - expanded_min_y + 1
        if expanded_component_width > int(max(1, live_config.marker_max_width_px) * 1.6):
            return False
        if expanded_component_height > int(max(1, live_config.marker_max_height_px) * 1.6):
            return False
    mean_marker_strength = marker_strength_sum / max(1.0, float(marker_pixel_count))
    confidence = max(hit.confidence, mean_marker_strength)
    hit.metadata.update(
        {
            "pixel_count": marker_pixel_count,
            "mean_marker_strength": mean_marker_strength,
            "fill_density": fill_density,
            "dark_core_ratio": dark_core_ratio,
            "validation_mode": "template_plus_color",
        }
    )
    object.__setattr__(hit, "confidence", confidence)
    return confidence >= max(0.40, live_config.marker_confidence_threshold - 0.08)


def _is_marker_pixel(
    *,
    red: int,
    green: int,
    blue: int,
    live_config: LiveConfig,
) -> bool:
    if live_config.marker_color_mode == "yellow":
        return (
            red >= live_config.marker_min_red
            and green >= live_config.marker_min_green
            and (red - blue) >= live_config.marker_red_blue_delta
            and (green - blue) >= live_config.marker_green_blue_delta
            and abs(red - green) <= live_config.marker_red_green_balance_delta
        )
    return (
        red >= live_config.marker_min_red
        and (red - green) >= live_config.marker_red_green_delta
        and (red - blue) >= live_config.marker_red_blue_delta
    )


def _compute_marker_pixel_strength(
    *,
    red: int,
    green: int,
    blue: int,
    live_config: LiveConfig,
) -> float:
    if live_config.marker_color_mode == "yellow":
        red_term = max(0.0, float(red - live_config.marker_min_red)) / max(
            1.0,
            float(255 - live_config.marker_min_red),
        )
        green_term = max(0.0, float(green - live_config.marker_min_green)) / max(
            1.0,
            float(255 - live_config.marker_min_green),
        )
        red_blue_term = max(
            0.0,
            float((red - blue) - live_config.marker_red_blue_delta),
        ) / max(1.0, float(255 - live_config.marker_red_blue_delta))
        green_blue_term = max(
            0.0,
            float((green - blue) - live_config.marker_green_blue_delta),
        ) / max(1.0, float(255 - live_config.marker_green_blue_delta))
        balance_term = 1.0 - min(
            1.0,
            abs(float(red - green)) / max(1.0, float(live_config.marker_red_green_balance_delta)),
        )
        return min(
            1.0,
            (red_term + green_term + red_blue_term + green_blue_term + balance_term) / 5.0,
        )
    red_term = max(0.0, float(red - live_config.marker_min_red)) / max(
        1.0,
        float(255 - live_config.marker_min_red),
    )
    green_delta_term = max(
        0.0,
        float((red - green) - live_config.marker_red_green_delta),
    ) / max(1.0, float(255 - live_config.marker_red_green_delta))
    blue_delta_term = max(
        0.0,
        float((red - blue) - live_config.marker_red_blue_delta),
    ) / max(1.0, float(255 - live_config.marker_red_blue_delta))
    return min(1.0, (red_term + green_delta_term + blue_delta_term) / 3.0)


def _compute_marker_dark_core_ratio(
    *,
    pixels: Any,
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
    dark_core_max_rgb: int,
) -> float:
    dark_pixels = 0
    total_pixels = 0
    for current_y in range(min_y, max_y + 1):
        for current_x in range(min_x, max_x + 1):
            red, green, blue = pixels[current_x, current_y]
            total_pixels += 1
            if max(red, green, blue) <= dark_core_max_rgb:
                dark_pixels += 1
    if total_pixels <= 0:
        return 0.0
    return float(dark_pixels) / float(total_pixels)


def _detect_player_veto(
    *,
    image: Any,
    box: tuple[int, int, int, int],
    live_config: LiveConfig,
) -> dict[str, Any]:
    if Image is None:
        return {
            "triggered": False,
            "green_pixel_count": 0,
            "green_pixel_ratio": 0.0,
            "green_bbox": None,
        }
    left, top, width, height = box
    if width <= 0 or height <= 0:
        return {
            "triggered": False,
            "green_pixel_count": 0,
            "green_pixel_ratio": 0.0,
            "green_bbox": None,
        }
    roi = image.crop((left, top, left + width, top + height)).convert("RGB")
    pixels = roi.load()
    green_pixels = 0
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            if (
                green >= live_config.player_veto_green_min_green
                and (green - red) >= live_config.player_veto_green_red_delta
                and (green - blue) >= live_config.player_veto_green_blue_delta
            ):
                green_pixels += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    area = max(1, width * height)
    green_ratio = float(green_pixels) / float(area)
    green_bbox = None
    bbox_width = 0
    bbox_height = 0
    if green_pixels > 0 and max_x >= min_x and max_y >= min_y:
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
        green_bbox = [
            left + min_x,
            top + min_y,
            bbox_width,
            bbox_height,
        ]
    triggered = (
        green_pixels >= live_config.player_veto_min_pixels
        and bbox_width >= live_config.player_veto_min_width_px
        and 0 < bbox_height <= live_config.player_veto_max_height_px
    )
    score = min(
        1.0,
        max(
            green_ratio * 4.0,
            green_pixels / max(1.0, float(live_config.player_veto_min_pixels * 4)),
            bbox_width / max(1.0, float(live_config.player_veto_min_width_px * 3)),
        ),
    )
    return {
        "triggered": triggered,
        "green_pixel_count": green_pixels,
        "green_pixel_ratio": green_ratio,
        "green_bbox": green_bbox,
        "score": score,
    }


def _detect_ice_mob_signature(
    *,
    image: Any,
    bbox: tuple[int, int, int, int],
    live_config: LiveConfig,
) -> dict[str, Any]:
    left, top, width, height = bbox
    if width <= 0 or height <= 0:
        return {
            "accepted": False,
            "ice_pixel_count": 0,
            "ice_pixel_ratio": 0.0,
        }
    focus_width = max(1, int(round(width * live_config.ice_mob_focus_width_ratio)))
    focus_height = max(1, int(round(height * live_config.ice_mob_focus_height_ratio)))
    focus_left = left + max(0, (width - focus_width) // 2)
    focus_top = top + max(0, height - focus_height - max(1, height // 12))
    focus_right = min(left + width, focus_left + focus_width)
    focus_bottom = min(top + height, focus_top + focus_height)
    crop = image.crop((focus_left, focus_top, focus_right, focus_bottom)).convert("RGB")
    pixels = list(crop.getdata())
    if not pixels:
        return {
            "accepted": False,
            "ice_pixel_count": 0,
            "ice_pixel_ratio": 0.0,
            "dark_ratio": 0.0,
            "brown_ratio": 0.0,
        }

    ice_pixels = 0
    dark_pixels = 0
    brown_pixels = 0
    for red, green, blue in pixels:
        brightness = (float(red) + float(green) + float(blue)) / 3.0
        if (
            blue >= live_config.ice_mob_min_blue
            and green >= live_config.ice_mob_min_green
            and brightness >= live_config.ice_mob_min_brightness
            and (blue + live_config.ice_mob_blue_red_tolerance) >= red
        ):
            ice_pixels += 1
        if brightness <= 96.0:
            dark_pixels += 1
        if (
            red >= 70
            and green >= 45
            and blue <= 120
            and red >= green
            and green >= blue
            and brightness <= 170.0
        ):
            brown_pixels += 1

    ice_ratio = ice_pixels / float(len(pixels))
    dark_ratio = dark_pixels / float(len(pixels))
    brown_ratio = brown_pixels / float(len(pixels))
    ice_score = max(
        0.0,
        min(
            1.0,
            (ice_ratio / max(live_config.ice_mob_min_ratio, 1e-6)) * 0.55
            + max(0.0, 1.0 - (dark_ratio / max(live_config.ice_mob_max_dark_ratio, 1e-6))) * 0.25
            + max(0.0, 1.0 - (brown_ratio / max(live_config.ice_mob_max_brown_ratio, 1e-6))) * 0.20,
        ),
    )
    accepted = (
        ice_pixels >= live_config.ice_mob_min_pixels
        and ice_ratio >= live_config.ice_mob_min_ratio
        and dark_ratio <= live_config.ice_mob_max_dark_ratio
        and brown_ratio <= live_config.ice_mob_max_brown_ratio
    )
    return {
        "accepted": accepted,
        "ice_pixel_count": ice_pixels,
        "ice_pixel_ratio": ice_ratio,
        "dark_ratio": dark_ratio,
        "brown_ratio": brown_ratio,
        "score": ice_score,
    }


def _build_local_roi_box(
    *,
    anchor_x: float,
    anchor_y: float,
    width: int,
    height: int,
    offset_y: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    left = int(round(anchor_x - (width / 2.0)))
    top = int(round(anchor_y + offset_y))
    left = max(0, min(left, max(0, frame_width - width)))
    top = max(0, min(top, max(0, frame_height - height)))
    return (
        left,
        top,
        min(width, frame_width - left),
        min(height, frame_height - top),
    )


def _match_local_variants(
    *,
    image: Any,
    box: tuple[int, int, int, int],
    variants: tuple[TemplateVariant, ...],
    confidence_threshold: float,
    stride_px: int,
) -> tuple[TemplateHit, ...]:
    if not variants:
        return ()
    left, top, width, height = box
    if width <= 0 or height <= 0:
        return ()
    roi_image = image.crop((left, top, left + width, top + height))
    return _match_template_variants(
        roi_image=roi_image,
        roi_offset_xy=(left, top),
        variants=variants,
        confidence_threshold=confidence_threshold,
        stride_px=stride_px,
    )


def _match_anchor_aligned_variants(
    *,
    image: Any,
    marker_hit: TemplateHit,
    box: tuple[int, int, int, int],
    variants: tuple[TemplateVariant, ...],
    confidence_threshold: float,
    stride_px: int,
) -> tuple[TemplateHit, ...]:
    if not variants or ImageChops is None or ImageStat is None or ImageOps is None:
        return ()
    left, top, width, height = box
    if width <= 0 or height <= 0:
        return ()
    right = left + width
    bottom = top + height
    marker_center_x, _ = marker_hit.center_xy
    marker_bottom_y = marker_hit.y + marker_hit.height
    horizontal_step = max(4, int(stride_px))
    x_offsets = (
        -(horizontal_step * 2),
        -horizontal_step,
        0,
        horizontal_step,
        horizontal_step * 2,
    )
    y_offsets = (
        -horizontal_step,
        -(horizontal_step // 2),
        0,
        horizontal_step // 2,
        horizontal_step,
        horizontal_step * 2,
    )
    hits: list[TemplateHit] = []
    for variant in variants:
        tested_positions: set[tuple[int, int]] = set()
        base_x = int(round(marker_center_x - (variant.width / 2.0)))
        for x_offset in x_offsets:
            candidate_x = base_x + x_offset
            for y_offset in y_offsets:
                candidate_y = int(round(marker_bottom_y + y_offset))
                if candidate_x < left or candidate_y < top:
                    continue
                if candidate_x + variant.width > right:
                    continue
                if candidate_y + variant.height > bottom:
                    continue
                position = (candidate_x, candidate_y)
                if position in tested_positions:
                    continue
                tested_positions.add(position)
                candidate_image = image.crop(
                    (
                        candidate_x,
                        candidate_y,
                        candidate_x + variant.width,
                        candidate_y + variant.height,
                    )
                )
                if variant.label == "occupied_swords" and variant.mask_image is not None:
                    confidence = _occupied_template_match_confidence(
                        candidate_image,
                        variant.image,
                        variant.mask_image,
                    )
                else:
                    confidence = _template_match_confidence(
                        candidate_image,
                        variant.image,
                        variant.mask_image,
                    )
                if confidence < confidence_threshold:
                    continue
                hits.append(
                    TemplateHit(
                        label=variant.label,
                        x=candidate_x,
                        y=candidate_y,
                        width=variant.width,
                        height=variant.height,
                        confidence=confidence,
                        rotation_deg=variant.rotation_deg,
                        target_id=None,
                        source=f"pixel_template_anchor:{variant.variant_name}",
                        metadata={
                            "template_path": str(variant.source_path),
                            "variant_name": variant.variant_name,
                            "search_mode": "anchor_aligned",
                        },
                    )
                )
    return tuple(hits)


def _select_best_confirmation_hit(
    *,
    marker_hit: TemplateHit,
    confirmation_hits: tuple[TemplateHit, ...],
    image: Any,
    live_config: LiveConfig,
) -> TemplateHit | None:
    if not confirmation_hits:
        return None
    marker_center_x, marker_center_y = marker_hit.center_xy
    alignment_weight = max(0.0, min(1.0, float(live_config.confirmation_alignment_weight)))
    foreground_weight = max(0.0, min(1.0, float(live_config.confirmation_foreground_weight)))
    template_weight = max(0.0, 1.0 - alignment_weight - foreground_weight)
    scored_hits: list[tuple[float, TemplateHit]] = []
    for hit in confirmation_hits:
        hit_center_x, hit_center_y = hit.center_xy
        horizontal_gap = abs(hit_center_x - marker_center_x)
        vertical_gap = hit_center_y - marker_center_y
        if horizontal_gap > live_config.confirmation_max_horizontal_offset_px:
            continue
        if vertical_gap < live_config.confirmation_min_vertical_offset_px:
            continue
        if vertical_gap > live_config.confirmation_max_vertical_offset_px:
            continue
        horizontal_penalty = horizontal_gap / max(
            1.0,
            float(live_config.confirmation_max_horizontal_offset_px),
        )
        vertical_midpoint = (
            live_config.confirmation_min_vertical_offset_px
            + live_config.confirmation_max_vertical_offset_px
        ) / 2.0
        vertical_span = max(
            1.0,
            float(
                live_config.confirmation_max_vertical_offset_px
                - live_config.confirmation_min_vertical_offset_px
            )
            / 2.0,
        )
        vertical_penalty = abs(vertical_gap - vertical_midpoint) / vertical_span
        alignment_score = max(0.0, 1.0 - ((horizontal_penalty * 0.6) + (vertical_penalty * 0.4)))
        foreground_score = _estimate_confirmation_foreground_score(
            image=image,
            bbox=hit.bbox,
        )
        combined_score = (
            (hit.confidence * template_weight)
            + (alignment_score * alignment_weight)
            + (foreground_score * foreground_weight)
        )
        scored_hits.append(
            (
                combined_score,
                replace(
                    hit,
                    metadata={
                        **hit.metadata,
                        "alignment_score": alignment_score,
                        "foreground_score": foreground_score,
                        "horizontal_gap_px": horizontal_gap,
                        "vertical_gap_px": vertical_gap,
                        "combined_confirmation_score": combined_score,
                    },
                ),
            )
        )
    if not scored_hits:
        return None
    scored_hits.sort(
        key=lambda item: (
            item[0],
            item[1].confidence,
            -float(item[1].metadata.get("horizontal_gap_px", 0.0)),
        ),
        reverse=True,
    )
    return scored_hits[0][1]


def _limit_seed_hits_for_confirmation(
    *,
    hits: tuple[TemplateHit, ...],
    reference_point_xy: tuple[float, float],
    max_seed_hits: int,
) -> tuple[TemplateHit, ...]:
    if max_seed_hits <= 0 or len(hits) <= max_seed_hits:
        return hits
    ranked_hits = sorted(
        hits,
        key=lambda hit: (
            float(hit.confidence) - min(math.dist(reference_point_xy, hit.center_xy), 1200.0) / 2400.0,
            float(hit.confidence),
            -math.dist(reference_point_xy, hit.center_xy),
        ),
        reverse=True,
    )
    return tuple(ranked_hits[:max_seed_hits])


def _estimate_confirmation_foreground_score(
    *,
    image: Any,
    bbox: tuple[int, int, int, int],
) -> float:
    if ImageStat is None or ImageOps is None or Image is None:
        return 0.0
    left, top, width, height = bbox
    if width <= 0 or height <= 0:
        return 0.0
    candidate_image = image.crop((left, top, left + width, top + height)).convert("RGB")
    if candidate_image.size[0] == 0 or candidate_image.size[1] == 0:
        return 0.0

    grayscale_image = ImageOps.grayscale(candidate_image)
    grayscale_stat = ImageStat.Stat(grayscale_image)
    brightness_stddev = float(grayscale_stat.stddev[0]) if grayscale_stat.stddev else 0.0

    hsv_image = candidate_image.convert("HSV")
    hsv_stat = ImageStat.Stat(hsv_image)
    saturation_mean = float(hsv_stat.mean[1]) if len(hsv_stat.mean) > 1 else 0.0

    brightness_score = max(0.0, min(1.0, brightness_stddev / 64.0))
    saturation_score = max(0.0, min(1.0, saturation_mean / 160.0))
    return (brightness_score * 0.55) + (saturation_score * 0.45)


def _detect_green_swords_hits(
    *,
    image: Any,
    box: tuple[int, int, int, int],
    live_config: LiveConfig,
) -> tuple[TemplateHit, ...]:
    if Image is None:
        return ()
    left, top, width, height = box
    if width <= 0 or height <= 0:
        return ()
    roi_image = image.crop((left, top, left + width, top + height)).convert("RGB")
    roi_width, roi_height = roi_image.size
    pixels = roi_image.load()
    green_mask = bytearray(roi_width * roi_height)
    for y in range(roi_height):
        for x in range(roi_width):
            red, green, blue = pixels[x, y]
            if (
                green >= live_config.swords_min_green
                and (green - red) >= live_config.swords_green_red_delta
                and (green - blue) >= live_config.swords_green_blue_delta
            ):
                green_mask[(y * roi_width) + x] = 1

    hits: list[TemplateHit] = []
    visited = bytearray(roi_width * roi_height)
    for y in range(roi_height):
        for x in range(roi_width):
            index = (y * roi_width) + x
            if green_mask[index] == 0 or visited[index] == 1:
                continue
            queue: list[tuple[int, int]] = [(x, y)]
            visited[index] = 1
            min_x = x
            max_x = x
            min_y = y
            max_y = y
            pixel_count = 0
            green_strength_sum = 0.0
            while queue:
                current_x, current_y = queue.pop()
                pixel_count += 1
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                red, green, blue = pixels[current_x, current_y]
                green_strength_sum += _compute_swords_pixel_strength(
                    red=red,
                    green=green,
                    blue=blue,
                    live_config=live_config,
                )
                for neighbor_x, neighbor_y in (
                    (current_x - 1, current_y),
                    (current_x + 1, current_y),
                    (current_x, current_y - 1),
                    (current_x, current_y + 1),
                    (current_x - 1, current_y - 1),
                    (current_x + 1, current_y - 1),
                    (current_x - 1, current_y + 1),
                    (current_x + 1, current_y + 1),
                ):
                    if not (0 <= neighbor_x < roi_width and 0 <= neighbor_y < roi_height):
                        continue
                    neighbor_index = (neighbor_y * roi_width) + neighbor_x
                    if green_mask[neighbor_index] == 0 or visited[neighbor_index] == 1:
                        continue
                    visited[neighbor_index] = 1
                    queue.append((neighbor_x, neighbor_y))

            if pixel_count < live_config.swords_min_blob_pixels:
                continue
            if pixel_count > live_config.swords_max_blob_pixels:
                continue
            component_width = max_x - min_x + 1
            component_height = max_y - min_y + 1
            if component_width < 3 or component_height < 3:
                continue
            if component_width > max(24, component_height * 2):
                continue
            if component_height > max(24, component_width * 2):
                continue
            confidence = min(1.0, green_strength_sum / max(1.0, float(pixel_count)))
            if confidence < live_config.swords_confidence_threshold:
                continue
            hits.append(
                TemplateHit(
                    label="occupied_swords",
                    x=left + min_x,
                    y=top + min_y,
                    width=component_width,
                    height=component_height,
                    confidence=confidence,
                    rotation_deg=0,
                    target_id=None,
                    source="pixel_green_swords",
                    metadata={
                        "pixel_count": pixel_count,
                        "mean_swords_strength": confidence,
                    },
                )
            )
    return tuple(hits)


def _estimate_green_pixel_ratio(
    *,
    image: Any,
    box: tuple[int, int, int, int],
    live_config: LiveConfig,
) -> float:
    left, top, width, height = box
    if width <= 0 or height <= 0:
        return 0.0
    roi_image = image.crop((left, top, left + width, top + height)).convert("RGB")
    pixels = list(roi_image.getdata())
    if not pixels:
        return 0.0
    matching_pixels = 0
    for red_value, green_value, blue_value in pixels:
        if (
            green_value >= live_config.swords_min_green
            and (green_value - red_value) >= live_config.swords_green_red_delta
            and (green_value - blue_value) >= live_config.swords_green_blue_delta
        ):
            matching_pixels += 1
    return matching_pixels / float(len(pixels))


def _compute_swords_pixel_strength(
    *,
    red: int,
    green: int,
    blue: int,
    live_config: LiveConfig,
) -> float:
    green_term = max(0.0, float(green - live_config.swords_min_green)) / max(
        1.0,
        float(255 - live_config.swords_min_green),
    )
    red_delta_term = max(
        0.0,
        float((green - red) - live_config.swords_green_red_delta),
    ) / max(1.0, float(255 - live_config.swords_green_red_delta))
    blue_delta_term = max(
        0.0,
        float((green - blue) - live_config.swords_green_blue_delta),
    ) / max(1.0, float(255 - live_config.swords_green_blue_delta))
    return min(1.0, (green_term + red_delta_term + blue_delta_term) / 3.0)


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(1, math.ceil((percentile / 100.0) * len(sorted_values)))
    return sorted_values[min(len(sorted_values) - 1, rank - 1)]


def _build_accuracy_summary(
    results: tuple[PerceptionFrameResult, ...],
) -> AccuracySummary | None:
    evaluated_results = tuple(
        result for result in results if isinstance(result.expectations, dict) and result.expectations
    )
    if not evaluated_results:
        return None

    behavior_match_count = 0
    selected_target_match_count = 0
    occupied_contract_match_count = 0

    for result in evaluated_results:
        expectations = result.expectations
        behavior_matches = True
        selected_matches = True
        occupied_matches = True

        selected_target_required = expectations.get("selected_target_required")
        if selected_target_required is True and result.selected_target is None:
            behavior_matches = False
            selected_matches = False
        if selected_target_required is False and result.selected_target is not None:
            behavior_matches = False
            selected_matches = False

        selected_target_screen_xy = expectations.get("selected_target_screen_xy")
        selected_target_max_error_px = float(expectations.get("selected_target_max_error_px", 48))
        if (
            isinstance(selected_target_screen_xy, list)
            and len(selected_target_screen_xy) == 2
            and all(isinstance(item, (int, float)) for item in selected_target_screen_xy)
        ):
            if result.selected_target is None:
                behavior_matches = False
                selected_matches = False
            else:
                selected_distance = math.dist(
                    (
                        float(result.selected_target.screen_x),
                        float(result.selected_target.screen_y),
                    ),
                    (
                        float(selected_target_screen_xy[0]),
                        float(selected_target_screen_xy[1]),
                    ),
                )
                if selected_distance > selected_target_max_error_px:
                    behavior_matches = False
                    selected_matches = False

        occupied_target_screen_xy = expectations.get("occupied_target_screen_xy")
        occupied_target_max_error_px = float(expectations.get("occupied_target_max_error_px", 56))
        if isinstance(occupied_target_screen_xy, list):
            actual_occupied_points = [
                (float(target.screen_x), float(target.screen_y))
                for target in result.occupied_detections
            ]
            for expected_xy in occupied_target_screen_xy:
                if not isinstance(expected_xy, list) or len(expected_xy) != 2:
                    continue
                if not any(
                    math.dist(
                        (float(expected_xy[0]), float(expected_xy[1])),
                        actual_xy,
                    )
                    <= occupied_target_max_error_px
                    for actual_xy in actual_occupied_points
                ):
                    behavior_matches = False
                    occupied_matches = False
                    break

        if behavior_matches:
            behavior_match_count += 1
        if selected_matches:
            selected_target_match_count += 1
        if occupied_matches:
            occupied_contract_match_count += 1

    return AccuracySummary(
        evaluated_frame_count=len(evaluated_results),
        behavior_match_count=behavior_match_count,
        selected_target_match_count=selected_target_match_count,
        occupied_contract_match_count=occupied_contract_match_count,
    )


def _build_benchmark_summary(
    results: tuple[PerceptionFrameResult, ...],
    *,
    strict_pixel_only: bool,
) -> BenchmarkSummary | None:
    evaluated_results = tuple(
        result
        for result in results
        if isinstance(result.ground_truth, dict) and bool(result.ground_truth.get("candidates"))
    )
    if not evaluated_results:
        return None

    frame_reports = tuple(_build_benchmark_frame_report(result) for result in evaluated_results)
    total_ground_truth_targets = sum(report.ground_truth_target_count for report in frame_reports)
    total_predicted_targets = sum(report.predicted_target_count for report in frame_reports)
    total_true_positive = sum(report.true_positive_count for report in frame_reports)
    total_false_positive = sum(report.false_positive_count for report in frame_reports)
    total_false_negative = sum(report.false_negative_count for report in frame_reports)
    total_player_false_positive = sum(report.player_false_positive_count for report in frame_reports)
    total_wrong_mob_false_positive = sum(report.wrong_mob_false_positive_count for report in frame_reports)
    total_ui_or_environment_false_positive = sum(
        report.ui_or_environment_false_positive_count for report in frame_reports
    )
    total_matched_ground_truth = sum(report.matched_ground_truth_count for report in frame_reports)
    total_occupied_correct = sum(report.occupied_correct_count for report in frame_reports)
    selected_accuracy_values = [1.0 if report.selected_target_correct else 0.0 for report in frame_reports]
    selected_in_zone_accuracy_values = [
        1.0 if report.selected_target_in_zone else 0.0 for report in frame_reports
    ]
    occupied_accuracy = None
    if total_matched_ground_truth > 0:
        occupied_accuracy = total_occupied_correct / float(total_matched_ground_truth)

    pixel_frame_count = sum(1 for result in evaluated_results if result.pipeline_mode == "pixel")
    fallback_frame_count = sum(1 for result in evaluated_results if result.pipeline_mode != "pixel")

    return BenchmarkSummary(
        evaluated_frame_count=len(frame_reports),
        strict_pixel_only=strict_pixel_only,
        pixel_frame_count=pixel_frame_count,
        fallback_frame_count=fallback_frame_count,
        target_true_positive_count=total_true_positive,
        target_false_positive_count=total_false_positive,
        target_false_negative_count=total_false_negative,
        target_recall=0.0
        if total_ground_truth_targets == 0
        else total_true_positive / float(total_ground_truth_targets),
        target_precision=0.0
        if total_predicted_targets == 0
        else total_true_positive / float(total_predicted_targets),
        occupied_classification_accuracy=occupied_accuracy,
        selected_target_accuracy=None
        if not selected_accuracy_values
        else sum(selected_accuracy_values) / float(len(selected_accuracy_values)),
        selected_target_in_zone_accuracy=None
        if not selected_in_zone_accuracy_values
        else sum(selected_in_zone_accuracy_values) / float(len(selected_in_zone_accuracy_values)),
        out_of_zone_rejection_count=NumericAggregate.from_values(
            "out_of_zone_rejection_count",
            (float(report.out_of_zone_target_count) for report in frame_reports),
        ),
        false_positive_reduction_after_zone_filtering=NumericAggregate.from_values(
            "false_positive_reduction_after_zone_filtering",
            (
                float(
                    len(
                        [
                            detection
                            for detection in result.out_of_zone_detections
                            if not detection.occupied
                        ]
                    )
                )
                for result in evaluated_results
            ),
        ),
        candidate_count=NumericAggregate.from_values(
            "candidate_count",
            (float(result.candidate_hit_count) for result in evaluated_results),
        ),
        merged_count=NumericAggregate.from_values(
            "merged_count",
            (float(result.merged_hit_count) for result in evaluated_results),
        ),
        false_positive_count=NumericAggregate.from_values(
            "false_positive_count",
            (float(report.false_positive_count) for report in frame_reports),
        ),
        false_negative_count=NumericAggregate.from_values(
            "false_negative_count",
            (float(report.false_negative_count) for report in frame_reports),
        ),
        player_false_positive_count=NumericAggregate.from_values(
            "player_false_positive_count",
            (float(report.player_false_positive_count) for report in frame_reports),
        ),
        wrong_mob_false_positive_count=NumericAggregate.from_values(
            "wrong_mob_false_positive_count",
            (float(report.wrong_mob_false_positive_count) for report in frame_reports),
        ),
        ui_or_environment_false_positive_count=NumericAggregate.from_values(
            "ui_or_environment_false_positive_count",
            (float(report.ui_or_environment_false_positive_count) for report in frame_reports),
        ),
        frame_reports=frame_reports,
    )


def _build_benchmark_frame_report(result: PerceptionFrameResult) -> BenchmarkFrameReport:
    ground_truth_candidates = _parse_ground_truth_candidates(result.ground_truth)
    unmatched_detection_indices = set(range(len(result.detections)))
    false_negative_candidate_ids: list[str] = []
    matched_ground_truth_count = 0
    occupied_correct_count = 0
    expected_selected_candidate_id: str | None = None
    selected_target_correct = True

    matched_detection_by_candidate_id: dict[str, LiveTargetDetection] = {}
    for candidate in ground_truth_candidates:
        if candidate.selected:
            expected_selected_candidate_id = candidate.candidate_id
            break

    for candidate in ground_truth_candidates:
        best_index: int | None = None
        best_distance = float("inf")
        for detection_index, detection in enumerate(result.detections):
            if detection_index not in unmatched_detection_indices:
                continue
            distance = math.dist(
                (float(detection.screen_x), float(detection.screen_y)),
                candidate.screen_xy,
            )
            if distance <= candidate.max_error_px and distance < best_distance:
                best_index = detection_index
                best_distance = distance
        if best_index is None:
            false_negative_candidate_ids.append(candidate.candidate_id)
            if candidate.selected:
                selected_target_correct = result.selected_target is None
            continue

        matched_ground_truth_count += 1
        unmatched_detection_indices.remove(best_index)
        matched_detection = result.detections[best_index]
        matched_detection_by_candidate_id[candidate.candidate_id] = matched_detection
        if matched_detection.occupied is candidate.occupied:
            occupied_correct_count += 1
        if candidate.selected:
            selected_target_correct = result.selected_target_id == matched_detection.target_id

    if expected_selected_candidate_id is None and result.selected_target_id is not None:
        selected_target_correct = False

    false_positive_target_ids = tuple(
        result.detections[index].target_id
        for index in sorted(unmatched_detection_indices)
    )
    false_positive_detections = tuple(
        result.detections[index]
        for index in sorted(unmatched_detection_indices)
    )
    player_false_positive_count = 0
    wrong_mob_false_positive_count = 0
    ui_or_environment_false_positive_count = 0
    for detection in false_positive_detections:
        bucket = _bucket_false_positive_detection(detection)
        if bucket == "player_fp":
            player_false_positive_count += 1
        elif bucket == "wrong_mob_fp":
            wrong_mob_false_positive_count += 1
        else:
            ui_or_environment_false_positive_count += 1
    false_positive_count = len(false_positive_target_ids)
    false_negative_count = len(false_negative_candidate_ids)
    predicted_target_count = len(result.detections)
    ground_truth_target_count = len(ground_truth_candidates)
    selected_target = result.selected_target
    selected_target_in_zone = True
    if selected_target is not None:
        selected_target_in_zone = bool(selected_target.metadata.get("in_scene_zone", True))

    return BenchmarkFrameReport(
        frame_source=result.frame_source,
        pipeline_mode=result.pipeline_mode,
        ground_truth_target_count=ground_truth_target_count,
        predicted_target_count=predicted_target_count,
        in_zone_target_count=len(result.in_zone_detections),
        out_of_zone_target_count=len(result.out_of_zone_detections),
        true_positive_count=matched_ground_truth_count,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        player_false_positive_count=player_false_positive_count,
        wrong_mob_false_positive_count=wrong_mob_false_positive_count,
        ui_or_environment_false_positive_count=ui_or_environment_false_positive_count,
        target_recall=0.0
        if ground_truth_target_count == 0
        else matched_ground_truth_count / float(ground_truth_target_count),
        target_precision=0.0
        if predicted_target_count == 0
        else matched_ground_truth_count / float(predicted_target_count),
        occupied_classification_accuracy=None
        if matched_ground_truth_count == 0
        else occupied_correct_count / float(matched_ground_truth_count),
        selected_target_correct=selected_target_correct,
        selected_target_in_zone=selected_target_in_zone,
        expected_selected_candidate_id=expected_selected_candidate_id,
        matched_ground_truth_count=matched_ground_truth_count,
        occupied_correct_count=occupied_correct_count,
        false_positive_target_ids=false_positive_target_ids,
        false_negative_candidate_ids=tuple(false_negative_candidate_ids),
    )


def _parse_ground_truth_candidates(ground_truth: dict[str, Any]) -> tuple[GroundTruthCandidate, ...]:
    raw_candidates = ground_truth.get("candidates", [])
    if not isinstance(raw_candidates, list):
        return ()
    candidates: list[GroundTruthCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates, start=1):
        if not isinstance(raw_candidate, dict):
            continue
        raw_screen_xy = raw_candidate.get("screen_xy")
        if (
            not isinstance(raw_screen_xy, (list, tuple))
            or len(raw_screen_xy) != 2
            or not all(isinstance(item, (int, float)) for item in raw_screen_xy)
        ):
            continue
        raw_bbox = raw_candidate.get("bbox")
        bbox: tuple[int, int, int, int] | None = None
        if (
            isinstance(raw_bbox, (list, tuple))
            and len(raw_bbox) == 4
            and all(isinstance(item, (int, float)) for item in raw_bbox)
        ):
            bbox = (
                int(raw_bbox[0]),
                int(raw_bbox[1]),
                int(raw_bbox[2]),
                int(raw_bbox[3]),
            )
        candidates.append(
            GroundTruthCandidate(
                candidate_id=str(raw_candidate.get("candidate_id", f"candidate_{index:03d}")),
                screen_xy=(float(raw_screen_xy[0]), float(raw_screen_xy[1])),
                max_error_px=float(raw_candidate.get("max_error_px", 48.0)),
                occupied=bool(raw_candidate.get("occupied", False)),
                selected=bool(raw_candidate.get("selected", False)),
                bbox=bbox,
            )
        )
    return tuple(candidates)


def _bucket_false_positive_detection(detection: LiveTargetDetection) -> str:
    player_veto_score = float(detection.metadata.get("player_veto_score", 0.0))
    brown_ratio = float(detection.metadata.get("ice_signature_brown_ratio", 0.0))
    ice_score = float(detection.metadata.get("ice_score", 0.0))
    marker_score = float(detection.metadata.get("marker_score", 0.0))
    if player_veto_score >= 0.10 or brown_ratio >= 0.12:
        return "player_fp"
    if ice_score >= 0.40 and marker_score >= 0.40:
        return "wrong_mob_fp"
    return "ui_or_environment_fp"


def _build_detection_tuning_parameters(live_config: LiveConfig) -> dict[str, Any]:
    return {
        "perception_confidence_threshold": live_config.perception_confidence_threshold,
        "confirmation_confidence_threshold": live_config.confirmation_confidence_threshold,
        "occupied_confidence_threshold": live_config.occupied_confidence_threshold,
        "template_match_stride_px": live_config.template_match_stride_px,
        "template_rotations_deg": list(live_config.template_rotations_deg),
        "merge_distance_px": live_config.merge_distance_px,
        "candidate_confirmation_frames": live_config.candidate_confirmation_frames,
        "occupied_confirmation_frames": live_config.occupied_confirmation_frames,
        "candidate_loss_frames": live_config.candidate_loss_frames,
        "enable_fallback_confirmation": live_config.enable_fallback_confirmation,
        "confirmation_anchor_search_enabled": live_config.confirmation_anchor_search_enabled,
        "confirmation_anchor_only": live_config.confirmation_anchor_only,
        "confirmation_template_stride_px": live_config.confirmation_template_stride_px,
        "max_seed_hits_for_confirmation": live_config.max_seed_hits_for_confirmation,
        "target_stability_enabled": live_config.target_stability_enabled,
        "target_stability_center_distance_px": live_config.target_stability_center_distance_px,
        "target_stability_switch_distance_gain_px": live_config.target_stability_switch_distance_gain_px,
        "target_stability_confidence_margin": live_config.target_stability_confidence_margin,
        "player_veto_enabled": live_config.player_veto_enabled,
        "player_veto_roi_width_px": live_config.player_veto_roi_width_px,
        "player_veto_roi_height_px": live_config.player_veto_roi_height_px,
        "player_veto_roi_offset_y_px": live_config.player_veto_roi_offset_y_px,
        "ice_mob_signature_enabled": live_config.ice_mob_signature_enabled,
        "ice_mob_min_blue": live_config.ice_mob_min_blue,
        "ice_mob_min_green": live_config.ice_mob_min_green,
        "ice_mob_min_brightness": live_config.ice_mob_min_brightness,
        "ice_mob_blue_red_tolerance": live_config.ice_mob_blue_red_tolerance,
        "ice_mob_min_pixels": live_config.ice_mob_min_pixels,
        "ice_mob_min_ratio": live_config.ice_mob_min_ratio,
        "ice_mob_focus_width_ratio": live_config.ice_mob_focus_width_ratio,
        "ice_mob_focus_height_ratio": live_config.ice_mob_focus_height_ratio,
        "ice_mob_max_dark_ratio": live_config.ice_mob_max_dark_ratio,
        "ice_mob_max_brown_ratio": live_config.ice_mob_max_brown_ratio,
        "rescue_upper_scan_confidence_threshold": live_config.rescue_upper_scan_confidence_threshold,
        "rescue_upper_scan_stride_px": live_config.rescue_upper_scan_stride_px,
        "rescue_pseudo_marker_size_px": live_config.rescue_pseudo_marker_size_px,
        "rescue_pseudo_marker_offset_y_px": live_config.rescue_pseudo_marker_offset_y_px,
        "preview_fast_mode": live_config.preview_fast_mode,
        "preview_skip_fallback_confirmation": live_config.preview_skip_fallback_confirmation,
        "preview_render_aux_boxes": live_config.preview_render_aux_boxes,
        "preview_analyze_every_nth_frame": live_config.preview_analyze_every_nth_frame,
        "scene_profile_path": None
        if live_config.scene_profile_path is None
        else str(live_config.scene_profile_path),
        "scene_zone_overlay_visible": live_config.scene_zone_overlay_visible,
    }


def _build_worst_frame_entries(
    results: tuple[PerceptionFrameResult, ...],
) -> tuple[dict[str, Any], ...]:
    scored_entries: list[tuple[float, dict[str, Any]]] = []
    benchmark_summary = _build_benchmark_summary(results, strict_pixel_only=False)
    benchmark_reports = {}
    if benchmark_summary is not None:
        benchmark_reports = {
            report.frame_source: report
            for report in benchmark_summary.frame_reports
        }
    for result in results:
        report = benchmark_reports.get(result.frame_source)
        false_positive_count = 0 if report is None else report.false_positive_count
        false_negative_count = 0 if report is None else report.false_negative_count
        selected_wrong = False if report is None else not report.selected_target_correct
        occupied_wrong = (
            False
            if report is None or report.occupied_classification_accuracy is None
            else report.occupied_classification_accuracy < 1.0
        )
        score = (
            (25.0 if selected_wrong else 0.0)
            + (15.0 if occupied_wrong else 0.0)
            + (10.0 * false_negative_count)
            + (4.0 * false_positive_count)
            + (result.timings.total_reaction_latency_ms / 250.0)
        )
        scored_entries.append(
            (
                score,
                {
                    "frame_source": result.frame_source,
                    "selected_target_id": result.selected_target_id,
                    "selected_target_correct": None if report is None else report.selected_target_correct,
                    "false_positive_count": false_positive_count,
                    "false_negative_count": false_negative_count,
                    "occupied_classification_accuracy": None
                    if report is None
                    else report.occupied_classification_accuracy,
                    "out_of_zone_target_count": len(result.out_of_zone_detections),
                    "detection_latency_ms": result.timings.detection_latency_ms,
                    "selection_latency_ms": result.timings.selection_latency_ms,
                    "total_reaction_latency_ms": result.timings.total_reaction_latency_ms,
                    "rejection_summary": {
                        "low_confidence": len(result.diagnostics.get("low_confidence_hits", [])),
                        "occupied": len(result.diagnostics.get("occupied_rejections", [])),
                        "out_of_zone": len(result.diagnostics.get("out_of_zone_rejections", [])),
                        "unstable": len(result.diagnostics.get("unstable_rejections", [])),
                        "player_veto": len(result.diagnostics.get("player_veto_rejections", [])),
                        "mob_signature": len(result.diagnostics.get("mob_signature_rejections", [])),
                    },
                    "score": score,
                },
            )
        )
    scored_entries.sort(key=lambda item: item[0], reverse=True)
    return tuple(entry for _, entry in scored_entries[:5])


def compare_perception_summary_payloads(
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_benchmark = dict(baseline_payload.get("benchmark_summary", {}))
    candidate_benchmark = dict(candidate_payload.get("benchmark_summary", {}))
    baseline_detection = dict(baseline_payload.get("detection_latency", {}))
    candidate_detection = dict(candidate_payload.get("detection_latency", {}))
    baseline_selection = dict(baseline_payload.get("selection_latency", {}))
    candidate_selection = dict(candidate_payload.get("selection_latency", {}))
    baseline_total = dict(baseline_payload.get("total_reaction_latency", {}))
    candidate_total = dict(candidate_payload.get("total_reaction_latency", {}))

    def _delta(left: Any, right: Any) -> float | None:
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        return float(right) - float(left)

    comparison = {
        "target_recall_delta": _delta(
            baseline_benchmark.get("target_recall"),
            candidate_benchmark.get("target_recall"),
        ),
        "target_precision_delta": _delta(
            baseline_benchmark.get("target_precision"),
            candidate_benchmark.get("target_precision"),
        ),
        "occupied_accuracy_delta": _delta(
            baseline_benchmark.get("occupied_classification_accuracy"),
            candidate_benchmark.get("occupied_classification_accuracy"),
        ),
        "selected_target_accuracy_delta": _delta(
            baseline_benchmark.get("selected_target_accuracy"),
            candidate_benchmark.get("selected_target_accuracy"),
        ),
        "false_positive_total_delta": _delta(
            baseline_benchmark.get("target_false_positive_count"),
            candidate_benchmark.get("target_false_positive_count"),
        ),
        "false_negative_total_delta": _delta(
            baseline_benchmark.get("target_false_negative_count"),
            candidate_benchmark.get("target_false_negative_count"),
        ),
        "detection_latency_avg_ms_delta": _delta(
            baseline_detection.get("avg_ms"),
            candidate_detection.get("avg_ms"),
        ),
        "selection_latency_avg_ms_delta": _delta(
            baseline_selection.get("avg_ms"),
            candidate_selection.get("avg_ms"),
        ),
        "total_reaction_latency_avg_ms_delta": _delta(
            baseline_total.get("avg_ms"),
            candidate_total.get("avg_ms"),
        ),
    }
    return comparison


def _template_hit_to_dict(hit: TemplateHit, *, rejection_reason: str | None = None) -> dict[str, Any]:
    payload = {
        "label": hit.label,
        "bbox": [hit.x, hit.y, hit.width, hit.height],
        "confidence": hit.confidence,
        "rotation_deg": hit.rotation_deg,
        "source": hit.source,
        "target_id": hit.target_id,
        "metadata": hit.metadata,
    }
    if rejection_reason is not None:
        payload["rejection_reason"] = rejection_reason
    return payload


def _detection_to_diagnostic_entry(
    detection: LiveTargetDetection,
    *,
    rejection_reason: str,
) -> dict[str, Any]:
    return {
        "target_id": detection.target_id,
        "screen_xy": [detection.screen_x, detection.screen_y],
        "bbox": None if detection.bbox is None else list(detection.bbox),
        "confidence": detection.confidence,
        "distance": detection.distance,
        "mob_variant": detection.mob_variant,
        "seen_frames": int(detection.metadata.get("seen_frames", 0)),
        "rejection_reason": rejection_reason,
    }


def _count_hits_by_label(hits: tuple[TemplateHit, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for hit in hits:
        counts[hit.label] = counts.get(hit.label, 0) + 1
    return counts


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_template_mask(image: Any) -> Any:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    mask = Image.new("L", (width, height), 0)
    rgb_pixels = rgb_image.load()
    mask_pixels = mask.load()
    for y in range(height):
        for x in range(width):
            red, green, blue = rgb_pixels[x, y]
            channel_span = max(red, green, blue) - min(red, green, blue)
            if channel_span >= 50 or red >= 140 or green >= 140:
                mask_pixels[x, y] = 255
    return mask


def _build_template_variants(
    *,
    label: str,
    template_path: Path,
    image: Any,
    rotations_deg: tuple[int, ...],
) -> list[TemplateVariant]:
    if ImageOps is None:
        raise RuntimeError("Pixel-based perception wymaga Pillow.")
    variants: list[TemplateVariant] = []
    base_mask = _build_template_mask(image) if label == "occupied_swords" else None
    for rotation_deg in rotations_deg:
        rotated_image = image.rotate(rotation_deg, expand=True)
        rotated_mask = None if base_mask is None else base_mask.rotate(rotation_deg, expand=True)
        variants.append(
            TemplateVariant(
                label=label,
                variant_name=f"{template_path.stem}_rot{rotation_deg}",
                rotation_deg=rotation_deg,
                image=rotated_image.convert("RGB"),
                source_path=template_path.resolve(),
                mask_image=rotated_mask,
            )
        )
    return variants


def _is_active_mob_template_path(template_path: Path) -> bool:
    stem = template_path.stem.lower()
    inactive_tokens = (
        "nie_targetowac",
        "player",
        "with_player",
        "behind_player",
        "sword",
        "swords",
        "occupied",
        "reference_only",
        "do_not_target",
    )
    return not any(token in stem for token in inactive_tokens)


def _match_template_variants(
    *,
    roi_image: Any,
    roi_offset_xy: tuple[int, int],
    variants: tuple[TemplateVariant, ...],
    confidence_threshold: float,
    stride_px: int,
) -> tuple[TemplateHit, ...]:
    if ImageChops is None or ImageStat is None or ImageOps is None:
        return ()
    roi_rgb = roi_image.convert("RGB")
    roi_width, roi_height = roi_rgb.size
    hits: list[TemplateHit] = []
    for variant in variants:
        if variant.width > roi_width or variant.height > roi_height:
            continue
        max_y = roi_height - variant.height
        max_x = roi_width - variant.width
        for y in range(0, max_y + 1, stride_px):
            for x in range(0, max_x + 1, stride_px):
                window = roi_rgb.crop((x, y, x + variant.width, y + variant.height))
                if variant.label == "occupied_swords" and variant.mask_image is not None:
                    confidence = _occupied_template_match_confidence(
                        window,
                        variant.image,
                        variant.mask_image,
                    )
                else:
                    confidence = _template_match_confidence(window, variant.image, variant.mask_image)
                if confidence < confidence_threshold:
                    continue
                hits.append(
                    TemplateHit(
                        label=variant.label,
                        x=roi_offset_xy[0] + x,
                        y=roi_offset_xy[1] + y,
                        width=variant.width,
                        height=variant.height,
                        confidence=confidence,
                        rotation_deg=variant.rotation_deg,
                        target_id=None,
                        source=f"pixel_template:{variant.variant_name}",
                        metadata={
                            "template_path": str(variant.source_path),
                            "variant_name": variant.variant_name,
                        },
                    )
                )
    return tuple(hits)


def _template_match_confidence(
    candidate_image: Any,
    template_image: Any,
    mask_image: Any | None = None,
) -> float:
    if mask_image is not None:
        return _masked_template_match_confidence(candidate_image, template_image, mask_image)
    difference = ImageChops.difference(candidate_image.convert("RGB"), template_image.convert("RGB"))
    grayscale_difference = ImageOps.grayscale(difference)
    mean_difference = float(ImageStat.Stat(grayscale_difference).mean[0])
    return max(0.0, 1.0 - (mean_difference / 255.0))


def _masked_template_match_confidence(
    candidate_image: Any,
    template_image: Any,
    mask_image: Any,
) -> float:
    candidate_rgb = candidate_image.convert("RGB")
    template_rgb = template_image.convert("RGB")
    mask_gray = mask_image.convert("L")
    candidate_pixels = candidate_rgb.load()
    template_pixels = template_rgb.load()
    mask_pixels = mask_gray.load()
    width, height = template_rgb.size
    total_difference = 0.0
    total_weight = 0.0
    for y in range(height):
        for x in range(width):
            weight = mask_pixels[x, y] / 255.0
            if weight <= 0.0:
                continue
            candidate_r, candidate_g, candidate_b = candidate_pixels[x, y]
            template_r, template_g, template_b = template_pixels[x, y]
            pixel_difference = (
                abs(candidate_r - template_r)
                + abs(candidate_g - template_g)
                + abs(candidate_b - template_b)
            ) / 3.0
            total_difference += pixel_difference * weight
            total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    mean_difference = total_difference / total_weight
    return max(0.0, 1.0 - (mean_difference / 255.0))


def _occupied_template_match_confidence(
    candidate_image: Any,
    template_image: Any,
    mask_image: Any,
) -> float:
    candidate_rgb = candidate_image.convert("RGB")
    template_rgb = template_image.convert("RGB")
    mask_gray = mask_image.convert("L")
    candidate_pixels = candidate_rgb.load()
    template_pixels = template_rgb.load()
    mask_pixels = mask_gray.load()
    width, height = template_rgb.size
    green_difference = 0.0
    green_weight = 0.0
    red_difference = 0.0
    red_weight = 0.0
    neutral_difference = 0.0
    neutral_weight = 0.0
    candidate_green_pixels = 0.0
    candidate_red_pixels = 0.0
    for y in range(height):
        for x in range(width):
            weight = mask_pixels[x, y] / 255.0
            if weight <= 0.0:
                continue
            candidate_r, candidate_g, candidate_b = candidate_pixels[x, y]
            template_r, template_g, template_b = template_pixels[x, y]
            pixel_difference = (
                abs(candidate_r - template_r)
                + abs(candidate_g - template_g)
                + abs(candidate_b - template_b)
            ) / 3.0
            if template_g >= template_r + 20 and template_g >= template_b + 20:
                green_difference += pixel_difference * weight
                green_weight += weight
                if candidate_g >= candidate_r + 20 and candidate_g >= candidate_b + 20:
                    candidate_green_pixels += weight
            elif template_r >= template_g + 20 and template_r >= template_b + 20:
                red_difference += pixel_difference * weight
                red_weight += weight
                if candidate_r >= candidate_g + 20 and candidate_r >= candidate_b + 20:
                    candidate_red_pixels += weight
            else:
                neutral_difference += pixel_difference * weight
                neutral_weight += weight
    if green_weight > 0.0 and candidate_green_pixels < (green_weight * 0.35):
        return 0.0
    if red_weight > 0.0 and candidate_red_pixels < (red_weight * 0.35):
        return 0.0
    scores: list[float] = []
    if green_weight > 0.0:
        scores.append(max(0.0, 1.0 - ((green_difference / green_weight) / 255.0)))
    if red_weight > 0.0:
        scores.append(max(0.0, 1.0 - ((red_difference / red_weight) / 255.0)))
    if neutral_weight > 0.0:
        scores.append(max(0.0, 1.0 - ((neutral_difference / neutral_weight) / 255.0)))
    if not scores:
        return 0.0
    return min(scores)


def _resolve_reference_point(
    frame: LiveFrame,
    *,
    scene_profile: SceneProfile | None = None,
) -> tuple[float, float]:
    raw_reference = frame.metadata.get("reference_point_xy")
    if (
        isinstance(raw_reference, (list, tuple))
        and len(raw_reference) == 2
        and all(isinstance(item, (int, float)) for item in raw_reference)
    ):
        return (float(raw_reference[0]), float(raw_reference[1]))
    if scene_profile is not None and scene_profile.reference_point_xy is not None:
        return scene_profile.reference_point_xy
    return (frame.width / 2.0, frame.height / 2.0)


def _synthesize_hits_from_targets(frame: LiveFrame) -> tuple[TemplateHit, ...]:
    raw_targets = frame.metadata.get("targets", [])
    if not isinstance(raw_targets, list):
        return ()
    hits: list[TemplateHit] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        center_x = int(raw_target.get("screen_x", 0))
        center_y = int(raw_target.get("screen_y", 0))
        raw_bbox_width = raw_target.get("bbox_width", 40)
        raw_bbox_height = raw_target.get("bbox_height", 52)
        bbox_width = 40 if raw_bbox_width is None else int(raw_bbox_width)
        bbox_height = 52 if raw_bbox_height is None else int(raw_bbox_height)
        bbox_x = center_x - (bbox_width // 2)
        bbox_y = center_y - (bbox_height // 2)
        target_id = _optional_str(raw_target.get("target_id"))
        mob_variant = str(raw_target.get("mob_variant", "mob_a"))
        orientation_deg = int(raw_target.get("orientation_deg", 0))
        confidence = float(raw_target.get("confidence", 0.95))
        metadata = dict(raw_target.get("metadata", {}))
        hits.append(
            TemplateHit(
                label=mob_variant,
                x=bbox_x,
                y=bbox_y,
                width=bbox_width,
                height=bbox_height,
                confidence=confidence,
                rotation_deg=orientation_deg,
                target_id=target_id,
                source="synthetic_target",
                metadata=metadata,
            )
        )
        duplicate_orientations = metadata.get("duplicate_orientations", [])
        if isinstance(duplicate_orientations, list):
            for duplicate in duplicate_orientations:
                if not isinstance(duplicate, dict):
                    continue
                hits.append(
                    TemplateHit(
                        label=str(duplicate.get("label", mob_variant)),
                        x=int(duplicate.get("x", bbox_x)),
                        y=int(duplicate.get("y", bbox_y)),
                        width=int(duplicate.get("width", bbox_width)),
                        height=int(duplicate.get("height", bbox_height)),
                        confidence=float(duplicate.get("confidence", max(0.5, confidence - 0.08))),
                        rotation_deg=int(duplicate.get("rotation_deg", orientation_deg)),
                        target_id=target_id,
                        source="synthetic_duplicate",
                        metadata=metadata,
                    )
                )
        if bool(raw_target.get("occupied", False)):
            hits.append(
                TemplateHit(
                    label="occupied_swords",
                    x=center_x - 16,
                    y=(bbox_y - 24),
                    width=32,
                    height=20,
                    confidence=float(raw_target.get("occupied_confidence", 0.98)),
                    rotation_deg=0,
                    target_id=target_id,
                    source="synthetic_occupied",
                    metadata={},
                )
            )
    return tuple(hits)


def _find_merge_index(
    merged_hits: list[TemplateHit],
    *,
    hit: TemplateHit,
    merge_distance_px: int,
) -> int | None:
    for index, existing in enumerate(merged_hits):
        if existing.target_id is not None and hit.target_id is not None and existing.target_id == hit.target_id:
            return index
        if math.dist(existing.center_xy, hit.center_xy) <= float(merge_distance_px):
            return index
    return None


def _merge_two_hits(previous: TemplateHit, new_hit: TemplateHit) -> TemplateHit:
    union_left = min(previous.x, new_hit.x)
    union_top = min(previous.y, new_hit.y)
    union_right = max(previous.x + previous.width, new_hit.x + new_hit.width)
    union_bottom = max(previous.y + previous.height, new_hit.y + new_hit.height)
    representative = new_hit if new_hit.confidence >= previous.confidence else previous
    previous_count = int(previous.metadata.get("raw_hit_count", 1))
    return TemplateHit(
        label=representative.label,
        x=union_left,
        y=union_top,
        width=union_right - union_left,
        height=union_bottom - union_top,
        confidence=max(previous.confidence, new_hit.confidence),
        rotation_deg=representative.rotation_deg,
        target_id=representative.target_id or previous.target_id or new_hit.target_id,
        source=representative.source,
        metadata={
            **representative.metadata,
            "raw_hit_count": previous_count + 1,
        },
    )
