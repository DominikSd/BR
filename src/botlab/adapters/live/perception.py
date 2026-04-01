from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from botlab.adapters.live.models import LiveFrame, LiveTargetDetection
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
    mob_variants: tuple[TemplateVariant, ...]
    occupied_variants: tuple[TemplateVariant, ...]


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
    expectations: dict[str, Any] = field(default_factory=dict)
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
            "expectations": self.expectations,
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
    detection_latency: LatencyAggregate
    selection_latency: LatencyAggregate
    total_reaction_latency: LatencyAggregate
    accuracy_summary: AccuracySummary | None = None

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
                    "free_target_count": len(result.free_detections),
                    "occupied_target_count": len(result.occupied_detections),
                    "selected_target_id": result.selected_target_id,
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
            "detection_latency": self.detection_latency.to_dict(),
            "selection_latency": self.selection_latency.to_dict(),
            "total_reaction_latency": self.total_reaction_latency.to_dict(),
            "frames": [result.to_dict() for result in self.frame_results],
        }
        if self.accuracy_summary is not None:
            payload["accuracy_summary"] = self.accuracy_summary.to_dict()
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
        mob_variants: list[TemplateVariant] = []
        mobs_root = self._live_config.mobs_template_directory
        if mobs_root.exists():
            for label_directory in sorted(path for path in mobs_root.iterdir() if path.is_dir()):
                label = label_directory.name
                template_paths = sorted(label_directory.glob("*.png"))
                preferred_paths = [
                    path for path in template_paths if "upper" in path.stem.lower()
                ]
                if preferred_paths:
                    template_paths = preferred_paths
                for template_path in template_paths:
                    base_image = Image.open(template_path).convert("RGB")
                    mob_variants.extend(
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
            mob_variants=tuple(mob_variants),
            occupied_variants=tuple(occupied_variants),
        )
        return self._cached_pack


class PerceptionAnalyzer:
    def __init__(
        self,
        live_config: LiveConfig,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._live_config = live_config
        self._clock = clock or time.perf_counter
        self._template_pack_loader = TemplatePackLoader(live_config)
        self._track_states: dict[str, TrackedDetectionState] = {}
        self._track_sequence = 0

    def analyze_frame(
        self,
        frame: LiveFrame,
        *,
        cycle_id: int,
        phase: str,
    ) -> PerceptionFrameResult:
        roi = extract_named_roi(frame, roi_name="spawn_roi", live_config=self._live_config)
        reference_point_xy = _resolve_reference_point(frame)

        detection_started_ts = frame.captured_at_ts
        detection_perf_started = self._clock()
        if frame.image is not None:
            roi_hits, detections = self._run_marker_first_pipeline(
                frame=frame,
                roi=roi,
                reference_point_xy=reference_point_xy,
            )
        else:
            raw_hits = self._load_metadata_template_hits(frame)
            roi_hits = self._filter_hits_to_roi(raw_hits=raw_hits, roi=roi)
            detections = self._build_detections_from_template_hits(
                hits=roi_hits,
                reference_point_xy=reference_point_xy,
            )
        detections = self._smooth_detections(detections)
        detection_finished_ts = detection_started_ts + self._resolve_duration_s(
            frame=frame,
            phase_key="detection_duration_s",
            measured_duration_s=max(0.0, self._clock() - detection_perf_started),
        )

        selection_perf_started = self._clock()
        selected_target = select_nearest_target(filter_occupied_targets(detections))
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
            expectations=dict(frame.metadata.get("expected_perception", {})),
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
            accuracy_summary=_build_accuracy_summary(results),
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

    def _run_marker_first_pipeline(
        self,
        *,
        frame: LiveFrame,
        roi: dict[str, Any],
        reference_point_xy: tuple[float, float],
    ) -> tuple[tuple[TemplateHit, ...], tuple[LiveTargetDetection, ...]]:
        if Image is None or frame.image is None:
            return (), ()
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

        marker_hits = _detect_red_marker_hits(
            roi_image=roi_image,
            roi_offset_xy=(roi_box[0], roi_box[1]),
            live_config=self._live_config,
        )
        merged_marker_hits = merge_template_hits(
            marker_hits,
            merge_distance_px=self._live_config.merge_distance_px,
        )

        raw_hits: list[TemplateHit] = list(marker_hits)
        detections: list[LiveTargetDetection] = []
        local_stride_px = max(1, min(4, stride_px))
        for index, marker_hit in enumerate(merged_marker_hits, start=1):
            occupied_roi_box = _build_local_roi_box(
                anchor_x=marker_hit.x + (marker_hit.width / 2.0),
                anchor_y=marker_hit.y + (marker_hit.height / 2.0),
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

            confirmation_roi_box = _build_local_roi_box(
                anchor_x=marker_hit.x + (marker_hit.width / 2.0),
                anchor_y=marker_hit.y + marker_hit.height,
                width=self._live_config.confirmation_roi_width_px,
                height=self._live_config.confirmation_roi_height_px,
                offset_y=self._live_config.confirmation_roi_offset_y_px,
                frame_width=frame.width,
                frame_height=frame.height,
            )
            confirmation_hits = _match_local_variants(
                image=image,
                box=confirmation_roi_box,
                variants=template_pack.mob_variants,
                confidence_threshold=self._live_config.confirmation_confidence_threshold,
                stride_px=local_stride_px,
            )
            raw_hits.extend(confirmation_hits)

            best_confirmation = _select_best_confirmation_hit(
                marker_hit=marker_hit,
                confirmation_hits=confirmation_hits,
                image=image,
                live_config=self._live_config,
            )
            if best_confirmation is None:
                continue

            occupied, occupied_confidence = classify_occupied(
                mob_hit=best_confirmation,
                occupied_hits=all_occupied_hits,
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
                    occupied=occupied,
                    mob_variant=best_confirmation.label,
                    reachable=True,
                    confidence=max(marker_hit.confidence, detection_confidence),
                    bbox=best_confirmation.bbox,
                    orientation_deg=best_confirmation.rotation_deg,
                    metadata={
                        "detection_pipeline": "marker_first",
                        "marker_bbox": list(marker_hit.bbox),
                        "marker_confidence": marker_hit.confidence,
                        "marker_pixel_count": int(marker_hit.metadata.get("pixel_count", 0)),
                        "occupied_confidence": occupied_confidence,
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
                        "confirmation_template": best_confirmation.metadata.get("template_path"),
                        "variant_name": best_confirmation.metadata.get("variant_name"),
                        "raw_hit_count": int(best_confirmation.metadata.get("raw_hit_count", 1)),
                    },
                )
            )
        ordered_detections = tuple(sorted(detections, key=lambda item: (item.distance, item.target_id)))
        return tuple(raw_hits), ordered_detections

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
                        "occupied_confidence": occupied_confidence,
                        "raw_hit_count": int(merged_hit.metadata.get("raw_hit_count", 1)),
                        **merged_hit.metadata,
                    },
                )
            )
        return tuple(sorted(detections, key=lambda item: (item.distance, item.target_id)))

    def _smooth_detections(
        self,
        detections: tuple[LiveTargetDetection, ...],
    ) -> tuple[LiveTargetDetection, ...]:
        if not detections and not self._track_states:
            return ()
        matched_track_ids: set[str] = set()
        smoothed: list[LiveTargetDetection] = []
        for detection in detections:
            track_id = self._match_track_id(detection, matched_track_ids)
            if track_id is None:
                self._track_sequence += 1
                track_id = f"track-{self._track_sequence:04d}"
                seen_frames = 1
                occupied_seen_frames = 1 if detection.occupied else 0
            else:
                previous = self._track_states[track_id]
                seen_frames = previous.seen_frames + 1
                occupied_seen_frames = (
                    previous.occupied_seen_frames + 1 if detection.occupied else 0
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
            stable = seen_frames >= self._live_config.candidate_confirmation_frames
            if not stable:
                continue
            smoothed_occupied = detection.occupied and (
                occupied_seen_frames >= self._live_config.occupied_confirmation_frames
            )
            smoothed.append(
                replace(
                    detection,
                    occupied=smoothed_occupied,
                    metadata={
                        **detection.metadata,
                        "track_id": track_id,
                        "seen_frames": seen_frames,
                        "occupied_seen_frames": occupied_seen_frames,
                        "stable_candidate": stable,
                    },
                )
            )

        stale_track_ids: list[str] = []
        for track_id, state in self._track_states.items():
            if track_id in matched_track_ids:
                continue
            state.missed_frames += 1
            if state.missed_frames >= self._live_config.candidate_loss_frames:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            del self._track_states[track_id]
        return tuple(sorted(smoothed, key=lambda item: (item.distance, item.target_id)))

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
        return artifact_paths

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
        for hit in result.raw_hits:
            hit_color = "#9ca3af"
            if hit.label == "red_marker":
                hit_color = "#ef4444"
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
            color = "#ef4444" if detection.occupied else "#22c55e"
            stroke_width = 4 if selected else 2
            label = "occupied" if detection.occupied else "free"
            marker_bbox = detection.metadata.get("marker_bbox")
            occupied_roi = detection.metadata.get("occupied_roi")
            confirmation_roi = detection.metadata.get("confirmation_roi")
            if isinstance(marker_bbox, list) and len(marker_bbox) == 4:
                lines.append(
                    f'<rect x="{marker_bbox[0]}" y="{marker_bbox[1]}" width="{marker_bbox[2]}" height="{marker_bbox[3]}" fill="none" stroke="#ef4444" stroke-width="2" />'
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
                f'<text x="{bbox[0]}" y="{max(14, bbox[1] - 6)}" fill="#f9fafb" font-size="14">{detection.target_id} {label} conf={detection.confidence:.2f} dist={detection.distance:.1f} marker={float(detection.metadata.get("marker_confidence", 0.0)):.2f}</text>'
            )
            if selected:
                lines.append(
                    f'<text x="{bbox[0]}" y="{bbox[1] + bbox[3] + 18}" fill="#fde047" font-size="14">selected</text>'
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
        for path in sorted(root.iterdir()):
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
            if path.name.endswith("_perception.json"):
                continue
            if path.name == "perception_session_summary.json":
                continue
            loaded_frames.append((path.stem, self.load_frame(path)))
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


class PerceptionAnalysisRunner:
    def __init__(
        self,
        *,
        live_config: LiveConfig,
        output_directory: Path,
        clock: Clock | None = None,
    ) -> None:
        self._analyzer = PerceptionAnalyzer(live_config, clock=clock)
        self._loader = PerceptionFrameLoader()
        self._artifact_writer = PerceptionArtifactWriter(output_directory)

    def analyze_frame_path(self, frame_path: str | Path) -> PerceptionSessionSummary:
        frame = self._loader.load_frame(frame_path)
        frame_name = Path(frame_path).stem
        result = self._analyzer.analyze_frame(frame, cycle_id=1, phase=frame_name)
        artifact_paths = self._artifact_writer.write_batch_result(
            frame_name=frame_name,
            frame=frame,
            result=result,
        )
        persisted_result = replace(result, artifact_paths=artifact_paths)
        self._artifact_writer.append_jsonl(record_name=frame_name, result=persisted_result)
        summary = self._analyzer.summarize_session((persisted_result,))
        self._artifact_writer.write_session_summary(summary)
        return summary

    def analyze_directory(self, directory: str | Path) -> PerceptionSessionSummary:
        frame_entries = self._loader.load_directory(directory)
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
    return (best_confidence > 0.0, best_confidence)


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
        for detection in perception_result.detections
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
            "free_target_count": len(perception_result.free_detections),
            "occupied_target_count": len(perception_result.occupied_detections),
            "selected_target_id": perception_result.selected_target_id,
            "detection_latency_ms": perception_result.timings.detection_latency_ms,
            "selection_latency_ms": perception_result.timings.selection_latency_ms,
            "total_reaction_latency_ms": perception_result.timings.total_reaction_latency_ms,
        },
    )


def _detect_red_marker_hits(
    *,
    roi_image: Any,
    roi_offset_xy: tuple[int, int],
    live_config: LiveConfig,
) -> tuple[TemplateHit, ...]:
    if Image is None:
        return ()
    rgb_image = roi_image.convert("RGB")
    width, height = rgb_image.size
    pixels = rgb_image.load()
    red_mask = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels[x, y]
            if (
                red >= live_config.marker_min_red
                and (red - green) >= live_config.marker_red_green_delta
                and (red - blue) >= live_config.marker_red_blue_delta
            ):
                red_mask[(y * width) + x] = 1

    hits: list[TemplateHit] = []
    visited = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            index = (y * width) + x
            if red_mask[index] == 0 or visited[index] == 1:
                continue
            queue: list[tuple[int, int]] = [(x, y)]
            visited[index] = 1
            component_pixels: list[tuple[int, int]] = []
            min_x = x
            max_x = x
            min_y = y
            max_y = y
            red_strength_sum = 0.0
            while queue:
                current_x, current_y = queue.pop()
                component_pixels.append((current_x, current_y))
                min_x = min(min_x, current_x)
                max_x = max(max_x, current_x)
                min_y = min(min_y, current_y)
                max_y = max(max_y, current_y)
                red, green, blue = pixels[current_x, current_y]
                red_strength_sum += _compute_marker_pixel_strength(
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
                    if red_mask[neighbor_index] == 0 or visited[neighbor_index] == 1:
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
            confidence = min(1.0, red_strength_sum / max(1.0, float(pixel_count)))
            if confidence < live_config.marker_confidence_threshold:
                continue
            hits.append(
                TemplateHit(
                    label="red_marker",
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
                    },
                )
            )
    return tuple(hits)


def _compute_marker_pixel_strength(
    *,
    red: int,
    green: int,
    blue: int,
    live_config: LiveConfig,
) -> float:
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


def _resolve_reference_point(frame: LiveFrame) -> tuple[float, float]:
    raw_reference = frame.metadata.get("reference_point_xy")
    if (
        isinstance(raw_reference, (list, tuple))
        and len(raw_reference) == 2
        and all(isinstance(item, (int, float)) for item in raw_reference)
    ):
        return (float(raw_reference[0]), float(raw_reference[1]))
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
