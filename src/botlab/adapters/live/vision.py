from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from botlab.adapters.live.models import (
    LiveFrame,
    LiveResourceSnapshot,
    LiveStateSnapshot,
    LiveTargetDetection,
)
from botlab.config import CombatConfig, LiveConfig
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency path
    Image = None


def extract_named_roi(frame: LiveFrame, *, roi_name: str, live_config: LiveConfig) -> dict[str, Any]:
    metadata_override = frame.metadata.get(roi_name)
    if (
        isinstance(metadata_override, (list, tuple))
        and len(metadata_override) == 4
        and all(isinstance(item, int) for item in metadata_override)
    ):
        x, y, width, height = metadata_override
    else:
        roi_map = {
            "spawn_roi": live_config.spawn_roi,
            "hp_bar_roi": live_config.hp_bar_roi,
            "condition_bar_roi": live_config.condition_bar_roi,
            "combat_indicator_roi": live_config.combat_indicator_roi,
            "reward_roi": live_config.reward_roi,
        }
        if roi_name not in roi_map:
            raise ValueError(f"Nieznany ROI '{roi_name}'.")
        x, y, width, height = roi_map[roi_name]
    if x >= frame.width or y >= frame.height:
        x, y, width, height = (0, 0, frame.width, frame.height)
    else:
        width = min(width, max(1, frame.width - x))
        height = min(height, max(1, frame.height - y))
    return {
        "name": roi_name,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "frame_width": frame.width,
        "frame_height": frame.height,
    }


def parse_target_detections(frame: LiveFrame) -> tuple[LiveTargetDetection, ...]:
    raw_targets = frame.metadata.get("targets", [])
    if not isinstance(raw_targets, list):
        return ()

    detections: list[LiveTargetDetection] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        detections.append(
            LiveTargetDetection(
                target_id=str(raw_target.get("target_id", "unknown")),
                screen_x=int(raw_target.get("screen_x", 0)),
                screen_y=int(raw_target.get("screen_y", 0)),
                distance=float(raw_target.get("distance", 0.0)),
                occupied=bool(raw_target.get("occupied", False)),
                mob_variant=str(raw_target.get("mob_variant", "mob_a")),
                reachable=bool(raw_target.get("reachable", True)),
                metadata=dict(raw_target.get("metadata", {})),
            )
        )
    return tuple(detections)


def filter_occupied_targets(
    detections: tuple[LiveTargetDetection, ...],
) -> tuple[LiveTargetDetection, ...]:
    return tuple(detection for detection in detections if not detection.occupied)


def select_nearest_target(
    detections: tuple[LiveTargetDetection, ...],
) -> LiveTargetDetection | None:
    if not detections:
        return None
    return min(detections, key=lambda item: (item.distance, item.target_id))


def should_start_rest(
    *,
    hp_ratio: float,
    condition_ratio: float,
    combat_config: CombatConfig,
) -> bool:
    return (
        hp_ratio < combat_config.rest_start_threshold
        or condition_ratio < combat_config.rest_start_threshold
    )


def ready_after_rest(
    *,
    hp_ratio: float,
    condition_ratio: float,
    combat_config: CombatConfig,
) -> bool:
    return (
        hp_ratio >= combat_config.rest_stop_threshold
        and condition_ratio >= combat_config.rest_stop_threshold
    )


@dataclass(slots=True, frozen=True)
class StallDetector:
    timeout_s: float = 1.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s musi byc wieksze od 0.")

    def is_stalled(
        self,
        *,
        last_progress_ts: float,
        now_ts: float,
        entered_combat: bool,
    ) -> bool:
        if entered_combat:
            return False
        return (now_ts - last_progress_ts) >= self.timeout_s


class SimpleTemplateMatcher:
    def match_flag(
        self,
        frame: LiveFrame,
        *,
        flag_name: str,
        default: bool = False,
    ) -> bool:
        template_flags = frame.metadata.get("template_flags", {})
        if not isinstance(template_flags, dict):
            return default
        return bool(template_flags.get(flag_name, default))


class LiveResourceProvider:
    def __init__(self, live_config: LiveConfig) -> None:
        self._live_config = live_config

    def read_resources(self, frame: LiveFrame) -> LiveResourceSnapshot:
        hp_roi = extract_named_roi(frame, roi_name="hp_bar_roi", live_config=self._live_config)
        condition_roi = extract_named_roi(
            frame,
            roi_name="condition_bar_roi",
            live_config=self._live_config,
        )
        if frame.image is not None and Image is not None:
            hp_ratio, hp_metadata = self._read_hp_ratio_from_pixels(frame, hp_roi)
            condition_ratio, condition_metadata = self._read_condition_ratio_from_pixels(
                frame,
                condition_roi,
            )
            return LiveResourceSnapshot(
                hp_ratio=hp_ratio,
                condition_ratio=condition_ratio,
                metadata={
                    "source": "pixel",
                    "hp": hp_metadata,
                    "condition": condition_metadata,
                },
            )
        return LiveResourceSnapshot(
            hp_ratio=float(frame.metadata.get("hp_ratio", 1.0)),
            condition_ratio=float(frame.metadata.get("condition_ratio", 1.0)),
            metadata={
                "source": "metadata_fallback",
                "hp": {"roi": hp_roi},
                "condition": {"roi": condition_roi},
            },
        )

    def aggregate_resource_reads(
        self,
        samples: tuple[LiveResourceSnapshot, ...],
        *,
        previous_snapshot: LiveResourceSnapshot | None = None,
    ) -> "LiveResourceReadAggregate":
        if not samples:
            return LiveResourceReadAggregate(
                sample_count=0,
                hp_ratio=0.0,
                condition_ratio=0.0,
                confidence=0.0,
                warnings=("resource_sampling_empty",),
                hp_median=0.0,
                condition_median=0.0,
                hp_spread=0.0,
                condition_spread=0.0,
                samples=(),
            )

        hp_values = [sample.hp_ratio for sample in samples]
        condition_values = [sample.condition_ratio for sample in samples]
        hp_median = float(median(hp_values))
        condition_median = float(median(condition_values))
        hp_spread = max(hp_values) - min(hp_values)
        condition_spread = max(condition_values) - min(condition_values)
        per_sample_confidences = [
            _resource_sample_confidence(sample)
            for sample in samples
        ]
        base_confidence = sum(per_sample_confidences) / len(per_sample_confidences)
        consistency_penalty = min(1.0, max(hp_spread, condition_spread) / max(
            self._live_config.rest_resource_warning_spread_threshold,
            1e-6,
        ))
        confidence = max(0.0, min(1.0, base_confidence * (1.0 - (0.5 * consistency_penalty))))

        warnings: list[str] = []
        if any(sample.metadata.get("source") != "pixel" for sample in samples):
            warnings.append("resource_metadata_fallback_used")
        if hp_spread > self._live_config.rest_resource_warning_spread_threshold:
            warnings.append("hp_read_spread_high")
        if condition_spread > self._live_config.rest_resource_warning_spread_threshold:
            warnings.append("condition_read_spread_high")
        if confidence < self._live_config.rest_resource_min_confidence:
            warnings.append("resource_confidence_low")
        if previous_snapshot is not None:
            if hp_median + 0.05 < previous_snapshot.hp_ratio:
                warnings.append("hp_dropped_during_rest")
            if condition_median + 0.05 < previous_snapshot.condition_ratio:
                warnings.append("condition_dropped_during_rest")

        return LiveResourceReadAggregate(
            sample_count=len(samples),
            hp_ratio=hp_median,
            condition_ratio=condition_median,
            confidence=confidence,
            warnings=tuple(warnings),
            hp_median=hp_median,
            condition_median=condition_median,
            hp_spread=hp_spread,
            condition_spread=condition_spread,
            samples=samples,
        )

    def _read_hp_ratio_from_pixels(
        self,
        frame: LiveFrame,
        roi: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        fill_ratio, matching_pixels, total_pixels = _compute_bar_fill_ratio(
            frame=frame,
            roi=roi,
            matcher=lambda red_value, green_value, blue_value: (
                red_value >= self._live_config.hp_bar_min_red
                and (red_value - green_value) >= self._live_config.hp_bar_red_green_delta
                and (red_value - blue_value) >= self._live_config.hp_bar_red_blue_delta
            ),
        )
        normalized_ratio = fill_ratio if fill_ratio >= self._live_config.hp_bar_min_fill_ratio else 0.0
        return (
            normalized_ratio,
            {
                "roi": roi,
                "fill_ratio": fill_ratio,
                "normalized_ratio": normalized_ratio,
                "matching_pixels": matching_pixels,
                "total_pixels": total_pixels,
            },
        )

    def _read_condition_ratio_from_pixels(
        self,
        frame: LiveFrame,
        roi: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        fill_ratio, matching_pixels, total_pixels = _compute_bar_fill_ratio(
            frame=frame,
            roi=roi,
            matcher=lambda red_value, green_value, blue_value: (
                green_value >= self._live_config.condition_bar_min_green
                and (green_value - red_value) >= self._live_config.condition_bar_green_red_delta
                and (green_value - blue_value) >= self._live_config.condition_bar_green_blue_delta
            ),
        )
        normalized_ratio = (
            fill_ratio if fill_ratio >= self._live_config.condition_bar_min_fill_ratio else 0.0
        )
        return (
            normalized_ratio,
            {
                "roi": roi,
                "fill_ratio": fill_ratio,
                "normalized_ratio": normalized_ratio,
                "matching_pixels": matching_pixels,
                "total_pixels": total_pixels,
            },
        )


class SimpleStateDetector:
    def __init__(
        self,
        live_config: LiveConfig | None = None,
        template_matcher: SimpleTemplateMatcher | None = None,
    ) -> None:
        self._live_config = live_config
        self._template_matcher = template_matcher or SimpleTemplateMatcher()

    def detect_state(self, frame: LiveFrame) -> LiveStateSnapshot:
        if self._live_config is not None and Image is not None and frame.image is not None:
            in_combat, combat_metadata = self._detect_in_combat_from_pixels(frame)
            reward_visible, reward_metadata = self._detect_reward_from_pixels(frame)
            detection_source = "pixel"
        else:
            in_combat = bool(frame.metadata.get("in_combat", False)) or self._template_matcher.match_flag(
                frame,
                flag_name="combat_indicator",
                default=False,
            )
            reward_visible = bool(frame.metadata.get("reward_visible", False)) or self._template_matcher.match_flag(
                frame,
                flag_name="reward_screen",
                default=False,
            )
            combat_metadata = {"source": "metadata_fallback"}
            reward_metadata = {"source": "metadata_fallback"}
            detection_source = "metadata_fallback"
        rest_available = bool(frame.metadata.get("rest_available", True))
        return LiveStateSnapshot(
            in_combat=in_combat,
            reward_visible=reward_visible,
            rest_available=rest_available,
            metadata={
                "source": detection_source,
                "combat_indicator": combat_metadata,
                "reward_visibility": reward_metadata,
            },
        )

    def _detect_in_combat_from_pixels(self, frame: LiveFrame) -> tuple[bool, dict[str, Any]]:
        if self._live_config is None or Image is None or frame.image is None:
            return False, {"source": "unavailable"}
        roi = extract_named_roi(frame, roi_name="combat_indicator_roi", live_config=self._live_config)
        left = int(roi["x"])
        top = int(roi["y"])
        right = left + int(roi["width"])
        bottom = top + int(roi["height"])
        if right <= left or bottom <= top:
            return False, {"source": "pixel", "roi": roi, "reason": "empty_roi"}
        roi_image = frame.image.crop((left, top, right, bottom)).convert("RGB")
        pixels = list(roi_image.getdata())
        if not pixels:
            return False, {"source": "pixel", "roi": roi, "reason": "empty_pixels"}
        matching_pixels = 0
        for red_value, green_value, blue_value in pixels:
            if (
                red_value >= self._live_config.combat_indicator_min_red
                and (red_value - green_value) >= self._live_config.combat_indicator_red_green_delta
                and (red_value - blue_value) >= self._live_config.combat_indicator_red_blue_delta
            ):
                matching_pixels += 1
        active_ratio = matching_pixels / float(len(pixels))
        detected = active_ratio >= self._live_config.combat_indicator_min_ratio
        return detected, {
            "source": "pixel",
            "roi": roi,
            "active_ratio": active_ratio,
            "matching_pixels": matching_pixels,
            "total_pixels": len(pixels),
        }

    def _detect_reward_from_pixels(self, frame: LiveFrame) -> tuple[bool, dict[str, Any]]:
        if self._live_config is None or Image is None or frame.image is None:
            return False, {"source": "unavailable"}
        roi = extract_named_roi(frame, roi_name="reward_roi", live_config=self._live_config)
        left = int(roi["x"])
        top = int(roi["y"])
        right = left + int(roi["width"])
        bottom = top + int(roi["height"])
        if right <= left or bottom <= top:
            return False, {"source": "pixel", "roi": roi, "reason": "empty_roi"}
        roi_image = frame.image.crop((left, top, right, bottom)).convert("RGB")
        pixels = list(roi_image.getdata())
        if not pixels:
            return False, {"source": "pixel", "roi": roi, "reason": "empty_pixels"}
        matching_pixels = 0
        for red_value, green_value, blue_value in pixels:
            if (
                red_value >= self._live_config.reward_min_red
                and green_value >= self._live_config.reward_min_green
                and blue_value <= self._live_config.reward_max_blue
            ):
                matching_pixels += 1
        active_ratio = matching_pixels / float(len(pixels))
        detected = active_ratio >= self._live_config.reward_min_ratio
        return detected, {
            "source": "pixel",
            "roi": roi,
            "active_ratio": active_ratio,
            "matching_pixels": matching_pixels,
            "total_pixels": len(pixels),
        }


def _compute_bar_fill_ratio(
    *,
    frame: LiveFrame,
    roi: dict[str, Any],
    matcher,
) -> tuple[float, int, int]:
    if frame.image is None or Image is None:
        return 0.0, 0, 0
    left = int(roi["x"])
    top = int(roi["y"])
    right = left + int(roi["width"])
    bottom = top + int(roi["height"])
    if right <= left or bottom <= top:
        return 0.0, 0, 0
    roi_image = frame.image.crop((left, top, right, bottom)).convert("RGB")
    roi_width, roi_height = roi_image.size
    if roi_width <= 0 or roi_height <= 0:
        return 0.0, 0, 0
    pixels = roi_image.load()
    matching_pixels = 0
    active_columns = 0
    required_matches_per_column = max(1, int(round(roi_height * 0.25)))
    for column_x in range(roi_width):
        column_matches = 0
        for row_y in range(roi_height):
            red_value, green_value, blue_value = pixels[column_x, row_y]
            if matcher(red_value, green_value, blue_value):
                column_matches += 1
        matching_pixels += column_matches
        if column_matches >= required_matches_per_column:
            active_columns += 1
    fill_ratio = active_columns / float(roi_width)
    return fill_ratio, matching_pixels, roi_width * roi_height


@dataclass(slots=True, frozen=True)
class LiveResourceReadAggregate:
    sample_count: int
    hp_ratio: float
    condition_ratio: float
    confidence: float
    warnings: tuple[str, ...]
    hp_median: float
    condition_median: float
    hp_spread: float
    condition_spread: float
    samples: tuple[LiveResourceSnapshot, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "hp_ratio": self.hp_ratio,
            "condition_ratio": self.condition_ratio,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "hp_median": self.hp_median,
            "condition_median": self.condition_median,
            "hp_spread": self.hp_spread,
            "condition_spread": self.condition_spread,
            "samples": [
                {
                    "hp_ratio": sample.hp_ratio,
                    "condition_ratio": sample.condition_ratio,
                    "metadata": sample.metadata,
                }
                for sample in self.samples
            ],
        }


def _resource_sample_confidence(sample: LiveResourceSnapshot) -> float:
    metadata_source = sample.metadata.get("source")
    if metadata_source != "pixel":
        return 0.35
    hp_meta = sample.metadata.get("hp", {})
    condition_meta = sample.metadata.get("condition", {})
    hp_fill = float(hp_meta.get("fill_ratio", sample.hp_ratio))
    condition_fill = float(condition_meta.get("fill_ratio", sample.condition_ratio))
    hp_match_ratio = _safe_ratio(
        float(hp_meta.get("matching_pixels", 0)),
        float(hp_meta.get("total_pixels", 1)),
    )
    condition_match_ratio = _safe_ratio(
        float(condition_meta.get("matching_pixels", 0)),
        float(condition_meta.get("total_pixels", 1)),
    )
    hp_confidence = max(0.0, min(1.0, (0.7 * hp_fill) + (0.3 * min(1.0, hp_match_ratio * 10.0))))
    condition_confidence = max(
        0.0,
        min(1.0, (0.7 * condition_fill) + (0.3 * min(1.0, condition_match_ratio * 10.0))),
    )
    return (hp_confidence + condition_confidence) / 2.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def build_world_snapshot(
    *,
    cycle_id: int,
    frame: LiveFrame,
    current_target_id: str | None,
    phase: str,
) -> WorldSnapshot:
    detections = parse_target_detections(frame)
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
                **detection.metadata,
            },
        )
        for detection in detections
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
        },
    )
