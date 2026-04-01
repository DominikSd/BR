from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from botlab.adapters.live.models import (
    LiveFrame,
    LiveResourceSnapshot,
    LiveStateSnapshot,
    LiveTargetDetection,
)
from botlab.config import CombatConfig, LiveConfig
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot


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
        extract_named_roi(frame, roi_name="hp_bar_roi", live_config=self._live_config)
        extract_named_roi(frame, roi_name="condition_bar_roi", live_config=self._live_config)
        return LiveResourceSnapshot(
            hp_ratio=float(frame.metadata.get("hp_ratio", 1.0)),
            condition_ratio=float(frame.metadata.get("condition_ratio", 1.0)),
        )


class SimpleStateDetector:
    def __init__(self, template_matcher: SimpleTemplateMatcher | None = None) -> None:
        self._template_matcher = template_matcher or SimpleTemplateMatcher()

    def detect_state(self, frame: LiveFrame) -> LiveStateSnapshot:
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
        rest_available = bool(frame.metadata.get("rest_available", True))
        return LiveStateSnapshot(
            in_combat=in_combat,
            reward_visible=reward_visible,
            rest_available=rest_available,
        )


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
