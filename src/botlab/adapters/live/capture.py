from __future__ import annotations

import json
from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path
from random import Random
from typing import Any

from botlab.adapters.live.models import LiveFrame, LiveSessionState, LiveTargetDetection
from botlab.config import LiveConfig

try:
    from PIL import ImageGrab
except Exception:  # pragma: no cover - optional dependency path
    ImageGrab = None


@dataclass(slots=True, frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class DebugArtifactWriter:
    def __init__(self, live_config: LiveConfig) -> None:
        self._live_config = live_config
        self._debug_directory = live_config.debug_directory

    def write_frame(
        self,
        *,
        cycle_id: int,
        phase: str,
        frame: LiveFrame,
    ) -> dict[str, Path]:
        if not self._live_config.save_frames and not self._live_config.save_overlays:
            return {}

        cycle_directory = self._debug_directory / f"cycle_{cycle_id:03d}"
        cycle_directory.mkdir(parents=True, exist_ok=True)

        artifact_paths: dict[str, Path] = {}
        if self._live_config.save_frames:
            metadata_path = cycle_directory / f"{phase}_frame.json"
            payload = {
                "width": frame.width,
                "height": frame.height,
                "captured_at_ts": frame.captured_at_ts,
                "source": frame.source,
                "metadata": frame.metadata,
            }
            metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            artifact_paths["frame_json"] = metadata_path

        if self._live_config.save_overlays:
            overlay_path = cycle_directory / f"{phase}_overlay.svg"
            overlay_path.write_text(
                self._build_overlay_svg(frame=frame),
                encoding="utf-8",
            )
            artifact_paths["overlay_svg"] = overlay_path

        return artifact_paths

    def _build_overlay_svg(self, *, frame: LiveFrame) -> str:
        targets = _targets_from_metadata(frame.metadata.get("targets", []))
        roi_items = [
            ("spawn_roi", self._live_config.spawn_roi),
            ("hp_bar_roi", self._live_config.hp_bar_roi),
            ("condition_bar_roi", self._live_config.condition_bar_roi),
            ("combat_indicator_roi", self._live_config.combat_indicator_roi),
            ("reward_roi", self._live_config.reward_roi),
        ]
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{frame.width}" height="{frame.height}" viewBox="0 0 {frame.width} {frame.height}">',
            '<rect width="100%" height="100%" fill="#111827" />',
        ]
        for roi_name, roi in roi_items:
            lines.append(
                f'<rect x="{roi[0]}" y="{roi[1]}" width="{roi[2]}" height="{roi[3]}" fill="none" stroke="#60a5fa" stroke-width="2" />'
            )
            lines.append(
                f'<text x="{roi[0] + 4}" y="{roi[1] + 16}" fill="#93c5fd" font-size="14">{roi_name}</text>'
            )
        for target in targets:
            color = "#ef4444" if target.occupied else "#22c55e"
            lines.append(
                f'<circle cx="{target.screen_x}" cy="{target.screen_y}" r="12" fill="none" stroke="{color}" stroke-width="3" />'
            )
            if target.occupied:
                lines.append(
                    f'<line x1="{target.screen_x - 10}" y1="{target.screen_y - 10}" x2="{target.screen_x + 10}" y2="{target.screen_y + 10}" stroke="#f97316" stroke-width="3" />'
                )
                lines.append(
                    f'<line x1="{target.screen_x + 10}" y1="{target.screen_y - 10}" x2="{target.screen_x - 10}" y2="{target.screen_y + 10}" stroke="#f97316" stroke-width="3" />'
                )
            lines.append(
                f'<text x="{target.screen_x + 16}" y="{target.screen_y - 6}" fill="#f9fafb" font-size="14">{target.target_id}</text>'
            )
        lines.append("</svg>")
        return "\n".join(lines)


class ForegroundWindowCapture:
    def __init__(self, live_config: LiveConfig) -> None:
        self._live_config = live_config
        self._region = CaptureRegion(*live_config.capture_region)

    def capture_frame(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> LiveFrame:
        if ImageGrab is None:
            raise RuntimeError(
                "Brak Pillow/ImageGrab. Tryb live bez dry-run wymaga Pillow albo dalszej implementacji capture."
            )

        image = ImageGrab.grab(
            bbox=(
                self._region.left,
                self._region.top,
                self._region.right,
                self._region.bottom,
            )
        )
        return LiveFrame(
            width=self._region.width,
            height=self._region.height,
            captured_at_ts=default_ts,
            source="foreground_window_capture",
            metadata={
                "cycle_id": cycle_id,
                "phase": phase,
                "hp_ratio": session_state.hp_ratio,
                "condition_ratio": session_state.condition_ratio,
                "targets": [],
                "in_combat": False,
                "reward_visible": False,
                "rest_available": True,
            },
            image=image,
        )


class DryRunWindowCapture:
    def __init__(self, live_config: LiveConfig) -> None:
        self._live_config = live_config
        self._region = CaptureRegion(*live_config.capture_region)

    def capture_frame(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> LiveFrame:
        metadata = self._build_frame_metadata(
            cycle_id=cycle_id,
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
        )
        return LiveFrame(
            width=self._region.width,
            height=self._region.height,
            captured_at_ts=default_ts,
            source=f"dry_run:{self._live_config.dry_run_profile}",
            metadata=metadata,
            image=None,
        )

    def _build_frame_metadata(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        if self._live_config.dry_run_profile != "single_spot_mvp":
            raise ValueError(
                f"Nieznany live dry-run profile '{self._live_config.dry_run_profile}'."
            )

        payload = self._single_spot_mvp_payload(
            cycle_id=cycle_id,
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
        )
        return {
            "cycle_id": cycle_id,
            "phase": phase,
            **payload,
        }

    def _single_spot_mvp_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        if cycle_id == 1:
            observation_targets = (
                LiveTargetDetection("occupied-near", 520, 280, 1.0, occupied=True, mob_variant="mob_a"),
                LiveTargetDetection(
                    "front-free",
                    620,
                    300,
                    2.4,
                    occupied=False,
                    mob_variant="mob_b",
                    confidence=0.94,
                    orientation_deg=90,
                    metadata={
                        "duplicate_orientations": [
                            {
                                "label": "mob_b",
                                "x": 598,
                                "y": 273,
                                "width": 42,
                                "height": 54,
                                "confidence": 0.88,
                                "rotation_deg": 180,
                            }
                        ]
                    },
                ),
                LiveTargetDetection(
                    "fallback-safe",
                    760,
                    318,
                    4.0,
                    occupied=False,
                    mob_variant="mob_a",
                    confidence=0.91,
                    orientation_deg=270,
                ),
            )
            approach_targets = (
                LiveTargetDetection("occupied-near", 520, 280, 1.0, occupied=True, mob_variant="mob_a"),
                LiveTargetDetection(
                    "front-free",
                    620,
                    300,
                    2.4,
                    occupied=True,
                    mob_variant="mob_b",
                    confidence=0.93,
                    orientation_deg=90,
                ),
                LiveTargetDetection(
                    "fallback-safe",
                    700,
                    310,
                    3.5,
                    occupied=False,
                    mob_variant="mob_a",
                    confidence=0.92,
                    orientation_deg=180,
                ),
            )
            if phase == "observation":
                return {
                    "targets": _targets_to_metadata(observation_targets),
                    "perception_profile": {
                        "detection_duration_s": 0.012,
                        "selection_duration_s": 0.004,
                        "action_ready_duration_s": 0.002,
                    },
                    "hp_ratio": session_state.hp_ratio,
                    "condition_ratio": session_state.condition_ratio,
                    "in_combat": False,
                    "reward_visible": False,
                    "rest_available": True,
                }
            if phase == "approach":
                return {
                    "stall_after_s": None,
                    "last_progress_ts": default_ts + 0.45,
                    "targets": _targets_to_metadata(observation_targets),
                }
            if phase == "approach_revalidation":
                return {
                    "targets": _targets_to_metadata(approach_targets),
                    "perception_profile": {
                        "detection_duration_s": 0.010,
                        "selection_duration_s": 0.003,
                        "action_ready_duration_s": 0.001,
                    },
                    "hp_ratio": session_state.hp_ratio,
                    "condition_ratio": session_state.condition_ratio,
                }
            if phase == "interaction":
                return {
                    "targets": _targets_to_metadata(
                        (
                            LiveTargetDetection("fallback-safe", 700, 310, 0.3, occupied=False, mob_variant="mob_a"),
                        )
                    ),
                    "interaction_ready": True,
                    "in_combat": False,
                }
            if phase == "verify":
                return {"in_combat": True, "reward_visible": False}
            if phase == "combat":
                return {
                    "combat_turns": 4,
                    "hp_ratio": 0.42,
                    "condition_ratio": 0.38,
                    "reward_visible": True,
                    "reward_duration_s": 0.4,
                }
            if phase == "rest":
                return {
                    "rest_tick_count": 3,
                    "hp_ratio": 1.0,
                    "condition_ratio": 0.98,
                    "rest_available": True,
                }

        generated_targets = _generate_circle_targets(
            cycle_id=cycle_id,
            center_xy=(700, 300),
            distance_center=3.0,
            distance_radius=2.5,
            count=3,
            seed=42,
        )
        nearest = min(generated_targets, key=lambda item: (item.distance, item.target_id))
        if phase == "observation":
            return {
                "targets": _targets_to_metadata(generated_targets),
                "perception_profile": {
                    "detection_duration_s": 0.011,
                    "selection_duration_s": 0.003,
                    "action_ready_duration_s": 0.002,
                },
                "hp_ratio": session_state.hp_ratio,
                "condition_ratio": session_state.condition_ratio,
                "in_combat": False,
                "reward_visible": False,
                "rest_available": True,
            }
        if phase == "approach":
            return {
                "stall_after_s": None,
                "last_progress_ts": default_ts + 0.12,
                "targets": _targets_to_metadata(generated_targets),
            }
        if phase == "approach_revalidation":
            return {
                "targets": _targets_to_metadata(generated_targets),
                "hp_ratio": session_state.hp_ratio,
                "condition_ratio": session_state.condition_ratio,
            }
        if phase == "interaction":
            return {
                "targets": _targets_to_metadata(
                    (
                        LiveTargetDetection(
                            nearest.target_id,
                            nearest.screen_x,
                            nearest.screen_y,
                            0.25,
                            occupied=False,
                            mob_variant=nearest.mob_variant,
                        ),
                    )
                ),
                "interaction_ready": True,
                "in_combat": False,
            }
        if phase == "verify":
            return {"in_combat": True, "reward_visible": False}
        if phase == "combat":
            return {
                "combat_turns": 4,
                "hp_ratio": 0.88,
                "condition_ratio": 0.86,
                "reward_visible": True,
                "reward_duration_s": 0.3,
            }
        if phase == "rest":
            return {
                "rest_tick_count": 0,
                "hp_ratio": session_state.hp_ratio,
                "condition_ratio": session_state.condition_ratio,
                "rest_available": True,
            }

        return {
            "targets": [],
            "hp_ratio": session_state.hp_ratio,
            "condition_ratio": session_state.condition_ratio,
            "in_combat": False,
            "reward_visible": False,
            "rest_available": True,
        }


def _targets_to_metadata(targets: tuple[LiveTargetDetection, ...]) -> list[dict[str, Any]]:
    return [
        {
            "target_id": target.target_id,
            "screen_x": target.screen_x,
            "screen_y": target.screen_y,
            "distance": target.distance,
            "occupied": target.occupied,
            "mob_variant": target.mob_variant,
            "reachable": target.reachable,
            "confidence": target.confidence,
            "orientation_deg": target.orientation_deg,
            "bbox_width": None if target.bbox is None else target.bbox[2],
            "bbox_height": None if target.bbox is None else target.bbox[3],
            "metadata": dict(target.metadata),
        }
        for target in targets
    ]


def _targets_from_metadata(raw_targets: object) -> tuple[LiveTargetDetection, ...]:
    if not isinstance(raw_targets, list):
        return ()
    parsed_targets: list[LiveTargetDetection] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        raw_bbox_width = raw_target.get("bbox_width", 40)
        raw_bbox_height = raw_target.get("bbox_height", 52)
        bbox_width = 40 if raw_bbox_width is None else int(raw_bbox_width)
        bbox_height = 52 if raw_bbox_height is None else int(raw_bbox_height)
        parsed_targets.append(
            LiveTargetDetection(
                target_id=str(raw_target.get("target_id", "unknown")),
                screen_x=int(raw_target.get("screen_x", 0)),
                screen_y=int(raw_target.get("screen_y", 0)),
                distance=float(raw_target.get("distance", 0.0)),
                occupied=bool(raw_target.get("occupied", False)),
                mob_variant=str(raw_target.get("mob_variant", "mob_a")),
                reachable=bool(raw_target.get("reachable", True)),
                confidence=float(raw_target.get("confidence", 1.0)),
                bbox=(
                    int(raw_target.get("screen_x", 0)) - (bbox_width // 2),
                    int(raw_target.get("screen_y", 0)) - (bbox_height // 2),
                    bbox_width,
                    bbox_height,
                ),
                orientation_deg=int(raw_target.get("orientation_deg", 0)),
                metadata=dict(raw_target.get("metadata", {})),
            )
        )
    return tuple(parsed_targets)


def _generate_circle_targets(
    *,
    cycle_id: int,
    center_xy: tuple[int, int],
    distance_center: float,
    distance_radius: float,
    count: int,
    seed: int,
) -> tuple[LiveTargetDetection, ...]:
    rng = Random(f"{seed}:{cycle_id}")
    variants = ("mob_a", "mob_b")
    center_x, center_y = center_xy
    targets: list[LiveTargetDetection] = []
    for index in range(1, count + 1):
        angle = rng.uniform(0.0, 2.0 * pi)
        radius = distance_radius * sqrt(rng.random())
        distance = round(distance_center + radius, 6)
        x = int(round(center_x + (radius * 40.0 * cos(angle))))
        y = int(round(center_y + (radius * 28.0 * sin(angle))))
        targets.append(
            LiveTargetDetection(
                target_id=f"live-spawn-{cycle_id}-{index}",
                screen_x=x,
                screen_y=y,
                distance=distance,
                occupied=False,
                mob_variant=variants[(index - 1) % len(variants)],
            )
        )
    return tuple(targets)
