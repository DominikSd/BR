from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path
from random import Random
import sys
from typing import Any

from botlab.adapters.live.models import LiveFrame, LiveSessionState, LiveTargetDetection
from botlab.config import LiveConfig

try:
    from PIL import Image, ImageGrab
except Exception:  # pragma: no cover - optional dependency path
    Image = None
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


@dataclass(slots=True, frozen=True)
class WindowCaptureStatus:
    configured_window_title: str
    matched_window_hwnd: int | None
    matched_window_title: str | None
    foreground_window_title: str | None
    window_bbox: tuple[int, int, int, int] | None
    capture_bbox: tuple[int, int, int, int]
    foreground_matches: bool
    reliable: bool
    real_input_allowed: bool
    block_reason: str | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_window_title": self.configured_window_title,
            "matched_window_hwnd": self.matched_window_hwnd,
            "matched_window_title": self.matched_window_title,
            "foreground_window_title": self.foreground_window_title,
            "window_bbox": None if self.window_bbox is None else list(self.window_bbox),
            "capture_bbox": list(self.capture_bbox),
            "foreground_matches": self.foreground_matches,
            "reliable": self.reliable,
            "real_input_allowed": self.real_input_allowed,
            "block_reason": self.block_reason,
            "warning": self.warning,
        }


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
        window_guard = frame.metadata.get("window_guard")
        if isinstance(window_guard, dict):
            lines.append(
                '<rect x="8" y="8" width="520" height="76" fill="#0f172a" fill-opacity="0.80" stroke="#334155" stroke-width="1" />'
            )
            configured_title = window_guard.get("configured_window_title")
            matched_title = window_guard.get("matched_window_title")
            foreground_title = window_guard.get("foreground_window_title")
            capture_bbox = window_guard.get("capture_bbox")
            lines.append(
                f'<text x="16" y="28" fill="#f8fafc" font-size="14">window={configured_title} matched={matched_title}</text>'
            )
            lines.append(
                f'<text x="16" y="46" fill="#f8fafc" font-size="14">foreground={foreground_title} matches={window_guard.get("foreground_matches")} reliable={window_guard.get("reliable")}</text>'
            )
            lines.append(
                f'<text x="16" y="64" fill="#f8fafc" font-size="14">capture_bbox={capture_bbox} block_reason={window_guard.get("block_reason")}</text>'
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
        self._last_window_status: WindowCaptureStatus | None = None

    def real_input_guard_status(self) -> tuple[bool, str, dict[str, Any]]:
        status = self._last_window_status
        if status is None:
            return (
                False,
                "window_guard_no_capture_yet",
                {
                    "configured_window_title": self._live_config.window_title,
                },
            )
        return (
            status.real_input_allowed,
            "window_guard_ok" if status.real_input_allowed else str(status.block_reason or "window_guard_blocked"),
            status.to_dict(),
        )

    def capture_frame(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
        allow_background_capture: bool = False,
    ) -> LiveFrame:
        if ImageGrab is None:
            raise RuntimeError(
                "Brak Pillow/ImageGrab. Tryb live bez dry-run wymaga Pillow albo dalszej implementacji capture."
            )
        window_status = self._resolve_window_status()
        self._last_window_status = window_status
        preview_background_bypass = False
        if window_status.reliable:
            image = ImageGrab.grab(bbox=window_status.capture_bbox)
            source = "foreground_window_capture"
            capture_reliability = "trusted"
        elif (
            allow_background_capture
            and window_status.window_bbox is not None
            and window_status.block_reason != "window_minimized"
        ):
            image = None
            if window_status.matched_window_hwnd is not None:
                image = _grab_window_content(
                    hwnd=window_status.matched_window_hwnd,
                    window_bbox=window_status.window_bbox,
                    capture_bbox=window_status.capture_bbox,
                )
            if image is not None:
                source = "window_content_capture_preview_bypass"
                capture_reliability = "preview_background_bypass"
                preview_background_bypass = True
            else:
                image = ImageGrab.grab(bbox=window_status.capture_bbox)
                source = "foreground_window_capture_preview_bypass"
                capture_reliability = "preview_background_bypass_fallback"
                preview_background_bypass = True
        else:
            if Image is None:
                raise RuntimeError("Brak Pillow/Image. Window guard wymaga mozliwosci utworzenia placeholder frame.")
            blocked_width = max(1, int(window_status.capture_bbox[2] - window_status.capture_bbox[0]))
            blocked_height = max(1, int(window_status.capture_bbox[3] - window_status.capture_bbox[1]))
            image = Image.new("RGB", (blocked_width, blocked_height), color=(8, 8, 8))
            source = "foreground_window_capture_blocked"
            capture_reliability = "blocked"
        return LiveFrame(
            width=int(image.size[0]),
            height=int(image.size[1]),
            captured_at_ts=default_ts,
            source=source,
            metadata={
                "cycle_id": cycle_id,
                "phase": phase,
                "targets": [],
                "rest_available": True,
                "resource_fallback_enabled": False,
                "state_fallback_enabled": False,
                "window_guard": window_status.to_dict(),
                "capture_reliability": capture_reliability,
                "preview_background_bypass": preview_background_bypass,
            },
            image=image,
        )

    def _resolve_window_status(self) -> WindowCaptureStatus:
        configured_title = self._live_config.window_title
        fallback_bbox = (
            self._region.left,
            self._region.top,
            self._region.right,
            self._region.bottom,
        )
        if sys.platform != "win32":
            return WindowCaptureStatus(
                configured_window_title=configured_title,
                matched_window_hwnd=None,
                matched_window_title=None,
                foreground_window_title=None,
                window_bbox=None,
                capture_bbox=fallback_bbox,
                foreground_matches=not self._live_config.foreground_only,
                reliable=not self._live_config.foreground_only,
                real_input_allowed=False,
                block_reason="window_guard_non_windows",
                warning="window_guard_non_windows",
            )

        matched_window = _find_window_by_title(configured_title)
        foreground_window = _get_foreground_window_info()
        if matched_window is None:
            return WindowCaptureStatus(
                configured_window_title=configured_title,
                matched_window_hwnd=None,
                matched_window_title=None,
                foreground_window_title=None if foreground_window is None else foreground_window["title"],
                window_bbox=None,
                capture_bbox=fallback_bbox,
                foreground_matches=False,
                reliable=False,
                real_input_allowed=False,
                block_reason="window_not_found",
                warning="window_not_found",
            )

        window_bbox = matched_window["bbox"]
        capture_bbox = _calculate_window_relative_capture_bbox(window_bbox, self._region)
        if _window_looks_minimized_or_invalid(int(matched_window["hwnd"]), window_bbox):
            previous_capture_bbox = fallback_bbox
            if self._last_window_status is not None:
                previous_capture_bbox = self._last_window_status.capture_bbox
            return WindowCaptureStatus(
                configured_window_title=configured_title,
                matched_window_hwnd=int(matched_window["hwnd"]),
                matched_window_title=str(matched_window["title"]),
                foreground_window_title=None if foreground_window is None else str(foreground_window["title"]),
                window_bbox=window_bbox,
                capture_bbox=previous_capture_bbox,
                foreground_matches=False if foreground_window is None else bool(foreground_window["hwnd"] == matched_window["hwnd"]),
                reliable=False,
                real_input_allowed=False,
                block_reason="window_minimized",
                warning="window_minimized",
            )
        foreground_matches = False
        foreground_title: str | None = None
        if foreground_window is not None:
            foreground_matches = bool(foreground_window["hwnd"] == matched_window["hwnd"])
            foreground_title = str(foreground_window["title"])

        reliable = True
        block_reason: str | None = None
        warning: str | None = None
        if self._live_config.foreground_only and not foreground_matches:
            reliable = False
            block_reason = "foreground_window_mismatch"
            warning = "foreground_window_mismatch"

        return WindowCaptureStatus(
            configured_window_title=configured_title,
            matched_window_hwnd=int(matched_window["hwnd"]),
            matched_window_title=str(matched_window["title"]),
            foreground_window_title=foreground_title,
            window_bbox=window_bbox,
            capture_bbox=capture_bbox,
            foreground_matches=foreground_matches,
            reliable=reliable,
            real_input_allowed=reliable and foreground_matches,
            block_reason=block_reason,
            warning=warning,
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
        allow_background_capture: bool = False,
    ) -> LiveFrame:
        _ = allow_background_capture
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
        normalized_phase = _normalize_dry_run_phase(phase)
        profile_builders = {
            "single_spot_mvp": self._single_spot_mvp_payload,
            "engage_target_stolen": self._engage_target_stolen_payload,
            "engage_target_stolen_noisy": self._engage_target_stolen_noisy_payload,
            "engage_misclick": self._engage_misclick_payload,
            "engage_misclick_partial": self._engage_misclick_partial_payload,
            "engage_approach_stalled": self._engage_approach_stalled_payload,
            "engage_timeout": self._engage_timeout_payload,
        }
        payload_builder = profile_builders.get(self._live_config.dry_run_profile)
        if payload_builder is None:
            raise ValueError(
                f"Nieznany live dry-run profile '{self._live_config.dry_run_profile}'."
            )

        payload = payload_builder(
            cycle_id=cycle_id,
            phase=normalized_phase,
            default_ts=default_ts,
            session_state=session_state,
        )
        return {
            "cycle_id": cycle_id,
            "phase": normalized_phase,
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
            if phase == "engage_verify":
                return {
                    "in_combat": True,
                    "reward_visible": False,
                    "targets": _targets_to_metadata(
                        (
                            LiveTargetDetection(
                                "fallback-safe",
                                700,
                                310,
                                0.2,
                                occupied=False,
                                mob_variant="mob_a",
                            ),
                        )
                    ),
                    "perception_profile": {
                        "detection_duration_s": 0.009,
                        "selection_duration_s": 0.002,
                        "action_ready_duration_s": 0.001,
                    },
                }
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
        if phase == "engage_verify":
            return {
                "in_combat": True,
                "reward_visible": False,
                "targets": _targets_to_metadata(
                    (
                        LiveTargetDetection(
                            nearest.target_id,
                            nearest.screen_x,
                            nearest.screen_y,
                            0.2,
                            occupied=False,
                            mob_variant=nearest.mob_variant,
                        ),
                    )
                ),
                "perception_profile": {
                    "detection_duration_s": 0.008,
                    "selection_duration_s": 0.002,
                    "action_ready_duration_s": 0.001,
                },
            }
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

    def _engage_target_stolen_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        observation_targets = (
            LiveTargetDetection("occupied-near", 520, 280, 1.0, occupied=True, mob_variant="mob_a"),
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=False, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        stolen_targets = (
            LiveTargetDetection("occupied-near", 520, 280, 1.0, occupied=True, mob_variant="mob_a"),
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=True, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        return self._build_engage_profile_payload(
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
            observation_targets=observation_targets,
            approach_targets=observation_targets,
            interaction_targets=(
                LiveTargetDetection("front-free", 620, 300, 0.2, occupied=False, mob_variant="mob_b"),
            ),
            verify_targets=stolen_targets,
            verify_metadata={
                "in_combat": False,
                "reward_visible": False,
                "engage_result": "target_stolen",
            },
        )

    def _engage_misclick_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        observation_targets = (
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=False, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        verify_targets = (
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        return self._build_engage_profile_payload(
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
            observation_targets=observation_targets,
            approach_targets=observation_targets,
            interaction_targets=(
                LiveTargetDetection("front-free", 620, 300, 0.2, occupied=False, mob_variant="mob_b"),
            ),
            verify_targets=verify_targets,
            verify_metadata={
                "in_combat": False,
                "reward_visible": False,
                "engage_result": "misclick",
            },
        )

    def _engage_target_stolen_noisy_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        observation_targets = (
            LiveTargetDetection("occupied-near", 520, 280, 1.0, occupied=True, mob_variant="mob_a"),
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=False, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        verify_targets = (
            LiveTargetDetection("front-free", 624, 304, 1.9, occupied=True, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 758, 320, 3.7, occupied=False, mob_variant="mob_a"),
            LiveTargetDetection("noise-side", 882, 346, 5.2, occupied=False, mob_variant="mob_b"),
        )
        return self._build_engage_profile_payload(
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
            observation_targets=observation_targets,
            approach_targets=observation_targets,
            interaction_targets=(
                LiveTargetDetection("front-free", 620, 300, 0.2, occupied=False, mob_variant="mob_b"),
            ),
            verify_targets=verify_targets,
            verify_metadata={
                "in_combat": False,
                "reward_visible": False,
                "engage_result": "target_stolen",
            },
        )

    def _engage_misclick_partial_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        observation_targets = (
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=False, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        verify_targets = (
            LiveTargetDetection("safe-far", 764, 321, 3.9, occupied=False, mob_variant="mob_a"),
            LiveTargetDetection("far-noise", 930, 270, 6.2, occupied=False, mob_variant="mob_b"),
        )
        return self._build_engage_profile_payload(
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
            observation_targets=observation_targets,
            approach_targets=observation_targets,
            interaction_targets=(
                LiveTargetDetection("front-free", 620, 300, 0.2, occupied=False, mob_variant="mob_b"),
            ),
            verify_targets=verify_targets,
            verify_metadata={
                "in_combat": False,
                "reward_visible": False,
                "engage_result": "misclick",
            },
        )

    def _engage_timeout_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        observation_targets = (
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=False, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        return self._build_engage_profile_payload(
            phase=phase,
            default_ts=default_ts,
            session_state=session_state,
            observation_targets=observation_targets,
            approach_targets=observation_targets,
            interaction_targets=(
                LiveTargetDetection("front-free", 620, 300, 0.2, occupied=False, mob_variant="mob_b"),
            ),
            verify_targets=(),
            verify_metadata={
                "in_combat": False,
                "reward_visible": False,
                "verify_timeout": True,
                "approach_timeout": True,
                "engage_result": "approach_timeout",
            },
        )

    def _engage_approach_stalled_payload(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
    ) -> dict[str, Any]:
        observation_targets = (
            LiveTargetDetection("front-free", 620, 300, 1.8, occupied=False, mob_variant="mob_b"),
            LiveTargetDetection("safe-far", 760, 318, 3.8, occupied=False, mob_variant="mob_a"),
        )
        if phase == "observation":
            return {
                "targets": _targets_to_metadata(observation_targets),
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
                "stall_after_s": 1.0,
                "last_progress_ts": default_ts,
                "targets": _targets_to_metadata(observation_targets),
            }
        if phase == "approach_revalidation":
            return {
                "targets": _targets_to_metadata(observation_targets),
                "hp_ratio": session_state.hp_ratio,
                "condition_ratio": session_state.condition_ratio,
            }
        if phase == "interaction":
            return {
                "targets": _targets_to_metadata(()),
                "interaction_ready": False,
                "in_combat": False,
            }
        if phase == "verify":
            return {"in_combat": False, "reward_visible": False}
        if phase == "engage_verify":
            return {
                "targets": [],
                "in_combat": False,
                "reward_visible": False,
                "engage_result": "approach_stalled",
            }
        return {
            "targets": [],
            "hp_ratio": session_state.hp_ratio,
            "condition_ratio": session_state.condition_ratio,
            "in_combat": False,
            "reward_visible": False,
            "rest_available": True,
        }

    def _build_engage_profile_payload(
        self,
        *,
        phase: str,
        default_ts: float,
        session_state: LiveSessionState,
        observation_targets: tuple[LiveTargetDetection, ...],
        approach_targets: tuple[LiveTargetDetection, ...],
        interaction_targets: tuple[LiveTargetDetection, ...],
        verify_targets: tuple[LiveTargetDetection, ...],
        verify_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if phase == "observation":
            return {
                "targets": _targets_to_metadata(observation_targets),
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
                "last_progress_ts": default_ts + 0.20,
                "targets": _targets_to_metadata(observation_targets),
            }
        if phase == "approach_revalidation":
            return {
                "targets": _targets_to_metadata(approach_targets),
                "perception_profile": {
                    "detection_duration_s": 0.009,
                    "selection_duration_s": 0.003,
                    "action_ready_duration_s": 0.001,
                },
                "hp_ratio": session_state.hp_ratio,
                "condition_ratio": session_state.condition_ratio,
            }
        if phase == "interaction":
            return {
                "targets": _targets_to_metadata(interaction_targets),
                "interaction_ready": True,
                "in_combat": False,
            }
        if phase == "verify":
            return {"in_combat": False, "reward_visible": False}
        if phase == "engage_verify":
            return {
                "targets": _targets_to_metadata(verify_targets),
                "perception_profile": {
                    "detection_duration_s": 0.008,
                    "selection_duration_s": 0.002,
                    "action_ready_duration_s": 0.001,
                },
                **verify_metadata,
            }
        return {
            "targets": [],
            "hp_ratio": session_state.hp_ratio,
            "condition_ratio": session_state.condition_ratio,
            "in_combat": False,
            "reward_visible": False,
            "rest_available": True,
        }


def create_capture(live_config: LiveConfig):
    return DryRunWindowCapture(live_config) if live_config.dry_run else ForegroundWindowCapture(live_config)


def _normalize_dry_run_phase(phase: str) -> str:
    if phase.startswith("rest_sample_"):
        return "rest"
    return phase


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


def _calculate_window_relative_capture_bbox(
    window_bbox: tuple[int, int, int, int],
    capture_region: CaptureRegion,
) -> tuple[int, int, int, int]:
    window_left, window_top, window_right, window_bottom = window_bbox
    window_width = max(1, window_right - window_left)
    window_height = max(1, window_bottom - window_top)
    if capture_region.width <= 0 or capture_region.height <= 0:
        return (
            int(window_left),
            int(window_top),
            int(window_right),
            int(window_bottom),
        )
    left = window_left + capture_region.left
    top = window_top + capture_region.top
    width = min(capture_region.width, max(1, window_width - capture_region.left))
    height = min(capture_region.height, max(1, window_height - capture_region.top))
    return (
        int(left),
        int(top),
        int(left + width),
        int(top + height),
    )


def _find_window_by_title(window_title: str) -> dict[str, Any] | None:
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    results: list[dict[str, Any]] = []
    enum_windows_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_title(hwnd)
        if not title:
            return True
        if window_title.lower() not in title.lower():
            return True
        bbox = _get_window_rect(hwnd)
        if bbox is None:
            return True
        results.append(
            {
                "hwnd": int(hwnd),
                "title": title,
                "bbox": bbox,
            }
        )
        return True

    user32.EnumWindows(enum_windows_proc(callback), 0)
    if not results:
        return None
    return results[0]


def _get_foreground_window_info() -> dict[str, Any] | None:
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    title = _get_window_title(hwnd)
    bbox = _get_window_rect(hwnd)
    return {
        "hwnd": int(hwnd),
        "title": title,
        "bbox": bbox,
    }


def _get_window_title(hwnd: int) -> str:
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return str(buffer.value)


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    user32 = ctypes.windll.user32
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _is_window_iconic(hwnd: int) -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    return bool(user32.IsIconic(hwnd))


def _window_bbox_looks_minimized(bbox: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = bbox
    width = max(0, right - left)
    height = max(0, bottom - top)
    if left <= -32000 and top <= -32000:
        return True
    if width <= 200 and height <= 80:
        return True
    return False


def _window_looks_minimized_or_invalid(hwnd: int, bbox: tuple[int, int, int, int]) -> bool:
    return _is_window_iconic(hwnd) or _window_bbox_looks_minimized(bbox)


def _grab_window_content(
    *,
    hwnd: int,
    window_bbox: tuple[int, int, int, int],
    capture_bbox: tuple[int, int, int, int],
):
    if sys.platform != "win32" or Image is None:
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    window_left, window_top, window_right, window_bottom = window_bbox
    width = max(1, int(window_right - window_left))
    height = max(1, int(window_bottom - window_top))

    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        return None
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    if not mem_dc:
        user32.ReleaseDC(hwnd, hwnd_dc)
        return None
    bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
    if not bitmap:
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)
        return None

    old_object = gdi32.SelectObject(mem_dc, bitmap)
    try:
        print_success = bool(user32.PrintWindow(hwnd, mem_dc, 0))
        if not print_success:
            print_success = bool(user32.PrintWindow(hwnd, mem_dc, 2))
        if not print_success:
            return None

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.wintypes.DWORD),
                ("biWidth", ctypes.wintypes.LONG),
                ("biHeight", ctypes.wintypes.LONG),
                ("biPlanes", ctypes.wintypes.WORD),
                ("biBitCount", ctypes.wintypes.WORD),
                ("biCompression", ctypes.wintypes.DWORD),
                ("biSizeImage", ctypes.wintypes.DWORD),
                ("biXPelsPerMeter", ctypes.wintypes.LONG),
                ("biYPelsPerMeter", ctypes.wintypes.LONG),
                ("biClrUsed", ctypes.wintypes.DWORD),
                ("biClrImportant", ctypes.wintypes.DWORD),
            ]

        class RGBQUAD(ctypes.Structure):
            _fields_ = [
                ("rgbBlue", ctypes.c_ubyte),
                ("rgbGreen", ctypes.c_ubyte),
                ("rgbRed", ctypes.c_ubyte),
                ("rgbReserved", ctypes.c_ubyte),
            ]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [
                ("bmiHeader", BITMAPINFOHEADER),
                ("bmiColors", RGBQUAD * 1),
            ]

        bitmap_info = BITMAPINFO()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = width
        bitmap_info.bmiHeader.biHeight = -height
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = 0

        buffer = ctypes.create_string_buffer(width * height * 4)
        dib_rows = gdi32.GetDIBits(
            mem_dc,
            bitmap,
            0,
            height,
            buffer,
            ctypes.byref(bitmap_info),
            0,
        )
        if dib_rows != height:
            return None

        image = Image.frombuffer(
            "RGB",
            (width, height),
            buffer,
            "raw",
            "BGRX",
            0,
            1,
        )
        crop_left = max(0, int(capture_bbox[0] - window_left))
        crop_top = max(0, int(capture_bbox[1] - window_top))
        crop_right = min(width, int(capture_bbox[2] - window_left))
        crop_bottom = min(height, int(capture_bbox[3] - window_top))
        if crop_right <= crop_left or crop_bottom <= crop_top:
            return image
        return image.crop((crop_left, crop_top, crop_right, crop_bottom))
    finally:
        gdi32.SelectObject(mem_dc, old_object)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(hwnd, hwnd_dc)
