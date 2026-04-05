from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from botlab.adapters.live.models import LiveEngageOutcome, LiveEngageResult, LiveTargetDetection
from botlab.adapters.live.perception import LatencyAggregate, PerceptionFrameResult
from botlab.adapters.live.vision import SimpleStateDetector
from botlab.application import TargetEngagementResult, TargetEngagementService
from botlab.config import LiveConfig
from botlab.types import BotState, Observation, TelemetryRecord

if TYPE_CHECKING:
    from botlab.adapters.live.input import LiveInputDriver, LiveInputEvent
    from botlab.adapters.live.runner import LiveRuntime
    from botlab.adapters.telemetry.storage import SQLiteTelemetryStorage


@dataclass(slots=True, frozen=True)
class LiveEngageRunReport:
    results: tuple[LiveEngageResult, ...]
    summary: "LiveEngageSessionSummary"
    log_path: Path
    sqlite_path: Path


@dataclass(slots=True, frozen=True)
class LiveEngageSessionSummary:
    total_attempts: int
    engaged_count: int
    target_stolen_count: int
    misclick_count: int
    approach_stalled_count: int
    approach_timeout_count: int
    no_target_available_count: int
    engage_quality_gate_rejection_count: int
    occupied_rejection_count: int
    out_of_zone_rejection_count: int
    detection_latency: LatencyAggregate
    selection_latency: LatencyAggregate
    total_reaction_latency: LatencyAggregate
    verification_latency: LatencyAggregate
    calibration_warning_count: int = 0
    right_click_action_count: int = 0
    key_press_action_count: int = 0
    key_sequence_action_count: int = 0
    real_input_action_count: int = 0

    @classmethod
    def from_results(cls, results: Iterable[LiveEngageResult]) -> "LiveEngageSessionSummary":
        parsed_results = tuple(results)
        return cls(
            total_attempts=len(parsed_results),
            engaged_count=sum(1 for result in parsed_results if result.outcome is LiveEngageOutcome.ENGAGED),
            target_stolen_count=sum(
                1 for result in parsed_results if result.outcome is LiveEngageOutcome.TARGET_STOLEN
            ),
            misclick_count=sum(1 for result in parsed_results if result.outcome is LiveEngageOutcome.MISCLICK),
            approach_stalled_count=sum(
                1 for result in parsed_results if result.outcome is LiveEngageOutcome.APPROACH_STALLED
            ),
            approach_timeout_count=sum(
                1 for result in parsed_results if result.outcome is LiveEngageOutcome.APPROACH_TIMEOUT
            ),
            no_target_available_count=sum(
                1 for result in parsed_results if result.outcome is LiveEngageOutcome.NO_TARGET_AVAILABLE
            ),
            engage_quality_gate_rejection_count=sum(
                int(result.metadata.get("engage_quality_gate_rejection_count", 0))
                for result in parsed_results
            ),
            occupied_rejection_count=sum(
                int(result.metadata.get("occupied_rejection_count", 0))
                for result in parsed_results
            ),
            out_of_zone_rejection_count=sum(
                int(result.metadata.get("out_of_zone_rejection_count", 0))
                for result in parsed_results
            ),
            calibration_warning_count=sum(
                1
                for result in parsed_results
                if bool(result.metadata.get("scene_calibration_warning"))
            ),
            right_click_action_count=sum(
                int(result.metadata.get("right_click_action_count", 0))
                for result in parsed_results
            ),
            key_press_action_count=sum(
                int(result.metadata.get("key_press_action_count", 0))
                for result in parsed_results
            ),
            key_sequence_action_count=sum(
                int(result.metadata.get("key_sequence_action_count", 0))
                for result in parsed_results
            ),
            real_input_action_count=sum(
                int(result.metadata.get("real_input_action_count", 0))
                for result in parsed_results
            ),
            detection_latency=LatencyAggregate.from_values(
                "engage_detection_latency_ms",
                (
                    result.detection_latency_ms
                    for result in parsed_results
                    if result.detection_latency_ms is not None
                ),
            ),
            selection_latency=LatencyAggregate.from_values(
                "engage_selection_latency_ms",
                (
                    result.selection_latency_ms
                    for result in parsed_results
                    if result.selection_latency_ms is not None
                ),
            ),
            total_reaction_latency=LatencyAggregate.from_values(
                "engage_total_reaction_latency_ms",
                (
                    result.total_reaction_latency_ms
                    for result in parsed_results
                    if result.total_reaction_latency_ms is not None
                ),
            ),
            verification_latency=LatencyAggregate.from_values(
                "engage_verification_latency_ms",
                (
                    result.verification_latency_ms
                    for result in parsed_results
                    if result.verification_latency_ms is not None
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "engaged_count": self.engaged_count,
            "target_stolen_count": self.target_stolen_count,
            "misclick_count": self.misclick_count,
            "approach_stalled_count": self.approach_stalled_count,
            "approach_timeout_count": self.approach_timeout_count,
            "no_target_available_count": self.no_target_available_count,
            "engage_quality_gate_rejection_count": self.engage_quality_gate_rejection_count,
            "occupied_rejection_count": self.occupied_rejection_count,
            "out_of_zone_rejection_count": self.out_of_zone_rejection_count,
            "calibration_warning_count": self.calibration_warning_count,
            "right_click_action_count": self.right_click_action_count,
            "key_press_action_count": self.key_press_action_count,
            "key_sequence_action_count": self.key_sequence_action_count,
            "real_input_action_count": self.real_input_action_count,
            "engage_success_rate": 0.0
            if self.total_attempts == 0
            else self.engaged_count / float(self.total_attempts),
            "detection_latency": self.detection_latency.to_dict(),
            "selection_latency": self.selection_latency.to_dict(),
            "total_reaction_latency": self.total_reaction_latency.to_dict(),
            "verification_latency": self.verification_latency.to_dict(),
        }


class LiveEngageArtifactWriter:
    def __init__(self, output_directory: Path, *, live_config: LiveConfig) -> None:
        self._output_directory = output_directory / "engage"
        self._live_config = live_config

    def write_result(
        self,
        *,
        result: LiveEngageResult,
        observation_result: PerceptionFrameResult | None,
        verification_result: PerceptionFrameResult | None,
        input_events: tuple["LiveInputEvent", ...],
    ) -> dict[str, Path]:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        directory = self._output_directory / f"engage_{result.cycle_id:03d}"
        directory.mkdir(parents=True, exist_ok=True)

        artifact_paths: dict[str, Path] = {}
        result_path = directory / "engage_result.json"
        result_payload = {
            **result.to_dict(),
            "observation_perception": None if observation_result is None else observation_result.to_dict(),
            "verification_perception": None if verification_result is None else verification_result.to_dict(),
            "input_events": [
                {
                    "action": event.action,
                    "payload": event.payload,
                }
                for event in input_events
            ],
        }
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifact_paths["engage_result_json"] = result_path

        overlay_path = directory / "engage_overlay.svg"
        overlay_path.write_text(
            self._build_overlay_svg(
                result=result,
                observation_result=observation_result,
                verification_result=verification_result,
            ),
            encoding="utf-8",
        )
        artifact_paths["engage_overlay_svg"] = overlay_path

        jsonl_path = self._output_directory / "engage_results.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "cycle_id": result.cycle_id,
                        **result.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
        artifact_paths["engage_results_jsonl"] = jsonl_path
        return artifact_paths

    def write_session_summary(self, summary: LiveEngageSessionSummary) -> Path:
        self._output_directory.mkdir(parents=True, exist_ok=True)
        summary_path = self._output_directory / "engage_session_summary.json"
        summary_path.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary_path

    def _build_overlay_svg(
        self,
        *,
        result: LiveEngageResult,
        observation_result: PerceptionFrameResult | None,
        verification_result: PerceptionFrameResult | None,
    ) -> str:
        base_result = observation_result or verification_result
        width = 1280 if base_result is None else base_result.frame_width
        height = 720 if base_result is None else base_result.frame_height
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#111827" />',
        ]
        if observation_result is not None:
            roi = observation_result.roi
            lines.append(
                f'<rect x="{roi["x"]}" y="{roi["y"]}" width="{roi["width"]}" height="{roi["height"]}" fill="none" stroke="#60a5fa" stroke-width="2" />'
            )
            if observation_result.scene_zone_polygon:
                polygon_points = " ".join(
                    f"{point_x},{point_y}" for point_x, point_y in observation_result.scene_zone_polygon
                )
                lines.append(
                    f'<polygon points="{polygon_points}" fill="#22c55e" fill-opacity="0.08" stroke="#22c55e" stroke-width="3" />'
                )
            if observation_result.scene_calibration.get("warning"):
                lines.append(
                    f'<text x="16" y="48" fill="#f59e0b" font-size="16">scene_calibration_warning={observation_result.scene_calibration["warning"]}</text>'
                )
            for detection in observation_result.detections:
                bbox = detection.bbox or (detection.screen_x - 18, detection.screen_y - 24, 36, 48)
                in_scene_zone = bool(detection.metadata.get("in_scene_zone", True))
                color = "#ef4444" if detection.occupied else "#22c55e"
                if not in_scene_zone:
                    color = "#9ca3af"
                stroke_width = 4 if detection.target_id == result.selected_target_id else 2
                lines.append(
                    f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}" fill="none" stroke="{color}" stroke-width="{stroke_width}" />'
                )
                lines.append(
                    f'<text x="{bbox[0]}" y="{max(14, bbox[1] - 4)}" fill="#f9fafb" font-size="14">{detection.target_id} {"occupied" if detection.occupied else "free"}{" / out_of_zone" if not in_scene_zone else ""}</text>'
                )
        if result.click_screen_xy is not None:
            click_x, click_y = result.click_screen_xy
            lines.append(
                f'<line x1="{click_x - 12}" y1="{click_y}" x2="{click_x + 12}" y2="{click_y}" stroke="#fde047" stroke-width="3" />'
            )
            lines.append(
                f'<line x1="{click_x}" y1="{click_y - 12}" x2="{click_x}" y2="{click_y + 12}" stroke="#fde047" stroke-width="3" />'
            )
            lines.append(
                f'<text x="{click_x + 16}" y="{max(16, click_y - 8)}" fill="#fde047" font-size="14">click</text>'
            )
        if verification_result is not None:
            for detection in verification_result.detections:
                bbox = detection.bbox or (detection.screen_x - 18, detection.screen_y - 24, 36, 48)
                color = "#f97316" if detection.occupied else "#38bdf8"
                lines.append(
                    f'<rect x="{bbox[0]}" y="{bbox[1]}" width="{bbox[2]}" height="{bbox[3]}" fill="none" stroke="{color}" stroke-dasharray="4 4" stroke-width="2" />'
                )
        for roi_name, roi_value, color in (
            ("hp_bar_roi", self._live_config.hp_bar_roi, "#f87171"),
            ("condition_bar_roi", self._live_config.condition_bar_roi, "#4ade80"),
            ("combat_indicator_roi", self._live_config.combat_indicator_roi, "#fb7185"),
            ("reward_roi", self._live_config.reward_roi, "#facc15"),
        ):
            lines.append(
                f'<rect x="{roi_value[0]}" y="{roi_value[1]}" width="{roi_value[2]}" height="{roi_value[3]}" fill="none" stroke="{color}" stroke-width="2" />'
            )
            lines.append(
                f'<text x="{roi_value[0] + 4}" y="{max(16, roi_value[1] - 4)}" fill="{color}" font-size="12">{roi_name}</text>'
            )
        lines.append(
            f'<text x="16" y="24" fill="#f9fafb" font-size="18">engage outcome={result.outcome.value} reason={result.reason}</text>'
        )
        verify_state_detection = result.metadata.get("verify_state_detection")
        if isinstance(verify_state_detection, dict):
            lines.append(
                f'<text x="16" y="{height - 44}" fill="#e5e7eb" font-size="14">verify_state_source={verify_state_detection.get("source")}</text>'
            )
        lines.append(
            f'<text x="16" y="{height - 24}" fill="#e5e7eb" font-size="14">input_counts=rc:{result.metadata.get("right_click_action_count", 0)} key:{result.metadata.get("key_press_action_count", 0)} seq:{result.metadata.get("key_sequence_action_count", 0)} real:{result.metadata.get("real_input_action_count", 0)}</text>'
        )
        lines.append("</svg>")
        return "\n".join(lines)


class LiveEngageService:
    def __init__(
        self,
        *,
        runtime: "LiveRuntime",
        target_engagement_service: TargetEngagementService,
        state_detector: SimpleStateDetector,
        input_driver: "LiveInputDriver",
        artifact_writer: LiveEngageArtifactWriter,
        verify_delay_s: float,
        click_offset_y_px: int,
        target_match_max_distance_px: int,
    ) -> None:
        self._runtime = runtime
        self._target_engagement_service = target_engagement_service
        self._state_detector = state_detector
        self._input_driver = input_driver
        self._artifact_writer = artifact_writer
        self._verify_delay_s = verify_delay_s
        self._click_offset_y_px = click_offset_y_px
        self._target_match_max_distance_px = target_match_max_distance_px

    def attempt_engage(self, *, cycle_id: int) -> LiveEngageResult:
        initial_event_count = len(self._input_driver.events)
        engagement = self._target_engagement_service.engage_target(cycle_id=cycle_id)
        self._runtime.set_target_resolution(engagement.target_resolution)
        self._runtime.set_approach_result(engagement.approach_result)
        self._runtime.set_interaction_result(engagement.interaction_result)

        observation_result = self._runtime.perception_result(cycle_id=cycle_id, phase="observation")
        observation_metrics_metadata = _build_observation_metrics_metadata(observation_result)
        new_events = self._input_driver.events[initial_event_count:]
        input_metrics_metadata = _build_input_event_metrics(new_events)
        click_point_xy = _resolve_click_point(
            input_events=new_events,
            target_resolution=engagement,
            click_offset_y_px=self._click_offset_y_px,
        )
        selected_target_id = engagement.target_resolution.selected_target_id
        final_target_id = engagement.interaction_result.target_id
        started_at_ts = (
            engagement.target_resolution.world_snapshot.observed_at_ts
            if observation_result is None
            else observation_result.timings.frame_captured_ts
        )

        if "stalled" in engagement.approach_result.reason:
            result = LiveEngageResult(
                cycle_id=cycle_id,
                outcome=LiveEngageOutcome.APPROACH_STALLED,
                reason=engagement.approach_result.reason,
                selected_target_id=selected_target_id,
                final_target_id=final_target_id,
                click_screen_xy=click_point_xy,
                started_at_ts=started_at_ts,
                completed_at_ts=engagement.approach_result.completed_at_ts,
                detection_latency_ms=None if observation_result is None else observation_result.timings.detection_latency_ms,
                selection_latency_ms=None if observation_result is None else observation_result.timings.selection_latency_ms,
                total_reaction_latency_ms=None if observation_result is None else observation_result.timings.total_reaction_latency_ms,
                verification_latency_ms=0.0,
                metadata={
                    "approach_reason": engagement.approach_result.reason,
                    "interaction_reason": engagement.interaction_result.reason,
                    **engagement.approach_result.metadata,
                    **observation_metrics_metadata,
                    "engage_ladder_diagnostics": _build_engage_ladder_diagnostics(
                        observation_result=observation_result,
                        quality_gate_rejected=False,
                    ),
                    **input_metrics_metadata,
                },
            )
            return self._persist_result(
                result=result,
                observation_result=observation_result,
                verification_result=None,
                input_events=new_events,
            )

        if engagement.approach_result.reason.startswith("engage_quality_gate_"):
            result = LiveEngageResult(
                cycle_id=cycle_id,
                outcome=LiveEngageOutcome.NO_TARGET_AVAILABLE,
                reason=engagement.approach_result.reason,
                selected_target_id=selected_target_id,
                final_target_id=final_target_id,
                click_screen_xy=click_point_xy,
                started_at_ts=started_at_ts,
                completed_at_ts=engagement.approach_result.completed_at_ts,
                detection_latency_ms=None if observation_result is None else observation_result.timings.detection_latency_ms,
                selection_latency_ms=None if observation_result is None else observation_result.timings.selection_latency_ms,
                total_reaction_latency_ms=None if observation_result is None else observation_result.timings.total_reaction_latency_ms,
                verification_latency_ms=0.0,
                metadata={
                    "approach_reason": engagement.approach_result.reason,
                    "interaction_reason": engagement.interaction_result.reason,
                    **engagement.approach_result.metadata,
                    **observation_metrics_metadata,
                    "engage_ladder_diagnostics": _build_engage_ladder_diagnostics(
                        observation_result=observation_result,
                        quality_gate_rejected=True,
                    ),
                    **input_metrics_metadata,
                },
            )
            return self._persist_result(
                result=result,
                observation_result=observation_result,
                verification_result=None,
                input_events=new_events,
            )

        if selected_target_id is None or final_target_id is None:
            result = LiveEngageResult(
                cycle_id=cycle_id,
                outcome=LiveEngageOutcome.NO_TARGET_AVAILABLE,
                reason="no_selectable_target",
                selected_target_id=selected_target_id,
                final_target_id=final_target_id,
                click_screen_xy=click_point_xy,
                started_at_ts=started_at_ts,
                completed_at_ts=engagement.interaction_result.observed_at_ts,
                detection_latency_ms=None if observation_result is None else observation_result.timings.detection_latency_ms,
                selection_latency_ms=None if observation_result is None else observation_result.timings.selection_latency_ms,
                total_reaction_latency_ms=None if observation_result is None else observation_result.timings.total_reaction_latency_ms,
                verification_latency_ms=0.0,
                metadata={
                    "approach_reason": engagement.approach_result.reason,
                    "interaction_reason": engagement.interaction_result.reason,
                    **engagement.approach_result.metadata,
                    **observation_metrics_metadata,
                    "engage_ladder_diagnostics": _build_engage_ladder_diagnostics(
                        observation_result=observation_result,
                        quality_gate_rejected=False,
                    ),
                    **input_metrics_metadata,
                },
            )
            return self._persist_result(
                result=result,
                observation_result=observation_result,
                verification_result=None,
                input_events=new_events,
            )

        verify_default_ts = engagement.interaction_result.observed_at_ts + self._verify_delay_s
        if not self._input_driver.dry_run and self._verify_delay_s > 0.0:
            time.sleep(self._verify_delay_s)
        verify_frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="engage_verify",
            default_ts=verify_default_ts,
        )
        verify_perception = self._runtime.analyze_frame(
            cycle_id=cycle_id,
            phase="engage_verify",
            default_ts=verify_default_ts,
        )
        verify_state = self._state_detector.detect_state(verify_frame)
        outcome, reason, classification_metadata = classify_engage_outcome(
            engagement=engagement,
            verify_state=verify_state,
            verify_frame_metadata=verify_frame.metadata,
            verify_perception=verify_perception,
            click_point_xy=click_point_xy,
            target_match_max_distance_px=self._target_match_max_distance_px,
        )
        verification_latency_ms = max(
            0.0,
            (verify_frame.captured_at_ts - engagement.interaction_result.observed_at_ts) * 1000.0,
        )
        result = LiveEngageResult(
            cycle_id=cycle_id,
            outcome=outcome,
            reason=reason,
            selected_target_id=selected_target_id,
            final_target_id=final_target_id,
            click_screen_xy=click_point_xy,
            started_at_ts=started_at_ts,
            completed_at_ts=verify_frame.captured_at_ts,
            detection_latency_ms=None if observation_result is None else observation_result.timings.detection_latency_ms,
            selection_latency_ms=None if observation_result is None else observation_result.timings.selection_latency_ms,
            total_reaction_latency_ms=None if observation_result is None else observation_result.timings.total_reaction_latency_ms,
            verification_latency_ms=verification_latency_ms,
            metadata={
                "approach_reason": engagement.approach_result.reason,
                "interaction_reason": engagement.interaction_result.reason,
                "verify_frame_source": verify_frame.source,
                "verify_state_detection": verify_state.metadata,
                "verify_selected_target_id": verify_perception.selected_target_id,
                **engagement.approach_result.metadata,
                **observation_metrics_metadata,
                **input_metrics_metadata,
                "engage_ladder_diagnostics": _build_engage_ladder_diagnostics(
                    observation_result=observation_result,
                    quality_gate_rejected=False,
                ),
                **classification_metadata,
            },
        )
        return self._persist_result(
            result=result,
            observation_result=observation_result,
            verification_result=verify_perception,
            input_events=new_events,
        )

    def _persist_result(
        self,
        *,
        result: LiveEngageResult,
        observation_result: PerceptionFrameResult | None,
        verification_result: PerceptionFrameResult | None,
        input_events: tuple["LiveInputEvent", ...],
    ) -> LiveEngageResult:
        artifact_paths = self._artifact_writer.write_result(
            result=result,
            observation_result=observation_result,
            verification_result=verification_result,
            input_events=input_events,
        )
        return replace(result, artifact_paths=artifact_paths)


def classify_engage_outcome(
    *,
    engagement: TargetEngagementResult,
    verify_state,
    verify_frame_metadata: dict[str, Any],
    verify_perception: PerceptionFrameResult,
    click_point_xy: tuple[int, int] | None,
    target_match_max_distance_px: int,
) -> tuple[LiveEngageOutcome, str, dict[str, Any]]:
    outcome_hint = verify_frame_metadata.get("engage_result")
    if isinstance(outcome_hint, str):
        normalized_hint = outcome_hint.strip().lower()
        for outcome in LiveEngageOutcome:
            if outcome.value == normalized_hint:
                return outcome, f"engage_result_hint:{normalized_hint}", {"result_hint": normalized_hint}

    if verify_state.in_combat:
        return LiveEngageOutcome.ENGAGED, "entered_combat_detected", {"verify_in_combat": True}

    if (
        bool(verify_frame_metadata.get("approach_timeout", False))
        or bool(verify_frame_metadata.get("verify_timeout", False))
        or "timeout" in engagement.approach_result.reason
    ):
        return LiveEngageOutcome.APPROACH_TIMEOUT, "engage_verify_timeout", {"verify_timeout": True}

    final_target = _find_detection_by_target_id(
        detections=verify_perception.detections,
        target_id=engagement.interaction_result.target_id,
    )
    if final_target is None and click_point_xy is not None:
        final_target = _find_nearest_detection(
            detections=verify_perception.detections,
            reference_xy=click_point_xy,
            max_distance_px=float(target_match_max_distance_px),
        )
    if final_target is not None and final_target.occupied:
        return (
            LiveEngageOutcome.TARGET_STOLEN,
            "target_became_occupied_after_click",
            {
                "verify_target_id": final_target.target_id,
                "verify_target_xy": [final_target.screen_x, final_target.screen_y],
            },
        )

    if click_point_xy is not None:
        occupied_near_click = _find_nearest_detection(
            detections=verify_perception.occupied_detections,
            reference_xy=click_point_xy,
            max_distance_px=float(target_match_max_distance_px),
        )
        if occupied_near_click is not None:
            return (
                LiveEngageOutcome.TARGET_STOLEN,
                "occupied_target_near_click_point",
                {
                    "verify_target_id": occupied_near_click.target_id,
                    "verify_target_xy": [occupied_near_click.screen_x, occupied_near_click.screen_y],
                },
            )

    return (
        LiveEngageOutcome.MISCLICK,
        "target_not_engaged_and_not_stolen",
        {
            "verify_target_count": len(verify_perception.detections),
            "verify_free_target_count": len(verify_perception.free_detections),
            "verify_occupied_target_count": len(verify_perception.occupied_detections),
        },
    )


def record_engage_attempt(
    *,
    storage: "SQLiteTelemetryStorage",
    result: LiveEngageResult,
) -> None:
    storage.record_attempt(
        TelemetryRecord(
            cycle_id=result.cycle_id,
            event_ts=result.completed_at_ts,
            state=BotState.ATTEMPT,
            reaction_ms=result.total_reaction_latency_ms,
            verification_ms=result.verification_latency_ms,
            result=result.outcome.value,
            reason=result.reason,
            metadata={
                "selected_target_id": result.selected_target_id,
                "final_target_id": result.final_target_id,
                "click_screen_xy": None if result.click_screen_xy is None else list(result.click_screen_xy),
                **result.metadata,
            },
        )
    )


def build_observation_for_engage(*, cycle_id: int, observed_at_ts: float) -> Observation:
    return Observation(
        cycle_id=cycle_id,
        observed_at_ts=observed_at_ts,
        signal_detected=True,
        actual_spawn_ts=observed_at_ts,
        source="live_engage",
        confidence=1.0,
        metadata={"engage_mvp": True},
    )


def _resolve_click_point(
    *,
    input_events: tuple["LiveInputEvent", ...],
    target_resolution: TargetEngagementResult,
    click_offset_y_px: int,
) -> tuple[int, int] | None:
    for event in reversed(input_events):
        if event.action != "right_click_target":
            continue
        payload = event.payload
        if "screen_x" in payload and "screen_y" in payload:
            return (int(payload["screen_x"]), int(payload["screen_y"]))

    target_id = target_resolution.interaction_result.target_id
    world_snapshot = target_resolution.target_resolution.world_snapshot
    if target_id is None:
        return None
    target_group = world_snapshot.group_by_id(target_id)
    if target_group is None:
        return None
    return (
        int(round(target_group.position.x)),
        int(round(target_group.position.y + click_offset_y_px)),
    )


def _find_detection_by_target_id(
    *,
    detections: Iterable[LiveTargetDetection],
    target_id: str | None,
) -> LiveTargetDetection | None:
    if target_id is None:
        return None
    for detection in detections:
        if detection.target_id == target_id:
            return detection
    return None


def _find_nearest_detection(
    *,
    detections: Iterable[LiveTargetDetection],
    reference_xy: tuple[int, int],
    max_distance_px: float,
) -> LiveTargetDetection | None:
    nearest: LiveTargetDetection | None = None
    nearest_distance = float("inf")
    for detection in detections:
        distance = math.dist(
            (float(detection.screen_x), float(detection.screen_y)),
            (float(reference_xy[0]), float(reference_xy[1])),
        )
        if distance <= max_distance_px and distance < nearest_distance:
            nearest = detection
            nearest_distance = distance
    return nearest


def _build_observation_metrics_metadata(
    observation_result: PerceptionFrameResult | None,
) -> dict[str, Any]:
    if observation_result is None:
        return {
            "occupied_rejection_count": 0,
            "out_of_zone_rejection_count": 0,
            "selected_target_in_zone": None,
            "scene_calibration_warning": None,
        }
    selected_target = observation_result.selected_target
    return {
        "occupied_rejection_count": len(observation_result.occupied_detections),
        "out_of_zone_rejection_count": len(observation_result.out_of_zone_detections),
        "selected_target_in_zone": None
        if selected_target is None
        else bool(selected_target.metadata.get("in_scene_zone", True)),
        "scene_calibration_warning": observation_result.scene_calibration.get("warning"),
    }


def _build_engage_ladder_diagnostics(
    *,
    observation_result: PerceptionFrameResult | None,
    quality_gate_rejected: bool,
) -> dict[str, int]:
    base_ladder = {}
    if observation_result is not None:
        raw_ladder = observation_result.diagnostics.get("ladder_diagnostics", {})
        if isinstance(raw_ladder, dict):
            base_ladder = dict(raw_ladder)
    base_ladder["engage_quality_gate_rejection_count"] = 1 if quality_gate_rejected else 0
    return {
        "seed_stage_count": int(base_ladder.get("seed_stage_count", 0)),
        "marker_hit_count": int(base_ladder.get("marker_hit_count", 0)),
        "template_marker_fallback_count": int(base_ladder.get("template_marker_fallback_count", 0)),
        "upper_rescue_count": int(base_ladder.get("upper_rescue_count", 0)),
        "confirmation_pass_count": int(base_ladder.get("confirmation_pass_count", 0)),
        "ice_signature_rejection_count": int(base_ladder.get("ice_signature_rejection_count", 0)),
        "player_veto_rejection_count": int(base_ladder.get("player_veto_rejection_count", 0)),
        "out_of_zone_rejection_count": int(base_ladder.get("out_of_zone_rejection_count", 0)),
        "engage_quality_gate_rejection_count": int(base_ladder.get("engage_quality_gate_rejection_count", 0)),
        "final_detection_count": int(base_ladder.get("final_detection_count", 0)),
    }


def _build_input_event_metrics(input_events: tuple["LiveInputEvent", ...]) -> dict[str, Any]:
    right_click_action_count = 0
    key_press_action_count = 0
    key_sequence_action_count = 0
    real_input_action_count = 0
    for event in input_events:
        if event.action == "right_click_target":
            right_click_action_count += 1
            if event.payload.get("execution_status") == "real_click_sent":
                real_input_action_count += 1
        elif event.action == "press_key":
            key_press_action_count += 1
            if event.payload.get("execution_status") == "real_key_sent":
                real_input_action_count += 1
        elif event.action == "press_sequence":
            key_sequence_action_count += 1
            execution_statuses = event.payload.get("execution_statuses", [])
            if isinstance(execution_statuses, list):
                real_input_action_count += sum(1 for status in execution_statuses if status == "real_key_sent")
    return {
        "right_click_action_count": right_click_action_count,
        "key_press_action_count": key_press_action_count,
        "key_sequence_action_count": key_sequence_action_count,
        "real_input_action_count": real_input_action_count,
    }
