from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

from botlab.adapters.live.capture import (
    DebugArtifactWriter,
    create_capture,
)
from botlab.adapters.live.engage import (
    LiveEngageArtifactWriter,
    LiveEngageRunReport,
    LiveEngageService,
    LiveEngageSessionSummary,
    record_engage_attempt,
)
from botlab.adapters.live.input import LiveInputDriver
from botlab.adapters.live.models import (
    LiveEngageResult,
    LiveFrame,
    LiveResourceSnapshot,
    LiveSessionState,
    LiveTargetDetection,
)
from botlab.adapters.live.perception import (
    PerceptionAnalyzer,
    PerceptionArtifactWriter,
    PerceptionFrameResult,
    build_world_snapshot_from_perception,
)
from botlab.adapters.live.vision import (
    LiveResourceProvider,
    SimpleStateDetector,
    StallDetector,
    ready_after_rest,
)
from botlab.adapters.simulation.combat_profiles import SimulatedCombatProfileCatalog
from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog
from botlab.adapters.telemetry.logger import configure_telemetry_logger, log_telemetry_record
from botlab.adapters.telemetry.storage import SQLiteTelemetryStorage
from botlab.application import (
    ActionContext,
    ActionExecutor,
    ActionResult,
    CombatTimeline,
    CycleOrchestrator,
    ObservationPreparationResult,
    ObservationProvider,
    ObservationWindow,
    RestProvider,
    RestTimeline,
    SimulationReport,
    TargetAcquisitionService,
    TargetApproachProvider,
    TargetApproachResult,
    TargetApproachService,
    TargetEngagementResult,
    TargetEngagementService,
    TargetInteractionProvider,
    TargetInteractionResult,
    TargetInteractionService,
    TargetResolution,
    TelemetrySink,
    TimedCombatSnapshot,
    VerificationOutcome,
    VerificationProvider,
    VerificationResult,
    WorldStateProvider,
)
from botlab.config import Settings
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.domain.targeting import RetargetPolicy, TargetSelectionPolicy, TargetValidationPolicy
from botlab.domain.world import GroupSnapshot
from botlab.types import BotState, CombatSnapshot, Observation, TelemetryRecord


class LiveRuntime:
    def __init__(
        self,
        *,
        settings: Settings,
        scheduler: CycleScheduler,
        capture,
        artifact_writer: DebugArtifactWriter,
        perception_analyzer: PerceptionAnalyzer,
        perception_artifact_writer: PerceptionArtifactWriter,
        logger: logging.Logger,
        default_allow_background_capture: bool = False,
    ) -> None:
        self._settings = settings
        self._scheduler = scheduler
        self._capture = capture
        self._artifact_writer = artifact_writer
        self._perception_analyzer = perception_analyzer
        self._perception_artifact_writer = perception_artifact_writer
        self._logger = logger
        self._default_allow_background_capture = default_allow_background_capture
        self._session_state = LiveSessionState()
        self._frames: dict[tuple[int, str, str], LiveFrame] = {}
        self._perception_results: dict[tuple[int, str, str], PerceptionFrameResult] = {}
        self._observation_preparations: dict[int, ObservationPreparationResult] = {}
        self._target_resolutions: dict[int, TargetResolution] = {}
        self._approach_results: dict[int, TargetApproachResult] = {}
        self._interaction_results: dict[int, TargetInteractionResult] = {}

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def session_state(self) -> LiveSessionState:
        return self._session_state

    @property
    def initial_cycle_id(self) -> int:
        return self._scheduler.predictor.anchor_cycle_id + 1

    def prediction_for_cycle(self, cycle_id: int):
        return self._scheduler.prediction_for_cycle(cycle_id)

    def capture_frame(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        allow_background_capture: bool = False,
    ) -> LiveFrame:
        effective_allow_background_capture = allow_background_capture or self._default_allow_background_capture
        capture_mode = "preview_bypass" if effective_allow_background_capture else "default"
        cache_key = (cycle_id, phase, capture_mode)
        cached_frame = self._frames.get(cache_key)
        if cached_frame is not None:
            return cached_frame

        capture_started = time.perf_counter()
        frame = self._capture.capture_frame(
            cycle_id=cycle_id,
            phase=phase,
            default_ts=default_ts,
            session_state=self._session_state,
            allow_background_capture=effective_allow_background_capture,
        )
        capture_latency_ms = max(0.0, (time.perf_counter() - capture_started) * 1000.0)
        frame = replace(
            frame,
            metadata={
                **frame.metadata,
                "capture_latency_ms": capture_latency_ms,
            },
        )
        artifact_paths = self._artifact_writer.write_frame(
            cycle_id=cycle_id,
            phase=phase,
            frame=frame,
        )
        if artifact_paths:
            frame = replace(frame, artifact_paths=artifact_paths)
        self._frames[cache_key] = frame
        self._logger.info(
            "live_capture cycle_id=%s phase=%s source=%s capture_ms=%.1f artifacts=%s",
            cycle_id,
            phase,
            frame.source,
            capture_latency_ms,
            {key: str(path) for key, path in frame.artifact_paths.items()},
        )
        return frame

    def analyze_frame(
        self,
        *,
        cycle_id: int,
        phase: str,
        default_ts: float,
        allow_background_capture: bool = False,
    ) -> PerceptionFrameResult:
        effective_allow_background_capture = allow_background_capture or self._default_allow_background_capture
        capture_mode = "preview_bypass" if effective_allow_background_capture else "default"
        cache_key = (cycle_id, phase, capture_mode)
        cached_result = self._perception_results.get(cache_key)
        if cached_result is not None:
            return cached_result

        frame = self.capture_frame(
            cycle_id=cycle_id,
            phase=phase,
            default_ts=default_ts,
            allow_background_capture=effective_allow_background_capture,
        )
        result = self._perception_analyzer.analyze_frame(frame, cycle_id=cycle_id, phase=phase)
        artifact_paths = self._perception_artifact_writer.write_cycle_result(
            cycle_id=cycle_id,
            phase=phase,
            frame=frame,
            result=result,
        )
        persisted_result = replace(result, artifact_paths=artifact_paths)
        self._perception_artifact_writer.append_jsonl(
            record_name=f"cycle_{cycle_id:03d}:{phase}",
            result=persisted_result,
        )
        self._perception_results[cache_key] = persisted_result
        self._logger.info(
            "live_perception cycle_id=%s phase=%s targets=%s free=%s selected=%s reaction_ms=%.3f",
            cycle_id,
            phase,
            len(persisted_result.detections),
            len(persisted_result.free_detections),
            persisted_result.selected_target_id,
            persisted_result.timings.total_reaction_latency_ms,
        )
        return persisted_result

    def perception_result(
        self,
        *,
        cycle_id: int,
        phase: str,
    ) -> PerceptionFrameResult | None:
        default_result = self._perception_results.get((cycle_id, phase, "default"))
        if default_result is not None:
            return default_result
        return self._perception_results.get((cycle_id, phase, "preview_bypass"))

    def frame_result(
        self,
        *,
        cycle_id: int,
        phase: str,
        capture_mode: str = "default",
    ) -> LiveFrame | None:
        return self._frames.get((cycle_id, phase, capture_mode))

    def set_observation_preparation(self, result: ObservationPreparationResult) -> None:
        self._observation_preparations[result.cycle_id] = result

    def set_target_resolution(self, result: TargetResolution) -> None:
        self._target_resolutions[result.cycle_id] = result

    def set_approach_result(self, result: TargetApproachResult) -> None:
        self._approach_results[result.cycle_id] = result

    def set_interaction_result(self, result: TargetInteractionResult) -> None:
        self._interaction_results[result.cycle_id] = result

    def observation_preparations(self) -> list[ObservationPreparationResult]:
        return [self._observation_preparations[key] for key in sorted(self._observation_preparations)]

    def target_resolutions(self) -> list[TargetResolution]:
        return [self._target_resolutions[key] for key in sorted(self._target_resolutions)]

    def approach_results(self) -> list[TargetApproachResult]:
        return [self._approach_results[key] for key in sorted(self._approach_results)]

    def interaction_results(self) -> list[TargetInteractionResult]:
        return [self._interaction_results[key] for key in sorted(self._interaction_results)]

    def perception_results(self) -> list[PerceptionFrameResult]:
        return [self._perception_results[key] for key in sorted(self._perception_results)]

    def write_perception_session_summary(self) -> None:
        summary = self._perception_analyzer.summarize_session(self.perception_results())
        self._perception_artifact_writer.write_session_summary(summary)

    def update_resources(self, *, hp_ratio: float, condition_ratio: float) -> None:
        self._session_state.hp_ratio = min(max(float(hp_ratio), 0.0), 1.0)
        self._session_state.condition_ratio = min(max(float(condition_ratio), 0.0), 1.0)


class LiveObservationProvider(ObservationProvider):
    def __init__(self, runtime: LiveRuntime) -> None:
        self._runtime = runtime

    def get_observation_window(self, cycle_id: int) -> ObservationWindow:
        prediction = self._runtime.prediction_for_cycle(cycle_id)
        frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="observation",
            default_ts=prediction.predicted_spawn_ts,
        )
        perception = self._runtime.analyze_frame(
            cycle_id=cycle_id,
            phase="observation",
            default_ts=prediction.predicted_spawn_ts,
        )
        preparation = ObservationPreparationResult(
            cycle_id=cycle_id,
            spawn_zone_visible=True,
            ready_for_observation=True,
            starting_position_xy=(0.0, 0.0),
            observation_position_xy=(0.0, 0.0),
            travel_s=0.0,
            arrived_at_ts=prediction.ready_window_start_ts,
            wait_for_spawn_s=max(0.0, prediction.predicted_spawn_ts - prediction.ready_window_start_ts),
            note="live_observation_ready",
            metadata={
                "ready_reason": "already_on_spot",
                "start_position_source": "live_current_spot",
                "selected_target_id": perception.selected_target_id,
                "detection_latency_ms": perception.timings.detection_latency_ms,
                "selection_latency_ms": perception.timings.selection_latency_ms,
                "total_reaction_latency_ms": perception.timings.total_reaction_latency_ms,
            },
        )
        self._runtime.set_observation_preparation(preparation)

        observation = None
        if perception.detections:
            observation = Observation(
                cycle_id=cycle_id,
                observed_at_ts=prediction.predicted_spawn_ts,
                signal_detected=True,
                actual_spawn_ts=prediction.predicted_spawn_ts,
                source=frame.source,
                confidence=1.0,
                metadata={
                    "target_count": len(perception.detections),
                    "free_target_count": len(perception.free_detections),
                    "occupied_target_count": len(perception.occupied_detections),
                    "selected_target_id": perception.selected_target_id,
                    "detection_latency_ms": perception.timings.detection_latency_ms,
                    "selection_latency_ms": perception.timings.selection_latency_ms,
                    "total_reaction_latency_ms": perception.timings.total_reaction_latency_ms,
                },
            )

        return ObservationWindow(
            cycle_id=cycle_id,
            observation=observation,
            actual_spawn_ts=prediction.predicted_spawn_ts,
            window_closed_ts=prediction.ready_window_end_ts,
            note="live_cycle",
            metadata={
                "observable_in_ready_window": bool(perception.detections),
                "frame_source": frame.source,
                "selected_target_id": perception.selected_target_id,
            },
        )


class LiveWorldStateProvider(WorldStateProvider):
    def __init__(self, runtime: LiveRuntime) -> None:
        self._runtime = runtime

    def get_world_snapshot(self, cycle_id: int):
        prediction = self._runtime.prediction_for_cycle(cycle_id)
        frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="observation",
            default_ts=prediction.predicted_spawn_ts,
        )
        perception = self._runtime.analyze_frame(
            cycle_id=cycle_id,
            phase="observation",
            default_ts=prediction.predicted_spawn_ts,
        )
        return build_world_snapshot_from_perception(
            cycle_id=cycle_id,
            frame=frame,
            perception_result=perception,
            current_target_id=None,
            phase="live_acquire",
        )

    def get_approach_world_snapshot(self, cycle_id: int):
        prediction = self._runtime.prediction_for_cycle(cycle_id)
        frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="approach_revalidation",
            default_ts=prediction.predicted_spawn_ts + 0.25,
        )
        perception = self._runtime.analyze_frame(
            cycle_id=cycle_id,
            phase="approach_revalidation",
            default_ts=prediction.predicted_spawn_ts + 0.25,
        )
        return build_world_snapshot_from_perception(
            cycle_id=cycle_id,
            frame=frame,
            perception_result=perception,
            current_target_id=None,
            phase="live_approach_revalidation",
        )

    def get_interaction_world_snapshot(self, cycle_id: int):
        prediction = self._runtime.prediction_for_cycle(cycle_id)
        frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="interaction",
            default_ts=prediction.predicted_spawn_ts + 0.50,
        )
        perception = self._runtime.analyze_frame(
            cycle_id=cycle_id,
            phase="interaction",
            default_ts=prediction.predicted_spawn_ts + 0.50,
        )
        return build_world_snapshot_from_perception(
            cycle_id=cycle_id,
            frame=frame,
            perception_result=perception,
            current_target_id=None,
            phase="live_interaction_revalidation",
        )


class LiveTargetApproachProvider(TargetApproachProvider):
    def __init__(
        self,
        runtime: LiveRuntime,
        *,
        input_driver: LiveInputDriver,
        stall_detector: StallDetector,
        movement_speed_units_per_s: float = 4.0,
        interaction_range: float = 0.5,
        step_distance_units: float = 1.0,
        engage_min_target_confidence: float = 0.70,
        engage_min_seen_frames: int = 1,
        engage_relaxed_target_confidence: float = 0.62,
        engage_relaxed_min_seen_frames: int = 2,
        engage_relaxed_min_confirmation_score: float = 0.72,
        engage_relaxed_min_ice_score: float = 0.35,
        engage_relaxed_max_player_veto_score: float = 0.45,
    ) -> None:
        self._runtime = runtime
        self._input_driver = input_driver
        self._stall_detector = stall_detector
        self._movement_speed_units_per_s = movement_speed_units_per_s
        self._interaction_range = interaction_range
        self._step_distance_units = step_distance_units
        self._engage_min_target_confidence = engage_min_target_confidence
        self._engage_min_seen_frames = engage_min_seen_frames
        self._engage_relaxed_target_confidence = engage_relaxed_target_confidence
        self._engage_relaxed_min_seen_frames = engage_relaxed_min_seen_frames
        self._engage_relaxed_min_confirmation_score = engage_relaxed_min_confirmation_score
        self._engage_relaxed_min_ice_score = engage_relaxed_min_ice_score
        self._engage_relaxed_max_player_veto_score = engage_relaxed_max_player_veto_score

    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        world_snapshot = target_resolution.world_snapshot
        started_at_ts = world_snapshot.observed_at_ts
        target_id = target_resolution.selected_target_id
        if target_id is None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=None,
                started_at_ts=started_at_ts,
                completed_at_ts=started_at_ts,
                travel_s=0.0,
                arrived=False,
                reason="no_target_selected",
                initial_target_id=None,
                retargeted=False,
                metadata={},
            )

        target_group = world_snapshot.group_by_id(target_id)
        if target_group is None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=target_id,
                started_at_ts=started_at_ts,
                completed_at_ts=started_at_ts,
                travel_s=0.0,
                arrived=False,
                reason="target_missing_from_world_snapshot",
                initial_target_id=target_id,
                retargeted=False,
                metadata={},
            )

        quality_gate_reason, engage_gate_metadata = self._evaluate_target_for_engage(target_group)
        if quality_gate_reason is not None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=target_id,
                started_at_ts=started_at_ts,
                completed_at_ts=started_at_ts,
                travel_s=0.0,
                arrived=False,
                reason=quality_gate_reason,
                initial_target_id=target_id,
                retargeted=False,
                metadata={
                    **engage_gate_metadata,
                    "engage_quality_gate_rejected": True,
                    "target_confidence": target_group.metadata.get("confidence"),
                    "target_seen_frames": target_group.metadata.get("seen_frames"),
                    "target_in_scene_zone": target_group.metadata.get("in_scene_zone"),
                },
            )

        self._input_driver.right_click_target(
            LiveTargetDetection(
                target_id=target_id,
                screen_x=int(target_group.position.x),
                screen_y=int(target_group.position.y),
                distance=target_group.distance,
                occupied=False,
                mob_variant=str(target_group.metadata.get("mob_variant", "mob_a")),
            )
        )

        travel_distance = max(0.0, target_group.distance - self._interaction_range)
        movement_steps = self._build_movement_steps(
            started_at_ts=started_at_ts,
            travel_distance=travel_distance,
        )
        approach_frame = self._runtime.capture_frame(
            cycle_id=target_resolution.cycle_id,
            phase="approach",
            default_ts=started_at_ts,
        )
        stall_after_s = approach_frame.metadata.get("stall_after_s")
        if isinstance(stall_after_s, (int, float)):
            last_progress_ts = float(approach_frame.metadata.get("last_progress_ts", started_at_ts))
            stalled_at_ts = last_progress_ts + float(stall_after_s)
            if self._stall_detector.is_stalled(
                last_progress_ts=last_progress_ts,
                now_ts=stalled_at_ts,
                entered_combat=bool(approach_frame.metadata.get("in_combat", False)),
            ):
                return TargetApproachResult(
                    cycle_id=target_resolution.cycle_id,
                    target_id=target_id,
                    started_at_ts=started_at_ts,
                    completed_at_ts=stalled_at_ts,
                    travel_s=max(0.0, stalled_at_ts - started_at_ts),
                    arrived=False,
                    reason="approach_stalled_no_progress_timeout",
                    initial_target_id=target_id,
                    retargeted=False,
                    metadata={
                        "movement_steps": movement_steps,
                        "stall_timeout_s": self._stall_detector.timeout_s,
                    },
                )

        completed_at_ts = started_at_ts + (travel_distance / self._movement_speed_units_per_s)
        return TargetApproachResult(
            cycle_id=target_resolution.cycle_id,
            target_id=target_id,
            started_at_ts=started_at_ts,
            completed_at_ts=completed_at_ts,
            travel_s=max(0.0, completed_at_ts - started_at_ts),
            arrived=True,
            reason="target_reached_in_live_adapter",
            initial_target_id=target_id,
            retargeted=False,
            metadata={
                **engage_gate_metadata,
                "movement_steps": movement_steps,
                "movement_step_count": len(movement_steps),
            },
        )

    def _evaluate_target_for_engage(self, target_group: GroupSnapshot) -> tuple[str | None, dict[str, object]]:
        metadata: dict[str, object] = {
            "engage_gate_decision": "reject",
            "engage_gate_reason": None,
            "engage_quality_gate_rejection_count": 0,
            "target_confidence": float(target_group.metadata.get("confidence", 1.0)),
            "target_seen_frames": int(target_group.metadata.get("seen_frames", 1)),
            "target_track_seen_frames": int(target_group.metadata.get("track_seen_frames", 1)),
            "target_in_scene_zone": bool(target_group.metadata.get("in_scene_zone", True)),
            "target_player_veto_triggered": bool(target_group.metadata.get("player_veto_triggered", False)),
            "target_player_veto_score": float(target_group.metadata.get("player_veto_score", 0.0)),
            "target_player_veto_threshold": float(
                target_group.metadata.get("player_veto_threshold", 1.0)
            ),
            "target_player_veto_gate_decision": str(
                target_group.metadata.get("player_veto_gate_decision", "allow")
            ),
            "target_confirmation_score": float(
                target_group.metadata.get(
                    "confirmation_selected_score",
                    target_group.metadata.get("confirmation_confidence", 0.0),
                )
            ),
            "target_ice_score": float(target_group.metadata.get("ice_score", 0.0)),
            "target_occupied_reason": target_group.metadata.get("occupied_reason"),
            "target_actionable": bool(target_group.metadata.get("actionable", True)),
            "engage_min_target_confidence": self._engage_min_target_confidence,
            "engage_relaxed_target_confidence": self._engage_relaxed_target_confidence,
            "engage_min_seen_frames": self._engage_min_seen_frames,
            "engage_relaxed_min_seen_frames": self._engage_relaxed_min_seen_frames,
            "engage_relaxed_min_confirmation_score": self._engage_relaxed_min_confirmation_score,
            "engage_relaxed_min_ice_score": self._engage_relaxed_min_ice_score,
            "engage_relaxed_max_player_veto_score": self._engage_relaxed_max_player_veto_score,
        }
        if target_group.engaged_by_other:
            metadata["engage_gate_reason"] = "engage_quality_gate_target_occupied"
            metadata["engage_quality_gate_rejection_count"] = 1
            return "engage_quality_gate_target_occupied", metadata
        if not target_group.reachable:
            metadata["engage_gate_reason"] = "engage_quality_gate_target_unreachable"
            metadata["engage_quality_gate_rejection_count"] = 1
            return "engage_quality_gate_target_unreachable", metadata
        if not bool(target_group.metadata.get("in_scene_zone", True)):
            metadata["engage_gate_reason"] = "engage_quality_gate_out_of_zone"
            metadata["engage_quality_gate_rejection_count"] = 1
            return "engage_quality_gate_out_of_zone", metadata
        if bool(target_group.metadata.get("player_veto_triggered", False)):
            metadata["engage_gate_reason"] = "engage_quality_gate_player_veto"
            metadata["engage_quality_gate_rejection_count"] = 1
            return "engage_quality_gate_player_veto", metadata
        confidence = float(target_group.metadata.get("confidence", 1.0))
        seen_frames = int(target_group.metadata.get("seen_frames", 1))
        track_seen_frames = int(target_group.metadata.get("track_seen_frames", seen_frames))
        actionable = bool(target_group.metadata.get("actionable", True))
        confirmation_score = float(
            target_group.metadata.get(
                "confirmation_selected_score",
                target_group.metadata.get("confirmation_confidence", 0.0),
            )
        )
        ice_score = float(target_group.metadata.get("ice_score", 0.0))
        player_veto_score = float(target_group.metadata.get("player_veto_score", 0.0))
        player_veto_gate_decision = str(target_group.metadata.get("player_veto_gate_decision", "allow"))
        if not actionable or seen_frames < self._engage_min_seen_frames:
            metadata["engage_gate_reason"] = "engage_quality_gate_not_stable"
            metadata["engage_quality_gate_rejection_count"] = 1
            return "engage_quality_gate_not_stable", metadata
        if confidence >= self._engage_min_target_confidence:
            metadata["engage_gate_decision"] = "strict_confidence_pass"
            metadata["engage_gate_reason"] = "engage_quality_gate_pass_strict_confidence"
            return None, metadata
        if (
            confidence >= self._engage_relaxed_target_confidence
            and track_seen_frames >= max(self._engage_min_seen_frames, self._engage_relaxed_min_seen_frames)
        ):
            if player_veto_gate_decision != "allow" or player_veto_score > self._engage_relaxed_max_player_veto_score:
                metadata["engage_gate_reason"] = "engage_quality_gate_player_veto_suspicious"
                metadata["engage_quality_gate_rejection_count"] = 1
                return "engage_quality_gate_player_veto_suspicious", metadata
            if confirmation_score < self._engage_relaxed_min_confirmation_score:
                metadata["engage_gate_reason"] = "engage_quality_gate_confirmation_too_weak"
                metadata["engage_quality_gate_rejection_count"] = 1
                return "engage_quality_gate_confirmation_too_weak", metadata
            if ice_score < self._engage_relaxed_min_ice_score:
                metadata["engage_gate_reason"] = "engage_quality_gate_not_icy_enough"
                metadata["engage_quality_gate_rejection_count"] = 1
                return "engage_quality_gate_not_icy_enough", metadata
            metadata["engage_gate_decision"] = "relaxed_stable_pass"
            metadata["engage_gate_reason"] = "engage_quality_gate_pass_relaxed_stable"
            return None, metadata
        metadata["engage_gate_reason"] = "engage_quality_gate_low_confidence"
        metadata["engage_quality_gate_rejection_count"] = 1
        return "engage_quality_gate_low_confidence", metadata

    def _build_movement_steps(
        self,
        *,
        started_at_ts: float,
        travel_distance: float,
    ) -> list[dict[str, float | int]]:
        if travel_distance <= 0.0:
            return []

        steps: list[dict[str, float | int]] = []
        remaining_distance = travel_distance
        current_ts = started_at_ts
        step_index = 0
        while remaining_distance > 0.0:
            step_index += 1
            step_distance = min(self._step_distance_units, remaining_distance)
            current_ts += step_distance / self._movement_speed_units_per_s
            remaining_distance = max(0.0, remaining_distance - step_distance)
            steps.append(
                {
                    "step_index": step_index,
                    "step_distance": round(step_distance, 6),
                    "arrived_ts": round(current_ts, 6),
                    "remaining_distance": round(remaining_distance, 6),
                }
            )
        return steps


class LiveTargetInteractionProvider(TargetInteractionProvider):
    def __init__(self, runtime: LiveRuntime, state_detector: SimpleStateDetector) -> None:
        self._runtime = runtime
        self._state_detector = state_detector

    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        interaction_frame = self._runtime.capture_frame(
            cycle_id=target_approach_result.cycle_id,
            phase="interaction",
            default_ts=target_approach_result.completed_at_ts,
        )
        state = self._state_detector.detect_state(interaction_frame)
        ready = bool(interaction_frame.metadata.get("interaction_ready", False)) or state.in_combat
        return TargetInteractionResult(
            cycle_id=target_approach_result.cycle_id,
            target_id=target_approach_result.target_id,
            ready=ready,
            observed_at_ts=interaction_frame.captured_at_ts,
            reason="interaction_ready" if ready else "interaction_not_ready",
            initial_target_id=target_approach_result.initial_target_id,
            retargeted=target_approach_result.retargeted,
            metadata={},
        )


class LiveActionExecutor(ActionExecutor):
    def __init__(self, runtime: LiveRuntime, target_engagement_service: TargetEngagementService) -> None:
        self._runtime = runtime
        self._target_engagement_service = target_engagement_service

    def execute_action(self, context: ActionContext) -> ActionResult:
        engagement = self._target_engagement_service.engage_target(cycle_id=context.cycle_id)
        self._runtime.set_target_resolution(engagement.target_resolution)
        self._runtime.set_approach_result(engagement.approach_result)
        self._runtime.set_interaction_result(engagement.interaction_result)

        metadata = self._build_metadata(engagement)
        if engagement.interaction_result.ready:
            return ActionResult(
                cycle_id=context.cycle_id,
                success=True,
                executed_at_ts=engagement.interaction_result.observed_at_ts,
                reason="action_executed",
                metadata=metadata,
            )

        result_reason = "no_target_available" if engagement.interaction_result.target_id is None else "approach_failed"
        return ActionResult(
            cycle_id=context.cycle_id,
            success=False,
            executed_at_ts=engagement.interaction_result.observed_at_ts,
            reason=result_reason,
            metadata=metadata,
        )

    def _build_metadata(self, engagement: TargetEngagementResult) -> dict[str, object]:
        target_resolution = engagement.target_resolution
        approach_result = engagement.approach_result
        interaction_result = engagement.interaction_result
        observation_perception = self._runtime.perception_result(
            cycle_id=target_resolution.cycle_id,
            phase="observation",
        )
        return {
            "session_hp_ratio_before_cycle": self._runtime.session_state.hp_ratio,
            "session_condition_ratio_before_cycle": self._runtime.session_state.condition_ratio,
            "initial_target_id": target_resolution.selected_target_id,
            "selected_target_id": interaction_result.target_id,
            "target_decision_reason": target_resolution.decision.reason,
            "approach_reason": approach_result.reason,
            "interaction_reason": interaction_result.reason,
            "approach_travel_s": approach_result.travel_s,
            "interaction_ready": interaction_result.ready,
            "retargeted_during_approach": approach_result.retargeted,
            "retarget_count": int(approach_result.retargeted),
            "target_loss_count": int(approach_result.retargeted),
            "perception_detection_latency_ms": None if observation_perception is None else observation_perception.timings.detection_latency_ms,
            "perception_selection_latency_ms": None if observation_perception is None else observation_perception.timings.selection_latency_ms,
            "perception_total_reaction_latency_ms": None if observation_perception is None else observation_perception.timings.total_reaction_latency_ms,
        }


class LiveVerificationProvider(VerificationProvider):
    def __init__(self, runtime: LiveRuntime, state_detector: SimpleStateDetector) -> None:
        self._runtime = runtime
        self._state_detector = state_detector

    def verify(self, cycle_id: int, observation: Observation) -> VerificationResult:
        frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="verify",
            default_ts=observation.observed_at_ts + 0.10,
        )
        state = self._state_detector.detect_state(frame)
        if state.in_combat:
            outcome = VerificationOutcome.SUCCESS
            reason = "entered_combat_detected"
        elif bool(frame.metadata.get("verify_timeout", False)):
            outcome = VerificationOutcome.TIMEOUT
            reason = "verify_timeout"
        else:
            outcome = VerificationOutcome.FAILURE
            reason = "combat_not_detected"
        return VerificationResult(
            cycle_id=cycle_id,
            outcome=outcome,
            started_at_ts=observation.observed_at_ts + 0.03,
            completed_at_ts=frame.captured_at_ts,
            reason=reason,
            metadata={
                "state_detection": state.metadata,
            },
        )


class LiveCombatResolver:
    def __init__(
        self,
        runtime: LiveRuntime,
        *,
        input_driver: LiveInputDriver,
        state_detector: SimpleStateDetector,
        resource_provider: LiveResourceProvider,
        combat_plan_catalog: SimulatedCombatPlanCatalog,
        combat_profile_catalog: SimulatedCombatProfileCatalog,
    ) -> None:
        self._runtime = runtime
        self._input_driver = input_driver
        self._state_detector = state_detector
        self._resource_provider = resource_provider
        self._combat_plan_catalog = combat_plan_catalog
        self._combat_profile_catalog = combat_profile_catalog

    def resolve_combat(
        self,
        cycle_id: int,
        *,
        combat_started_ts: float,
        observation: Observation,
    ) -> CombatTimeline:
        frame = self._runtime.capture_frame(
            cycle_id=cycle_id,
            phase="combat",
            default_ts=combat_started_ts + 0.30,
        )
        state = self._state_detector.detect_state(frame)
        resources = self._resource_provider.read_resources(frame)
        turn_count = int(frame.metadata.get("combat_turns", 4))
        reward_duration_s = float(frame.metadata.get("reward_duration_s", 0.35))

        if self._runtime.settings.combat.default_profile_name is not None:
            selection = self._combat_profile_catalog.select_profile(
                self._runtime.settings.combat.default_profile_name
            )
        else:
            selection = self._combat_plan_catalog.select_plan()

        snapshots: list[TimedCombatSnapshot] = []
        for turn_index in range(1, turn_count):
            round_keys = ("1", "space") if turn_index == 1 else ("3", "space")
            self._input_driver.press_sequence(round_keys)
            event_ts = combat_started_ts + (turn_index * 0.30)
            snapshots.append(
                TimedCombatSnapshot(
                    event_ts=event_ts,
                    snapshot=CombatSnapshot(
                        hp_ratio=max(resources.hp_ratio, 0.40),
                        condition_ratio=max(resources.condition_ratio, 0.40),
                        turn_index=turn_index,
                        enemy_count=1,
                        strategy="live_mvp_basic",
                        in_combat=True,
                        combat_started_ts=combat_started_ts,
                        combat_finished_ts=None,
                        metadata={
                            "cycle_id": cycle_id,
                            "phase": "combat_turn",
                            "combat_plan_name": selection.plan_name,
                            "combat_plan_source": selection.source,
                            "combat_profile_name": selection.metadata.get("combat_profile_name"),
                            "input_sequence": list(round_keys),
                            "input_key": round_keys[0],
                            "reward_visible": state.reward_visible,
                            "state_detection": state.metadata,
                            "resource_detection": resources.metadata,
                            "starting_hp_ratio": self._runtime.session_state.hp_ratio,
                            "starting_condition_ratio": self._runtime.session_state.condition_ratio,
                        },
                    ),
                )
            )

        final_event_ts = combat_started_ts + (turn_count * 0.30)
        final_round_keys = ("1", "space") if turn_count == 1 else ("3", "space")
        self._input_driver.press_sequence(final_round_keys)
        final_snapshot = CombatSnapshot(
            hp_ratio=resources.hp_ratio,
            condition_ratio=resources.condition_ratio,
            turn_index=turn_count,
            enemy_count=0,
            strategy="live_mvp_basic",
            in_combat=False,
            combat_started_ts=combat_started_ts,
            combat_finished_ts=final_event_ts,
            metadata={
                "cycle_id": cycle_id,
                "phase": "combat_finished",
                "combat_plan_name": selection.plan_name,
                "combat_plan_source": selection.source,
                "combat_profile_name": selection.metadata.get("combat_profile_name"),
                "input_sequence": list(final_round_keys),
                "input_key": final_round_keys[0],
                "reward_started_ts": final_event_ts,
                "reward_completed_ts": final_event_ts + reward_duration_s,
                "reward_visible": state.reward_visible,
                "state_detection": state.metadata,
                "resource_detection": resources.metadata,
                "starting_hp_ratio": self._runtime.session_state.hp_ratio,
                "starting_condition_ratio": self._runtime.session_state.condition_ratio,
            },
        )
        snapshots.append(TimedCombatSnapshot(event_ts=final_event_ts, snapshot=final_snapshot))
        self._runtime.update_resources(hp_ratio=resources.hp_ratio, condition_ratio=resources.condition_ratio)
        return CombatTimeline(cycle_id=cycle_id, snapshots=snapshots, metadata={})


class LiveRestProvider(RestProvider):
    def __init__(
        self,
        runtime: LiveRuntime,
        *,
        input_driver: LiveInputDriver,
        resource_provider: LiveResourceProvider,
    ) -> None:
        self._runtime = runtime
        self._input_driver = input_driver
        self._resource_provider = resource_provider

    def apply_rest(
        self,
        cycle_id: int,
        *,
        rest_started_ts: float,
        starting_hp_ratio: float,
        starting_condition_ratio: float = 1.0,
        observation: Observation,
    ) -> RestTimeline:
        self._input_driver.press_key("esc")
        self._input_driver.press_key("r")
        snapshots: list[TimedCombatSnapshot] = []
        previous_resources = LiveResourceSnapshot(
            hp_ratio=starting_hp_ratio,
            condition_ratio=starting_condition_ratio,
            metadata={"source": "starting_snapshot"},
        )
        previous_aggregate = None
        warning_count = 0
        stalled_or_uncertain_count = 0
        threshold_reached_count = 0
        stabilized_tick_index = 0
        consecutive_stalled_ticks = 0

        for tick_index in range(1, self._runtime.settings.live.rest_resource_max_ticks + 1):
            sample_snapshots: list[LiveResourceSnapshot] = []
            sample_frame_sources: list[str] = []
            sample_base_ts = rest_started_ts + ((tick_index - 1) * 0.50)
            for sample_index in range(1, self._runtime.settings.live.rest_resource_sample_count + 1):
                sample_ts = sample_base_ts + (
                    (sample_index - 1) * self._runtime.settings.live.rest_resource_sample_interval_s
                )
                frame = self._runtime.capture_frame(
                    cycle_id=cycle_id,
                    phase=f"rest_sample_{tick_index:02d}_{sample_index:02d}",
                    default_ts=sample_ts,
                )
                sample_snapshots.append(self._resource_provider.read_resources(frame))
                sample_frame_sources.append(frame.source)

            aggregate = self._resource_provider.aggregate_resource_reads(
                tuple(sample_snapshots),
                previous_snapshot=previous_resources,
            )
            hp_growth = aggregate.hp_ratio - previous_resources.hp_ratio
            condition_growth = aggregate.condition_ratio - previous_resources.condition_ratio
            progress_detected = (
                hp_growth >= self._runtime.settings.live.rest_resource_growth_min_delta
                or condition_growth >= self._runtime.settings.live.rest_resource_growth_min_delta
            )
            ready = ready_after_rest(
                hp_ratio=aggregate.hp_ratio,
                condition_ratio=aggregate.condition_ratio,
                combat_config=self._runtime.settings.combat,
            )
            stop_reason = "rest_continue"
            aggregate_warnings = list(aggregate.warnings)
            if ready and aggregate.confidence >= self._runtime.settings.live.rest_resource_min_confidence:
                stop_reason = "rest_stop_threshold_reached"
                threshold_reached_count += 1
                consecutive_stalled_ticks = 0
                if stabilized_tick_index == 0:
                    stabilized_tick_index = tick_index
            elif not progress_detected or aggregate.confidence < self._runtime.settings.live.rest_resource_min_confidence:
                stalled_or_uncertain_count += 1
                consecutive_stalled_ticks += 1
                stop_reason = "rest_stalled_or_uncertain"
                if not progress_detected:
                    aggregate_warnings.append("rest_progress_not_detected")
                if aggregate.confidence < self._runtime.settings.live.rest_resource_min_confidence:
                    aggregate_warnings.append("rest_confidence_below_threshold")
                if (
                    consecutive_stalled_ticks
                    >= self._runtime.settings.live.rest_resource_stall_warning_ticks
                ):
                    aggregate_warnings.append("rest_sampling_stalled")
            else:
                consecutive_stalled_ticks = 0

            warning_count += len(aggregate_warnings)

            event_ts = sample_base_ts + (
                self._runtime.settings.live.rest_resource_sample_count
                * self._runtime.settings.live.rest_resource_sample_interval_s
            )
            snapshots.append(
                TimedCombatSnapshot(
                    event_ts=event_ts,
                    snapshot=CombatSnapshot(
                        hp_ratio=aggregate.hp_ratio,
                        condition_ratio=aggregate.condition_ratio,
                        turn_index=tick_index,
                        enemy_count=0,
                        strategy="live_rest",
                        in_combat=False,
                        metadata={
                            "cycle_id": cycle_id,
                            "phase": "rest_tick",
                            "resource_detection": aggregate.to_dict(),
                            "resource_confidence": aggregate.confidence,
                            "resource_warning_count": len(aggregate_warnings),
                            "resource_warnings": aggregate_warnings,
                            "rest_decision": stop_reason,
                            "rest_progress_detected": progress_detected,
                            "sample_frame_sources": sample_frame_sources,
                            "sample_count": aggregate.sample_count,
                        },
                    ),
                )
            )
            previous_resources = LiveResourceSnapshot(
                hp_ratio=aggregate.hp_ratio,
                condition_ratio=aggregate.condition_ratio,
                metadata={"source": "aggregated"},
            )
            previous_aggregate = aggregate
            if stop_reason == "rest_stop_threshold_reached":
                break

        if snapshots:
            final_snapshot = snapshots[-1].snapshot
            self._runtime.update_resources(
                hp_ratio=final_snapshot.hp_ratio,
                condition_ratio=final_snapshot.condition_ratio,
            )
        return RestTimeline(
            cycle_id=cycle_id,
            snapshots=snapshots,
            metadata={
                "resource_sample_count": self._runtime.settings.live.rest_resource_sample_count,
                "resource_confidence": None if previous_aggregate is None else previous_aggregate.confidence,
                "resource_warning_count": warning_count,
                "rest_stop_threshold_reached_count": threshold_reached_count,
                "rest_stalled_or_uncertain_count": stalled_or_uncertain_count,
                "rest_stabilized_tick_index": stabilized_tick_index,
                "resource_final_hp_ratio": None if previous_aggregate is None else previous_aggregate.hp_ratio,
                "resource_final_condition_ratio": None
                if previous_aggregate is None
                else previous_aggregate.condition_ratio,
            },
        )


class LiveTelemetrySink(TelemetrySink):
    def __init__(self, storage: SQLiteTelemetryStorage, logger: logging.Logger) -> None:
        self._storage = storage
        self._logger = logger

    def record_cycle(self, record: TelemetryRecord) -> None:
        self._storage.record_cycle(record)
        log_telemetry_record(self._logger, record)

    def record_attempt(self, record: TelemetryRecord) -> None:
        self._storage.record_attempt(record)
        log_telemetry_record(self._logger, record)

    def record_state_transition(self, record: TelemetryRecord) -> None:
        self._storage.record_state_transition(record)
        log_telemetry_record(self._logger, record)


class LiveRunner:
    def __init__(
        self,
        *,
        orchestrator: CycleOrchestrator,
        engage_service: LiveEngageService,
        runtime: LiveRuntime,
        storage: SQLiteTelemetryStorage,
        logger: logging.Logger,
    ) -> None:
        self._orchestrator = orchestrator
        self._engage_service = engage_service
        self._runtime = runtime
        self._storage = storage
        self._logger = logger
        self._next_live_cycle_id = runtime.initial_cycle_id

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        initial_anchor_spawn_ts: float = 100.0,
        initial_anchor_cycle_id: int = 0,
        logger_name: str = "botlab.live",
        enable_console: bool = False,
        force_input_dry_run: bool = False,
        force_background_capture: bool = False,
    ) -> "LiveRunner":
        scheduler = CycleScheduler.from_cycle_config(settings.cycle)
        scheduler.bootstrap(
            anchor_spawn_ts=initial_anchor_spawn_ts,
            anchor_cycle_id=initial_anchor_cycle_id,
        )
        storage = SQLiteTelemetryStorage.from_config(settings.telemetry)
        logger = configure_telemetry_logger(
            telemetry_config=settings.telemetry,
            logger_name=logger_name,
            enable_console=enable_console,
        )
        capture = create_capture(settings.live)
        runtime = LiveRuntime(
            settings=settings,
            scheduler=scheduler,
            capture=capture,
            artifact_writer=DebugArtifactWriter(settings.live),
            perception_analyzer=PerceptionAnalyzer(settings.live),
            perception_artifact_writer=PerceptionArtifactWriter(settings.live.debug_directory, settings.live),
            logger=logger,
            default_allow_background_capture=force_background_capture,
        )
        decision_engine = DecisionEngine(settings.cycle, settings.combat)
        fsm = CycleFSM(
            decision_engine=decision_engine,
            initial_state=BotState.IDLE,
            started_at_ts=0.0,
            cycle_id=None,
        )
        recovery = RecoveryManager(settings.cycle)
        input_driver = LiveInputDriver(
            logger=logger,
            dry_run=settings.live.dry_run or force_input_dry_run,
            enable_real_input=False if force_input_dry_run else settings.live.enable_real_input,
            enable_real_clicks=False if force_input_dry_run else settings.live.enable_real_clicks,
            enable_real_keys=False if force_input_dry_run else settings.live.enable_real_keys,
            screen_offset_xy=(
                settings.live.capture_region[0],
                settings.live.capture_region[1],
            ),
            real_input_guard=getattr(capture, "real_input_guard_status", None),
        )
        state_detector = SimpleStateDetector(settings.live)
        resource_provider = LiveResourceProvider(settings.live)
        world_provider = LiveWorldStateProvider(runtime)
        retarget_policy = RetargetPolicy(
            selection_policy=TargetSelectionPolicy(),
            validation_policy=TargetValidationPolicy(),
        )
        target_acquisition_service = TargetAcquisitionService(
            world_state_provider=world_provider,
            retarget_policy=retarget_policy,
        )
        target_approach_service = TargetApproachService(
            target_approach_provider=LiveTargetApproachProvider(
                runtime,
                input_driver=input_driver,
                stall_detector=StallDetector(settings.live.stall_timeout_s),
                engage_min_target_confidence=settings.live.engage_min_target_confidence,
                engage_min_seen_frames=settings.live.engage_min_seen_frames,
                engage_relaxed_target_confidence=settings.live.engage_relaxed_target_confidence,
                engage_relaxed_min_seen_frames=settings.live.engage_relaxed_min_seen_frames,
                engage_relaxed_min_confirmation_score=settings.live.engage_relaxed_min_confirmation_score,
                engage_relaxed_min_ice_score=settings.live.engage_relaxed_min_ice_score,
                engage_relaxed_max_player_veto_score=settings.live.engage_relaxed_max_player_veto_score,
            ),
            approach_world_state_provider=world_provider,
            retarget_policy=retarget_policy,
        )
        target_interaction_service = TargetInteractionService(
            target_interaction_provider=LiveTargetInteractionProvider(runtime, state_detector),
            interaction_world_state_provider=world_provider,
            retarget_policy=retarget_policy,
        )
        target_engagement_service = TargetEngagementService(
            acquisition_service=target_acquisition_service,
            approach_service=target_approach_service,
            interaction_service=target_interaction_service,
        )
        combat_plan_catalog = SimulatedCombatPlanCatalog()
        combat_profile_catalog = SimulatedCombatProfileCatalog(combat_plan_catalog=combat_plan_catalog)
        telemetry_sink = LiveTelemetrySink(storage, logger)
        engage_service = LiveEngageService(
            runtime=runtime,
            target_engagement_service=target_engagement_service,
            state_detector=state_detector,
            input_driver=input_driver,
            artifact_writer=LiveEngageArtifactWriter(
                settings.live.debug_directory,
                live_config=settings.live,
            ),
            verify_delay_s=settings.live.engage_verify_delay_s,
            click_offset_y_px=settings.live.engage_click_offset_y_px,
            target_match_max_distance_px=settings.live.engage_target_match_max_distance_px,
        )
        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            fsm=fsm,
            recovery=recovery,
            observation_provider=LiveObservationProvider(runtime),
            action_executor=LiveActionExecutor(runtime, target_engagement_service),
            verification_provider=LiveVerificationProvider(runtime, state_detector),
            combat_resolver=LiveCombatResolver(
                runtime,
                input_driver=input_driver,
                state_detector=state_detector,
                resource_provider=resource_provider,
                combat_plan_catalog=combat_plan_catalog,
                combat_profile_catalog=combat_profile_catalog,
            ),
            rest_provider=LiveRestProvider(
                runtime,
                input_driver=input_driver,
                resource_provider=resource_provider,
            ),
            telemetry_sink=telemetry_sink,
            cycle_config=settings.cycle,
        )
        return cls(
            orchestrator=orchestrator,
            engage_service=engage_service,
            runtime=runtime,
            storage=storage,
            logger=logger,
        )

    @property
    def storage(self) -> SQLiteTelemetryStorage:
        return self._storage

    @property
    def runtime(self) -> LiveRuntime:
        return self._runtime

    def run_cycles(self, total_cycles: int) -> SimulationReport:
        if total_cycles <= 0:
            raise ValueError("total_cycles musi byc wieksze od 0.")
        self._storage.initialize()
        initial_cycle_id = self._runtime.initial_cycle_id
        cycle_results = self._orchestrator.run_cycles(total_cycles, initial_cycle_id=initial_cycle_id)
        self._runtime.write_perception_session_summary()
        cycle_ids = {initial_cycle_id + offset for offset in range(total_cycles)}
        cycle_records = [
            record
            for record in self._storage.fetch_cycles()
            if int(record["cycle_id"]) in cycle_ids
        ]
        return SimulationReport(
            cycle_results=cycle_results,
            log_path=self._resolve_log_path(),
            sqlite_path=self._storage.sqlite_path,
            observation_preparations=self._runtime.observation_preparations(),
            target_resolutions=self._runtime.target_resolutions(),
            approach_results=self._runtime.approach_results(),
            interaction_results=self._runtime.interaction_results(),
            cycle_records=cycle_records,
        )

    def run_engage_attempts(self, total_attempts: int) -> LiveEngageRunReport:
        if total_attempts <= 0:
            raise ValueError("total_attempts musi byc wieksze od 0.")
        self._storage.initialize()
        results: list[LiveEngageResult] = []
        for offset in range(total_attempts):
            cycle_id = self._next_live_cycle_id + offset
            result = self._engage_service.attempt_engage(cycle_id=cycle_id)
            record_engage_attempt(storage=self._storage, result=result)
            results.append(result)
        self._next_live_cycle_id += total_attempts
        self._runtime.write_perception_session_summary()
        summary = LiveEngageSessionSummary.from_results(results)
        LiveEngageArtifactWriter(
            self._runtime.settings.live.debug_directory,
            live_config=self._runtime.settings.live,
        ).write_session_summary(summary)
        return LiveEngageRunReport(
            results=tuple(results),
            summary=summary,
            log_path=self._resolve_log_path(),
            sqlite_path=self._storage.sqlite_path,
        )

    def _resolve_log_path(self) -> Path:
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename).resolve()
        raise RuntimeError("Logger nie ma skonfigurowanego FileHandler.")
