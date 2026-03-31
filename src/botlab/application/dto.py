from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from botlab.domain.combat_plan import CombatPlan
from botlab.domain.targeting import RetargetDecision
from botlab.domain.world import WorldSnapshot
from botlab.types import BotState, CombatSnapshot, Observation


class VerificationOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ObservationWindow:
    cycle_id: int
    observation: Observation | None
    actual_spawn_ts: float | None
    window_closed_ts: float
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionContext:
    cycle_id: int
    now_ts: float
    predicted_spawn_ts: float | None = None
    observation: Observation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    cycle_id: int
    success: bool
    executed_at_ts: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    cycle_id: int
    outcome: VerificationOutcome
    started_at_ts: float
    completed_at_ts: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimedCombatSnapshot:
    event_ts: float
    snapshot: CombatSnapshot


@dataclass(frozen=True)
class CombatTimeline:
    cycle_id: int
    snapshots: list[TimedCombatSnapshot]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RestTimeline:
    cycle_id: int
    snapshots: list[TimedCombatSnapshot]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CombatPlanSelection:
    plan_name: str
    plan: CombatPlan
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ObservationPreparationResult:
    cycle_id: int
    spawn_zone_visible: bool
    ready_for_observation: bool
    starting_position_xy: tuple[float, float] = (0.0, 0.0)
    observation_position_xy: tuple[float, float] = (0.0, 0.0)
    travel_s: float = 0.0
    arrived_at_ts: float | None = None
    wait_for_spawn_s: float = 0.0
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TargetResolution:
    cycle_id: int
    current_target_id: str | None
    selected_target_id: str | None
    world_snapshot: WorldSnapshot
    decision: RetargetDecision


@dataclass(slots=True, frozen=True)
class TargetApproachResult:
    cycle_id: int
    target_id: str | None
    started_at_ts: float
    completed_at_ts: float
    travel_s: float
    arrived: bool
    reason: str
    initial_target_id: str | None = None
    retargeted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TargetInteractionResult:
    cycle_id: int
    target_id: str | None
    ready: bool
    observed_at_ts: float
    reason: str
    initial_target_id: str | None = None
    retargeted: bool = False
    world_snapshot: WorldSnapshot | None = None
    decision: RetargetDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TargetEngagementResult:
    cycle_id: int
    target_resolution: TargetResolution
    approach_result: TargetApproachResult
    interaction_result: TargetInteractionResult


@dataclass(slots=True, frozen=True)
class CycleRunResult:
    cycle_id: int
    predicted_spawn_ts: float
    actual_spawn_ts: float | None
    drift_s: float | None
    result: str
    final_state: BotState
    reaction_ms: float | None
    verification_ms: float | None
    observation_used: bool
    note: str


@dataclass(slots=True, frozen=True)
class CombatTelemetrySummary:
    key: str
    total_cycles: int
    success_cycles: int
    failure_cycles: int
    no_target_cycles: int
    timeout_cycles: int
    execution_error_cycles: int
    rest_cycles: int
    avg_final_hp_ratio: float | None


@dataclass(slots=True, frozen=True)
class SimulationReport:
    cycle_results: list[CycleRunResult]
    log_path: Path
    sqlite_path: Path
    observation_preparations: list[ObservationPreparationResult] = field(default_factory=list)
    target_resolutions: list[TargetResolution] = field(default_factory=list)
    approach_results: list[TargetApproachResult] = field(default_factory=list)
    interaction_results: list[TargetInteractionResult] = field(default_factory=list)
    cycle_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_cycles(self) -> int:
        return len(self.cycle_results)

    def count_result(self, result: str) -> int:
        return sum(1 for item in self.cycle_results if item.result == result)

    def combat_plan_summaries(self) -> list[CombatTelemetrySummary]:
        return self._build_combat_summaries(group_key="combat_plan_name")

    def combat_profile_summaries(self) -> list[CombatTelemetrySummary]:
        return self._build_combat_summaries(group_key="combat_profile_name")

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "results": {
                "success": self.count_result("success"),
                "failure": self.count_result("failure"),
                "no_event": self.count_result("no_event"),
                "no_target_available": self.count_result("no_target_available"),
                "late_event_missed": self.count_result("late_event_missed"),
                "verify_timeout": self.count_result("verify_timeout"),
                "execution_error": self.count_result("execution_error"),
            },
            "combat_plan_summaries": [
                {
                    "key": summary.key,
                    "total_cycles": summary.total_cycles,
                    "success_cycles": summary.success_cycles,
                    "failure_cycles": summary.failure_cycles,
                    "no_target_cycles": summary.no_target_cycles,
                    "timeout_cycles": summary.timeout_cycles,
                    "execution_error_cycles": summary.execution_error_cycles,
                    "rest_cycles": summary.rest_cycles,
                    "avg_final_hp_ratio": summary.avg_final_hp_ratio,
                }
                for summary in self.combat_plan_summaries()
            ],
            "combat_profile_summaries": [
                {
                    "key": summary.key,
                    "total_cycles": summary.total_cycles,
                    "success_cycles": summary.success_cycles,
                    "failure_cycles": summary.failure_cycles,
                    "no_target_cycles": summary.no_target_cycles,
                    "timeout_cycles": summary.timeout_cycles,
                    "execution_error_cycles": summary.execution_error_cycles,
                    "rest_cycles": summary.rest_cycles,
                    "avg_final_hp_ratio": summary.avg_final_hp_ratio,
                }
                for summary in self.combat_profile_summaries()
            ],
        }

    def decision_trace_lines(self) -> list[str]:
        observation_by_cycle = {
            item.cycle_id: item for item in self.observation_preparations
        }
        resolution_by_cycle = {
            item.cycle_id: item for item in self.target_resolutions
        }
        approach_by_cycle = {item.cycle_id: item for item in self.approach_results}
        interaction_by_cycle = {
            item.cycle_id: item for item in self.interaction_results
        }
        cycle_record_by_cycle = {
            int(record["cycle_id"]): record
            for record in self.cycle_records
            if "cycle_id" in record
        }

        lines: list[str] = []
        for cycle_result in self.cycle_results:
            cycle_id = cycle_result.cycle_id
            observation = observation_by_cycle.get(cycle_id)
            resolution = resolution_by_cycle.get(cycle_id)
            approach = approach_by_cycle.get(cycle_id)
            interaction = interaction_by_cycle.get(cycle_id)
            cycle_record = cycle_record_by_cycle.get(cycle_id, {})
            metadata = cycle_record.get("metadata", {})

            if observation is not None:
                reposition_required = bool(
                    observation.metadata.get("reposition_required", False)
                )
                start_position_source = observation.metadata.get("start_position_source")
                if reposition_required or start_position_source != "scenario_current_position":
                    lines.append(
                        "cycle_trace="
                        f"{cycle_id} "
                        "phase=staging "
                        f"source={start_position_source} "
                        f"start_xy={observation.starting_position_xy} "
                        f"observation_xy={observation.observation_position_xy} "
                        f"travel_s={self._format_optional_float(observation.travel_s)} "
                        f"arrived_ts={self._format_optional_float(observation.arrived_at_ts)}"
                    )
                if observation.wait_for_spawn_s > 0.0:
                    lines.append(
                        "cycle_trace="
                        f"{cycle_id} "
                        "phase=wait "
                        f"wait_for_spawn_s={self._format_optional_float(observation.wait_for_spawn_s)} "
                        f"observation_xy={observation.observation_position_xy}"
                    )
                if not observation.ready_for_observation:
                    lines.append(
                        "cycle_trace="
                        f"{cycle_id} "
                        "phase=staging_missed "
                        f"reason={observation.metadata.get('ready_reason')} "
                        f"arrived_ts={self._format_optional_float(observation.arrived_at_ts)} "
                        f"ready_window_start_ts={self._format_optional_float(observation.metadata.get('ready_window_start_ts'))}"
                    )
                    lines.append(
                        "cycle_trace="
                        f"{cycle_id} "
                        "phase=cycle "
                        f"predicted_spawn_ts={self._format_optional_float(cycle_result.predicted_spawn_ts)} "
                        f"actual_spawn_ts={self._format_optional_float(cycle_result.actual_spawn_ts)} "
                        f"result={cycle_result.result} "
                        f"final_state={cycle_result.final_state.value}"
                    )
                    continue
                lines.append(
                    "cycle_trace="
                    f"{cycle_id} "
                    "phase=observation "
                    f"ts={self._format_optional_float(resolution.world_snapshot.observed_at_ts if resolution is not None else None)} "
                    f"spawn_zone_visible={observation.spawn_zone_visible} "
                    f"ready_for_observation={observation.ready_for_observation}"
                )

            if resolution is not None:
                lines.extend(
                    self._build_target_resolution_lines(
                        cycle_id=cycle_id,
                        resolution=resolution,
                    )
                )

            if approach is not None:
                lines.extend(
                    self._build_approach_lines(
                        cycle_id=cycle_id,
                        approach=approach,
                    )
                )

            if interaction is not None:
                lines.extend(
                    self._build_interaction_lines(
                        cycle_id=cycle_id,
                        interaction=interaction,
                    )
                )

            if metadata.get("combat_completed") is True:
                lines.append(
                    "cycle_trace="
                    f"{cycle_id} "
                    "phase=combat "
                    f"ts={self._format_optional_float(interaction.observed_at_ts if interaction is not None else None)} "
                    f"started_target={metadata.get('selected_target_id')} "
                    f"plan={metadata.get('combat_plan_name')}"
                )
                lines.append(
                    "cycle_trace="
                    f"{cycle_id} "
                    "phase=combat "
                    "result=completed "
                    f"turns={metadata.get('combat_turn_count')} "
                    f"final_hp_ratio={self._format_optional_float(metadata.get('combat_final_hp_ratio'))}"
                )

            if metadata.get("combat_finished_with_rest") is True:
                lines.append(
                    "cycle_trace="
                    f"{cycle_id} "
                    "phase=rest "
                    "result=completed "
                    f"ticks={metadata.get('rest_tick_count')} "
                    f"final_hp_ratio={self._format_optional_float(metadata.get('rest_final_hp_ratio'))}"
                )

            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=cycle "
                f"predicted_spawn_ts={self._format_optional_float(cycle_result.predicted_spawn_ts)} "
                f"actual_spawn_ts={self._format_optional_float(cycle_result.actual_spawn_ts)} "
                f"result={cycle_result.result} "
                f"final_state={cycle_result.final_state.value}"
            )

        return lines

    def _build_combat_summaries(self, *, group_key: str) -> list[CombatTelemetrySummary]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in self.cycle_records:
            metadata = record.get("metadata", {})
            key = metadata.get(group_key)
            if not isinstance(key, str) or not key:
                continue
            grouped.setdefault(key, []).append(record)

        summaries: list[CombatTelemetrySummary] = []
        for key in sorted(grouped):
            records = grouped[key]
            final_hp_values = [
                float(record["metadata"]["combat_final_hp_ratio"])
                for record in records
                if record.get("metadata", {}).get("combat_final_hp_ratio") is not None
            ]
            avg_final_hp_ratio = None
            if final_hp_values:
                avg_final_hp_ratio = round(sum(final_hp_values) / len(final_hp_values), 6)

            summaries.append(
                CombatTelemetrySummary(
                    key=key,
                    total_cycles=len(records),
                    success_cycles=sum(1 for record in records if record.get("result") == "success"),
                    failure_cycles=sum(1 for record in records if record.get("result") == "failure"),
                    no_target_cycles=sum(
                        1 for record in records if record.get("result") == "no_target_available"
                    ),
                    timeout_cycles=sum(
                        1 for record in records if record.get("result") == "verify_timeout"
                    ),
                    execution_error_cycles=sum(
                        1 for record in records if record.get("result") == "execution_error"
                    ),
                    rest_cycles=sum(
                        1
                        for record in records
                        if record.get("metadata", {}).get("combat_finished_with_rest") is True
                    ),
                    avg_final_hp_ratio=avg_final_hp_ratio,
                )
            )

        return summaries

    def _build_target_resolution_lines(
        self,
        *,
        cycle_id: int,
        resolution: TargetResolution,
    ) -> list[str]:
        world = resolution.world_snapshot
        selected_target_id = resolution.selected_target_id
        lines: list[str] = []

        if not world.spawn_zone_visible:
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=targeting "
                "status=spawn_zone_not_visible"
            )
            return lines

        if not world.groups:
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=targeting "
                f"selected=None reason={resolution.decision.reason}"
            )
            return lines

        selected_group = None if selected_target_id is None else world.group_by_id(selected_target_id)
        selected_distance = None if selected_group is None else selected_group.distance
        lines.append(
            "cycle_trace="
            f"{cycle_id} "
            "phase=targeting "
            f"ts={self._format_optional_float(world.observed_at_ts)} "
            f"selected={selected_target_id} "
            f"distance={self._format_optional_float(selected_distance)} "
            f"reason={resolution.decision.reason}"
        )

        for group in sorted(world.groups, key=lambda item: (item.distance, item.group_id)):
            if group.group_id == selected_target_id:
                continue
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=targeting "
                f"rejected={group.group_id} "
                f"reason={self._group_rejection_reason(group, selected_target_id is not None)} "
                f"distance={self._format_optional_float(group.distance)}"
            )

        return lines

    def _build_approach_lines(
        self,
        *,
        cycle_id: int,
        approach: TargetApproachResult,
    ) -> list[str]:
        lines: list[str] = []
        movement_steps = approach.metadata.get("movement_steps", [])
        if isinstance(movement_steps, list) and len(movement_steps) > 1:
            for raw_step in movement_steps:
                if not isinstance(raw_step, dict):
                    continue
                lines.append(
                    "cycle_trace="
                    f"{cycle_id} "
                    "phase=approach_step "
                    f"target={approach.initial_target_id or approach.target_id} "
                    f"step={raw_step.get('step_index')} "
                    f"arrived_ts={self._format_optional_float(raw_step.get('arrived_ts'))} "
                    f"remaining_distance={self._format_optional_float(raw_step.get('remaining_distance'))}"
                )

        revalidation_step_index = approach.metadata.get("revalidation_step_index")
        if isinstance(revalidation_step_index, int):
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=approach_revalidate "
                f"target={approach.initial_target_id or approach.target_id} "
                f"step={revalidation_step_index} "
                f"reason={approach.metadata.get('revalidation_reason')}"
            )

        if approach.initial_target_id is None and approach.target_id is None:
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=approach "
                f"status=skipped reason={approach.reason}"
            )
            return lines

        if approach.retargeted and approach.initial_target_id is not None:
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=approach "
                f"ts={self._format_optional_float(approach.completed_at_ts)} "
                f"retarget_from={approach.initial_target_id} "
                f"retarget_to={approach.target_id} "
                f"reason={approach.metadata.get('revalidation_reason', approach.reason)}"
            )

        if approach.target_id is None:
            lines.append(
                "cycle_trace="
                f"{cycle_id} "
                "phase=approach "
                f"ts={self._format_optional_float(approach.completed_at_ts)} "
                f"lost_target={approach.initial_target_id} "
                f"reason={approach.reason}"
            )
            return lines

        lines.append(
            "cycle_trace="
            f"{cycle_id} "
            "phase=approach "
            f"ts={self._format_optional_float(approach.completed_at_ts)} "
            f"arrived_target={approach.target_id} "
            f"travel_s={self._format_optional_float(approach.travel_s)} "
            f"reason={approach.reason}"
        )
        return lines

    def _build_interaction_lines(
        self,
        *,
        cycle_id: int,
        interaction: TargetInteractionResult,
    ) -> list[str]:
        if interaction.ready:
            return [
                "cycle_trace="
                f"{cycle_id} "
                "phase=interaction "
                f"ts={self._format_optional_float(interaction.observed_at_ts)} "
                f"target={interaction.target_id} "
                f"ready={interaction.ready} "
                f"reason={interaction.reason}"
            ]

        if interaction.reason == "retarget_before_interaction_requires_new_approach":
            return [
                "cycle_trace="
                f"{cycle_id} "
                "phase=interaction "
                f"ts={self._format_optional_float(interaction.observed_at_ts)} "
                f"retarget_to={interaction.target_id} "
                f"reason={interaction.metadata.get('revalidation_reason', interaction.reason)}"
            ]

        return [
            "cycle_trace="
            f"{cycle_id} "
            "phase=interaction "
            f"ts={self._format_optional_float(interaction.observed_at_ts)} "
            f"target={interaction.target_id} "
            f"ready={interaction.ready} "
            f"reason={interaction.reason}"
        ]

    @staticmethod
    def _group_rejection_reason(group: Any, has_selected_target: bool) -> str:
        if not group.is_alive:
            return "defeated"
        if group.engaged_by_other:
            return "engaged_by_other"
        if not group.reachable:
            return "unreachable"
        if has_selected_target:
            return "better_target_selected"
        return "not_selected"

    @staticmethod
    def _format_optional_float(value: object) -> str:
        if value is None:
            return "None"
        return f"{float(value):.3f}"
