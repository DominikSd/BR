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
