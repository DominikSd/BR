from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from botlab.application import (
    ActionContext,
    ActionResult,
    CombatTimeline,
    CycleOrchestrator,
    ObservationWindow,
    RestTimeline,
    TimedCombatSnapshot,
    VerificationOutcome,
    VerificationResult,
)
from botlab.config import CombatConfig, CycleConfig
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, CombatSnapshot, Observation, TelemetryRecord


def _cycle_config() -> CycleConfig:
    return CycleConfig(
        interval_s=60.0,
        prepare_before_s=10.0,
        ready_before_s=5.0,
        ready_after_s=5.0,
        verify_timeout_s=0.5,
        recover_timeout_s=2.0,
    )


def _combat_config() -> CombatConfig:
    return CombatConfig(
        low_hp_threshold=0.35,
        rest_start_threshold=0.50,
        rest_stop_threshold=0.90,
    )


def _build_scheduler() -> CycleScheduler:
    scheduler = CycleScheduler.from_cycle_config(_cycle_config())
    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    return scheduler


def _build_fsm() -> CycleFSM:
    decision_engine = DecisionEngine(_cycle_config(), _combat_config())
    return CycleFSM(
        decision_engine=decision_engine,
        initial_state=BotState.IDLE,
        started_at_ts=0.0,
        cycle_id=None,
    )


@dataclass
class RecordingTelemetrySink:
    cycles: list[TelemetryRecord] = field(default_factory=list)
    attempts: list[TelemetryRecord] = field(default_factory=list)
    transitions: list[TelemetryRecord] = field(default_factory=list)

    def record_cycle(self, record: TelemetryRecord) -> None:
        self.cycles.append(record)

    def record_attempt(self, record: TelemetryRecord) -> None:
        self.attempts.append(record)

    def record_state_transition(self, record: TelemetryRecord) -> None:
        self.transitions.append(record)


class StaticObservationProvider:
    def __init__(self, window: ObservationWindow) -> None:
        self._window = window

    def get_observation_window(self, cycle_id: int) -> ObservationWindow:
        assert cycle_id == self._window.cycle_id
        return self._window


class StaticActionExecutor:
    def __init__(self, executed_at_ts: float) -> None:
        self._executed_at_ts = executed_at_ts

    def execute_action(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            cycle_id=context.cycle_id,
            success=True,
            executed_at_ts=self._executed_at_ts,
            reason="action_executed",
        )


class StaticVerificationProvider:
    def __init__(self, result: VerificationResult) -> None:
        self._result = result

    def verify(self, cycle_id: int, observation: Observation) -> VerificationResult:
        assert cycle_id == self._result.cycle_id
        return self._result


class StaticCombatResolver:
    def __init__(self, timeline: CombatTimeline) -> None:
        self._timeline = timeline

    def resolve_combat(
        self,
        cycle_id: int,
        *,
        combat_started_ts: float,
        observation: Observation,
    ) -> CombatTimeline:
        assert cycle_id == self._timeline.cycle_id
        assert combat_started_ts <= self._timeline.snapshots[0].event_ts
        return self._timeline


class StaticRestProvider:
    def __init__(self, timeline: RestTimeline) -> None:
        self._timeline = timeline

    def apply_rest(
        self,
        cycle_id: int,
        *,
        rest_started_ts: float,
        starting_hp_ratio: float,
        observation: Observation,
    ) -> RestTimeline:
        assert cycle_id == self._timeline.cycle_id
        assert rest_started_ts <= self._timeline.snapshots[0].event_ts
        return self._timeline


def _build_orchestrator(
    *,
    observation_window: ObservationWindow,
    action_executed_at_ts: float,
    verification_result: VerificationResult,
    combat_timeline: CombatTimeline,
    rest_timeline: RestTimeline,
    telemetry_sink: RecordingTelemetrySink | None = None,
) -> tuple[CycleOrchestrator, RecordingTelemetrySink]:
    telemetry = telemetry_sink or RecordingTelemetrySink()
    orchestrator = CycleOrchestrator(
        scheduler=_build_scheduler(),
        fsm=_build_fsm(),
        recovery=RecoveryManager(_cycle_config()),
        observation_provider=StaticObservationProvider(observation_window),
        action_executor=StaticActionExecutor(action_executed_at_ts),
        verification_provider=StaticVerificationProvider(verification_result),
        combat_resolver=StaticCombatResolver(combat_timeline),
        rest_provider=StaticRestProvider(rest_timeline),
        telemetry_sink=telemetry,
        cycle_config=_cycle_config(),
    )
    return orchestrator, telemetry


def _success_observation() -> Observation:
    return Observation(
        cycle_id=1,
        observed_at_ts=160.0,
        signal_detected=True,
        actual_spawn_ts=160.0,
    )


def _combat_snapshot(*, event_ts: float, hp_ratio: float, in_combat: bool) -> TimedCombatSnapshot:
    snapshot = CombatSnapshot(
        hp_ratio=hp_ratio,
        turn_index=1,
        enemy_count=0 if not in_combat else 1,
        strategy="default",
        in_combat=in_combat,
        combat_started_ts=event_ts - 0.1,
        combat_finished_ts=None if in_combat else event_ts,
    )
    return TimedCombatSnapshot(event_ts=event_ts, snapshot=snapshot)


def _rest_snapshot(*, event_ts: float, hp_ratio: float) -> TimedCombatSnapshot:
    snapshot = CombatSnapshot(
        hp_ratio=hp_ratio,
        turn_index=1,
        enemy_count=0,
        strategy="rest",
        in_combat=False,
    )
    return TimedCombatSnapshot(event_ts=event_ts, snapshot=snapshot)


def test_run_cycles_zero_cycles_raises_value_error() -> None:
    observation = _success_observation()
    observation_window = ObservationWindow(
        cycle_id=1,
        observation=observation,
        actual_spawn_ts=160.0,
        window_closed_ts=165.0,
    )
    verification_result = VerificationResult(
        cycle_id=1,
        outcome=VerificationOutcome.SUCCESS,
        started_at_ts=160.03,
        completed_at_ts=160.13,
    )
    orchestrator, _ = _build_orchestrator(
        observation_window=observation_window,
        action_executed_at_ts=160.02,
        verification_result=verification_result,
        combat_timeline=CombatTimeline(cycle_id=1, snapshots=[_combat_snapshot(event_ts=160.5, hp_ratio=0.95, in_combat=False)]),
        rest_timeline=RestTimeline(cycle_id=1, snapshots=[]),
    )

    with pytest.raises(ValueError, match="total_cycles musi być większe od 0"):
        orchestrator.run_cycles(0)


def test_run_cycles_negative_cycles_raises_value_error() -> None:
    observation = _success_observation()
    observation_window = ObservationWindow(
        cycle_id=1,
        observation=observation,
        actual_spawn_ts=160.0,
        window_closed_ts=165.0,
    )
    verification_result = VerificationResult(
        cycle_id=1,
        outcome=VerificationOutcome.SUCCESS,
        started_at_ts=160.03,
        completed_at_ts=160.13,
    )
    orchestrator, _ = _build_orchestrator(
        observation_window=observation_window,
        action_executed_at_ts=160.02,
        verification_result=verification_result,
        combat_timeline=CombatTimeline(cycle_id=1, snapshots=[_combat_snapshot(event_ts=160.5, hp_ratio=0.95, in_combat=False)]),
        rest_timeline=RestTimeline(cycle_id=1, snapshots=[]),
    )

    with pytest.raises(ValueError, match="total_cycles musi być większe od 0"):
        orchestrator.run_cycles(-1)


def test_run_cycles_with_observation_success() -> None:
    observation = _success_observation()
    observation_window = ObservationWindow(
        cycle_id=1,
        observation=observation,
        actual_spawn_ts=160.0,
        window_closed_ts=165.0,
        note="success-no-rest",
    )
    verification_result = VerificationResult(
        cycle_id=1,
        outcome=VerificationOutcome.SUCCESS,
        started_at_ts=160.03,
        completed_at_ts=160.13,
        reason="success",
    )
    orchestrator, telemetry = _build_orchestrator(
        observation_window=observation_window,
        action_executed_at_ts=160.02,
        verification_result=verification_result,
        combat_timeline=CombatTimeline(
            cycle_id=1,
            snapshots=[_combat_snapshot(event_ts=160.5, hp_ratio=0.95, in_combat=False)],
        ),
        rest_timeline=RestTimeline(cycle_id=1, snapshots=[]),
    )

    results = orchestrator.run_cycles(1, initial_cycle_id=1)

    assert len(results) == 1
    result = results[0]
    assert result.cycle_id == 1
    assert result.result == "success"
    assert result.final_state is BotState.WAIT_NEXT_CYCLE
    assert result.observation_used is True
    assert result.reaction_ms == pytest.approx(20.0)
    assert result.verification_ms == pytest.approx(100.0)
    assert telemetry.attempts[0].result == "success"
    assert telemetry.cycles[0].result == "success"
