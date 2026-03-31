from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    return CycleFSM(
        decision_engine=DecisionEngine(_cycle_config(), _combat_config()),
        initial_state=BotState.IDLE,
        started_at_ts=0.0,
        cycle_id=None,
    )


def _observation(*, cycle_id: int = 1, observed_at_ts: float, actual_spawn_ts: float) -> Observation:
    return Observation(
        cycle_id=cycle_id,
        observed_at_ts=observed_at_ts,
        signal_detected=True,
        actual_spawn_ts=actual_spawn_ts,
    )


def _combat_snapshot(
    *,
    event_ts: float,
    hp_ratio: float,
    in_combat: bool,
    turn_index: int = 1,
) -> TimedCombatSnapshot:
    return TimedCombatSnapshot(
        event_ts=event_ts,
        snapshot=CombatSnapshot(
            hp_ratio=hp_ratio,
            turn_index=turn_index,
            enemy_count=1 if in_combat else 0,
            strategy="default",
            in_combat=in_combat,
            combat_started_ts=event_ts - 0.3,
            combat_finished_ts=None if in_combat else event_ts,
        ),
    )


def _rest_snapshot(*, event_ts: float, hp_ratio: float) -> TimedCombatSnapshot:
    return TimedCombatSnapshot(
        event_ts=event_ts,
        snapshot=CombatSnapshot(
            hp_ratio=hp_ratio,
            turn_index=1,
            enemy_count=0,
            strategy="rest",
            in_combat=False,
        ),
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


@dataclass
class ScenarioPorts:
    observation_window: ObservationWindow
    action_result: ActionResult | None = None
    verification_result: VerificationResult | None = None
    combat_timeline: CombatTimeline | None = None
    rest_timeline: RestTimeline | None = None
    combat_error: Exception | None = None
    rest_error: Exception | None = None
    action_calls: int = 0
    verify_calls: int = 0
    combat_calls: int = 0
    rest_calls: int = 0

    def get_observation_window(self, cycle_id: int) -> ObservationWindow:
        observation = self.observation_window.observation
        if observation is not None:
            observation = replace(observation, cycle_id=cycle_id)
        return replace(self.observation_window, cycle_id=cycle_id, observation=observation)

    def execute_action(self, context: ActionContext) -> ActionResult:
        self.action_calls += 1
        if self.action_result is None:
            raise AssertionError("Action should not be executed in this scenario.")
        return replace(self.action_result, cycle_id=context.cycle_id)

    def verify(self, cycle_id: int, observation: Observation) -> VerificationResult:
        self.verify_calls += 1
        if self.verification_result is None:
            raise AssertionError("Verification should not run in this scenario.")
        return replace(self.verification_result, cycle_id=cycle_id)

    def resolve_combat(
        self,
        cycle_id: int,
        *,
        combat_started_ts: float,
        observation: Observation,
    ) -> CombatTimeline:
        self.combat_calls += 1
        if self.combat_error is not None:
            raise self.combat_error
        if self.combat_timeline is None:
            raise AssertionError("Combat should not run in this scenario.")
        return self.combat_timeline

    def apply_rest(
        self,
        cycle_id: int,
        *,
        rest_started_ts: float,
        starting_hp_ratio: float,
        observation: Observation,
    ) -> RestTimeline:
        self.rest_calls += 1
        if self.rest_error is not None:
            raise self.rest_error
        if self.rest_timeline is None:
            raise AssertionError("Rest should not run in this scenario.")
        return self.rest_timeline


def _build_orchestrator(
    ports: ScenarioPorts,
    telemetry_sink: RecordingTelemetrySink | None = None,
) -> tuple[CycleOrchestrator, RecordingTelemetrySink]:
    telemetry = telemetry_sink or RecordingTelemetrySink()
    orchestrator = CycleOrchestrator(
        scheduler=_build_scheduler(),
        fsm=_build_fsm(),
        recovery=RecoveryManager(_cycle_config()),
        observation_provider=ports,
        action_executor=ports,
        verification_provider=ports,
        combat_resolver=ports,
        rest_provider=ports,
        telemetry_sink=telemetry,
        cycle_config=_cycle_config(),
    )
    return orchestrator, telemetry


class TestScenario1SuccessWithRestRequired:
    def test_success_with_combat_rest_completes_with_wait_next_cycle_state(self) -> None:
        observation = _observation(observed_at_ts=160.0, actual_spawn_ts=160.0)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=160.0,
                window_closed_ts=165.01,
                note="rest-required",
            ),
            action_result=ActionResult(
                cycle_id=1,
                success=True,
                executed_at_ts=160.02,
                reason="action_executed",
            ),
            verification_result=VerificationResult(
                cycle_id=1,
                outcome=VerificationOutcome.SUCCESS,
                started_at_ts=160.03,
                completed_at_ts=160.13,
                reason="success",
            ),
            combat_timeline=CombatTimeline(
                cycle_id=1,
                snapshots=[_combat_snapshot(event_ts=160.60, hp_ratio=0.40, in_combat=False)],
            ),
            rest_timeline=RestTimeline(
                cycle_id=1,
                snapshots=[_rest_snapshot(event_ts=161.10, hp_ratio=0.95)],
            ),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "success"
        assert results[0].final_state is BotState.WAIT_NEXT_CYCLE
        assert ports.combat_calls == 1
        assert ports.rest_calls == 1
        assert any(item.state_exit == BotState.REST for item in telemetry.transitions)
        assert any(item.reason == "rest_completed_hp_restored" for item in telemetry.transitions)


class TestScenario2SuccessWithoutRestRequired:
    def test_success_with_high_hp_no_rest_required(self) -> None:
        observation = _observation(observed_at_ts=160.0, actual_spawn_ts=160.0)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=160.0,
                window_closed_ts=165.01,
                note="no-rest",
            ),
            action_result=ActionResult(
                cycle_id=1,
                success=True,
                executed_at_ts=160.02,
                reason="action_executed",
            ),
            verification_result=VerificationResult(
                cycle_id=1,
                outcome=VerificationOutcome.SUCCESS,
                started_at_ts=160.03,
                completed_at_ts=160.13,
                reason="success",
            ),
            combat_timeline=CombatTimeline(
                cycle_id=1,
                snapshots=[_combat_snapshot(event_ts=160.60, hp_ratio=0.95, in_combat=False)],
            ),
            rest_timeline=RestTimeline(cycle_id=1, snapshots=[]),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "success"
        assert results[0].final_state is BotState.WAIT_NEXT_CYCLE
        assert ports.combat_calls == 1
        assert ports.rest_calls == 0
        assert any(item.reason == "combat_finished_no_rest_needed" for item in telemetry.transitions)


class TestScenario3VerifyTimeoutWithRecovery:
    def test_verify_timeout_triggers_recovery_and_returns_to_wait_next_cycle(self) -> None:
        observation = _observation(observed_at_ts=160.0, actual_spawn_ts=160.0)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=160.0,
                window_closed_ts=165.01,
                note="timeout",
            ),
            action_result=ActionResult(
                cycle_id=1,
                success=True,
                executed_at_ts=160.02,
                reason="action_executed",
            ),
            verification_result=VerificationResult(
                cycle_id=1,
                outcome=VerificationOutcome.TIMEOUT,
                started_at_ts=160.03,
                completed_at_ts=160.53,
                reason="timeout",
            ),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "verify_timeout"
        assert results[0].final_state is BotState.WAIT_NEXT_CYCLE
        assert ports.combat_calls == 0
        assert ports.rest_calls == 0
        assert any(item.state_exit == BotState.RECOVER for item in telemetry.transitions)
        assert any(item.reason == "recover_timeout_elapsed" for item in telemetry.transitions)


class TestScenario4ExecutionErrorWithRecovery:
    def test_exception_during_success_path_triggers_recovery(self) -> None:
        observation = _observation(observed_at_ts=160.0, actual_spawn_ts=160.0)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=160.0,
                window_closed_ts=165.01,
                note="execution-error",
            ),
            action_result=ActionResult(
                cycle_id=1,
                success=True,
                executed_at_ts=160.02,
                reason="action_executed",
            ),
            verification_result=VerificationResult(
                cycle_id=1,
                outcome=VerificationOutcome.SUCCESS,
                started_at_ts=160.03,
                completed_at_ts=160.13,
                reason="success",
            ),
            combat_error=RuntimeError("Combat system malfunction"),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "execution_error"
        assert results[0].final_state is BotState.WAIT_NEXT_CYCLE
        assert ports.combat_calls == 1
        assert ports.rest_calls == 0
        assert any(item.state_exit == BotState.RECOVER for item in telemetry.transitions)
        assert any(
            item.reason == "execution_error_RuntimeError_reset_complete"
            for item in telemetry.transitions
        )


class TestScenario5NoEvent:
    def test_no_observation_results_in_no_event(self) -> None:
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=None,
                actual_spawn_ts=None,
                window_closed_ts=165.01,
                note="no-event",
            ),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "no_event"
        assert results[0].observation_used is False
        assert ports.action_calls == 0
        assert ports.verify_calls == 0
        assert ports.combat_calls == 0
        assert ports.rest_calls == 0
        assert telemetry.cycles[0].result == "no_event"


class TestScenario6LateEventMissed:
    def test_event_predicted_but_not_observed_is_late_missed(self) -> None:
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=None,
                actual_spawn_ts=166.5,
                window_closed_ts=165.01,
                note="late-event",
            ),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "late_event_missed"
        assert results[0].observation_used is False
        assert results[0].drift_s == pytest.approx(6.5)
        assert telemetry.cycles[0].result == "late_event_missed"


class TestScenario7VisibleEventButNoTargetAvailable:
    def test_visible_event_without_free_target_completes_without_attempt(self) -> None:
        observation = _observation(observed_at_ts=160.0, actual_spawn_ts=160.0)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=160.0,
                window_closed_ts=165.01,
                note="no-target",
            ),
            action_result=ActionResult(
                cycle_id=1,
                success=False,
                executed_at_ts=160.0,
                reason="no_target_available",
                metadata={
                    "selected_target_id": None,
                    "target_decision_reason": "no_target_available",
                },
            ),
        )
        orchestrator, telemetry = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].result == "no_target_available"
        assert results[0].final_state is BotState.WAIT_NEXT_CYCLE
        assert results[0].observation_used is True
        assert ports.action_calls == 1
        assert ports.verify_calls == 0
        assert ports.combat_calls == 0
        assert ports.rest_calls == 0
        assert telemetry.attempts == []
        assert telemetry.cycles[0].result == "no_target_available"
        assert telemetry.cycles[0].metadata["selected_target_id"] is None
        assert telemetry.cycles[0].metadata["target_decision_reason"] == "no_target_available"


class TestMultipleCycles:
    def test_run_multiple_cycles_produces_correct_count(self) -> None:
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=None,
                actual_spawn_ts=None,
                window_closed_ts=165.01,
            ),
        )
        orchestrator, _ = _build_orchestrator(ports)

        results = orchestrator.run_cycles(3, initial_cycle_id=1)

        assert [item.cycle_id for item in results] == [1, 2, 3]


class TestDriftCalculation:
    def test_positive_drift_when_event_is_late(self) -> None:
        observation = _observation(observed_at_ts=161.5, actual_spawn_ts=161.5)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=161.5,
                window_closed_ts=165.01,
            ),
            action_result=ActionResult(cycle_id=1, success=True, executed_at_ts=161.52),
            verification_result=VerificationResult(
                cycle_id=1,
                outcome=VerificationOutcome.SUCCESS,
                started_at_ts=161.53,
                completed_at_ts=161.63,
            ),
            combat_timeline=CombatTimeline(
                cycle_id=1,
                snapshots=[_combat_snapshot(event_ts=162.0, hp_ratio=0.95, in_combat=False)],
            ),
            rest_timeline=RestTimeline(cycle_id=1, snapshots=[]),
        )
        orchestrator, _ = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].drift_s == pytest.approx(1.5)

    def test_negative_drift_when_event_is_early(self) -> None:
        observation = _observation(observed_at_ts=158.8, actual_spawn_ts=158.8)
        ports = ScenarioPorts(
            observation_window=ObservationWindow(
                cycle_id=1,
                observation=observation,
                actual_spawn_ts=158.8,
                window_closed_ts=165.01,
            ),
            action_result=ActionResult(cycle_id=1, success=True, executed_at_ts=158.82),
            verification_result=VerificationResult(
                cycle_id=1,
                outcome=VerificationOutcome.SUCCESS,
                started_at_ts=158.83,
                completed_at_ts=158.93,
            ),
            combat_timeline=CombatTimeline(
                cycle_id=1,
                snapshots=[_combat_snapshot(event_ts=159.3, hp_ratio=0.95, in_combat=False)],
            ),
            rest_timeline=RestTimeline(cycle_id=1, snapshots=[]),
        )
        orchestrator, _ = _build_orchestrator(ports)

        results = orchestrator.run_cycles(1, initial_cycle_id=1)

        assert results[0].drift_s == pytest.approx(-1.2)
