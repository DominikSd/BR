from __future__ import annotations

from botlab.application.dto import (
    ActionContext,
    CycleRunResult,
    ObservationWindow,
    TimedCombatSnapshot,
    VerificationOutcome,
    VerificationResult,
)
from botlab.application.ports import (
    ActionExecutor,
    CombatResolver,
    ObservationProvider,
    RestProvider,
    TelemetrySink,
    VerificationProvider,
)
from botlab.config import CycleConfig
from botlab.domain.fsm import CycleFSM, StateTransition
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, Decision, Observation, TelemetryRecord


class CycleOrchestrator:
    """
    Neutralny orchestrator cyklu.

    Orkiestruje przebieg cyklu używając:
    - scheduler dla predykcji,
    - FSM dla stanów,
    - recovery dla odzysku,
    - portów application dla zewnętrznych zależności.

    Logika application nie zna szczegółów symulacyjnych opóźnień ani timeline.
    """

    def __init__(
        self,
        *,
        scheduler: CycleScheduler,
        fsm: CycleFSM,
        recovery: RecoveryManager,
        observation_provider: ObservationProvider,
        action_executor: ActionExecutor,
        verification_provider: VerificationProvider,
        combat_resolver: CombatResolver,
        rest_provider: RestProvider,
        telemetry_sink: TelemetrySink,
        cycle_config: CycleConfig,
    ) -> None:
        self._scheduler = scheduler
        self._fsm = fsm
        self._recovery = recovery
        self._observation_provider = observation_provider
        self._action_executor = action_executor
        self._verification_provider = verification_provider
        self._combat_resolver = combat_resolver
        self._rest_provider = rest_provider
        self._telemetry_sink = telemetry_sink
        self._cycle_config = cycle_config

    def run_cycles(self, total_cycles: int, initial_cycle_id: int = 0) -> list[CycleRunResult]:
        if total_cycles <= 0:
            raise ValueError("total_cycles musi być większe od 0.")

        cycle_results: list[CycleRunResult] = []

        for cycle_offset in range(total_cycles):
            cycle_id = initial_cycle_id + cycle_offset
            cycle_results.append(self._run_single_cycle(cycle_id))

        return cycle_results

    def _run_single_cycle(self, cycle_id: int) -> CycleRunResult:
        prediction = self._scheduler.prediction_for_cycle(cycle_id)

        self._tick_and_record_transition(
            now_ts=prediction.prepare_window_start_ts,
            prediction=prediction,
            actual_spawn_ts=None,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )
        self._tick_and_record_transition(
            now_ts=prediction.ready_window_start_ts,
            prediction=prediction,
            actual_spawn_ts=None,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        observation_window = self._observation_provider.get_observation_window(cycle_id)

        if observation_window.observation is None:
            return self._complete_cycle_without_observation(
                cycle_id=cycle_id,
                prediction=prediction,
                observation_window=observation_window,
            )

        return self._complete_cycle_with_observation(
            cycle_id=cycle_id,
            prediction=prediction,
            observation_window=observation_window,
        )

    def _complete_cycle_without_observation(
        self,
        *,
        cycle_id: int,
        prediction,
        observation_window: ObservationWindow,
    ) -> CycleRunResult:
        decision = self._tick_and_record_transition(
            now_ts=observation_window.window_closed_ts,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        event_in_ready_window = (
            observation_window.actual_spawn_ts is not None
            and prediction.ready_window_start_ts
            <= observation_window.actual_spawn_ts
            <= prediction.ready_window_end_ts
        )

        if observation_window.actual_spawn_ts is None:
            result = "no_event"
            reason = decision.reason
        elif event_in_ready_window:
            result = "no_event"
            reason = "event_not_observable_in_ready_window"
        else:
            result = "late_event_missed"
            reason = "actual_event_outside_ready_window"

        final_state = self._fsm.current_state
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation_window.actual_spawn_ts,
        )

        self._telemetry_sink.record_cycle(
            TelemetryRecord(
                cycle_id=cycle_id,
                event_ts=observation_window.window_closed_ts,
                state=final_state,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                reason=reason,
                reaction_ms=None,
                verification_ms=None,
                result=result,
                final_state=final_state,
                metadata={
                    "note": observation_window.note,
                    "observation_used": False,
                    **observation_window.metadata,
                },
            )
        )

        return CycleRunResult(
            cycle_id=cycle_id,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            drift_s=drift_s,
            result=result,
            final_state=final_state,
            reaction_ms=None,
            verification_ms=None,
            observation_used=False,
            note=observation_window.note,
        )

    def _complete_cycle_with_observation(
        self,
        *,
        cycle_id: int,
        prediction,
        observation_window: ObservationWindow,
    ) -> CycleRunResult:
        observation = observation_window.observation
        assert observation is not None

        self._scheduler.register_observation(observation)

        action_context = ActionContext(
            cycle_id=cycle_id,
            now_ts=observation.observed_at_ts,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            observation=observation,
            metadata=dict(observation_window.metadata),
        )
        action_result = self._action_executor.execute_action(action_context)
        action_metadata = dict(action_result.metadata)
        if not action_result.success:
            if action_result.reason == "no_target_available":
                return self._complete_cycle_without_target(
                    cycle_id=cycle_id,
                    prediction=prediction,
                    observation_window=observation_window,
                    completed_at_ts=observation_window.window_closed_ts,
                    action_metadata=action_metadata,
                )
            if action_result.reason == "approach_failed":
                return self._complete_cycle_without_target(
                    cycle_id=cycle_id,
                    prediction=prediction,
                    observation_window=observation_window,
                    completed_at_ts=action_result.executed_at_ts,
                    action_metadata=action_metadata,
                    result="approach_failed",
                    reason="approach_failed",
                )
            raise RuntimeError(f"action_execution_failed: {action_result.reason}")

        attempt_decision = self._tick_and_record_transition(
            now_ts=action_result.executed_at_ts,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=observation,
            verify_result=None,
            combat_snapshot=None,
        )

        verification_result = self._verification_provider.verify(cycle_id, observation)
        self._tick_and_record_transition(
            now_ts=verification_result.started_at_ts,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        reaction_ms = (action_result.executed_at_ts - observation.observed_at_ts) * 1000.0
        verification_ms = (
            verification_result.completed_at_ts - verification_result.started_at_ts
        ) * 1000.0
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation_window.actual_spawn_ts,
        )

        if verification_result.outcome == VerificationOutcome.TIMEOUT:
            self._handle_verify_timeout_transition(
                prediction=prediction,
                observation_window=observation_window,
                verification_result=verification_result,
            )

            final_state = self._fsm.current_state
            self._telemetry_sink.record_attempt(
                TelemetryRecord(
                    cycle_id=cycle_id,
                    event_ts=verification_result.completed_at_ts,
                    state=BotState.ATTEMPT,
                    expected_spawn_ts=prediction.predicted_spawn_ts,
                    actual_spawn_ts=observation_window.actual_spawn_ts,
                    drift_s=drift_s,
                    reason=verification_result.reason or "verify_timeout",
                    reaction_ms=reaction_ms,
                    verification_ms=verification_ms,
                    result="verify_timeout",
                    final_state=final_state,
                    metadata={
                        "note": observation_window.note,
                        "attempt_action": attempt_decision.action,
                        **action_metadata,
                        **verification_result.metadata,
                    },
                )
            )
            self._telemetry_sink.record_cycle(
                TelemetryRecord(
                    cycle_id=cycle_id,
                    event_ts=verification_result.completed_at_ts
                    + self._cycle_config.recover_timeout_s,
                    state=final_state,
                    expected_spawn_ts=prediction.predicted_spawn_ts,
                    actual_spawn_ts=observation_window.actual_spawn_ts,
                    drift_s=drift_s,
                    reason=verification_result.reason or "verify_timeout",
                    reaction_ms=reaction_ms,
                    verification_ms=verification_ms,
                    result="verify_timeout",
                    final_state=final_state,
                    metadata={
                        "note": observation_window.note,
                        "observation_used": True,
                        **action_metadata,
                        **observation_window.metadata,
                    },
                )
            )

            return CycleRunResult(
                cycle_id=cycle_id,
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                result="verify_timeout",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note=observation_window.note,
            )

        verify_decision = self._tick_and_record_transition(
            now_ts=verification_result.completed_at_ts,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=None,
            verify_result=verification_result.outcome.value,
            combat_snapshot=None,
        )

        result = (
            "success"
            if verification_result.outcome == VerificationOutcome.SUCCESS
            else "failure"
        )
        self._telemetry_sink.record_attempt(
            TelemetryRecord(
                cycle_id=cycle_id,
                event_ts=verification_result.completed_at_ts,
                state=BotState.ATTEMPT,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                reason=verify_decision.reason,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result=result,
                final_state=self._fsm.current_state,
                metadata={
                    "note": observation_window.note,
                    "attempt_action": attempt_decision.action,
                    **action_metadata,
                    **verification_result.metadata,
                },
            )
        )

        if result == "failure":
            final_state = self._fsm.current_state
            self._telemetry_sink.record_cycle(
                TelemetryRecord(
                    cycle_id=cycle_id,
                    event_ts=verification_result.completed_at_ts,
                    state=final_state,
                    expected_spawn_ts=prediction.predicted_spawn_ts,
                    actual_spawn_ts=observation_window.actual_spawn_ts,
                    drift_s=drift_s,
                    reason=verify_decision.reason,
                    reaction_ms=reaction_ms,
                    verification_ms=verification_ms,
                    result="failure",
                    final_state=final_state,
                    metadata={
                        "note": observation_window.note,
                        "observation_used": True,
                        **action_metadata,
                        **observation_window.metadata,
                    },
                )
            )

            return CycleRunResult(
                cycle_id=cycle_id,
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                result="failure",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note=observation_window.note,
            )

        try:
            final_state, completed_ts, completed_reason, combat_metadata = self._run_success_path(
                cycle_id=cycle_id,
                prediction=prediction,
                observation_window=observation_window,
                combat_started_ts=verification_result.completed_at_ts,
            )
        except Exception as exc:
            return self._complete_cycle_with_execution_error(
                cycle_id=cycle_id,
                prediction=prediction,
                observation_window=observation_window,
                observation=observation,
                exception=exc,
                completed_from_ts=verification_result.completed_at_ts,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                drift_s=drift_s,
                attempt_action=attempt_decision.action,
                action_metadata=action_metadata,
            )

        self._telemetry_sink.record_cycle(
            TelemetryRecord(
                cycle_id=cycle_id,
                event_ts=completed_ts,
                state=final_state,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                reason=completed_reason,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result="success",
                final_state=final_state,
                metadata={
                    "note": observation_window.note,
                    "observation_used": True,
                    **action_metadata,
                    **combat_metadata,
                    **observation_window.metadata,
                },
            )
        )

        return CycleRunResult(
            cycle_id=cycle_id,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            drift_s=drift_s,
            result="success",
            final_state=final_state,
            reaction_ms=reaction_ms,
            verification_ms=verification_ms,
            observation_used=True,
            note=observation_window.note,
        )

    def _complete_cycle_without_target(
        self,
        *,
        cycle_id: int,
        prediction,
        observation_window: ObservationWindow,
        completed_at_ts: float,
        action_metadata: dict[str, object],
        result: str = "no_target_available",
        reason: str = "no_target_available",
    ) -> CycleRunResult:
        self._tick_and_record_transition(
            now_ts=completed_at_ts,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        final_state = self._fsm.current_state
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation_window.actual_spawn_ts,
        )

        self._telemetry_sink.record_cycle(
            TelemetryRecord(
                cycle_id=cycle_id,
                event_ts=completed_at_ts,
                state=final_state,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                reason=reason,
                reaction_ms=None,
                verification_ms=None,
                result=result,
                final_state=final_state,
                metadata={
                    "note": observation_window.note,
                    "observation_used": True,
                    **action_metadata,
                    **observation_window.metadata,
                },
            )
        )

        return CycleRunResult(
            cycle_id=cycle_id,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            drift_s=drift_s,
            result=result,
            final_state=final_state,
            reaction_ms=None,
            verification_ms=None,
            observation_used=True,
            note=observation_window.note,
        )

    def _handle_verify_timeout_transition(
        self,
        *,
        prediction,
        observation_window: ObservationWindow,
        verification_result: VerificationResult,
    ) -> None:
        self._tick_and_record_transition(
            now_ts=verification_result.completed_at_ts,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )
        self._tick_and_record_transition(
            now_ts=verification_result.completed_at_ts + self._cycle_config.recover_timeout_s,
            prediction=prediction,
            actual_spawn_ts=observation_window.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

    def _run_success_path(
        self,
        *,
        cycle_id: int,
        prediction,
        observation_window: ObservationWindow,
        combat_started_ts: float,
    ) -> tuple[BotState, float, str, dict[str, object]]:
        observation = observation_window.observation
        assert observation is not None

        combat_timeline = self._combat_resolver.resolve_combat(
            cycle_id,
            combat_started_ts=combat_started_ts,
            observation=observation,
        )

        last_reason = "verification_success"
        last_event_ts = combat_started_ts
        last_snapshot: TimedCombatSnapshot | None = None
        combat_metadata: dict[str, object] = {
            "combat_completed": False,
            "combat_turn_count": 0,
            "combat_final_hp_ratio": None,
            "combat_final_condition_ratio": None,
            "combat_finished_with_rest": False,
            "rest_tick_count": 0,
        }

        for timed_snapshot in combat_timeline.snapshots:
            decision = self._tick_and_record_transition(
                now_ts=timed_snapshot.event_ts,
                prediction=prediction,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                observation=None,
                verify_result=None,
                combat_snapshot=timed_snapshot.snapshot,
            )
            last_reason = decision.reason
            last_event_ts = timed_snapshot.event_ts
            last_snapshot = timed_snapshot
            combat_metadata.update(
                {
                    "combat_completed": True,
                    "combat_turn_count": len(combat_timeline.snapshots),
                    "combat_final_hp_ratio": timed_snapshot.snapshot.hp_ratio,
                    "combat_final_condition_ratio": timed_snapshot.snapshot.condition_ratio,
                    "combat_strategy": timed_snapshot.snapshot.strategy,
                    **timed_snapshot.snapshot.metadata,
                }
            )

        if self._fsm.current_state is BotState.WAIT_NEXT_CYCLE:
            return self._fsm.current_state, last_event_ts, last_reason, combat_metadata

        if self._fsm.current_state is not BotState.REST:
            raise RuntimeError(f"Nieoczekiwany stan po walce: {self._fsm.current_state}")

        if last_snapshot is None:
            raise RuntimeError("Brak końcowego snapshotu walki dla ścieżki REST.")

        rest_timeline = self._rest_provider.apply_rest(
            cycle_id,
            rest_started_ts=float(
                combat_metadata.get("reward_completed_ts", last_event_ts)
            ),
            starting_hp_ratio=last_snapshot.snapshot.hp_ratio,
            starting_condition_ratio=last_snapshot.snapshot.condition_ratio,
            observation=observation,
        )

        for timed_snapshot in rest_timeline.snapshots:
            decision = self._tick_and_record_transition(
                now_ts=timed_snapshot.event_ts,
                prediction=prediction,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                observation=None,
                verify_result=None,
                combat_snapshot=timed_snapshot.snapshot,
            )
            last_reason = decision.reason
            last_event_ts = timed_snapshot.event_ts
            combat_metadata["combat_finished_with_rest"] = True
            combat_metadata["rest_tick_count"] = len(rest_timeline.snapshots)
            combat_metadata["rest_final_hp_ratio"] = timed_snapshot.snapshot.hp_ratio
            combat_metadata["rest_final_condition_ratio"] = timed_snapshot.snapshot.condition_ratio

        if rest_timeline.metadata:
            combat_metadata.update(rest_timeline.metadata)

        if self._fsm.current_state is not BotState.WAIT_NEXT_CYCLE:
            raise RuntimeError("REST nie zakończył się przejściem do WAIT_NEXT_CYCLE.")

        return self._fsm.current_state, last_event_ts, last_reason, combat_metadata

    def _complete_cycle_with_execution_error(
        self,
        *,
        cycle_id: int,
        prediction,
        observation_window: ObservationWindow,
        observation: Observation,
        exception: Exception,
        completed_from_ts: float,
        reaction_ms: float,
        verification_ms: float,
        drift_s: float | None,
        attempt_action: str,
        action_metadata: dict[str, object],
    ) -> CycleRunResult:
        recovery_plan = self._recovery.build_exception_recovery_plan(
            now_ts=completed_from_ts,
            current_state=self._fsm.current_state,
            cycle_id=cycle_id,
            exception=exception,
        )

        last_event_ts = completed_from_ts
        for recovery_step in recovery_plan:
            self._fsm.force_state(
                new_state=recovery_step.target_state,
                now_ts=recovery_step.at_ts,
                reason=recovery_step.reason,
                cycle_id=recovery_step.cycle_id,
            )
            self._telemetry_sink.record_state_transition(
                TelemetryRecord(
                    cycle_id=recovery_step.cycle_id,
                    event_ts=recovery_step.at_ts,
                    state=recovery_step.target_state,
                    expected_spawn_ts=prediction.predicted_spawn_ts,
                    actual_spawn_ts=observation_window.actual_spawn_ts,
                    drift_s=drift_s,
                    state_enter=recovery_step.from_state,
                    state_exit=recovery_step.target_state,
                    reason=recovery_step.reason,
                    final_state=recovery_step.target_state,
                    metadata={"action": "recovery"},
                )
            )
            last_event_ts = recovery_step.at_ts

        final_state = self._fsm.current_state

        self._telemetry_sink.record_attempt(
            TelemetryRecord(
                cycle_id=cycle_id,
                event_ts=completed_from_ts,
                state=BotState.ATTEMPT,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                reason="execution_error",
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result="execution_error",
                final_state=final_state,
                metadata={
                    "note": observation_window.note,
                    "attempt_action": attempt_action,
                    **action_metadata,
                },
            )
        )
        self._telemetry_sink.record_cycle(
            TelemetryRecord(
                cycle_id=cycle_id,
                event_ts=last_event_ts,
                state=final_state,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation_window.actual_spawn_ts,
                drift_s=drift_s,
                reason=recovery_plan[-1].reason if recovery_plan else "execution_error_unknown",
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result="execution_error",
                final_state=final_state,
                metadata={
                    "note": observation_window.note,
                    "observation_used": True,
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                },
            )
        )

        return CycleRunResult(
            cycle_id=cycle_id,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation.actual_spawn_ts,
            drift_s=drift_s,
            result="execution_error",
            final_state=final_state,
            reaction_ms=reaction_ms,
            verification_ms=verification_ms,
            observation_used=True,
            note=observation_window.note,
        )

    def _tick_and_record_transition(
        self,
        *,
        now_ts: float,
        prediction,
        actual_spawn_ts: float | None,
        observation,
        verify_result,
        combat_snapshot,
    ) -> Decision:
        previous_transition_count = self._fsm.transition_count()

        decision = self._fsm.tick(
            now_ts=now_ts,
            prediction=prediction,
            temporal_state=self._scheduler.state_for_time(now_ts, prediction),
            observation=observation,
            verify_result=verify_result,
            combat_snapshot=combat_snapshot,
        )

        if self._fsm.transition_count() > previous_transition_count:
            transition = self._fsm.transition_history()[-1]
            self._record_transition(
                transition=transition,
                decision=decision,
                prediction=prediction,
                actual_spawn_ts=actual_spawn_ts,
            )

        return decision

    def _record_transition(
        self,
        *,
        transition: StateTransition,
        decision: Decision,
        prediction,
        actual_spawn_ts: float | None,
    ) -> None:
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=actual_spawn_ts,
        )

        self._telemetry_sink.record_state_transition(
            TelemetryRecord(
                cycle_id=transition.cycle_id,
                event_ts=transition.at_ts,
                state=transition.to_state,
                expected_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=actual_spawn_ts,
                drift_s=drift_s,
                state_enter=transition.from_state,
                state_exit=transition.to_state,
                reason=transition.reason,
                final_state=transition.to_state,
                metadata={"action": decision.action},
            )
        )

    def _compute_drift_s(
        self,
        *,
        predicted_spawn_ts: float,
        actual_spawn_ts: float | None,
    ) -> float | None:
        if actual_spawn_ts is None:
            return None
        return actual_spawn_ts - predicted_spawn_ts
