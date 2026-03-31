from __future__ import annotations

from botlab.application.dto import (
    ActionContext,
    ActionResult,
    CombatOutcome,
    CycleRunResult,
    RestOutcome,
    VerificationOutcome,
)
from botlab.application.ports import (
    ActionExecutor,
    Clock,
    CombatResolver,
    ObservationProvider,
    RestProvider,
    TelemetrySink,
    VerificationProvider,
)
from botlab.config import CycleConfig
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM, StateTransition
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, Decision, Observation, TelemetryRecord


class CycleOrchestrator:
    """
    Neutralny orchestrator cyklu.

    Orkiestruje przebieg cyklu używając:
    - scheduler dla predykcji,
    - decision_engine dla decyzji,
    - FSM dla stanów,
    - recovery dla odzysku,
    - portów application dla zewnętrznych zależności.

    Nie zależy od konkretnych implementacji symulacji.
    """

    def __init__(
        self,
        *,
        scheduler: CycleScheduler,
        decision_engine: DecisionEngine,
        fsm: CycleFSM,
        recovery: RecoveryManager,
        clock: Clock,
        observation_provider: ObservationProvider,
        action_executor: ActionExecutor,
        verification_provider: VerificationProvider,
        combat_resolver: CombatResolver,
        rest_provider: RestProvider,
        telemetry_sink: TelemetrySink,
        cycle_config: CycleConfig,
    ) -> None:
        self._scheduler = scheduler
        self._decision_engine = decision_engine
        self._fsm = fsm
        self._recovery = recovery
        self._clock = clock
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

        # Symulacja prepare window
        self._tick_and_record_transition(
            now_ts=prediction.prepare_window_start_ts,
            prediction=prediction,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        # Symulacja ready window
        self._tick_and_record_transition(
            now_ts=prediction.ready_window_start_ts,
            prediction=prediction,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        observation = self._observation_provider.get_latest_observation(cycle_id)

        if observation is None:
            return self._complete_cycle_without_observation(
                prediction_cycle_id=cycle_id,
                prediction=prediction,
            )

        return self._complete_cycle_with_observation(
            prediction_cycle_id=cycle_id,
            prediction=prediction,
            observation=observation,
        )

    def _complete_cycle_without_observation(
        self,
        *,
        prediction_cycle_id: int,
        prediction,
    ) -> CycleRunResult:
        decision = self._tick_and_record_transition(
            now_ts=prediction.ready_window_end_ts,
            prediction=prediction,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        if prediction.predicted_spawn_ts is None:
            result = "no_event"
            reason = decision.reason
        else:
            result = "late_event_missed"
            reason = "actual_event_outside_ready_window"

        final_state = self._fsm.current_state
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=None,
        )

        self._telemetry_sink.record_cycle({
            "cycle_id": prediction_cycle_id,
            "event_ts": prediction.ready_window_end_ts,
            "state": final_state.value,
            "expected_spawn_ts": prediction.predicted_spawn_ts,
            "actual_spawn_ts": None,
            "drift_s": drift_s,
            "reason": reason,
            "reaction_ms": None,
            "verification_ms": None,
            "result": result,
            "final_state": final_state.value,
            "metadata": {"scenario_note": "", "observation_used": False},
        })

        return CycleRunResult(
            cycle_id=prediction_cycle_id,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=None,
            drift_s=drift_s,
            result=result,
            final_state=final_state,
            reaction_ms=None,
            verification_ms=None,
            observation_used=False,
            note="",
        )

    def _complete_cycle_with_observation(
        self,
        *,
        prediction_cycle_id: int,
        prediction,
        observation: Observation,
    ) -> CycleRunResult:
        self._scheduler.register_observation(observation)

        # Attempt
        action_context = ActionContext(
            cycle_id=prediction_cycle_id,
            now_ts=observation.observed_at_ts,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            observation=observation,
            metadata={},
        )
        action_result = self._action_executor.execute_action(action_context)

        attempt_decision = self._tick_and_record_transition(
            now_ts=observation.observed_at_ts,
            prediction=prediction,
            observation=observation,
            verify_result=None,
            combat_snapshot=None,
        )

        # Verify start
        verify_start_ts = observation.observed_at_ts + 0.020  # Placeholder latency
        self._tick_and_record_transition(
            now_ts=verify_start_ts,
            prediction=prediction,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        # Verify resolution
        verify_result = self._verification_provider.verify(prediction_cycle_id, observation)
        verify_resolution_ts = verify_start_ts + 0.100  # Placeholder latency

        if verify_result == VerificationOutcome.TIMEOUT:
            # Handle timeout with recovery
            recover_ts = verify_resolution_ts + self._cycle_config.recover_timeout_s
            self._tick_and_record_transition(
                now_ts=verify_resolution_ts,
                prediction=prediction,
                observation=None,
                verify_result=verify_result.value,
                combat_snapshot=None,
            )
            self._tick_and_record_transition(
                now_ts=recover_ts,
                prediction=prediction,
                observation=None,
                verify_result=None,
                combat_snapshot=None,
            )

            final_state = self._fsm.current_state
            reaction_ms = (observation.observed_at_ts - prediction.predicted_spawn_ts) * 1000.0
            verification_ms = (verify_resolution_ts - verify_start_ts) * 1000.0
            drift_s = self._compute_drift_s(
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation.observed_at_ts,
            )

            self._telemetry_sink.record_attempt({
                "cycle_id": prediction_cycle_id,
                "event_ts": verify_resolution_ts,
                "state": "ATTEMPT",
                "expected_spawn_ts": prediction.predicted_spawn_ts,
                "actual_spawn_ts": observation.observed_at_ts,
                "drift_s": drift_s,
                "reaction_ms": reaction_ms,
                "verification_ms": verification_ms,
                "result": "verify_timeout",
                "reason": "verify_timeout",
                "metadata": {"attempt_action": attempt_decision.action},
            })
            self._telemetry_sink.record_cycle({
                "cycle_id": prediction_cycle_id,
                "event_ts": recover_ts,
                "state": final_state.value,
                "expected_spawn_ts": prediction.predicted_spawn_ts,
                "actual_spawn_ts": observation.observed_at_ts,
                "drift_s": drift_s,
                "reason": "verify_timeout",
                "reaction_ms": reaction_ms,
                "verification_ms": verification_ms,
                "result": "verify_timeout",
                "final_state": final_state.value,
                "metadata": {"observation_used": True},
            })

            return CycleRunResult(
                cycle_id=prediction_cycle_id,
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation.observed_at_ts,
                drift_s=drift_s,
                result="verify_timeout",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note="",
            )

        verify_decision = self._tick_and_record_transition(
            now_ts=verify_resolution_ts,
            prediction=prediction,
            observation=None,
            verify_result=verify_result.value,
            combat_snapshot=None,
        )

        if verify_result == VerificationOutcome.FAILURE:
            final_state = self._fsm.current_state
            reaction_ms = (observation.observed_at_ts - prediction.predicted_spawn_ts) * 1000.0
            verification_ms = (verify_resolution_ts - verify_start_ts) * 1000.0
            drift_s = self._compute_drift_s(
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation.observed_at_ts,
            )

            self._telemetry_sink.record_attempt({
                "cycle_id": prediction_cycle_id,
                "event_ts": verify_resolution_ts,
                "state": "ATTEMPT",
                "expected_spawn_ts": prediction.predicted_spawn_ts,
                "actual_spawn_ts": observation.observed_at_ts,
                "drift_s": drift_s,
                "reaction_ms": reaction_ms,
                "verification_ms": verification_ms,
                "result": "failure",
                "reason": verify_decision.reason,
                "metadata": {"attempt_action": attempt_decision.action},
            })
            self._telemetry_sink.record_cycle({
                "cycle_id": prediction_cycle_id,
                "event_ts": verify_resolution_ts,
                "state": final_state.value,
                "expected_spawn_ts": prediction.predicted_spawn_ts,
                "actual_spawn_ts": observation.observed_at_ts,
                "drift_s": drift_s,
                "reason": verify_decision.reason,
                "reaction_ms": reaction_ms,
                "verification_ms": verification_ms,
                "result": "failure",
                "final_state": final_state.value,
                "metadata": {"observation_used": True},
            })

            return CycleRunResult(
                cycle_id=prediction_cycle_id,
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation.observed_at_ts,
                drift_s=drift_s,
                result="failure",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note="",
            )

        # Success path
        try:
            final_state, completed_ts, completed_reason = self._run_success_path(
                prediction=prediction,
                combat_started_ts=verify_resolution_ts,
                cycle_id=prediction_cycle_id,
                observation=observation,
            )
        except Exception as e:
            # Handle exception with recovery
            recovery_plan = self._recovery.build_exception_recovery_plan(
                now_ts=verify_resolution_ts,
                current_state=self._fsm.current_state,
                cycle_id=prediction_cycle_id,
                exception=e,
            )
            if recovery_plan is None:
                raise

            last_event_ts = verify_resolution_ts
            for recovery_step in recovery_plan:
                self._fsm.force_state(
                    new_state=recovery_step.target_state,
                    now_ts=recovery_step.at_ts,
                    reason=recovery_step.reason,
                    cycle_id=recovery_step.cycle_id,
                )
                self._telemetry_sink.record_state_transition({
                    "cycle_id": recovery_step.cycle_id,
                    "event_ts": recovery_step.at_ts,
                    "state_enter": recovery_step.from_state.value if recovery_step.from_state else None,
                    "state_exit": recovery_step.target_state.value,
                    "reason": recovery_step.reason,
                    "final_state": recovery_step.target_state.value,
                    "metadata": {"action": "recovery"},
                })
                last_event_ts = recovery_step.at_ts

            final_state = self._fsm.current_state
            completed_ts = last_event_ts
            completed_reason = recovery_plan[-1].reason if recovery_plan else "execution_error_unknown"

            reaction_ms = (observation.observed_at_ts - prediction.predicted_spawn_ts) * 1000.0
            verification_ms = (verify_resolution_ts - verify_start_ts) * 1000.0
            drift_s = self._compute_drift_s(
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation.observed_at_ts,
            )

            self._telemetry_sink.record_attempt({
                "cycle_id": prediction_cycle_id,
                "event_ts": verify_resolution_ts,
                "state": "ATTEMPT",
                "expected_spawn_ts": prediction.predicted_spawn_ts,
                "actual_spawn_ts": observation.observed_at_ts,
                "drift_s": drift_s,
                "reaction_ms": reaction_ms,
                "verification_ms": verification_ms,
                "result": "execution_error",
                "reason": "execution_error",
                "metadata": {"attempt_action": attempt_decision.action},
            })
            self._telemetry_sink.record_cycle({
                "cycle_id": prediction_cycle_id,
                "event_ts": completed_ts,
                "state": final_state.value,
                "expected_spawn_ts": prediction.predicted_spawn_ts,
                "actual_spawn_ts": observation.observed_at_ts,
                "drift_s": drift_s,
                "reason": completed_reason,
                "reaction_ms": reaction_ms,
                "verification_ms": verification_ms,
                "result": "execution_error",
                "final_state": final_state.value,
                "metadata": {"observation_used": True, "exception_type": type(e).__name__, "exception_message": str(e)},
            })

            return CycleRunResult(
                cycle_id=prediction_cycle_id,
                predicted_spawn_ts=prediction.predicted_spawn_ts,
                actual_spawn_ts=observation.observed_at_ts,
                drift_s=drift_s,
                result="execution_error",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note="",
            )

        reaction_ms = (observation.observed_at_ts - prediction.predicted_spawn_ts) * 1000.0
        verification_ms = (verify_resolution_ts - verify_start_ts) * 1000.0
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation.observed_at_ts,
        )

        self._telemetry_sink.record_attempt({
            "cycle_id": prediction_cycle_id,
            "event_ts": verify_resolution_ts,
            "state": "ATTEMPT",
            "expected_spawn_ts": prediction.predicted_spawn_ts,
            "actual_spawn_ts": observation.observed_at_ts,
            "drift_s": drift_s,
            "reaction_ms": reaction_ms,
            "verification_ms": verification_ms,
            "result": "success",
            "reason": verify_decision.reason,
            "metadata": {"attempt_action": attempt_decision.action},
        })
        self._telemetry_sink.record_cycle({
            "cycle_id": prediction_cycle_id,
            "event_ts": completed_ts,
            "state": final_state.value,
            "expected_spawn_ts": prediction.predicted_spawn_ts,
            "actual_spawn_ts": observation.observed_at_ts,
            "drift_s": drift_s,
            "reason": completed_reason,
            "reaction_ms": reaction_ms,
            "verification_ms": verification_ms,
            "result": "success",
            "final_state": final_state.value,
            "metadata": {"observation_used": True},
        })

        return CycleRunResult(
            cycle_id=prediction_cycle_id,
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=observation.observed_at_ts,
            drift_s=drift_s,
            result="success",
            final_state=final_state,
            reaction_ms=reaction_ms,
            verification_ms=verification_ms,
            observation_used=True,
            note="",
        )

    def _run_success_path(
        self,
        *,
        prediction,
        combat_started_ts: float,
        cycle_id: int,
        observation: Observation,
    ) -> tuple[BotState, float, str]:
        # Placeholder for combat and rest
        combat_outcome = self._combat_resolver.resolve_combat(cycle_id, observation)
        rest_outcome = self._rest_provider.apply_rest(cycle_id, observation)

        # Simplified: assume combat and rest complete immediately
        completed_ts = combat_started_ts + 1.0  # Placeholder
        completed_reason = "combat_and_rest_completed"
        final_state = BotState.WAIT_NEXT_CYCLE

        return final_state, completed_ts, completed_reason

    def _tick_and_record_transition(
        self,
        *,
        now_ts: float,
        prediction,
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
            )

        return decision

    def _record_transition(
        self,
        *,
        transition: StateTransition,
        decision: Decision,
        prediction,
    ) -> None:
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=None,  # Simplified
        )

        self._telemetry_sink.record_state_transition({
            "cycle_id": transition.cycle_id,
            "event_ts": transition.at_ts,
            "state_enter": transition.from_state.value,
            "state_exit": transition.to_state.value,
            "reason": transition.reason,
            "final_state": transition.to_state.value,
            "metadata": {"action": decision.action},
        })

    def _compute_drift_s(
        self,
        *,
        predicted_spawn_ts: float,
        actual_spawn_ts: float | None,
    ) -> float | None:
        if actual_spawn_ts is None:
            return None
        return actual_spawn_ts - predicted_spawn_ts