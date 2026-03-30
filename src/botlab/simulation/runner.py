from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from botlab.config import Settings
from botlab.core.decision_engine import DecisionEngine
from botlab.core.fsm import CycleFSM, StateTransition
from botlab.core.recovery import RecoveryManager
from botlab.core.scheduler import CycleScheduler
from botlab.simulation.battle import SimulatedBattle, SimulatedRest, TimedCombatSnapshot
from botlab.simulation.spawner import SimulatedSpawner, SpawnEvent
from botlab.simulation.world import CycleTrace, SimulatedWorld
from botlab.telemetry.logger import configure_telemetry_logger, log_telemetry_record
from botlab.telemetry.storage import SQLiteTelemetryStorage
from botlab.types import BotState, Decision, TelemetryRecord


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
class SimulationReport:
    cycle_results: list[CycleRunResult]
    log_path: Path
    sqlite_path: Path

    @property
    def total_cycles(self) -> int:
        return len(self.cycle_results)

    def count_result(self, result: str) -> int:
        return sum(1 for item in self.cycle_results if item.result == result)


class SimulationRunner:
    """
    Minimalny runner spinający:
    - scheduler,
    - predictor,
    - FSM,
    - spawner,
    - world,
    - battle,
    - rest,
    - telemetry logger,
    - SQLite storage.

    Zasada działania:
    - każdy cykl jest wykonywany krokami czasowymi,
    - każde realne przejście stanu jest logowane i zapisywane do SQLite,
    - po sukcesie VERIFY wykonywana jest prosta walka,
    - po walce system przechodzi do REST albo WAIT_NEXT_CYCLE.
    """

    def __init__(
        self,
        *,
        scheduler: CycleScheduler,
        fsm: CycleFSM,
        spawner: SimulatedSpawner,
        world: SimulatedWorld,
        battle: SimulatedBattle,
        rest: SimulatedRest,
        storage: SQLiteTelemetryStorage,
        logger: logging.Logger,
        cycle_config,
    ) -> None:
        self._scheduler = scheduler
        self._fsm = fsm
        self._spawner = spawner
        self._world = world
        self._battle = battle
        self._rest = rest
        self._storage = storage
        self._logger = logger
        self._recovery = RecoveryManager(cycle_config)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        spawner: SimulatedSpawner | None = None,
        initial_anchor_spawn_ts: float = 100.0,
        initial_anchor_cycle_id: int = 0,
        logger_name: str = "botlab.simulation",
        enable_console: bool = False,
    ) -> "SimulationRunner":
        scheduler = CycleScheduler.from_cycle_config(settings.cycle)
        scheduler.bootstrap(
            anchor_spawn_ts=initial_anchor_spawn_ts,
            anchor_cycle_id=initial_anchor_cycle_id,
        )

        decision_engine = DecisionEngine(settings.cycle, settings.combat)
        fsm = CycleFSM(
            decision_engine=decision_engine,
            initial_state=BotState.IDLE,
            started_at_ts=0.0,
            cycle_id=None,
        )

        world = SimulatedWorld(settings.cycle)
        battle = SimulatedBattle()
        rest = SimulatedRest(settings.combat)

        storage = SQLiteTelemetryStorage.from_config(settings.telemetry)
        logger = configure_telemetry_logger(
            telemetry_config=settings.telemetry,
            logger_name=logger_name,
            enable_console=enable_console,
        )

        return cls(
            scheduler=scheduler,
            fsm=fsm,
            spawner=spawner or SimulatedSpawner(),
            world=world,
            battle=battle,
            rest=rest,
            storage=storage,
            logger=logger,
            cycle_config=settings.cycle,
        )

    @property
    def storage(self) -> SQLiteTelemetryStorage:
        return self._storage

    def run_cycles(self, total_cycles: int) -> SimulationReport:
        if total_cycles <= 0:
            raise ValueError("total_cycles musi być większe od 0.")

        self._storage.initialize()

        start_cycle_id = self._scheduler.predictor.anchor_cycle_id + 1
        cycle_results: list[CycleRunResult] = []

        for cycle_id in range(start_cycle_id, start_cycle_id + total_cycles):
            cycle_results.append(self._run_single_cycle(cycle_id))

        return SimulationReport(
            cycle_results=cycle_results,
            log_path=self._resolve_log_path(),
            sqlite_path=self._storage.sqlite_path,
        )

    def _resolve_log_path(self) -> Path:
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename).resolve()

        raise RuntimeError("Logger nie ma skonfigurowanego FileHandler.")

    def _run_single_cycle(self, cycle_id: int) -> CycleRunResult:
        prediction = self._scheduler.prediction_for_cycle(cycle_id)
        spawn_event = self._spawner.build_spawn_event(prediction)
        trace = self._world.build_cycle_trace(spawn_event)

        self._tick_and_record_transition(
            now_ts=trace.prepare_ts,
            prediction=prediction,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        self._tick_and_record_transition(
            now_ts=trace.ready_ts,
            prediction=prediction,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        if spawn_event.observation is None:
            return self._complete_cycle_without_observation(
                prediction_cycle_id=cycle_id,
                spawn_event=spawn_event,
                trace=trace,
            )

        return self._complete_cycle_with_observation(
            prediction_cycle_id=cycle_id,
            spawn_event=spawn_event,
            trace=trace,
        )

    def _complete_cycle_without_observation(
        self,
        *,
        prediction_cycle_id: int,
        spawn_event: SpawnEvent,
        trace: CycleTrace,
    ) -> CycleRunResult:
        decision = self._tick_and_record_transition(
            now_ts=trace.ready_window_close_ts,
            prediction=spawn_event.prediction,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        if spawn_event.actual_spawn_ts is None:
            result = "no_event"
            reason = decision.reason
        else:
            result = "late_event_missed"
            reason = "actual_event_outside_ready_window"

        final_state = self._fsm.current_state
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
        )

        cycle_record = TelemetryRecord(
            cycle_id=prediction_cycle_id,
            event_ts=trace.ready_window_close_ts,
            state=final_state,
            expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            drift_s=drift_s,
            state_enter=None,
            state_exit=None,
            reason=reason,
            reaction_ms=None,
            verification_ms=None,
            result=result,
            final_state=final_state,
            metadata={
                "scenario_note": spawn_event.scenario.note,
                "observation_used": False,
            },
        )
        self._storage.record_cycle(cycle_record)
        log_telemetry_record(self._logger, cycle_record)

        return CycleRunResult(
            cycle_id=prediction_cycle_id,
            predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            drift_s=drift_s,
            result=result,
            final_state=final_state,
            reaction_ms=None,
            verification_ms=None,
            observation_used=False,
            note=spawn_event.scenario.note,
        )

    def _complete_cycle_with_observation(
        self,
        *,
        prediction_cycle_id: int,
        spawn_event: SpawnEvent,
        trace: CycleTrace,
    ) -> CycleRunResult:
        assert spawn_event.observation is not None
        assert trace.attempt_ts is not None
        assert trace.verify_start_ts is not None

        self._scheduler.register_observation(spawn_event.observation)

        attempt_decision = self._tick_and_record_transition(
            now_ts=trace.attempt_ts,
            prediction=spawn_event.prediction,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            observation=spawn_event.observation,
            verify_result=None,
            combat_snapshot=None,
        )

        self._tick_and_record_transition(
            now_ts=trace.verify_start_ts,
            prediction=spawn_event.prediction,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            observation=None,
            verify_result=None,
            combat_snapshot=None,
        )

        reaction_ms = (trace.attempt_ts - spawn_event.observation.observed_at_ts) * 1000.0
        drift_s = self._compute_drift_s(
            predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
        )

        if spawn_event.verify_result == "timeout":
            assert trace.verify_timeout_ts is not None
            assert trace.recover_complete_ts is not None

            self._tick_and_record_transition(
                now_ts=trace.verify_timeout_ts,
                prediction=spawn_event.prediction,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                observation=None,
                verify_result=None,
                combat_snapshot=None,
            )

            verification_ms = (trace.verify_timeout_ts - trace.verify_start_ts) * 1000.0

            attempt_record = TelemetryRecord(
                cycle_id=prediction_cycle_id,
                event_ts=trace.verify_timeout_ts,
                state=BotState.ATTEMPT,
                expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                state_enter=BotState.READY_WINDOW,
                state_exit=BotState.VERIFY,
                reason="verify_timeout",
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result="verify_timeout",
                final_state=BotState.RECOVER,
                metadata={
                    "scenario_note": spawn_event.scenario.note,
                    "attempt_action": attempt_decision.action,
                },
            )
            self._storage.record_attempt(attempt_record)
            log_telemetry_record(self._logger, attempt_record)

            self._tick_and_record_transition(
                now_ts=trace.recover_complete_ts,
                prediction=spawn_event.prediction,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                observation=None,
                verify_result=None,
                combat_snapshot=None,
            )

            final_state = self._fsm.current_state
            cycle_record = TelemetryRecord(
                cycle_id=prediction_cycle_id,
                event_ts=trace.recover_complete_ts,
                state=final_state,
                expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                state_enter=None,
                state_exit=None,
                reason="verify_timeout",
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result="verify_timeout",
                final_state=final_state,
                metadata={
                    "scenario_note": spawn_event.scenario.note,
                    "observation_used": True,
                },
            )
            self._storage.record_cycle(cycle_record)
            log_telemetry_record(self._logger, cycle_record)

            return CycleRunResult(
                cycle_id=prediction_cycle_id,
                predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                result="verify_timeout",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note=spawn_event.scenario.note,
            )

        assert trace.verify_resolution_ts is not None

        verify_decision = self._tick_and_record_transition(
            now_ts=trace.verify_resolution_ts,
            prediction=spawn_event.prediction,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            observation=None,
            verify_result=spawn_event.verify_result,
            combat_snapshot=None,
        )

        verification_ms = (trace.verify_resolution_ts - trace.verify_start_ts) * 1000.0
        result = "success" if spawn_event.verify_result == "success" else "failure"

        attempt_record = TelemetryRecord(
            cycle_id=prediction_cycle_id,
            event_ts=trace.verify_resolution_ts,
            state=BotState.ATTEMPT,
            expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            drift_s=drift_s,
            state_enter=BotState.READY_WINDOW,
            state_exit=BotState.VERIFY,
            reason=verify_decision.reason,
            reaction_ms=reaction_ms,
            verification_ms=verification_ms,
            result=result,
            final_state=self._fsm.current_state,
            metadata={
                "scenario_note": spawn_event.scenario.note,
                "attempt_action": attempt_decision.action,
            },
        )
        self._storage.record_attempt(attempt_record)
        log_telemetry_record(self._logger, attempt_record)

        if result == "failure":
            final_state = self._fsm.current_state
            cycle_record = TelemetryRecord(
                cycle_id=prediction_cycle_id,
                event_ts=trace.verify_resolution_ts,
                state=final_state,
                expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                state_enter=None,
                state_exit=None,
                reason=verify_decision.reason,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result=result,
                final_state=final_state,
                metadata={
                    "scenario_note": spawn_event.scenario.note,
                    "observation_used": True,
                },
            )
            self._storage.record_cycle(cycle_record)
            log_telemetry_record(self._logger, cycle_record)

            return CycleRunResult(
                cycle_id=prediction_cycle_id,
                predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                result=result,
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note=spawn_event.scenario.note,
            )

        try:
            final_state, completed_ts, completed_reason = self._run_success_path(
                spawn_event=spawn_event,
                combat_started_ts=trace.verify_resolution_ts,
            )
        except Exception as e:
            exception_type = type(e).__name__
            recovery_plan = self._recovery.build_exception_recovery_plan(
                now_ts=trace.verify_resolution_ts,
                current_state=self._fsm.current_state,
                cycle_id=prediction_cycle_id,
                exception=e,
            )

            if recovery_plan is None:
                raise

            last_event_ts = trace.verify_resolution_ts
            
            for recovery_step in recovery_plan:
                from_state = recovery_step.from_state
                self._fsm.force_state(
                    new_state=recovery_step.target_state,
                    now_ts=recovery_step.at_ts,
                    reason=recovery_step.reason,
                    cycle_id=recovery_step.cycle_id,
                )
                
                record = TelemetryRecord(
                    cycle_id=recovery_step.cycle_id,
                    event_ts=recovery_step.at_ts,
                    state=recovery_step.target_state,
                    expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                    actual_spawn_ts=spawn_event.actual_spawn_ts,
                    drift_s=drift_s,
                    state_enter=from_state,
                    state_exit=recovery_step.target_state,
                    reason=recovery_step.reason,
                    reaction_ms=None,
                    verification_ms=None,
                    result=None,
                    final_state=recovery_step.target_state,
                    metadata={
                        "action": "recovery",
                    },
                )
                self._storage.record_state_transition(record)
                log_telemetry_record(self._logger, record)
                
                last_event_ts = recovery_step.at_ts

            final_state = self._fsm.current_state
            completed_ts = last_event_ts
            completed_reason = recovery_plan[-1].reason if recovery_plan else "execution_error_unknown"

            cycle_record = TelemetryRecord(
                cycle_id=prediction_cycle_id,
                event_ts=completed_ts,
                state=final_state,
                expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                state_enter=None,
                state_exit=None,
                reason=completed_reason,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                result="execution_error",
                final_state=final_state,
                metadata={
                    "scenario_note": spawn_event.scenario.note,
                    "observation_used": True,
                    "exception_type": exception_type,
                    "exception_message": str(e),
                },
            )
            self._storage.record_cycle(cycle_record)
            log_telemetry_record(self._logger, cycle_record)

            return CycleRunResult(
                cycle_id=prediction_cycle_id,
                predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                drift_s=drift_s,
                result="execution_error",
                final_state=final_state,
                reaction_ms=reaction_ms,
                verification_ms=verification_ms,
                observation_used=True,
                note=spawn_event.scenario.note,
            )

        cycle_record = TelemetryRecord(
            cycle_id=prediction_cycle_id,
            event_ts=completed_ts,
            state=final_state,
            expected_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            drift_s=drift_s,
            state_enter=None,
            state_exit=None,
            reason=completed_reason,
            reaction_ms=reaction_ms,
            verification_ms=verification_ms,
            result="success",
            final_state=final_state,
            metadata={
                "scenario_note": spawn_event.scenario.note,
                "observation_used": True,
            },
        )
        self._storage.record_cycle(cycle_record)
        log_telemetry_record(self._logger, cycle_record)

        return CycleRunResult(
            cycle_id=prediction_cycle_id,
            predicted_spawn_ts=spawn_event.prediction.predicted_spawn_ts,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            drift_s=drift_s,
            result="success",
            final_state=final_state,
            reaction_ms=reaction_ms,
            verification_ms=verification_ms,
            observation_used=True,
            note=spawn_event.scenario.note,
        )

    def _run_success_path(
        self,
        *,
        spawn_event: SpawnEvent,
        combat_started_ts: float,
    ) -> tuple[BotState, float, str]:
        battle_timeline = self._battle.build_timeline(
            cycle_id=spawn_event.cycle_id,
            combat_started_ts=combat_started_ts,
            scenario=spawn_event.scenario,
        )

        last_reason = "verification_success"
        last_event_ts = combat_started_ts
        last_snapshot: TimedCombatSnapshot | None = None

        for timed_snapshot in battle_timeline:
            decision = self._tick_and_record_transition(
                now_ts=timed_snapshot.event_ts,
                prediction=spawn_event.prediction,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                observation=None,
                verify_result=None,
                combat_snapshot=timed_snapshot.snapshot,
            )
            last_reason = decision.reason
            last_event_ts = timed_snapshot.event_ts
            last_snapshot = timed_snapshot

        if self._fsm.current_state is BotState.WAIT_NEXT_CYCLE:
            return self._fsm.current_state, last_event_ts, last_reason

        if self._fsm.current_state is not BotState.REST:
            raise RuntimeError(
                f"Nieoczekiwany stan po zakończeniu walki: {self._fsm.current_state}"
            )

        if last_snapshot is None:
            raise RuntimeError("Brak snapshotu kończącego walkę dla ścieżki REST.")

        rest_timeline = self._rest.build_timeline(
            cycle_id=spawn_event.cycle_id,
            rest_started_ts=last_event_ts,
            starting_hp_ratio=last_snapshot.snapshot.hp_ratio,
            scenario=spawn_event.scenario,
        )

        for timed_snapshot in rest_timeline:
            decision = self._tick_and_record_transition(
                now_ts=timed_snapshot.event_ts,
                prediction=spawn_event.prediction,
                actual_spawn_ts=spawn_event.actual_spawn_ts,
                observation=None,
                verify_result=None,
                combat_snapshot=timed_snapshot.snapshot,
            )
            last_reason = decision.reason
            last_event_ts = timed_snapshot.event_ts

        if self._fsm.current_state is not BotState.WAIT_NEXT_CYCLE:
            raise RuntimeError("REST nie zakończył się przejściem do WAIT_NEXT_CYCLE.")

        return self._fsm.current_state, last_event_ts, last_reason

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

        record = TelemetryRecord(
            cycle_id=transition.cycle_id,
            event_ts=transition.at_ts,
            state=transition.to_state,
            expected_spawn_ts=prediction.predicted_spawn_ts,
            actual_spawn_ts=actual_spawn_ts,
            drift_s=drift_s,
            state_enter=transition.from_state,
            state_exit=transition.to_state,
            reason=transition.reason,
            reaction_ms=None,
            verification_ms=None,
            result=None,
            final_state=transition.to_state,
            metadata={
                "action": decision.action,
            },
        )
        self._storage.record_state_transition(record)
        log_telemetry_record(self._logger, record)

    def _compute_drift_s(
        self,
        *,
        predicted_spawn_ts: float,
        actual_spawn_ts: float | None,
    ) -> float | None:
        if actual_spawn_ts is None:
            return None
        return actual_spawn_ts - predicted_spawn_ts
