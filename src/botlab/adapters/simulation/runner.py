from __future__ import annotations

import logging
from pathlib import Path

from botlab.application import (
    ActionContext,
    ActionResult,
    CombatOutcome,
    CycleOrchestrator,
    CycleRunResult,
    RestOutcome,
    SimulationReport,
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
from botlab.config import Settings
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.adapters.simulation.battle import SimulatedBattle, SimulatedRest
from botlab.adapters.simulation.spawner import SimulatedSpawner
from botlab.adapters.telemetry.logger import configure_telemetry_logger
from botlab.adapters.telemetry.storage import SQLiteTelemetryStorage
from botlab.types import BotState, Observation


class SimulationRunner:
    """
    Composition root dla symulacji.

    Tworzy adaptery portów application layer i używa CycleOrchestrator
    do orkiestracji cyklu, zachowując semantykę symulacji.
    """

    def __init__(
        self,
        *,
        orchestrator: CycleOrchestrator,
        storage: SQLiteTelemetryStorage,
        logger: logging.Logger,
    ) -> None:
        self._orchestrator = orchestrator
        self._storage = storage
        self._logger = logger

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

        recovery = RecoveryManager(settings.cycle)

        battle = SimulatedBattle()
        rest = SimulatedRest(settings.combat)

        storage = SQLiteTelemetryStorage.from_config(settings.telemetry)
        logger = configure_telemetry_logger(
            telemetry_config=settings.telemetry,
            logger_name=logger_name,
            enable_console=enable_console,
        )

        # Create port adapters
        clock = SimulatedClock()
        observation_provider = SimulatedObservationProvider(
            spawner=spawner or SimulatedSpawner(),
            scheduler=scheduler,
        )
        action_executor = SimulatedActionExecutor()
        verification_provider = SimulatedVerificationProvider(
            spawner=spawner or SimulatedSpawner(),
            scheduler=scheduler,
        )
        combat_resolver = SimulatedCombatResolver(battle)
        rest_provider = SimulatedRestProvider(rest)
        telemetry_sink = SimulatedTelemetrySink(storage, logger)

        # Create orchestrator
        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            decision_engine=decision_engine,
            fsm=fsm,
            recovery=recovery,
            clock=clock,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=settings.cycle,
        )

        return cls(
            orchestrator=orchestrator,
            storage=storage,
            logger=logger,
        )

    @property
    def storage(self) -> SQLiteTelemetryStorage:
        return self._storage

    def run_cycles(self, total_cycles: int) -> SimulationReport:
        if total_cycles <= 0:
            raise ValueError("total_cycles musi być większe od 0.")

        self._storage.initialize()

        cycle_results = self._orchestrator.run_cycles(total_cycles)

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


# Port Adapters

class SimulatedClock(Clock):
    """Adapter dla zegara symulacji."""

    def now(self) -> float:
        # W symulacji czas jest kontrolowany przez orchestratora
        raise NotImplementedError("SimulatedClock.now() nie powinno być wywoływane bezpośrednio")


class SimulatedObservationProvider(ObservationProvider):
    """Adapter dostarczający obserwacje z symulacji."""

    def __init__(self, spawner: SimulatedSpawner, scheduler: CycleScheduler):
        self._spawner = spawner
        self._scheduler = scheduler

    def get_latest_observation(self, cycle_id: int) -> Observation | None:
        prediction = self._scheduler.prediction_for_cycle(cycle_id)
        spawn_event = self._spawner.build_spawn_event(prediction)
        return spawn_event.observation


class SimulatedActionExecutor(ActionExecutor):
    """Adapter wykonujący akcje w symulacji."""

    def execute_action(self, context: ActionContext) -> ActionResult:
        # W symulacji akcja jest zawsze udana
        return ActionResult(
            cycle_id=context.cycle_id,
            success=True,
            reason="action_executed",
            metadata={},
        )


class SimulatedVerificationProvider(VerificationProvider):
    """Adapter weryfikujący w symulacji."""

    def __init__(self, spawner: SimulatedSpawner, scheduler: CycleScheduler):
        self._spawner = spawner
        self._scheduler = scheduler

    def verify(self, cycle_id: int, observation) -> VerificationOutcome:
        prediction = self._scheduler.prediction_for_cycle(cycle_id)
        spawn_event = self._spawner.build_spawn_event(prediction)
        if spawn_event.verify_result == "success":
            return VerificationOutcome.SUCCESS
        elif spawn_event.verify_result == "failure":
            return VerificationOutcome.FAILURE
        elif spawn_event.verify_result == "timeout":
            return VerificationOutcome.TIMEOUT
        else:
            raise ValueError(f"Nieznany verify_result: {spawn_event.verify_result}")


class SimulatedCombatResolver(CombatResolver):
    """Adapter rozwiązujący walkę w symulacji."""

    def __init__(self, battle: SimulatedBattle):
        self._battle = battle

    def resolve_combat(self, cycle_id: int, state_snapshot: Observation) -> CombatOutcome:
        # Symulacja walki - w symulacji zawsze udana
        return CombatOutcome(
            cycle_id=cycle_id,
            won=True,
            hp_ratio=0.7,
            metadata={"battle_duration": 5.0},
        )


class SimulatedRestProvider(RestProvider):
    """Adapter dostarczający odpoczynek w symulacji."""

    def __init__(self, rest: SimulatedRest):
        self._rest = rest

    def apply_rest(self, cycle_id: int, state_snapshot: Observation) -> RestOutcome:
        # Symulacja odpoczynku - regeneracja zawsze powodzeniem
        return RestOutcome(
            cycle_id=cycle_id,
            hp_ratio=0.95,
            recovered=True,
            metadata={"rest_duration": 10.0},
        )


class SimulatedTelemetrySink(TelemetrySink):
    """Adapter zapisujący telemetrię."""

    def __init__(self, storage: SQLiteTelemetryStorage, logger: logging.Logger):
        self._storage = storage
        self._logger = logger

    def record_cycle(self, record: dict) -> None:
        from botlab.adapters.telemetry.logger import log_telemetry_record
        from botlab.types import TelemetryRecord

        telemetry_record = TelemetryRecord(**record)
        self._storage.record_cycle(telemetry_record)
        log_telemetry_record(self._logger, telemetry_record)

    def record_attempt(self, record: dict) -> None:
        from botlab.adapters.telemetry.logger import log_telemetry_record
        from botlab.types import TelemetryRecord

        telemetry_record = TelemetryRecord(**record)
        self._storage.record_attempt(telemetry_record)
        log_telemetry_record(self._logger, telemetry_record)

    def record_state_transition(self, record: dict) -> None:
        from botlab.adapters.telemetry.logger import log_telemetry_record
        from botlab.types import TelemetryRecord

        telemetry_record = TelemetryRecord(**record)
        self._storage.record_state_transition(telemetry_record)
        log_telemetry_record(self._logger, telemetry_record)
