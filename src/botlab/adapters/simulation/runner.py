from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from botlab.adapters.simulation.battle import SimulatedBattle, SimulatedRest
from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog
from botlab.adapters.simulation.combat_profiles import SimulatedCombatProfileCatalog
from botlab.adapters.simulation.spawner import SimulatedSpawner
from botlab.adapters.simulation.targeting import (
    SimulatedApproachWorldStateProvider,
    SimulatedInteractionWorldStateProvider,
    SimulatedObservationPreparationProvider,
    SimulatedTargetApproachProvider,
    SimulatedTargetInteractionProvider,
    SimulatedWorldStateProvider,
)
from botlab.adapters.simulation.world import CycleTrace, SimulatedWorld
from botlab.adapters.telemetry.logger import configure_telemetry_logger, log_telemetry_record
from botlab.adapters.telemetry.storage import SQLiteTelemetryStorage
from botlab.application import (
    ActionContext,
    ActionResult,
    CombatTimeline,
    CycleOrchestrator,
    ObservationPreparationService,
    ObservationWindow,
    RestTimeline,
    SimulationReport,
    TargetAcquisitionService,
    TargetApproachService,
    TargetEngagementService,
    TargetResolution,
    TargetInteractionService,
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
from botlab.config import Settings
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.domain.targeting import RetargetPolicy, TargetSelectionPolicy, TargetValidationPolicy
from botlab.types import BotState, Observation, TelemetryRecord

if TYPE_CHECKING:
    from botlab.application.dto import (
        TargetApproachResult,
        TargetInteractionResult,
        TargetResolution,
    )


@dataclass(slots=True, frozen=True)
class SimulatedCycleContext:
    cycle_id: int
    prediction: object
    spawn_event: object
    trace: CycleTrace


class SimulatedCycleRuntime:
    """Współdzielona materializacja cyklu dla wszystkich adapterów portów."""

    def __init__(
        self,
        *,
        scheduler: CycleScheduler,
        spawner: SimulatedSpawner,
        world: SimulatedWorld,
    ) -> None:
        self._scheduler = scheduler
        self._spawner = spawner
        self._world = world
        self._contexts: dict[int, SimulatedCycleContext] = {}
        self._target_resolutions: dict[int, TargetResolution] = {}
        self._approach_results: dict[int, TargetApproachResult] = {}
        self._interaction_results: dict[int, TargetInteractionResult] = {}

    @property
    def initial_cycle_id(self) -> int:
        return self._scheduler.predictor.anchor_cycle_id + 1

    def context_for_cycle(self, cycle_id: int) -> SimulatedCycleContext:
        context = self._contexts.get(cycle_id)
        if context is not None:
            return context

        prediction = self._scheduler.prediction_for_cycle(cycle_id)
        spawn_event = self._spawner.build_spawn_event(prediction)
        trace = self._world.build_cycle_trace(spawn_event)
        context = SimulatedCycleContext(
            cycle_id=cycle_id,
            prediction=prediction,
            spawn_event=spawn_event,
            trace=trace,
        )
        self._contexts[cycle_id] = context
        return context

    def set_target_resolution(self, cycle_id: int, resolution: "TargetResolution") -> None:
        self._target_resolutions[cycle_id] = resolution

    def target_resolution_for_cycle(self, cycle_id: int) -> "TargetResolution | None":
        return self._target_resolutions.get(cycle_id)

    def set_approach_result(self, cycle_id: int, result: "TargetApproachResult") -> None:
        self._approach_results[cycle_id] = result

    def approach_result_for_cycle(self, cycle_id: int) -> "TargetApproachResult | None":
        return self._approach_results.get(cycle_id)

    def set_interaction_result(self, cycle_id: int, result: "TargetInteractionResult") -> None:
        self._interaction_results[cycle_id] = result

    def interaction_result_for_cycle(self, cycle_id: int) -> "TargetInteractionResult | None":
        return self._interaction_results.get(cycle_id)


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
        observation_preparation_service: ObservationPreparationService,
        target_engagement_service: TargetEngagementService,
        runtime: SimulatedCycleRuntime,
        storage: SQLiteTelemetryStorage,
        logger: logging.Logger,
    ) -> None:
        self._orchestrator = orchestrator
        self._observation_preparation_service = observation_preparation_service
        self._target_engagement_service = target_engagement_service
        self._runtime = runtime
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

        shared_spawner = spawner or SimulatedSpawner()
        world = SimulatedWorld(settings.cycle)
        combat_plan_catalog = SimulatedCombatPlanCatalog()
        combat_profile_catalog = SimulatedCombatProfileCatalog(
            combat_plan_catalog=combat_plan_catalog
        )
        battle = SimulatedBattle(
            combat_plan_catalog=combat_plan_catalog,
            combat_profile_catalog=combat_profile_catalog,
        )
        rest = SimulatedRest(settings.combat)
        runtime = SimulatedCycleRuntime(
            scheduler=scheduler,
            spawner=shared_spawner,
            world=world,
        )

        storage = SQLiteTelemetryStorage.from_config(settings.telemetry)
        logger = configure_telemetry_logger(
            telemetry_config=settings.telemetry,
            logger_name=logger_name,
            enable_console=enable_console,
        )

        observation_provider = SimulatedObservationProvider(runtime)
        action_executor = SimulatedActionExecutor(
            runtime,
            combat_plan_catalog=combat_plan_catalog,
            combat_profile_catalog=combat_profile_catalog,
        )
        verification_provider = SimulatedVerificationProvider(runtime)
        combat_resolver = SimulatedCombatResolver(runtime, battle)
        rest_provider = SimulatedRestProvider(runtime, rest)
        telemetry_sink = SimulatedTelemetrySink(storage, logger)

        orchestrator = CycleOrchestrator(
            scheduler=scheduler,
            fsm=fsm,
            recovery=recovery,
            observation_provider=observation_provider,
            action_executor=action_executor,
            verification_provider=verification_provider,
            combat_resolver=combat_resolver,
            rest_provider=rest_provider,
            telemetry_sink=telemetry_sink,
            cycle_config=settings.cycle,
        )

        observation_preparation_service = ObservationPreparationService(
            observation_preparation_provider=SimulatedObservationPreparationProvider(runtime),
        )
        retarget_policy = RetargetPolicy(
            selection_policy=TargetSelectionPolicy(),
            validation_policy=TargetValidationPolicy(),
        )
        target_acquisition_service = TargetAcquisitionService(
            world_state_provider=SimulatedWorldStateProvider(runtime),
            retarget_policy=retarget_policy,
        )
        target_approach_service = TargetApproachService(
            target_approach_provider=SimulatedTargetApproachProvider(runtime),
            approach_world_state_provider=SimulatedApproachWorldStateProvider(runtime),
            retarget_policy=retarget_policy,
        )
        target_interaction_service = TargetInteractionService(
            target_interaction_provider=SimulatedTargetInteractionProvider(),
            interaction_world_state_provider=SimulatedInteractionWorldStateProvider(runtime),
            retarget_policy=retarget_policy,
        )
        target_engagement_service = TargetEngagementService(
            acquisition_service=target_acquisition_service,
            approach_service=target_approach_service,
            interaction_service=target_interaction_service,
        )

        return cls(
            orchestrator=orchestrator,
            observation_preparation_service=observation_preparation_service,
            target_engagement_service=target_engagement_service,
            runtime=runtime,
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
        initial_cycle_id = self._runtime.initial_cycle_id
        cycle_ids = [initial_cycle_id + offset for offset in range(total_cycles)]
        observation_preparations = []
        target_resolutions = []
        approach_results = []
        interaction_results = []
        cycle_results = []
        for cycle_id in cycle_ids:
            observation_preparation = self._observation_preparation_service.prepare_observation(
                cycle_id=cycle_id
            )
            engagement = self._target_engagement_service.engage_target(
                cycle_id=cycle_id,
                current_target_id=self._runtime.context_for_cycle(
                    cycle_id
                ).spawn_event.scenario.current_target_id,
            )
            target_resolution = engagement.target_resolution
            approach_result = engagement.approach_result
            interaction_result = engagement.interaction_result
            self._runtime.set_target_resolution(cycle_id, target_resolution)
            self._runtime.set_approach_result(cycle_id, approach_result)
            self._runtime.set_interaction_result(cycle_id, interaction_result)
            observation_preparations.append(observation_preparation)
            target_resolutions.append(target_resolution)
            approach_results.append(approach_result)
            interaction_results.append(interaction_result)
            cycle_results.extend(
                self._orchestrator.run_cycles(
                    1,
                    initial_cycle_id=cycle_id,
                )
            )

        return SimulationReport(
            cycle_results=cycle_results,
            log_path=self._resolve_log_path(),
            sqlite_path=self._storage.sqlite_path,
            observation_preparations=observation_preparations,
            target_resolutions=target_resolutions,
            approach_results=approach_results,
            interaction_results=interaction_results,
            cycle_records=self._storage.fetch_cycles(),
        )

    def _resolve_log_path(self) -> Path:
        for handler in self._logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return Path(handler.baseFilename).resolve()

        raise RuntimeError("Logger nie ma skonfigurowanego FileHandler.")


class SimulatedObservationProvider(ObservationProvider):
    def __init__(self, runtime: SimulatedCycleRuntime) -> None:
        self._runtime = runtime

    def get_observation_window(self, cycle_id: int) -> ObservationWindow:
        context = self._runtime.context_for_cycle(cycle_id)
        spawn_event = context.spawn_event
        trace = context.trace
        return ObservationWindow(
            cycle_id=cycle_id,
            observation=spawn_event.observation,
            actual_spawn_ts=spawn_event.actual_spawn_ts,
            window_closed_ts=trace.ready_window_close_ts,
            note=spawn_event.scenario.note,
            metadata={
                "observable_in_ready_window": spawn_event.observable_in_ready_window,
            },
        )


class SimulatedActionExecutor(ActionExecutor):
    def __init__(
        self,
        runtime: SimulatedCycleRuntime,
        *,
        combat_plan_catalog: SimulatedCombatPlanCatalog,
        combat_profile_catalog: SimulatedCombatProfileCatalog,
    ) -> None:
        self._runtime = runtime
        self._combat_plan_catalog = combat_plan_catalog
        self._combat_profile_catalog = combat_profile_catalog

    def execute_action(self, context: ActionContext) -> ActionResult:
        cycle_context = self._runtime.context_for_cycle(context.cycle_id)
        target_resolution = self._runtime.target_resolution_for_cycle(context.cycle_id)
        approach_result = self._runtime.approach_result_for_cycle(context.cycle_id)
        interaction_result = self._runtime.interaction_result_for_cycle(context.cycle_id)
        engagement_metadata = self._build_engagement_metadata(
            cycle_id=context.cycle_id,
            target_resolution=target_resolution,
            approach_result=approach_result,
            interaction_result=interaction_result,
        )
        selected_target_id = None if interaction_result is None else interaction_result.target_id
        if interaction_result is None:
            selected_target_id = None if approach_result is None else approach_result.target_id
        if interaction_result is None and approach_result is None and target_resolution is not None:
            selected_target_id = target_resolution.selected_target_id

        if selected_target_id is None:
            return ActionResult(
                cycle_id=context.cycle_id,
                success=False,
                executed_at_ts=context.now_ts,
                reason="no_target_available",
                metadata=engagement_metadata,
            )

        attempt_ts = cycle_context.trace.attempt_ts
        if attempt_ts is None:
            raise RuntimeError("Brak attempt_ts dla cyklu z obserwacją.")

        return ActionResult(
            cycle_id=context.cycle_id,
            success=True,
            executed_at_ts=attempt_ts,
            reason="action_executed",
            metadata=engagement_metadata,
        )

    def _build_engagement_metadata(
        self,
        *,
        cycle_id: int,
        target_resolution: "TargetResolution | None",
        approach_result: "TargetApproachResult | None",
        interaction_result: "TargetInteractionResult | None",
    ) -> dict[str, object]:
        combat_metadata = self._build_combat_metadata(cycle_id=cycle_id)
        initial_target_id = None
        if interaction_result is not None and interaction_result.initial_target_id is not None:
            initial_target_id = interaction_result.initial_target_id
        elif approach_result is not None and approach_result.initial_target_id is not None:
            initial_target_id = approach_result.initial_target_id
        elif target_resolution is not None:
            initial_target_id = target_resolution.selected_target_id

        selected_target_id = None
        if interaction_result is not None:
            selected_target_id = interaction_result.target_id
        elif approach_result is not None:
            selected_target_id = approach_result.target_id
        elif target_resolution is not None:
            selected_target_id = target_resolution.selected_target_id

        retargeted_during_approach = False if approach_result is None else approach_result.retargeted
        retargeted_before_interaction = False
        if interaction_result is not None:
            retargeted_before_interaction = bool(
                interaction_result.metadata.get("retargeted_before_interaction_phase", False)
            )

        target_loss_count = 0
        if approach_result is not None and (
            approach_result.retargeted
            or approach_result.reason == "target_lost_during_approach_no_replacement"
        ):
            target_loss_count += 1
        if interaction_result is not None and (
            interaction_result.metadata.get("retargeted_before_interaction_phase", False)
            or interaction_result.reason == "target_lost_before_interaction_no_replacement"
        ):
            target_loss_count += 1

        retarget_count = int(retargeted_during_approach) + int(retargeted_before_interaction)

        engagement_duration_s = None
        if approach_result is not None and interaction_result is not None:
            engagement_duration_s = max(
                interaction_result.observed_at_ts,
                approach_result.completed_at_ts,
            ) - approach_result.started_at_ts
        elif approach_result is not None:
            engagement_duration_s = approach_result.completed_at_ts - approach_result.started_at_ts

        return {
            **combat_metadata,
            "initial_target_id": initial_target_id,
            "selected_target_id": selected_target_id,
            "target_decision_reason": (
                None if target_resolution is None else target_resolution.decision.reason
            ),
            "approach_reason": None if approach_result is None else approach_result.reason,
            "interaction_reason": None if interaction_result is None else interaction_result.reason,
            "approach_travel_s": None if approach_result is None else approach_result.travel_s,
            "engagement_duration_s": engagement_duration_s,
            "interaction_ready": (
                None if interaction_result is None else interaction_result.ready
            ),
            "target_loss_count": target_loss_count,
            "retarget_count": retarget_count,
            "retargeted_during_approach": retargeted_during_approach,
            "retargeted_before_interaction": retargeted_before_interaction,
        }

    def _build_combat_metadata(self, *, cycle_id: int) -> dict[str, object]:
        scenario = self._runtime.context_for_cycle(cycle_id).spawn_event.scenario

        if scenario.combat_profile_name is not None:
            selection = self._combat_profile_catalog.select_profile(scenario.combat_profile_name)
        else:
            selection = self._combat_plan_catalog.select_plan(
                plan_name=scenario.combat_plan_name,
                round_sequences=scenario.combat_plan_rounds,
                input_sequence=scenario.combat_inputs,
            )

        metadata = {
            "combat_plan_name": selection.plan_name,
            "combat_plan_source": selection.source,
            "combat_input_sequence": selection.plan.to_input_sequence(),
            "combat_strategy": scenario.combat_strategy,
        }
        if "combat_profile_name" in selection.metadata:
            metadata["combat_profile_name"] = selection.metadata["combat_profile_name"]
        elif scenario.combat_profile_name is not None:
            metadata["combat_profile_name"] = scenario.combat_profile_name
        else:
            metadata["combat_profile_name"] = None

        return metadata


class SimulatedVerificationProvider(VerificationProvider):
    def __init__(self, runtime: SimulatedCycleRuntime) -> None:
        self._runtime = runtime

    def verify(self, cycle_id: int, observation: Observation) -> VerificationResult:
        context = self._runtime.context_for_cycle(cycle_id)
        trace = context.trace
        verify_result = context.spawn_event.verify_result
        if verify_result is None:
            raise RuntimeError("Brak verify_result dla cyklu z obserwacją.")
        if trace.verify_start_ts is None:
            raise RuntimeError("Brak verify_start_ts dla cyklu z obserwacją.")

        if verify_result == "timeout":
            completed_at_ts = trace.verify_timeout_ts
            outcome = VerificationOutcome.TIMEOUT
        elif verify_result == "success":
            completed_at_ts = trace.verify_resolution_ts
            outcome = VerificationOutcome.SUCCESS
        elif verify_result == "failure":
            completed_at_ts = trace.verify_resolution_ts
            outcome = VerificationOutcome.FAILURE
        else:
            raise ValueError(f"Nieznany verify_result: {verify_result}")

        if completed_at_ts is None:
            raise RuntimeError("Brak completed_at_ts dla verify.")

        return VerificationResult(
            cycle_id=cycle_id,
            outcome=outcome,
            started_at_ts=trace.verify_start_ts,
            completed_at_ts=completed_at_ts,
            reason=verify_result,
            metadata={},
        )


class SimulatedCombatResolver(CombatResolver):
    def __init__(self, runtime: SimulatedCycleRuntime, battle: SimulatedBattle) -> None:
        self._runtime = runtime
        self._battle = battle

    def resolve_combat(
        self,
        cycle_id: int,
        *,
        combat_started_ts: float,
        observation: Observation,
    ) -> CombatTimeline:
        context = self._runtime.context_for_cycle(cycle_id)
        snapshots = self._battle.build_timeline(
            cycle_id=cycle_id,
            combat_started_ts=combat_started_ts,
            scenario=context.spawn_event.scenario,
        )
        return CombatTimeline(
            cycle_id=cycle_id,
            snapshots=snapshots,
            metadata={"note": context.spawn_event.scenario.note},
        )


class SimulatedRestProvider(RestProvider):
    def __init__(self, runtime: SimulatedCycleRuntime, rest: SimulatedRest) -> None:
        self._runtime = runtime
        self._rest = rest

    def apply_rest(
        self,
        cycle_id: int,
        *,
        rest_started_ts: float,
        starting_hp_ratio: float,
        observation: Observation,
    ) -> RestTimeline:
        context = self._runtime.context_for_cycle(cycle_id)
        snapshots = self._rest.build_timeline(
            cycle_id=cycle_id,
            rest_started_ts=rest_started_ts,
            starting_hp_ratio=starting_hp_ratio,
            scenario=context.spawn_event.scenario,
        )
        return RestTimeline(
            cycle_id=cycle_id,
            snapshots=snapshots,
            metadata={"note": context.spawn_event.scenario.note},
        )


class SimulatedTelemetrySink(TelemetrySink):
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
