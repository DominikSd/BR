from botlab.application import (
    ActionContext,
    ActionExecutor,
    ActionResult,
    ApproachWorldStateProvider,
    Clock,
    CombatPlanCatalog,
    CombatPlanSelection,
    CombatResolver,
    CombatTimeline,
    InteractionWorldStateProvider,
    Observation,
    ObservationPreparationProvider,
    ObservationPreparationResult,
    ObservationProvider,
    ObservationWindow,
    RestProvider,
    RestTimeline,
    TelemetrySink,
    TimedCombatSnapshot,
    TargetApproachProvider,
    TargetApproachResult,
    TargetInteractionProvider,
    TargetInteractionResult,
    TargetResolution,
    VerificationOutcome,
    VerificationProvider,
    VerificationResult,
    WorldStateProvider,
)
from botlab.domain.combat_plan import CombatPlan
from botlab.domain.world import Position, WorldSnapshot
from botlab.types import CombatSnapshot, TelemetryRecord


class DummyClock:
    def now(self) -> float:
        return 1234.5

    def sleep(self, delay_s: float) -> None:
        assert delay_s >= 0


class DummyObservationProvider:
    def get_observation_window(self, cycle_id: int) -> ObservationWindow:
        observation = Observation(
            cycle_id=cycle_id,
            observed_at_ts=100.0,
            signal_detected=True,
            actual_spawn_ts=100.0,
        )
        return ObservationWindow(
            cycle_id=cycle_id,
            observation=observation,
            actual_spawn_ts=100.0,
            window_closed_ts=101.0,
        )


class DummyObservationPreparationProvider:
    def prepare_observation(self, cycle_id: int) -> ObservationPreparationResult:
        return ObservationPreparationResult(
            cycle_id=cycle_id,
            spawn_zone_visible=True,
            ready_for_observation=True,
        )


class DummyWorldStateProvider:
    def get_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        return WorldSnapshot(
            observed_at_ts=100.0,
            bot_position=Position(x=0.0, y=0.0),
            groups=(),
        )


class DummyApproachWorldStateProvider:
    def get_approach_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        return WorldSnapshot(
            observed_at_ts=100.25,
            bot_position=Position(x=1.0, y=0.0),
            groups=(),
        )


class DummyInteractionWorldStateProvider:
    def get_interaction_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        return WorldSnapshot(
            observed_at_ts=100.5,
            bot_position=Position(x=1.5, y=0.0),
            groups=(),
        )


class DummyTargetApproachProvider:
    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        return TargetApproachResult(
            cycle_id=target_resolution.cycle_id,
            target_id=target_resolution.selected_target_id,
            started_at_ts=100.0,
            completed_at_ts=100.5,
            travel_s=0.5,
            arrived=target_resolution.selected_target_id is not None,
            reason="approach_finished",
        )


class DummyTargetInteractionProvider:
    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        return TargetInteractionResult(
            cycle_id=target_approach_result.cycle_id,
            target_id=target_approach_result.target_id,
            ready=target_approach_result.target_id is not None,
            observed_at_ts=100.75,
            reason="interaction_ready",
        )


class DummyCombatPlanCatalog:
    def select_plan(
        self,
        *,
        plan_name: str | None = None,
        input_sequence: tuple[str, ...] | None = None,
        round_sequences: tuple[tuple[str, ...], ...] | None = None,
    ) -> CombatPlanSelection:
        if round_sequences is not None:
            plan = CombatPlan.from_round_sequences(round_sequences, name=plan_name or "dummy-rounds")
        else:
            plan = CombatPlan.from_input_sequence(input_sequence or ("1", "space"), name=plan_name or "dummy")
        return CombatPlanSelection(
            plan_name=plan.name,
            plan=plan,
            source="dummy",
        )

    def available_plan_names(self) -> tuple[str, ...]:
        return ("dummy",)


class DummyActionExecutor:
    def execute_action(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            cycle_id=context.cycle_id,
            success=True,
            executed_at_ts=context.now_ts + 0.02,
            reason="executed",
        )


class DummyVerificationProvider:
    def verify(self, cycle_id: int, observation: Observation) -> VerificationResult:
        return VerificationResult(
            cycle_id=cycle_id,
            outcome=VerificationOutcome.SUCCESS,
            started_at_ts=observation.observed_at_ts + 0.03,
            completed_at_ts=observation.observed_at_ts + 0.13,
            reason="success",
        )


class DummyCombatResolver:
    def resolve_combat(
        self,
        cycle_id: int,
        *,
        combat_started_ts: float,
        observation: Observation,
    ) -> CombatTimeline:
        snapshot = CombatSnapshot(
            hp_ratio=0.7,
            turn_index=1,
            enemy_count=0,
            strategy="default",
            in_combat=False,
            combat_started_ts=combat_started_ts,
            combat_finished_ts=combat_started_ts + 0.3,
        )
        return CombatTimeline(
            cycle_id=cycle_id,
            snapshots=[TimedCombatSnapshot(event_ts=combat_started_ts + 0.3, snapshot=snapshot)],
        )


class DummyRestProvider:
    def apply_rest(
        self,
        cycle_id: int,
        *,
        rest_started_ts: float,
        starting_hp_ratio: float,
        starting_condition_ratio: float = 1.0,
        observation: Observation,
    ) -> RestTimeline:
        snapshot = CombatSnapshot(
            hp_ratio=0.95,
            turn_index=1,
            enemy_count=0,
            strategy="rest",
            in_combat=False,
            condition_ratio=max(starting_condition_ratio, 0.95),
        )
        return RestTimeline(
            cycle_id=cycle_id,
            snapshots=[TimedCombatSnapshot(event_ts=rest_started_ts + 0.5, snapshot=snapshot)],
        )


class DummyTelemetrySink:
    def record_cycle(self, record: TelemetryRecord) -> None:
        assert isinstance(record, TelemetryRecord)

    def record_state_transition(self, record: TelemetryRecord) -> None:
        assert isinstance(record, TelemetryRecord)

    def record_attempt(self, record: TelemetryRecord) -> None:
        assert isinstance(record, TelemetryRecord)


def test_application_ports_protocols_runtime() -> None:
    assert isinstance(DummyClock(), Clock)
    assert isinstance(DummyObservationProvider(), ObservationProvider)
    assert isinstance(DummyObservationPreparationProvider(), ObservationPreparationProvider)
    assert isinstance(DummyWorldStateProvider(), WorldStateProvider)
    assert isinstance(DummyApproachWorldStateProvider(), ApproachWorldStateProvider)
    assert isinstance(DummyInteractionWorldStateProvider(), InteractionWorldStateProvider)
    assert isinstance(DummyTargetApproachProvider(), TargetApproachProvider)
    assert isinstance(DummyTargetInteractionProvider(), TargetInteractionProvider)
    assert isinstance(DummyCombatPlanCatalog(), CombatPlanCatalog)
    assert isinstance(DummyActionExecutor(), ActionExecutor)
    assert isinstance(DummyVerificationProvider(), VerificationProvider)
    assert isinstance(DummyCombatResolver(), CombatResolver)
    assert isinstance(DummyRestProvider(), RestProvider)
    assert isinstance(DummyTelemetrySink(), TelemetrySink)
