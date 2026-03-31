from botlab.application import (
    ActionContext,
    ActionExecutor,
    ActionResult,
    Clock,
    CombatOutcome,
    CombatResolver,
    Observation,
    ObservationProvider,
    RestOutcome,
    RestProvider,
    TelemetrySink,
    VerificationOutcome,
    VerificationProvider,
)


class DummyClock:
    def now(self) -> float:
        return 1234.5

    def sleep(self, delay_s: float) -> None:
        assert delay_s >= 0


class DummyObservationProvider:
    def get_latest_observation(self, cycle_id: int) -> Observation | None:
        return Observation(
            cycle_id=cycle_id,
            observed_at_ts=100.0,
            signal_detected=True,
            actual_spawn_ts=100.0,
        )


class DummyActionExecutor:
    def execute_action(self, context: ActionContext) -> ActionResult:
        return ActionResult(cycle_id=context.cycle_id, success=True, reason="executed")


class DummyVerificationProvider:
    def verify(self, cycle_id: int, observation: Observation | None) -> VerificationOutcome:
        return VerificationOutcome.SUCCESS


class DummyCombatResolver:
    def resolve_combat(self, cycle_id: int, state_snapshot: Observation) -> CombatOutcome:
        return CombatOutcome(cycle_id=cycle_id, won=True, hp_ratio=0.7, metadata={})


class DummyRestProvider:
    def apply_rest(self, cycle_id: int, state_snapshot: Observation) -> RestOutcome:
        return RestOutcome(cycle_id=cycle_id, hp_ratio=0.95, recovered=True, metadata={})


class DummyTelemetrySink:
    def record_cycle(self, payload: dict) -> None:
        assert isinstance(payload, dict)

    def record_state_transition(self, payload: dict) -> None:
        assert isinstance(payload, dict)

    def record_attempt(self, payload: dict) -> None:
        assert isinstance(payload, dict)


def test_application_ports_protocols_runtime() -> None:
    assert isinstance(DummyClock(), Clock)
    assert isinstance(DummyObservationProvider(), ObservationProvider)
    assert isinstance(DummyActionExecutor(), ActionExecutor)
    assert isinstance(DummyVerificationProvider(), VerificationProvider)
    assert isinstance(DummyCombatResolver(), CombatResolver)
    assert isinstance(DummyRestProvider(), RestProvider)
    assert isinstance(DummyTelemetrySink(), TelemetrySink)
