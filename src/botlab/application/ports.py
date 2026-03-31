from __future__ import annotations

from typing import Protocol, runtime_checkable

from botlab.application.dto import (
    ActionContext,
    ActionResult,
    CombatPlanSelection,
    CombatTimeline,
    ObservationPreparationResult,
    ObservationWindow,
    RestTimeline,
    TargetApproachResult,
    TargetInteractionResult,
    TargetResolution,
    VerificationResult,
)
from botlab.domain.world import WorldSnapshot
from botlab.types import Observation, TelemetryRecord


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Obecny czas w sekundach (timestamp)."""

    def sleep(self, delay_s: float) -> None:
        """Czekaj asynchronicznie lub synchronnie przez delay_s sekund."""


@runtime_checkable
class ObservationProvider(Protocol):
    def get_observation_window(self, cycle_id: int) -> ObservationWindow:
        """Zwraca wynik obserwacji dla danego cyklu."""


@runtime_checkable
class ObservationPreparationProvider(Protocol):
    def prepare_observation(self, cycle_id: int) -> ObservationPreparationResult:
        """Zwraca wynik przygotowania bota do obserwacji strefy spawnu."""


@runtime_checkable
class WorldStateProvider(Protocol):
    def get_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        """Zwraca snapshot swiata dla danego cyklu."""


@runtime_checkable
class ApproachWorldStateProvider(Protocol):
    def get_approach_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        """Zwraca snapshot swiata do rewalidacji celu podczas dojscia."""


@runtime_checkable
class InteractionWorldStateProvider(Protocol):
    def get_interaction_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        """Zwraca snapshot swiata do koncowej walidacji celu przed interakcja."""


@runtime_checkable
class TargetApproachProvider(Protocol):
    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        """Zwraca wynik dojscia do wybranego celu."""


@runtime_checkable
class TargetInteractionProvider(Protocol):
    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        """Zwraca techniczny wynik przygotowania do interakcji po dojsciu."""


@runtime_checkable
class CombatPlanCatalog(Protocol):
    def select_plan(
        self,
        *,
        plan_name: str | None = None,
        input_sequence: tuple[str, ...] | None = None,
        round_sequences: tuple[tuple[str, ...], ...] | None = None,
    ) -> CombatPlanSelection:
        """Rozwiazuje nazwany plan walki albo buduje plan z jawnej sekwencji wejsc."""

    def available_plan_names(self) -> tuple[str, ...]:
        """Zwraca dostepne nazwy planow walki."""


@runtime_checkable
class ActionExecutor(Protocol):
    def execute_action(self, context: ActionContext) -> ActionResult:
        """Wykonuje zadanie sterowania, np. reakcje na zdarzenie."""


@runtime_checkable
class VerificationProvider(Protocol):
    def verify(self, cycle_id: int, observation: Observation) -> VerificationResult:
        """Weryfikuje wynik podjetej akcji dla danego cyklu."""


@runtime_checkable
class CombatResolver(Protocol):
    def resolve_combat(
        self,
        cycle_id: int,
        *,
        combat_started_ts: float,
        observation: Observation,
    ) -> CombatTimeline:
        """Zwraca timeline walki dla danego cyklu."""


@runtime_checkable
class RestProvider(Protocol):
    def apply_rest(
        self,
        cycle_id: int,
        *,
        rest_started_ts: float,
        starting_hp_ratio: float,
        observation: Observation,
    ) -> RestTimeline:
        """Zwraca timeline regeneracji po walce."""


@runtime_checkable
class TelemetrySink(Protocol):
    def record_cycle(self, record: TelemetryRecord) -> None:
        """Zapisuje podsumowanie cyklu."""

    def record_state_transition(self, record: TelemetryRecord) -> None:
        """Zapisuje pojedyncze przejscie stanu."""

    def record_attempt(self, record: TelemetryRecord) -> None:
        """Zapisuje dane proby dzialania lub reakcji w cyklu."""
