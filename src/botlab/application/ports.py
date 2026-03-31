from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from botlab.application.dto import (
    ActionContext,
    ActionResult,
    CombatOutcome,
    RestOutcome,
    VerificationOutcome,
)
from botlab.types import Observation


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """Obecny czas w sekundach (timestamp)."""

    def sleep(self, delay_s: float) -> None:
        """Czekaj asynchronicznie lub synchronnie przez delay_s sekund."""


@runtime_checkable
class ObservationProvider(Protocol):
    def get_latest_observation(self, cycle_id: int) -> Observation | None:
        """Zwróć obserwację dla cyklu, jeśli dostępna."""


@runtime_checkable
class ActionExecutor(Protocol):
    def execute_action(self, context: ActionContext) -> ActionResult:
        """Wykonuje zadanie sterowania (np. reakcję na zdarzenie)."""


@runtime_checkable
class VerificationProvider(Protocol):
    def verify(self, cycle_id: int, observation: Observation | None) -> VerificationOutcome:
        """Weryfikuje, czy podjęto poprawne działanie w danym cyklu."""


@runtime_checkable
class CombatResolver(Protocol):
    def resolve_combat(self, cycle_id: int, state_snapshot: Observation) -> CombatOutcome:
        """Symuluje przeszłą walkę i zwraca wynik końcowy."""


@runtime_checkable
class RestProvider(Protocol):
    def apply_rest(self, cycle_id: int, state_snapshot: Observation) -> RestOutcome:
        """Symuluje regenerację po walce."""


@runtime_checkable
class TelemetrySink(Protocol):
    def record_cycle(self, payload: dict) -> None:
        """Zapisuje podsumowanie cyklu.

        payload zawiera dostateczne pole z telemetrią; może pochodzić z rodzaju
        TelemetryRecord w domenie.
        """

    def record_state_transition(self, payload: dict) -> None:
        """Zapisuje pojedyncze przejście stanu."""

    def record_attempt(self, payload: dict) -> None:
        """Zapisuje dane próby działania/reakcji w cyklu."""
