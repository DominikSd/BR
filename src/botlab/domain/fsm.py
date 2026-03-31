from __future__ import annotations

from dataclasses import dataclass

from botlab.domain.decision_engine import DecisionContext, DecisionEngine, VerifyResult
from botlab.types import BotState, CombatSnapshot, CyclePrediction, Decision, Observation


class FSMError(RuntimeError):
    """Błąd automatu stanów."""


@dataclass(slots=True, frozen=True)
class StateTransition:
    cycle_id: int | None
    at_ts: float
    from_state: BotState
    to_state: BotState
    reason: str


class CycleFSM:
    """
    Minimalny automat stanów dla cyklicznego systemu PvE.

    Odpowiada za:
    - przechowywanie aktualnego stanu,
    - pamiętanie momentu wejścia do stanu,
    - wołanie DecisionEngine,
    - stosowanie przejść,
    - zapisywanie historii przejść.

    Automat nie zawiera schedulera ani predictora wewnątrz.
    Dostaje tylko gotowy kontekst na każdy tick.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        *,
        initial_state: BotState = BotState.IDLE,
        started_at_ts: float = 0.0,
        cycle_id: int | None = None,
    ) -> None:
        self._decision_engine = decision_engine
        self._current_state = initial_state
        self._state_entered_ts = started_at_ts
        self._current_cycle_id = cycle_id
        self._history: list[StateTransition] = []
        self._last_decision: Decision | None = None

    @property
    def current_state(self) -> BotState:
        return self._current_state

    @property
    def state_entered_ts(self) -> float:
        return self._state_entered_ts

    @property
    def current_cycle_id(self) -> int | None:
        return self._current_cycle_id

    @property
    def last_decision(self) -> Decision | None:
        return self._last_decision

    def state_elapsed_s(self, now_ts: float) -> float:
        return now_ts - self._state_entered_ts

    def transition_history(self) -> list[StateTransition]:
        return list(self._history)

    def transition_count(self) -> int:
        return len(self._history)

    def force_state(
        self,
        *,
        new_state: BotState,
        now_ts: float,
        reason: str,
        cycle_id: int | None = None,
    ) -> None:
        previous_state = self._current_state

        if cycle_id is not None:
            self._current_cycle_id = cycle_id

        if previous_state != new_state:
            self._history.append(
                StateTransition(
                    cycle_id=self._current_cycle_id,
                    at_ts=now_ts,
                    from_state=previous_state,
                    to_state=new_state,
                    reason=reason,
                )
            )

        self._current_state = new_state
        self._state_entered_ts = now_ts

    def tick(
        self,
        *,
        now_ts: float,
        prediction: CyclePrediction | None = None,
        temporal_state: BotState | None = None,
        observation: Observation | None = None,
        verify_result: VerifyResult | None = None,
        combat_snapshot: CombatSnapshot | None = None,
    ) -> Decision:
        context = DecisionContext(
            now_ts=now_ts,
            current_state=self._current_state,
            state_entered_ts=self._state_entered_ts,
            cycle_id=self._current_cycle_id,
            prediction=prediction,
            temporal_state=temporal_state,
            observation=observation,
            verify_result=verify_result,
            combat_snapshot=combat_snapshot,
        )

        decision = self._decision_engine.decide(context)
        self._apply_decision(decision=decision, now_ts=now_ts)

        return decision

    def _apply_decision(self, *, decision: Decision, now_ts: float) -> None:
        if decision.cycle_id is not None:
            self._current_cycle_id = decision.cycle_id

        previous_state = self._current_state
        next_state = decision.next_state

        if next_state != previous_state:
            self._history.append(
                StateTransition(
                    cycle_id=self._current_cycle_id,
                    at_ts=now_ts,
                    from_state=previous_state,
                    to_state=next_state,
                    reason=decision.reason,
                )
            )
            self._current_state = next_state
            self._state_entered_ts = now_ts

        self._last_decision = decision
