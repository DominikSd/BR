from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from botlab.config import CombatConfig, CycleConfig
from botlab.types import BotState, CombatSnapshot, CyclePrediction, Decision, Observation


VerifyResult = Literal["success", "failure"]


class DecisionEngineError(RuntimeError):
    """Błąd silnika decyzji."""


@dataclass(slots=True, frozen=True)
class DecisionContext:
    now_ts: float
    current_state: BotState
    state_entered_ts: float
    cycle_id: int | None = None
    prediction: CyclePrediction | None = None
    temporal_state: BotState | None = None
    observation: Observation | None = None
    verify_result: VerifyResult | None = None
    combat_snapshot: CombatSnapshot | None = None

    @property
    def state_elapsed_s(self) -> float:
        return self.now_ts - self.state_entered_ts

    @property
    def effective_cycle_id(self) -> int | None:
        if self.cycle_id is not None:
            return self.cycle_id
        if self.prediction is not None:
            return self.prediction.cycle_id
        return None


class DecisionEngine:
    """
    Minimalny silnik decyzji dla FSM.

    Odpowiada za wybór następnego stanu na podstawie:
    - aktualnego stanu,
    - czasu,
    - pozycji względem okna cyklu,
    - obserwacji sygnału,
    - wyniku weryfikacji,
    - stanu walki / HP.

    Zasady:
    - logika ma być jawna,
    - bez ukrytej magii,
    - bez automatycznej optymalizacji,
    - priorytet: stabilność i testowalność.
    """

    def __init__(self, cycle_config: CycleConfig, combat_config: CombatConfig) -> None:
        self._cycle_config = cycle_config
        self._combat_config = combat_config

    def decide(self, context: DecisionContext) -> Decision:
        if context.current_state in {
            BotState.IDLE,
            BotState.WAIT_NEXT_CYCLE,
            BotState.PREPARE_WINDOW,
            BotState.READY_WINDOW,
        }:
            self._require_temporal_context(context)

        if context.current_state == BotState.IDLE:
            return self._decide_idle(context)

        if context.current_state == BotState.WAIT_NEXT_CYCLE:
            return self._decide_wait_next_cycle(context)

        if context.current_state == BotState.PREPARE_WINDOW:
            return self._decide_prepare_window(context)

        if context.current_state == BotState.READY_WINDOW:
            return self._decide_ready_window(context)

        if context.current_state == BotState.ATTEMPT:
            return self._decide_attempt(context)

        if context.current_state == BotState.VERIFY:
            return self._decide_verify(context)

        if context.current_state == BotState.COMBAT:
            return self._decide_combat(context)

        if context.current_state == BotState.REST:
            return self._decide_rest(context)

        if context.current_state == BotState.RECOVER:
            return self._decide_recover(context)

        raise DecisionEngineError(f"Nieobsługiwany stan: {context.current_state}")

    def _decide_idle(self, context: DecisionContext) -> Decision:
        assert context.temporal_state is not None

        if context.temporal_state == BotState.PREPARE_WINDOW:
            return self._build_decision(
                context=context,
                next_state=BotState.PREPARE_WINDOW,
                action="enter_prepare_window",
                reason="cycle_prepare_window_reached",
            )

        if context.temporal_state == BotState.READY_WINDOW:
            return self._build_decision(
                context=context,
                next_state=BotState.READY_WINDOW,
                action="enter_ready_window",
                reason="cycle_ready_window_reached",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.WAIT_NEXT_CYCLE,
            action="wait_for_cycle",
            reason="waiting_for_next_cycle",
        )

    def _decide_wait_next_cycle(self, context: DecisionContext) -> Decision:
        assert context.temporal_state is not None

        if context.temporal_state == BotState.PREPARE_WINDOW:
            return self._build_decision(
                context=context,
                next_state=BotState.PREPARE_WINDOW,
                action="enter_prepare_window",
                reason="prepare_window_opened",
            )

        if context.temporal_state == BotState.READY_WINDOW:
            return self._build_decision(
                context=context,
                next_state=BotState.READY_WINDOW,
                action="enter_ready_window",
                reason="ready_window_opened",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.WAIT_NEXT_CYCLE,
            action="hold_state",
            reason="still_waiting_for_cycle",
        )

    def _decide_prepare_window(self, context: DecisionContext) -> Decision:
        assert context.temporal_state is not None
        assert context.prediction is not None

        if context.temporal_state == BotState.READY_WINDOW:
            return self._build_decision(
                context=context,
                next_state=BotState.READY_WINDOW,
                action="enter_ready_window",
                reason="ready_window_opened",
            )

        if context.now_ts > context.prediction.ready_window_end_ts:
            return self._build_decision(
                context=context,
                next_state=BotState.WAIT_NEXT_CYCLE,
                action="skip_cycle",
                reason="ready_window_missed_from_prepare",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.PREPARE_WINDOW,
            action="hold_state",
            reason="preparing_for_cycle",
        )

    def _decide_ready_window(self, context: DecisionContext) -> Decision:
        assert context.temporal_state is not None
        assert context.prediction is not None

        if self._has_valid_signal(context.observation):
            return self._build_decision(
                context=context,
                next_state=BotState.ATTEMPT,
                action="attempt_reaction",
                reason="signal_detected_in_ready_window",
            )

        if context.now_ts > context.prediction.ready_window_end_ts:
            return self._build_decision(
                context=context,
                next_state=BotState.WAIT_NEXT_CYCLE,
                action="close_cycle_window",
                reason="no_signal_before_ready_window_timeout",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.READY_WINDOW,
            action="hold_state",
            reason="waiting_for_signal_in_ready_window",
        )

    def _decide_attempt(self, context: DecisionContext) -> Decision:
        return self._build_decision(
            context=context,
            next_state=BotState.VERIFY,
            action="start_verification",
            reason="attempt_dispatched_waiting_for_verification",
        )

    def _decide_verify(self, context: DecisionContext) -> Decision:
        if context.verify_result == "success":
            return self._build_decision(
                context=context,
                next_state=BotState.COMBAT,
                action="enter_combat",
                reason="verification_success",
            )

        if context.verify_result == "failure":
            return self._build_decision(
                context=context,
                next_state=BotState.WAIT_NEXT_CYCLE,
                action="return_to_wait",
                reason="verification_failure",
            )

        if context.state_elapsed_s >= self._cycle_config.verify_timeout_s:
            return self._build_decision(
                context=context,
                next_state=BotState.RECOVER,
                action="enter_recovery",
                reason="verify_timeout",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.VERIFY,
            action="hold_state",
            reason="verification_pending",
        )

    def _decide_combat(self, context: DecisionContext) -> Decision:
        snapshot = context.combat_snapshot

        if snapshot is None or snapshot.in_combat:
            return self._build_decision(
                context=context,
                next_state=BotState.COMBAT,
                action="hold_state",
                reason="combat_in_progress",
            )

        if snapshot.hp_ratio < self._combat_config.rest_start_threshold:
            return self._build_decision(
                context=context,
                next_state=BotState.REST,
                action="enter_rest",
                reason="combat_finished_low_hp",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.WAIT_NEXT_CYCLE,
            action="return_to_wait",
            reason="combat_finished_no_rest_needed",
        )

    def _decide_rest(self, context: DecisionContext) -> Decision:
        snapshot = context.combat_snapshot

        if snapshot is None:
            return self._build_decision(
                context=context,
                next_state=BotState.REST,
                action="hold_state",
                reason="resting_without_snapshot",
            )

        if snapshot.hp_ratio >= self._combat_config.rest_stop_threshold:
            return self._build_decision(
                context=context,
                next_state=BotState.WAIT_NEXT_CYCLE,
                action="finish_rest",
                reason="rest_completed_hp_restored",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.REST,
            action="hold_state",
            reason="rest_in_progress",
        )

    def _decide_recover(self, context: DecisionContext) -> Decision:
        if context.state_elapsed_s >= self._cycle_config.recover_timeout_s:
            return self._build_decision(
                context=context,
                next_state=BotState.WAIT_NEXT_CYCLE,
                action="complete_recovery",
                reason="recover_timeout_elapsed",
            )

        return self._build_decision(
            context=context,
            next_state=BotState.RECOVER,
            action="hold_state",
            reason="recovery_in_progress",
        )

    def _build_decision(
        self,
        *,
        context: DecisionContext,
        next_state: BotState,
        action: str,
        reason: str,
    ) -> Decision:
        return Decision(
            cycle_id=context.effective_cycle_id,
            state=context.current_state,
            next_state=next_state,
            action=action,
            reason=reason,
            decided_at_ts=context.now_ts,
            metadata={
                "state_elapsed_s": context.state_elapsed_s,
            },
        )

    def _require_temporal_context(self, context: DecisionContext) -> None:
        if context.temporal_state is None:
            raise DecisionEngineError(
                "Dla stanów czasowych wymagane jest pole temporal_state."
            )

        if context.current_state in {BotState.PREPARE_WINDOW, BotState.READY_WINDOW}:
            if context.prediction is None:
                raise DecisionEngineError(
                    "Dla PREPARE_WINDOW i READY_WINDOW wymagane jest pole prediction."
                )

    def _has_valid_signal(self, observation: Observation | None) -> bool:
        if observation is None:
            return False
        return observation.signal_detected is True
