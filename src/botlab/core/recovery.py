from __future__ import annotations

from dataclasses import dataclass

from botlab.config import CycleConfig
from botlab.types import BotState, CyclePrediction


@dataclass(slots=True, frozen=True)
class RecoveryStep:
    cycle_id: int | None
    at_ts: float
    from_state: BotState | None
    target_state: BotState
    reason: str


class RecoveryManager:
    """
    Minimalna warstwa recovery / odporności.

    Odpowiada za:
    - wykrywanie stanów utkniętych,
    - budowanie jawnego planu bezpiecznego resetu,
    - budowanie planu recovery po wyjątkach wykonania.

    Zasada działania:
    - jeśli system jest w stanie niebezpiecznym albo niespójnym,
      budujemy plan przejścia:
        * RECOVER
        * WAIT_NEXT_CYCLE
    - dla niektórych stanów czasowych można bezpiecznie przejść od razu
      do WAIT_NEXT_CYCLE.
    """

    def __init__(self, cycle_config: CycleConfig) -> None:
        self._cycle_config = cycle_config

    def is_neutral_state(self, state: BotState) -> bool:
        return state in {BotState.IDLE, BotState.WAIT_NEXT_CYCLE}

    def detect_stuck_state(
        self,
        *,
        now_ts: float,
        current_state: BotState,
        state_entered_ts: float,
        cycle_id: int | None = None,
        prediction: CyclePrediction | None = None,
    ) -> list[RecoveryStep] | None:
        state_elapsed_s = max(0.0, now_ts - state_entered_ts)

        if current_state == BotState.PREPARE_WINDOW:
            if prediction is not None and now_ts > prediction.ready_window_end_ts:
                return [
                    RecoveryStep(
                        cycle_id=cycle_id,
                        at_ts=now_ts,
                        from_state=current_state,
                        target_state=BotState.WAIT_NEXT_CYCLE,
                        reason="prepare_window_stuck_past_ready_window",
                    )
                ]
            return None

        if current_state == BotState.READY_WINDOW:
            if prediction is not None and now_ts > prediction.ready_window_end_ts:
                return [
                    RecoveryStep(
                        cycle_id=cycle_id,
                        at_ts=now_ts,
                        from_state=current_state,
                        target_state=BotState.WAIT_NEXT_CYCLE,
                        reason="ready_window_stuck_past_ready_window",
                    )
                ]
            return None

        if current_state == BotState.ATTEMPT:
            if state_elapsed_s >= self._cycle_config.verify_timeout_s:
                return self.build_safe_reset_plan(
                    now_ts=now_ts,
                    current_state=current_state,
                    cycle_id=cycle_id,
                    reason="attempt_stuck_timeout",
                )
            return None

        if current_state == BotState.VERIFY:
            if state_elapsed_s >= self._cycle_config.verify_timeout_s:
                return self.build_safe_reset_plan(
                    now_ts=now_ts,
                    current_state=current_state,
                    cycle_id=cycle_id,
                    reason="verify_stuck_timeout",
                )
            return None

        if current_state == BotState.COMBAT:
            if state_elapsed_s >= self._cycle_config.interval_s:
                return self.build_safe_reset_plan(
                    now_ts=now_ts,
                    current_state=current_state,
                    cycle_id=cycle_id,
                    reason="combat_stuck_timeout",
                )
            return None

        if current_state == BotState.REST:
            if state_elapsed_s >= self._cycle_config.interval_s:
                return self.build_safe_reset_plan(
                    now_ts=now_ts,
                    current_state=current_state,
                    cycle_id=cycle_id,
                    reason="rest_stuck_timeout",
                )
            return None

        if current_state == BotState.RECOVER:
            if state_elapsed_s >= self._cycle_config.recover_timeout_s:
                return [
                    RecoveryStep(
                        cycle_id=cycle_id,
                        at_ts=now_ts,
                        from_state=current_state,
                        target_state=BotState.WAIT_NEXT_CYCLE,
                        reason="recover_timeout_force_reset",
                    )
                ]
            return None

        return None

    def build_exception_recovery_plan(
        self,
        *,
        now_ts: float,
        current_state: BotState,
        cycle_id: int | None,
        exception: Exception | str,
    ) -> list[RecoveryStep]:
        if isinstance(exception, Exception):
            suffix = type(exception).__name__
        else:
            suffix = str(exception)

        return self.build_safe_reset_plan(
            now_ts=now_ts,
            current_state=current_state,
            cycle_id=cycle_id,
            reason=f"execution_error_{suffix}",
        )

    def ensure_neutral_state_plan(
        self,
        *,
        now_ts: float,
        current_state: BotState,
        cycle_id: int | None,
        reason: str,
    ) -> list[RecoveryStep] | None:
        if self.is_neutral_state(current_state):
            return None

        return self.build_safe_reset_plan(
            now_ts=now_ts,
            current_state=current_state,
            cycle_id=cycle_id,
            reason=reason,
        )

    def build_safe_reset_plan(
        self,
        *,
        now_ts: float,
        current_state: BotState,
        cycle_id: int | None,
        reason: str,
    ) -> list[RecoveryStep]:
        if current_state == BotState.WAIT_NEXT_CYCLE:
            return [
                RecoveryStep(
                    cycle_id=cycle_id,
                    at_ts=now_ts,
                    from_state=current_state,
                    target_state=BotState.WAIT_NEXT_CYCLE,
                    reason=f"{reason}_already_neutral",
                )
            ]

        if current_state == BotState.RECOVER:
            return [
                RecoveryStep(
                    cycle_id=cycle_id,
                    at_ts=now_ts + self._cycle_config.recover_timeout_s,
                    from_state=BotState.RECOVER,
                    target_state=BotState.WAIT_NEXT_CYCLE,
                    reason=f"{reason}_reset_complete",
                )
            ]

        return [
            RecoveryStep(
                cycle_id=cycle_id,
                at_ts=now_ts,
                from_state=current_state,
                target_state=BotState.RECOVER,
                reason=reason,
            ),
            RecoveryStep(
                cycle_id=cycle_id,
                at_ts=now_ts + self._cycle_config.recover_timeout_s,
                from_state=BotState.RECOVER,
                target_state=BotState.WAIT_NEXT_CYCLE,
                reason=f"{reason}_reset_complete",
            ),
        ]
