from __future__ import annotations

from botlab.config import load_default_config
from botlab.domain.recovery import RecoveryManager
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState


def test_recovery_detects_prepare_window_stuck_past_ready_window() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)
    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    prediction = scheduler.next_prediction()

    recovery = RecoveryManager(settings.cycle)

    plan = recovery.detect_stuck_state(
        now_ts=146.5,
        current_state=BotState.PREPARE_WINDOW,
        state_entered_ts=140.0,
        cycle_id=1,
        prediction=prediction,
    )

    assert plan is not None
    assert len(plan) == 1
    assert plan[0].target_state is BotState.WAIT_NEXT_CYCLE
    assert plan[0].reason == "prepare_window_stuck_past_ready_window"


def test_recovery_builds_safe_reset_plan_for_verify_timeout() -> None:
    settings = load_default_config()
    recovery = RecoveryManager(settings.cycle)

    plan = recovery.detect_stuck_state(
        now_ts=145.8,
        current_state=BotState.VERIFY,
        state_entered_ts=145.0,
        cycle_id=1,
        prediction=None,
    )

    assert plan is not None
    assert len(plan) == 2
    assert plan[0].target_state is BotState.RECOVER
    assert plan[0].reason == "verify_stuck_timeout"
    assert plan[1].target_state is BotState.WAIT_NEXT_CYCLE
    assert plan[1].reason == "verify_stuck_timeout_reset_complete"


def test_recovery_detects_recover_timeout_and_forces_wait_next_cycle() -> None:
    settings = load_default_config()
    recovery = RecoveryManager(settings.cycle)

    plan = recovery.detect_stuck_state(
        now_ts=200.5,
        current_state=BotState.RECOVER,
        state_entered_ts=198.0,
        cycle_id=3,
        prediction=None,
    )

    assert plan is not None
    assert len(plan) == 1
    assert plan[0].target_state is BotState.WAIT_NEXT_CYCLE
    assert plan[0].reason == "recover_timeout_force_reset"


def test_recovery_builds_exception_recovery_plan() -> None:
    settings = load_default_config()
    recovery = RecoveryManager(settings.cycle)

    plan = recovery.build_exception_recovery_plan(
        now_ts=200.0,
        current_state=BotState.COMBAT,
        cycle_id=7,
        exception=RuntimeError("boom"),
    )

    assert len(plan) == 2
    assert plan[0].target_state is BotState.RECOVER
    assert plan[0].reason == "execution_error_RuntimeError"
    assert plan[1].target_state is BotState.WAIT_NEXT_CYCLE
    assert plan[1].reason == "execution_error_RuntimeError_reset_complete"


def test_recovery_returns_none_when_state_is_already_neutral() -> None:
    settings = load_default_config()
    recovery = RecoveryManager(settings.cycle)

    plan = recovery.ensure_neutral_state_plan(
        now_ts=100.0,
        current_state=BotState.WAIT_NEXT_CYCLE,
        cycle_id=1,
        reason="should_not_reset",
    )

    assert plan is None
