from __future__ import annotations

from botlab.config import load_default_config
from botlab.domain.decision_engine import DecisionEngine
from botlab.domain.fsm import CycleFSM
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, CombatSnapshot, Observation


def _build_stack() -> tuple[CycleFSM, CycleScheduler]:
    settings = load_default_config()

    engine = DecisionEngine(settings.cycle, settings.combat)
    fsm = CycleFSM(
        decision_engine=engine,
        initial_state=BotState.IDLE,
        started_at_ts=0.0,
    )

    scheduler = CycleScheduler.from_cycle_config(settings.cycle)
    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    return fsm, scheduler


def test_fsm_progresses_from_idle_to_prepare_to_ready() -> None:
    fsm, scheduler = _build_stack()
    prediction = scheduler.next_prediction()

    decision_1 = fsm.tick(
        now_ts=130.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(130.0, prediction),
    )
    assert decision_1.next_state is BotState.WAIT_NEXT_CYCLE
    assert fsm.current_state is BotState.WAIT_NEXT_CYCLE

    decision_2 = fsm.tick(
        now_ts=141.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(141.0, prediction),
    )
    assert decision_2.next_state is BotState.PREPARE_WINDOW
    assert fsm.current_state is BotState.PREPARE_WINDOW

    decision_3 = fsm.tick(
        now_ts=144.2,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(144.2, prediction),
    )
    assert decision_3.next_state is BotState.READY_WINDOW
    assert fsm.current_state is BotState.READY_WINDOW

    history = fsm.transition_history()
    assert len(history) == 3
    assert history[0].from_state is BotState.IDLE
    assert history[0].to_state is BotState.WAIT_NEXT_CYCLE
    assert history[1].to_state is BotState.PREPARE_WINDOW
    assert history[2].to_state is BotState.READY_WINDOW


def test_fsm_ready_attempt_verify_combat_path() -> None:
    fsm, scheduler = _build_stack()
    prediction = scheduler.next_prediction()

    fsm.tick(
        now_ts=130.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(130.0, prediction),
    )
    fsm.tick(
        now_ts=141.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(141.0, prediction),
    )
    fsm.tick(
        now_ts=144.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(144.0, prediction),
    )

    observation = Observation(
        cycle_id=1,
        observed_at_ts=145.0,
        signal_detected=True,
        actual_spawn_ts=145.0,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    decision_attempt = fsm.tick(
        now_ts=145.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(145.0, prediction),
        observation=observation,
    )
    assert decision_attempt.next_state is BotState.ATTEMPT
    assert fsm.current_state is BotState.ATTEMPT

    decision_verify = fsm.tick(
        now_ts=145.05,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(145.05, prediction),
    )
    assert decision_verify.next_state is BotState.VERIFY
    assert fsm.current_state is BotState.VERIFY

    decision_combat = fsm.tick(
        now_ts=145.20,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(145.20, prediction),
        verify_result="success",
    )
    assert decision_combat.next_state is BotState.COMBAT
    assert fsm.current_state is BotState.COMBAT

    history = fsm.transition_history()
    transition_states = [(item.from_state, item.to_state) for item in history]

    assert (BotState.READY_WINDOW, BotState.ATTEMPT) in transition_states
    assert (BotState.ATTEMPT, BotState.VERIFY) in transition_states
    assert (BotState.VERIFY, BotState.COMBAT) in transition_states


def test_fsm_returns_to_wait_next_cycle_when_no_signal_arrives() -> None:
    fsm, scheduler = _build_stack()
    prediction = scheduler.next_prediction()

    fsm.tick(
        now_ts=141.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(141.0, prediction),
    )
    fsm.tick(
        now_ts=144.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(144.0, prediction),
    )

    decision = fsm.tick(
        now_ts=146.2,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(146.2, prediction),
        observation=None,
    )

    assert decision.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision.reason == "no_signal_before_ready_window_timeout"
    assert fsm.current_state is BotState.WAIT_NEXT_CYCLE


def test_fsm_verify_timeout_then_recover_then_wait_next_cycle() -> None:
    fsm, scheduler = _build_stack()
    prediction = scheduler.next_prediction()

    fsm.force_state(
        new_state=BotState.VERIFY,
        now_ts=145.0,
        reason="test_setup",
        cycle_id=1,
    )

    decision_recover = fsm.tick(
        now_ts=145.7,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(145.7, prediction),
    )

    assert decision_recover.next_state is BotState.RECOVER
    assert decision_recover.reason == "verify_timeout"
    assert fsm.current_state is BotState.RECOVER

    decision_wait = fsm.tick(
        now_ts=147.8,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(147.8, prediction),
    )

    assert decision_wait.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision_wait.reason == "recover_timeout_elapsed"
    assert fsm.current_state is BotState.WAIT_NEXT_CYCLE


def test_fsm_combat_to_rest_to_wait_next_cycle() -> None:
    fsm, scheduler = _build_stack()
    prediction = scheduler.next_prediction()

    fsm.force_state(
        new_state=BotState.COMBAT,
        now_ts=145.3,
        reason="test_setup",
        cycle_id=1,
    )

    low_hp_snapshot = CombatSnapshot(
        hp_ratio=0.40,
        turn_index=4,
        enemy_count=0,
        strategy="default",
        in_combat=False,
        combat_started_ts=145.3,
        combat_finished_ts=150.0,
        metadata={},
    )

    decision_rest = fsm.tick(
        now_ts=150.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(150.0, prediction),
        combat_snapshot=low_hp_snapshot,
    )

    assert decision_rest.next_state is BotState.REST
    assert decision_rest.reason == "combat_finished_low_resources"
    assert fsm.current_state is BotState.REST

    restored_hp_snapshot = CombatSnapshot(
        hp_ratio=0.95,
        turn_index=4,
        enemy_count=0,
        strategy="default",
        in_combat=False,
        combat_started_ts=145.3,
        combat_finished_ts=150.0,
        metadata={},
    )

    decision_wait = fsm.tick(
        now_ts=154.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(154.0, prediction),
        combat_snapshot=restored_hp_snapshot,
    )

    assert decision_wait.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision_wait.reason == "rest_completed_resources_restored"
    assert fsm.current_state is BotState.WAIT_NEXT_CYCLE
