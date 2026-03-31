from __future__ import annotations

from botlab.config import load_default_config
from botlab.domain.decision_engine import DecisionContext, DecisionEngine
from botlab.domain.scheduler import CycleScheduler
from botlab.types import BotState, CombatSnapshot, Observation


def _build_engine_and_prediction() -> tuple[DecisionEngine, CycleScheduler]:
    settings = load_default_config()
    engine = DecisionEngine(settings.cycle, settings.combat)
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)
    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    return engine, scheduler


def test_decision_engine_moves_from_idle_to_wait_next_cycle() -> None:
    engine, scheduler = _build_engine_and_prediction()
    prediction = scheduler.next_prediction()
    temporal_state = scheduler.state_for_time(130.0, prediction)

    context = DecisionContext(
        now_ts=130.0,
        current_state=BotState.IDLE,
        state_entered_ts=0.0,
        prediction=prediction,
        temporal_state=temporal_state,
    )

    decision = engine.decide(context)

    assert decision.state is BotState.IDLE
    assert decision.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision.reason == "waiting_for_next_cycle"


def test_decision_engine_moves_to_prepare_and_ready_based_on_time_windows() -> None:
    engine, scheduler = _build_engine_and_prediction()
    prediction = scheduler.next_prediction()

    prepare_context = DecisionContext(
        now_ts=141.0,
        current_state=BotState.WAIT_NEXT_CYCLE,
        state_entered_ts=130.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(141.0, prediction),
    )

    prepare_decision = engine.decide(prepare_context)

    assert prepare_decision.next_state is BotState.PREPARE_WINDOW
    assert prepare_decision.reason == "prepare_window_opened"

    ready_context = DecisionContext(
        now_ts=144.2,
        current_state=BotState.PREPARE_WINDOW,
        state_entered_ts=141.0,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(144.2, prediction),
    )

    ready_decision = engine.decide(ready_context)

    assert ready_decision.next_state is BotState.READY_WINDOW
    assert ready_decision.reason == "ready_window_opened"


def test_decision_engine_enters_attempt_when_signal_detected_in_ready_window() -> None:
    engine, scheduler = _build_engine_and_prediction()
    prediction = scheduler.next_prediction()

    observation = Observation(
        cycle_id=1,
        observed_at_ts=145.0,
        signal_detected=True,
        actual_spawn_ts=145.0,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    context = DecisionContext(
        now_ts=145.0,
        current_state=BotState.READY_WINDOW,
        state_entered_ts=144.0,
        cycle_id=1,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(145.0, prediction),
        observation=observation,
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.ATTEMPT
    assert decision.action == "attempt_reaction"
    assert decision.reason == "signal_detected_in_ready_window"


def test_decision_engine_returns_to_wait_when_ready_window_is_missed() -> None:
    engine, scheduler = _build_engine_and_prediction()
    prediction = scheduler.next_prediction()

    context = DecisionContext(
        now_ts=146.2,
        current_state=BotState.READY_WINDOW,
        state_entered_ts=144.0,
        cycle_id=1,
        prediction=prediction,
        temporal_state=scheduler.state_for_time(146.2, prediction),
        observation=None,
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision.reason == "no_signal_before_ready_window_timeout"


def test_decision_engine_moves_from_attempt_to_verify() -> None:
    engine, _ = _build_engine_and_prediction()

    context = DecisionContext(
        now_ts=145.05,
        current_state=BotState.ATTEMPT,
        state_entered_ts=145.0,
        cycle_id=1,
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.VERIFY
    assert decision.reason == "attempt_dispatched_waiting_for_verification"


def test_decision_engine_verify_success_goes_to_combat() -> None:
    engine, _ = _build_engine_and_prediction()

    context = DecisionContext(
        now_ts=145.20,
        current_state=BotState.VERIFY,
        state_entered_ts=145.10,
        cycle_id=1,
        verify_result="success",
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.COMBAT
    assert decision.reason == "verification_success"


def test_decision_engine_verify_failure_goes_to_wait_next_cycle() -> None:
    engine, _ = _build_engine_and_prediction()

    context = DecisionContext(
        now_ts=145.20,
        current_state=BotState.VERIFY,
        state_entered_ts=145.10,
        cycle_id=1,
        verify_result="failure",
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision.reason == "verification_failure"


def test_decision_engine_verify_timeout_goes_to_recover() -> None:
    engine, _ = _build_engine_and_prediction()

    context = DecisionContext(
        now_ts=146.0,
        current_state=BotState.VERIFY,
        state_entered_ts=145.0,
        cycle_id=1,
        verify_result=None,
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.RECOVER
    assert decision.reason == "verify_timeout"


def test_decision_engine_recover_timeout_goes_to_wait_next_cycle() -> None:
    engine, _ = _build_engine_and_prediction()

    context = DecisionContext(
        now_ts=150.0,
        current_state=BotState.RECOVER,
        state_entered_ts=147.5,
        cycle_id=1,
    )

    decision = engine.decide(context)

    assert decision.next_state is BotState.WAIT_NEXT_CYCLE
    assert decision.reason == "recover_timeout_elapsed"


def test_decision_engine_combat_to_rest_and_rest_to_wait_next_cycle() -> None:
    engine, _ = _build_engine_and_prediction()

    combat_finished_low_hp = CombatSnapshot(
        hp_ratio=0.40,
        turn_index=5,
        enemy_count=0,
        strategy="default",
        in_combat=False,
        combat_started_ts=145.3,
        combat_finished_ts=150.0,
        metadata={},
    )

    combat_context = DecisionContext(
        now_ts=150.0,
        current_state=BotState.COMBAT,
        state_entered_ts=145.3,
        cycle_id=1,
        combat_snapshot=combat_finished_low_hp,
    )

    combat_decision = engine.decide(combat_context)

    assert combat_decision.next_state is BotState.REST
    assert combat_decision.reason == "combat_finished_low_hp"

    restored_hp_snapshot = CombatSnapshot(
        hp_ratio=0.95,
        turn_index=5,
        enemy_count=0,
        strategy="default",
        in_combat=False,
        combat_started_ts=145.3,
        combat_finished_ts=150.0,
        metadata={},
    )

    rest_context = DecisionContext(
        now_ts=154.0,
        current_state=BotState.REST,
        state_entered_ts=150.0,
        cycle_id=1,
        combat_snapshot=restored_hp_snapshot,
    )

    rest_decision = engine.decide(rest_context)

    assert rest_decision.next_state is BotState.WAIT_NEXT_CYCLE
    assert rest_decision.reason == "rest_completed_hp_restored"
