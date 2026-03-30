from __future__ import annotations

import pytest

from botlab.config import load_default_config
from botlab.core.scheduler import CycleScheduler, SchedulerError
from botlab.types import BotState, Observation


def test_scheduler_requires_bootstrap_before_prediction() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    with pytest.raises(SchedulerError, match="bootstrap"):
        scheduler.next_prediction()


def test_scheduler_builds_expected_windows_for_next_cycle() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    prediction = scheduler.next_prediction()

    assert prediction.cycle_id == 1
    assert prediction.predicted_spawn_ts == 145.0
    assert prediction.prepare_window_start_ts == 140.0
    assert prediction.ready_window_start_ts == 144.0
    assert prediction.ready_window_end_ts == 146.0


def test_scheduler_keeps_ideal_prediction_after_multiple_ideal_observations() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    scheduler.register_observation(
        Observation(
            cycle_id=1,
            observed_at_ts=145.0,
            signal_detected=True,
            actual_spawn_ts=145.0,
            source="simulation",
            confidence=1.0,
            metadata={},
        )
    )
    scheduler.register_observation(
        Observation(
            cycle_id=2,
            observed_at_ts=190.0,
            signal_detected=True,
            actual_spawn_ts=190.0,
            source="simulation",
            confidence=1.0,
            metadata={},
        )
    )

    prediction = scheduler.next_prediction()

    assert prediction.cycle_id == 3
    assert prediction.predicted_spawn_ts == pytest.approx(235.0)
    assert prediction.prepare_window_start_ts == pytest.approx(230.0)
    assert prediction.ready_window_start_ts == pytest.approx(234.0)
    assert prediction.ready_window_end_ts == pytest.approx(236.0)


def test_scheduler_state_for_time_before_prepare_prepare_ready_and_after_ready() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    prediction = scheduler.next_prediction()

    assert scheduler.state_for_time(139.0, prediction) is BotState.WAIT_NEXT_CYCLE
    assert scheduler.state_for_time(140.0, prediction) is BotState.PREPARE_WINDOW
    assert scheduler.state_for_time(143.999, prediction) is BotState.PREPARE_WINDOW
    assert scheduler.state_for_time(144.0, prediction) is BotState.READY_WINDOW
    assert scheduler.state_for_time(145.0, prediction) is BotState.READY_WINDOW
    assert scheduler.state_for_time(146.0, prediction) is BotState.READY_WINDOW
    assert scheduler.state_for_time(146.001, prediction) is BotState.WAIT_NEXT_CYCLE


def test_scheduler_helper_methods_match_cycle_windows() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    prediction = scheduler.next_prediction()

    assert scheduler.is_before_prepare_window(139.9, prediction) is True
    assert scheduler.is_prepare_window(140.0, prediction) is True
    assert scheduler.is_prepare_window(143.5, prediction) is True
    assert scheduler.is_ready_window(144.0, prediction) is True
    assert scheduler.is_ready_window(145.5, prediction) is True
    assert scheduler.has_ready_window_passed(146.1, prediction) is True

    assert scheduler.seconds_until_prepare_window(135.0, prediction) == pytest.approx(5.0)
    assert scheduler.seconds_until_prepare_window(140.0, prediction) == pytest.approx(0.0)

    assert scheduler.seconds_until_ready_window(140.0, prediction) == pytest.approx(4.0)
    assert scheduler.seconds_until_ready_window(144.0, prediction) == pytest.approx(0.0)

    assert scheduler.seconds_until_spawn(140.0, prediction) == pytest.approx(5.0)
    assert scheduler.seconds_until_spawn(145.0, prediction) == pytest.approx(0.0)
    assert scheduler.seconds_until_spawn(145.5, prediction) == pytest.approx(0.0)


def test_scheduler_register_observation_updates_future_prediction() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    observation = Observation(
        cycle_id=1,
        observed_at_ts=145.2,
        signal_detected=True,
        actual_spawn_ts=145.2,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    updated = scheduler.register_observation(observation)
    next_prediction = scheduler.next_prediction()

    assert updated is True
    assert next_prediction.cycle_id == 2
    assert next_prediction.predicted_spawn_ts == pytest.approx(190.4)
    assert next_prediction.prepare_window_start_ts == pytest.approx(185.4)
    assert next_prediction.ready_window_start_ts == pytest.approx(189.4)
    assert next_prediction.ready_window_end_ts == pytest.approx(191.4)


def test_scheduler_treats_late_time_as_window_already_missed() -> None:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)

    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    prediction = scheduler.next_prediction()

    now_ts = 146.5

    assert scheduler.has_ready_window_passed(now_ts, prediction) is True
    assert scheduler.state_for_time(now_ts, prediction) is BotState.WAIT_NEXT_CYCLE
