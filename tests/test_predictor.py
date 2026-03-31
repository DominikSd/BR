from __future__ import annotations

import pytest

from botlab.config import load_default_config
from botlab.domain.predictor import PredictorError, SpawnPredictor
from botlab.types import Observation


def test_predictor_bootstrap_and_predict_next_cycle_from_default_config() -> None:
    settings = load_default_config()

    predictor = SpawnPredictor.from_cycle_config(settings.cycle)
    predictor.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    prediction = predictor.predict_next()

    assert prediction.cycle_id == 1
    assert prediction.predicted_spawn_ts == 145.0
    assert prediction.interval_s == 45.0
    assert prediction.prepare_window_start_ts == 140.0
    assert prediction.ready_window_start_ts == 144.0
    assert prediction.ready_window_end_ts == 146.0
    assert prediction.based_on_observation_count == 1


def test_predictor_updates_for_positive_drift_plus_point_two_seconds() -> None:
    settings = load_default_config()

    predictor = SpawnPredictor.from_cycle_config(settings.cycle)
    predictor.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    observation_cycle_1 = Observation(
        cycle_id=1,
        observed_at_ts=145.2,
        signal_detected=True,
        actual_spawn_ts=145.2,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    predictor.record_observation(observation_cycle_1)

    next_prediction = predictor.predict_next()

    assert predictor.current_effective_interval_s() == pytest.approx(45.2)
    assert predictor.average_drift_s() == pytest.approx(0.2)
    assert next_prediction.cycle_id == 2
    assert next_prediction.predicted_spawn_ts == pytest.approx(190.4)
    assert next_prediction.prepare_window_start_ts == pytest.approx(185.4)
    assert next_prediction.ready_window_start_ts == pytest.approx(189.4)
    assert next_prediction.ready_window_end_ts == pytest.approx(191.4)


def test_predictor_updates_for_negative_drift_minus_point_two_seconds() -> None:
    settings = load_default_config()

    predictor = SpawnPredictor.from_cycle_config(settings.cycle)
    predictor.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    observation_cycle_1 = Observation(
        cycle_id=1,
        observed_at_ts=144.8,
        signal_detected=True,
        actual_spawn_ts=144.8,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    predictor.record_observation(observation_cycle_1)

    next_prediction = predictor.predict_next()

    assert predictor.current_effective_interval_s() == pytest.approx(44.8)
    assert predictor.average_drift_s() == pytest.approx(-0.2)
    assert next_prediction.cycle_id == 2
    assert next_prediction.predicted_spawn_ts == pytest.approx(189.6)
    assert next_prediction.prepare_window_start_ts == pytest.approx(184.6)
    assert next_prediction.ready_window_start_ts == pytest.approx(188.6)
    assert next_prediction.ready_window_end_ts == pytest.approx(190.6)


def test_predictor_stabilizes_when_drift_alternates_plus_and_minus_point_two() -> None:
    settings = load_default_config()

    predictor = SpawnPredictor.from_cycle_config(settings.cycle)
    predictor.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    observation_cycle_1 = Observation(
        cycle_id=1,
        observed_at_ts=145.2,
        signal_detected=True,
        actual_spawn_ts=145.2,
        source="simulation",
        confidence=1.0,
        metadata={},
    )
    observation_cycle_2 = Observation(
        cycle_id=2,
        observed_at_ts=190.0,
        signal_detected=True,
        actual_spawn_ts=190.0,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    predictor.record_observation(observation_cycle_1)
    predictor.record_observation(observation_cycle_2)

    next_prediction = predictor.predict_next()

    assert predictor.current_effective_interval_s() == pytest.approx(45.0)
    assert predictor.average_drift_s() == pytest.approx(0.0)
    assert next_prediction.cycle_id == 3
    assert next_prediction.predicted_spawn_ts == pytest.approx(235.0)


def test_predictor_ignores_observation_without_actual_spawn_ts() -> None:
    settings = load_default_config()

    predictor = SpawnPredictor.from_cycle_config(settings.cycle)
    predictor.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)

    observation = Observation(
        cycle_id=1,
        observed_at_ts=145.0,
        signal_detected=False,
        actual_spawn_ts=None,
        source="simulation",
        confidence=0.0,
        metadata={},
    )

    updated = predictor.record_observation(observation)
    next_prediction = predictor.predict_next()

    assert updated is False
    assert predictor.current_effective_interval_s() == 45.0
    assert predictor.average_drift_s() == 0.0
    assert next_prediction.predicted_spawn_ts == 145.0


def test_predictor_requires_monotonic_cycle_ids() -> None:
    settings = load_default_config()

    predictor = SpawnPredictor.from_cycle_config(settings.cycle)
    predictor.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=1)

    observation = Observation(
        cycle_id=1,
        observed_at_ts=145.0,
        signal_detected=True,
        actual_spawn_ts=145.0,
        source="simulation",
        confidence=1.0,
        metadata={},
    )

    with pytest.raises(PredictorError, match="greater|większe|anchor_cycle_id"):
        predictor.record_observation(observation)
