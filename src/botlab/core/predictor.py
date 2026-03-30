from __future__ import annotations

from collections import deque
from typing import Deque

from botlab.config import CycleConfig
from botlab.types import CyclePrediction, Observation


class PredictorError(RuntimeError):
    """Błąd predictora cyklu."""


class SpawnPredictor:
    """
    Minimalny predictor kolejnych cykli.

    Założenia:
    - działa na bazowym interwale, np. 45 s,
    - po każdej poprawnej obserwacji aktualizuje anchor czasu,
    - efektywny interwał wyznacza jako średnią z ostatnich obserwowanych odstępów,
    - nie zawiera jeszcze zaawansowanego modelu ani heurystyk odpornościowych.

    Predictor nie zarządza stanami FSM. Jego jedyną rolą jest przewidywanie
    expected_spawn_ts i zbudowanie okien czasowych wokół tego punktu.
    """

    def __init__(
        self,
        *,
        interval_s: float,
        prepare_before_s: float,
        ready_before_s: float,
        ready_after_s: float,
        max_history: int = 20,
    ) -> None:
        if interval_s <= 0.0:
            raise PredictorError("interval_s musi być większe od 0.")
        if prepare_before_s < 0.0:
            raise PredictorError("prepare_before_s nie może być ujemne.")
        if ready_before_s < 0.0:
            raise PredictorError("ready_before_s nie może być ujemne.")
        if ready_after_s < 0.0:
            raise PredictorError("ready_after_s nie może być ujemne.")
        if max_history <= 0:
            raise PredictorError("max_history musi być większe od 0.")
        if ready_before_s > prepare_before_s:
            raise PredictorError("ready_before_s nie może być większe niż prepare_before_s.")

        self._base_interval_s = interval_s
        self._prepare_before_s = prepare_before_s
        self._ready_before_s = ready_before_s
        self._ready_after_s = ready_after_s
        self._max_history = max_history

        self._anchor_cycle_id: int | None = None
        self._anchor_spawn_ts: float | None = None

        self._observed_spawns: Deque[tuple[int, float]] = deque(maxlen=max_history)
        self._interval_history: Deque[float] = deque(maxlen=max_history)
        self._drift_history: Deque[float] = deque(maxlen=max_history)

    @classmethod
    def from_cycle_config(
        cls,
        cycle_config: CycleConfig,
        max_history: int = 20,
    ) -> "SpawnPredictor":
        return cls(
            interval_s=cycle_config.interval_s,
            prepare_before_s=cycle_config.prepare_before_s,
            ready_before_s=cycle_config.ready_before_s,
            ready_after_s=cycle_config.ready_after_s,
            max_history=max_history,
        )

    def bootstrap(self, *, anchor_spawn_ts: float, anchor_cycle_id: int = 0) -> None:
        """
        Inicjalizuje predictor znanym punktem odniesienia.

        Ten punkt traktujemy jako pierwszy znany spawn.
        Dzięki temu kolejna obserwacja może już wyznaczyć pierwszy realny interwał.
        """
        self._anchor_cycle_id = anchor_cycle_id
        self._anchor_spawn_ts = anchor_spawn_ts

        self._observed_spawns.clear()
        self._interval_history.clear()
        self._drift_history.clear()

        self._observed_spawns.append((anchor_cycle_id, anchor_spawn_ts))

    def is_bootstrapped(self) -> bool:
        return self._anchor_cycle_id is not None and self._anchor_spawn_ts is not None

    @property
    def anchor_cycle_id(self) -> int:
        self._require_bootstrapped()
        assert self._anchor_cycle_id is not None
        return self._anchor_cycle_id

    @property
    def anchor_spawn_ts(self) -> float:
        self._require_bootstrapped()
        assert self._anchor_spawn_ts is not None
        return self._anchor_spawn_ts

    @property
    def observation_count(self) -> int:
        return len(self._observed_spawns)

    @property
    def interval_history_count(self) -> int:
        return len(self._interval_history)

    def current_effective_interval_s(self) -> float:
        if not self._interval_history:
            return self._base_interval_s
        return sum(self._interval_history) / len(self._interval_history)

    def average_drift_s(self) -> float:
        if not self._drift_history:
            return 0.0
        return sum(self._drift_history) / len(self._drift_history)

    def record_observation(self, observation: Observation) -> bool:
        """
        Rejestruje obserwację cyklu.

        Zwraca:
        - True: obserwacja została użyta do aktualizacji predictora,
        - False: obserwacja nie zawierała actual_spawn_ts i nie mogła poprawić modelu.

        Zasady:
        - ignorujemy obserwacje bez actual_spawn_ts,
        - wymagamy rosnących cycle_id,
        - drift liczymy względem oczekiwania sprzed aktualizacji anchor.
        """
        if observation.actual_spawn_ts is None:
            return False

        if not self.is_bootstrapped():
            self.bootstrap(
                anchor_spawn_ts=observation.actual_spawn_ts,
                anchor_cycle_id=observation.cycle_id,
            )
            return True

        if observation.cycle_id <= self.anchor_cycle_id:
            raise PredictorError(
                "Observation.cycle_id musi być większe od aktualnego anchor_cycle_id."
            )

        cycle_delta_from_anchor = observation.cycle_id - self.anchor_cycle_id
        expected_before_update = self.anchor_spawn_ts + (self._base_interval_s * cycle_delta_from_anchor)
        drift_s = observation.actual_spawn_ts - expected_before_update
        self._drift_history.append(drift_s)

        previous_cycle_id, previous_spawn_ts = self._observed_spawns[-1]
        cycle_delta = observation.cycle_id - previous_cycle_id

        if cycle_delta <= 0:
            raise PredictorError("Obserwacje muszą być rejestrowane w rosnącej kolejności cycle_id.")

        observed_interval_s = (observation.actual_spawn_ts - previous_spawn_ts) / cycle_delta
        self._interval_history.append(observed_interval_s)

        self._observed_spawns.append((observation.cycle_id, observation.actual_spawn_ts))
        self._anchor_cycle_id = observation.cycle_id
        self._anchor_spawn_ts = observation.actual_spawn_ts

        return True

    def predict_next(self) -> CyclePrediction:
        self._require_bootstrapped()
        return self.predict_for_cycle(self.anchor_cycle_id + 1)

    def predict_for_cycle(self, cycle_id: int) -> CyclePrediction:
        self._require_bootstrapped()

        cycle_offset = cycle_id - self.anchor_cycle_id
        effective_interval_s = self.current_effective_interval_s()
        predicted_spawn_ts = self.anchor_spawn_ts + (effective_interval_s * cycle_offset)

        return self._build_prediction(
            cycle_id=cycle_id,
            predicted_spawn_ts=predicted_spawn_ts,
            interval_s=effective_interval_s,
        )

    def _build_prediction(
        self,
        *,
        cycle_id: int,
        predicted_spawn_ts: float,
        interval_s: float,
    ) -> CyclePrediction:
        prepare_window_start_ts = predicted_spawn_ts - self._prepare_before_s
        ready_window_start_ts = predicted_spawn_ts - self._ready_before_s
        ready_window_end_ts = predicted_spawn_ts + self._ready_after_s

        return CyclePrediction(
            cycle_id=cycle_id,
            predicted_spawn_ts=predicted_spawn_ts,
            interval_s=interval_s,
            prepare_window_start_ts=prepare_window_start_ts,
            ready_window_start_ts=ready_window_start_ts,
            ready_window_end_ts=ready_window_end_ts,
            based_on_observation_count=self.observation_count,
        )

    def _require_bootstrapped(self) -> None:
        if not self.is_bootstrapped():
            raise PredictorError("Predictor nie został jeszcze zainicjalizowany przez bootstrap().")
