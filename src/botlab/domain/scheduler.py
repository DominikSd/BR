from __future__ import annotations

from botlab.config import CycleConfig
from botlab.domain.predictor import SpawnPredictor
from botlab.types import BotState, CyclePrediction, Observation


class SchedulerError(RuntimeError):
    """Błąd schedulera cyklu."""


class CycleScheduler:
    """
    Minimalny scheduler cyklu.

    Odpowiada za:
    - trzymanie predictora,
    - budowanie przewidywanego cyklu,
    - określenie, czy jesteśmy przed prepare window,
      w prepare window, w ready window albo już po nim,
    - udostępnienie prostych metod pomocniczych dla późniejszego FSM.

    Na tym etapie scheduler jeszcze nie wykonuje przejść stanów samodzielnie.
    Zwraca tylko wynik logiki czasowej.
    """

    def __init__(self, predictor: SpawnPredictor) -> None:
        self._predictor = predictor

    @classmethod
    def from_cycle_config(
        cls,
        cycle_config: CycleConfig,
        max_history: int = 20,
    ) -> "CycleScheduler":
        predictor = SpawnPredictor.from_cycle_config(
            cycle_config=cycle_config,
            max_history=max_history,
        )
        return cls(predictor=predictor)

    @property
    def predictor(self) -> SpawnPredictor:
        return self._predictor

    def bootstrap(self, *, anchor_spawn_ts: float, anchor_cycle_id: int = 0) -> None:
        self._predictor.bootstrap(
            anchor_spawn_ts=anchor_spawn_ts,
            anchor_cycle_id=anchor_cycle_id,
        )

    def register_observation(self, observation: Observation) -> bool:
        return self._predictor.record_observation(observation)

    def next_prediction(self) -> CyclePrediction:
        self._require_bootstrapped()
        return self._predictor.predict_next()

    def prediction_for_cycle(self, cycle_id: int) -> CyclePrediction:
        self._require_bootstrapped()
        return self._predictor.predict_for_cycle(cycle_id)

    def state_for_time(self, now_ts: float, prediction: CyclePrediction) -> BotState:
        """
        Zwraca minimalny stan czasowy z punktu widzenia schedulera.

        Reguły:
        - przed prepare window -> WAIT_NEXT_CYCLE
        - prepare window -> PREPARE_WINDOW
        - ready window -> READY_WINDOW
        - po ready window -> WAIT_NEXT_CYCLE

        Właściwe przejścia ATTEMPT / VERIFY / COMBAT / REST obsłuży później FSM.
        """
        if now_ts < prediction.prepare_window_start_ts:
            return BotState.WAIT_NEXT_CYCLE

        if prediction.prepare_window_start_ts <= now_ts < prediction.ready_window_start_ts:
            return BotState.PREPARE_WINDOW

        if prediction.ready_window_start_ts <= now_ts <= prediction.ready_window_end_ts:
            return BotState.READY_WINDOW

        return BotState.WAIT_NEXT_CYCLE

    def is_before_prepare_window(self, now_ts: float, prediction: CyclePrediction) -> bool:
        return now_ts < prediction.prepare_window_start_ts

    def is_prepare_window(self, now_ts: float, prediction: CyclePrediction) -> bool:
        return prediction.prepare_window_start_ts <= now_ts < prediction.ready_window_start_ts

    def is_ready_window(self, now_ts: float, prediction: CyclePrediction) -> bool:
        return prediction.ready_window_start_ts <= now_ts <= prediction.ready_window_end_ts

    def has_ready_window_passed(self, now_ts: float, prediction: CyclePrediction) -> bool:
        return now_ts > prediction.ready_window_end_ts

    def seconds_until_prepare_window(self, now_ts: float, prediction: CyclePrediction) -> float:
        return max(0.0, prediction.prepare_window_start_ts - now_ts)

    def seconds_until_ready_window(self, now_ts: float, prediction: CyclePrediction) -> float:
        return max(0.0, prediction.ready_window_start_ts - now_ts)

    def seconds_until_spawn(self, now_ts: float, prediction: CyclePrediction) -> float:
        return max(0.0, prediction.predicted_spawn_ts - now_ts)

    def _require_bootstrapped(self) -> None:
        if not self._predictor.is_bootstrapped():
            raise SchedulerError("Scheduler nie został jeszcze zainicjalizowany przez bootstrap().")
