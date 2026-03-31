from __future__ import annotations

from dataclasses import dataclass

from botlab.config import CycleConfig
from botlab.adapters.simulation.spawner import SpawnEvent


@dataclass(slots=True, frozen=True)
class CycleTrace:
    """
    Kompletny harmonogram jednego cyklu w symulowanym świecie.

    Pole ready_window_close_ts jest zawsze ustawione.
    Pozostałe pola zależą od tego, czy wystąpiła obserwacja i czy doszło do ATTEMPT/VERIFY.
    """

    cycle_id: int
    prepare_ts: float
    ready_ts: float
    ready_window_close_ts: float
    attempt_ts: float | None
    verify_start_ts: float | None
    verify_resolution_ts: float | None
    verify_timeout_ts: float | None
    recover_complete_ts: float | None
    post_cycle_reset_ts: float | None


class SimulatedWorld:
    """
    Minimalny model czasu dla symulacji cyklu.

    Odpowiada za wyliczenie chwil:
    - wejścia do PREPARE_WINDOW,
    - wejścia do READY_WINDOW,
    - ATTEMPT,
    - VERIFY start,
    - VERIFY finish albo VERIFY timeout,
    - RECOVER finish,
    - technicznego resetu po wejściu do COMBAT.

    Na tym etapie nie modelujemy jeszcze faktycznej walki.
    """

    def __init__(
        self,
        cycle_config: CycleConfig,
        *,
        attempt_latency_s: float = 0.020,
        verify_latency_s: float = 0.100,
        epsilon_s: float = 0.010,
    ) -> None:
        if attempt_latency_s <= 0.0:
            raise ValueError("attempt_latency_s musi być większe od 0.")
        if verify_latency_s <= 0.0:
            raise ValueError("verify_latency_s musi być większe od 0.")
        if epsilon_s <= 0.0:
            raise ValueError("epsilon_s musi być większe od 0.")

        self._cycle_config = cycle_config
        self._attempt_latency_s = attempt_latency_s
        self._verify_latency_s = verify_latency_s
        self._epsilon_s = epsilon_s

    def build_cycle_trace(self, spawn_event: SpawnEvent) -> CycleTrace:
        prediction = spawn_event.prediction

        prepare_ts = prediction.prepare_window_start_ts
        ready_ts = prediction.ready_window_start_ts
        ready_window_close_ts = prediction.ready_window_end_ts + self._epsilon_s

        if spawn_event.observation is None:
            return CycleTrace(
                cycle_id=prediction.cycle_id,
                prepare_ts=prepare_ts,
                ready_ts=ready_ts,
                ready_window_close_ts=ready_window_close_ts,
                attempt_ts=None,
                verify_start_ts=None,
                verify_resolution_ts=None,
                verify_timeout_ts=None,
                recover_complete_ts=None,
                post_cycle_reset_ts=None,
            )

        attempt_ts = spawn_event.observation.observed_at_ts + self._attempt_latency_s
        verify_start_ts = attempt_ts + self._epsilon_s

        if spawn_event.verify_result == "timeout":
            verify_resolution_ts = None
            verify_timeout_ts = verify_start_ts + self._cycle_config.verify_timeout_s + self._epsilon_s
            recover_complete_ts = (
                verify_timeout_ts + self._cycle_config.recover_timeout_s + self._epsilon_s
            )
            post_cycle_reset_ts = None
        else:
            verify_resolution_ts = verify_start_ts + self._verify_latency_s
            verify_timeout_ts = None
            recover_complete_ts = None
            if spawn_event.verify_result == "success":
                post_cycle_reset_ts = verify_resolution_ts + self._epsilon_s
            else:
                post_cycle_reset_ts = None

        return CycleTrace(
            cycle_id=prediction.cycle_id,
            prepare_ts=prepare_ts,
            ready_ts=ready_ts,
            ready_window_close_ts=ready_window_close_ts,
            attempt_ts=attempt_ts,
            verify_start_ts=verify_start_ts,
            verify_resolution_ts=verify_resolution_ts,
            verify_timeout_ts=verify_timeout_ts,
            recover_complete_ts=recover_complete_ts,
            post_cycle_reset_ts=post_cycle_reset_ts,
        )
