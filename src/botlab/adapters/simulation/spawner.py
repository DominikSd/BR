from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from botlab.types import CyclePrediction, Observation


VerifyOutcome = Literal["success", "failure", "timeout"]


@dataclass(slots=True, frozen=True)
class CycleScenario:
    """
    Plan jednego cyklu w symulacji.

    Pola:
    - has_event:
        Czy w ogóle istnieje realne zdarzenie w tym cyklu.
    - drift_s:
        Przesunięcie rzeczywistego spawnu względem predicted_spawn_ts.
    - verify_result:
        Wynik etapu VERIFY, jeśli zdarzenie zostało zauważone i wykonano ATTEMPT.
    - combat_turns:
        Liczba tur walki po udanym VERIFY.
    - combat_turn_duration_s:
        Czas jednej tury walki. Jeśli None, użyta zostanie wartość domyślna battle engine.
    - combat_final_hp_ratio:
        HP po zakończeniu walki. Decyduje, czy system przejdzie do REST.
    - combat_strategy:
        Nazwa strategii wpisywana do CombatSnapshot.
    - force_battle_error:
        Wymusza wyjątek w warstwie battle.
    - force_rest_error:
        Wymusza wyjątek w warstwie rest.
    - note:
        Dodatkowy opis pomocniczy do testów i telemetry.
    """

    has_event: bool = True
    drift_s: float = 0.0
    verify_result: VerifyOutcome = "success"
    combat_turns: int = 3
    combat_turn_duration_s: float | None = None
    combat_final_hp_ratio: float = 0.80
    combat_strategy: str = "default"
    force_battle_error: bool = False
    force_rest_error: bool = False
    note: str = ""


@dataclass(slots=True, frozen=True)
class SpawnEvent:
    """
    Zmaterializowany wynik planu cyklu względem konkretnej predykcji.

    Pola:
    - actual_spawn_ts:
        Rzeczywisty moment pojawienia się zdarzenia, jeśli istnieje.
    - observable_in_ready_window:
        Czy zdarzenie mieści się w ready window i może zostać zauważone przez system.
    - observation:
        Obserwacja dostarczona do FSM, jeśli zdarzenie było zauważalne.
    - verify_result:
        Wynik VERIFY dla zauważalnego zdarzenia.
    """

    cycle_id: int
    prediction: CyclePrediction
    scenario: CycleScenario
    actual_spawn_ts: float | None
    observable_in_ready_window: bool
    observation: Observation | None
    verify_result: VerifyOutcome | None


class SimulatedSpawner:
    """
    Minimalny generator planów cykli dla symulacji.

    Zasady:
    - jeśli has_event=False, w cyklu nie ma realnego spawnu,
    - jeśli actual_spawn_ts wypada poza ready window, FSM nie dostaje obserwacji,
    - jeśli actual_spawn_ts mieści się w ready window, runner dostaje Observation
      i może przejść do ATTEMPT / VERIFY.
    """

    def __init__(
        self,
        *,
        default_scenario: CycleScenario | None = None,
        overrides: Mapping[int, CycleScenario] | None = None,
        observation_source: str = "simulation",
    ) -> None:
        self._default_scenario = default_scenario or CycleScenario()
        self._overrides = dict(overrides or {})
        self._observation_source = observation_source

    def scenario_for_cycle(self, cycle_id: int) -> CycleScenario:
        return self._overrides.get(cycle_id, self._default_scenario)

    def build_spawn_event(self, prediction: CyclePrediction) -> SpawnEvent:
        scenario = self.scenario_for_cycle(prediction.cycle_id)

        if not scenario.has_event:
            return SpawnEvent(
                cycle_id=prediction.cycle_id,
                prediction=prediction,
                scenario=scenario,
                actual_spawn_ts=None,
                observable_in_ready_window=False,
                observation=None,
                verify_result=None,
            )

        actual_spawn_ts = prediction.predicted_spawn_ts + scenario.drift_s
        observable_in_ready_window = (
            prediction.ready_window_start_ts <= actual_spawn_ts <= prediction.ready_window_end_ts
        )

        if observable_in_ready_window:
            observation = Observation(
                cycle_id=prediction.cycle_id,
                observed_at_ts=actual_spawn_ts,
                signal_detected=True,
                actual_spawn_ts=actual_spawn_ts,
                source=self._observation_source,
                confidence=1.0,
                metadata={
                    "note": scenario.note,
                    "drift_s": scenario.drift_s,
                },
            )
            verify_result: VerifyOutcome | None = scenario.verify_result
        else:
            observation = None
            verify_result = None

        return SpawnEvent(
            cycle_id=prediction.cycle_id,
            prediction=prediction,
            scenario=scenario,
            actual_spawn_ts=actual_spawn_ts,
            observable_in_ready_window=observable_in_ready_window,
            observation=observation,
            verify_result=verify_result,
        )
