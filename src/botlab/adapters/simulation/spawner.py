from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from botlab.types import CyclePrediction, Observation


VerifyOutcome = Literal["success", "failure", "timeout"]


@dataclass(slots=True, frozen=True)
class SimulatedGroupState:
    group_id: str
    position_xy: tuple[float, float]
    alive_count: int = 3
    engaged_by_other: bool = False
    reachable: bool = True
    threat_score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CycleScenario:
    """
    Plan jednego cyklu w symulacji.

    Pola:
    - has_event:
        Czy w ogole istnieje realne zdarzenie w tym cyklu.
    - drift_s:
        Przesuniecie rzeczywistego spawnu wzgledem predicted_spawn_ts.
    - verify_result:
        Wynik etapu VERIFY, jesli zdarzenie zostalo zauwazone i wykonano ATTEMPT.
    - combat_turns:
        Liczba tur walki po udanym VERIFY.
    - combat_turn_duration_s:
        Czas jednej tury walki. Jesli None, uzyta zostanie wartosc domyslna battle engine.
    - combat_final_hp_ratio:
        HP po zakonczeniu walki. Decyduje, czy system przejdzie do REST.
    - combat_strategy:
        Nazwa strategii wpisywana do CombatSnapshot.
    - combat_profile_name:
        Nazwa profilu walki mapowanego na named combat plan.
    - combat_plan_name:
        Nazwa gotowego planu walki do rozwiazania przez katalog planow.
    - combat_plan_rounds:
        Jawny plan "per runda", np. (("1", "space"), ("2",)).
    - combat_inputs:
        Jawna sekwencja wejsc walki, fallback gdy nie wybrano combat_plan_name.
    - force_battle_error:
        Wymusza wyjatek w warstwie battle.
    - force_rest_error:
        Wymusza wyjatek w warstwie rest.
    - spawn_zone_visible:
        Czy bot jest ustawiony tak, ze widzi strefe spawnu.
    - bot_position_xy:
        Pozycja bota uzywana do obliczen dystansu widocznych grupek.
    - approach_revalidation_delay_s:
        Offset czasowy snapshotu rewalidacji celu podczas dojscia.
    - approach_bot_position_xy:
        Opcjonalna pozycja bota dla snapshotu w fazie dojscia.
    - interaction_revalidation_delay_s:
        Offset czasowy snapshotu koncowej walidacji celu przed interakcja.
    - interaction_bot_position_xy:
        Opcjonalna pozycja bota dla snapshotu tuz przed interakcja.
    - current_target_id:
        Target ustawiony na starcie cyklu, jesli juz istnial.
    - approach_groups:
        Opcjonalny widok grupek do rewalidacji celu podczas dojscia.
    - interaction_groups:
        Opcjonalny widok grupek do koncowej walidacji celu przed interakcja.
    - groups:
        Widok grup PvE dla modelu swiata.
    - note:
        Dodatkowy opis pomocniczy do testow i telemetry.
    """

    has_event: bool = True
    drift_s: float = 0.0
    verify_result: VerifyOutcome = "success"
    combat_turns: int = 3
    combat_turn_duration_s: float | None = None
    combat_final_hp_ratio: float = 0.80
    combat_strategy: str = "default"
    combat_profile_name: str | None = None
    combat_plan_name: str | None = None
    combat_plan_rounds: tuple[tuple[str, ...], ...] | None = None
    combat_inputs: tuple[str, ...] = ("1", "space")
    force_battle_error: bool = False
    force_rest_error: bool = False
    spawn_zone_visible: bool = True
    bot_position_xy: tuple[float, float] = (0.0, 0.0)
    approach_revalidation_delay_s: float = 0.250
    approach_bot_position_xy: tuple[float, float] | None = None
    interaction_revalidation_delay_s: float = 0.450
    interaction_bot_position_xy: tuple[float, float] | None = None
    current_target_id: str | None = None
    approach_groups: tuple[SimulatedGroupState, ...] | None = None
    interaction_groups: tuple[SimulatedGroupState, ...] | None = None
    groups: tuple[SimulatedGroupState, ...] = field(default_factory=tuple)
    note: str = ""


@dataclass(slots=True, frozen=True)
class SpawnEvent:
    """
    Zmaterializowany wynik planu cyklu wzgledem konkretnej predykcji.

    Pola:
    - actual_spawn_ts:
        Rzeczywisty moment pojawienia sie zdarzenia, jesli istnieje.
    - observable_in_ready_window:
        Czy zdarzenie miesci sie w ready window i moze zostac zauwazone przez system.
    - observation:
        Obserwacja dostarczona do FSM, jesli zdarzenie bylo zauwazalne.
    - verify_result:
        Wynik VERIFY dla zauwazalnego zdarzenia.
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
    Minimalny generator planow cykli dla symulacji.

    Zasady:
    - jesli has_event=False, w cyklu nie ma realnego spawnu,
    - jesli actual_spawn_ts wypada poza ready window, FSM nie dostaje obserwacji,
    - jesli actual_spawn_ts miesci sie w ready window, runner dostaje Observation
      i moze przejsc do ATTEMPT / VERIFY.
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
        observable_in_ready_window = scenario.spawn_zone_visible and (
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
