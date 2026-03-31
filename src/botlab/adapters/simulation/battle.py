from __future__ import annotations

from botlab.adapters.simulation.combat_profiles import SimulatedCombatProfileCatalog
from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog
from botlab.adapters.simulation.spawner import CycleScenario
from botlab.application.dto import TimedCombatSnapshot
from botlab.application.ports import CombatPlanCatalog
from botlab.config import CombatConfig
from botlab.types import CombatSnapshot


class SimulatedBattle:
    """
    Minimalny, deterministyczny symulator walki.

    Zalozenia:
    - walka trwa ustalona liczbe tur,
    - kazda tura ma staly czas,
    - koncowe HP jest okreslone przez CycleScenario.combat_final_hp_ratio,
    - ostatni snapshot konczy walke i ma in_combat=False.
    """

    def __init__(
        self,
        *,
        default_turn_duration_s: float = 0.300,
        combat_plan_catalog: CombatPlanCatalog | None = None,
        combat_profile_catalog: SimulatedCombatProfileCatalog | None = None,
    ) -> None:
        if default_turn_duration_s <= 0.0:
            raise ValueError("default_turn_duration_s musi byc wieksze od 0.")

        self._default_turn_duration_s = default_turn_duration_s
        self._combat_plan_catalog = combat_plan_catalog or SimulatedCombatPlanCatalog()
        self._combat_profile_catalog = combat_profile_catalog or SimulatedCombatProfileCatalog(
            combat_plan_catalog=self._combat_plan_catalog
        )

    def build_timeline(
        self,
        *,
        cycle_id: int,
        combat_started_ts: float,
        scenario: CycleScenario,
    ) -> list[TimedCombatSnapshot]:
        if scenario.force_battle_error:
            raise RuntimeError("forced_battle_error")

        if scenario.combat_turns <= 0:
            raise ValueError("scenario.combat_turns musi byc wieksze od 0.")

        turn_duration_s = (
            self._default_turn_duration_s
            if scenario.combat_turn_duration_s is None
            else scenario.combat_turn_duration_s
        )
        if turn_duration_s <= 0.0:
            raise ValueError("combat_turn_duration_s musi byc wieksze od 0.")

        final_hp_ratio = min(max(scenario.combat_final_hp_ratio, 0.0), 1.0)
        total_turns = scenario.combat_turns
        if scenario.combat_profile_name is not None:
            combat_plan_selection = self._combat_profile_catalog.select_profile(
                scenario.combat_profile_name
            )
        else:
            combat_plan_selection = self._combat_plan_catalog.select_plan(
                plan_name=scenario.combat_plan_name,
                round_sequences=scenario.combat_plan_rounds,
                input_sequence=scenario.combat_inputs,
            )
        combat_plan = combat_plan_selection.plan
        combat_inputs = combat_plan.to_input_sequence()

        snapshots: list[TimedCombatSnapshot] = []

        if total_turns == 1:
            event_ts = combat_started_ts + turn_duration_s
            final_snapshot = CombatSnapshot(
                hp_ratio=final_hp_ratio,
                turn_index=1,
                enemy_count=0,
                strategy=scenario.combat_strategy,
                in_combat=False,
                combat_started_ts=combat_started_ts,
                combat_finished_ts=event_ts,
                metadata={
                    "cycle_id": cycle_id,
                    "phase": "combat_finished",
                    "combat_plan_name": combat_plan_selection.plan_name,
                    "combat_plan_source": combat_plan_selection.source,
                    "combat_profile_name": combat_plan_selection.metadata.get("combat_profile_name"),
                    "input_sequence": combat_inputs,
                    "input_key": combat_plan.action_for_turn(1).key,
                },
            )
            snapshots.append(TimedCombatSnapshot(event_ts=event_ts, snapshot=final_snapshot))
            return snapshots

        hp_start = 1.0
        hp_drop = hp_start - final_hp_ratio

        for turn_index in range(1, total_turns):
            planned_action = combat_plan.action_for_turn(turn_index)
            progress = turn_index / total_turns
            hp_ratio = hp_start - (hp_drop * progress)
            event_ts = combat_started_ts + (turn_index * turn_duration_s)

            snapshot = CombatSnapshot(
                hp_ratio=max(final_hp_ratio, hp_ratio),
                turn_index=turn_index,
                enemy_count=1,
                strategy=scenario.combat_strategy,
                in_combat=True,
                combat_started_ts=combat_started_ts,
                combat_finished_ts=None,
                metadata={
                    "cycle_id": cycle_id,
                    "phase": "combat_turn",
                    "combat_plan_name": combat_plan_selection.plan_name,
                    "combat_plan_source": combat_plan_selection.source,
                    "combat_profile_name": combat_plan_selection.metadata.get("combat_profile_name"),
                    "input_sequence": combat_inputs,
                    "input_key": planned_action.key,
                },
            )
            snapshots.append(TimedCombatSnapshot(event_ts=event_ts, snapshot=snapshot))

        combat_finished_ts = combat_started_ts + (total_turns * turn_duration_s)
        final_snapshot = CombatSnapshot(
            hp_ratio=final_hp_ratio,
            turn_index=total_turns,
            enemy_count=0,
            strategy=scenario.combat_strategy,
            in_combat=False,
            combat_started_ts=combat_started_ts,
            combat_finished_ts=combat_finished_ts,
            metadata={
                "cycle_id": cycle_id,
                "phase": "combat_finished",
                "combat_plan_name": combat_plan_selection.plan_name,
                "combat_plan_source": combat_plan_selection.source,
                "combat_profile_name": combat_plan_selection.metadata.get("combat_profile_name"),
                "input_sequence": combat_inputs,
                "input_key": combat_plan.action_for_turn(total_turns).key,
            },
        )
        snapshots.append(TimedCombatSnapshot(event_ts=combat_finished_ts, snapshot=final_snapshot))

        return snapshots


class SimulatedRest:
    """
    Minimalny, deterministyczny symulator regeneracji.

    Zalozenia:
    - HP rosnie o stala wartosc co tick,
    - rest konczy sie dopiero po osiagnieciu rest_stop_threshold,
    - wszystkie snapshoty maja in_combat=False.
    """

    def __init__(
        self,
        combat_config: CombatConfig,
        *,
        rest_tick_s: float = 0.500,
        heal_per_tick: float = 0.200,
    ) -> None:
        if rest_tick_s <= 0.0:
            raise ValueError("rest_tick_s musi byc wieksze od 0.")
        if heal_per_tick <= 0.0:
            raise ValueError("heal_per_tick musi byc wieksze od 0.")

        self._combat_config = combat_config
        self._rest_tick_s = rest_tick_s
        self._heal_per_tick = heal_per_tick

    def build_timeline(
        self,
        *,
        cycle_id: int,
        rest_started_ts: float,
        starting_hp_ratio: float,
        scenario: CycleScenario,
    ) -> list[TimedCombatSnapshot]:
        if scenario.force_rest_error:
            raise RuntimeError("forced_rest_error")

        hp_ratio = min(max(starting_hp_ratio, 0.0), 1.0)
        threshold = self._combat_config.rest_stop_threshold
        snapshots: list[TimedCombatSnapshot] = []
        tick_index = 0

        while hp_ratio < threshold:
            tick_index += 1
            hp_ratio = min(1.0, hp_ratio + self._heal_per_tick)
            event_ts = rest_started_ts + (tick_index * self._rest_tick_s)
            snapshot = CombatSnapshot(
                hp_ratio=hp_ratio,
                turn_index=tick_index,
                enemy_count=0,
                strategy="rest",
                in_combat=False,
                combat_started_ts=None,
                combat_finished_ts=None,
                metadata={
                    "cycle_id": cycle_id,
                    "phase": "rest_tick",
                },
            )
            snapshots.append(TimedCombatSnapshot(event_ts=event_ts, snapshot=snapshot))

        return snapshots
