from __future__ import annotations

from botlab.config import load_default_config
from botlab.simulation.battle import SimulatedBattle, SimulatedRest
from botlab.simulation.spawner import CycleScenario
import pytest


def test_simulated_battle_finishes_with_final_snapshot_and_expected_hp() -> None:
    battle = SimulatedBattle(default_turn_duration_s=0.25)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=4,
        combat_turn_duration_s=0.25,
        combat_final_hp_ratio=0.40,
        combat_strategy="aggressive",
        note="battle-test",
    )

    timeline = battle.build_timeline(
        cycle_id=1,
        combat_started_ts=200.0,
        scenario=scenario,
    )

    assert len(timeline) == 4
    assert timeline[-1].event_ts == 201.0
    assert timeline[-1].snapshot.in_combat is False
    assert timeline[-1].snapshot.hp_ratio == 0.40
    assert timeline[-1].snapshot.strategy == "aggressive"

    assert any(item.snapshot.in_combat is True for item in timeline[:-1])


def test_simulated_rest_recovers_hp_to_stop_threshold() -> None:
    settings = load_default_config()
    rest = SimulatedRest(settings.combat, rest_tick_s=0.5, heal_per_tick=0.2)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=3,
        combat_final_hp_ratio=0.40,
        combat_strategy="default",
        note="rest-test",
    )

    timeline = rest.build_timeline(
        cycle_id=1,
        rest_started_ts=300.0,
        starting_hp_ratio=0.40,
        scenario=scenario,
    )

    assert len(timeline) >= 1
    assert timeline[0].event_ts == 300.5
    assert timeline[-1].snapshot.in_combat is False
    assert timeline[-1].snapshot.hp_ratio >= settings.combat.rest_stop_threshold


def test_simulated_battle_raises_error_on_force_battle_error() -> None:
    battle = SimulatedBattle(default_turn_duration_s=0.25)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=4,
        combat_turn_duration_s=0.25,
        combat_final_hp_ratio=0.40,
        combat_strategy="default",
        force_battle_error=True,
        note="battle-error-test",
    )

    with pytest.raises(RuntimeError, match="forced_battle_error"):
        battle.build_timeline(
            cycle_id=1,
            combat_started_ts=200.0,
            scenario=scenario,
        )


def test_simulated_rest_raises_error_on_force_rest_error() -> None:
    settings = load_default_config()
    rest = SimulatedRest(settings.combat, rest_tick_s=0.5, heal_per_tick=0.2)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=3,
        combat_final_hp_ratio=0.40,
        combat_strategy="default",
        force_rest_error=True,
        note="rest-error-test",
    )

    with pytest.raises(RuntimeError, match="forced_rest_error"):
        rest.build_timeline(
            cycle_id=1,
            rest_started_ts=300.0,
            starting_hp_ratio=0.40,
            scenario=scenario,
        )
