from __future__ import annotations

from botlab.config import load_default_config
from botlab.adapters.simulation.battle import SimulatedBattle, SimulatedRest
from botlab.adapters.simulation.spawner import CycleScenario
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
    assert timeline[0].snapshot.metadata["combat_plan_name"] == "basic_1_space"
    assert timeline[0].snapshot.metadata["combat_plan_source"] == "default_catalog_plan"
    assert timeline[0].snapshot.metadata["input_sequence"] == ("1", "space", "3", "space")
    assert timeline[0].snapshot.metadata["input_key"] == "1"
    assert timeline[1].snapshot.metadata["input_key"] == "space"
    assert timeline[2].snapshot.metadata["input_key"] == "3"
    assert timeline[3].snapshot.metadata["input_key"] == "space"

    assert any(item.snapshot.in_combat is True for item in timeline[:-1])


def test_simulated_battle_uses_custom_input_sequence_in_turn_metadata() -> None:
    battle = SimulatedBattle(default_turn_duration_s=0.25)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=3,
        combat_turn_duration_s=0.25,
        combat_inputs=("1", "space", "2"),
    )

    timeline = battle.build_timeline(
        cycle_id=7,
        combat_started_ts=50.0,
        scenario=scenario,
    )

    assert [item.snapshot.metadata["input_key"] for item in timeline] == ["1", "space", "2"]
    assert all(item.snapshot.metadata["combat_plan_name"] == "custom_input_sequence" for item in timeline)
    assert all(item.snapshot.metadata["combat_plan_source"] == "custom_input_sequence" for item in timeline)
    assert all(item.snapshot.metadata["input_sequence"] == ("1", "space", "2") for item in timeline)


def test_simulated_battle_can_use_named_combat_plan_from_catalog() -> None:
    battle = SimulatedBattle(default_turn_duration_s=0.25)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=3,
        combat_turn_duration_s=0.25,
        combat_plan_name="spam_1_space",
    )

    timeline = battle.build_timeline(
        cycle_id=8,
        combat_started_ts=60.0,
        scenario=scenario,
    )

    assert [item.snapshot.metadata["input_key"] for item in timeline] == ["1", "1", "space"]
    assert all(item.snapshot.metadata["combat_plan_name"] == "spam_1_space" for item in timeline)
    assert all(item.snapshot.metadata["combat_plan_source"] == "named_catalog" for item in timeline)


def test_simulated_battle_can_use_combat_profile_mapping() -> None:
    battle = SimulatedBattle(default_turn_duration_s=0.25)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=3,
        combat_turn_duration_s=0.25,
        combat_profile_name="fast_farmer",
    )

    timeline = battle.build_timeline(
        cycle_id=10,
        combat_started_ts=80.0,
        scenario=scenario,
    )

    assert [item.snapshot.metadata["input_key"] for item in timeline] == ["1", "1", "space"]
    assert all(item.snapshot.metadata["combat_plan_name"] == "spam_1_space" for item in timeline)
    assert all(item.snapshot.metadata["combat_plan_source"] == "combat_profile" for item in timeline)
    assert all(item.snapshot.metadata["combat_profile_name"] == "fast_farmer" for item in timeline)


def test_simulated_battle_can_use_round_based_plan_definition() -> None:
    battle = SimulatedBattle(default_turn_duration_s=0.25)

    scenario = CycleScenario(
        has_event=True,
        drift_s=0.0,
        verify_result="success",
        combat_turns=3,
        combat_turn_duration_s=0.25,
        combat_plan_rounds=(("1", "space"), ("2",)),
    )

    timeline = battle.build_timeline(
        cycle_id=9,
        combat_started_ts=70.0,
        scenario=scenario,
    )

    assert [item.snapshot.metadata["input_key"] for item in timeline] == ["1", "space", "2"]
    assert all(item.snapshot.metadata["combat_plan_name"] == "custom_round_plan" for item in timeline)
    assert all(item.snapshot.metadata["combat_plan_source"] == "custom_round_plan" for item in timeline)


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
