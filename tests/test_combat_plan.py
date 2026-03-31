from __future__ import annotations

import pytest

from botlab.domain.combat_plan import CombatAction, CombatPlan, CombatRoundPlan


def test_combat_plan_can_be_built_from_simple_input_sequence() -> None:
    plan = CombatPlan.from_input_sequence(("1", "space"), name="simple")

    assert plan.name == "simple"
    assert len(plan.rounds) == 2
    assert plan.rounds[0].round_index == 1
    assert plan.rounds[0].actions[0].key == "1"
    assert plan.rounds[1].actions[0].key == "space"
    assert plan.to_input_sequence() == ("1", "space")


def test_combat_plan_cycles_actions_when_turn_count_exceeds_declared_rounds() -> None:
    plan = CombatPlan.from_input_sequence(("1", "space"), name="loop")

    assert plan.action_for_turn(1).key == "1"
    assert plan.action_for_turn(2).key == "space"
    assert plan.action_for_turn(3).key == "1"
    assert plan.action_for_turn(4).key == "space"


def test_combat_plan_supports_multiple_actions_inside_single_round() -> None:
    plan = CombatPlan(
        name="round-combo",
        rounds=(
            CombatRoundPlan(
                round_index=1,
                actions=(
                    CombatAction(key="1"),
                    CombatAction(key="space"),
                ),
            ),
            CombatRoundPlan(
                round_index=2,
                actions=(CombatAction(key="2"),),
            ),
        ),
    )

    assert plan.to_input_sequence() == ("1", "space", "2")
    assert plan.action_for_turn(1).key == "1"
    assert plan.action_for_turn(2).key == "space"
    assert plan.action_for_turn(3).key == "2"
    assert plan.action_for_turn(4).key == "1"


def test_combat_plan_can_be_built_from_round_sequences() -> None:
    plan = CombatPlan.from_round_sequences(
        (("1", "space"), ("2",)),
        name="rounds",
    )

    assert plan.name == "rounds"
    assert len(plan.rounds) == 2
    assert plan.rounds[0].actions[0].key == "1"
    assert plan.rounds[0].actions[1].key == "space"
    assert plan.rounds[1].actions[0].key == "2"
    assert plan.to_input_sequence() == ("1", "space", "2")


def test_combat_plan_rejects_empty_or_non_sequential_rounds() -> None:
    with pytest.raises(ValueError, match="CombatPlan.rounds nie moze byc puste"):
        CombatPlan(name="invalid", rounds=())

    with pytest.raises(ValueError, match="CombatPlan.rounds musza byc kolejne"):
        CombatPlan(
            name="invalid",
            rounds=(
                CombatRoundPlan(round_index=1, actions=(CombatAction(key="1"),)),
                CombatRoundPlan(round_index=3, actions=(CombatAction(key="space"),)),
            ),
        )


def test_combat_round_plan_and_action_validate_required_values() -> None:
    with pytest.raises(ValueError, match="CombatAction.key nie moze byc pusty"):
        CombatAction(key=" ")

    with pytest.raises(ValueError, match="CombatRoundPlan.actions nie moze byc puste"):
        CombatRoundPlan(round_index=1, actions=())

    with pytest.raises(ValueError, match="round_sequences nie moze byc puste"):
        CombatPlan.from_round_sequences(())
