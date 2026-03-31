from __future__ import annotations

import pytest

from pathlib import Path

from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog, load_combat_plans


def test_simulated_combat_plan_catalog_lists_builtin_plans() -> None:
    catalog = SimulatedCombatPlanCatalog()

    assert "basic_1_space" in catalog.available_plan_names()
    assert "spam_1_space" in catalog.available_plan_names()
    assert "round_combo_demo" in catalog.available_plan_names()


def test_simulated_combat_plan_catalog_resolves_named_plan() -> None:
    catalog = SimulatedCombatPlanCatalog()

    selection = catalog.select_plan(plan_name="spam_1_space")

    assert selection.plan_name == "spam_1_space"
    assert selection.source == "named_catalog"
    assert selection.plan.to_input_sequence() == ("1", "1", "space")


def test_simulated_combat_plan_catalog_builds_custom_sequence_when_plan_is_not_named() -> None:
    catalog = SimulatedCombatPlanCatalog()

    selection = catalog.select_plan(input_sequence=("2", "space"))

    assert selection.plan_name == "custom_input_sequence"
    assert selection.source == "custom_input_sequence"
    assert selection.plan.to_input_sequence() == ("2", "space")


def test_simulated_combat_plan_catalog_builds_custom_round_plan() -> None:
    catalog = SimulatedCombatPlanCatalog()

    selection = catalog.select_plan(round_sequences=(("1", "space"), ("2",)))

    assert selection.plan_name == "custom_round_plan"
    assert selection.source == "custom_round_plan"
    assert selection.plan.to_input_sequence() == ("1", "space", "2")


def test_simulated_combat_plan_catalog_raises_for_unknown_named_plan() -> None:
    catalog = SimulatedCombatPlanCatalog()

    with pytest.raises(ValueError, match="Nieznany combat plan"):
        catalog.select_plan(plan_name="unknown")


def test_load_combat_plans_from_yaml_file(tmp_path: Path) -> None:
    plan_path = tmp_path / "combat-plans.yaml"
    plan_path.write_text(
        "\n".join(
            [
                "plans:",
                "  alpha:",
                "    inputs:",
                '      - "1"',
                '      - "space"',
                "  beta:",
                "    rounds:",
                '      - ["2", "space"]',
                '      - ["3"]',
            ]
        ),
        encoding="utf-8",
    )

    plans = load_combat_plans(plan_path)

    assert plans["alpha"].to_input_sequence() == ("1", "space")
    assert plans["beta"].to_input_sequence() == ("2", "space", "3")
