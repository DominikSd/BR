from __future__ import annotations

from pathlib import Path

import pytest

from botlab.adapters.simulation.combat_profiles import (
    SimulatedCombatProfileCatalog,
    load_combat_profiles,
)
from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog


def test_simulated_combat_profile_catalog_lists_builtin_profiles() -> None:
    catalog = SimulatedCombatProfileCatalog()

    assert "basic_farmer" in catalog.available_profile_names()
    assert "fast_farmer" in catalog.available_profile_names()


def test_simulated_combat_profile_catalog_resolves_profile_to_named_plan() -> None:
    catalog = SimulatedCombatProfileCatalog()

    selection = catalog.select_profile("fast_farmer")

    assert selection.plan_name == "spam_1_space"
    assert selection.source == "combat_profile"
    assert selection.metadata["combat_profile_name"] == "fast_farmer"
    assert selection.plan.to_input_sequence() == ("1", "1", "space")


def test_load_combat_profiles_from_yaml_file(tmp_path: Path) -> None:
    profile_path = tmp_path / "combat-profiles.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "profiles:",
                "  alpha:",
                '    description: "Alpha profile"',
                '    combat_plan_name: "basic_1_space"',
            ]
        ),
        encoding="utf-8",
    )

    profiles = load_combat_profiles(
        profile_path,
        combat_plan_catalog=SimulatedCombatPlanCatalog(),
    )

    assert profiles["alpha"].combat_plan_name == "basic_1_space"


def test_simulated_combat_profile_catalog_raises_for_unknown_profile() -> None:
    catalog = SimulatedCombatProfileCatalog()

    with pytest.raises(ValueError, match="Nieznany combat profile"):
        catalog.select_profile("unknown")
