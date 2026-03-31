from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog
from botlab.application import CombatPlanSelection
from botlab.constants import DEFAULT_COMBAT_PROFILES_PATH


@dataclass(slots=True, frozen=True)
class CombatProfile:
    name: str
    combat_plan_name: str
    description: str = ""


class SimulatedCombatProfileCatalog:
    def __init__(
        self,
        *,
        profile_file_path: str | Path | None = None,
        profiles: Mapping[str, CombatProfile] | None = None,
        combat_plan_catalog: SimulatedCombatPlanCatalog | None = None,
    ) -> None:
        self._combat_plan_catalog = combat_plan_catalog or SimulatedCombatPlanCatalog()
        loaded_profiles: dict[str, CombatProfile] = {}
        if profile_file_path is not None or profiles is None:
            loaded_profiles.update(
                load_combat_profiles(
                    profile_file_path or DEFAULT_COMBAT_PROFILES_PATH,
                    combat_plan_catalog=self._combat_plan_catalog,
                )
            )
        if profiles is not None:
            loaded_profiles.update(dict(profiles))
        if not loaded_profiles:
            raise ValueError("Combat profile catalog nie moze byc pusty.")

        self._profiles = loaded_profiles

    def available_profile_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def select_profile(self, profile_name: str) -> CombatPlanSelection:
        try:
            profile = self._profiles[profile_name]
        except KeyError as exc:
            available = ", ".join(self.available_profile_names())
            raise ValueError(
                f"Nieznany combat profile '{profile_name}'. Dostepne: {available}"
            ) from exc

        plan_selection = self._combat_plan_catalog.select_plan(
            plan_name=profile.combat_plan_name,
        )
        return CombatPlanSelection(
            plan_name=plan_selection.plan_name,
            plan=plan_selection.plan,
            source="combat_profile",
            metadata={
                "combat_profile_name": profile.name,
                "combat_profile_description": profile.description,
            },
        )


def load_combat_profiles(
    profile_file_path: str | Path,
    *,
    combat_plan_catalog: SimulatedCombatPlanCatalog | None = None,
) -> dict[str, CombatProfile]:
    path = Path(profile_file_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Plik combat profiles nie istnieje: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError("Combat profiles YAML musi byc mapa.")

    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("Sekcja 'profiles' musi byc mapa nazw profili.")

    plan_catalog = combat_plan_catalog or SimulatedCombatPlanCatalog()
    profiles: dict[str, CombatProfile] = {}
    for profile_name, raw_profile in raw_profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("Kazda nazwa combat profile musi byc niepustym stringiem.")
        profile = _parse_profile(profile_name.strip(), raw_profile)
        plan_catalog.select_plan(plan_name=profile.combat_plan_name)
        profiles[profile.name] = profile

    if not profiles:
        raise ValueError("Sekcja 'profiles' nie moze byc pusta.")
    return profiles


def _parse_profile(profile_name: str, raw_profile: object) -> CombatProfile:
    if not isinstance(raw_profile, Mapping):
        raise ValueError(f"Profil '{profile_name}' musi byc mapa.")

    combat_plan_name = raw_profile.get("combat_plan_name")
    if not isinstance(combat_plan_name, str) or not combat_plan_name.strip():
        raise ValueError(
            f"Profil '{profile_name}' musi definiowac niepuste 'combat_plan_name'."
        )

    description = raw_profile.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"Profil '{profile_name}'.description musi byc stringiem.")

    return CombatProfile(
        name=profile_name,
        combat_plan_name=combat_plan_name.strip(),
        description=description.strip(),
    )
