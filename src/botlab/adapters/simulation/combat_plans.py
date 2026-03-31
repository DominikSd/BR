from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml

from botlab.application import CombatPlanCatalog, CombatPlanSelection
from botlab.constants import DEFAULT_COMBAT_PLANS_PATH
from botlab.domain.combat_plan import CombatPlan


class SimulatedCombatPlanCatalog(CombatPlanCatalog):
    def __init__(
        self,
        *,
        plan_file_path: str | Path | None = None,
        plans: Mapping[str, CombatPlan] | None = None,
    ) -> None:
        loaded_plans: dict[str, CombatPlan] = {}
        if plan_file_path is not None or plans is None:
            loaded_plans.update(load_combat_plans(plan_file_path or DEFAULT_COMBAT_PLANS_PATH))
        if plans is not None:
            loaded_plans.update(dict(plans))
        if not loaded_plans:
            raise ValueError("Combat plan catalog nie moze byc pusty.")

        self._plans = loaded_plans

    def select_plan(
        self,
        *,
        plan_name: str | None = None,
        input_sequence: tuple[str, ...] | None = None,
        round_sequences: tuple[tuple[str, ...], ...] | None = None,
    ) -> CombatPlanSelection:
        if plan_name is not None:
            try:
                plan = self._plans[plan_name]
            except KeyError as exc:
                available = ", ".join(self.available_plan_names())
                raise ValueError(
                    f"Nieznany combat plan '{plan_name}'. Dostepne: {available}"
                ) from exc
            return CombatPlanSelection(
                plan_name=plan.name,
                plan=plan,
                source="named_catalog",
            )

        if round_sequences is not None:
            custom_round_plan = CombatPlan.from_round_sequences(
                round_sequences,
                name="custom_round_plan",
            )
            return CombatPlanSelection(
                plan_name=custom_round_plan.name,
                plan=custom_round_plan,
                source="custom_round_plan",
                metadata={"round_sequences": round_sequences},
            )

        if input_sequence is None:
            default_plan = self._default_plan()
            return CombatPlanSelection(
                plan_name=default_plan.name,
                plan=default_plan,
                source="default_catalog_plan",
            )

        if input_sequence == self._default_plan().to_input_sequence():
            default_plan = self._default_plan()
            return CombatPlanSelection(
                plan_name=default_plan.name,
                plan=default_plan,
                source="default_catalog_plan",
            )

        custom_plan = CombatPlan.from_input_sequence(
            input_sequence,
            name="custom_input_sequence",
        )
        return CombatPlanSelection(
            plan_name=custom_plan.name,
            plan=custom_plan,
            source="custom_input_sequence",
            metadata={"input_sequence": custom_plan.to_input_sequence()},
        )

    def available_plan_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plans))

    def _default_plan(self) -> CombatPlan:
        if "basic_1_space" in self._plans:
            return self._plans["basic_1_space"]
        first_plan_name = self.available_plan_names()[0]
        return self._plans[first_plan_name]


def load_combat_plans(plan_file_path: str | Path) -> dict[str, CombatPlan]:
    path = Path(plan_file_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Plik combat plans nie istnieje: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError("Combat plans YAML musi byc mapa.")

    raw_plans = data.get("plans")
    if not isinstance(raw_plans, Mapping):
        raise ValueError("Sekcja 'plans' musi byc mapa nazw planow.")

    plans: dict[str, CombatPlan] = {}
    for plan_name, raw_plan in raw_plans.items():
        if not isinstance(plan_name, str) or not plan_name.strip():
            raise ValueError("Kazda nazwa combat plan musi byc niepustym stringiem.")
        plans[plan_name.strip()] = _parse_named_plan(plan_name.strip(), raw_plan)

    if not plans:
        raise ValueError("Sekcja 'plans' nie moze byc pusta.")
    return plans


def _parse_named_plan(plan_name: str, raw_plan: object) -> CombatPlan:
    if not isinstance(raw_plan, Mapping):
        raise ValueError(f"Plan '{plan_name}' musi byc mapa.")

    raw_inputs = raw_plan.get("inputs")
    raw_rounds = raw_plan.get("rounds")
    if raw_inputs is not None and raw_rounds is not None:
        raise ValueError(f"Plan '{plan_name}' nie moze miec jednoczesnie 'inputs' i 'rounds'.")

    if raw_rounds is not None:
        round_sequences = _parse_round_sequences(raw_rounds, plan_name=plan_name)
        return CombatPlan.from_round_sequences(round_sequences, name=plan_name)

    if raw_inputs is not None:
        input_sequence = _parse_input_sequence(raw_inputs, plan_name=plan_name)
        return CombatPlan.from_input_sequence(input_sequence, name=plan_name)

    raise ValueError(f"Plan '{plan_name}' musi definiowac 'inputs' albo 'rounds'.")


def _parse_input_sequence(raw_inputs: object, *, plan_name: str) -> tuple[str, ...]:
    if not isinstance(raw_inputs, list):
        raise ValueError(f"Plan '{plan_name}'.inputs musi byc lista stringow.")

    inputs: list[str] = []
    for item in raw_inputs:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Plan '{plan_name}'.inputs musi zawierac niepuste stringi.")
        inputs.append(item.strip())

    if not inputs:
        raise ValueError(f"Plan '{plan_name}'.inputs nie moze byc puste.")
    return tuple(inputs)


def _parse_round_sequences(
    raw_rounds: object,
    *,
    plan_name: str,
) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw_rounds, list):
        raise ValueError(f"Plan '{plan_name}'.rounds musi byc lista rund.")

    rounds: list[tuple[str, ...]] = []
    for raw_round in raw_rounds:
        if not isinstance(raw_round, list):
            raise ValueError(f"Plan '{plan_name}'.rounds musi zawierac listy akcji.")

        actions: list[str] = []
        for action_key in raw_round:
            if not isinstance(action_key, str) or not action_key.strip():
                raise ValueError(
                    f"Plan '{plan_name}'.rounds musi zawierac niepuste stringi."
                )
            actions.append(action_key.strip())

        if not actions:
            raise ValueError(f"Plan '{plan_name}'.rounds nie moze zawierac pustej rundy.")
        rounds.append(tuple(actions))

    if not rounds:
        raise ValueError(f"Plan '{plan_name}'.rounds nie moze byc puste.")
    return tuple(rounds)
