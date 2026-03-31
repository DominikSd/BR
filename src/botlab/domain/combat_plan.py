from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class CombatAction:
    key: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("CombatAction.key nie moze byc pusty.")


@dataclass(slots=True, frozen=True)
class CombatRoundPlan:
    round_index: int
    actions: tuple[CombatAction, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_index <= 0:
            raise ValueError("CombatRoundPlan.round_index musi byc wieksze od 0.")
        if not self.actions:
            raise ValueError("CombatRoundPlan.actions nie moze byc puste.")


@dataclass(slots=True, frozen=True)
class CombatPlan:
    name: str
    rounds: tuple[CombatRoundPlan, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CombatPlan.name nie moze byc puste.")
        if not self.rounds:
            raise ValueError("CombatPlan.rounds nie moze byc puste.")

        expected_round_index = 1
        for round_plan in self.rounds:
            if round_plan.round_index != expected_round_index:
                raise ValueError("CombatPlan.rounds musza byc kolejne i zaczynac od 1.")
            expected_round_index += 1

    @classmethod
    def from_input_sequence(
        cls,
        inputs: tuple[str, ...],
        *,
        name: str = "default_click_sequence",
    ) -> "CombatPlan":
        if not inputs:
            raise ValueError("inputs nie moze byc puste.")

        rounds = tuple(
            CombatRoundPlan(
                round_index=index,
                actions=(CombatAction(key=input_key),),
            )
            for index, input_key in enumerate(inputs, start=1)
        )
        return cls(name=name, rounds=rounds)

    @classmethod
    def from_round_sequences(
        cls,
        round_sequences: tuple[tuple[str, ...], ...],
        *,
        name: str = "round_based_plan",
    ) -> "CombatPlan":
        if not round_sequences:
            raise ValueError("round_sequences nie moze byc puste.")

        rounds = []
        for round_index, action_keys in enumerate(round_sequences, start=1):
            if not action_keys:
                raise ValueError("Kazda runda musi zawierac co najmniej jedna akcje.")
            rounds.append(
                CombatRoundPlan(
                    round_index=round_index,
                    actions=tuple(CombatAction(key=key) for key in action_keys),
                )
            )

        return cls(name=name, rounds=tuple(rounds))

    def to_input_sequence(self) -> tuple[str, ...]:
        return tuple(action.key for round_plan in self.rounds for action in round_plan.actions)

    def action_for_turn(self, turn_index: int) -> CombatAction:
        if turn_index <= 0:
            raise ValueError("turn_index musi byc wieksze od 0.")

        flattened_actions = tuple(
            action
            for round_plan in self.rounds
            for action in round_plan.actions
        )
        sequence_index = (turn_index - 1) % len(flattened_actions)
        return flattened_actions[sequence_index]
