from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from botlab.adapters.simulation.combat_profiles import SimulatedCombatProfileCatalog
from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog
from botlab.adapters.simulation.runner import SimulationRunner
from botlab.adapters.simulation.spawner import (
    CycleScenario,
    SimulatedGroupState,
    SimulatedSpawner,
)
from botlab.application import SimulationReport
from botlab.config import Settings


@dataclass(slots=True, frozen=True)
class ScenarioReplay:
    name: str
    description: str = ""
    total_cycles: int = 1
    initial_anchor_spawn_ts: float = 100.0
    initial_anchor_cycle_id: int = 0
    default_scenario: CycleScenario = field(default_factory=CycleScenario)
    overrides: dict[int, CycleScenario] = field(default_factory=dict)

    def __post_init__(self) -> None:
        combat_plan_catalog = SimulatedCombatPlanCatalog()
        combat_profile_catalog = SimulatedCombatProfileCatalog(
            combat_plan_catalog=combat_plan_catalog
        )
        if not self.name.strip():
            raise ValueError("ScenarioReplay.name nie moze byc pusty.")
        if self.total_cycles <= 0:
            raise ValueError("ScenarioReplay.total_cycles musi byc wieksze od 0.")
        if self.initial_anchor_cycle_id < 0:
            raise ValueError("ScenarioReplay.initial_anchor_cycle_id nie moze byc ujemne.")
        _validate_scenario_combat_config(
            self.default_scenario,
            combat_plan_catalog,
            combat_profile_catalog,
        )
        for cycle_id, scenario in self.overrides.items():
            if cycle_id <= 0:
                raise ValueError("Override cycle_id musi byc wieksze od 0.")
            _validate_scenario_combat_config(
                scenario,
                combat_plan_catalog,
                combat_profile_catalog,
            )

    def build_spawner(self) -> SimulatedSpawner:
        return SimulatedSpawner(
            default_scenario=self.default_scenario,
            overrides=self.overrides,
        )


class ScenarioReplayRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        replay: ScenarioReplay,
        enable_console: bool = False,
    ) -> None:
        self._settings = settings
        self._replay = replay
        self._enable_console = enable_console

    @property
    def replay(self) -> ScenarioReplay:
        return self._replay

    @classmethod
    def from_preset(
        cls,
        settings: Settings,
        *,
        preset_name: str,
        enable_console: bool = False,
    ) -> "ScenarioReplayRunner":
        return cls(
            settings=settings,
            replay=get_scenario_replay_preset(preset_name),
            enable_console=enable_console,
        )

    @classmethod
    def from_file(
        cls,
        settings: Settings,
        *,
        replay_path: str | Path,
        enable_console: bool = False,
    ) -> "ScenarioReplayRunner":
        return cls(
            settings=settings,
            replay=load_scenario_replay(replay_path),
            enable_console=enable_console,
        )

    def build_runner(
        self,
        *,
        initial_anchor_spawn_ts: float | None = None,
        initial_anchor_cycle_id: int | None = None,
    ) -> SimulationRunner:
        return SimulationRunner.from_settings(
            self._settings,
            spawner=self._replay.build_spawner(),
            initial_anchor_spawn_ts=(
                self._replay.initial_anchor_spawn_ts
                if initial_anchor_spawn_ts is None
                else initial_anchor_spawn_ts
            ),
            initial_anchor_cycle_id=(
                self._replay.initial_anchor_cycle_id
                if initial_anchor_cycle_id is None
                else initial_anchor_cycle_id
            ),
            enable_console=self._enable_console,
        )

    def run(
        self,
        *,
        total_cycles: int | None = None,
        initial_anchor_spawn_ts: float | None = None,
        initial_anchor_cycle_id: int | None = None,
    ) -> SimulationReport:
        runner = self.build_runner(
            initial_anchor_spawn_ts=initial_anchor_spawn_ts,
            initial_anchor_cycle_id=initial_anchor_cycle_id,
        )
        cycles_to_run = self._replay.total_cycles if total_cycles is None else total_cycles
        return runner.run_cycles(cycles_to_run)


def list_scenario_replay_presets() -> tuple[ScenarioReplay, ...]:
    presets = _build_preset_catalog()
    return tuple(presets[name] for name in sorted(presets))


def get_scenario_replay_preset(name: str) -> ScenarioReplay:
    presets = _build_preset_catalog()
    try:
        return presets[name]
    except KeyError as exc:
        available = ", ".join(sorted(presets))
        raise ValueError(
            f"Nieznany scenario preset '{name}'. Dostepne: {available}"
        ) from exc


def load_scenario_replay(replay_path: str | Path) -> ScenarioReplay:
    path = Path(replay_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Plik scenario replay nie istnieje: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, Mapping):
        raise ValueError("Scenario replay musi byc mapa YAML.")

    name = _require_non_empty_str(data, "name")
    description = _optional_str(data, "description", "")
    total_cycles = _optional_positive_int(data, "total_cycles", 1)
    initial_anchor_spawn_ts = _optional_float(data, "initial_anchor_spawn_ts", 100.0)
    initial_anchor_cycle_id = _optional_non_negative_int(data, "initial_anchor_cycle_id", 0)

    raw_default_scenario = data.get("default_scenario")
    if raw_default_scenario is None:
        default_scenario = CycleScenario()
    else:
        default_scenario = _parse_cycle_scenario(raw_default_scenario)

    raw_overrides = data.get("overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("Pole 'overrides' musi byc mapa cycle_id -> scenario.")

    overrides: dict[int, CycleScenario] = {}
    for raw_cycle_id, raw_scenario in raw_overrides.items():
        cycle_id = _coerce_cycle_id(raw_cycle_id)
        overrides[cycle_id] = _parse_cycle_scenario(raw_scenario)

    return ScenarioReplay(
        name=name,
        description=description,
        total_cycles=total_cycles,
        initial_anchor_spawn_ts=initial_anchor_spawn_ts,
        initial_anchor_cycle_id=initial_anchor_cycle_id,
        default_scenario=default_scenario,
        overrides=overrides,
    )


def _build_preset_catalog() -> dict[str, ScenarioReplay]:
    baseline = ScenarioReplay(
        name="baseline_mixed_cycle",
        description="Mieszany replay z success, failure, no_event, late, timeout i no_target.",
        total_cycles=6,
        overrides={
            1: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="success",
                combat_final_hp_ratio=0.40,
                note="preset-success-rest",
            ),
            2: CycleScenario(
                has_event=True,
                drift_s=0.2,
                verify_result="failure",
                note="preset-failure",
            ),
            3: CycleScenario(
                has_event=False,
                note="preset-no-event",
            ),
            4: CycleScenario(
                has_event=True,
                drift_s=1.2,
                verify_result="success",
                note="preset-late",
            ),
            5: CycleScenario(
                has_event=True,
                drift_s=0.0,
                verify_result="timeout",
                note="preset-timeout",
            ),
            6: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(
                        group_id="busy-near",
                        position_xy=(1.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(
                        group_id="blocked-far",
                        position_xy=(3.0, 0.0),
                        reachable=False,
                    ),
                ),
                note="preset-no-target",
            ),
        },
    )

    retarget = ScenarioReplay(
        name="retarget_path",
        description="Replay pokazujacy utrate targetu podczas dojscia i przed interakcja.",
        total_cycles=2,
        overrides={
            1: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(group_id="current", position_xy=(2.0, 0.0)),
                    SimulatedGroupState(group_id="replacement", position_xy=(6.0, 0.0)),
                ),
                approach_revalidation_delay_s=0.3,
                approach_bot_position_xy=(1.0, 0.0),
                approach_groups=(
                    SimulatedGroupState(
                        group_id="current",
                        position_xy=(2.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(group_id="replacement", position_xy=(4.0, 0.0)),
                ),
                note="preset-retarget-during-approach",
            ),
            2: CycleScenario(
                has_event=True,
                spawn_zone_visible=True,
                bot_position_xy=(0.0, 0.0),
                groups=(
                    SimulatedGroupState(group_id="current", position_xy=(2.0, 0.0)),
                    SimulatedGroupState(group_id="replacement", position_xy=(6.0, 0.0)),
                ),
                interaction_revalidation_delay_s=0.6,
                interaction_bot_position_xy=(2.0, 0.0),
                interaction_groups=(
                    SimulatedGroupState(
                        group_id="current",
                        position_xy=(2.0, 0.0),
                        engaged_by_other=True,
                    ),
                    SimulatedGroupState(group_id="replacement", position_xy=(3.0, 0.0)),
                ),
                note="preset-retarget-before-interaction",
            ),
        },
    )

    return {
        baseline.name: baseline,
        retarget.name: retarget,
    }


def _parse_cycle_scenario(raw_scenario: object) -> CycleScenario:
    if not isinstance(raw_scenario, Mapping):
        raise ValueError("Scenario musi byc mapa YAML.")

    return CycleScenario(
        has_event=_optional_bool(raw_scenario, "has_event", True),
        drift_s=_optional_float(raw_scenario, "drift_s", 0.0),
        verify_result=_optional_verify_outcome(raw_scenario, "verify_result", "success"),
        combat_turns=_optional_positive_int(raw_scenario, "combat_turns", 3),
        combat_turn_duration_s=_optional_nullable_float(raw_scenario, "combat_turn_duration_s", None),
        combat_final_hp_ratio=_optional_float(raw_scenario, "combat_final_hp_ratio", 0.80),
        combat_strategy=_optional_str(raw_scenario, "combat_strategy", "default"),
        combat_profile_name=_optional_nullable_str(raw_scenario, "combat_profile_name", None),
        combat_plan_name=_optional_nullable_str(raw_scenario, "combat_plan_name", None),
        combat_plan_rounds=_optional_round_sequences(raw_scenario, "combat_plan_rounds", None),
        combat_inputs=_optional_str_tuple(raw_scenario, "combat_inputs", ("1", "space")),
        force_battle_error=_optional_bool(raw_scenario, "force_battle_error", False),
        force_rest_error=_optional_bool(raw_scenario, "force_rest_error", False),
        spawn_zone_visible=_optional_bool(raw_scenario, "spawn_zone_visible", True),
        bot_position_xy=_optional_point(raw_scenario, "bot_position_xy", (0.0, 0.0)),
        approach_revalidation_delay_s=_optional_float(
            raw_scenario,
            "approach_revalidation_delay_s",
            0.250,
        ),
        approach_bot_position_xy=_optional_nullable_point(
            raw_scenario,
            "approach_bot_position_xy",
            None,
        ),
        interaction_revalidation_delay_s=_optional_float(
            raw_scenario,
            "interaction_revalidation_delay_s",
            0.450,
        ),
        interaction_bot_position_xy=_optional_nullable_point(
            raw_scenario,
            "interaction_bot_position_xy",
            None,
        ),
        current_target_id=_optional_nullable_str(raw_scenario, "current_target_id", None),
        approach_groups=_optional_groups(raw_scenario, "approach_groups", None),
        interaction_groups=_optional_groups(raw_scenario, "interaction_groups", None),
        groups=_optional_groups(raw_scenario, "groups", ()),
        note=_optional_str(raw_scenario, "note", ""),
    )


def _parse_group(raw_group: object) -> SimulatedGroupState:
    if not isinstance(raw_group, Mapping):
        raise ValueError("Kazda grupa musi byc mapa YAML.")

    return SimulatedGroupState(
        group_id=_require_non_empty_str(raw_group, "group_id"),
        position_xy=_optional_point(raw_group, "position_xy", (0.0, 0.0)),
        alive_count=_optional_non_negative_int(raw_group, "alive_count", 3),
        engaged_by_other=_optional_bool(raw_group, "engaged_by_other", False),
        reachable=_optional_bool(raw_group, "reachable", True),
        threat_score=_optional_float(raw_group, "threat_score", 0.0),
        metadata=_optional_mapping(raw_group, "metadata", {}),
    )


def _coerce_cycle_id(value: object) -> int:
    if isinstance(value, int):
        cycle_id = value
    elif isinstance(value, str) and value.strip().isdigit():
        cycle_id = int(value.strip())
    else:
        raise ValueError("Klucze 'overrides' musza byc dodatnimi liczbami calkowitymi.")

    if cycle_id <= 0:
        raise ValueError("Klucze 'overrides' musza byc dodatnimi liczbami calkowitymi.")
    return cycle_id


def _require_non_empty_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pole '{key}' musi byc niepustym stringiem.")
    return value.strip()


def _optional_str(data: Mapping[str, object], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Pole '{key}' musi byc stringiem.")
    return value.strip()


def _optional_nullable_str(
    data: Mapping[str, object],
    key: str,
    default: str | None,
) -> str | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Pole '{key}' musi byc stringiem albo null.")
    stripped = value.strip()
    return stripped or None


def _optional_bool(data: Mapping[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Pole '{key}' musi byc bool.")
    return value


def _optional_float(data: Mapping[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"Pole '{key}' musi byc liczba.")
    return float(value)


def _optional_nullable_float(
    data: Mapping[str, object],
    key: str,
    default: float | None,
) -> float | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Pole '{key}' musi byc liczba albo null.")
    return float(value)


def _optional_positive_int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Pole '{key}' musi byc liczba calkowita.")
    if value <= 0:
        raise ValueError(f"Pole '{key}' musi byc wieksze od 0.")
    return value


def _optional_non_negative_int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Pole '{key}' musi byc liczba calkowita.")
    if value < 0:
        raise ValueError(f"Pole '{key}' nie moze byc ujemne.")
    return value


def _optional_verify_outcome(data: Mapping[str, object], key: str, default: str) -> str:
    value = _optional_str(data, key, default)
    if value not in {"success", "failure", "timeout"}:
        raise ValueError("Pole 'verify_result' musi byc jednym z: success, failure, timeout.")
    return value


def _optional_point(
    data: Mapping[str, object],
    key: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Pole '{key}' musi byc para [x, y].")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError(f"Pole '{key}' musi zawierac liczby.")
    return (float(x), float(y))


def _optional_nullable_point(
    data: Mapping[str, object],
    key: str,
    default: tuple[float, float] | None,
) -> tuple[float, float] | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Pole '{key}' musi byc para [x, y] albo null.")
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError(f"Pole '{key}' musi zawierac liczby.")
    return (float(x), float(y))


def _optional_groups(
    data: Mapping[str, object],
    key: str,
    default: tuple[SimulatedGroupState, ...] | None,
) -> tuple[SimulatedGroupState, ...] | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if not isinstance(value, list):
        raise ValueError(f"Pole '{key}' musi byc lista grup albo null.")
    return tuple(_parse_group(item) for item in value)


def _optional_str_tuple(
    data: Mapping[str, object],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = data.get(key, default)
    if not isinstance(value, list):
        if isinstance(value, tuple):
            items = list(value)
        else:
            raise ValueError(f"Pole '{key}' musi byc lista stringow.")
    else:
        items = value
    result: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Pole '{key}' musi zawierac niepuste stringi.")
        result.append(item.strip())
    if not result:
        raise ValueError(f"Pole '{key}' nie moze byc puste.")
    return tuple(result)


def _optional_round_sequences(
    data: Mapping[str, object],
    key: str,
    default: tuple[tuple[str, ...], ...] | None,
) -> tuple[tuple[str, ...], ...] | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if not isinstance(value, list):
        raise ValueError(f"Pole '{key}' musi byc lista rund albo null.")

    rounds: list[tuple[str, ...]] = []
    for raw_round in value:
        if not isinstance(raw_round, list):
            raise ValueError(f"Pole '{key}' musi zawierac listy akcji.")
        round_actions: list[str] = []
        for action_key in raw_round:
            if not isinstance(action_key, str) or not action_key.strip():
                raise ValueError(f"Pole '{key}' musi zawierac niepuste stringi.")
            round_actions.append(action_key.strip())
        if not round_actions:
            raise ValueError(f"Pole '{key}' nie moze zawierac pustej rundy.")
        rounds.append(tuple(round_actions))

    if not rounds:
        raise ValueError(f"Pole '{key}' nie moze byc puste.")
    return tuple(rounds)


def _optional_mapping(
    data: Mapping[str, object],
    key: str,
    default: dict[str, object],
) -> dict[str, object]:
    value = data.get(key, default)
    if not isinstance(value, Mapping):
        raise ValueError(f"Pole '{key}' musi byc mapa.")
    return dict(value)


def _validate_scenario_combat_config(
    scenario: CycleScenario,
    combat_plan_catalog: SimulatedCombatPlanCatalog,
    combat_profile_catalog: SimulatedCombatProfileCatalog,
) -> None:
    if scenario.combat_profile_name is not None:
        combat_profile_catalog.select_profile(scenario.combat_profile_name)
        return

    if scenario.combat_plan_name is not None:
        combat_plan_catalog.select_plan(plan_name=scenario.combat_plan_name)
        return

    if scenario.combat_plan_rounds is not None:
        combat_plan_catalog.select_plan(round_sequences=scenario.combat_plan_rounds)
        return

    if not scenario.combat_inputs:
        raise ValueError("combat_inputs nie moze byc puste, jesli nie ustawiono combat_plan_name.")
