from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from botlab.types import BotState, Observation


class VerificationOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ActionContext:
    cycle_id: int
    now_ts: float
    predicted_spawn_ts: float | None = None
    observation: Observation | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActionResult:
    cycle_id: int
    success: bool
    reason: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CombatOutcome:
    cycle_id: int
    won: bool
    hp_ratio: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RestOutcome:
    cycle_id: int
    hp_ratio: float
    recovered: bool
    metadata: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class CycleRunResult:
    cycle_id: int
    predicted_spawn_ts: float
    actual_spawn_ts: float | None
    drift_s: float | None
    result: str
    final_state: BotState
    reaction_ms: float | None
    verification_ms: float | None
    observation_used: bool
    note: str


@dataclass(slots=True, frozen=True)
class SimulationReport:
    cycle_results: list[CycleRunResult]
    log_path: Path
    sqlite_path: Path

    @property
    def total_cycles(self) -> int:
        return len(self.cycle_results)

    def count_result(self, result: str) -> int:
        return sum(1 for item in self.cycle_results if item.result == result)
