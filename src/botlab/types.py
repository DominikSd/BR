from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BotState(str, Enum):
    IDLE = "IDLE"
    PREPARE_WINDOW = "PREPARE_WINDOW"
    READY_WINDOW = "READY_WINDOW"
    ATTEMPT = "ATTEMPT"
    VERIFY = "VERIFY"
    COMBAT = "COMBAT"
    REST = "REST"
    RECOVER = "RECOVER"
    WAIT_NEXT_CYCLE = "WAIT_NEXT_CYCLE"


@dataclass(slots=True, frozen=True)
class CyclePrediction:
    cycle_id: int
    predicted_spawn_ts: float
    interval_s: float
    prepare_window_start_ts: float
    ready_window_start_ts: float
    ready_window_end_ts: float
    based_on_observation_count: int = 0

    def is_in_prepare_window(self, now_ts: float) -> bool:
        return self.prepare_window_start_ts <= now_ts < self.ready_window_start_ts

    def is_in_ready_window(self, now_ts: float) -> bool:
        return self.ready_window_start_ts <= now_ts <= self.ready_window_end_ts


@dataclass(slots=True, frozen=True)
class Observation:
    cycle_id: int
    observed_at_ts: float
    signal_detected: bool
    actual_spawn_ts: float | None = None
    source: str = "simulation"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Decision:
    cycle_id: int | None
    state: BotState
    next_state: BotState
    action: str
    reason: str
    decided_at_ts: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CombatSnapshot:
    hp_ratio: float
    turn_index: int
    enemy_count: int
    strategy: str
    in_combat: bool
    combat_started_ts: float | None = None
    combat_finished_ts: float | None = None
    condition_ratio: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TelemetryRecord:
    cycle_id: int | None
    event_ts: float
    state: BotState
    expected_spawn_ts: float | None = None
    actual_spawn_ts: float | None = None
    drift_s: float | None = None
    state_enter: BotState | None = None
    state_exit: BotState | None = None
    reason: str = ""
    reaction_ms: float | None = None
    verification_ms: float | None = None
    result: str | None = None
    final_state: BotState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
