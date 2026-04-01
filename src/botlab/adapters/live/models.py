from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class LiveTargetDetection:
    target_id: str
    screen_x: int
    screen_y: int
    distance: float
    occupied: bool = False
    mob_variant: str = "mob_a"
    reachable: bool = True
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] | None = None
    orientation_deg: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LiveFrame:
    width: int
    height: int
    captured_at_ts: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    image: Any | None = None
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LiveResourceSnapshot:
    hp_ratio: float
    condition_ratio: float


@dataclass(slots=True, frozen=True)
class LiveStateSnapshot:
    in_combat: bool
    reward_visible: bool
    rest_available: bool


@dataclass(slots=True)
class LiveSessionState:
    hp_ratio: float = 1.0
    condition_ratio: float = 1.0


class LiveEngageOutcome(str, Enum):
    ENGAGED = "engaged"
    TARGET_STOLEN = "target_stolen"
    MISCLICK = "misclick"
    APPROACH_STALLED = "approach_stalled"
    APPROACH_TIMEOUT = "approach_timeout"
    NO_TARGET_AVAILABLE = "no_target_available"


@dataclass(slots=True, frozen=True)
class LiveEngageResult:
    cycle_id: int
    outcome: LiveEngageOutcome
    reason: str
    selected_target_id: str | None
    final_target_id: str | None
    click_screen_xy: tuple[int, int] | None
    started_at_ts: float
    completed_at_ts: float
    detection_latency_ms: float | None = None
    selection_latency_ms: float | None = None
    total_reaction_latency_ms: float | None = None
    verification_latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "selected_target_id": self.selected_target_id,
            "final_target_id": self.final_target_id,
            "click_screen_xy": None if self.click_screen_xy is None else list(self.click_screen_xy),
            "started_at_ts": self.started_at_ts,
            "completed_at_ts": self.completed_at_ts,
            "detection_latency_ms": self.detection_latency_ms,
            "selection_latency_ms": self.selection_latency_ms,
            "total_reaction_latency_ms": self.total_reaction_latency_ms,
            "verification_latency_ms": self.verification_latency_ms,
            "metadata": self.metadata,
            "artifact_paths": {key: str(path) for key, path in self.artifact_paths.items()},
        }
