from __future__ import annotations

from dataclasses import dataclass, field
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
