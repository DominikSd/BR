from __future__ import annotations

from dataclasses import dataclass, field
from math import dist, isfinite
from typing import Any


@dataclass(slots=True, frozen=True)
class Position:
    x: float
    y: float

    def __post_init__(self) -> None:
        _require_finite(self.x, "x")
        _require_finite(self.y, "y")

    def distance_to(self, other: "Position") -> float:
        return dist((self.x, self.y), (other.x, other.y))


@dataclass(slots=True, frozen=True)
class GroupSnapshot:
    group_id: str
    position: Position
    distance: float
    alive_count: int
    engaged_by_other: bool
    reachable: bool
    threat_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError("group_id musi być niepusty.")
        if self.distance < 0.0:
            raise ValueError("distance nie może być ujemny.")
        if self.alive_count < 0:
            raise ValueError("alive_count nie może być ujemny.")
        _require_finite(self.distance, "distance")
        _require_finite(self.threat_score, "threat_score")

    @property
    def is_alive(self) -> bool:
        return self.alive_count > 0

    @property
    def is_targetable(self) -> bool:
        return self.is_alive and self.reachable and not self.engaged_by_other


@dataclass(slots=True, frozen=True)
class WorldSnapshot:
    observed_at_ts: float
    bot_position: Position
    groups: tuple[GroupSnapshot, ...]
    in_combat: bool = False
    current_target_id: str | None = None
    spawn_zone_visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_finite(self.observed_at_ts, "observed_at_ts")
        if self.current_target_id is not None and not self.current_target_id.strip():
            raise ValueError("current_target_id nie może być pusty.")

    def group_by_id(self, group_id: str) -> GroupSnapshot | None:
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None

    def targetable_groups(self) -> tuple[GroupSnapshot, ...]:
        return tuple(group for group in self.groups if group.is_targetable)

    @property
    def can_search_for_targets(self) -> bool:
        return self.spawn_zone_visible and not self.in_combat


@dataclass(slots=True, frozen=True)
class TargetCandidate:
    group_id: str
    score: float
    reason: str
    reachable: bool
    engaged_by_other: bool
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise ValueError("group_id musi być niepusty.")
        if not self.reason.strip():
            raise ValueError("reason musi być niepusty.")
        if self.distance < 0.0:
            raise ValueError("distance nie może być ujemny.")
        _require_finite(self.score, "score")
        _require_finite(self.distance, "distance")

    @property
    def is_selectable(self) -> bool:
        return self.reachable and not self.engaged_by_other


def _require_finite(value: float, field_name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{field_name} musi być liczbą skończoną.")
