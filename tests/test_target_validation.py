from __future__ import annotations

from botlab.domain.targeting import (
    TargetValidationPolicy,
    TargetValidationStatus,
)
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot


def _group(
    group_id: str,
    *,
    alive_count: int = 3,
    reachable: bool = True,
    engaged_by_other: bool = False,
    distance: float = 5.0,
) -> GroupSnapshot:
    return GroupSnapshot(
        group_id=group_id,
        position=Position(x=distance, y=0.0),
        distance=distance,
        alive_count=alive_count,
        engaged_by_other=engaged_by_other,
        reachable=reachable,
    )


def _world(*groups: GroupSnapshot) -> WorldSnapshot:
    return WorldSnapshot(
        observed_at_ts=100.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=groups,
    )


def test_validate_returns_valid_when_target_still_exists_and_is_available() -> None:
    policy = TargetValidationPolicy()
    world = _world(_group("g-1", alive_count=4, reachable=True, engaged_by_other=False, distance=3.5))

    result = policy.validate(world, "g-1")

    assert result.status is TargetValidationStatus.VALID
    assert result.can_continue is True
    assert result.reason == "target_still_valid"


def test_validate_returns_missing_when_target_disappears() -> None:
    policy = TargetValidationPolicy()
    world = _world(_group("g-2"))

    result = policy.validate(world, "g-1")

    assert result.status is TargetValidationStatus.MISSING
    assert result.can_continue is False


def test_validate_returns_defeated_when_target_has_no_alive_units() -> None:
    policy = TargetValidationPolicy()
    world = _world(_group("g-1", alive_count=0))

    result = policy.validate(world, "g-1")

    assert result.status is TargetValidationStatus.DEFEATED
    assert result.can_continue is False


def test_validate_returns_engaged_when_other_player_took_target() -> None:
    policy = TargetValidationPolicy()
    world = _world(_group("g-1", engaged_by_other=True))

    result = policy.validate(world, "g-1")

    assert result.status is TargetValidationStatus.ENGAGED_BY_OTHER
    assert result.can_continue is False


def test_validate_returns_unreachable_when_target_cannot_be_reached() -> None:
    policy = TargetValidationPolicy()
    world = _world(_group("g-1", reachable=False))

    result = policy.validate(world, "g-1")

    assert result.status is TargetValidationStatus.UNREACHABLE
    assert result.can_continue is False
