from __future__ import annotations

from botlab.domain.targeting import (
    RetargetPolicy,
    TargetSelectionPolicy,
    TargetValidationPolicy,
    TargetValidationStatus,
)
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot


def _group(
    group_id: str,
    *,
    distance: float,
    alive_count: int = 3,
    reachable: bool = True,
    engaged_by_other: bool = False,
    threat_score: float = 0.0,
) -> GroupSnapshot:
    return GroupSnapshot(
        group_id=group_id,
        position=Position(x=distance, y=0.0),
        distance=distance,
        alive_count=alive_count,
        engaged_by_other=engaged_by_other,
        reachable=reachable,
        threat_score=threat_score,
    )


def _world(*groups: GroupSnapshot, current_target_id: str | None = None) -> WorldSnapshot:
    return WorldSnapshot(
        observed_at_ts=100.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=groups,
        current_target_id=current_target_id,
    )


def _policy() -> RetargetPolicy:
    return RetargetPolicy(
        selection_policy=TargetSelectionPolicy(),
        validation_policy=TargetValidationPolicy(),
    )


def test_resolve_keeps_current_target_when_it_is_still_valid() -> None:
    world = _world(
        _group("current", distance=5.0),
        _group("other", distance=3.0),
        current_target_id="current",
    )

    decision = _policy().resolve(world, "current")

    assert decision.changed is False
    assert decision.reason == "current_target_still_valid"
    assert decision.selected_target is not None
    assert decision.selected_target.group_id == "current"
    assert decision.validation is not None
    assert decision.validation.status is TargetValidationStatus.VALID


def test_resolve_retargets_when_current_target_is_taken() -> None:
    world = _world(
        _group("current", distance=2.0, engaged_by_other=True),
        _group("replacement", distance=4.0),
        current_target_id="current",
    )

    decision = _policy().resolve(world, "current")

    assert decision.changed is True
    assert decision.reason == "current_target_invalid_retargeted"
    assert decision.selected_target is not None
    assert decision.selected_target.group_id == "replacement"
    assert decision.validation is not None
    assert decision.validation.status is TargetValidationStatus.ENGAGED_BY_OTHER


def test_resolve_clears_target_when_current_is_invalid_and_no_replacement_exists() -> None:
    world = _world(
        _group("current", distance=2.0, alive_count=0),
        _group("blocked", distance=3.0, reachable=False),
        current_target_id="current",
    )

    decision = _policy().resolve(world, "current")

    assert decision.changed is True
    assert decision.reason == "current_target_invalid_and_no_replacement"
    assert decision.selected_target is None
    assert decision.validation is not None
    assert decision.validation.status is TargetValidationStatus.DEFEATED


def test_resolve_selects_initial_target_when_no_current_target_exists() -> None:
    world = _world(
        _group("far", distance=8.0),
        _group("near", distance=3.0),
    )

    decision = _policy().resolve(world, None)

    assert decision.changed is True
    assert decision.reason == "selected_initial_target"
    assert decision.validation is None
    assert decision.selected_target is not None
    assert decision.selected_target.group_id == "near"


def test_resolve_returns_no_target_when_world_has_no_targetable_groups() -> None:
    world = _world(
        _group("dead", distance=2.0, alive_count=0),
        _group("taken", distance=1.0, engaged_by_other=True),
    )

    decision = _policy().resolve(world, None)

    assert decision.changed is False
    assert decision.reason == "no_target_available"
    assert decision.selected_target is None
    assert decision.validation is None
