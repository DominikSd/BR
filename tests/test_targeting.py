from __future__ import annotations

import pytest

from botlab.domain.targeting import TargetSelectionPolicy
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot


def _group(
    group_id: str,
    *,
    distance: float,
    threat_score: float = 0.0,
    alive_count: int = 3,
    reachable: bool = True,
    engaged_by_other: bool = False,
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
        observed_at_ts=123.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=groups,
        current_target_id=current_target_id,
    )


def test_select_best_target_returns_none_when_no_targetable_groups() -> None:
    policy = TargetSelectionPolicy()
    world = _world(
        _group("dead", distance=2.0, alive_count=0),
        _group("taken", distance=1.0, engaged_by_other=True),
        _group("blocked", distance=3.0, reachable=False),
    )

    assert policy.select_best_target(world) is None


def test_target_selection_policy_defaults_to_nearest_free_without_biases() -> None:
    policy = TargetSelectionPolicy()

    assert policy.threat_weight == 0.0
    assert policy.current_target_bonus == 0.0


def test_select_best_target_prefers_nearest_reachable_group() -> None:
    policy = TargetSelectionPolicy(threat_weight=0.0, current_target_bonus=0.0)
    world = _world(
        _group("far", distance=10.0),
        _group("near", distance=3.0),
        _group("mid", distance=5.0),
    )

    candidate = policy.select_best_target(world)

    assert candidate is not None
    assert candidate.group_id == "near"
    assert candidate.reason == "best_effective_distance"
    assert candidate.distance == pytest.approx(3.0)


def test_select_best_target_can_prefer_lower_threat_when_weighted() -> None:
    policy = TargetSelectionPolicy(threat_weight=2.0, current_target_bonus=0.0)
    world = _world(
        _group("risky", distance=4.0, threat_score=4.0),
        _group("safer", distance=5.0, threat_score=0.5),
    )

    candidate = policy.select_best_target(world)

    assert candidate is not None
    assert candidate.group_id == "safer"
    assert candidate.metadata["effective_distance"] == pytest.approx(6.0)


def test_select_best_target_applies_small_bonus_for_current_target() -> None:
    policy = TargetSelectionPolicy(threat_weight=0.0, current_target_bonus=0.5)
    world = _world(
        _group("current", distance=5.0),
        _group("new", distance=4.8),
        current_target_id="current",
    )

    candidate = policy.select_best_target(world)

    assert candidate is not None
    assert candidate.group_id == "current"
    assert candidate.metadata["current_target_bonus_applied"] is True


def test_select_best_target_breaks_ties_by_group_id_for_stability() -> None:
    policy = TargetSelectionPolicy(threat_weight=0.0, current_target_bonus=0.0)
    world = _world(
        _group("b-group", distance=4.0),
        _group("a-group", distance=4.0),
    )

    candidate = policy.select_best_target(world)

    assert candidate is not None
    assert candidate.group_id == "a-group"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threat_weight": -1.0}, "threat_weight"),
        ({"current_target_bonus": -0.1}, "current_target_bonus"),
        ({"threat_weight": float("inf")}, "threat_weight"),
        ({"current_target_bonus": float("nan")}, "current_target_bonus"),
    ],
)
def test_target_selection_policy_rejects_invalid_configuration(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TargetSelectionPolicy(**kwargs)
