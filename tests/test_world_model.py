from __future__ import annotations

import pytest

from botlab.domain.world import GroupSnapshot, Position, TargetCandidate, WorldSnapshot


def test_position_distance_to_other_position() -> None:
    origin = Position(x=0.0, y=0.0)
    target = Position(x=3.0, y=4.0)

    assert origin.distance_to(target) == pytest.approx(5.0)


def test_group_snapshot_exposes_liveness_and_targetability() -> None:
    group = GroupSnapshot(
        group_id="g-1",
        position=Position(x=10.0, y=20.0),
        distance=7.5,
        alive_count=3,
        engaged_by_other=False,
        reachable=True,
        threat_score=0.25,
    )

    assert group.is_alive is True
    assert group.is_targetable is True


def test_group_snapshot_not_targetable_when_taken_or_unreachable_or_dead() -> None:
    taken_group = GroupSnapshot(
        group_id="taken",
        position=Position(x=1.0, y=1.0),
        distance=2.0,
        alive_count=2,
        engaged_by_other=True,
        reachable=True,
    )
    unreachable_group = GroupSnapshot(
        group_id="unreachable",
        position=Position(x=2.0, y=2.0),
        distance=5.0,
        alive_count=2,
        engaged_by_other=False,
        reachable=False,
    )
    dead_group = GroupSnapshot(
        group_id="dead",
        position=Position(x=3.0, y=3.0),
        distance=6.0,
        alive_count=0,
        engaged_by_other=False,
        reachable=True,
    )

    assert taken_group.is_targetable is False
    assert unreachable_group.is_targetable is False
    assert dead_group.is_targetable is False


def test_world_snapshot_can_find_group_by_id_and_filter_targetable_groups() -> None:
    targetable_group = GroupSnapshot(
        group_id="g-1",
        position=Position(x=5.0, y=6.0),
        distance=3.0,
        alive_count=4,
        engaged_by_other=False,
        reachable=True,
    )
    blocked_group = GroupSnapshot(
        group_id="g-2",
        position=Position(x=8.0, y=9.0),
        distance=10.0,
        alive_count=4,
        engaged_by_other=True,
        reachable=True,
    )
    world = WorldSnapshot(
        observed_at_ts=123.456,
        bot_position=Position(x=1.0, y=2.0),
        groups=(targetable_group, blocked_group),
        current_target_id="g-1",
    )

    assert world.group_by_id("g-1") == targetable_group
    assert world.group_by_id("missing") is None
    assert world.targetable_groups() == (targetable_group,)


def test_world_snapshot_reports_when_target_search_is_possible() -> None:
    visible_world = WorldSnapshot(
        observed_at_ts=50.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(),
        in_combat=False,
        spawn_zone_visible=True,
    )
    hidden_world = WorldSnapshot(
        observed_at_ts=50.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(),
        in_combat=False,
        spawn_zone_visible=False,
    )
    combat_world = WorldSnapshot(
        observed_at_ts=50.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(),
        in_combat=True,
        spawn_zone_visible=True,
    )

    assert visible_world.can_search_for_targets is True
    assert hidden_world.can_search_for_targets is False
    assert combat_world.can_search_for_targets is False


def test_target_candidate_marks_selectable_targets() -> None:
    candidate = TargetCandidate(
        group_id="g-1",
        score=0.9,
        reason="nearest_reachable_group",
        reachable=True,
        engaged_by_other=False,
        distance=2.5,
    )
    blocked_candidate = TargetCandidate(
        group_id="g-2",
        score=0.4,
        reason="already_taken",
        reachable=True,
        engaged_by_other=True,
        distance=1.5,
    )

    assert candidate.is_selectable is True
    assert blocked_candidate.is_selectable is False


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (
            lambda: Position(x=float("inf"), y=0.0),
            "x",
        ),
        (
            lambda: GroupSnapshot(
                group_id="",
                position=Position(x=1.0, y=1.0),
                distance=1.0,
                alive_count=1,
                engaged_by_other=False,
                reachable=True,
            ),
            "group_id",
        ),
        (
            lambda: GroupSnapshot(
                group_id="g-1",
                position=Position(x=1.0, y=1.0),
                distance=-1.0,
                alive_count=1,
                engaged_by_other=False,
                reachable=True,
            ),
            "distance",
        ),
        (
            lambda: WorldSnapshot(
                observed_at_ts=10.0,
                bot_position=Position(x=0.0, y=0.0),
                groups=(),
                current_target_id="",
            ),
            "current_target_id",
        ),
        (
            lambda: TargetCandidate(
                group_id="g-1",
                score=0.5,
                reason="",
                reachable=True,
                engaged_by_other=False,
                distance=1.0,
            ),
            "reason",
        ),
    ],
)
def test_world_model_rejects_invalid_values(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()
