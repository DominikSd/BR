from __future__ import annotations

from botlab.application import (
    TargetApproachResult,
    TargetApproachService,
    TargetResolution,
)
from botlab.domain.targeting import (
    RetargetDecision,
    RetargetPolicy,
    TargetSelectionPolicy,
    TargetValidationPolicy,
)
from botlab.domain.world import GroupSnapshot, Position, TargetCandidate, WorldSnapshot


def _group(
    group_id: str,
    *,
    distance: float,
    reachable: bool = True,
    engaged_by_other: bool = False,
) -> GroupSnapshot:
    return GroupSnapshot(
        group_id=group_id,
        position=Position(x=distance, y=0.0),
        distance=distance,
        alive_count=3,
        engaged_by_other=engaged_by_other,
        reachable=reachable,
    )


def _world(
    observed_at_ts: float,
    *groups: GroupSnapshot,
    current_target_id: str | None = None,
) -> WorldSnapshot:
    return WorldSnapshot(
        observed_at_ts=observed_at_ts,
        bot_position=Position(x=0.0, y=0.0),
        groups=groups,
        current_target_id=current_target_id,
    )


def _resolution(world_snapshot: WorldSnapshot, selected_target_id: str | None) -> TargetResolution:
    selected_target = None
    if selected_target_id is not None:
        group = world_snapshot.group_by_id(selected_target_id)
        assert group is not None
        selected_target = TargetCandidate(
            group_id=group.group_id,
            score=1.0,
            reason="selected_initial_target",
            reachable=group.reachable,
            engaged_by_other=group.engaged_by_other,
            distance=group.distance,
        )

    return TargetResolution(
        cycle_id=5,
        current_target_id=None,
        selected_target_id=selected_target_id,
        world_snapshot=world_snapshot,
        decision=RetargetDecision(
            current_target_id=None,
            selected_target=selected_target,
            validation=None,
            changed=selected_target_id is not None,
            reason="selected_initial_target" if selected_target_id is not None else "no_target_available",
        ),
    )


class RecordingTargetApproachProvider:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        self.calls.append(target_resolution.selected_target_id)
        world = target_resolution.world_snapshot
        target_id = target_resolution.selected_target_id
        if target_id is None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=None,
                started_at_ts=world.observed_at_ts,
                completed_at_ts=world.observed_at_ts,
                travel_s=0.0,
                arrived=False,
                reason="no_target_selected",
            )

        group = world.group_by_id(target_id)
        assert group is not None
        return TargetApproachResult(
            cycle_id=target_resolution.cycle_id,
            target_id=target_id,
            started_at_ts=world.observed_at_ts,
            completed_at_ts=world.observed_at_ts + group.distance,
            travel_s=group.distance,
            arrived=True,
            reason="target_reached",
            metadata={"distance": group.distance},
        )


class StaticApproachWorldStateProvider:
    def __init__(self, world_snapshot: WorldSnapshot) -> None:
        self._world_snapshot = world_snapshot

    def get_approach_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        assert cycle_id == 5
        return self._world_snapshot


def _policy() -> RetargetPolicy:
    return RetargetPolicy(
        selection_policy=TargetSelectionPolicy(),
        validation_policy=TargetValidationPolicy(),
    )


def test_approach_target_returns_provider_result_without_revalidation() -> None:
    provider = RecordingTargetApproachProvider()
    initial_world = _world(100.0, _group("group-1", distance=2.0))
    service = TargetApproachService(target_approach_provider=provider)

    result = service.approach_target(_resolution(initial_world, "group-1"))

    assert result.target_id == "group-1"
    assert result.arrived is True
    assert result.initial_target_id == "group-1"
    assert result.retargeted is False
    assert provider.calls == ["group-1"]


def test_approach_target_retargets_immediately_when_current_target_becomes_unavailable() -> None:
    provider = RecordingTargetApproachProvider()
    initial_world = _world(
        100.0,
        _group("current", distance=2.0),
        _group("replacement", distance=5.0),
    )
    approach_world = _world(
        100.25,
        _group("current", distance=1.0, engaged_by_other=True),
        _group("replacement", distance=3.0),
        current_target_id="current",
    )
    service = TargetApproachService(
        target_approach_provider=provider,
        approach_world_state_provider=StaticApproachWorldStateProvider(approach_world),
        retarget_policy=_policy(),
    )

    result = service.approach_target(_resolution(initial_world, "current"))

    assert result.target_id == "replacement"
    assert result.arrived is True
    assert result.initial_target_id == "current"
    assert result.retargeted is True
    assert result.metadata["revalidation_reason"] == "current_target_invalid_retargeted"
    assert result.metadata["validation_reason"] == "target_engaged_by_other_player"
    assert provider.calls == ["current", "replacement"]


def test_approach_target_stops_immediately_when_lost_and_no_replacement_exists() -> None:
    provider = RecordingTargetApproachProvider()
    initial_world = _world(100.0, _group("current", distance=2.0))
    approach_world = _world(
        100.25,
        _group("current", distance=1.0, reachable=False),
        current_target_id="current",
    )
    service = TargetApproachService(
        target_approach_provider=provider,
        approach_world_state_provider=StaticApproachWorldStateProvider(approach_world),
        retarget_policy=_policy(),
    )

    result = service.approach_target(_resolution(initial_world, "current"))

    assert result.target_id is None
    assert result.arrived is False
    assert result.initial_target_id == "current"
    assert result.retargeted is False
    assert result.reason == "target_lost_during_approach_no_replacement"
    assert result.metadata["revalidation_reason"] == "current_target_invalid_and_no_replacement"
    assert result.metadata["validation_reason"] == "target_not_reachable"
    assert provider.calls == ["current"]
