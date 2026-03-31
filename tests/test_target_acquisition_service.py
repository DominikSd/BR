from __future__ import annotations

from botlab.application import TargetAcquisitionService
from botlab.domain.targeting import RetargetPolicy, TargetSelectionPolicy, TargetValidationPolicy
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


class StaticWorldStateProvider:
    def __init__(self, world_snapshot: WorldSnapshot) -> None:
        self._world_snapshot = world_snapshot

    def get_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        return self._world_snapshot


def _service(world_snapshot: WorldSnapshot) -> TargetAcquisitionService:
    return TargetAcquisitionService(
        world_state_provider=StaticWorldStateProvider(world_snapshot),
        retarget_policy=RetargetPolicy(
            selection_policy=TargetSelectionPolicy(),
            validation_policy=TargetValidationPolicy(),
        ),
    )


def test_resolve_target_selects_initial_target_when_none_exists() -> None:
    world = WorldSnapshot(
        observed_at_ts=10.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(
            _group("far", distance=10.0),
            _group("near", distance=2.0),
        ),
    )

    resolution = _service(world).resolve_target(cycle_id=7, current_target_id=None)

    assert resolution.cycle_id == 7
    assert resolution.current_target_id is None
    assert resolution.selected_target_id == "near"
    assert resolution.decision.reason == "selected_initial_target"


def test_resolve_target_keeps_current_target_when_still_valid() -> None:
    world = WorldSnapshot(
        observed_at_ts=10.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(
            _group("current", distance=4.0),
            _group("other", distance=2.0),
        ),
        current_target_id="current",
    )

    resolution = _service(world).resolve_target(cycle_id=7, current_target_id="current")

    assert resolution.selected_target_id == "current"
    assert resolution.decision.changed is False
    assert resolution.decision.reason == "current_target_still_valid"


def test_resolve_target_retargets_when_current_target_becomes_invalid() -> None:
    world = WorldSnapshot(
        observed_at_ts=10.0,
        bot_position=Position(x=0.0, y=0.0),
        groups=(
            _group("current", distance=2.0, engaged_by_other=True),
            _group("replacement", distance=4.0),
        ),
        current_target_id="current",
    )

    resolution = _service(world).resolve_target(cycle_id=7, current_target_id="current")

    assert resolution.selected_target_id == "replacement"
    assert resolution.decision.changed is True
    assert resolution.decision.reason == "current_target_invalid_retargeted"
