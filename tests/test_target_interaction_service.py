from __future__ import annotations

from botlab.application import (
    TargetApproachResult,
    TargetInteractionResult,
    TargetInteractionService,
)
from botlab.domain.targeting import RetargetPolicy, TargetSelectionPolicy, TargetValidationPolicy
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot


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


class StaticInteractionWorldStateProvider:
    def __init__(self, world_snapshot: WorldSnapshot) -> None:
        self._world_snapshot = world_snapshot

    def get_interaction_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        assert cycle_id == 7
        return self._world_snapshot


class RecordingTargetInteractionProvider:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        self.calls.append(target_approach_result.target_id)
        return TargetInteractionResult(
            cycle_id=target_approach_result.cycle_id,
            target_id=target_approach_result.target_id,
            ready=target_approach_result.target_id is not None,
            observed_at_ts=target_approach_result.completed_at_ts,
            reason=(
                "interaction_ready"
                if target_approach_result.target_id is not None
                else "interaction_not_ready"
            ),
        )


def _policy() -> RetargetPolicy:
    return RetargetPolicy(
        selection_policy=TargetSelectionPolicy(),
        validation_policy=TargetValidationPolicy(),
    )


def test_prepare_interaction_keeps_valid_target() -> None:
    provider = RecordingTargetInteractionProvider()
    world = _world(100.5, _group("current", distance=1.0), current_target_id="current")
    service = TargetInteractionService(
        target_interaction_provider=provider,
        interaction_world_state_provider=StaticInteractionWorldStateProvider(world),
        retarget_policy=_policy(),
    )

    result = service.prepare_interaction(
        TargetApproachResult(
            cycle_id=7,
            target_id="current",
            started_at_ts=100.0,
            completed_at_ts=100.5,
            travel_s=0.5,
            arrived=True,
            reason="target_reached_in_simulation",
            initial_target_id="current",
        )
    )

    assert result.target_id == "current"
    assert result.ready is True
    assert result.retargeted is False
    assert result.metadata["revalidation_reason"] == "current_target_still_valid"
    assert provider.calls == ["current"]


def test_prepare_interaction_retargets_when_target_is_taken_before_interaction() -> None:
    provider = RecordingTargetInteractionProvider()
    world = _world(
        100.6,
        _group("current", distance=1.0, engaged_by_other=True),
        _group("replacement", distance=1.5),
        current_target_id="current",
    )
    service = TargetInteractionService(
        target_interaction_provider=provider,
        interaction_world_state_provider=StaticInteractionWorldStateProvider(world),
        retarget_policy=_policy(),
    )

    result = service.prepare_interaction(
        TargetApproachResult(
            cycle_id=7,
            target_id="current",
            started_at_ts=100.0,
            completed_at_ts=100.5,
            travel_s=0.5,
            arrived=True,
            reason="target_reached_in_simulation",
            initial_target_id="current",
        )
    )

    assert result.target_id == "replacement"
    assert result.ready is False
    assert result.retargeted is True
    assert result.initial_target_id == "current"
    assert result.reason == "retarget_before_interaction_requires_new_approach"
    assert result.metadata["revalidation_reason"] == "current_target_invalid_retargeted"
    assert result.metadata["validation_reason"] == "target_engaged_by_other_player"
    assert provider.calls == ["current"]


def test_prepare_interaction_returns_no_target_when_lost_without_replacement() -> None:
    provider = RecordingTargetInteractionProvider()
    world = _world(
        100.6,
        _group("current", distance=1.0, reachable=False),
        current_target_id="current",
    )
    service = TargetInteractionService(
        target_interaction_provider=provider,
        interaction_world_state_provider=StaticInteractionWorldStateProvider(world),
        retarget_policy=_policy(),
    )

    result = service.prepare_interaction(
        TargetApproachResult(
            cycle_id=7,
            target_id="current",
            started_at_ts=100.0,
            completed_at_ts=100.5,
            travel_s=0.5,
            arrived=True,
            reason="target_reached_in_simulation",
            initial_target_id="current",
        )
    )

    assert result.target_id is None
    assert result.ready is False
    assert result.reason == "target_lost_before_interaction_no_replacement"
    assert result.metadata["revalidation_reason"] == "current_target_invalid_and_no_replacement"
    assert result.metadata["validation_reason"] == "target_not_reachable"
    assert provider.calls == ["current"]
