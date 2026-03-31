from __future__ import annotations

from botlab.application import (
    TargetApproachResult,
    TargetEngagementService,
    TargetInteractionResult,
    TargetResolution,
)
from botlab.domain.targeting import RetargetDecision
from botlab.domain.world import Position, WorldSnapshot


def _resolution(
    *,
    cycle_id: int,
    target_id: str | None,
    observed_at_ts: float = 100.0,
) -> TargetResolution:
    return TargetResolution(
        cycle_id=cycle_id,
        current_target_id=None,
        selected_target_id=target_id,
        world_snapshot=WorldSnapshot(
            observed_at_ts=observed_at_ts,
            bot_position=Position(x=0.0, y=0.0),
            groups=(),
        ),
        decision=RetargetDecision(
            current_target_id=None,
            selected_target=None,
            validation=None,
            changed=target_id is not None,
            reason="selected_initial_target" if target_id is not None else "no_target_available",
        ),
    )


class StaticAcquisitionService:
    def __init__(self, resolution: TargetResolution) -> None:
        self._resolution = resolution

    def resolve_target(
        self,
        *,
        cycle_id: int,
        current_target_id: str | None = None,
    ) -> TargetResolution:
        assert cycle_id == self._resolution.cycle_id
        return self._resolution


class QueueApproachService:
    def __init__(self, results: list[TargetApproachResult]) -> None:
        self._results = list(results)
        self.calls: list[str | None] = []

    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        self.calls.append(target_resolution.selected_target_id)
        return self._results.pop(0)


class QueueInteractionService:
    def __init__(self, results: list[TargetInteractionResult]) -> None:
        self._results = list(results)
        self.calls: list[str | None] = []

    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        self.calls.append(target_approach_result.target_id)
        return self._results.pop(0)


def test_engage_target_returns_ready_result_without_retry() -> None:
    acquisition = StaticAcquisitionService(_resolution(cycle_id=9, target_id="current"))
    approach = QueueApproachService(
        [
            TargetApproachResult(
                cycle_id=9,
                target_id="current",
                started_at_ts=100.0,
                completed_at_ts=100.5,
                travel_s=0.5,
                arrived=True,
                reason="target_reached_in_simulation",
                initial_target_id="current",
            )
        ]
    )
    interaction = QueueInteractionService(
        [
            TargetInteractionResult(
                cycle_id=9,
                target_id="current",
                ready=True,
                observed_at_ts=100.5,
                reason="interaction_ready",
                initial_target_id="current",
            )
        ]
    )
    service = TargetEngagementService(
        acquisition_service=acquisition,
        approach_service=approach,
        interaction_service=interaction,
    )

    result = service.engage_target(cycle_id=9)

    assert result.target_resolution.selected_target_id == "current"
    assert result.approach_result.target_id == "current"
    assert result.interaction_result.target_id == "current"
    assert result.interaction_result.ready is True
    assert approach.calls == ["current"]
    assert interaction.calls == ["current"]


def test_engage_target_retries_approach_after_retarget_before_interaction() -> None:
    acquisition = StaticAcquisitionService(_resolution(cycle_id=9, target_id="current"))
    approach = QueueApproachService(
        [
            TargetApproachResult(
                cycle_id=9,
                target_id="current",
                started_at_ts=100.0,
                completed_at_ts=100.5,
                travel_s=0.5,
                arrived=True,
                reason="target_reached_in_simulation",
                initial_target_id="current",
            ),
            TargetApproachResult(
                cycle_id=9,
                target_id="replacement",
                started_at_ts=100.6,
                completed_at_ts=101.0,
                travel_s=0.4,
                arrived=True,
                reason="target_reached_in_simulation",
                initial_target_id="replacement",
            ),
        ]
    )
    retry_world = WorldSnapshot(
        observed_at_ts=100.6,
        bot_position=Position(x=1.0, y=0.0),
        groups=(),
    )
    interaction = QueueInteractionService(
        [
            TargetInteractionResult(
                cycle_id=9,
                target_id="replacement",
                ready=False,
                observed_at_ts=100.6,
                reason="retarget_before_interaction_requires_new_approach",
                initial_target_id="current",
                retargeted=True,
                world_snapshot=retry_world,
                decision=RetargetDecision(
                    current_target_id="current",
                    selected_target=None,
                    validation=None,
                    changed=True,
                    reason="current_target_invalid_retargeted",
                ),
            ),
            TargetInteractionResult(
                cycle_id=9,
                target_id="replacement",
                ready=True,
                observed_at_ts=101.0,
                reason="interaction_ready",
                initial_target_id="replacement",
            ),
        ]
    )
    service = TargetEngagementService(
        acquisition_service=acquisition,
        approach_service=approach,
        interaction_service=interaction,
    )

    result = service.engage_target(cycle_id=9)

    assert result.target_resolution.selected_target_id == "replacement"
    assert result.approach_result.target_id == "replacement"
    assert result.interaction_result.target_id == "replacement"
    assert result.interaction_result.ready is True
    assert result.interaction_result.initial_target_id == "current"
    assert result.interaction_result.retargeted is True
    assert approach.calls == ["current", "replacement"]
    assert interaction.calls == ["current", "replacement"]
