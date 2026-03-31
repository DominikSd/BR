from __future__ import annotations

from botlab.adapters.simulation.runner import SimulatedCycleRuntime
from botlab.adapters.simulation.spawner import CycleScenario, SimulatedGroupState, SimulatedSpawner
from botlab.adapters.simulation.targeting import (
    SimulatedApproachWorldStateProvider,
    SimulatedInteractionWorldStateProvider,
    SimulatedObservationPreparationProvider,
    SimulatedTargetApproachProvider,
    SimulatedTargetInteractionProvider,
    SimulatedWorldStateProvider,
)
from botlab.application import TargetApproachResult, TargetResolution
from botlab.domain.targeting import RetargetDecision
from botlab.adapters.simulation.world import SimulatedWorld
from botlab.config import load_default_config
from botlab.domain.scheduler import CycleScheduler


def _runtime_for_scenario(cycle_id: int, scenario: CycleScenario) -> SimulatedCycleRuntime:
    settings = load_default_config()
    scheduler = CycleScheduler.from_cycle_config(settings.cycle)
    scheduler.bootstrap(anchor_spawn_ts=100.0, anchor_cycle_id=0)
    spawner = SimulatedSpawner(overrides={cycle_id: scenario})
    world = SimulatedWorld(settings.cycle)
    return SimulatedCycleRuntime(
        scheduler=scheduler,
        spawner=spawner,
        world=world,
    )


def test_world_state_provider_builds_visible_groups_with_computed_distances() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            spawn_zone_visible=True,
            bot_position_xy=(1.0, 1.0),
            groups=(
                SimulatedGroupState(group_id="g-1", position_xy=(4.0, 5.0), threat_score=0.5),
                SimulatedGroupState(group_id="g-2", position_xy=(2.0, 1.0), engaged_by_other=True),
            ),
            note="visible-zone",
        ),
    )
    provider = SimulatedWorldStateProvider(runtime)

    snapshot = provider.get_world_snapshot(1)

    assert snapshot.spawn_zone_visible is True
    assert snapshot.can_search_for_targets is True
    assert snapshot.metadata["visible_group_count"] == 2
    assert snapshot.group_by_id("g-1") is not None
    assert snapshot.group_by_id("g-1").distance == 5.0
    assert snapshot.group_by_id("g-2") is not None
    assert snapshot.group_by_id("g-2").engaged_by_other is True


def test_world_state_provider_hides_groups_when_spawn_zone_is_not_visible() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            spawn_zone_visible=False,
            bot_position_xy=(0.0, 0.0),
            groups=(
                SimulatedGroupState(group_id="g-1", position_xy=(3.0, 4.0)),
            ),
            note="hidden-zone",
        ),
    )
    provider = SimulatedWorldStateProvider(runtime)

    snapshot = provider.get_world_snapshot(1)

    assert snapshot.spawn_zone_visible is False
    assert snapshot.can_search_for_targets is False
    assert snapshot.groups == ()


def test_world_state_provider_creates_default_group_for_visible_spawn_event() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            has_event=True,
            spawn_zone_visible=True,
            bot_position_xy=(0.0, 0.0),
            groups=(),
        ),
    )
    provider = SimulatedWorldStateProvider(runtime)

    snapshot = provider.get_world_snapshot(1)

    assert len(snapshot.groups) == 1
    assert snapshot.groups[0].group_id == "group-1"
    assert snapshot.groups[0].distance == 5.0


def test_observation_preparation_provider_reports_spawn_zone_visibility() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            spawn_zone_visible=False,
            bot_position_xy=(3.0, 4.0),
            groups=(SimulatedGroupState(group_id="g-1", position_xy=(5.0, 5.0)),),
            note="need-reposition",
        ),
    )
    provider = SimulatedObservationPreparationProvider(runtime)

    result = provider.prepare_observation(1)

    assert result.cycle_id == 1
    assert result.spawn_zone_visible is False
    assert result.ready_for_observation is False
    assert result.note == "need-reposition"
    assert result.metadata["bot_position_xy"] == (3.0, 4.0)
    assert result.metadata["configured_group_count"] == 1


def test_target_approach_provider_computes_travel_time_for_selected_target() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            spawn_zone_visible=True,
            bot_position_xy=(0.0, 0.0),
            groups=(SimulatedGroupState(group_id="g-1", position_xy=(3.0, 4.0)),),
        ),
    )
    world_snapshot = SimulatedWorldStateProvider(runtime).get_world_snapshot(1)
    provider = SimulatedTargetApproachProvider(runtime, movement_speed_units_per_s=3.0)

    result = provider.approach_target(
        target_resolution=TargetResolution(
            cycle_id=1,
            current_target_id=None,
            selected_target_id="g-1",
            world_snapshot=world_snapshot,
            decision=RetargetDecision(
                current_target_id=None,
                selected_target=None,
                validation=None,
                changed=True,
                reason="selected_initial_target",
            ),
        )
    )

    assert result.target_id == "g-1"
    assert result.arrived is True
    assert result.travel_s == 1.5
    assert result.completed_at_ts == world_snapshot.observed_at_ts + 1.5
    assert result.initial_target_id == "g-1"
    assert result.metadata["travel_distance"] == 4.5


def test_target_approach_provider_skips_when_no_target_selected() -> None:
    runtime = _runtime_for_scenario(1, CycleScenario(spawn_zone_visible=True))
    world_snapshot = SimulatedWorldStateProvider(runtime).get_world_snapshot(1)
    provider = SimulatedTargetApproachProvider(runtime)

    result = provider.approach_target(
        target_resolution=TargetResolution(
            cycle_id=1,
            current_target_id=None,
            selected_target_id=None,
            world_snapshot=world_snapshot,
            decision=RetargetDecision(
                current_target_id=None,
                selected_target=None,
                validation=None,
                changed=False,
                reason="no_target_available",
            ),
        )
    )

    assert result.target_id is None
    assert result.arrived is False
    assert result.travel_s == 0.0
    assert result.reason == "no_target_selected"


def test_approach_world_state_provider_uses_approach_specific_groups_and_position() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            spawn_zone_visible=True,
            bot_position_xy=(0.0, 0.0),
            groups=(SimulatedGroupState(group_id="initial", position_xy=(4.0, 0.0)),),
            approach_revalidation_delay_s=0.4,
            approach_bot_position_xy=(2.0, 0.0),
            approach_groups=(
                SimulatedGroupState(group_id="initial", position_xy=(4.0, 0.0), engaged_by_other=True),
                SimulatedGroupState(group_id="replacement", position_xy=(5.0, 0.0)),
            ),
        ),
    )
    provider = SimulatedApproachWorldStateProvider(runtime)

    snapshot = provider.get_approach_world_snapshot(1)

    assert snapshot.metadata["phase"] == "approach_revalidation"
    assert snapshot.observed_at_ts == 145.4
    assert snapshot.bot_position.x == 2.0
    assert snapshot.group_by_id("initial") is not None
    assert snapshot.group_by_id("initial").engaged_by_other is True
    assert snapshot.group_by_id("replacement") is not None
    assert snapshot.group_by_id("replacement").distance == 3.0


def test_interaction_world_state_provider_uses_interaction_specific_groups_and_position() -> None:
    runtime = _runtime_for_scenario(
        1,
        CycleScenario(
            spawn_zone_visible=True,
            bot_position_xy=(0.0, 0.0),
            groups=(SimulatedGroupState(group_id="initial", position_xy=(4.0, 0.0)),),
            interaction_revalidation_delay_s=0.7,
            interaction_bot_position_xy=(3.0, 0.0),
            interaction_groups=(
                SimulatedGroupState(group_id="initial", position_xy=(4.0, 0.0), engaged_by_other=True),
                SimulatedGroupState(group_id="replacement", position_xy=(5.0, 0.0)),
            ),
        ),
    )
    provider = SimulatedInteractionWorldStateProvider(runtime)

    snapshot = provider.get_interaction_world_snapshot(1)

    assert snapshot.metadata["phase"] == "interaction_revalidation"
    assert snapshot.observed_at_ts == 145.7
    assert snapshot.bot_position.x == 3.0
    assert snapshot.group_by_id("initial") is not None
    assert snapshot.group_by_id("initial").engaged_by_other is True
    assert snapshot.group_by_id("replacement") is not None
    assert snapshot.group_by_id("replacement").distance == 2.0


def test_target_interaction_provider_marks_ready_after_successful_approach() -> None:
    provider = SimulatedTargetInteractionProvider()

    result = provider.prepare_interaction(
        target_approach_result=TargetApproachResult(
            cycle_id=1,
            target_id="g-1",
            started_at_ts=145.0,
            completed_at_ts=145.5,
            travel_s=0.5,
            arrived=True,
            reason="target_reached_in_simulation",
            initial_target_id="g-1",
        )
    )

    assert result.target_id == "g-1"
    assert result.ready is True
    assert result.reason == "interaction_ready"
