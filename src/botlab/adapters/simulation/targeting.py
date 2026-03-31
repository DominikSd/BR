from __future__ import annotations

from typing import TYPE_CHECKING

from botlab.adapters.simulation.spawner import SimulatedGroupState
from botlab.application.dto import (
    ObservationPreparationResult,
    TargetApproachResult,
    TargetInteractionResult,
    TargetResolution,
)
from botlab.domain.world import GroupSnapshot, Position, WorldSnapshot

if TYPE_CHECKING:
    from botlab.adapters.simulation.runner import SimulatedCycleRuntime


class SimulatedObservationPreparationProvider:
    """Zwraca prosty wynik ustawienia bota do obserwacji strefy spawnu."""

    def __init__(self, runtime: SimulatedCycleRuntime) -> None:
        self._runtime = runtime

    def prepare_observation(self, cycle_id: int) -> ObservationPreparationResult:
        context = self._runtime.context_for_cycle(cycle_id)
        scenario = context.spawn_event.scenario
        return ObservationPreparationResult(
            cycle_id=cycle_id,
            spawn_zone_visible=scenario.spawn_zone_visible,
            ready_for_observation=scenario.spawn_zone_visible,
            note=scenario.note,
            metadata={
                "bot_position_xy": scenario.bot_position_xy,
                "configured_group_count": len(scenario.groups),
            },
        )


class SimulatedWorldStateProvider:
    """Buduje domenowy WorldSnapshot z kontekstu cyklu symulacyjnego."""

    def __init__(self, runtime: SimulatedCycleRuntime) -> None:
        self._runtime = runtime

    def get_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        context = self._runtime.context_for_cycle(cycle_id)
        scenario = context.spawn_event.scenario
        observed_at_ts = (
            context.spawn_event.actual_spawn_ts
            if context.spawn_event.actual_spawn_ts is not None
            else context.prediction.ready_window_start_ts
        )
        return self._build_snapshot(
            cycle_id=cycle_id,
            bot_position_xy=scenario.bot_position_xy,
            groups=self._resolve_groups(cycle_id),
            observed_at_ts=observed_at_ts,
            current_target_id=scenario.current_target_id,
            spawn_zone_visible=scenario.spawn_zone_visible,
            metadata={
                "cycle_id": cycle_id,
                "scenario_note": scenario.note,
                "visible_group_count": len(self._resolve_groups(cycle_id))
                if scenario.spawn_zone_visible
                else 0,
                "phase": "acquire",
            },
        )

    def _resolve_groups(self, cycle_id: int) -> tuple[SimulatedGroupState, ...]:
        scenario = self._runtime.context_for_cycle(cycle_id).spawn_event.scenario
        if scenario.groups:
            return scenario.groups

        if not scenario.has_event:
            return ()

        return (
            SimulatedGroupState(
                group_id=f"group-{cycle_id}",
                position_xy=(5.0, 0.0),
            ),
        )

    def _build_group_snapshot(
        self,
        *,
        bot_position: Position,
        group: SimulatedGroupState,
    ) -> GroupSnapshot:
        group_position = Position(*group.position_xy)
        return GroupSnapshot(
            group_id=group.group_id,
            position=group_position,
            distance=bot_position.distance_to(group_position),
            alive_count=group.alive_count,
            engaged_by_other=group.engaged_by_other,
            reachable=group.reachable,
            threat_score=group.threat_score,
            metadata=dict(group.metadata),
        )

    def _build_snapshot(
        self,
        *,
        cycle_id: int,
        bot_position_xy: tuple[float, float],
        groups: tuple[SimulatedGroupState, ...],
        observed_at_ts: float,
        current_target_id: str | None,
        spawn_zone_visible: bool,
        metadata: dict[str, object],
    ) -> WorldSnapshot:
        bot_position = Position(*bot_position_xy)
        visible_groups: tuple[GroupSnapshot, ...]
        if spawn_zone_visible:
            visible_groups = tuple(
                self._build_group_snapshot(bot_position=bot_position, group=group)
                for group in groups
            )
        else:
            visible_groups = ()

        return WorldSnapshot(
            observed_at_ts=observed_at_ts,
            bot_position=bot_position,
            groups=visible_groups,
            in_combat=False,
            current_target_id=current_target_id,
            spawn_zone_visible=spawn_zone_visible,
            metadata=metadata,
        )


class SimulatedApproachWorldStateProvider(SimulatedWorldStateProvider):
    """Buduje snapshot swiata dla rewalidacji celu podczas dojscia."""

    def get_approach_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        context = self._runtime.context_for_cycle(cycle_id)
        scenario = context.spawn_event.scenario
        base_observed_at_ts = (
            context.spawn_event.actual_spawn_ts
            if context.spawn_event.actual_spawn_ts is not None
            else context.prediction.ready_window_start_ts
        )
        groups = scenario.approach_groups
        if groups is None:
            groups = self._resolve_groups(cycle_id)

        bot_position_xy = (
            scenario.bot_position_xy
            if scenario.approach_bot_position_xy is None
            else scenario.approach_bot_position_xy
        )
        return self._build_snapshot(
            cycle_id=cycle_id,
            bot_position_xy=bot_position_xy,
            groups=groups,
            observed_at_ts=base_observed_at_ts + scenario.approach_revalidation_delay_s,
            current_target_id=scenario.current_target_id,
            spawn_zone_visible=scenario.spawn_zone_visible,
            metadata={
                "cycle_id": cycle_id,
                "scenario_note": scenario.note,
                "visible_group_count": len(groups) if scenario.spawn_zone_visible else 0,
                "phase": "approach_revalidation",
            },
        )


class SimulatedInteractionWorldStateProvider(SimulatedWorldStateProvider):
    """Buduje snapshot swiata do końcowej walidacji tuż przed interakcją."""

    def get_interaction_world_snapshot(self, cycle_id: int) -> WorldSnapshot:
        context = self._runtime.context_for_cycle(cycle_id)
        scenario = context.spawn_event.scenario
        base_observed_at_ts = (
            context.spawn_event.actual_spawn_ts
            if context.spawn_event.actual_spawn_ts is not None
            else context.prediction.ready_window_start_ts
        )
        groups = (
            self._resolve_groups(cycle_id)
            if scenario.interaction_groups is None
            else scenario.interaction_groups
        )
        bot_position_xy = (
            scenario.bot_position_xy
            if scenario.interaction_bot_position_xy is None
            else scenario.interaction_bot_position_xy
        )
        return self._build_snapshot(
            cycle_id=cycle_id,
            bot_position_xy=bot_position_xy,
            groups=groups,
            observed_at_ts=base_observed_at_ts + scenario.interaction_revalidation_delay_s,
            current_target_id=scenario.current_target_id,
            spawn_zone_visible=scenario.spawn_zone_visible,
            metadata={
                "cycle_id": cycle_id,
                "scenario_note": scenario.note,
                "visible_group_count": len(groups) if scenario.spawn_zone_visible else 0,
                "phase": "interaction_revalidation",
            },
        )


class SimulatedTargetApproachProvider:
    """Liczy prosty, deterministyczny czas dojścia do wybranego targetu."""

    def __init__(
        self,
        runtime: SimulatedCycleRuntime,
        *,
        movement_speed_units_per_s: float = 4.0,
        interaction_range: float = 0.5,
    ) -> None:
        if movement_speed_units_per_s <= 0.0:
            raise ValueError("movement_speed_units_per_s musi być większe od 0.")
        if interaction_range < 0.0:
            raise ValueError("interaction_range nie może być ujemny.")

        self._runtime = runtime
        self._movement_speed_units_per_s = movement_speed_units_per_s
        self._interaction_range = interaction_range

    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        world_snapshot = target_resolution.world_snapshot
        started_at_ts = world_snapshot.observed_at_ts
        target_id = target_resolution.selected_target_id

        if target_id is None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=None,
                started_at_ts=started_at_ts,
                completed_at_ts=started_at_ts,
                travel_s=0.0,
                arrived=False,
                reason="no_target_selected",
                initial_target_id=target_resolution.selected_target_id,
                retargeted=False,
                metadata={"decision_reason": target_resolution.decision.reason},
            )

        group = world_snapshot.group_by_id(target_id)
        if group is None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=target_id,
                started_at_ts=started_at_ts,
                completed_at_ts=started_at_ts,
                travel_s=0.0,
                arrived=False,
                reason="target_missing_from_world_snapshot",
                initial_target_id=target_resolution.selected_target_id,
                retargeted=False,
                metadata={"decision_reason": target_resolution.decision.reason},
            )

        travel_distance = max(0.0, group.distance - self._interaction_range)
        travel_s = travel_distance / self._movement_speed_units_per_s
        completed_at_ts = started_at_ts + travel_s
        return TargetApproachResult(
            cycle_id=target_resolution.cycle_id,
            target_id=target_id,
            started_at_ts=started_at_ts,
            completed_at_ts=completed_at_ts,
            travel_s=travel_s,
            arrived=True,
            reason="target_reached_in_simulation",
            initial_target_id=target_resolution.selected_target_id,
            retargeted=False,
            metadata={
                "decision_reason": target_resolution.decision.reason,
                "movement_speed_units_per_s": self._movement_speed_units_per_s,
                "interaction_range": self._interaction_range,
                "target_distance": group.distance,
                "travel_distance": travel_distance,
            },
        )


class SimulatedTargetInteractionProvider:
    """Zwraca prosty wynik gotowosci do interakcji z celem po dojściu."""

    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        return TargetInteractionResult(
            cycle_id=target_approach_result.cycle_id,
            target_id=target_approach_result.target_id,
            ready=target_approach_result.target_id is not None and target_approach_result.arrived,
            observed_at_ts=target_approach_result.completed_at_ts,
            reason=(
                "interaction_ready"
                if target_approach_result.target_id is not None and target_approach_result.arrived
                else "interaction_not_ready"
            ),
            initial_target_id=target_approach_result.initial_target_id,
            retargeted=target_approach_result.retargeted,
            metadata={
                "approach_reason": target_approach_result.reason,
            },
        )
