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
    """Returns a simple result for moving into spawn observation position."""

    def __init__(
        self,
        runtime: SimulatedCycleRuntime,
        *,
        movement_speed_units_per_s: float = 4.0,
    ) -> None:
        if movement_speed_units_per_s <= 0.0:
            raise ValueError("movement_speed_units_per_s musi byc wieksze od 0.")
        self._runtime = runtime
        self._movement_speed_units_per_s = movement_speed_units_per_s

    def prepare_observation(self, cycle_id: int) -> ObservationPreparationResult:
        context = self._runtime.context_for_cycle(cycle_id)
        scenario = context.spawn_event.scenario
        previous_preparation = self._runtime.observation_preparation_for_cycle(cycle_id - 1)
        if scenario.observation_start_position_xy is not None:
            starting_position_xy = scenario.observation_start_position_xy
            start_position_source = "scenario_override"
        elif (
            previous_preparation is not None
            and previous_preparation.ready_for_observation is False
            and previous_preparation.metadata.get("ready_reason")
            == "arrived_after_ready_window_start"
        ):
            starting_position_xy = previous_preparation.observation_position_xy
            start_position_source = "carryover_from_previous_missed_cycle"
        else:
            starting_position_xy = scenario.bot_position_xy
            start_position_source = "scenario_current_position"

        starting_position = Position(*starting_position_xy)
        observation_position = Position(*scenario.bot_position_xy)
        travel_distance = starting_position.distance_to(observation_position)
        travel_s = travel_distance / self._movement_speed_units_per_s
        arrived_at_ts = context.trace.prepare_ts + travel_s
        ready_for_observation = (
            scenario.spawn_zone_visible
            and arrived_at_ts <= context.prediction.ready_window_start_ts
        )
        wait_for_spawn_s = max(0.0, context.prediction.predicted_spawn_ts - arrived_at_ts)

        if not scenario.spawn_zone_visible:
            ready_reason = "spawn_zone_hidden"
        elif ready_for_observation:
            ready_reason = "ready_in_observation_position"
        else:
            ready_reason = "arrived_after_ready_window_start"

        return ObservationPreparationResult(
            cycle_id=cycle_id,
            spawn_zone_visible=scenario.spawn_zone_visible,
            ready_for_observation=ready_for_observation,
            starting_position_xy=starting_position_xy,
            observation_position_xy=scenario.bot_position_xy,
            travel_s=travel_s,
            arrived_at_ts=arrived_at_ts,
            wait_for_spawn_s=wait_for_spawn_s,
            note=scenario.note,
            metadata={
                "ready_reason": ready_reason,
                "start_position_source": start_position_source,
                "reposition_required": travel_distance > 0.0,
                "bot_position_xy": scenario.bot_position_xy,
                "observation_start_position_xy": starting_position_xy,
                "configured_group_count": len(scenario.groups),
                "travel_distance": travel_distance,
                "movement_speed_units_per_s": self._movement_speed_units_per_s,
                "prepare_started_ts": context.trace.prepare_ts,
                "ready_window_start_ts": context.prediction.ready_window_start_ts,
            },
        )


class SimulatedWorldStateProvider:
    """Builds a domain WorldSnapshot from simulation cycle context."""

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

        generated_groups = self._runtime.groups_for_cycle(cycle_id)
        if generated_groups:
            return tuple(generated_groups)

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
    """Builds a world snapshot for target revalidation during approach."""

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
    """Builds a world snapshot for final validation before interaction."""

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
    """Builds a deterministic step-by-step approach to the selected target."""

    def __init__(
        self,
        runtime: SimulatedCycleRuntime,
        *,
        movement_speed_units_per_s: float = 4.0,
        interaction_range: float = 0.5,
        step_distance_units: float = 1.0,
    ) -> None:
        if movement_speed_units_per_s <= 0.0:
            raise ValueError("movement_speed_units_per_s musi byc wieksze od 0.")
        if interaction_range < 0.0:
            raise ValueError("interaction_range nie moze byc ujemny.")
        if step_distance_units <= 0.0:
            raise ValueError("step_distance_units musi byc wieksze od 0.")

        self._runtime = runtime
        self._movement_speed_units_per_s = movement_speed_units_per_s
        self._interaction_range = interaction_range
        self._step_distance_units = step_distance_units

    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        world_snapshot = target_resolution.world_snapshot
        scenario = self._runtime.context_for_cycle(target_resolution.cycle_id).spawn_event.scenario
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
        movement_steps = self._build_movement_steps(
            started_at_ts=started_at_ts,
            travel_distance=travel_distance,
        )
        stall_result = self._build_stall_result(
            cycle_id=target_resolution.cycle_id,
            target_id=target_id,
            started_at_ts=started_at_ts,
            travel_distance=travel_distance,
            movement_steps=movement_steps,
            decision_reason=target_resolution.decision.reason,
            scenario=scenario,
        )
        if stall_result is not None:
            return stall_result

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
                "step_distance_units": self._step_distance_units,
                "movement_step_count": len(movement_steps),
                "movement_steps": movement_steps,
            },
        )

    def _build_stall_result(
        self,
        *,
        cycle_id: int,
        target_id: str,
        started_at_ts: float,
        travel_distance: float,
        movement_steps: list[dict[str, float | int]],
        decision_reason: str,
        scenario,
    ) -> TargetApproachResult | None:
        stall_after_step = scenario.approach_stall_after_step
        if stall_after_step is None or travel_distance <= 0.0:
            return None
        if stall_after_step < 0:
            raise ValueError("approach_stall_after_step nie moze byc ujemny.")
        if scenario.approach_stall_timeout_s <= 0.0:
            raise ValueError("approach_stall_timeout_s musi byc wieksze od 0.")
        if stall_after_step >= len(movement_steps):
            return None

        completed_steps = movement_steps[:stall_after_step]
        last_progress_ts = started_at_ts
        if completed_steps:
            last_step_arrived_ts = completed_steps[-1].get("arrived_ts")
            if isinstance(last_step_arrived_ts, (int, float)):
                last_progress_ts = float(last_step_arrived_ts)
        stalled_at_ts = last_progress_ts + scenario.approach_stall_timeout_s

        return TargetApproachResult(
            cycle_id=cycle_id,
            target_id=target_id,
            started_at_ts=started_at_ts,
            completed_at_ts=stalled_at_ts,
            travel_s=max(0.0, stalled_at_ts - started_at_ts),
            arrived=False,
            reason="approach_stalled_no_progress_timeout",
            initial_target_id=target_id,
            retargeted=False,
            metadata={
                "decision_reason": decision_reason,
                "movement_speed_units_per_s": self._movement_speed_units_per_s,
                "interaction_range": self._interaction_range,
                "travel_distance": travel_distance,
                "step_distance_units": self._step_distance_units,
                "movement_step_count": len(completed_steps),
                "movement_steps": completed_steps,
                "stall_after_step": stall_after_step,
                "stall_timeout_s": scenario.approach_stall_timeout_s,
                "stalled_at_ts": stalled_at_ts,
            },
        )

    def _build_movement_steps(
        self,
        *,
        started_at_ts: float,
        travel_distance: float,
    ) -> list[dict[str, float | int]]:
        if travel_distance <= 0.0:
            return []

        remaining_distance = travel_distance
        accumulated_distance = 0.0
        current_ts = started_at_ts
        step_index = 0
        steps: list[dict[str, float | int]] = []
        while remaining_distance > 0.0:
            step_index += 1
            step_distance = min(self._step_distance_units, remaining_distance)
            step_duration_s = step_distance / self._movement_speed_units_per_s
            accumulated_distance += step_distance
            current_ts += step_duration_s
            remaining_distance = max(0.0, travel_distance - accumulated_distance)
            steps.append(
                {
                    "step_index": step_index,
                    "step_distance": round(step_distance, 6),
                    "arrived_ts": round(current_ts, 6),
                    "remaining_distance": round(remaining_distance, 6),
                }
            )
        return steps


class SimulatedTargetInteractionProvider:
    """Returns a simple readiness result after approach."""

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
