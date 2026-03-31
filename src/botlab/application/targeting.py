from __future__ import annotations

from dataclasses import replace

from botlab.application.dto import (
    ObservationPreparationResult,
    TargetApproachResult,
    TargetEngagementResult,
    TargetInteractionResult,
    TargetResolution,
)
from botlab.application.ports import (
    ApproachWorldStateProvider,
    InteractionWorldStateProvider,
    ObservationPreparationProvider,
    TargetApproachProvider,
    TargetInteractionProvider,
    WorldStateProvider,
)
from botlab.domain.targeting import RetargetPolicy


class ObservationPreparationService:
    """Mały use-case application dla przygotowania obserwacji strefy spawnu."""

    def __init__(self, *, observation_preparation_provider: ObservationPreparationProvider) -> None:
        self._observation_preparation_provider = observation_preparation_provider

    def prepare_observation(self, *, cycle_id: int) -> ObservationPreparationResult:
        return self._observation_preparation_provider.prepare_observation(cycle_id)


class TargetAcquisitionService:
    """Mały use-case application do utrzymania lub pozyskania celu."""

    def __init__(
        self,
        *,
        world_state_provider: WorldStateProvider,
        retarget_policy: RetargetPolicy,
    ) -> None:
        self._world_state_provider = world_state_provider
        self._retarget_policy = retarget_policy

    def resolve_target(
        self,
        *,
        cycle_id: int,
        current_target_id: str | None = None,
    ) -> TargetResolution:
        world_snapshot = self._world_state_provider.get_world_snapshot(cycle_id)
        decision = self._retarget_policy.resolve(world_snapshot, current_target_id)
        selected_target_id = (
            None if decision.selected_target is None else decision.selected_target.group_id
        )

        return TargetResolution(
            cycle_id=cycle_id,
            current_target_id=current_target_id,
            selected_target_id=selected_target_id,
            world_snapshot=world_snapshot,
            decision=decision,
        )


class TargetApproachService:
    """Mały use-case application dla dojścia do wcześniej wybranego celu."""

    def __init__(
        self,
        *,
        target_approach_provider: TargetApproachProvider,
        approach_world_state_provider: ApproachWorldStateProvider | None = None,
        retarget_policy: RetargetPolicy | None = None,
    ) -> None:
        self._target_approach_provider = target_approach_provider
        self._approach_world_state_provider = approach_world_state_provider
        self._retarget_policy = retarget_policy

    def approach_target(self, target_resolution: TargetResolution) -> TargetApproachResult:
        initial_result = self._target_approach_provider.approach_target(target_resolution)
        initial_target_id = target_resolution.selected_target_id

        if (
            initial_target_id is None
            or self._approach_world_state_provider is None
            or self._retarget_policy is None
        ):
            return self._with_tracking_metadata(
                initial_result,
                initial_target_id=initial_target_id,
                retargeted=False,
            )

        approach_world_snapshot = self._approach_world_state_provider.get_approach_world_snapshot(
            target_resolution.cycle_id
        )
        decision = self._retarget_policy.resolve(approach_world_snapshot, initial_target_id)
        final_target_id = None if decision.selected_target is None else decision.selected_target.group_id

        if final_target_id == initial_target_id and decision.reason == "current_target_still_valid":
            return self._with_tracking_metadata(
                initial_result,
                initial_target_id=initial_target_id,
                retargeted=False,
                extra_metadata={
                    "revalidation_reason": decision.reason,
                    **self._revalidation_step_metadata(
                        initial_result=initial_result,
                        revalidation_ts=approach_world_snapshot.observed_at_ts,
                    ),
                },
            )

        if final_target_id is None:
            return TargetApproachResult(
                cycle_id=target_resolution.cycle_id,
                target_id=None,
                started_at_ts=initial_result.started_at_ts,
                completed_at_ts=approach_world_snapshot.observed_at_ts,
                travel_s=max(0.0, approach_world_snapshot.observed_at_ts - initial_result.started_at_ts),
                arrived=False,
                reason="target_lost_during_approach_no_replacement",
                initial_target_id=initial_target_id,
                retargeted=False,
                metadata={
                    "revalidation_reason": decision.reason,
                    "validation_reason": (
                        None if decision.validation is None else decision.validation.reason
                    ),
                    **self._revalidation_step_metadata(
                        initial_result=initial_result,
                        revalidation_ts=approach_world_snapshot.observed_at_ts,
                    ),
                },
            )

        replacement_resolution = TargetResolution(
            cycle_id=target_resolution.cycle_id,
            current_target_id=initial_target_id,
            selected_target_id=final_target_id,
            world_snapshot=approach_world_snapshot,
            decision=decision,
        )
        replacement_result = self._target_approach_provider.approach_target(replacement_resolution)
        return self._with_tracking_metadata(
            replacement_result,
            initial_target_id=initial_target_id,
            retargeted=final_target_id != initial_target_id,
            extra_metadata={
                "revalidation_reason": decision.reason,
                "validation_reason": (
                    None if decision.validation is None else decision.validation.reason
                ),
                **self._revalidation_step_metadata(
                    initial_result=initial_result,
                    revalidation_ts=approach_world_snapshot.observed_at_ts,
                ),
            },
        )

    def _with_tracking_metadata(
        self,
        result: TargetApproachResult,
        *,
        initial_target_id: str | None,
        retargeted: bool,
        extra_metadata: dict[str, object] | None = None,
    ) -> TargetApproachResult:
        metadata = dict(result.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        return TargetApproachResult(
            cycle_id=result.cycle_id,
            target_id=result.target_id,
            started_at_ts=result.started_at_ts,
            completed_at_ts=result.completed_at_ts,
            travel_s=result.travel_s,
            arrived=result.arrived,
            reason=result.reason,
            initial_target_id=initial_target_id,
            retargeted=retargeted,
            metadata=metadata,
        )

    def _revalidation_step_metadata(
        self,
        *,
        initial_result: TargetApproachResult,
        revalidation_ts: float,
    ) -> dict[str, object]:
        raw_steps = initial_result.metadata.get("movement_steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            return {}

        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                continue
            arrived_ts = raw_step.get("arrived_ts")
            step_index = raw_step.get("step_index")
            if isinstance(arrived_ts, (int, float)) and isinstance(step_index, int):
                if revalidation_ts <= float(arrived_ts):
                    return {"revalidation_step_index": step_index}

        last_step = raw_steps[-1]
        if isinstance(last_step, dict) and isinstance(last_step.get("step_index"), int):
            return {"revalidation_step_index": last_step["step_index"]}
        return {}


class TargetInteractionService:
    """Mały use-case application dla końcowej walidacji celu przed interakcją."""

    def __init__(
        self,
        *,
        target_interaction_provider: TargetInteractionProvider,
        interaction_world_state_provider: InteractionWorldStateProvider | None = None,
        retarget_policy: RetargetPolicy | None = None,
    ) -> None:
        self._target_interaction_provider = target_interaction_provider
        self._interaction_world_state_provider = interaction_world_state_provider
        self._retarget_policy = retarget_policy

    def prepare_interaction(
        self,
        target_approach_result: TargetApproachResult,
    ) -> TargetInteractionResult:
        initial_result = self._target_interaction_provider.prepare_interaction(
            target_approach_result
        )
        current_target_id = target_approach_result.target_id

        if (
            current_target_id is None
            or self._interaction_world_state_provider is None
            or self._retarget_policy is None
        ):
            return self._with_tracking_metadata(
                initial_result,
                initial_target_id=target_approach_result.initial_target_id,
                retargeted=target_approach_result.retargeted,
            )

        interaction_world_snapshot = self._interaction_world_state_provider.get_interaction_world_snapshot(
            target_approach_result.cycle_id
        )
        decision = self._retarget_policy.resolve(interaction_world_snapshot, current_target_id)
        final_target_id = None if decision.selected_target is None else decision.selected_target.group_id

        if final_target_id == current_target_id and decision.reason == "current_target_still_valid":
            return self._with_tracking_metadata(
                initial_result,
                initial_target_id=target_approach_result.initial_target_id,
                retargeted=target_approach_result.retargeted,
                extra_metadata={"revalidation_reason": decision.reason},
            )

        if final_target_id is None:
            return TargetInteractionResult(
                cycle_id=target_approach_result.cycle_id,
                target_id=None,
                ready=False,
                observed_at_ts=interaction_world_snapshot.observed_at_ts,
                reason="target_lost_before_interaction_no_replacement",
                initial_target_id=target_approach_result.initial_target_id,
                retargeted=target_approach_result.retargeted,
                world_snapshot=interaction_world_snapshot,
                decision=decision,
                metadata={
                    "revalidation_reason": decision.reason,
                    "validation_reason": (
                        None if decision.validation is None else decision.validation.reason
                    ),
                },
            )

        return self._with_tracking_metadata(
            TargetInteractionResult(
                cycle_id=target_approach_result.cycle_id,
                target_id=final_target_id,
                ready=False,
                observed_at_ts=interaction_world_snapshot.observed_at_ts,
                reason="retarget_before_interaction_requires_new_approach",
                initial_target_id=target_approach_result.initial_target_id,
                retargeted=True,
                world_snapshot=interaction_world_snapshot,
                decision=decision,
                metadata={},
            ),
            initial_target_id=target_approach_result.initial_target_id,
            retargeted=True,
            extra_metadata={
                "revalidation_reason": decision.reason,
                "validation_reason": (
                    None if decision.validation is None else decision.validation.reason
                ),
            },
        )

    def _with_tracking_metadata(
        self,
        result: TargetInteractionResult,
        *,
        initial_target_id: str | None,
        retargeted: bool,
        extra_metadata: dict[str, object] | None = None,
    ) -> TargetInteractionResult:
        metadata = dict(result.metadata)
        if extra_metadata:
            metadata.update(extra_metadata)
        return TargetInteractionResult(
            cycle_id=result.cycle_id,
            target_id=result.target_id,
            ready=result.ready,
            observed_at_ts=result.observed_at_ts,
            reason=result.reason,
            initial_target_id=initial_target_id,
            retargeted=retargeted,
            world_snapshot=result.world_snapshot,
            decision=result.decision,
            metadata=metadata,
        )


class TargetEngagementService:
    """Składa acquire, approach i interaction w jeden jawny use-case application."""

    def __init__(
        self,
        *,
        acquisition_service: TargetAcquisitionService,
        approach_service: TargetApproachService,
        interaction_service: TargetInteractionService,
    ) -> None:
        self._acquisition_service = acquisition_service
        self._approach_service = approach_service
        self._interaction_service = interaction_service

    def engage_target(
        self,
        *,
        cycle_id: int,
        current_target_id: str | None = None,
    ) -> TargetEngagementResult:
        current_resolution = self._acquisition_service.resolve_target(
            cycle_id=cycle_id,
            current_target_id=current_target_id,
        )
        original_target_id = current_resolution.selected_target_id
        retargeted_before_interaction = False

        while True:
            approach_result = self._approach_service.approach_target(current_resolution)
            interaction_result = self._interaction_service.prepare_interaction(approach_result)
            if retargeted_before_interaction:
                interaction_metadata = dict(interaction_result.metadata)
                interaction_metadata["retargeted_before_interaction_phase"] = True
                interaction_result = replace(
                    interaction_result,
                    initial_target_id=original_target_id,
                    retargeted=True,
                    metadata=interaction_metadata,
                )

            if (
                interaction_result.ready
                or interaction_result.target_id is None
                or interaction_result.reason != "retarget_before_interaction_requires_new_approach"
                or interaction_result.world_snapshot is None
                or interaction_result.decision is None
            ):
                return TargetEngagementResult(
                    cycle_id=cycle_id,
                    target_resolution=current_resolution,
                    approach_result=approach_result,
                    interaction_result=interaction_result,
                )

            retargeted_before_interaction = True
            current_resolution = TargetResolution(
                cycle_id=cycle_id,
                current_target_id=approach_result.target_id,
                selected_target_id=interaction_result.target_id,
                world_snapshot=interaction_result.world_snapshot,
                decision=interaction_result.decision,
            )
