from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from botlab.domain.world import GroupSnapshot, TargetCandidate, WorldSnapshot


class TargetValidationStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    DEFEATED = "defeated"
    ENGAGED_BY_OTHER = "engaged_by_other"
    UNREACHABLE = "unreachable"


@dataclass(slots=True, frozen=True)
class TargetValidationResult:
    group_id: str
    status: TargetValidationStatus
    reason: str
    can_continue: bool
    metadata: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class RetargetDecision:
    current_target_id: str | None
    selected_target: TargetCandidate | None
    validation: TargetValidationResult | None
    changed: bool
    reason: str


@dataclass(slots=True, frozen=True)
class TargetSelectionPolicy:
    threat_weight: float = 0.0
    current_target_bonus: float = 0.0

    def __post_init__(self) -> None:
        if self.threat_weight < 0.0:
            raise ValueError("threat_weight nie może być ujemny.")
        if self.current_target_bonus < 0.0:
            raise ValueError("current_target_bonus nie może być ujemny.")
        if not isfinite(self.threat_weight):
            raise ValueError("threat_weight musi być liczbą skończoną.")
        if not isfinite(self.current_target_bonus):
            raise ValueError("current_target_bonus musi być liczbą skończoną.")

    def select_best_target(self, world: WorldSnapshot) -> TargetCandidate | None:
        targetable_groups = world.targetable_groups()
        if not targetable_groups:
            return None

        best_group = min(
            targetable_groups,
            key=lambda group: (self._effective_distance(group, world), group.group_id),
        )
        effective_distance = self._effective_distance(best_group, world)
        score = 1.0 / (1.0 + effective_distance)

        return TargetCandidate(
            group_id=best_group.group_id,
            score=score,
            reason="best_effective_distance",
            reachable=best_group.reachable,
            engaged_by_other=best_group.engaged_by_other,
            distance=best_group.distance,
            metadata={
                "effective_distance": effective_distance,
                "threat_score": best_group.threat_score,
                "current_target_bonus_applied": world.current_target_id == best_group.group_id,
            },
        )

    def _effective_distance(self, group: GroupSnapshot, world: WorldSnapshot) -> float:
        bonus = self.current_target_bonus if world.current_target_id == group.group_id else 0.0
        effective_distance = group.distance + (group.threat_score * self.threat_weight) - bonus
        return max(0.0, effective_distance)


@dataclass(slots=True, frozen=True)
class TargetValidationPolicy:
    def validate(self, world: WorldSnapshot, group_id: str) -> TargetValidationResult:
        group = world.group_by_id(group_id)
        if group is None:
            return TargetValidationResult(
                group_id=group_id,
                status=TargetValidationStatus.MISSING,
                reason="target_missing_from_world_snapshot",
                can_continue=False,
                metadata=None,
            )

        if not group.is_alive:
            return TargetValidationResult(
                group_id=group_id,
                status=TargetValidationStatus.DEFEATED,
                reason="target_has_no_alive_units",
                can_continue=False,
                metadata={"alive_count": group.alive_count},
            )

        if group.engaged_by_other:
            return TargetValidationResult(
                group_id=group_id,
                status=TargetValidationStatus.ENGAGED_BY_OTHER,
                reason="target_engaged_by_other_player",
                can_continue=False,
                metadata={"distance": group.distance},
            )

        if not group.reachable:
            return TargetValidationResult(
                group_id=group_id,
                status=TargetValidationStatus.UNREACHABLE,
                reason="target_not_reachable",
                can_continue=False,
                metadata={"distance": group.distance},
            )

        return TargetValidationResult(
            group_id=group_id,
            status=TargetValidationStatus.VALID,
            reason="target_still_valid",
            can_continue=True,
            metadata={"distance": group.distance, "alive_count": group.alive_count},
        )


@dataclass(slots=True, frozen=True)
class RetargetPolicy:
    selection_policy: TargetSelectionPolicy
    validation_policy: TargetValidationPolicy

    def resolve(self, world: WorldSnapshot, current_target_id: str | None) -> RetargetDecision:
        if current_target_id is not None:
            validation = self.validation_policy.validate(world, current_target_id)
            if validation.can_continue:
                current_group = world.group_by_id(current_target_id)
                assert current_group is not None
                return RetargetDecision(
                    current_target_id=current_target_id,
                    selected_target=TargetCandidate(
                        group_id=current_group.group_id,
                        score=1.0,
                        reason="keep_current_target",
                        reachable=current_group.reachable,
                        engaged_by_other=current_group.engaged_by_other,
                        distance=current_group.distance,
                        metadata={"validation_reason": validation.reason},
                    ),
                    validation=validation,
                    changed=False,
                    reason="current_target_still_valid",
                )

            selection_world = WorldSnapshot(
                observed_at_ts=world.observed_at_ts,
                bot_position=world.bot_position,
                groups=world.groups,
                in_combat=world.in_combat,
                current_target_id=None,
                metadata=dict(world.metadata),
            )
            replacement = self.selection_policy.select_best_target(selection_world)
            if replacement is None:
                return RetargetDecision(
                    current_target_id=current_target_id,
                    selected_target=None,
                    validation=validation,
                    changed=True,
                    reason="current_target_invalid_and_no_replacement",
                )

            return RetargetDecision(
                current_target_id=current_target_id,
                selected_target=replacement,
                validation=validation,
                changed=replacement.group_id != current_target_id,
                reason="current_target_invalid_retargeted",
            )

        selected_target = self.selection_policy.select_best_target(world)
        if selected_target is None:
            return RetargetDecision(
                current_target_id=None,
                selected_target=None,
                validation=None,
                changed=False,
                reason="no_target_available",
            )

        return RetargetDecision(
            current_target_id=None,
            selected_target=selected_target,
            validation=None,
            changed=True,
            reason="selected_initial_target",
        )
