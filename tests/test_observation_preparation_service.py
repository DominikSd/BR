from __future__ import annotations

from botlab.application import ObservationPreparationResult, ObservationPreparationService


class StaticObservationPreparationProvider:
    def __init__(self, result: ObservationPreparationResult) -> None:
        self._result = result

    def prepare_observation(self, cycle_id: int) -> ObservationPreparationResult:
        assert cycle_id == self._result.cycle_id
        return self._result


def test_prepare_observation_returns_provider_result() -> None:
    expected = ObservationPreparationResult(
        cycle_id=7,
        spawn_zone_visible=True,
        ready_for_observation=True,
        starting_position_xy=(0.0, 0.0),
        observation_position_xy=(1.0, 2.0),
        travel_s=0.5,
        arrived_at_ts=95.5,
        wait_for_spawn_s=49.5,
        note="spawn-zone-ready",
        metadata={"bot_position_xy": (1.0, 2.0)},
    )
    service = ObservationPreparationService(
        observation_preparation_provider=StaticObservationPreparationProvider(expected),
    )

    result = service.prepare_observation(cycle_id=7)

    assert result == expected
