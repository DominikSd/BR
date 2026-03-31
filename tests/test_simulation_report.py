from __future__ import annotations

from pathlib import Path

from botlab.application.dto import CycleRunResult, SimulationReport
from botlab.types import BotState


def test_simulation_report_builds_combat_plan_summaries() -> None:
    report = SimulationReport(
        cycle_results=[
            CycleRunResult(
                cycle_id=1,
                predicted_spawn_ts=100.0,
                actual_spawn_ts=100.0,
                drift_s=0.0,
                result="success",
                final_state=BotState.WAIT_NEXT_CYCLE,
                reaction_ms=20.0,
                verification_ms=100.0,
                observation_used=True,
                note="a",
            ),
            CycleRunResult(
                cycle_id=2,
                predicted_spawn_ts=145.0,
                actual_spawn_ts=145.0,
                drift_s=0.0,
                result="success",
                final_state=BotState.WAIT_NEXT_CYCLE,
                reaction_ms=20.0,
                verification_ms=100.0,
                observation_used=True,
                note="b",
            ),
        ],
        log_path=Path("logs/botlab.log"),
        sqlite_path=Path("data/telemetry/botlab.sqlite3"),
        cycle_records=[
            {
                "cycle_id": 1,
                "result": "success",
                "metadata": {
                    "combat_plan_name": "basic_1_space",
                    "combat_final_hp_ratio": 0.4,
                    "combat_finished_with_rest": True,
                },
            },
            {
                "cycle_id": 2,
                "result": "success",
                "metadata": {
                    "combat_plan_name": "basic_1_space",
                    "combat_final_hp_ratio": 0.8,
                    "combat_finished_with_rest": False,
                },
            },
        ],
    )

    summaries = report.combat_plan_summaries()

    assert len(summaries) == 1
    assert summaries[0].key == "basic_1_space"
    assert summaries[0].total_cycles == 2
    assert summaries[0].success_cycles == 2
    assert summaries[0].rest_cycles == 1
    assert summaries[0].avg_final_hp_ratio == 0.6


def test_simulation_report_builds_combat_profile_summaries() -> None:
    report = SimulationReport(
        cycle_results=[],
        log_path=Path("logs/botlab.log"),
        sqlite_path=Path("data/telemetry/botlab.sqlite3"),
        cycle_records=[
            {
                "cycle_id": 1,
                "result": "success",
                "metadata": {
                    "combat_profile_name": "fast_farmer",
                    "combat_final_hp_ratio": 0.5,
                    "combat_finished_with_rest": True,
                },
            },
            {
                "cycle_id": 2,
                "result": "execution_error",
                "metadata": {
                    "combat_profile_name": "fast_farmer",
                    "combat_final_hp_ratio": None,
                    "combat_finished_with_rest": False,
                },
            },
        ],
    )

    summaries = report.combat_profile_summaries()

    assert len(summaries) == 1
    assert summaries[0].key == "fast_farmer"
    assert summaries[0].total_cycles == 2
    assert summaries[0].success_cycles == 1
    assert summaries[0].execution_error_cycles == 1
    assert summaries[0].rest_cycles == 1
    assert summaries[0].avg_final_hp_ratio == 0.5
