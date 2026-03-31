from __future__ import annotations

from pathlib import Path

from botlab.adapters.simulation.replay import ScenarioReplayRunner
from botlab.config import Settings, TelemetryConfig, load_default_config


def _build_settings(tmp_path: Path) -> Settings:
    base = load_default_config()
    telemetry = TelemetryConfig(
        sqlite_path=(tmp_path / "data" / "telemetry" / "botlab.sqlite3").resolve(),
        log_path=(tmp_path / "logs" / "botlab.log").resolve(),
        log_level="INFO",
    )
    return Settings(
        app=base.app,
        cycle=base.cycle,
        combat=base.combat,
        telemetry=telemetry,
        vision=base.vision,
        source_path=base.source_path,
    )


def test_demo_farming_session_is_stable_golden_path(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    replay_runner = ScenarioReplayRunner.from_preset(
        settings,
        preset_name="demo_farming_session",
        enable_console=False,
    )

    report = replay_runner.run()

    assert report.total_cycles == 2
    assert [item.result for item in report.cycle_results] == ["success", "success"]

    assert report.target_resolutions[0].selected_target_id == "front-free"
    assert report.approach_results[0].initial_target_id == "front-free"
    assert report.approach_results[0].target_id == "fallback-safe"
    assert report.approach_results[0].retargeted is True
    assert report.interaction_results[0].target_id == "fallback-safe"
    assert report.interaction_results[0].ready is True

    cycle_records = {int(record["cycle_id"]): record for record in report.cycle_records}
    cycle_1_metadata = cycle_records[1]["metadata"]
    cycle_2_metadata = cycle_records[2]["metadata"]

    assert cycle_1_metadata["combat_completed"] is True
    assert cycle_1_metadata["reward_started_ts"] is not None
    assert cycle_1_metadata["reward_completed_ts"] is not None
    assert cycle_1_metadata["combat_finished_with_rest"] is True
    assert cycle_1_metadata["rest_final_hp_ratio"] >= 0.9
    assert cycle_1_metadata["rest_final_condition_ratio"] >= 0.9

    assert cycle_2_metadata["session_hp_ratio_before_cycle"] == cycle_1_metadata["rest_final_hp_ratio"]
    assert (
        cycle_2_metadata["session_condition_ratio_before_cycle"]
        == cycle_1_metadata["rest_final_condition_ratio"]
    )
    assert cycle_2_metadata["combat_completed"] is True
    assert cycle_2_metadata["combat_finished_with_rest"] is False

    trace = report.decision_trace_lines()
    assert any("phase=targeting " in line and "selected=front-free" in line for line in trace)
    assert any("phase=targeting rejected=occupied-near reason=engaged_by_other" in line for line in trace)
    assert any(
        "phase=approach " in line and "retarget_from=front-free retarget_to=fallback-safe" in line
        for line in trace
    )
    assert any("phase=reward " in line for line in trace)
    assert any("phase=rest result=completed" in line for line in trace)
    assert any("cycle_trace=2 phase=cycle " in line and "result=success" in line for line in trace)
