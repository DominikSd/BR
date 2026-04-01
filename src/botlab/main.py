from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from botlab.adapters.live import LiveRunner, PerceptionAnalysisRunner
from botlab.adapters.simulation.combat_profiles import SimulatedCombatProfileCatalog
from botlab.adapters.simulation.combat_plans import SimulatedCombatPlanCatalog
from botlab.adapters.simulation.replay import (
    ScenarioReplay,
    ScenarioReplayRunner,
    list_scenario_replay_presets,
    load_scenario_replay,
)
from botlab.adapters.simulation.runner import SimulationReport, SimulationRunner
from botlab.adapters.simulation.spawner import CycleScenario, SimulatedSpawner
from botlab.config import Settings, load_config, load_default_config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="botlab",
        description=(
            "Uruchamia lokalny przebieg symulacyjny rdzenia PvE cycle controller "
            "z telemetry do logu i SQLite."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Sciezka do pliku YAML z konfiguracja. Domyslnie: config/default.yaml",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Liczba cykli do zasymulowania. Domyslnie: 10 lub total_cycles replaya.",
    )
    parser.add_argument(
        "--anchor-spawn-ts",
        type=float,
        default=None,
        help="Poczatkowy znany timestamp spawnu dla bootstrapu predictora.",
    )
    parser.add_argument(
        "--anchor-cycle-id",
        type=int,
        default=None,
        help="Poczatkowy cycle_id dla bootstrapu predictora.",
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Wlacza logowanie telemetry rowniez na konsoli.",
    )
    parser.add_argument(
        "--scenario-preset",
        type=str,
        default=None,
        help="Nazwa wbudowanego replay/scenario presetu do uruchomienia.",
    )
    parser.add_argument(
        "--scenario-file",
        type=str,
        default=None,
        help="Sciezka do pliku YAML z replayem/scenario.",
    )
    parser.add_argument(
        "--list-scenario-presets",
        action="store_true",
        help="Wypisuje dostepne wbudowane presety replay/scenario i konczy dzialanie.",
    )
    parser.add_argument(
        "--combat-plan",
        type=str,
        default=None,
        help="Nazwa gotowego planu walki dla standardowego uruchomienia symulacji.",
    )
    parser.add_argument(
        "--list-combat-plans",
        action="store_true",
        help="Wypisuje dostepne nazwy planow walki i konczy dzialanie.",
    )
    parser.add_argument(
        "--combat-profile",
        type=str,
        default=None,
        help="Nazwa profilu walki mapowanego na named combat plan.",
    )
    parser.add_argument(
        "--list-combat-profiles",
        action="store_true",
        help="Wypisuje dostepne profile walki i konczy dzialanie.",
    )
    parser.add_argument(
        "--export-report-json",
        type=str,
        default=None,
        help="Opcjonalna sciezka do zapisu raportu skutecznosci w JSON.",
    )
    parser.add_argument(
        "--show-cycle-trace",
        action="store_true",
        help="Wypisuje czytelny trace decyzji cyklu: targetowanie, retarget, combat i rest.",
    )
    parser.add_argument(
        "--analyze-frame",
        type=str,
        default=None,
        help="Analizuje pojedyncza klatke lub plik JSON z frame spec i zapisuje artefakty perception.",
    )
    parser.add_argument(
        "--analyze-batch-dir",
        type=str,
        default=None,
        help="Analizuje katalog klatek/plikow JSON i zapisuje artefakty perception oraz agregaty latencji.",
    )
    parser.add_argument(
        "--perception-output-dir",
        type=str,
        default=None,
        help="Opcjonalny katalog wyjsciowy dla artefaktow perception. Domyslnie: live.debug_directory/perception.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.list_scenario_presets:
        _print_scenario_presets()
        return 0
    if args.list_combat_plans:
        _print_combat_plans()
        return 0
    if args.list_combat_profiles:
        _print_combat_profiles()
        return 0

    _validate_scenario_args(
        scenario_preset=args.scenario_preset,
        scenario_file=args.scenario_file,
        combat_plan=args.combat_plan,
        combat_profile=args.combat_profile,
    )
    _validate_perception_args(
        analyze_frame=args.analyze_frame,
        analyze_batch_dir=args.analyze_batch_dir,
        scenario_preset=args.scenario_preset,
        scenario_file=args.scenario_file,
    )

    settings = _load_settings(args.config)
    if args.analyze_frame is not None or args.analyze_batch_dir is not None:
        summary, output_directory = _run_perception_analysis(
            settings=settings,
            analyze_frame=args.analyze_frame,
            analyze_batch_dir=args.analyze_batch_dir,
            output_directory=args.perception_output_dir,
        )
        _print_perception_report(summary=summary, output_directory=output_directory)
        return 0

    replay = _load_replay(
        scenario_preset=args.scenario_preset,
        scenario_file=args.scenario_file,
    )
    cycles = _resolve_total_cycles(args.cycles, replay)
    anchor_spawn_ts = _resolve_anchor_spawn_ts(args.anchor_spawn_ts, replay)
    anchor_cycle_id = _resolve_anchor_cycle_id(args.anchor_cycle_id, replay)
    _validate_runtime_args(cycles=cycles, anchor_cycle_id=anchor_cycle_id)

    report = _run_simulation(
        settings=settings,
        replay=replay,
        cycles=cycles,
        anchor_spawn_ts=anchor_spawn_ts,
        anchor_cycle_id=anchor_cycle_id,
        combat_plan_name=_resolve_combat_plan_name(args.combat_plan, args.combat_profile),
        combat_profile_name=_resolve_combat_profile_name(settings, args.combat_profile),
        enable_console=args.console_log,
    )
    _export_report_json(report=report, export_path=args.export_report_json)
    _print_report(
        settings=settings,
        report=report,
        replay=replay,
        show_cycle_trace=_should_show_cycle_trace(
            replay=replay,
            show_cycle_trace=args.show_cycle_trace,
        ),
    )

    return 0


def _run_simulation(
    *,
    settings: Settings,
    replay: ScenarioReplay | None,
    cycles: int,
    anchor_spawn_ts: float,
    anchor_cycle_id: int,
    combat_plan_name: str | None,
    combat_profile_name: str | None,
    enable_console: bool,
) -> SimulationReport:
    if settings.app.mode == "live":
        if replay is not None:
            raise ValueError("Tryb live nie obsluguje scenario replay. Uzyj konfiguracji live.")
        runner = LiveRunner.from_settings(
            settings,
            initial_anchor_spawn_ts=anchor_spawn_ts,
            initial_anchor_cycle_id=anchor_cycle_id,
            enable_console=enable_console,
        )
        return runner.run_cycles(cycles)

    if replay is None:
        spawner = None
        if combat_plan_name is not None or combat_profile_name is not None:
            spawner = SimulatedSpawner(
                default_scenario=CycleScenario(
                    combat_plan_name=combat_plan_name,
                    combat_profile_name=combat_profile_name,
                )
            )
        runner = SimulationRunner.from_settings(
            settings,
            spawner=spawner,
            initial_anchor_spawn_ts=anchor_spawn_ts,
            initial_anchor_cycle_id=anchor_cycle_id,
            enable_console=enable_console,
        )
        return runner.run_cycles(cycles)

    replay_runner = ScenarioReplayRunner(
        settings=settings,
        replay=replay,
        enable_console=enable_console,
    )
    return replay_runner.run(
        total_cycles=cycles,
        initial_anchor_spawn_ts=anchor_spawn_ts,
        initial_anchor_cycle_id=anchor_cycle_id,
    )


def _run_perception_analysis(
    *,
    settings: Settings,
    analyze_frame: str | None,
    analyze_batch_dir: str | None,
    output_directory: str | None,
):
    if analyze_frame is None and analyze_batch_dir is None:
        raise ValueError("Brak zrodla dla trybu perception-only.")
    resolved_output_directory = _resolve_perception_output_directory(settings, output_directory)
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=resolved_output_directory,
    )
    if analyze_frame is not None:
        return runner.analyze_frame_path(analyze_frame), resolved_output_directory
    return runner.analyze_directory(analyze_batch_dir), resolved_output_directory


def _load_settings(config_path: str | None) -> Settings:
    if config_path is None:
        return load_default_config()

    return load_config(config_path)


def _load_replay(
    *,
    scenario_preset: str | None,
    scenario_file: str | None,
) -> ScenarioReplay | None:
    if scenario_preset is not None:
        for preset in list_scenario_replay_presets():
            if preset.name == scenario_preset:
                return preset
        available = ", ".join(preset.name for preset in list_scenario_replay_presets())
        raise ValueError(
            f"Nieznany scenario preset '{scenario_preset}'. Dostepne: {available}"
        )

    if scenario_file is not None:
        return load_scenario_replay(scenario_file)

    return None


def _validate_scenario_args(
    *,
    scenario_preset: str | None,
    scenario_file: str | None,
    combat_plan: str | None,
    combat_profile: str | None,
) -> None:
    if scenario_preset is not None and scenario_file is not None:
        raise ValueError("--scenario-preset i --scenario-file nie moga byc uzyte razem.")
    if combat_plan is not None and combat_profile is not None:
        raise ValueError("--combat-plan i --combat-profile nie moga byc uzyte razem.")


def _validate_perception_args(
    *,
    analyze_frame: str | None,
    analyze_batch_dir: str | None,
    scenario_preset: str | None,
    scenario_file: str | None,
) -> None:
    if analyze_frame is not None and analyze_batch_dir is not None:
        raise ValueError("--analyze-frame i --analyze-batch-dir nie moga byc uzyte razem.")
    if (analyze_frame is not None or analyze_batch_dir is not None) and (
        scenario_preset is not None or scenario_file is not None
    ):
        raise ValueError("Tryb perception-only nie obsluguje scenario replay.")


def _resolve_total_cycles(cycles: int | None, replay: ScenarioReplay | None) -> int:
    if cycles is not None:
        return cycles
    if replay is not None:
        return replay.total_cycles
    return 10


def _resolve_anchor_spawn_ts(
    anchor_spawn_ts: float | None,
    replay: ScenarioReplay | None,
) -> float:
    if anchor_spawn_ts is not None:
        return anchor_spawn_ts
    if replay is not None:
        return replay.initial_anchor_spawn_ts
    return 100.0


def _resolve_anchor_cycle_id(
    anchor_cycle_id: int | None,
    replay: ScenarioReplay | None,
) -> int:
    if anchor_cycle_id is not None:
        return anchor_cycle_id
    if replay is not None:
        return replay.initial_anchor_cycle_id
    return 0


def _resolve_combat_profile_name(
    settings: Settings,
    combat_profile_name: str | None,
) -> str | None:
    if combat_profile_name is not None:
        return combat_profile_name
    return settings.combat.default_profile_name


def _resolve_combat_plan_name(
    combat_plan_name: str | None,
    combat_profile_name: str | None,
) -> str | None:
    if combat_profile_name is not None:
        return None
    return combat_plan_name


def _validate_runtime_args(*, cycles: int, anchor_cycle_id: int) -> None:
    if cycles <= 0:
        raise ValueError("--cycles musi byc wieksze od 0.")
    if anchor_cycle_id < 0:
        raise ValueError("--anchor-cycle-id nie moze byc ujemne.")


def _print_scenario_presets() -> None:
    for preset in list_scenario_replay_presets():
        print(
            f"{preset.name}: total_cycles={preset.total_cycles} description={preset.description}"
        )


def _print_combat_plans() -> None:
    combat_plan_catalog = SimulatedCombatPlanCatalog()
    for plan_name in combat_plan_catalog.available_plan_names():
        print(plan_name)


def _print_combat_profiles() -> None:
    combat_profile_catalog = SimulatedCombatProfileCatalog()
    for profile_name in combat_profile_catalog.available_profile_names():
        print(profile_name)


def _export_report_json(*, report: SimulationReport, export_path: str | None) -> None:
    if export_path is None:
        return

    path = Path(export_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_export_dict()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _resolve_perception_output_directory(settings: Settings, output_directory: str | None) -> Path:
    if output_directory is not None:
        return Path(output_directory).expanduser().resolve()
    return (settings.live.debug_directory / "perception").resolve()


def _print_report(
    *,
    settings: Settings,
    report: SimulationReport,
    replay: ScenarioReplay | None,
    show_cycle_trace: bool = False,
) -> None:
    print(f"app.name={settings.app.name}")
    print(f"app.mode={settings.app.mode}")
    print(f"source_config={settings.source_path}")
    if replay is not None:
        print(f"scenario_replay={replay.name}")
    print(f"total_cycles={report.total_cycles}")
    print(f"success={report.count_result('success')}")
    print(f"failure={report.count_result('failure')}")
    print(f"no_event={report.count_result('no_event')}")
    print(f"no_target_available={report.count_result('no_target_available')}")
    print(f"approach_failed={report.count_result('approach_failed')}")
    print(f"late_event_missed={report.count_result('late_event_missed')}")
    print(f"verify_timeout={report.count_result('verify_timeout')}")
    print(f"execution_error={report.count_result('execution_error')}")
    print(f"log_path={report.log_path}")
    print(f"sqlite_path={report.sqlite_path}")

    for cycle_result in report.cycle_results:
        actual_spawn_repr = (
            "None"
            if cycle_result.actual_spawn_ts is None
            else f"{cycle_result.actual_spawn_ts:.3f}"
        )
        drift_repr = "None" if cycle_result.drift_s is None else f"{cycle_result.drift_s:.3f}"
        reaction_repr = (
            "None"
            if cycle_result.reaction_ms is None
            else f"{cycle_result.reaction_ms:.3f}"
        )
        verification_repr = (
            "None"
            if cycle_result.verification_ms is None
            else f"{cycle_result.verification_ms:.3f}"
        )

        print(
            "cycle="
            f"{cycle_result.cycle_id} "
            f"result={cycle_result.result} "
            f"final_state={cycle_result.final_state.value} "
            f"predicted_spawn_ts={cycle_result.predicted_spawn_ts:.3f} "
            f"actual_spawn_ts={actual_spawn_repr} "
            f"drift_s={drift_repr} "
            f"reaction_ms={reaction_repr} "
            f"verification_ms={verification_repr} "
            f"observation_used={cycle_result.observation_used} "
            f"note={cycle_result.note}"
        )

    for summary in report.combat_plan_summaries():
        avg_hp_repr = "None" if summary.avg_final_hp_ratio is None else f"{summary.avg_final_hp_ratio:.3f}"
        print(
            "combat_plan_summary="
            f"{summary.key} "
            f"total_cycles={summary.total_cycles} "
            f"success={summary.success_cycles} "
            f"failure={summary.failure_cycles} "
            f"no_target={summary.no_target_cycles} "
            f"verify_timeout={summary.timeout_cycles} "
            f"execution_error={summary.execution_error_cycles} "
            f"rest_cycles={summary.rest_cycles} "
            f"avg_final_hp_ratio={avg_hp_repr}"
        )

    for summary in report.combat_profile_summaries():
        avg_hp_repr = "None" if summary.avg_final_hp_ratio is None else f"{summary.avg_final_hp_ratio:.3f}"
        print(
            "combat_profile_summary="
            f"{summary.key} "
            f"total_cycles={summary.total_cycles} "
            f"success={summary.success_cycles} "
            f"failure={summary.failure_cycles} "
            f"no_target={summary.no_target_cycles} "
            f"verify_timeout={summary.timeout_cycles} "
            f"execution_error={summary.execution_error_cycles} "
            f"rest_cycles={summary.rest_cycles} "
            f"avg_final_hp_ratio={avg_hp_repr}"
        )

    if show_cycle_trace:
        for line in report.decision_trace_lines():
            print(line)


def _print_perception_report(*, summary, output_directory: Path) -> None:
    print("perception_mode=analysis")
    print(f"frame_count={len(summary.frame_results)}")
    print(f"perception_output_dir={output_directory}")
    for result in summary.frame_results:
        print(
            "perception_frame="
            f"{result.phase} "
            f"source={result.frame_source} "
            f"targets={len(result.detections)} "
            f"free_targets={len(result.free_detections)} "
            f"occupied_targets={len(result.occupied_detections)} "
            f"selected_target={result.selected_target_id} "
            f"detection_latency_ms={result.timings.detection_latency_ms:.3f} "
            f"selection_latency_ms={result.timings.selection_latency_ms:.3f} "
            f"total_reaction_latency_ms={result.timings.total_reaction_latency_ms:.3f}"
        )
    for aggregate in (
        summary.detection_latency,
        summary.selection_latency,
        summary.total_reaction_latency,
    ):
        min_repr = "None" if aggregate.min_ms is None else f"{aggregate.min_ms:.3f}"
        avg_repr = "None" if aggregate.avg_ms is None else f"{aggregate.avg_ms:.3f}"
        p50_repr = "None" if aggregate.p50_ms is None else f"{aggregate.p50_ms:.3f}"
        p95_repr = "None" if aggregate.p95_ms is None else f"{aggregate.p95_ms:.3f}"
        max_repr = "None" if aggregate.max_ms is None else f"{aggregate.max_ms:.3f}"
        print(
            "perception_latency_summary="
            f"{aggregate.name} "
            f"count={aggregate.count} "
            f"min_ms={min_repr} "
            f"avg_ms={avg_repr} "
            f"p50_ms={p50_repr} "
            f"p95_ms={p95_repr} "
            f"max_ms={max_repr}"
        )


def _should_show_cycle_trace(
    *,
    replay: ScenarioReplay | None,
    show_cycle_trace: bool,
) -> bool:
    if show_cycle_trace:
        return True
    if replay is None:
        return False
    return replay.name in {
        "demo_farming_cycle",
        "demo_farming_session",
        "demo_farming_showcase",
        "demo_observation_miss",
        "demo_observation_reposition",
    }


if __name__ == "__main__":
    raise SystemExit(main())
