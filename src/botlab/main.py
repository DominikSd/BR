from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from botlab.adapters.live import LiveEngageObserve, LiveRunner, LiveVisionPreview, PerceptionAnalysisRunner
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
        "--benchmark-split",
        type=str,
        default=None,
        help="Uruchamia benchmark vision dla zdefiniowanego splitu datasetu, np. regression albo holdout.",
    )
    parser.add_argument(
        "--perception-output-dir",
        type=str,
        default=None,
        help="Opcjonalny katalog wyjsciowy dla artefaktow perception. Domyslnie: live.debug_directory/perception.",
    )
    parser.add_argument(
        "--strict-pixel-benchmark",
        action="store_true",
        help="Wylacza fallback metadata-only i wymaga prawdziwego obrazu rasterowego dla perception benchmarku.",
    )
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="Uruchamia osobne okno preview/debug dla live vision bez wykonywania akcji w grze.",
    )
    parser.add_argument(
        "--live-engage-observe",
        action="store_true",
        help="Uruchamia okno debugowe dla pionu detect -> select -> engage -> verify na zywo.",
    )
    parser.add_argument(
        "--live-engage-mvp",
        action="store_true",
        help="Uruchamia minimalny pion live engage: perception -> nearest free target -> engage attempt -> outcome verification.",
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
        benchmark_split=args.benchmark_split,
        scenario_preset=args.scenario_preset,
        scenario_file=args.scenario_file,
    )

    settings = _load_settings(args.config)
    if args.live_preview:
        _validate_live_preview_args(
            settings=settings,
            scenario_preset=args.scenario_preset,
            scenario_file=args.scenario_file,
            analyze_frame=args.analyze_frame,
            analyze_batch_dir=args.analyze_batch_dir,
        )
        preview = LiveVisionPreview(
            settings=settings,
            enable_console=args.console_log,
        )
        return preview.run()
    if args.live_engage_observe:
        _validate_live_engage_observe_args(
            settings=settings,
            scenario_preset=args.scenario_preset,
            scenario_file=args.scenario_file,
            analyze_frame=args.analyze_frame,
            analyze_batch_dir=args.analyze_batch_dir,
        )
        observe = LiveEngageObserve(
            settings=settings,
            enable_console=args.console_log,
        )
        return observe.run()
    if args.live_engage_mvp:
        _validate_live_engage_args(
            settings=settings,
            scenario_preset=args.scenario_preset,
            scenario_file=args.scenario_file,
            analyze_frame=args.analyze_frame,
            analyze_batch_dir=args.analyze_batch_dir,
        )
        attempts = _resolve_total_cycles(args.cycles, replay=None)
        _validate_runtime_args(cycles=attempts, anchor_cycle_id=0)
        report = _run_live_engage_mvp(
            settings=settings,
            attempts=attempts,
            anchor_spawn_ts=_resolve_anchor_spawn_ts(args.anchor_spawn_ts, replay=None),
            anchor_cycle_id=_resolve_anchor_cycle_id(args.anchor_cycle_id, replay=None),
            enable_console=args.console_log,
        )
        _print_live_engage_report(report)
        return 0
    if (
        args.analyze_frame is not None
        or args.analyze_batch_dir is not None
        or args.benchmark_split is not None
    ):
        summary, output_directory = _run_perception_analysis(
            settings=settings,
            analyze_frame=args.analyze_frame,
            analyze_batch_dir=args.analyze_batch_dir,
            benchmark_split=args.benchmark_split,
            output_directory=args.perception_output_dir,
            strict_pixel_only=args.strict_pixel_benchmark,
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


def _run_live_engage_mvp(
    *,
    settings: Settings,
    attempts: int,
    anchor_spawn_ts: float,
    anchor_cycle_id: int,
    enable_console: bool,
):
    runner = LiveRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=anchor_spawn_ts,
        initial_anchor_cycle_id=anchor_cycle_id,
        enable_console=enable_console,
    )
    return runner.run_engage_attempts(attempts)


def _run_perception_analysis(
    *,
    settings: Settings,
    analyze_frame: str | None,
    analyze_batch_dir: str | None,
    benchmark_split: str | None,
    output_directory: str | None,
    strict_pixel_only: bool,
):
    if analyze_frame is None and analyze_batch_dir is None and benchmark_split is None:
        raise ValueError("Brak zrodla dla trybu perception-only.")
    resolved_output_directory = _resolve_perception_output_directory(settings, output_directory)
    runner = PerceptionAnalysisRunner(
        live_config=settings.live,
        output_directory=resolved_output_directory,
        strict_pixel_only=strict_pixel_only,
    )
    if analyze_frame is not None:
        return runner.analyze_frame_path(analyze_frame), resolved_output_directory
    if benchmark_split is not None:
        return runner.analyze_benchmark_split(benchmark_split), resolved_output_directory
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
    benchmark_split: str | None,
    scenario_preset: str | None,
    scenario_file: str | None,
) -> None:
    if analyze_frame is not None and analyze_batch_dir is not None:
        raise ValueError("--analyze-frame i --analyze-batch-dir nie moga byc uzyte razem.")
    if benchmark_split is not None and (analyze_frame is not None or analyze_batch_dir is not None):
        raise ValueError("--benchmark-split nie moze byc laczony z --analyze-frame ani --analyze-batch-dir.")
    if (analyze_frame is not None or analyze_batch_dir is not None) and (
        scenario_preset is not None or scenario_file is not None
    ):
        raise ValueError("Tryb perception-only nie obsluguje scenario replay.")
    if benchmark_split is not None and (scenario_preset is not None or scenario_file is not None):
        raise ValueError("Tryb benchmark perception nie obsluguje scenario replay.")


def _validate_live_preview_args(
    *,
    settings: Settings,
    scenario_preset: str | None,
    scenario_file: str | None,
    analyze_frame: str | None,
    analyze_batch_dir: str | None,
) -> None:
    if settings.app.mode != "live":
        raise ValueError("--live-preview wymaga konfiguracji z app.mode=live.")
    if scenario_preset is not None or scenario_file is not None:
        raise ValueError("--live-preview nie obsluguje scenario replay.")
    if analyze_frame is not None or analyze_batch_dir is not None:
        raise ValueError("--live-preview nie moze byc laczony z trybem perception-only.")


def _validate_live_engage_observe_args(
    *,
    settings: Settings,
    scenario_preset: str | None,
    scenario_file: str | None,
    analyze_frame: str | None,
    analyze_batch_dir: str | None,
) -> None:
    if settings.app.mode != "live":
        raise ValueError("--live-engage-observe wymaga konfiguracji z app.mode=live.")
    if scenario_preset is not None or scenario_file is not None:
        raise ValueError("--live-engage-observe nie obsluguje scenario replay.")
    if analyze_frame is not None or analyze_batch_dir is not None:
        raise ValueError("--live-engage-observe nie moze byc laczony z trybem perception-only.")


def _validate_live_engage_args(
    *,
    settings: Settings,
    scenario_preset: str | None,
    scenario_file: str | None,
    analyze_frame: str | None,
    analyze_batch_dir: str | None,
) -> None:
    if settings.app.mode != "live":
        raise ValueError("--live-engage-mvp wymaga konfiguracji z app.mode=live.")
    if scenario_preset is not None or scenario_file is not None:
        raise ValueError("--live-engage-mvp nie obsluguje scenario replay.")
    if analyze_frame is not None or analyze_batch_dir is not None:
        raise ValueError("--live-engage-mvp nie moze byc laczony z trybem perception-only.")


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
            f"candidate_hits={result.candidate_hit_count} "
            f"merged_hits={result.merged_hit_count} "
            f"targets={len(result.detections)} "
            f"free_targets={len(result.free_detections)} "
            f"occupied_targets={len(result.occupied_detections)} "
            f"selected_target={result.selected_target_id} "
            f"detection_latency_ms={result.timings.detection_latency_ms:.3f} "
            f"selection_latency_ms={result.timings.selection_latency_ms:.3f} "
            f"total_reaction_latency_ms={result.timings.total_reaction_latency_ms:.3f}"
        )
    for aggregate in (
        summary.candidate_hits,
        summary.merged_hits,
        summary.free_targets,
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
    if summary.accuracy_summary is not None:
        print(
            "perception_accuracy_summary="
            f"evaluated_frames={summary.accuracy_summary.evaluated_frame_count} "
            f"behavior_match={summary.accuracy_summary.behavior_match_count} "
            f"selected_match={summary.accuracy_summary.selected_target_match_count} "
            f"occupied_match={summary.accuracy_summary.occupied_contract_match_count}"
        )
    if summary.benchmark_summary is not None:
        occupied_accuracy_repr = (
            "None"
            if summary.benchmark_summary.occupied_classification_accuracy is None
            else f"{summary.benchmark_summary.occupied_classification_accuracy:.3f}"
        )
        selected_accuracy_repr = (
            "None"
            if summary.benchmark_summary.selected_target_accuracy is None
            else f"{summary.benchmark_summary.selected_target_accuracy:.3f}"
        )
        selected_in_zone_accuracy_repr = (
            "None"
            if summary.benchmark_summary.selected_target_in_zone_accuracy is None
            else f"{summary.benchmark_summary.selected_target_in_zone_accuracy:.3f}"
        )
        print(
            "perception_benchmark_summary="
            f"evaluated_frames={summary.benchmark_summary.evaluated_frame_count} "
            f"strict_pixel_only={summary.benchmark_summary.strict_pixel_only} "
            f"pixel_frames={summary.benchmark_summary.pixel_frame_count} "
            f"fallback_frames={summary.benchmark_summary.fallback_frame_count} "
            f"target_recall={summary.benchmark_summary.target_recall:.3f} "
            f"target_precision={summary.benchmark_summary.target_precision:.3f} "
            f"occupied_accuracy={occupied_accuracy_repr} "
            f"selected_target_accuracy={selected_accuracy_repr} "
            f"selected_target_in_zone_accuracy={selected_in_zone_accuracy_repr} "
            f"out_of_zone_rejections_avg={summary.benchmark_summary.out_of_zone_rejection_count.avg_value} "
            f"fp_reduction_after_zone_avg={summary.benchmark_summary.false_positive_reduction_after_zone_filtering.avg_value} "
            f"false_positive={summary.benchmark_summary.target_false_positive_count} "
            f"false_negative={summary.benchmark_summary.target_false_negative_count}"
        )
    for entry in summary.real_scene_regression_entries():
        print(
            "real_scene_regression="
            f"{entry['frame_source']} "
            f"targets={entry['target_count']} "
            f"free_targets={entry['free_target_count']} "
            f"occupied_targets={entry['occupied_target_count']} "
            f"out_of_zone_targets={entry['out_of_zone_target_count']} "
            f"selected_target={entry['selected_target_id']} "
            f"selected_in_zone={entry['selected_target_in_zone']} "
            f"selected_xy={entry['selected_target_xy']} "
            f"occupied_xy={entry['occupied_target_xy']} "
            f"detection_latency_ms={entry['detection_latency_ms']:.3f} "
            f"selection_latency_ms={entry['selection_latency_ms']:.3f} "
            f"total_reaction_latency_ms={entry['total_reaction_latency_ms']:.3f}"
        )


def _print_live_engage_report(report) -> None:
    print("engage_mode=mvp")
    print(f"total_attempts={report.summary.total_attempts}")
    print(f"engaged={report.summary.engaged_count}")
    success_rate = 0.0
    if report.summary.total_attempts > 0:
        success_rate = report.summary.engaged_count / float(report.summary.total_attempts)
    print(f"engage_success_rate={success_rate:.3f}")
    print(f"target_stolen={report.summary.target_stolen_count}")
    print(f"misclick={report.summary.misclick_count}")
    print(f"approach_stalled={report.summary.approach_stalled_count}")
    print(f"approach_timeout={report.summary.approach_timeout_count}")
    print(f"no_target_available={report.summary.no_target_available_count}")
    print(f"log_path={report.log_path}")
    print(f"sqlite_path={report.sqlite_path}")
    for aggregate in (
        report.summary.detection_latency,
        report.summary.selection_latency,
        report.summary.total_reaction_latency,
        report.summary.verification_latency,
    ):
        min_repr = "None" if aggregate.min_ms is None else f"{aggregate.min_ms:.3f}"
        avg_repr = "None" if aggregate.avg_ms is None else f"{aggregate.avg_ms:.3f}"
        p50_repr = "None" if aggregate.p50_ms is None else f"{aggregate.p50_ms:.3f}"
        p95_repr = "None" if aggregate.p95_ms is None else f"{aggregate.p95_ms:.3f}"
        max_repr = "None" if aggregate.max_ms is None else f"{aggregate.max_ms:.3f}"
        print(
            "engage_latency_summary="
            f"{aggregate.name} "
            f"count={aggregate.count} "
            f"min_ms={min_repr} "
            f"avg_ms={avg_repr} "
            f"p50_ms={p50_repr} "
            f"p95_ms={p95_repr} "
            f"max_ms={max_repr}"
        )
    for result in report.results:
        print(
            "engage_attempt="
            f"{result.cycle_id} "
            f"outcome={result.outcome.value} "
            f"reason={result.reason} "
            f"selected_target={result.selected_target_id} "
            f"final_target={result.final_target_id} "
            f"click_xy={None if result.click_screen_xy is None else list(result.click_screen_xy)} "
            f"detection_latency_ms={_format_optional_ms(result.detection_latency_ms)} "
            f"selection_latency_ms={_format_optional_ms(result.selection_latency_ms)} "
            f"total_reaction_latency_ms={_format_optional_ms(result.total_reaction_latency_ms)} "
            f"verification_latency_ms={_format_optional_ms(result.verification_latency_ms)}"
        )


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.3f}"


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
