from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from botlab.config import Settings, load_config, load_default_config
from botlab.adapters.simulation.runner import SimulationReport, SimulationRunner


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
        help="Ścieżka do pliku YAML z konfiguracją. Domyślnie: config/default.yaml",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Liczba cykli do zasymulowania. Domyślnie: 10",
    )
    parser.add_argument(
        "--anchor-spawn-ts",
        type=float,
        default=100.0,
        help="Początkowy znany timestamp spawnu dla bootstrapu predictora. Domyślnie: 100.0",
    )
    parser.add_argument(
        "--anchor-cycle-id",
        type=int,
        default=0,
        help="Początkowy cycle_id dla bootstrapu predictora. Domyślnie: 0",
    )
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Włącza logowanie telemetry również na konsolę.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    settings = _load_settings(args.config)
    _validate_runtime_args(cycles=args.cycles)

    runner = SimulationRunner.from_settings(
        settings,
        initial_anchor_spawn_ts=args.anchor_spawn_ts,
        initial_anchor_cycle_id=args.anchor_cycle_id,
        enable_console=args.console_log,
    )

    report = runner.run_cycles(args.cycles)
    _print_report(settings=settings, report=report)

    return 0


def _load_settings(config_path: str | None) -> Settings:
    if config_path is None:
        return load_default_config()

    return load_config(config_path)


def _validate_runtime_args(*, cycles: int) -> None:
    if cycles <= 0:
        raise ValueError("--cycles musi być większe od 0.")


def _print_report(*, settings: Settings, report: SimulationReport) -> None:
    print(f"app.name={settings.app.name}")
    print(f"app.mode={settings.app.mode}")
    print(f"source_config={settings.source_path}")
    print(f"total_cycles={report.total_cycles}")
    print(f"success={report.count_result('success')}")
    print(f"failure={report.count_result('failure')}")
    print(f"no_event={report.count_result('no_event')}")
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


if __name__ == "__main__":
    raise SystemExit(main())
