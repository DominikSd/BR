from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from botlab.constants import (
    DEFAULT_APP_MODE,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOG_LEVEL,
    PROJECT_ROOT,
    REQUIRED_CONFIG_SECTIONS,
    VALID_LOG_LEVELS,
)


class ConfigError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class AppConfig:
    name: str
    mode: str


@dataclass(slots=True, frozen=True)
class CycleConfig:
    interval_s: float
    prepare_before_s: float
    ready_before_s: float
    ready_after_s: float
    verify_timeout_s: float
    recover_timeout_s: float


@dataclass(slots=True, frozen=True)
class CombatConfig:
    low_hp_threshold: float
    rest_start_threshold: float
    rest_stop_threshold: float


@dataclass(slots=True, frozen=True)
class TelemetryConfig:
    sqlite_path: Path
    log_path: Path
    log_level: str


@dataclass(slots=True, frozen=True)
class VisionConfig:
    enabled: bool


@dataclass(slots=True, frozen=True)
class Settings:
    app: AppConfig
    cycle: CycleConfig
    combat: CombatConfig
    telemetry: TelemetryConfig
    vision: VisionConfig
    source_path: Path


def load_default_config() -> Settings:
    return load_config(DEFAULT_CONFIG_PATH)


def load_config(config_path: str | Path) -> Settings:
    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise ConfigError(f"Plik konfiguracji nie istnieje: {path}")

    raw_data = _read_yaml(path)
    _validate_required_sections(raw_data)

    app_section = _require_mapping(raw_data, "app")
    cycle_section = _require_mapping(raw_data, "cycle")
    combat_section = _require_mapping(raw_data, "combat")
    telemetry_section = _require_mapping(raw_data, "telemetry")
    vision_section = _require_mapping(raw_data, "vision")

    app_config = AppConfig(
        name=_require_str(app_section, "name"),
        mode=_optional_str(app_section, "mode", DEFAULT_APP_MODE),
    )

    cycle_config = CycleConfig(
        interval_s=_require_positive_float(cycle_section, "interval_s"),
        prepare_before_s=_require_non_negative_float(cycle_section, "prepare_before_s"),
        ready_before_s=_require_non_negative_float(cycle_section, "ready_before_s"),
        ready_after_s=_require_non_negative_float(cycle_section, "ready_after_s"),
        verify_timeout_s=_require_positive_float(cycle_section, "verify_timeout_s"),
        recover_timeout_s=_require_positive_float(cycle_section, "recover_timeout_s"),
    )
    _validate_cycle_config(cycle_config)

    combat_config = CombatConfig(
        low_hp_threshold=_require_ratio_float(combat_section, "low_hp_threshold"),
        rest_start_threshold=_require_ratio_float(combat_section, "rest_start_threshold"),
        rest_stop_threshold=_require_ratio_float(combat_section, "rest_stop_threshold"),
    )
    _validate_combat_config(combat_config)

    telemetry_config = TelemetryConfig(
        sqlite_path=_resolve_project_relative_path(_require_str(telemetry_section, "sqlite_path")),
        log_path=_resolve_project_relative_path(_require_str(telemetry_section, "log_path")),
        log_level=_normalize_log_level(_optional_str(telemetry_section, "log_level", DEFAULT_LOG_LEVEL)),
    )

    vision_config = VisionConfig(
        enabled=_require_bool(vision_section, "enabled"),
    )

    return Settings(
        app=app_config,
        cycle=cycle_config,
        combat=combat_config,
        telemetry=telemetry_config,
        vision=vision_config,
        source_path=path,
    )


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ConfigError(f"Plik konfiguracji jest pusty: {config_path}")

    if not isinstance(data, dict):
        raise ConfigError(
            f"Nieprawidłowy format YAML w pliku {config_path}. Oczekiwano mapy klucz-wartość."
        )

    return data


def _validate_required_sections(data: Mapping[str, Any]) -> None:
    missing_sections = [section for section in REQUIRED_CONFIG_SECTIONS if section not in data]
    if missing_sections:
        joined = ", ".join(missing_sections)
        raise ConfigError(f"Brak wymaganych sekcji konfiguracji: {joined}")


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Sekcja '{key}' musi być mapą.")
    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Pole '{key}' musi być niepustym napisem.")
    return value.strip()


def _optional_str(data: Mapping[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Pole '{key}' musi być niepustym napisem.")
    return value.strip()


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Pole '{key}' musi być typu bool.")
    return value


def _require_positive_float(data: Mapping[str, Any], key: str) -> float:
    value = _coerce_float(data, key)
    if value <= 0.0:
        raise ConfigError(f"Pole '{key}' musi być większe od 0.")
    return value


def _require_non_negative_float(data: Mapping[str, Any], key: str) -> float:
    value = _coerce_float(data, key)
    if value < 0.0:
        raise ConfigError(f"Pole '{key}' nie może być ujemne.")
    return value


def _require_ratio_float(data: Mapping[str, Any], key: str) -> float:
    value = _coerce_float(data, key)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"Pole '{key}' musi być w zakresie od 0.0 do 1.0.")
    return value


def _coerce_float(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Pole '{key}' musi być liczbą.")
    return float(value)


def _normalize_log_level(value: str) -> str:
    normalized = value.upper()
    if normalized not in VALID_LOG_LEVELS:
        allowed = ", ".join(VALID_LOG_LEVELS)
        raise ConfigError(
            f"Nieprawidłowy poziom logowania '{value}'. Dozwolone wartości: {allowed}."
        )
    return normalized


def _resolve_project_relative_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _validate_cycle_config(config: CycleConfig) -> None:
    if config.prepare_before_s >= config.interval_s:
        raise ConfigError("cycle.prepare_before_s musi być mniejsze niż cycle.interval_s.")

    if config.ready_before_s > config.prepare_before_s:
        raise ConfigError(
            "cycle.ready_before_s nie może być większe niż cycle.prepare_before_s."
        )

    ready_window_size = config.ready_before_s + config.ready_after_s
    if ready_window_size >= config.interval_s:
        raise ConfigError(
            "Suma cycle.ready_before_s i cycle.ready_after_s musi być mniejsza niż interval_s."
        )


def _validate_combat_config(config: CombatConfig) -> None:
    if config.rest_start_threshold < config.low_hp_threshold:
        raise ConfigError(
            "combat.rest_start_threshold nie może być mniejsze niż combat.low_hp_threshold."
        )

    if config.rest_stop_threshold < config.rest_start_threshold:
        raise ConfigError(
            "combat.rest_stop_threshold nie może być mniejsze niż combat.rest_start_threshold."
        )
