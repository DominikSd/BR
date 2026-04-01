from __future__ import annotations

from dataclasses import dataclass, field
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
    default_profile_name: str | None = None


@dataclass(slots=True, frozen=True)
class TelemetryConfig:
    sqlite_path: Path
    log_path: Path
    log_level: str


@dataclass(slots=True, frozen=True)
class VisionConfig:
    enabled: bool


@dataclass(slots=True, frozen=True)
class LiveConfig:
    dry_run: bool = False
    foreground_only: bool = True
    window_title: str = "Game Window"
    capture_region: tuple[int, int, int, int] = (0, 0, 1280, 720)
    spawn_roi: tuple[int, int, int, int] = (320, 140, 640, 320)
    hp_bar_roi: tuple[int, int, int, int] = (40, 40, 220, 18)
    condition_bar_roi: tuple[int, int, int, int] = (40, 68, 220, 18)
    combat_indicator_roi: tuple[int, int, int, int] = (560, 620, 160, 60)
    reward_roi: tuple[int, int, int, int] = (500, 120, 260, 120)
    debug_directory: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "live_debug")
    save_frames: bool = True
    save_overlays: bool = True
    stall_timeout_s: float = 1.0
    dry_run_profile: str = "single_spot_mvp"
    perception_confidence_threshold: float = 0.75
    occupied_confidence_threshold: float = 0.75
    merge_distance_px: int = 28
    sample_frames_directory: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "live" / "sample_frames" / "raw"
    )
    mobs_template_directory: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "live" / "templates" / "mobs"
    )
    occupied_template_directory: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets" / "live" / "templates" / "occupied"
    )
    template_match_stride_px: int = 4
    template_rotations_deg: tuple[int, ...] = (0, 90, 180, 270)
    marker_min_red: int = 170
    marker_red_green_delta: int = 35
    marker_red_blue_delta: int = 25
    marker_min_blob_pixels: int = 6
    marker_max_blob_pixels: int = 180
    marker_min_width_px: int = 3
    marker_max_width_px: int = 36
    marker_min_height_px: int = 3
    marker_max_height_px: int = 36
    marker_confidence_threshold: float = 0.55
    swords_min_green: int = 120
    swords_green_red_delta: int = 20
    swords_green_blue_delta: int = 10
    swords_min_blob_pixels: int = 2
    swords_max_blob_pixels: int = 220
    swords_confidence_threshold: float = 0.25
    occupied_template_match_min_green_ratio: float = 0.01
    occupied_local_roi_width_px: int = 64
    occupied_local_roi_height_px: int = 72
    occupied_local_roi_offset_y_px: int = -42
    confirmation_roi_width_px: int = 88
    confirmation_roi_height_px: int = 120
    confirmation_roi_offset_y_px: int = 4
    confirmation_confidence_threshold: float = 0.60
    confirmation_alignment_weight: float = 0.25
    confirmation_foreground_weight: float = 0.10
    confirmation_max_horizontal_offset_px: int = 56
    confirmation_min_vertical_offset_px: int = 12
    confirmation_max_vertical_offset_px: int = 180
    candidate_confirmation_frames: int = 1
    candidate_loss_frames: int = 2
    occupied_confirmation_frames: int = 1
    preview_refresh_interval_ms: int = 120
    preview_max_width_px: int = 1600
    preview_max_height_px: int = 900


@dataclass(slots=True, frozen=True)
class Settings:
    app: AppConfig
    cycle: CycleConfig
    combat: CombatConfig
    telemetry: TelemetryConfig
    vision: VisionConfig
    source_path: Path
    live: LiveConfig = field(default_factory=LiveConfig)


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
    live_section = _optional_mapping(raw_data, "live", {})

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
        default_profile_name=_optional_nullable_str(combat_section, "default_profile_name", None),
    )
    _validate_combat_config(combat_config)

    telemetry_config = TelemetryConfig(
        sqlite_path=_resolve_project_relative_path(_require_str(telemetry_section, "sqlite_path")),
        log_path=_resolve_project_relative_path(_require_str(telemetry_section, "log_path")),
        log_level=_normalize_log_level(
            _optional_str(telemetry_section, "log_level", DEFAULT_LOG_LEVEL)
        ),
    )

    vision_config = VisionConfig(
        enabled=_require_bool(vision_section, "enabled"),
    )

    live_config = LiveConfig(
        dry_run=_optional_bool(live_section, "dry_run", False),
        foreground_only=_optional_bool(live_section, "foreground_only", True),
        window_title=_optional_str(live_section, "window_title", "Game Window"),
        capture_region=_optional_int_quad(live_section, "capture_region", (0, 0, 1280, 720)),
        spawn_roi=_optional_int_quad(live_section, "spawn_roi", (320, 140, 640, 320)),
        hp_bar_roi=_optional_int_quad(live_section, "hp_bar_roi", (40, 40, 220, 18)),
        condition_bar_roi=_optional_int_quad(live_section, "condition_bar_roi", (40, 68, 220, 18)),
        combat_indicator_roi=_optional_int_quad(
            live_section,
            "combat_indicator_roi",
            (560, 620, 160, 60),
        ),
        reward_roi=_optional_int_quad(live_section, "reward_roi", (500, 120, 260, 120)),
        debug_directory=_resolve_project_relative_path(
            _optional_str(live_section, "debug_directory", "data/live_debug")
        ),
        save_frames=_optional_bool(live_section, "save_frames", True),
        save_overlays=_optional_bool(live_section, "save_overlays", True),
        stall_timeout_s=_optional_positive_float(live_section, "stall_timeout_s", 1.0),
        dry_run_profile=_optional_str(live_section, "dry_run_profile", "single_spot_mvp"),
        perception_confidence_threshold=_optional_ratio_float(
            live_section,
            "perception_confidence_threshold",
            0.75,
        ),
        occupied_confidence_threshold=_optional_ratio_float(
            live_section,
            "occupied_confidence_threshold",
            0.75,
        ),
        merge_distance_px=_optional_positive_int(live_section, "merge_distance_px", 28),
        sample_frames_directory=_resolve_project_relative_path(
            _optional_str(live_section, "sample_frames_directory", "assets/live/sample_frames/raw")
        ),
        mobs_template_directory=_resolve_project_relative_path(
            _optional_str(live_section, "mobs_template_directory", "assets/live/templates/mobs")
        ),
        occupied_template_directory=_resolve_project_relative_path(
            _optional_str(live_section, "occupied_template_directory", "assets/live/templates/occupied")
        ),
        template_match_stride_px=_optional_positive_int(
            live_section,
            "template_match_stride_px",
            4,
        ),
        template_rotations_deg=_optional_int_list(
            live_section,
            "template_rotations_deg",
            (0, 90, 180, 270),
        ),
        marker_min_red=_optional_int_range(live_section, "marker_min_red", 170, min_value=0, max_value=255),
        marker_red_green_delta=_optional_int_range(
            live_section,
            "marker_red_green_delta",
            35,
            min_value=0,
            max_value=255,
        ),
        marker_red_blue_delta=_optional_int_range(
            live_section,
            "marker_red_blue_delta",
            25,
            min_value=0,
            max_value=255,
        ),
        marker_min_blob_pixels=_optional_positive_int(live_section, "marker_min_blob_pixels", 6),
        marker_max_blob_pixels=_optional_positive_int(live_section, "marker_max_blob_pixels", 180),
        marker_min_width_px=_optional_positive_int(live_section, "marker_min_width_px", 3),
        marker_max_width_px=_optional_positive_int(live_section, "marker_max_width_px", 36),
        marker_min_height_px=_optional_positive_int(live_section, "marker_min_height_px", 3),
        marker_max_height_px=_optional_positive_int(live_section, "marker_max_height_px", 36),
        marker_confidence_threshold=_optional_ratio_float(
            live_section,
            "marker_confidence_threshold",
            0.55,
        ),
        swords_min_green=_optional_int_range(
            live_section,
            "swords_min_green",
            120,
            min_value=0,
            max_value=255,
        ),
        swords_green_red_delta=_optional_int_range(
            live_section,
            "swords_green_red_delta",
            20,
            min_value=0,
            max_value=255,
        ),
        swords_green_blue_delta=_optional_int_range(
            live_section,
            "swords_green_blue_delta",
            10,
            min_value=0,
            max_value=255,
        ),
        swords_min_blob_pixels=_optional_positive_int(
            live_section,
            "swords_min_blob_pixels",
            2,
        ),
        swords_max_blob_pixels=_optional_positive_int(
            live_section,
            "swords_max_blob_pixels",
            220,
        ),
        swords_confidence_threshold=_optional_ratio_float(
            live_section,
            "swords_confidence_threshold",
            0.25,
        ),
        occupied_template_match_min_green_ratio=_optional_ratio_float(
            live_section,
            "occupied_template_match_min_green_ratio",
            0.01,
        ),
        occupied_local_roi_width_px=_optional_positive_int(
            live_section,
            "occupied_local_roi_width_px",
            64,
        ),
        occupied_local_roi_height_px=_optional_positive_int(
            live_section,
            "occupied_local_roi_height_px",
            72,
        ),
        occupied_local_roi_offset_y_px=_optional_int(
            live_section,
            "occupied_local_roi_offset_y_px",
            -42,
        ),
        confirmation_roi_width_px=_optional_positive_int(
            live_section,
            "confirmation_roi_width_px",
            88,
        ),
        confirmation_roi_height_px=_optional_positive_int(
            live_section,
            "confirmation_roi_height_px",
            120,
        ),
        confirmation_roi_offset_y_px=_optional_int(
            live_section,
            "confirmation_roi_offset_y_px",
            4,
        ),
        confirmation_confidence_threshold=_optional_ratio_float(
            live_section,
            "confirmation_confidence_threshold",
            0.60,
        ),
        confirmation_alignment_weight=_optional_ratio_float(
            live_section,
            "confirmation_alignment_weight",
            0.25,
        ),
        confirmation_foreground_weight=_optional_ratio_float(
            live_section,
            "confirmation_foreground_weight",
            0.10,
        ),
        confirmation_max_horizontal_offset_px=_optional_positive_int(
            live_section,
            "confirmation_max_horizontal_offset_px",
            56,
        ),
        confirmation_min_vertical_offset_px=_optional_positive_int(
            live_section,
            "confirmation_min_vertical_offset_px",
            12,
        ),
        confirmation_max_vertical_offset_px=_optional_positive_int(
            live_section,
            "confirmation_max_vertical_offset_px",
            180,
        ),
        candidate_confirmation_frames=_optional_positive_int(
            live_section,
            "candidate_confirmation_frames",
            1,
        ),
        candidate_loss_frames=_optional_positive_int(
            live_section,
            "candidate_loss_frames",
            2,
        ),
        occupied_confirmation_frames=_optional_positive_int(
            live_section,
            "occupied_confirmation_frames",
            1,
        ),
        preview_refresh_interval_ms=_optional_positive_int(
            live_section,
            "preview_refresh_interval_ms",
            120,
        ),
        preview_max_width_px=_optional_positive_int(
            live_section,
            "preview_max_width_px",
            1600,
        ),
        preview_max_height_px=_optional_positive_int(
            live_section,
            "preview_max_height_px",
            900,
        ),
    )

    return Settings(
        app=app_config,
        cycle=cycle_config,
        combat=combat_config,
        telemetry=telemetry_config,
        vision=vision_config,
        source_path=path,
        live=live_config,
    )


def _read_yaml(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ConfigError(f"Plik konfiguracji jest pusty: {config_path}")

    if not isinstance(data, dict):
        raise ConfigError(
            f"Nieprawidlowy format YAML w pliku {config_path}. Oczekiwano mapy klucz-wartosc."
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
        raise ConfigError(f"Sekcja '{key}' musi byc mapa.")
    return value


def _optional_mapping(
    data: Mapping[str, Any],
    key: str,
    default: Mapping[str, Any],
) -> Mapping[str, Any]:
    value = data.get(key, default)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Sekcja '{key}' musi byc mapa.")
    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Pole '{key}' musi byc niepustym napisem.")
    return value.strip()


def _optional_str(data: Mapping[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Pole '{key}' musi byc niepustym napisem.")
    return value.strip()


def _optional_positive_float(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Pole '{key}' musi byc liczba.")
    numeric_value = float(value)
    if numeric_value <= 0.0:
        raise ConfigError(f"Pole '{key}' musi byc wieksze od 0.")
    return numeric_value


def _optional_positive_int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ConfigError(f"Pole '{key}' musi byc liczba calkowita.")
    if value <= 0:
        raise ConfigError(f"Pole '{key}' musi byc wieksze od 0.")
    return value


def _optional_int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ConfigError(f"Pole '{key}' musi byc liczba calkowita.")
    return value


def _optional_int_range(
    data: Mapping[str, Any],
    key: str,
    default: int,
    *,
    min_value: int,
    max_value: int,
) -> int:
    value = _optional_int(data, key, default)
    if not min_value <= value <= max_value:
        raise ConfigError(
            f"Pole '{key}' musi byc w zakresie od {min_value} do {max_value}."
        )
    return value


def _optional_ratio_float(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Pole '{key}' musi byc liczba.")
    numeric_value = float(value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ConfigError(f"Pole '{key}' musi byc w zakresie od 0.0 do 1.0.")
    return numeric_value


def _optional_int_list(
    data: Mapping[str, Any],
    key: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ConfigError(f"Pole '{key}' musi byc niepusta lista liczb calkowitych.")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ConfigError(f"Pole '{key}' musi zawierac liczby calkowite.")
        normalized = item % 360
        if normalized not in parsed:
            parsed.append(normalized)
    return tuple(parsed)


def _optional_nullable_str(
    data: Mapping[str, Any],
    key: str,
    default: str | None,
) -> str | None:
    value = data.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Pole '{key}' musi byc niepustym napisem albo null.")
    return value.strip()


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Pole '{key}' musi byc typu bool.")
    return value


def _optional_bool(data: Mapping[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"Pole '{key}' musi byc typu bool.")
    return value


def _require_positive_float(data: Mapping[str, Any], key: str) -> float:
    value = _coerce_float(data, key)
    if value <= 0.0:
        raise ConfigError(f"Pole '{key}' musi byc wieksze od 0.")
    return value


def _require_non_negative_float(data: Mapping[str, Any], key: str) -> float:
    value = _coerce_float(data, key)
    if value < 0.0:
        raise ConfigError(f"Pole '{key}' nie moze byc ujemne.")
    return value


def _require_ratio_float(data: Mapping[str, Any], key: str) -> float:
    value = _coerce_float(data, key)
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"Pole '{key}' musi byc w zakresie od 0.0 do 1.0.")
    return value


def _coerce_float(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"Pole '{key}' musi byc liczba.")
    return float(value)


def _normalize_log_level(value: str) -> str:
    normalized = value.upper()
    if normalized not in VALID_LOG_LEVELS:
        allowed = ", ".join(VALID_LOG_LEVELS)
        raise ConfigError(
            f"Nieprawidlowy poziom logowania '{value}'. Dozwolone wartosci: {allowed}."
        )
    return normalized


def _resolve_project_relative_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _optional_int_quad(
    data: Mapping[str, Any],
    key: str,
    default: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    value = data.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ConfigError(f"Pole '{key}' musi byc czworka [x, y, width, height].")
    parsed: list[int] = []
    for item in value:
        if not isinstance(item, int):
            raise ConfigError(f"Pole '{key}' musi zawierac liczby calkowite.")
        parsed.append(item)
    x, y, width, height = parsed
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ConfigError(
            f"Pole '{key}' musi zawierac nieujemne x/y i dodatnie width/height."
        )
    return (x, y, width, height)


def _validate_cycle_config(config: CycleConfig) -> None:
    if config.prepare_before_s >= config.interval_s:
        raise ConfigError("cycle.prepare_before_s musi byc mniejsze niz cycle.interval_s.")

    if config.ready_before_s > config.prepare_before_s:
        raise ConfigError(
            "cycle.ready_before_s nie moze byc wieksze niz cycle.prepare_before_s."
        )

    ready_window_size = config.ready_before_s + config.ready_after_s
    if ready_window_size >= config.interval_s:
        raise ConfigError(
            "Suma cycle.ready_before_s i cycle.ready_after_s musi byc mniejsza niz interval_s."
        )


def _validate_combat_config(config: CombatConfig) -> None:
    if config.rest_start_threshold < config.low_hp_threshold:
        raise ConfigError(
            "combat.rest_start_threshold nie moze byc mniejsze niz combat.low_hp_threshold."
        )

    if config.rest_stop_threshold < config.rest_start_threshold:
        raise ConfigError(
            "combat.rest_stop_threshold nie moze byc mniejsze niz combat.rest_start_threshold."
        )
