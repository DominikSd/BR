from __future__ import annotations

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SRC_ROOT: Final[Path] = PROJECT_ROOT / "src"
PACKAGE_ROOT: Final[Path] = SRC_ROOT / "botlab"

CONFIG_DIRECTORY: Final[Path] = PROJECT_ROOT / "config"
DATA_DIRECTORY: Final[Path] = PROJECT_ROOT / "data"
TELEMETRY_DATA_DIRECTORY: Final[Path] = DATA_DIRECTORY / "telemetry"
LOGS_DIRECTORY: Final[Path] = PROJECT_ROOT / "logs"

DEFAULT_CONFIG_FILENAME: Final[str] = "default.yaml"
DEFAULT_CONFIG_PATH: Final[Path] = CONFIG_DIRECTORY / DEFAULT_CONFIG_FILENAME
DEFAULT_COMBAT_PLANS_FILENAME: Final[str] = "combat_plans.yaml"
DEFAULT_COMBAT_PLANS_PATH: Final[Path] = CONFIG_DIRECTORY / DEFAULT_COMBAT_PLANS_FILENAME
DEFAULT_COMBAT_PROFILES_FILENAME: Final[str] = "combat_profiles.yaml"
DEFAULT_COMBAT_PROFILES_PATH: Final[Path] = CONFIG_DIRECTORY / DEFAULT_COMBAT_PROFILES_FILENAME

DEFAULT_APP_NAME: Final[str] = "botlab"
DEFAULT_APP_MODE: Final[str] = "simulation"

DEFAULT_CYCLE_INTERVAL_S: Final[float] = 45.0
DEFAULT_PREPARE_BEFORE_S: Final[float] = 5.0
DEFAULT_READY_BEFORE_S: Final[float] = 1.0
DEFAULT_READY_AFTER_S: Final[float] = 1.0
DEFAULT_VERIFY_TIMEOUT_S: Final[float] = 0.5
DEFAULT_RECOVER_TIMEOUT_S: Final[float] = 2.0

DEFAULT_LOW_HP_THRESHOLD: Final[float] = 0.35
DEFAULT_REST_START_THRESHOLD: Final[float] = 0.50
DEFAULT_REST_STOP_THRESHOLD: Final[float] = 0.90

DEFAULT_SQLITE_RELATIVE_PATH: Final[str] = "data/telemetry/botlab.sqlite3"
DEFAULT_LOG_RELATIVE_PATH: Final[str] = "logs/botlab.log"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

DEFAULT_VISION_ENABLED: Final[bool] = False

REQUIRED_CONFIG_SECTIONS: Final[tuple[str, ...]] = (
    "app",
    "cycle",
    "combat",
    "telemetry",
    "vision",
)

VALID_LOG_LEVELS: Final[tuple[str, ...]] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)
