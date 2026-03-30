from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_paths_exist() -> None:
    required_paths = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "config" / "default.yaml",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "telemetry",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "src" / "botlab",
        PROJECT_ROOT / "src" / "botlab" / "__init__.py",
        PROJECT_ROOT / "src" / "botlab" / "core" / "__init__.py",
        PROJECT_ROOT / "src" / "botlab" / "telemetry" / "__init__.py",
        PROJECT_ROOT / "src" / "botlab" / "simulation" / "__init__.py",
        PROJECT_ROOT / "src" / "botlab" / "vision" / "__init__.py",
        PROJECT_ROOT / "src" / "botlab" / "api" / "__init__.py",
        PROJECT_ROOT / "tests" / "test_bootstrap.py",
    ]

    missing_paths = [path for path in required_paths if not path.exists()]
    assert missing_paths == []


def test_default_yaml_has_required_sections() -> None:
    config_path = PROJECT_ROOT / "config" / "default.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    assert isinstance(data, dict)
    assert "app" in data
    assert "cycle" in data
    assert "combat" in data
    assert "telemetry" in data
    assert "vision" in data


def test_package_can_be_imported() -> None:
    import botlab

    assert botlab.__version__ == "0.1.0"
