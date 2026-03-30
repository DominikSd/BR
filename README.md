# botlab

Minimalny, testowalny szkielet projektu w Pythonie do budowy stabilnego rdzenia logicznego
dla cyklicznego systemu PvE w symulatorze / własnej grze.

Na tym etapie repozytorium zawiera wyłącznie bootstrap projektu:
- strukturę katalogów,
- podstawowy pakiet Python,
- plik konfiguracyjny YAML,
- minimalny test weryfikujący poprawność bootstrapu.

## Założenia projektu

Docelowo system ma obsługiwać cykliczne zdarzenia pojawiające się mniej więcej co 45 sekund
w modelu:

1. przewidywanie kolejnego cyklu,
2. przygotowanie przed cyklem,
3. wejście w okno gotowości,
4. szybka próba reakcji,
5. weryfikacja wyniku,
6. przejście do walki albo oczekiwania na kolejny cykl,
7. odpoczynek / regeneracja,
8. bezpieczne odzyskiwanie po błędach.

## Status etapu

Aktualny etap: **Etap 1 — bootstrap projektu**

Zrealizowane:
- `pyproject.toml`
- `README.md`
- `config/default.yaml`
- podstawowe `__init__.py`
- minimalny test bootstrapu

Jeszcze niezaimplementowane:
- modele danych,
- loader konfiguracji,
- telemetry,
- SQLite storage,
- predictor,
- scheduler,
- FSM,
- symulacja,
- battle/rest/recovery,
- integracja `main.py`.

## Struktura repozytorium

```text
project/
├── pyproject.toml
├── README.md
├── config/
│   └── default.yaml
├── data/
│   └── telemetry/
├── logs/
├── src/
│   └── botlab/
│       ├── __init__.py
│       ├── api/
│       │   └── __init__.py
│       ├── core/
│       │   └── __init__.py
│       ├── simulation/
│       │   └── __init__.py
│       ├── telemetry/
│       │   └── __init__.py
│       └── vision/
│           └── __init__.py
└── tests/
    └── test_bootstrap.py
```

## 3) `config/default.yaml`

```yaml
app:
  name: "botlab"
  mode: "simulation"

cycle:
  interval_s: 45.0
  prepare_before_s: 5.0
  ready_before_s: 1.0
  ready_after_s: 1.0
  verify_timeout_s: 0.5
  recover_timeout_s: 2.0

combat:
  low_hp_threshold: 0.35
  rest_start_threshold: 0.50
  rest_stop_threshold: 0.90

telemetry:
  sqlite_path: "data/telemetry/botlab.sqlite3"
  log_path: "logs/botlab.log"
  log_level: "INFO"

vision:
  enabled: false
```

## 4) `src/botlab/__init__.py`

```python
"""
botlab

Minimalny pakiet startowy projektu do budowy rdzenia logicznego
cyklicznego systemu PvE w symulatorze / własnej grze.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
```

## 5) Katalogi i puste pakiety modułów

- `src/botlab/api/__init__.py`
- `src/botlab/core/__init__.py`
- `src/botlab/simulation/__init__.py`
- `src/botlab/telemetry/__init__.py`
- `src/botlab/vision/__init__.py`

## 6) `tests/test_bootstrap.py`

```python
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
```
