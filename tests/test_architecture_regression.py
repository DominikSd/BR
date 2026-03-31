from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_do_not_import_obsolete_namespaces() -> None:
    forbidden_patterns = (
        "from botlab.core",
        "import botlab.core",
        "from botlab.simulation",
        "import botlab.simulation",
        "from botlab.telemetry",
        "import botlab.telemetry",
    )
    search_roots = (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
    )

    offenders: list[str] = []

    for root in search_roots:
        for path in root.rglob("*.py"):
            if path.name == "test_architecture_regression.py":
                continue
            content = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                if pattern in content:
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)} -> {pattern}")

    assert offenders == []
