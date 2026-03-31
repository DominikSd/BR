from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path() -> Path:
    root = Path(__file__).resolve().parents[1] / "data" / "test-runtime-temp"
    root.mkdir(parents=True, exist_ok=True)
    temp_dir = root / f"case-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
