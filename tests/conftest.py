"""Shared test fixtures.

Two rules hold for every test in this suite: it never reads the developer's
real `config/settings.yaml` by accident, and it never writes inside the
repository. Both are enforced here rather than remembered test by test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from finmgr.config import SETTINGS_PATH_ENV


@pytest.fixture(autouse=True)
def _no_inherited_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset $FINMGR_SETTINGS so the shell can't change what a test loads."""
    monkeypatch.delenv(SETTINGS_PATH_ENV, raising=False)


@pytest.fixture
def write_settings(tmp_path: Path) -> Callable[..., Path]:
    """Return a helper writing a settings YAML file into a temp directory.

    Called with no arguments it writes an empty mapping, which is how the
    "every key is optional" case is exercised.
    """

    def _write(values: dict | None = None, name: str = "settings.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(values or {}), encoding="utf-8")
        return path

    return _write
