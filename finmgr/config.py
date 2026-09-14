"""Project configuration.

One `Settings` object, loaded from `config/settings.yaml`, is the single place
that answers "where is the data, which currency, which timezone, how far back".
Everything downstream reads it rather than hard-coding paths.

    from finmgr.config import load_settings
    settings = load_settings()
    settings.data_dir          # absolute Path
    settings.history_start     # datetime.date
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# The repo root: finmgr/config.py -> finmgr/ -> repo root. The package is
# installed editable, so this points at the checkout rather than site-packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

#: Environment variable pointing at an alternative settings file.
SETTINGS_PATH_ENV = "FINMGR_SETTINGS"


class Settings(BaseModel):
    """Validated project settings.

    Relative paths are resolved against :data:`PROJECT_ROOT`, so a run started
    from any working directory reads and writes the same files.
    """

    # validate_default so an omitted path still gets resolved against the repo
    # root; without it a defaulted data_dir stays relative and follows the
    # working directory around.
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    data_dir: Path = Field(
        default=Path("data"),
        description="Root of the market data store. Never committed to git.",
    )
    universe_path: Path = Field(
        default=Path("config/universe.csv"),
        description="CSV listing the companies the project tracks.",
    )
    base_currency: str = Field(
        default="EUR",
        description="Currency every price and P&L figure is expressed in.",
    )
    timezone: str = Field(
        default="Europe/Madrid",
        description="IANA timezone for timestamps, calendars and scheduling.",
    )
    history_start: date = Field(
        default=date(2010, 1, 1),
        description="Earliest date the backfill asks for.",
    )

    @field_validator("data_dir", "universe_path")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("base_currency")
    @classmethod
    def _currency_code(cls, value: str) -> str:
        code = value.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError(f"base_currency must be a 3-letter ISO code, got {value!r}")
        return code

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

    @field_validator("history_start")
    @classmethod
    def _not_in_the_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError(f"history_start {value} is in the future")
        return value

    def ensure_data_dir(self) -> Path:
        """Create the data directory if it isn't there yet and return it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


def settings_path(path: Path | str | None = None) -> Path:
    """Resolve which settings file to read.

    Explicit argument wins, then ``$FINMGR_SETTINGS``, then the default.
    """
    if path is not None:
        return Path(path).expanduser()
    from_env = os.environ.get(SETTINGS_PATH_ENV)
    if from_env:
        return Path(from_env).expanduser()
    return DEFAULT_SETTINGS_PATH


def load_settings(path: Path | str | None = None) -> Settings:
    """Load and validate settings from YAML.

    A missing key falls back to the model default; a missing file is an error,
    because silently running on defaults is how you end up wondering why the
    data landed somewhere unexpected.
    """
    resolved = settings_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"settings file not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping, got {type(raw).__name__}")

    return Settings(**raw)
