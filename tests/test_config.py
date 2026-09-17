"""The settings layer: YAML in, validated `Settings` out, directories on demand.

Everything downstream — the store, the feature panel, the daily run — asks
`load_settings()` where things live. If this is wrong, data lands somewhere
unexpected and the mistake is only noticed weeks later, so it is the first
thing the project tests.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from finmgr.config import (
    DEFAULT_SETTINGS_PATH,
    PROJECT_ROOT,
    SETTINGS_PATH_ENV,
    Settings,
    load_settings,
    settings_path,
)

# ---------------------------------------------------------------------------
# Loading from YAML
# ---------------------------------------------------------------------------


def test_every_value_comes_from_the_yaml(
    write_settings: Callable[..., Path], tmp_path: Path
) -> None:
    path = write_settings(
        {
            "data_dir": str(tmp_path / "store"),
            "universe_path": str(tmp_path / "universe.csv"),
            "log_dir": str(tmp_path / "logs"),
            "base_currency": "USD",
            "timezone": "America/New_York",
            "history_start": date(2015, 6, 30),
        }
    )

    settings = load_settings(path)

    assert settings.data_dir == tmp_path / "store"
    assert settings.universe_path == tmp_path / "universe.csv"
    assert settings.log_dir == tmp_path / "logs"
    assert settings.base_currency == "USD"
    assert settings.timezone == "America/New_York"
    assert settings.history_start == date(2015, 6, 30)


def test_omitted_keys_fall_back_to_defaults(write_settings: Callable[..., Path]) -> None:
    """An empty file is valid: every key has a default in the model."""
    settings = load_settings(write_settings())

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.universe_path == PROJECT_ROOT / "config" / "universe.csv"
    assert settings.log_dir == PROJECT_ROOT / "logs"
    assert settings.base_currency == "EUR"
    assert settings.timezone == "Europe/Madrid"
    assert settings.history_start == date(2010, 1, 1)


def test_a_partial_file_mixes_given_values_and_defaults(
    write_settings: Callable[..., Path],
) -> None:
    settings = load_settings(write_settings({"base_currency": "GBP"}))

    assert settings.base_currency == "GBP"
    assert settings.timezone == "Europe/Madrid"


def test_relative_paths_resolve_against_the_repository_root(
    write_settings: Callable[..., Path],
) -> None:
    """So a run started from any working directory reads the same files."""
    settings = load_settings(write_settings({"data_dir": "elsewhere/bars"}))

    assert settings.data_dir.is_absolute()
    assert settings.data_dir == PROJECT_ROOT / "elsewhere" / "bars"


def test_the_project_settings_file_loads() -> None:
    """The file the project actually ships with has to be valid too."""
    settings = load_settings(DEFAULT_SETTINGS_PATH)

    assert settings.base_currency == "EUR"
    assert settings.timezone == "Europe/Madrid"
    assert settings.data_dir == PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Which file gets read
# ---------------------------------------------------------------------------


def test_default_path_is_the_repo_config() -> None:
    """The autouse fixture clears $FINMGR_SETTINGS, so this is the fallback."""
    assert settings_path() == DEFAULT_SETTINGS_PATH


def test_env_var_overrides_the_default(
    write_settings: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_settings({"base_currency": "CHF"}, name="override.yaml")
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(path))

    assert settings_path() == path
    assert load_settings().base_currency == "CHF"


def test_explicit_path_beats_the_env_var(
    write_settings: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = write_settings({"base_currency": "SEK"}, name="explicit.yaml")
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(write_settings({"base_currency": "CHF"})))

    assert load_settings(explicit).base_currency == "SEK"


def test_a_missing_file_is_an_error_not_a_silent_default(tmp_path: Path) -> None:
    """Running on defaults you never asked for is how data lands in odd places."""
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nowhere.yaml")


def test_yaml_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("- data_dir\n- logs\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_settings(path)


# ---------------------------------------------------------------------------
# Directories created on demand
# ---------------------------------------------------------------------------


def test_ensure_data_dir_creates_the_directory(
    write_settings: Callable[..., Path], tmp_path: Path
) -> None:
    target = tmp_path / "store" / "bars"
    settings = load_settings(write_settings({"data_dir": str(target)}))
    assert not target.exists()

    created = settings.ensure_data_dir()

    assert created == target
    assert target.is_dir()


def test_ensure_data_dir_is_safe_to_call_twice(
    write_settings: Callable[..., Path], tmp_path: Path
) -> None:
    """The daily run calls it every time; an existing directory is not an error."""
    target = tmp_path / "store"
    settings = load_settings(write_settings({"data_dir": str(target)}))

    settings.ensure_data_dir()
    (target / "marker.txt").write_text("keep me", encoding="utf-8")
    settings.ensure_data_dir()

    assert (target / "marker.txt").read_text(encoding="utf-8") == "keep me"


def test_ensure_log_dir_creates_the_directory(
    write_settings: Callable[..., Path], tmp_path: Path
) -> None:
    target = tmp_path / "logs"
    settings = load_settings(write_settings({"log_dir": str(target)}))

    assert settings.ensure_log_dir() == target
    assert target.is_dir()


def test_loading_settings_creates_nothing_by_itself(
    write_settings: Callable[..., Path], tmp_path: Path
) -> None:
    """Directories appear when something needs them, not on import."""
    settings = load_settings(
        write_settings({"data_dir": str(tmp_path / "store"), "log_dir": str(tmp_path / "logs")})
    )

    assert not settings.data_dir.exists()
    assert not settings.log_dir.exists()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_currency_is_normalised_to_upper_case(write_settings: Callable[..., Path]) -> None:
    assert load_settings(write_settings({"base_currency": " eur "})).base_currency == "EUR"


@pytest.mark.parametrize("bad", ["EURO", "€", "12", ""])
def test_a_bad_currency_code_is_rejected(bad: str, write_settings: Callable[..., Path]) -> None:
    with pytest.raises(ValidationError, match="3-letter ISO code"):
        load_settings(write_settings({"base_currency": bad}))


def test_an_unknown_timezone_is_rejected(write_settings: Callable[..., Path]) -> None:
    with pytest.raises(ValidationError, match="unknown timezone"):
        load_settings(write_settings({"timezone": "Europe/Atlantis"}))


def test_a_future_history_start_is_rejected(write_settings: Callable[..., Path]) -> None:
    tomorrow = date.today() + timedelta(days=1)

    with pytest.raises(ValidationError, match="is in the future"):
        load_settings(write_settings({"history_start": tomorrow}))


def test_an_unknown_key_is_rejected(write_settings: Callable[..., Path]) -> None:
    """A typo in the YAML must fail loudly, not be quietly ignored."""
    with pytest.raises(ValidationError, match="data_directory"):
        load_settings(write_settings({"data_directory": "data"}))


def test_settings_are_immutable() -> None:
    """Nothing downstream may retarget the data directory mid-run."""
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.base_currency = "USD"
