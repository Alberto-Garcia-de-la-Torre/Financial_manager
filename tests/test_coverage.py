"""Day 13: the coverage table after the full backfill.

The acceptance criterion is that a coverage table prints first date, last date
and row count for every ticker. "Every" is the part worth testing: a ticker the
backfill never got onto disk has to appear as a gap, not vanish from the table.

Offline and inside `tmp_path`, like the rest of the suite.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data.store import COVERAGE_COLUMNS, coverage, write_bars


def bars(ticker: str, start: str, sessions: int) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=sessions)
    prices = np.linspace(10.0, 20.0, sessions)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": dates.date,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": 1_000.0,
        }
    )


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "bars" / "daily"
    write_bars(bars("AAPL", "2010-01-04", 30), root=root)
    write_bars(bars("SAN.MC", "2020-06-01", 12), root=root)
    return root


def test_coverage_reports_first_last_and_rows_per_ticker(store: Path) -> None:
    frame = coverage(root=store)

    assert tuple(frame.columns) == COVERAGE_COLUMNS
    assert frame["ticker"].tolist() == ["AAPL", "SAN.MC"]
    aapl, san = frame.to_dict("records")
    assert (aapl["first_date"], aapl["last_date"], aapl["rows"]) == (
        dt.date(2010, 1, 4),
        dt.date(2010, 2, 12),
        30,
    )
    assert (san["first_date"], san["last_date"], san["rows"]) == (
        dt.date(2020, 6, 1),
        dt.date(2020, 6, 16),
        12,
    )


def test_a_ticker_with_nothing_stored_is_kept_as_a_gap(store: Path) -> None:
    frame = coverage(["SAN.MC", "DEAD.XX", "AAPL"], root=store)

    assert frame["ticker"].tolist() == ["SAN.MC", "DEAD.XX", "AAPL"]
    gap = frame.set_index("ticker").loc["DEAD.XX"]
    assert gap["rows"] == 0
    assert pd.isna(gap["first_date"]) and pd.isna(gap["last_date"])


def test_an_empty_store_still_lists_every_ticker_asked(tmp_path: Path) -> None:
    frame = coverage(["AAPL", "SAN.MC"], root=tmp_path)

    assert frame["rows"].tolist() == [0, 0]
    assert coverage(root=tmp_path).empty


def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe: list[str]) -> CliRunner:
    universe_file = tmp_path / "universe.csv"
    universe_file.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        + "".join(f"{t},{t},X,EUR,Energy,ES\n" for t in universe),
        encoding="utf-8",
    )
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\n"
        f"universe_path: {universe_file}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    return CliRunner()


@pytest.mark.usefixtures("store")
def test_the_cli_prints_a_row_for_every_universe_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = cli(tmp_path, monkeypatch, ["AAPL", "SAN.MC"]).invoke(
        app, ["coverage"], terminal_width=200
    )

    assert result.exit_code == 0, result.output
    for expected in ("AAPL", "2010-01-04", "2010-02-12", "30", "SAN.MC", "2020-06-16", "12"):
        assert expected in result.output
    assert "2 of 2 tickers stored; 42 bars, 2010-01-04 to 2020-06-16" in result.output


@pytest.mark.usefixtures("store")
def test_the_cli_fails_when_a_universe_ticker_has_no_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = cli(tmp_path, monkeypatch, ["AAPL", "DEAD.XX", "SAN.MC"]).invoke(
        app, ["coverage"], terminal_width=200
    )

    assert result.exit_code == 1
    assert "DEAD.XX" in result.output
    assert "2 of 3 tickers stored" in result.output
