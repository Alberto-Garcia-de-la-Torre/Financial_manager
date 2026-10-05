"""Day 15: splits and dividends, ingested into their own store.

The acceptance criterion — Apple's 4:1 split on 2020-08-31, printed out of the
store — is `test_aapl_split_round_trips_through_the_store` and, end to end
through the CLI, `test_cli_show_prints_the_aapl_split`. Both run on the saved
Yahoo responses in `tests/fixtures/yahoo/`; nothing here reaches the network.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yahoo_fixtures
from typer.testing import CliRunner

from finmgr.data import actions, fetch
from finmgr.data.actions import (
    ACTION_COLUMNS,
    ACTION_DTYPES,
    conform_actions,
    describe,
    fetch_actions,
    ingest_actions,
    merge_actions,
    read_actions,
    to_actions,
)

AAPL = "AAPL_2020-08-24_2020-09-04"
SANTANDER = "SAN.MC_2024-10-28_2024-11-11"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(fetch, "download_history", refuse)


def from_fixture(name: str):
    raw = yahoo_fixtures.load_raw(name)
    return lambda *_: raw.copy()


def fixture_fetcher(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    name = {"AAPL": AAPL, "SAN.MC": SANTANDER}[ticker]
    return fetch_actions(ticker, start, end, download=from_fixture(name))


def test_to_actions_finds_the_aapl_split_and_nothing_else() -> None:
    found = to_actions("AAPL", yahoo_fixtures.load_raw(AAPL))

    assert found.to_dict("records") == [
        {"ticker": "AAPL", "date": dt.date(2020, 8, 31), "action": "split", "value": 4.0}
    ]
    assert dict(found.dtypes.astype(str)) == ACTION_DTYPES


def test_to_actions_dates_a_madrid_dividend_on_its_local_session() -> None:
    found = to_actions("SAN.MC", yahoo_fixtures.load_raw(SANTANDER))

    assert found[["date", "action", "value"]].to_dict("records") == [
        {"date": dt.date(2024, 10, 30), "action": "dividend", "value": 0.1}
    ]


def test_to_actions_without_action_columns_is_empty() -> None:
    raw = yahoo_fixtures.load_raw(AAPL).drop(columns=["Dividends", "Stock Splits"])

    found = to_actions("AAPL", raw)

    assert found.empty
    assert list(found.columns) == list(ACTION_COLUMNS)


def test_aapl_split_round_trips_through_the_store(tmp_path: Path) -> None:
    found = fetch_actions("AAPL", "2020-08-24", "2020-09-04", download=from_fixture(AAPL))
    merge_actions(found, root=tmp_path)

    split = read_actions("AAPL", action="split", root=tmp_path)

    assert (tmp_path / "ticker=AAPL" / "part.parquet").is_file()
    assert len(split) == 1
    row = split.iloc[0]
    assert (row["date"], describe(row["action"], row["value"])) == (dt.date(2020, 8, 31), "4:1")
    assert dict(split.dtypes.astype(str)) == ACTION_DTYPES


def test_fetch_actions_honours_the_window() -> None:
    found = fetch_actions("AAPL", "2020-09-01", "2020-09-04", download=from_fixture(AAPL))

    assert found.empty


def test_merge_is_idempotent_and_keeps_what_the_source_forgot(tmp_path: Path) -> None:
    def frame(*rows: tuple[str, str, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [("X", dt.date.fromisoformat(d), a, v) for d, a, v in rows],
            columns=list(ACTION_COLUMNS),
        )

    first = merge_actions(
        frame(("2020-01-02", "dividend", 0.5), ("2020-01-02", "split", 2.0)), root=tmp_path
    )
    again = merge_actions(frame(("2020-01-02", "dividend", 0.5)), root=tmp_path)
    revised = merge_actions(frame(("2020-01-02", "dividend", 0.55)), root=tmp_path)

    assert (first[0].added, first[0].path is not None) == (2, True)
    assert (again[0].added, again[0].updated, again[0].path) == (0, 0, None)
    assert (revised[0].added, revised[0].updated) == (0, 1)
    stored = read_actions("X", root=tmp_path)
    assert stored[["action", "value"]].to_dict("records") == [
        {"action": "dividend", "value": 0.55},
        {"action": "split", "value": 2.0},
    ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"action": "merger"}, "unknown action"),
        ({"value": 0.0}, "non-positive"),
        ({"date": None}, "no date"),
    ],
)
def test_conform_refuses_bad_rows(change: dict, message: str) -> None:
    row = {"ticker": "X", "date": dt.date(2020, 1, 2), "action": "split", "value": 2.0} | change

    with pytest.raises(ValueError, match=message):
        conform_actions(pd.DataFrame([row]))


def test_conform_refuses_a_repeated_action() -> None:
    row = {"ticker": "X", "date": dt.date(2020, 1, 2), "action": "split", "value": 2.0}

    with pytest.raises(ValueError, match="repeat"):
        conform_actions(pd.DataFrame([row, row]))


@pytest.mark.parametrize(
    ("action", "value", "text"),
    [
        ("split", 4.0, "4:1"),
        ("split", 1.5, "1.5:1"),
        ("split", 0.1, "1:10"),
        ("dividend", 0.24, "0.24"),
    ],
)
def test_describe(action: str, value: float, text: str) -> None:
    assert describe(action, value) == text


def test_ingest_survives_a_bad_ticker(tmp_path: Path) -> None:
    results = ingest_actions(
        ["AAPL", "NOPE", "SAN.MC"],
        start="2020-01-01",
        end="2024-12-31",
        attempts=2,
        pause=0,
        root=tmp_path,
        fetch=fixture_fetcher,
        sleep=lambda _: None,
    )

    by_ticker = {result.ticker: result for result in results}
    assert (by_ticker["AAPL"].splits, by_ticker["AAPL"].added) == (1, 1)
    assert (by_ticker["SAN.MC"].dividends, by_ticker["SAN.MC"].added) == (1, 1)
    assert not by_ticker["NOPE"].ok
    assert by_ticker["NOPE"].attempts == 2
    assert set(read_actions(root=tmp_path)["ticker"]) == {"AAPL", "SAN.MC"}


def test_cli_show_prints_the_aapl_split(
    tmp_path: Path, write_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finmgr import cli
    from finmgr.config import SETTINGS_PATH_ENV, load_settings

    values = {"data_dir": str(tmp_path), "log_dir": str(tmp_path / "logs")}
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(write_settings(values)))
    found = fetch_actions("AAPL", "2020-08-24", "2020-09-04", download=from_fixture(AAPL))
    merge_actions(found, settings=load_settings())
    assert actions.actions_root(settings=load_settings()) == tmp_path / "actions"

    result = CliRunner().invoke(cli.app, ["actions", "--show", "-t", "AAPL", "--kind", "split"])

    assert result.exit_code == 0, result.output
    assert "2020-08-31" in result.output
    assert "4:1" in result.output
