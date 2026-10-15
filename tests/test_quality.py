"""Day 19: the data quality checks and the triage register.

Each check gets a frame with exactly one thing wrong and must flag exactly
that; a clean frame must flag nothing. Then the register: a rule triages the
flags it matches (by ticker or by venue) and no others, a rule matching nothing
is reported, and a malformed rule is refused. The end-to-end test runs
`finmgr check` over a store in `tmp_path` — offline, like every other test.
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
from finmgr.data import quality
from finmgr.data.actions import merge_actions
from finmgr.data.calendars import expected_sessions
from finmgr.data.quality import Flag, Severity, Verdict, check_ticker, load_triage, run_checks
from finmgr.data.store import write_bars

D = dt.date
SESSIONS = expected_sessions("XMAD", 2024, start=D(2024, 3, 1), end=D(2024, 3, 28))


def bars(ticker: str = "SAN.MC", days: list[dt.date] | None = None) -> pd.DataFrame:
    """A well-behaved series: a gentle random walk, every bar consistent."""
    days = list(SESSIONS if days is None else days)
    close = 10 * np.cumprod(1 + 0.01 * np.sin(np.arange(len(days))))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": days,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000.0,
        }
    )


def checks(flags: list[Flag]) -> list[tuple[str, dt.date]]:
    return [(flag.check, flag.date) for flag in flags]


# ---------------------------------------------------------------------------
# One check at a time
# ---------------------------------------------------------------------------


def test_a_clean_series_raises_nothing() -> None:
    assert check_ticker("SAN.MC", bars()) == []


@pytest.mark.parametrize("value", [np.nan, 0.0, -1.0])
def test_nan_zero_and_negative_prices_are_errors(value: float) -> None:
    frame = bars()
    frame.loc[3, "open"] = value

    flags = check_ticker("SAN.MC", frame)

    assert checks(flags) == [("bad_price", SESSIONS[3])]
    assert flags[0].severity is Severity.ERROR
    assert "open=" in flags[0].detail


def test_high_below_low_is_flagged_once_not_also_as_out_of_range() -> None:
    frame = bars()
    frame.loc[4, ["high", "low"]] = [9.0, 11.0]

    assert checks(check_ticker("SAN.MC", frame)) == [("high_below_low", SESSIONS[4])]


def test_close_outside_the_days_range() -> None:
    frame = bars()
    frame.loc[5, "close"] = frame.loc[5, "high"] * 1.02
    frame.loc[6, "close"] = frame.loc[6, "low"] * 0.98

    flags = check_ticker("SAN.MC", frame)

    assert checks(flags) == [
        ("close_out_of_range", SESSIONS[5]),
        ("close_out_of_range", SESSIONS[6]),
    ]
    assert {flag.severity for flag in flags} == {Severity.WARNING}


def test_a_close_on_the_high_is_not_out_of_range() -> None:
    frame = bars()
    frame["close"] = frame["high"]
    frame.loc[2, "close"] = frame.loc[2, "low"]

    assert check_ticker("SAN.MC", frame) == []


def test_duplicate_dates() -> None:
    frame = bars()
    frame = pd.concat([frame, frame.iloc[[7]]], ignore_index=True)

    flags = check_ticker("SAN.MC", frame)

    assert checks(flags) == [("duplicate_date", SESSIONS[7])]
    assert "2 bars" in flags[0].detail


def test_a_one_day_spike_flags_both_legs() -> None:
    frame = bars()
    frame.loc[8, ["open", "high", "low", "close"]] *= 2

    assert checks(check_ticker("SAN.MC", frame)) == [
        ("unexplained_move", SESSIONS[8]),
        ("unexplained_move", SESSIONS[9]),
    ]


def test_a_move_just_under_the_threshold_passes() -> None:
    frame = bars()
    frame.loc[10:, ["open", "high", "low", "close"]] *= 1.35

    assert check_ticker("SAN.MC", frame) == []


def test_a_split_explains_the_move_it_causes() -> None:
    # Unadjusted bars across a 4:1: a 75% overnight drop that is no loss.
    frame = bars()
    frame.loc[10:, ["open", "high", "low", "close"]] /= 4
    split = pd.Series([4.0], index=[SESSIONS[10]])

    assert check_ticker("SAN.MC", frame, split) == []
    assert checks(check_ticker("SAN.MC", frame)) == [("unexplained_move", SESSIONS[10])]


def test_a_split_on_a_closed_day_explains_the_next_sessions_move() -> None:
    monday = next(i for i, day in enumerate(SESSIONS) if i and day.weekday() == 0)
    frame = bars()
    frame.loc[monday:, ["open", "high", "low", "close"]] /= 4
    sunday = SESSIONS[monday] - dt.timedelta(days=1)

    assert check_ticker("SAN.MC", frame, pd.Series([4.0], index=[sunday])) == []


def test_a_split_does_not_excuse_a_move_it_cannot_account_for() -> None:
    frame = bars()
    frame.loc[10:, ["open", "high", "low", "close"]] /= 4
    flags = check_ticker("SAN.MC", frame, pd.Series([2.0], index=[SESSIONS[10]]))

    assert checks(flags) == [("unexplained_move", SESSIONS[10])]
    assert "split 2:1 does not explain it" in flags[0].detail


def test_five_flat_sessions_are_one_flag_four_are_none() -> None:
    frame = bars()
    frame.loc[3:7, "close"] = 10.0  # five sessions
    frame.loc[3:7, ["open", "high", "low"]] = 10.0

    flags = check_ticker("SAN.MC", frame)
    assert checks(flags) == [("flat_price", SESSIONS[3])]
    assert f"for 5 sessions through {SESSIONS[7]}" in flags[0].detail

    frame.loc[7, ["open", "high", "low", "close"]] = 10.5
    assert check_ticker("SAN.MC", frame) == []


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------


def write_register(path: Path, *rows: str) -> Path:
    path.write_text(
        "# a comment\ncheck,scope,start,end,verdict,reason\n" + "".join(r + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def test_rules_match_by_ticker_or_venue_within_their_dates(tmp_path: Path) -> None:
    rules = load_triage(
        write_register(
            tmp_path / "t.csv",
            "off_calendar,XMAD,2024-03-01,2024-03-31,bad-data,placeholder bars",
            "unexplained_move,SAN.MC,2024-03-05,2024-03-05,genuine,results day",
        )
    )
    flags = [
        Flag("off_calendar", "BBVA.MC", D(2024, 3, 29), ""),  # venue rule
        Flag("off_calendar", "AAPL", D(2024, 3, 29), ""),  # other venue
        Flag("unexplained_move", "SAN.MC", D(2024, 3, 5), ""),
        Flag("unexplained_move", "SAN.MC", D(2024, 3, 6), ""),  # outside the dates
        Flag("bad_price", "SAN.MC", D(2024, 3, 5), ""),  # other check
    ]
    venues = {"BBVA.MC": "XMAD", "SAN.MC": "XMAD", "AAPL": "XNAS"}

    out, unused = quality.apply_triage(flags, rules, venues)

    assert [flag.triage.verdict if flag.triage else None for flag in out] == [
        Verdict.BAD_DATA,
        None,
        Verdict.GENUINE,
        None,
        None,
    ]
    assert unused == []


def test_a_rule_that_matches_nothing_is_reported(tmp_path: Path) -> None:
    rules = load_triage(
        write_register(tmp_path / "t.csv", "flat_price,SAN.MC,2020-01-01,2020-01-01,genuine,x")
    )

    _, unused = quality.apply_triage([], rules, {"SAN.MC": "XMAD"})
    assert unused == rules

    # ... but only when that ticker was checked at all.
    assert quality.apply_triage([], rules, {"AAPL": "XNAS"})[1] == []


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("flat,SAN.MC,2024-01-01,2024-01-01,genuine,x", "unknown check"),
        ("flat_price,SAN.MC,2024-01-01,2024-01-01,fine,x", "verdict 'fine'"),
        ("flat_price,SAN.MC,2024-02-01,2024-01-01,genuine,x", "after end"),
        ("flat_price,SAN.MC,2024-01-01,2024-01-01,genuine,", "reason are required"),
    ],
)
def test_a_malformed_rule_is_refused_not_skipped(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_triage(write_register(tmp_path / "t.csv", row))


def test_a_missing_register_is_empty(tmp_path: Path) -> None:
    assert load_triage(tmp_path / "absent.csv") == []


def test_the_projects_register_loads() -> None:
    # The real config/quality_triage.csv: every rule must parse.
    assert load_triage(Path(__file__).parents[1] / "config" / "quality_triage.csv")


# ---------------------------------------------------------------------------
# The whole store, and the command
# ---------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A settings file, a two-ticker universe, and a store with one of everything."""
    (tmp_path / "universe.csv").write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "SAN.MC,Banco Santander,XMAD,EUR,Financials,Spain\n"
        "AAPL,Apple Inc.,XNAS,USD,Information Technology,United States\n",
        encoding="utf-8",
    )
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        f"data_dir: {tmp_path / 'data'}\n"
        f"universe_path: {tmp_path / 'universe.csv'}\n"
        f"log_dir: {tmp_path / 'logs'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))

    root = tmp_path / "data" / "bars" / "daily"
    san = bars("SAN.MC", [d for d in SESSIONS if d != D(2024, 3, 12)])  # one lost session
    apple_days = expected_sessions("XNAS", 2024, start=D(2024, 3, 1), end=D(2024, 3, 28))
    apple = bars("AAPL", apple_days)
    apple.loc[10:, ["open", "high", "low", "close"]] /= 4  # unadjusted 4:1 split
    write_bars(pd.concat([san, apple]), root=root)
    merge_actions(
        pd.DataFrame(
            {"ticker": ["AAPL"], "date": [apple_days[10]], "action": ["split"], "value": [4.0]}
        ),
        root=tmp_path / "data" / "actions",
    )
    return tmp_path


@pytest.mark.usefixtures("project")
def test_run_checks_covers_calendar_gaps_and_honours_splits() -> None:
    report = run_checks(as_of=D(2024, 3, 28))

    assert report.tickers == ("SAN.MC", "AAPL")
    assert checks(list(report.flags)) == [("missing_session", D(2024, 3, 12))]
    assert report.flags[0].ticker == "SAN.MC"
    assert not report.clean


def test_finmgr_check_fails_until_triaged_then_passes(project: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(app, ["check", "--as-of", "2024-03-28"], env={"COLUMNS": "200"})
    assert first.exit_code == 1, first.output
    assert "missing_session" in first.output and "1 untriaged" in first.output

    write_register(
        project / "quality_triage.csv",
        "missing_session,SAN.MC,2024-03-12,2024-03-12,source-gap,Yahoo has no bar",
    )
    second = runner.invoke(app, ["check", "--as-of", "2024-03-28"], env={"COLUMNS": "200"})
    assert second.exit_code == 0, second.output
    assert "1 triaged, 0 untriaged" in second.output
    assert "source-gap" in second.output
