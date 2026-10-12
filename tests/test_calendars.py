"""Day 17: holding the store against each venue's exchange calendar.

The acceptance criterion is a list of genuinely missing sessions per ticker
for 2024, holidays excluded. So the cases that matter are a Madrid holiday the
store rightly has no bar for (not missing), a Madrid weekday it has lost
(missing), and the edges of the window — a late listing, a year still in
progress, an unscheduled closure. `exchange_calendars` is offline, so none of
this touches the network; the bars live in `tmp_path`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data import calendars
from finmgr.data.calendars import expected_sessions, session_gaps, ticker_venues
from finmgr.data.store import write_bars
from finmgr.data.universe import load_universe

D = dt.date
GOOD_FRIDAY = D(2024, 3, 29)
EASTER_MONDAY = D(2024, 4, 1)
LOST = D(2024, 3, 27)  # an ordinary Wednesday in Madrid
END_2024 = D(2024, 12, 31)


def bars(ticker: str, days: Iterable[dt.date]) -> pd.DataFrame:
    days = sorted(days)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": days,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1_000.0,
        }
    )


@pytest.fixture
def universe(tmp_path: Path) -> Path:
    path = tmp_path / "universe.csv"
    path.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "SAN.MC,Banco Santander,XMAD,EUR,Financials,Spain\n"
        "AAPL,Apple Inc.,XNAS,USD,Information Technology,United States\n"
        "2330.TW,TSMC,XTAI,TWD,Information Technology,Taiwan\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "data" / "bars" / "daily"


def test_madrid_2024_sessions_exclude_its_public_holidays() -> None:
    sessions = expected_sessions("XMAD", 2024)

    assert len(sessions) == 256
    assert sessions[0] == D(2024, 1, 2) and sessions[-1] == END_2024
    for holiday in (D(2024, 1, 1), GOOD_FRIDAY, EASTER_MONDAY, D(2024, 5, 1), D(2024, 12, 25)):
        assert holiday not in sessions
    assert LOST in sessions
    assert all(day.weekday() < 5 for day in sessions)


def test_venues_differ_in_the_same_year() -> None:
    # Good Friday shuts both; Madrid trades on US Thanksgiving, New York on
    # Easter Monday.
    madrid, new_york = set(expected_sessions("XMAD", 2024)), set(expected_sessions("XNYS", 2024))
    assert GOOD_FRIDAY not in madrid | new_york
    assert D(2024, 11, 28) in madrid - new_york
    assert EASTER_MONDAY in new_york - madrid


def test_every_universe_venue_has_a_calendar() -> None:
    venues = {company.exchange for company in load_universe(Path("config/universe.csv"))}
    for venue in sorted(venues):
        assert expected_sessions(venue, 2024), venue
    assert {"XNYS", "XMAD", "XPAR", "XETR", "XAMS"} <= venues


def test_an_unknown_venue_is_an_error() -> None:
    with pytest.raises(ValueError, match="no exchange calendar"):
        expected_sessions("XNOPE", 2024)


def test_venues_come_from_the_universe(universe: Path) -> None:
    assert ticker_venues(path=universe) == {"SAN.MC": "XMAD", "AAPL": "XNAS", "2330.TW": "XTAI"}
    with pytest.raises(ValueError, match=r"DEAD\.XX"):
        ticker_venues(["SAN.MC", "DEAD.XX"], path=universe)


def test_a_lost_weekday_is_missing_and_a_holiday_is_not(universe: Path, root: Path) -> None:
    held = [day for day in expected_sessions("XMAD", 2024) if day != LOST]
    write_bars(bars("SAN.MC", held), root=root)

    (gap,) = session_gaps(["SAN.MC"], year=2024, as_of=D(2026, 1, 1), universe=universe, root=root)

    assert gap.venue == "XMAD"
    assert gap.missing == (LOST,)
    assert (gap.expected, gap.stored) == (256, 255)
    assert gap.off_calendar == () and not gap.clean


def test_a_bar_on_a_holiday_is_off_calendar(universe: Path, root: Path) -> None:
    write_bars(bars("SAN.MC", [*expected_sessions("XMAD", 2024), GOOD_FRIDAY]), root=root)

    (gap,) = session_gaps(["SAN.MC"], year=2024, as_of=D(2026, 1, 1), universe=universe, root=root)

    assert gap.missing == ()
    assert gap.off_calendar == (GOOD_FRIDAY,)


def test_a_late_listing_is_only_checked_from_its_first_bar(universe: Path, root: Path) -> None:
    listed = D(2024, 6, 3)
    write_bars(bars("AAPL", expected_sessions("XNAS", 2024, start=listed)), root=root)

    (gap,) = session_gaps(["AAPL"], year=2024, as_of=D(2026, 1, 1), universe=universe, root=root)

    assert gap.clean
    assert gap.window == (listed, END_2024)
    assert gap.expected == len(expected_sessions("XNAS", 2024, start=listed))


def test_the_year_in_progress_stops_at_as_of(universe: Path, root: Path) -> None:
    write_bars(bars("AAPL", expected_sessions("XNAS", 2024, end=D(2024, 6, 28))), root=root)

    (gap,) = session_gaps(["AAPL"], year=2024, as_of=D(2024, 6, 30), universe=universe, root=root)

    assert gap.clean and gap.window == (D(2024, 1, 2), D(2024, 6, 30))


def test_a_ticker_with_no_bars_misses_every_session(universe: Path, root: Path) -> None:
    (gap,) = session_gaps(["AAPL"], year=2024, as_of=D(2026, 1, 1), universe=universe, root=root)

    assert gap.stored == 0
    assert gap.missing == tuple(expected_sessions("XNAS", 2024))


def test_an_unscheduled_closure_is_excused_and_reported(universe: Path, root: Path) -> None:
    typhoon = D(2024, 10, 31)
    assert typhoon in calendars.UNSCHEDULED_CLOSURES["XTAI"]
    held = [day for day in expected_sessions("XTAI", 2024) if day != typhoon]
    write_bars(bars("2330.TW", held), root=root)

    (gap,) = session_gaps(["2330.TW"], year=2024, as_of=D(2026, 1, 1), universe=universe, root=root)

    assert gap.clean
    assert [day for day, _ in gap.excused] == [typhoon]
    assert gap.expected == gap.stored == len(held)


def test_a_bar_on_an_excused_closure_is_flagged(universe: Path, root: Path) -> None:
    # If the venue did trade, the closure entry is wrong — say so.
    write_bars(bars("2330.TW", expected_sessions("XTAI", 2024)), root=root)

    (gap,) = session_gaps(["2330.TW"], year=2024, as_of=D(2026, 1, 1), universe=universe, root=root)

    assert gap.off_calendar == (D(2024, 10, 31),)


def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe: Path) -> CliRunner:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\nuniverse_path: {universe}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    return CliRunner()


def test_the_cli_lists_missing_sessions_and_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe: Path, root: Path
) -> None:
    write_bars(bars("SAN.MC", [d for d in expected_sessions("XMAD", 2024) if d != LOST]), root=root)
    write_bars(bars("AAPL", expected_sessions("XNAS", 2024)), root=root)
    write_bars(
        bars("2330.TW", [d for d in expected_sessions("XTAI", 2024) if d != D(2024, 10, 31)]),
        root=root,
    )

    result = cli(tmp_path, monkeypatch, universe).invoke(
        app, ["sessions", "--year", "2024"], terminal_width=200
    )

    assert result.exit_code == 1, result.output
    assert "SAN.MC" in result.output and "2024-03-27" in result.output
    assert str(GOOD_FRIDAY) not in result.output
    assert "typhoon" in result.output
    assert "2 of 3 tickers complete for 2024; 1 missing sessions" in result.output


def test_the_cli_passes_on_a_complete_year(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe: Path, root: Path
) -> None:
    write_bars(bars("SAN.MC", expected_sessions("XMAD", 2024)), root=root)

    result = cli(tmp_path, monkeypatch, universe).invoke(
        app, ["sessions", "-y", "2024", "-t", "SAN.MC", "--all"], terminal_width=200
    )

    assert result.exit_code == 0, result.output
    assert "256" in result.output
    assert "1 of 1 tickers complete for 2024; 0 missing sessions" in result.output


def test_the_cli_rejects_a_ticker_outside_the_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, universe: Path
) -> None:
    result = cli(tmp_path, monkeypatch, universe).invoke(
        app, ["sessions", "-y", "2024", "-t", "DEAD.XX"], terminal_width=200
    )

    assert result.exit_code == 2
