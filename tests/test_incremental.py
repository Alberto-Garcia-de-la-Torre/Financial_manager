"""Day 12: incremental windows, and a re-run that writes nothing.

The acceptance criterion is
`test_running_ingest_twice_writes_zero_new_rows_the_second_time` and its CLI
twin below. Both run a whole ingest, run it again against the same store, and
assert the second one added no rows — checked three ways, because "zero new
rows" is easy to claim and easy to get wrong in different places: the report
says zero, the bars on disk are byte-identical, and the Parquet files were not
even reopened for writing.

Two mechanisms produce that, and they are tested apart as well as together.
The *window* narrows: each ticker is asked from its own newest stored bar
(less a few days of deliberate overlap) rather than from `history_start`. And
the *write* merges on `(ticker, date)`, so a session that comes back twice is
overwritten rather than appended — which is what makes even a forced `--full`
re-request idempotent.

Offline, like every other test here: a fake Yahoo holds a fixed history per
ticker and serves the slice a window asks for, recording every window it was
given.
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
from finmgr.data import ingest
from finmgr.data.ingest import (
    DEFAULT_REFETCH_DAYS,
    Status,
    incremental_start,
    ingest_universe,
)
from finmgr.data.store import (
    conform_bars,
    max_stored_date,
    max_stored_dates,
    merge_bars,
    partition_path,
    read_bars,
    write_bars,
)

#: The window every run below is given. Its end runs past the fake history on
#: purpose — that is what "up to today" looks like on a Sunday, and it leaves
#: room for a few more sessions to happen between two runs.
WINDOW = (dt.date(2026, 9, 1), dt.date(2026, 10, 9))

TICKERS = ("AAPL", "SAN.MC")


def history(ticker: str, sessions: int = 20, start: str = "2026-09-01") -> pd.DataFrame:
    """A deterministic run of daily bars in the canonical schema."""
    rng = np.random.default_rng(abs(hash(ticker)) % 2**32)
    dates = pd.bdate_range(start, periods=sessions).date
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, sessions))
    return conform_bars(
        pd.DataFrame(
            {
                "ticker": ticker,
                "date": dates,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 1e6 + np.arange(sessions, dtype=float),
            }
        )
    )


class Yahoo:
    """A fake Yahoo: fixed histories, sliced to whatever window is asked for.

    It records every `(ticker, start, end)` it was given, which is how the
    tests assert that the second run asked for less than the first rather than
    merely storing less.
    """

    def __init__(self, tickers: tuple[str, ...] = TICKERS, sessions: int = 20) -> None:
        self.bars = {ticker: history(ticker, sessions) for ticker in tickers}
        self.windows: list[tuple[str, dt.date, dt.date]] = []

    def __call__(self, ticker: str, start: dt.date, end: dt.date, **_: object) -> pd.DataFrame:
        self.windows.append((ticker, start, end))
        frame = self.bars[ticker]
        return frame.loc[frame["date"].between(start, end)].reset_index(drop=True)

    def starts(self, ticker: str) -> list[dt.date]:
        return [window[1] for window in self.windows if window[0] == ticker]

    def extend(self, sessions: int) -> None:
        """Let a few more sessions happen, as they do between two evenings."""
        for ticker, frame in self.bars.items():
            last = frame["date"].iloc[-1]
            more = history(ticker, len(frame) + sessions, start=last.isoformat())
            self.bars[ticker] = conform_bars(
                pd.concat([frame, more.iloc[1 : sessions + 1]], ignore_index=True)
            )

    def revise(self, ticker: str, close: float = 123.45) -> dt.date:
        """Restate the most recent session, the way a late volume fix does."""
        frame = self.bars[ticker].copy()
        frame.loc[frame.index[-1], "close"] = close
        self.bars[ticker] = frame
        return frame["date"].iloc[-1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take Yahoo away from every test in this module."""

    def refuse(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(ingest, "fetch_daily", refuse)


@pytest.fixture
def slept() -> list[float]:
    return []


def mtimes(root: Path) -> dict[str, int]:
    """When each partition was last written, to the nanosecond."""
    return {path.parent.name: path.stat().st_mtime_ns for path in root.glob("ticker=*/*.parquet")}


def run(root: Path, yahoo: Yahoo, slept: list[float], **kwargs: object) -> ingest.IngestReport:
    return ingest_universe(
        TICKERS,
        start=WINDOW[0],
        end=WINDOW[1],
        root=root,
        fetch=yahoo,
        sleep=slept.append,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def test_running_ingest_twice_writes_zero_new_rows_the_second_time(
    tmp_path: Path, slept: list[float]
) -> None:
    """Day 12: the second run adds nothing, changes nothing, rewrites nothing."""
    yahoo = Yahoo()

    first = run(tmp_path, yahoo, slept)
    stored = read_bars(root=tmp_path)
    written_at = mtimes(tmp_path)

    assert first.added == len(stored) == 40
    assert first.updated == 0

    second = run(tmp_path, yahoo, slept)

    # The report says so...
    assert second.added == 0
    assert second.updated == 0
    assert all(result.status is Status.OK for result in second.results)
    assert "0 new" in second.summary()
    # ... the store agrees ...
    pd.testing.assert_frame_equal(read_bars(root=tmp_path), stored)
    # ... and no partition was reopened for writing.
    assert mtimes(tmp_path) == written_at
    assert not any(result.written for result in second.results)

    # It did ask Yahoo, and asked for less than the first run: the window was
    # narrowed to the overlap, not skipped outright.
    assert 0 < second.rows < first.rows
    assert yahoo.starts("AAPL") == [
        WINDOW[0],
        dt.date(2026, 9, 28) - dt.timedelta(days=DEFAULT_REFETCH_DAYS),
    ]


def test_the_cli_run_twice_writes_zero_new_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same criterion through `finmgr ingest`, twice, exit code 0 both times."""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\n", encoding="utf-8"
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    monkeypatch.setattr(ingest, "fetch_daily", Yahoo())

    bars = tmp_path / "data" / "bars" / "daily"
    argv = "ingest -t AAPL -t SAN.MC --start 2026-09-01 --end 2026-09-30 --pause 0"

    first = CliRunner().invoke(app, argv.split())
    stored = read_bars(root=bars)
    written_at = mtimes(bars)

    assert first.exit_code == 0, first.output
    assert len(stored) == 40

    second = CliRunner().invoke(app, argv.split())

    assert second.exit_code == 0, second.output
    assert "0 new" in second.output
    assert mtimes(bars) == written_at
    pd.testing.assert_frame_equal(read_bars(root=bars), stored)


# ---------------------------------------------------------------------------
# Where the next request starts
# ---------------------------------------------------------------------------


def test_the_max_stored_date_is_read_per_ticker(tmp_path: Path) -> None:
    write_bars(history("AAPL", sessions=5), root=tmp_path)
    write_bars(history("SAN.MC", sessions=12), root=tmp_path)

    assert max_stored_dates(root=tmp_path) == {
        "AAPL": dt.date(2026, 9, 7),
        "SAN.MC": dt.date(2026, 9, 16),
    }
    assert max_stored_dates(["SAN.MC"], root=tmp_path) == {"SAN.MC": dt.date(2026, 9, 16)}
    assert max_stored_date("AAPL", root=tmp_path) == dt.date(2026, 9, 7)
    # A ticker nobody has stored yet is absent rather than dated in 1970.
    assert max_stored_date("MSFT", root=tmp_path) is None
    assert max_stored_dates([], root=tmp_path) == {}


def test_the_window_starts_where_the_store_ends() -> None:
    requested = dt.date(2010, 1, 4)

    # Nothing stored: the full backfill stands.
    assert incremental_start(requested, None) == requested
    # Stored: forward to a few days before the newest bar.
    assert incremental_start(requested, dt.date(2026, 9, 25), refetch_days=5) == dt.date(
        2026, 9, 20
    )
    # No overlap asked for: start on the last stored session itself.
    assert incremental_start(requested, dt.date(2026, 9, 25), refetch_days=0) == dt.date(
        2026, 9, 25
    )
    # The requested start is a floor: a store that only reaches 2010-01-06 does
    # not drag the window back before what was asked for.
    assert incremental_start(requested, dt.date(2010, 1, 6), refetch_days=30) == requested
    with pytest.raises(ValueError, match="must not be negative"):
        incremental_start(requested, None, refetch_days=-1)


def test_each_ticker_gets_its_own_window(tmp_path: Path, slept: list[float]) -> None:
    """A stale ticker is backfilled while a current one is barely touched."""
    yahoo = Yahoo()
    write_bars(history("AAPL", sessions=18), root=tmp_path)  # stored through 2026-09-24

    run(tmp_path, yahoo, slept)

    assert yahoo.starts("AAPL") == [dt.date(2026, 9, 24) - dt.timedelta(days=DEFAULT_REFETCH_DAYS)]
    assert yahoo.starts("SAN.MC") == [WINDOW[0]], "a ticker with no history gets the full window"


def test_a_store_already_past_the_window_is_not_asked_at_all(
    tmp_path: Path, slept: list[float]
) -> None:
    """`--end` in the past against a current store: status `current`, no request."""
    yahoo = Yahoo()
    write_bars(history("AAPL", sessions=20), root=tmp_path)

    report = ingest_universe(
        ["AAPL"],
        start=WINDOW[0],
        end=dt.date(2026, 9, 10),
        root=tmp_path,
        fetch=yahoo,
        sleep=slept.append,
    )

    (result,) = report.results
    assert result.status is Status.CURRENT
    assert result.succeeded and report.complete
    assert result.stored_max == dt.date(2026, 9, 28)
    assert "past the requested 2026-09-10" in result.detail
    assert yahoo.windows == [], "nothing was asked for"


def test_full_re_requests_the_whole_window_and_still_adds_nothing(
    tmp_path: Path, slept: list[float]
) -> None:
    """Idempotence comes from the merge, not only from the narrower window."""
    yahoo = Yahoo()

    first = run(tmp_path, yahoo, slept, incremental=False)
    written_at = mtimes(tmp_path)
    second = run(tmp_path, yahoo, slept, incremental=False)

    assert yahoo.starts("AAPL") == [WINDOW[0], WINDOW[0]]
    assert second.rows == first.rows, "the whole window was fetched again"
    assert second.added == 0 and second.updated == 0
    assert mtimes(tmp_path) == written_at
    assert not second.incremental, "the report records how it was run"


# ---------------------------------------------------------------------------
# What the merge does
# ---------------------------------------------------------------------------


def test_new_sessions_are_added_to_the_stored_history(tmp_path: Path, slept: list[float]) -> None:
    """Tomorrow's run: three more sessions, three more rows, nothing else."""
    yahoo = Yahoo()
    run(tmp_path, yahoo, slept)

    yahoo.extend(3)
    second = run(tmp_path, yahoo, slept)

    assert second.added == 6, "three new sessions for each of two tickers"
    assert second.updated == 0
    stored = read_bars("AAPL", root=tmp_path)
    assert len(stored) == 23
    assert stored["date"].is_monotonic_increasing
    assert not stored.duplicated(subset=["ticker", "date"]).any()


def test_a_revised_bar_overwrites_rather_than_appends(tmp_path: Path, slept: list[float]) -> None:
    """Yahoo restating a close is an update in place, not a second row."""
    yahoo = Yahoo()
    run(tmp_path, yahoo, slept)
    revised = yahoo.revise("AAPL", close=123.45)

    report = run(tmp_path, yahoo, slept)

    assert report.added == 0
    assert report.updated == 1
    stored = read_bars("AAPL", root=tmp_path)
    assert len(stored) == 20
    assert not stored.duplicated(subset=["ticker", "date"]).any()
    assert stored.loc[stored["date"] == revised, "close"].tolist() == [123.45]
    # Only the ticker that changed was rewritten.
    (aapl,) = [result for result in report.results if result.ticker == "AAPL"]
    (santander,) = [result for result in report.results if result.ticker == "SAN.MC"]
    assert aapl.written and not santander.written


def test_merge_counts_what_it_did_per_ticker(tmp_path: Path) -> None:
    stored = history("AAPL", sessions=10)
    merge_bars(stored, root=tmp_path)

    incoming = history("AAPL", sessions=14).iloc[8:].reset_index(drop=True)
    incoming.loc[0, "close"] = 42.0  # one restated session, one identical, four new

    (result,) = merge_bars(incoming, root=tmp_path)

    assert (result.added, result.updated, result.unchanged) == (4, 1, 1)
    assert result.stored == 14
    assert result.changed and result.path == partition_path("AAPL", root=tmp_path)
    assert len(read_bars("AAPL", root=tmp_path)) == 14


def test_an_unchanged_merge_leaves_the_file_alone(tmp_path: Path) -> None:
    bars = history("AAPL", sessions=10)
    merge_bars(bars, root=tmp_path)
    written_at = mtimes(tmp_path)

    (result,) = merge_bars(bars, root=tmp_path)

    assert (result.added, result.updated, result.unchanged) == (0, 0, 10)
    assert not result.changed and result.path is None
    assert mtimes(tmp_path) == written_at


def test_merging_nothing_is_not_an_error(tmp_path: Path) -> None:
    assert merge_bars(history("AAPL").iloc[:0], root=tmp_path) == []
    assert read_bars(root=tmp_path).empty


def test_a_nan_volume_is_not_mistaken_for_a_revision(tmp_path: Path) -> None:
    """NaN != NaN would rewrite every partition holding a volume-less session."""
    bars = history("AAPL", sessions=6)
    bars.loc[bars.index[-1], "volume"] = float("nan")
    merge_bars(bars, root=tmp_path)

    (result,) = merge_bars(bars, root=tmp_path)

    assert (result.added, result.updated, result.unchanged) == (0, 0, 6)


def test_merging_several_tickers_at_once(tmp_path: Path) -> None:
    both = pd.concat([history("AAPL", 5), history("SAN.MC", 5)], ignore_index=True)

    results = merge_bars(both, root=tmp_path)

    assert [result.ticker for result in results] == ["AAPL", "SAN.MC"]
    assert all(result.added == 5 for result in results)
    assert len(read_bars(root=tmp_path)) == 10


def test_the_status_row_records_the_window_it_actually_asked_for(
    tmp_path: Path, slept: list[float]
) -> None:
    """Day 14 writes these rows to a manifest; the narrowed start belongs in it."""
    yahoo = Yahoo()
    run(tmp_path, yahoo, slept)
    report = run(tmp_path, yahoo, slept)

    row = report.results[0].as_dict()
    assert row["asked_from"] == "2026-09-23"
    assert row["stored_max"] == "2026-09-28"
    assert row["added"] == 0 and row["unchanged"] == row["rows"]
    assert report.as_dict()["incremental"] is True
    assert report.as_dict()["added"] == 0
