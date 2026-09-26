"""Day 11: the ingestion loop, and what it does when the network goes away.

The acceptance criterion is
`test_pulling_the_cable_mid_run_still_reports_what_succeeded` and its CLI twin
below. Both simulate the cable coming out the only way a test can and still be
deterministic: a fetcher that serves real bars for the first few tickers and
then, for every call after that, forever, raises a real
`requests.exceptions.ConnectionError` carrying the message a severed link
produces — `[Errno 101] Network is unreachable`, through
`urllib3.NewConnectionError`.

Which exception class arrives is not the point, and the loop is written not to
care: yfinance's own stack is curl-based and reports the same outage as
`curl: (7) Failed to connect`. `tools/pull_the_cable.py` shows that, by taking
the network away from a running ingest against live Yahoo for real. This module
is the version that runs in `make test`, on any machine, offline.

Offline by construction, in fact: the autouse fixture replaces the one name in
:mod:`finmgr.data.ingest` that can reach Yahoo with a stub that fails the test,
and every other test injects its own fetcher. Sleeps are injected too — a test
that really waited two seconds per retry would take minutes, and the delays are
exactly what needs asserting on.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yahoo_fixtures
from requests.exceptions import ConnectionError as RequestsConnectionError
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV, Settings
from finmgr.data import ingest
from finmgr.data.fetch import to_bars
from finmgr.data.ingest import (
    DEFAULT_ATTEMPTS,
    IngestReport,
    Status,
    backoff_delays,
    fetch_with_retries,
    ingest_ticker,
    ingest_universe,
)
from finmgr.data.store import empty_bars, read_bars, stored_tickers
from finmgr.data.universe import Company

AAPL = "AAPL_2020-08-24_2020-09-04"

#: What every run below is told to ask for. The dates only matter in that the
#: fake fetchers ignore them and answer with the same saved bars.
WINDOW = (dt.date(2020, 8, 24), dt.date(2020, 9, 4))

#: The message a real severed link produces, near enough word for word.
UNREACHABLE = (
    "HTTPSConnectionPool(host='query2.finance.yahoo.com', port=443): "
    "Max retries exceeded with url: /v8/finance/chart/AAPL "
    "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object>: "
    "Failed to establish a new connection: [Errno 101] Network is unreachable'))"
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take Yahoo away from every test in this module."""

    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(ingest, "fetch_daily", refuse)


@pytest.fixture
def slept() -> list[float]:
    """Collects the waits a run asks for; pass `sleep=slept.append`."""
    return []


def unreachable() -> pd.DataFrame:
    """Raise what a call into a dead network raises."""
    raise RequestsConnectionError(UNREACHABLE)


def bars_for(ticker: str) -> pd.DataFrame:
    """Real Yahoo bars, relabelled as `ticker`.

    The prices come from day 10's saved AAPL response, so what the loop writes
    to the store is a response that genuinely came off the wire once.
    """
    return to_bars(ticker, yahoo_fixtures.load_raw(AAPL))


def working(calls: list[str] | None = None) -> Callable[..., pd.DataFrame]:
    """A fetcher that always answers."""

    def fetch(ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if calls is not None:
            calls.append(ticker)
        return bars_for(ticker)

    return fetch


def cable_pulled_after(alive: int, calls: list[str] | None = None) -> Callable[..., pd.DataFrame]:
    """A fetcher that answers for `alive` tickers, after which the network is gone.

    Not "then fails once": every later call raises, on every retry, for every
    remaining ticker — which is what an unplugged cable looks like from inside
    the process.
    """
    seen: set[str] = set()

    def fetch(ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if calls is not None:
            calls.append(ticker)
        seen.add(ticker)
        if len(seen) > alive:
            return unreachable()
        return bars_for(ticker)

    return fetch


def tickers(count: int) -> list[str]:
    return [f"T{index:02d}" for index in range(count)]


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def test_pulling_the_cable_mid_run_still_reports_what_succeeded(
    tmp_path: Path, slept: list[float]
) -> None:
    """Day 11: the network dies after three tickers; the run still ends cleanly.

    Cleanly means three things, all asserted here: nothing propagates out of
    `ingest_universe`, the three tickers that made it are on disk and named in
    the report, and every ticker that did not make it says why.
    """
    report = ingest_universe(
        tickers(20),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        fetch=cable_pulled_after(3),
        sleep=slept.append,
    )

    assert [result.ticker for result in report.succeeded] == ["T00", "T01", "T02"]
    assert all(result.status is Status.OK for result in report.succeeded)
    assert report.rows == 3 * len(bars_for("T00"))
    # What the report says succeeded is what is actually on disk.
    assert stored_tickers(root=tmp_path) == ["T00", "T01", "T02"]
    assert len(read_bars("T01", root=tmp_path)) == len(bars_for("T01"))

    # Everything else is accounted for: no ticker silently disappears.
    assert len(report.results) == 20
    assert report.failed and report.skipped
    assert len(report.failed) + len(report.skipped) == 17
    assert "Network is unreachable" in report.failed[0].detail
    assert report.failed[0].attempts == DEFAULT_ATTEMPTS
    assert all("consecutive failures" in result.detail for result in report.skipped)
    assert not report.complete
    assert "stopped early" in report.summary()
    assert slept, "the run paused between calls and between retries"


def test_the_cli_exits_cleanly_when_the_network_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same failure through `finmgr ingest`: exit code 1, a report, no traceback."""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\n", encoding="utf-8"
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))

    def fetch_daily(ticker: str, *_: object, **__: object) -> pd.DataFrame:
        if ticker in {"AAA", "BBB"}:
            return bars_for(ticker)
        return unreachable()

    monkeypatch.setattr(ingest, "fetch_daily", fetch_daily)

    # Zero pause and zero backoff so the test does not wait the real ones out;
    # the retries themselves still happen.
    argv = "ingest -t AAA -t BBB -t CCC -t DDD --start 2020-08-24 --end 2020-09-04"
    argv += " --pause 0 --backoff 0 --abort-after 0"

    result = CliRunner().invoke(app, argv.split())

    assert isinstance(result.exception, SystemExit) or result.exception is None
    assert result.exit_code == 1
    assert "Ingested" in result.output
    assert "AAA" in result.output and "BBB" in result.output
    assert "Traceback" not in result.output
    assert "2 of 4 tickers ingested" in result.output
    assert stored_tickers(root=tmp_path / "data" / "bars" / "daily") == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# Retries and backoff
# ---------------------------------------------------------------------------


def test_backoff_is_exponential_and_capped() -> None:
    assert backoff_delays(3, backoff=2.0, factor=2.0) == [2.0, 4.0]
    assert backoff_delays(1) == []
    assert backoff_delays(5, backoff=1.0, factor=10.0, maximum=50.0) == [1.0, 10.0, 50.0, 50.0]
    with pytest.raises(ValueError, match="at least 1"):
        backoff_delays(0)


def test_three_attempts_with_a_doubling_wait_between_them(slept: list[float]) -> None:
    """Two failures then an answer: three tries, two waits."""
    answers: list[Callable[[], pd.DataFrame]] = [
        unreachable,
        unreachable,
        lambda: bars_for("AAPL"),
    ]

    def flaky(_ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        return answers.pop(0)()

    bars, attempts = fetch_with_retries("AAPL", *WINDOW, fetch=flaky, sleep=slept.append)

    assert attempts == 3
    assert len(bars) == len(bars_for("AAPL"))
    assert slept == [2.0, 4.0]


def test_the_last_failure_is_raised_when_every_attempt_fails(slept: list[float]) -> None:
    """`fetch_with_retries` still raises; turning that into a row is the loop's job."""
    with pytest.raises(RequestsConnectionError):
        fetch_with_retries("AAPL", *WINDOW, fetch=lambda *_: unreachable(), sleep=slept.append)

    assert slept == [2.0, 4.0]


def test_a_ticker_that_never_recovers_becomes_a_status_row(
    tmp_path: Path, slept: list[float]
) -> None:
    result = ingest_ticker(
        "AAPL", *WINDOW, root=tmp_path, fetch=lambda *_: unreachable(), sleep=slept.append
    )

    assert result.status is Status.FAILED
    assert result.attempts == DEFAULT_ATTEMPTS
    assert "ConnectionError" in result.detail
    assert result.rows == 0
    assert not result.written
    assert slept == [2.0, 4.0]


def test_a_failing_write_is_the_tickers_failure_not_the_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk must not end the run either."""

    def refuse(*_: object, **__: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ingest, "write_bars", refuse)

    result = ingest_ticker("AAPL", *WINDOW, root=tmp_path, fetch=working())

    assert result.status is Status.FAILED
    assert "No space left on device" in result.detail
    # One request was made and it worked; only the write failed.
    assert result.attempts == 1


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_one_bad_symbol_does_not_kill_the_run(tmp_path: Path, slept: list[float]) -> None:
    """The point of the per-ticker try/except, in one test."""

    def fetch(ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if ticker == "T02":
            raise ValueError("T02: possibly delisted; no price data found")
        return bars_for(ticker)

    report = ingest_universe(
        tickers(5),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        fetch=fetch,
        sleep=slept.append,
    )

    assert [result.ticker for result in report.failed] == ["T02"]
    assert stored_tickers(root=tmp_path) == ["T00", "T01", "T03", "T04"]
    assert report.counts == {"ok": 4, "failed": 1}
    assert not report.skipped


def test_a_delay_separates_one_ticker_from_the_next(tmp_path: Path, slept: list[float]) -> None:
    """A pause between calls, and none before the first."""
    report = ingest_universe(
        tickers(4),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        pause=0.5,
        fetch=working(),
        sleep=slept.append,
    )

    assert len(report.succeeded) == 4
    assert slept == [0.5, 0.5, 0.5]


def test_an_empty_window_is_not_a_failure(tmp_path: Path, slept: list[float]) -> None:
    """A holiday week, or day 12's weekend re-run: zero rows, status `empty`."""
    report = ingest_universe(
        ["AAPL"],
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        fetch=lambda *_: empty_bars(),
        sleep=slept.append,
    )

    (result,) = report.results
    assert result.status is Status.EMPTY
    assert result.succeeded and report.complete
    assert result.attempts == 1
    assert stored_tickers(root=tmp_path) == []


def test_a_run_that_fails_everywhere_gives_up_and_skips_the_rest(
    tmp_path: Path, slept: list[float]
) -> None:
    """Nothing is gained by 100 doomed requests; the skipped rows say so."""
    calls: list[str] = []

    report = ingest_universe(
        tickers(30),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        abort_after=5,
        fetch=cable_pulled_after(0, calls=calls),
        sleep=slept.append,
    )

    assert len(report.failed) == 5
    assert len(report.skipped) == 25
    assert len(set(calls)) == 5, "no request was made for a skipped ticker"
    assert report.stopped_early == "5 consecutive failures"


def test_giving_up_early_can_be_switched_off(tmp_path: Path, slept: list[float]) -> None:
    report = ingest_universe(
        tickers(12),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        abort_after=0,
        fetch=cable_pulled_after(0),
        sleep=slept.append,
    )

    assert len(report.failed) == 12
    assert not report.skipped


def test_an_occasional_failure_does_not_trip_the_abort(tmp_path: Path, slept: list[float]) -> None:
    """The counter is consecutive failures, not total ones."""

    def fetch(ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if ticker in {"T01", "T03", "T05"}:
            return unreachable()
        return bars_for(ticker)

    report = ingest_universe(
        tickers(8),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        abort_after=2,
        fetch=fetch,
        sleep=slept.append,
    )

    assert len(report.failed) == 3
    assert not report.skipped


def test_ctrl_c_stops_the_run_and_keeps_what_arrived(tmp_path: Path, slept: list[float]) -> None:
    """The other way a run ends early. Same outcome: a report, not a traceback."""

    def fetch(ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if ticker == "T02":
            raise KeyboardInterrupt
        return bars_for(ticker)

    report = ingest_universe(
        tickers(6),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        fetch=fetch,
        sleep=slept.append,
    )

    assert report.interrupted
    assert [result.ticker for result in report.succeeded] == ["T00", "T01"]
    assert len(report.skipped) == 4
    assert stored_tickers(root=tmp_path) == ["T00", "T01"]
    assert "interrupted" in report.summary()


def test_companies_are_accepted_as_well_as_symbols(tmp_path: Path, slept: list[float]) -> None:
    """The loop takes what `load_universe()` returns, which is its default."""
    companies = [
        Company("SAN.MC", "Banco Santander", "XMAD", "EUR", "Financials", "ES"),
        Company("SAP.DE", "SAP SE", "XETR", "EUR", "Information Technology", "DE"),
    ]
    calls: list[str] = []

    ingest_universe(
        companies,
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        fetch=working(calls),
        sleep=slept.append,
    )

    assert calls == ["SAN.MC", "SAP.DE"]


def test_a_dry_run_fetches_but_writes_nothing(tmp_path: Path, slept: list[float]) -> None:
    report = ingest_universe(
        tickers(3),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        write=False,
        fetch=working(),
        sleep=slept.append,
    )

    assert report.rows > 0
    assert all(not result.written for result in report.succeeded)
    assert stored_tickers(root=tmp_path) == []


def test_every_ticker_is_asked_for_the_same_window(tmp_path: Path, slept: list[float]) -> None:
    """Resolved once, up front — so the report can state it, skipped rows included."""
    windows: list[tuple[dt.date, dt.date]] = []

    def fetch(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        windows.append((start, end))
        return bars_for(ticker)

    settings = Settings(data_dir=tmp_path, history_start=dt.date(2010, 1, 4))
    report = ingest_universe(
        tickers(3), settings=settings, root=tmp_path, fetch=fetch, sleep=slept.append
    )

    assert {window[0] for window in windows} == {dt.date(2010, 1, 4)}
    assert len({window[1] for window in windows}) == 1
    assert report.start == dt.date(2010, 1, 4)
    assert report.end == windows[0][1]


def test_a_reversed_window_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="after end"):
        ingest_universe(
            ["AAPL"], start="2020-09-04", end="2020-08-24", root=tmp_path, fetch=working()
        )


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


def test_the_report_summarises_and_serialises(tmp_path: Path, slept: list[float]) -> None:
    """Day 14 writes these rows to a manifest; they have to say everything."""
    report = ingest_universe(
        tickers(6),
        start=WINDOW[0],
        end=WINDOW[1],
        root=tmp_path,
        fetch=cable_pulled_after(2),
        sleep=slept.append,
    )

    as_dict = report.as_dict()
    assert as_dict["counts"] == {"ok": 2, "failed": 4}
    assert as_dict["start"] == "2020-08-24"
    assert [row["ticker"] for row in as_dict["results"]] == tickers(6)
    first = as_dict["results"][0]
    assert first["status"] == "ok"
    assert first["rows"] == len(bars_for("T00"))
    assert first["first_date"] == "2020-08-24" and first["last_date"] == "2020-09-04"
    assert first["written"] is True
    assert "2 of 6 tickers ingested" in report.summary()


def test_an_empty_report_says_so() -> None:
    report = IngestReport()

    assert report.complete
    assert report.rows == 0
    assert "0 of 0 tickers ingested" in report.summary()


def test_the_network_guard_is_really_armed(tmp_path: Path, slept: list[float]) -> None:
    """Proof that `no_network` would catch a live call, not just claim to."""
    report = ingest_universe(
        ["AAPL"], start=WINDOW[0], end=WINDOW[1], root=tmp_path, sleep=slept.append
    )

    assert report.failed[0].status is Status.FAILED
    assert "tried to reach Yahoo" in report.failed[0].detail
