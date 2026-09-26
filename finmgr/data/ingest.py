"""Batch ingestion: the whole universe, assuming it will partly fail.

    from finmgr.data.ingest import ingest_universe

    report = ingest_universe()          # every ticker in config/universe.csv
    report.rows                         # bars written
    report.failed                       # the ones that did not make it

Day 10 fetches one ticker and raises when Yahoo says no. This module is the
loop around it, and its entire job is to be *boring when things go wrong*:

**A small delay between calls.** yfinance is an unofficial scraper of a
rate-limited endpoint. A hundred requests as fast as the socket allows is the
quickest way to be throttled, so :data:`DEFAULT_PAUSE` seconds separate one
ticker from the next.

**Three attempts with exponential backoff.** A timeout, a 502 or a rate-limit
rejection is a statement about this second, not about the symbol. So each
ticker gets :data:`DEFAULT_ATTEMPTS` tries with :func:`backoff_delays` — 2s,
then 4s — between them. Every exception is retried rather than only a chosen
few: the failure modes of a scraper are not enumerable, and treating a
retryable error as fatal loses a company's history for the day, while treating
a fatal error as retryable costs four seconds.

**A try/except per ticker.** One bad symbol cannot end the run. Whatever goes
wrong — the request, the conversion, the write — is caught, written down as
that ticker's status row, and the loop moves on to the next one.

So nothing here raises for a ticker-level problem. The result is an
:class:`IngestReport`: one :class:`TickerIngest` per symbol, saying `ok`,
`empty`, `failed` or `skipped`, with the rows, the window, the attempts spent
and the error message. Day 14 writes those rows to a manifest under
`data/meta/`; today they are printed and returned.

Two behaviours exist specifically for the day the network dies mid-run:

*Each ticker is written as soon as it arrives* — :func:`ingest_ticker` fetches
and calls `write_bars` before the loop moves on — so an interrupted run keeps
everything it had already downloaded instead of losing the lot.

*A run that is failing everywhere stops early.* After
:data:`DEFAULT_ABORT_AFTER` consecutive failures the loop gives up and marks
the remaining tickers `skipped`, because 100 symbols at 3 attempts and 6
seconds of backoff each is half an hour of pointless waiting when the cause is
that the cable is out. `abort_after=0` disables it. A KeyboardInterrupt is handled the
same way: stop, mark the rest skipped, return the report.

The window is still the whole `history_start`..today by default; asking Yahoo
only for what is missing is day 12, and the full fifteen-year backfill is day
13.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from rich.console import Console
from rich.table import Table

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data.fetch import fetch_daily
from finmgr.data.store import as_date, write_bars
from finmgr.data.universe import Company, load_universe

#: Tries per ticker, including the first. Three is the roadmap's number and a
#: sensible one: it covers a blip and a retry landing in the same blip, without
#: turning a genuinely dead symbol into a minute of waiting.
DEFAULT_ATTEMPTS = 3

#: Backoff between attempts: `base * factor ** (attempt - 1)`, capped. With
#: these values a ticker waits 2s then 4s before its last try.
DEFAULT_BACKOFF = 2.0
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_BACKOFF = 60.0

#: Seconds between one ticker and the next. Politeness, and cheap insurance
#: against the rate limiter: 100 tickers cost under a minute of pausing.
DEFAULT_PAUSE = 0.5

#: Consecutive failures after which the run gives up and skips what is left.
#: Ten in a row is not ten bad symbols, it is the network. 0 disables it.
DEFAULT_ABORT_AFTER = 10

log = runlog.get_logger(__name__)


class Status(StrEnum):
    """What happened to one ticker in one run."""

    #: Bars came back for the requested window.
    OK = "ok"
    #: The request succeeded and the window holds no sessions. Not a failure:
    #: a re-run on a Sunday asks for a window with nothing in it.
    EMPTY = "empty"
    #: Every attempt raised. The reason is in `detail`.
    FAILED = "failed"
    #: Never attempted, because the run stopped before reaching it.
    SKIPPED = "skipped"


#: A downloader with :func:`~finmgr.data.fetch.fetch_daily`'s signature and its
#: inclusive bounds. Taken as an argument so the tests can run the whole loop —
#: retries, backoff, per-ticker recovery — without a network.
Fetcher = Callable[[str, date, date], pd.DataFrame]
Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class TickerIngest:
    """One status row: what this run did about one ticker.

    Returned rather than raised, for every outcome. Day 14 turns these into
    `data/meta/ingest_runs.parquet`, which is how "why is Iberdrola stale?"
    gets answered in eight weeks without guessing.
    """

    ticker: str
    status: Status
    rows: int = 0
    first_date: date | None = None
    last_date: date | None = None
    attempts: int = 0
    seconds: float = 0.0
    written: bool = False
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        """True when Yahoo answered — with bars or with an empty window."""
        return self.status in (Status.OK, Status.EMPTY)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "status": str(self.status),
            "rows": self.rows,
            "first_date": self.first_date.isoformat() if self.first_date else None,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "attempts": self.attempts,
            "seconds": round(self.seconds, 3),
            "written": self.written,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Every status row from one run, plus what the run was asked to do."""

    results: list[TickerIngest] = field(default_factory=list)
    start: date | None = None
    end: date | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    interrupted: bool = False
    #: Why the loop stopped before the end, empty when it did not.
    stopped_early: str = ""
    run_id: str | None = None

    @property
    def succeeded(self) -> list[TickerIngest]:
        """What worked — the list the acceptance criterion cares about."""
        return [result for result in self.results if result.succeeded]

    @property
    def failed(self) -> list[TickerIngest]:
        return [result for result in self.results if result.status is Status.FAILED]

    @property
    def skipped(self) -> list[TickerIngest]:
        return [result for result in self.results if result.status is Status.SKIPPED]

    @property
    def rows(self) -> int:
        """Bars ingested across every ticker."""
        return sum(result.rows for result in self.results)

    @property
    def counts(self) -> Counter[str]:
        return Counter(str(result.status) for result in self.results)

    @property
    def complete(self) -> bool:
        """True when every ticker was reached and answered for."""
        return not self.failed and not self.skipped and not self.interrupted

    @property
    def seconds(self) -> float:
        if self.started_at is None or self.finished_at is None:
            return sum(result.seconds for result in self.results)
        return (self.finished_at - self.started_at).total_seconds()

    def summary(self) -> str:
        """One line: what succeeded, out of how many, in how long."""
        counts = self.counts
        detail = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        ok = counts[str(Status.OK)] + counts[str(Status.EMPTY)]
        line = (
            f"{ok} of {len(self.results)} tickers ingested "
            f"({detail}); {self.rows} bars in {self.seconds:.1f}s"
        )
        if self.interrupted:
            line += " — interrupted"
        elif self.stopped_early:
            line += f" — stopped early: {self.stopped_early}"
        return line

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "interrupted": self.interrupted,
            "stopped_early": self.stopped_early,
            "rows": self.rows,
            "counts": dict(self.counts),
            "results": [result.as_dict() for result in self.results],
        }


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


def backoff_delays(
    attempts: int = DEFAULT_ATTEMPTS,
    *,
    backoff: float = DEFAULT_BACKOFF,
    factor: float = DEFAULT_BACKOFF_FACTOR,
    maximum: float = DEFAULT_MAX_BACKOFF,
) -> list[float]:
    """Seconds to wait after each failed attempt. Pure, so it can be tested.

    `attempts` tries need `attempts - 1` waits — there is nothing to wait for
    after the last one. Exponential rather than constant because the thing being
    backed off is a rate limiter: hammering it at a fixed interval is how a
    throttle that would have lifted in ten seconds lasts the whole run. The cap
    stops a large `attempts` from producing a wait longer than the session.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    if backoff < 0:
        raise ValueError(f"backoff must not be negative, got {backoff}")
    return [min(backoff * factor**index, maximum) for index in range(max(0, attempts - 1))]


def fetch_with_retries(
    ticker: str,
    start: date,
    end: date,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    factor: float = DEFAULT_BACKOFF_FACTOR,
    maximum: float = DEFAULT_MAX_BACKOFF,
    settings: Settings | None = None,
    fetch: Fetcher | None = None,
    sleep: Sleeper | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch one ticker, retrying a failure with exponential backoff.

    Returns `(bars, attempts_used)`. Raises the last exception if every attempt
    failed — :func:`ingest_ticker` is what turns that into a status row, so
    that this function stays usable on its own when a caller does want the
    error.

    An *empty* frame is a successful answer and is returned immediately: Yahoo
    legitimately has no sessions in a window over a holiday, and day 12 re-runs
    over weekends constantly. Day 8's validation is where an empty answer is
    treated with suspicion, because there the question being asked is "does
    this symbol still exist".
    """
    delays = backoff_delays(attempts, backoff=backoff, factor=factor, maximum=maximum)
    sleeper = time.sleep if sleep is None else sleep

    for attempt in range(1, len(delays) + 2):
        try:
            if fetch is None:
                # Resolved through the module global at call time, so a test —
                # or day 55's dry run — can replace `fetch_daily` wholesale.
                bars = fetch_daily(ticker, start, end, settings=settings)
            else:
                bars = fetch(ticker, start, end)
        except Exception as exc:
            if attempt > len(delays):
                raise
            delay = delays[attempt - 1]
            log.warning(
                "%s: attempt %d/%d failed (%s: %s); retrying in %.1fs",
                ticker,
                attempt,
                len(delays) + 1,
                type(exc).__name__,
                exc,
                delay,
            )
            sleeper(delay)
        else:
            return bars, attempt

    # Unreachable: the loop either returns or raises on its last pass.
    raise AssertionError(f"{ticker}: retry loop fell through")


# ---------------------------------------------------------------------------
# One ticker
# ---------------------------------------------------------------------------


def ingest_ticker(
    ticker: str,
    start: date,
    end: date,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    factor: float = DEFAULT_BACKOFF_FACTOR,
    maximum: float = DEFAULT_MAX_BACKOFF,
    write: bool = True,
    settings: Settings | None = None,
    root: Path | str | None = None,
    fetch: Fetcher | None = None,
    sleep: Sleeper | None = None,
) -> TickerIngest:
    """Fetch one ticker, store it, and say what happened. Never raises.

    "Never raises" is the whole point, and it covers the write as well as the
    request: a full disk or a permission problem is this ticker's failure, not
    the run's. The only thing that gets through is a KeyboardInterrupt, which
    is the user asking for the run to stop rather than a symbol misbehaving.

    The write happens here, one ticker at a time, rather than being collected
    and done at the end. A run that dies halfway through then keeps everything
    it had already downloaded — which is the difference between losing fifteen
    minutes and losing the lot.
    """
    symbol = str(ticker).strip()
    started = time.monotonic()
    used = 0
    try:
        bars, used = fetch_with_retries(
            symbol,
            start,
            end,
            attempts=attempts,
            backoff=backoff,
            factor=factor,
            maximum=maximum,
            settings=settings,
            fetch=fetch,
            sleep=sleep,
        )
        stored = False
        if write and not bars.empty:
            write_bars(bars, settings=settings, root=root)
            stored = True
    except Exception as exc:
        # Every failure mode of a scraper ends up here — a timeout, a 404, a
        # rate limit, a response that would not conform to the schema, a failed
        # write. The run carries on; this ticker is one row saying why.
        detail = f"{type(exc).__name__}: {exc}"
        # `used` is still 0 when the request itself never succeeded, in which
        # case every attempt was spent on it. When it is set, the fetch worked
        # and what failed was the write, on that many requests.
        spent = used or attempts
        log.warning("%s: giving up after %d attempt(s): %s", symbol, spent, detail)
        log.debug("%s: the failure in full", symbol, exc_info=exc)
        return TickerIngest(
            ticker=symbol,
            status=Status.FAILED,
            attempts=spent,
            seconds=time.monotonic() - started,
            detail=detail,
        )

    if bars.empty:
        return TickerIngest(
            ticker=symbol,
            status=Status.EMPTY,
            attempts=used,
            seconds=time.monotonic() - started,
            detail=f"no sessions between {start} and {end}",
        )
    return TickerIngest(
        ticker=symbol,
        status=Status.OK,
        rows=len(bars),
        first_date=bars["date"].iloc[0],
        last_date=bars["date"].iloc[-1],
        attempts=used,
        seconds=time.monotonic() - started,
        written=stored,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def resolve_window(
    start: date | datetime | str | None,
    end: date | datetime | str | None,
    settings: Settings | None = None,
) -> tuple[date, date]:
    """The window to ask for: given bounds, else settings, else today.

    :func:`~finmgr.data.fetch.fetch_daily` defaults the same way on its own,
    but the loop resolves the dates once up front so that every ticker in one
    run is asked for exactly the same window — and so the report can say which
    window that was, including for the tickers that never got there.
    """
    first = as_date(start, "start")
    last = as_date(end, "end")
    if first is None or last is None:
        resolved = settings or load_settings()
        first = first or resolved.history_start
        last = last or datetime.now(ZoneInfo(resolved.timezone)).date()
    if first > last:
        raise ValueError(f"start {first} is after end {last}")
    return first, last


def ingest_universe(
    tickers: Iterable[Company | str] | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    factor: float = DEFAULT_BACKOFF_FACTOR,
    maximum: float = DEFAULT_MAX_BACKOFF,
    pause: float = DEFAULT_PAUSE,
    abort_after: int = DEFAULT_ABORT_AFTER,
    write: bool = True,
    settings: Settings | None = None,
    root: Path | str | None = None,
    fetch: Fetcher | None = None,
    sleep: Sleeper | None = None,
    on_result: Callable[[TickerIngest], None] | None = None,
) -> IngestReport:
    """Ingest every ticker in order and return one status row each.

    `tickers` defaults to `config/universe.csv`. Nothing in here raises for a
    ticker that fails; what comes back is an :class:`IngestReport` covering
    every symbol asked about, including the ones a stopped run never reached.

    The loop stops early in two cases, and in both it fills the remainder of
    the report in as `skipped` rather than pretending those tickers were fine:
    a KeyboardInterrupt, and `abort_after` consecutive failures — the shape a
    dead network makes, where continuing means hundreds of doomed requests with
    backoff between them.
    """
    sleeper = time.sleep if sleep is None else sleep
    symbols = [
        company.ticker if isinstance(company, Company) else str(company)
        for company in (load_universe(settings=settings) if tickers is None else tickers)
    ]
    first, last = resolve_window(start, end, settings)
    run = runlog.current_run()
    started_at = datetime.now(UTC)

    log.info(
        "ingesting %d ticker(s) from %s to %s (attempts=%d, pause=%.1fs, write=%s)",
        len(symbols),
        first,
        last,
        attempts,
        pause,
        write,
    )

    results: list[TickerIngest] = []
    consecutive = 0
    interrupted = False
    stopped_early = ""

    for position, symbol in enumerate(symbols):
        try:
            if position and pause:
                sleeper(pause)
            result = ingest_ticker(
                symbol,
                first,
                last,
                attempts=attempts,
                backoff=backoff,
                factor=factor,
                maximum=maximum,
                write=write,
                settings=settings,
                root=root,
                fetch=fetch,
                sleep=sleep,
            )
        except KeyboardInterrupt:
            # Ctrl-C, or the runner being told to stop. Not an error, and not a
            # reason to throw away the rows already on disk.
            log.warning("interrupted after %d of %d ticker(s)", len(results), len(symbols))
            interrupted = True
            stopped_early = f"interrupted after {len(results)} of {len(symbols)}"
            break

        results.append(result)
        getattr(log, "info" if result.succeeded else "warning")(
            "%s [%d/%d]: %s (%d bars%s)",
            symbol,
            position + 1,
            len(symbols),
            result.status,
            result.rows,
            f", {result.detail}" if result.detail else "",
        )
        if on_result is not None:
            on_result(result)

        if result.status is Status.FAILED:
            consecutive += 1
            if abort_after and consecutive >= abort_after:
                stopped_early = f"{consecutive} consecutive failures"
                log.error(
                    "%s after %d of %d ticker(s); skipping the rest",
                    stopped_early,
                    len(results),
                    len(symbols),
                )
                break
        else:
            consecutive = 0

    for symbol in symbols[len(results) :]:
        results.append(
            TickerIngest(
                ticker=symbol,
                status=Status.SKIPPED,
                detail=stopped_early or "not attempted",
            )
        )

    report = IngestReport(
        results=results,
        start=first,
        end=last,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        interrupted=interrupted,
        stopped_early=stopped_early,
        run_id=run.run_id if run is not None else None,
    )
    log.info("ingest finished: %s", report.summary())
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_console(report: IngestReport, console: Console, *, show_all: bool = False) -> None:
    """Print what succeeded, then what did not.

    This is what has to appear on screen after the cable comes out: the run may
    have got nowhere, but it still says exactly which tickers are on disk and
    why the others are not.
    """
    succeeded = report.succeeded
    if succeeded:
        table = Table(title="Ingested")
        table.add_column("ticker", style="bold")
        table.add_column("status")
        table.add_column("rows", justify="right")
        table.add_column("first bar")
        table.add_column("last bar")
        table.add_column("tries", justify="right")
        shown: Sequence[TickerIngest] = succeeded if show_all else succeeded[:20]
        for result in shown:
            table.add_row(
                result.ticker,
                str(result.status),
                str(result.rows),
                result.first_date.isoformat() if result.first_date else "-",
                result.last_date.isoformat() if result.last_date else "-",
                str(result.attempts),
            )
        console.print(table)
        if len(shown) < len(succeeded):
            console.print(f"[dim]... and {len(succeeded) - len(shown)} more (--all to list).[/dim]")
    else:
        console.print("[yellow]Nothing was ingested.[/yellow]")

    unfinished = report.failed + report.skipped
    if unfinished:
        table = Table(title="Not ingested")
        table.add_column("ticker", style="bold")
        table.add_column("status")
        table.add_column("tries", justify="right")
        table.add_column("detail", overflow="fold")
        for result in unfinished:
            table.add_row(result.ticker, str(result.status), str(result.attempts), result.detail)
        console.print(table)

    console.print(report.summary())
