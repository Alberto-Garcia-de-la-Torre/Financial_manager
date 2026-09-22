"""Does every ticker in the universe still return bars from Yahoo?

`config/universe.csv` is hand-written, and Yahoo's suffixes are the easiest
thing in this project to get quietly wrong: `AIR.PA` is Airbus in Paris,
`AIR.DE` is a different listing, `AIR` on its own is AAR Corp in New York.
None of those mistakes raise — yfinance answers an unknown or delisted symbol
with an *empty* DataFrame, so a typo becomes a company that silently never
appears in any ranking.

So this module asks Yahoo for the last few sessions of every ticker and sorts
the answers into four buckets:

    ok      a non-empty frame whose newest bar is recent
    stale   bars, but the newest one is older than --max-age-days
    empty   a frame with no rows: wrong suffix, or the listing is gone
    error   the request itself failed

Run it as a script; it exits non-zero if anything is not `ok`:

    python -m finmgr.data.validate
    python -m finmgr.data.validate --report docs/universe_validation.md

This is deliberately *not* the ingestion path. It fetches five days, keeps
nothing, and writes nothing under `data/`. Day 11 is where downloading becomes
a pipeline with backoff and a status row per ticker.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.table import Table

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data.universe import Company, load_universe

#: How much history to ask for. Five sessions is enough to tell "trading
#: normally" from "nothing here", and small enough to be polite 100 times over.
DEFAULT_PERIOD = "5d"

#: How old the newest bar may be before the ticker counts as stale. A run on a
#: Monday morning legitimately sees Friday's close, and a public holiday adds a
#: day or two on top, so a week is the smallest threshold that is not noisy.
DEFAULT_MAX_AGE_DAYS = 7

#: Attempts per ticker, and the pause between them. yfinance is an unofficial,
#: rate-limited scraper: a single empty answer is not proof a listing is dead.
DEFAULT_ATTEMPTS = 3
DEFAULT_RETRY_PAUSE = 2.0

#: Pause between tickers, to stay under Yahoo's rate limiting.
DEFAULT_PAUSE = 0.3

log = runlog.get_logger(__name__)


class Status(StrEnum):
    """The verdict on one ticker."""

    OK = "ok"
    STALE = "stale"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TickerCheck:
    """What Yahoo said about one ticker."""

    ticker: str
    status: Status
    rows: int = 0
    first_date: date | None = None
    last_date: date | None = None
    age_days: int | None = None
    attempts: int = 1
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "status": str(self.status),
            "rows": self.rows,
            "first_date": self.first_date.isoformat() if self.first_date else None,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "age_days": self.age_days,
            "attempts": self.attempts,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Talking to Yahoo
# ---------------------------------------------------------------------------


def fetch_history(ticker: str, period: str = DEFAULT_PERIOD) -> Any:
    """The one network call: `yf.Ticker(t).history(period=...)`.

    Imported lazily so the module can be imported — and tested against a fake
    fetcher — without paying for yfinance's import.
    """
    import yfinance as yf

    return yf.Ticker(ticker).history(period=period)


#: Signature of a replacement fetcher, which is how the tests stay offline.
#: Both functions below take it as an optional argument and fall back to
#: :func:`fetch_history` *at call time* rather than binding it as a default, so
#: monkeypatching the module attribute is enough to take the network away.
Fetcher = Callable[[str, str], Any]
Sleeper = Callable[[float], None]


def _frame_dates(frame: Any) -> list[date]:
    """Bar dates out of whatever yfinance handed back, newest last.

    The index is a tz-aware DatetimeIndex in exchange-local time. Only the
    calendar date matters here, and taking it before converting anywhere else
    keeps a Tokyo bar dated the day it traded in Tokyo.
    """
    if frame is None or len(frame) == 0:
        return []
    dates = []
    for stamp in frame.index:
        as_date = getattr(stamp, "date", None)
        dates.append(as_date() if callable(as_date) else date.fromisoformat(str(stamp)[:10]))
    return sorted(dates)


def classify(
    ticker: str,
    frame: Any,
    *,
    today: date,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    attempts: int = 1,
) -> TickerCheck:
    """Turn one response into a verdict. Pure: no network, no clock."""
    dates = _frame_dates(frame)
    if not dates:
        return TickerCheck(
            ticker=ticker,
            status=Status.EMPTY,
            attempts=attempts,
            detail="no bars returned",
        )

    first, last = dates[0], dates[-1]
    age = (today - last).days
    stale = age > max_age_days
    return TickerCheck(
        ticker=ticker,
        status=Status.STALE if stale else Status.OK,
        rows=len(dates),
        first_date=first,
        last_date=last,
        age_days=age,
        attempts=attempts,
        detail=f"newest bar is {age} days old" if stale else "",
    )


def check_ticker(
    ticker: str,
    *,
    today: date,
    period: str = DEFAULT_PERIOD,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    attempts: int = DEFAULT_ATTEMPTS,
    retry_pause: float = DEFAULT_RETRY_PAUSE,
    fetch: Fetcher | None = None,
    sleep: Sleeper | None = None,
) -> TickerCheck:
    """Check one ticker, retrying an empty or failed answer.

    An empty frame is retried as well as an exception, and on purpose: Yahoo
    answers a throttled request the same way it answers a dead symbol, and
    condemning a live listing on one empty response would send someone editing
    the universe file for no reason. A symbol that is genuinely gone comes back
    empty every time, so the retries cost a few seconds and settle the question.
    """
    fetch = fetch_history if fetch is None else fetch
    sleep = time.sleep if sleep is None else sleep

    result = TickerCheck(ticker=ticker, status=Status.ERROR, detail="not attempted")
    for attempt in range(1, max(1, attempts) + 1):
        try:
            frame = fetch(ticker, period)
        # Any failure is this ticker's failure, not the run's: a timeout, a
        # JSON decode error and a rate-limit rejection all mean "ask again".
        except Exception as exc:
            result = TickerCheck(
                ticker=ticker,
                status=Status.ERROR,
                attempts=attempt,
                detail=f"{type(exc).__name__}: {exc}",
            )
        else:
            result = classify(
                ticker,
                frame,
                today=today,
                max_age_days=max_age_days,
                attempts=attempt,
            )

        if result.ok:
            return result
        if attempt < max(1, attempts):
            log.debug("%s: %s on attempt %d/%d, retrying", ticker, result.status, attempt, attempts)
            sleep(retry_pause)
    return result


def check_universe(
    companies: Iterable[Company | str],
    *,
    today: date | None = None,
    period: str = DEFAULT_PERIOD,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    attempts: int = DEFAULT_ATTEMPTS,
    retry_pause: float = DEFAULT_RETRY_PAUSE,
    pause: float = DEFAULT_PAUSE,
    fetch: Fetcher | None = None,
    sleep: Sleeper | None = None,
    on_result: Callable[[TickerCheck], None] | None = None,
) -> list[TickerCheck]:
    """Check every ticker in order, returning one verdict each."""
    sleep = time.sleep if sleep is None else sleep
    symbols = [c.ticker if isinstance(c, Company) else str(c) for c in companies]
    reference = today or date.today()

    results: list[TickerCheck] = []
    for position, ticker in enumerate(symbols):
        if position and pause:
            sleep(pause)
        result = check_ticker(
            ticker,
            today=reference,
            period=period,
            max_age_days=max_age_days,
            attempts=attempts,
            retry_pause=retry_pause,
            fetch=fetch,
            sleep=sleep,
        )
        results.append(result)
        level = "debug" if result.ok else "warning"
        getattr(log, level)(
            "%s [%d/%d]: %s (%d bars, last %s)",
            ticker,
            position + 1,
            len(symbols),
            result.status,
            result.rows,
            result.last_date or "-",
        )
        if on_result is not None:
            on_result(result)
    return results


def failures(results: Sequence[TickerCheck]) -> list[TickerCheck]:
    """Everything that was not `ok` — the list day 8 needs to be empty."""
    return [result for result in results if not result.ok]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _counts(results: Sequence[TickerCheck]) -> Counter[str]:
    return Counter(str(result.status) for result in results)


def render_console(results: Sequence[TickerCheck], console: Console) -> None:
    """Print the failure list, which is the whole point of the script."""
    bad = failures(results)
    counts = _counts(results)

    if bad:
        table = Table(title="Tickers that did not return recent bars")
        table.add_column("ticker", style="bold")
        table.add_column("status")
        table.add_column("rows", justify="right")
        table.add_column("last bar")
        table.add_column("detail", overflow="fold")
        for result in bad:
            table.add_row(
                result.ticker,
                str(result.status),
                str(result.rows),
                result.last_date.isoformat() if result.last_date else "-",
                result.detail,
            )
        console.print(table)
    else:
        console.print("[green]Failure list is empty.[/green]")

    summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    console.print(f"{counts['ok']} of {len(results)} returned recent bars ({summary}).")


def render_markdown(
    results: Sequence[TickerCheck],
    *,
    checked_at: datetime,
    period: str,
    max_age_days: int,
    run_id: str | None = None,
) -> str:
    """A committable record of one validation run."""
    bad = failures(results)
    counts = _counts(results)

    lines = [
        "# Universe validation",
        "",
        "Generated by `python -m finmgr.data.validate --report` (roadmap day 8).",
        "Every ticker in `config/universe.csv` is asked for its last few sessions;",
        "the acceptance criterion is that the failure list below is empty.",
        "",
        f"- Checked at: {checked_at.isoformat(timespec='seconds')}",
        f'- Request: `yf.Ticker(t).history(period="{period}")`',
        f"- Recent means: newest bar at most {max_age_days} days old",
        f"- Result: **{counts['ok']} of {len(results)} returned recent bars**",
    ]
    if run_id:
        lines.append(f"- Run id: `{run_id}`")
    lines += [
        "",
        "## Failures",
        "",
    ]
    if bad:
        lines += [
            "| ticker | status | rows | last bar | detail |",
            "| --- | --- | ---: | --- | --- |",
        ]
        lines += [
            f"| {r.ticker} | {r.status} | {r.rows} | "
            f"{r.last_date.isoformat() if r.last_date else '-'} | {r.detail} |"
            for r in bad
        ]
    else:
        lines.append("None. All tickers returned recent bars.")
    lines += [
        "",
        "## Every ticker",
        "",
        "| ticker | status | rows | first bar | last bar | age (days) |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    lines += [
        f"| {r.ticker} | {r.status} | {r.rows} | "
        f"{r.first_date.isoformat() if r.first_date else '-'} | "
        f"{r.last_date.isoformat() if r.last_date else '-'} | "
        f"{'-' if r.age_days is None else r.age_days} |"
        for r in results
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finmgr.data.validate",
        description="Check that every ticker in the universe returns recent bars from Yahoo.",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        metavar="CSV",
        help="Universe file to check (default: the one in config/settings.yaml).",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        default=None,
        metavar="SYMBOL",
        help="Check only this symbol; repeatable. Skips the universe file.",
    )
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD, help=f"History window (default: {DEFAULT_PERIOD})."
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"How old the newest bar may be (default: {DEFAULT_MAX_AGE_DAYS}).",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"Tries per ticker before giving up (default: {DEFAULT_ATTEMPTS}).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help=f"Seconds between tickers (default: {DEFAULT_PAUSE}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write a Markdown report of the run to this file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        metavar="LEVEL",
        help="Console verbosity (default: INFO). The file log always keeps DEBUG.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """Validate the universe. Returns 0 only if every ticker came back `ok`."""
    args = _parse_args(argv)
    resolved = settings or load_settings()

    console = Console()
    run = runlog.start_run(
        resolved,
        level=args.log_level.upper(),
        command="validate-universe",
        console=console,
    )
    status, exit_code = "ok", 0
    try:
        companies: list[Company | str]
        if args.ticker:
            companies = list(args.ticker)
        else:
            companies = list(load_universe(args.universe, settings=resolved))

        today = datetime.now(ZoneInfo(resolved.timezone)).date()
        log.info(
            "checking %d tickers against Yahoo (period=%s, max_age_days=%d)",
            len(companies),
            args.period,
            args.max_age_days,
        )

        results = check_universe(
            companies,
            today=today,
            period=args.period,
            max_age_days=args.max_age_days,
            attempts=args.attempts,
            pause=args.pause,
        )

        render_console(results, console)
        bad = failures(results)

        if args.report is not None:
            report = render_markdown(
                results,
                checked_at=datetime.now(ZoneInfo(resolved.timezone)),
                period=args.period,
                max_age_days=args.max_age_days,
                run_id=run.run_id,
            )
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
            log.info("wrote report to %s", args.report)

        if bad:
            log.error(
                "%d of %d tickers did not return recent bars: %s",
                len(bad),
                len(results),
                ", ".join(result.ticker for result in bad),
            )
            status, exit_code = "failed", 1
        else:
            log.info("all %d tickers returned recent bars", len(results))
    except Exception:
        log.exception("universe validation crashed")
        status, exit_code = "crashed", 1
    finally:
        runlog.end_run(status=status, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
