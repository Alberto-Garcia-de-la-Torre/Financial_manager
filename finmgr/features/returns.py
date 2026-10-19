"""Returns: the series every later feature is built on.

    from finmgr.features.returns import compute_returns, read_returns

    panel = read_returns(as_of="2024-06-28")                 # all 100, every horizon
    panel.loc[panel["date"] == pd.Timestamp("2024-06-28").date(), ["ticker", "ret_5d"]]

For each horizon `h` in :data:`HORIZONS` — 1, 5, 21, 63, 126 and 252 sessions,
a day, a week, a month, a quarter, half a year and a year — two columns:

    ret_{h}d     simple return   p[t] / p[t-h] - 1
    logret_{h}d  log return      ln p[t] - ln p[t-h]

**Strictly backward-looking.** The value on date `t` is built from the price
on `t` and the price `h` sessions *before* it, and nothing else. There is no
centring, no `shift(-h)`, no fill from a later row. `tests/test_returns.py`
proves it the blunt way: scramble every price after `t` and check that
`ret_5d` at `t` does not move.

**The price.** `adj_close_eur` from
:func:`~finmgr.data.benchmarks.read_total_return` by default: the back-adjusted
close (dividends reinvested, splits undone) in euros, so a return is what a
euro investor actually earned, and the 100 compare on one footing. Pass
`price="adj_close"` for the listing-currency return without the FX leg.
Back-adjusting as of a later date multiplies both ends of every window that
closes before the event by the same factor, so a return dated `t` is the same
whether the series was read as of `t` or as of today — the test checks that
too.

**What a session is.** A ticker's own venue calendar (day 17), not "the
previous row". When a bar is missing — never stored, or hidden by day 20's
quarantine — counting rows would quietly stretch a 5-session window to six.
So each ticker's prices are first laid on its venue's sessions between its
first bar and its last; a missing session becomes a NaN price, and only the
returns with that session as an *endpoint* turn NaN. A window that merely
spans it is still exact, because a simple return only needs its two ends.
Sessions in :data:`~finmgr.data.calendars.UNSCHEDULED_CLOSURES` are taken out
(the venue did not trade); a bar on a day the calendar calls closed is kept as
a session, since the store holds it. :func:`compute_returns` without
`sessions` falls back to counting rows, for callers that have already aligned.

A NaN price — no EUR close yet — gives NaN returns at both ends of its windows;
it is never filled from a neighbour.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.data.benchmarks import read_total_return
from finmgr.data.calendars import UNSCHEDULED_CLOSURES, expected_sessions
from finmgr.data.universe import load_universe

#: Lookbacks in sessions: a day, a week, a month, a quarter, half a year, a year.
HORIZONS: tuple[int, ...] = (1, 5, 21, 63, 126, 252)

#: The price column returns are taken on unless asked otherwise.
DEFAULT_PRICE = "adj_close_eur"


def simple_column(horizon: int) -> str:
    """`ret_5d` for 5."""
    return f"ret_{horizon}d"


def log_column(horizon: int) -> str:
    """`logret_5d` for 5."""
    return f"logret_{horizon}d"


def return_columns(horizons: Sequence[int] = HORIZONS) -> list[str]:
    """Every return column for `horizons`, simple ones first."""
    return [simple_column(h) for h in horizons] + [log_column(h) for h in horizons]


def _check_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(h) for h in horizons)
    if not out or any(h < 1 for h in out):
        # A horizon of 0 is a constant, and a negative one is a forward return.
        raise ValueError(f"horizons must be positive session counts, got {list(horizons)}")
    return out


def _one_ticker(
    frame: pd.DataFrame, price: str, horizons: tuple[int, ...], grid: Sequence[date] | None
) -> pd.DataFrame:
    """Returns for one ticker's rows, sorted by date, optionally on a session grid."""
    prices = frame.set_index("date")[price].astype(float)
    if grid is not None:
        # The stored dates are always on the grid: a bar the calendar did not
        # expect is still a bar.
        index = pd.Index(sorted(set(grid) | set(prices.index)), name="date")
        prices = prices.reindex(index)
    # A non-positive adjusted price is bad data, not a -100% return; and log
    # of it is undefined. Either way the windows touching it are NaN.
    prices = prices.where(prices > 0)
    logs = np.log(prices)
    out = {}
    for h in horizons:
        # shift(h) moves each value h sessions *later*: the row for t sees the
        # price from t-h. Never a negative shift.
        out[simple_column(h)] = prices / prices.shift(h) - 1.0
    for h in horizons:
        out[log_column(h)] = logs - logs.shift(h)
    result = pd.DataFrame(out, index=prices.index)
    return result.loc[frame["date"]]


def compute_returns(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    horizons: Sequence[int] = HORIZONS,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> pd.DataFrame:
    """Simple and log returns over `horizons` sessions, per ticker, backward only.

    `prices` is long: one row per `(ticker, date)` with a `price` column.
    `sessions` maps a ticker to the sessions its venue held; a ticker not in
    it, or no mapping at all, counts its own rows as sessions. Returns
    `ticker, date, <price>` and :func:`return_columns`, one row per input row,
    sorted by `(ticker, date)`.
    """
    horizons = _check_horizons(horizons)
    missing = {"ticker", "date", price} - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    base = prices[["ticker", "date", price]].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in ["ticker", "date", price, *return_columns(horizons)]})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, horizons, grid).reset_index(drop=True)
        parts.append(pd.concat([frame.reset_index(drop=True), part], axis=1))
    return pd.concat(parts, ignore_index=True)


def venue_sessions(venue: str, start: date, end: date) -> list[date]:
    """The sessions `venue` actually held in [start, end], unscheduled closures out."""
    closed = UNSCHEDULED_CLOSURES.get(venue, {})
    days: list[date] = []
    for year in range(start.year, end.year + 1):
        days += expected_sessions(venue, year, start=start, end=end)
    return [day for day in days if day not in closed]


def _venues(settings: Settings) -> dict[str, str]:
    """`{ticker: MIC}` for the universe and the benchmarks together."""
    venues = {c.ticker: c.exchange for c in load_universe(settings=settings)}
    for company in load_universe(settings.benchmarks_path):
        venues.setdefault(company.ticker, company.exchange)
    return venues


def read_returns(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    horizons: Sequence[int] = HORIZONS,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_returns` on :func:`read_total_return`, on each venue's calendar.

    Nothing after `as_of` is read. `start` trims the *output*; enough history
    before it is read that the longest horizon is defined from `start` on.
    `root` is a data directory as in :func:`read_total_return` (the benchmarks
    one included); venues come from the universe and benchmark files.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    horizons = _check_horizons(horizons)
    first = date.fromisoformat(start) if isinstance(start, str) else start
    # The read below starts before `start`, so the data layer would not see a
    # `start` in the future; refuse it here, as it would.
    pit.bound_end(None, cutoff, start=first)
    # Two calendar days per session covers weekends and every holiday season.
    read_from = None if first is None else first - timedelta(days=2 * max(horizons) + 14)
    prices = read_total_return(tickers, start=read_from, as_of=cutoff, settings=resolved, root=root)
    venues = _venues(resolved)
    sessions = {}
    for ticker, frame in prices.groupby("ticker", sort=False):
        if ticker in venues:
            sessions[ticker] = venue_sessions(
                venues[ticker], frame["date"].min(), frame["date"].max()
            )
    out = compute_returns(prices, price=price, horizons=horizons, sessions=sessions)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_returns")
