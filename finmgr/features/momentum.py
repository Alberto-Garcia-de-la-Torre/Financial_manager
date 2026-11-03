"""Momentum: 12-1, 6-1 and 5-day short-term reversal.

    from finmgr.features.momentum import read_momentum

    panel = read_momentum(as_of="2024-06-28", start="2024-01-01")

Three columns, every one counted in venue sessions exactly as
:mod:`finmgr.features.returns` counts them:

    mom_12_1   p[t-21] / p[t-252] - 1   the year, minus its last month
    mom_6_1    p[t-21] / p[t-126] - 1   the half year, minus its last month
    rev_5d     -(p[t] / p[t-5] - 1)     last week's return, sign flipped

**Why skip the last month.** Over 3-12 months, winners keep winning
(Jegadeesh-Titman); over the most recent month they tend to give some of it
back. A 252-session return that includes the last 21 sessions mixes the two and
dilutes the signal, so 12-1 is the return from t-252 to t-21 and the last month
is left out — it is the short-term reversal's territory instead.

**Why rev_5d is negated.** It is a signal, not a return: the 5-day loser is the
expected rebounder, so `rev_5d` is high for last week's losers. Every column
here then reads the same way round — higher is the side the effect favours.
`ret_5d` from the returns module is the un-negated value.

**Strictly backward-looking.** Every column at `t` uses prices at `t` and
earlier only (`shift` by a positive count, never a negative one); 12-1 and 6-1
do not even use `t` itself. A missing session at either end of a window makes
that value NaN; a missing session inside it does not matter, since only the two
ends are used.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.data.benchmarks import read_total_return
from finmgr.features.returns import DEFAULT_PRICE, session_prices, ticker_sessions

#: `{column: (lookback, skip)}` in sessions: the window runs from t-lookback to t-skip.
WINDOWS: dict[str, tuple[int, int]] = {
    "mom_12_1": (252, 21),
    "mom_6_1": (126, 21),
}

#: The short-term reversal lookback, in sessions.
REVERSAL_SESSIONS = 5
REVERSAL_COLUMN = "rev_5d"

MOMENTUM_COLUMNS: tuple[str, ...] = (*WINDOWS, REVERSAL_COLUMN)

#: The longest lookback any column needs.
MAX_LOOKBACK = max(max(lookback for lookback, _ in WINDOWS.values()), REVERSAL_SESSIONS)


def skip_return(prices: pd.Series, lookback: int, skip: int) -> pd.Series:
    """`p[t-skip] / p[t-lookback] - 1` on a session-aligned series.

    `skip=0` is the plain `lookback`-session return. Both counts are positive
    shifts, so nothing after t is ever read.
    """
    if not 0 <= skip < lookback:
        raise ValueError(f"need 0 <= skip < lookback, got lookback={lookback}, skip={skip}")
    return prices.shift(skip) / prices.shift(lookback) - 1.0


def _one_ticker(frame: pd.DataFrame, price: str, grid: Sequence[date] | None) -> pd.DataFrame:
    prices = session_prices(frame, price, grid)
    out = {col: skip_return(prices, lookback, skip) for col, (lookback, skip) in WINDOWS.items()}
    out[REVERSAL_COLUMN] = -skip_return(prices, REVERSAL_SESSIONS, 0)
    return pd.DataFrame(out, index=prices.index).loc[frame["date"]]


def compute_momentum(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> pd.DataFrame:
    """12-1, 6-1 momentum and 5-day reversal per ticker, backward only.

    Same contract as :func:`~finmgr.features.returns.compute_returns`: `prices`
    is long, one row per `(ticker, date)`; `sessions` maps a ticker to its
    venue's sessions, and without it rows are counted as sessions. Returns
    `ticker, date, <price>` and :data:`MOMENTUM_COLUMNS`, sorted by
    `(ticker, date)`.
    """
    missing = {"ticker", "date", price} - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    base = prices[["ticker", "date", price]].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in ["ticker", "date", price, *MOMENTUM_COLUMNS]})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, grid).reset_index(drop=True)
        parts.append(pd.concat([frame.reset_index(drop=True), part], axis=1))
    return pd.concat(parts, ignore_index=True)


def read_momentum(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_momentum` on :func:`read_total_return`, on each venue's calendar.

    Nothing after `as_of` is read; `start` trims the output, and enough history
    before it is read for 12-1 to be defined from `start` on.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    read_from = None if first is None else first - timedelta(days=2 * MAX_LOOKBACK + 14)
    prices = read_total_return(tickers, start=read_from, as_of=cutoff, settings=resolved, root=root)
    sessions = ticker_sessions(prices, resolved)
    out = compute_momentum(prices, price=price, sessions=sessions)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_momentum")
