"""Volatility and risk: realized vol, downside deviation, ATR(14), max drawdown.

    from finmgr.features.volatility import read_volatility

    panel = read_volatility(as_of="2024-06-28", start="2024-01-01")

Columns, every window counted in venue sessions exactly as
:mod:`finmgr.features.returns` counts them:

    vol_21d, vol_63d, vol_252d   std of daily log returns over the window, x sqrt(252)
    downdev_63d                  sqrt(mean(min(r, 0)^2)) over 63 sessions, x sqrt(252)
    atr_14                       Wilder's average true range, 14 sessions, in price units
    atr_14_pct                   atr_14 / close: the same, as a fraction of the price
    mdd_252d                     worst peak-to-trough fall within the last 252 sessions (<= 0)

**Annualized, as decimals.** `vol_63d = 0.22` is 22% a year. Daily log returns
are scaled by sqrt(252), the sessions in a trading year; the standard deviation
uses ddof=1. A typical large cap lands in 0.15-0.35.

**Downside deviation** is the target-zero semideviation: only the down days
contribute, but the mean is over every day in the window, so a stock that never
falls scores 0 and one whose falls are as large as its rises scores about
`vol / sqrt(2)`.

**ATR(14)** is Wilder's: the true range `max(H - L, |H - C[t-1]|, |L - C[t-1]|)`,
seeded with the plain mean of the first 14 and then smoothed as
`ATR[t] = (13 * ATR[t-1] + TR[t]) / 14`. Highs and lows are back-adjusted with
the close's own factor (`adj_close / close`), so a split is not a true range of
50%. The recursion means `atr_14` in price units depends on every adjustment
known at read time; `atr_14_pct` does not (each factor scales numerator and
denominator alike) and is the column to compare across tickers and dates.

**Max drawdown** at t looks at the 253 prices from t-252 to t — the same span as
`ret_252d` — and returns the largest fall from a running peak to a later trough
inside that window, as a non-positive fraction: `-0.30` is a 30% drawdown.

**Prices.** Vol, downside deviation and drawdown use `adj_close_eur` by default,
the series the returns are on, so a non-euro stock's risk includes its currency
leg. ATR is a range measure and uses the listing-currency adjusted bar.

**Strictly backward-looking.** Every value at t uses bars at t and earlier only.

**Missing sessions.** Prices are laid on the venue's session grid first; a
missing session is a NaN, which removes the two daily returns touching it (and
one true range). A window still produces a value if at least
:data:`MIN_COVERAGE` of its observations are there — one quarantined bar should
not blank out a whole year of `vol_252d` — but never before the window's full
span of history exists: the first `vol_63d` is at the 64th bar, not earlier on
fewer returns. ATR skips a NaN true range (NaN at that row) and resumes the
recursion from the last value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit, store
from finmgr.data.benchmarks import read_total_return
from finmgr.features.returns import DEFAULT_PRICE, session_prices, ticker_sessions

#: Sessions in a trading year: the annualization factor is its square root.
SESSIONS_PER_YEAR = 252

#: Realized-volatility windows, in sessions.
VOL_WINDOWS: tuple[int, ...] = (21, 63, 252)

#: The downside-deviation window, in sessions.
DOWNSIDE_WINDOW = 63

#: Wilder's ATR period.
ATR_PERIOD = 14

#: The max-drawdown window: the span from t-252 to t.
DRAWDOWN_WINDOW = 252

#: The fraction of a window's observations that must be present for a value.
MIN_COVERAGE = 0.9

#: The bar columns ATR needs, besides `ticker`, `date` and the price.
BAR_INPUTS: tuple[str, ...] = ("high", "low", "close", "adj_close")


def vol_column(window: int) -> str:
    """`vol_63d` for 63."""
    return f"vol_{window}d"


DOWNSIDE_COLUMN = f"downdev_{DOWNSIDE_WINDOW}d"
ATR_COLUMN = f"atr_{ATR_PERIOD}"
ATR_PCT_COLUMN = f"atr_{ATR_PERIOD}_pct"
DRAWDOWN_COLUMN = f"mdd_{DRAWDOWN_WINDOW}d"

VOLATILITY_COLUMNS: tuple[str, ...] = (
    *(vol_column(w) for w in VOL_WINDOWS),
    DOWNSIDE_COLUMN,
    ATR_COLUMN,
    ATR_PCT_COLUMN,
    DRAWDOWN_COLUMN,
)

#: The longest lookback any column needs, in sessions.
MAX_LOOKBACK = max(*VOL_WINDOWS, DOWNSIDE_WINDOW, ATR_PERIOD, DRAWDOWN_WINDOW)


def _needed(window: int) -> int:
    return math.ceil(MIN_COVERAGE * window)


def _warm(values: pd.Series, span: int) -> pd.Series:
    """NaN for the first `span` positions: the window has not got its history yet."""
    out = values.copy()
    out.iloc[:span] = np.nan
    return out


def realized_vol(log_returns: pd.Series, window: int) -> pd.Series:
    """Annualized std of `window` daily log returns ending at each row."""
    std = log_returns.rolling(window, min_periods=_needed(window)).std(ddof=1)
    return _warm(std * math.sqrt(SESSIONS_PER_YEAR), window)


def downside_deviation(log_returns: pd.Series, window: int) -> pd.Series:
    """Annualized target-zero semideviation of `window` daily log returns."""
    squared = log_returns.clip(upper=0.0) ** 2
    mean = squared.rolling(window, min_periods=_needed(window)).mean()
    return _warm(np.sqrt(mean) * math.sqrt(SESSIONS_PER_YEAR), window)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """`max(H - L, |H - C[t-1]|, |L - C[t-1]|)`; NaN if any input, or C[t-1], is."""
    prev = close.shift(1)
    parts = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1)
    return parts.max(axis=1, skipna=False)


def wilder_atr(tr: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder's ATR: mean of the first `period` true ranges, then `(n-1)/n` smoothing.

    A NaN true range gives NaN at that row and leaves the running value as it
    was; the recursion picks up again at the next valid one.
    """
    values = tr.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    seed: list[float] = []
    atr = math.nan
    for i, x in enumerate(values):
        if math.isnan(x):
            continue
        if len(seed) < period:
            seed.append(x)
            if len(seed) == period:
                atr = sum(seed) / period
                out[i] = atr
            continue
        atr = (atr * (period - 1) + x) / period
        out[i] = atr
    return pd.Series(out, index=tr.index)


def max_drawdown(prices: pd.Series, window: int = DRAWDOWN_WINDOW) -> pd.Series:
    """Worst peak-to-trough fall within the `window + 1` prices ending at each row.

    Non-positive; NaN until `window + 1` positions exist or when fewer than
    :data:`MIN_COVERAGE` of them have a price.
    """
    values = prices.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    span = window + 1
    if len(values) >= span:
        windows = np.lib.stride_tricks.sliding_window_view(values, span)
        # fmax ignores NaN, so a missing price neither sets nor breaks a peak.
        peaks = np.fmax.accumulate(windows, axis=1)
        with np.errstate(invalid="ignore"):
            falls = windows / peaks - 1.0
        enough = np.sum(~np.isnan(windows), axis=1) >= _needed(span)
        worst = np.full(len(windows), np.nan)
        worst[enough] = np.nanmin(falls[enough], axis=1)
        out[window:] = worst
    return pd.Series(out, index=prices.index)


def _one_ticker(
    frame: pd.DataFrame, price: str, grid: Sequence[date] | None, with_atr: bool
) -> pd.DataFrame:
    prices = session_prices(frame, price, grid)
    logret = np.log(prices) - np.log(prices.shift(1))
    out = {vol_column(w): realized_vol(logret, w) for w in VOL_WINDOWS}
    out[DOWNSIDE_COLUMN] = downside_deviation(logret, DOWNSIDE_WINDOW)
    if with_atr:
        factor = (frame["adj_close"] / frame["close"]).to_numpy()
        bars = pd.DataFrame(
            {
                "high": frame["high"].to_numpy() * factor,
                "low": frame["low"].to_numpy() * factor,
                "close": frame["adj_close"].to_numpy(),
            },
            index=frame["date"].to_numpy(),
        )
        bars = bars.where(bars > 0).reindex(prices.index)
        atr = wilder_atr(true_range(bars["high"], bars["low"], bars["close"]))
        out[ATR_COLUMN] = atr
        out[ATR_PCT_COLUMN] = atr / bars["close"]
    else:
        out[ATR_COLUMN] = out[ATR_PCT_COLUMN] = pd.Series(np.nan, index=prices.index)
    out[DRAWDOWN_COLUMN] = max_drawdown(prices)
    return pd.DataFrame(out, index=prices.index).loc[frame["date"]]


def compute_volatility(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> pd.DataFrame:
    """Every :data:`VOLATILITY_COLUMNS` per ticker, backward only.

    Same contract as :func:`~finmgr.features.returns.compute_returns`. ATR also
    needs :data:`BAR_INPUTS` (raw `high`, `low`, `close` and the back-adjusted
    `adj_close`); without them the two ATR columns are NaN. Returns `ticker,
    date, <price>` and :data:`VOLATILITY_COLUMNS`, sorted by `(ticker, date)`.
    """
    missing = {"ticker", "date", price} - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    with_atr = set(BAR_INPUTS) <= set(prices.columns)
    keep = ["ticker", "date", price, *(c for c in BAR_INPUTS if with_atr and c != price)]
    base = prices[keep].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in ["ticker", "date", price, *VOLATILITY_COLUMNS]})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, grid, with_atr).reset_index(drop=True)
        head = frame[["ticker", "date", price]].reset_index(drop=True)
        parts.append(pd.concat([head, part], axis=1))
    return pd.concat(parts, ignore_index=True)


def read_volatility(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_volatility` on the store, on each venue's calendar.

    Closes come from :func:`read_total_return`, highs and lows from the bars.
    Nothing after `as_of` is read; `start` trims the output, and enough history
    before it is read for every column to be defined from `start` on. ATR's
    recursion starts at the first bar read, so a later `start` can move
    `atr_14` in the last decimals; it converges within a few dozen sessions.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    # Twice the lookback, plus room for ATR's recursion to forget its seed.
    read_from = None if first is None else first - timedelta(days=2 * MAX_LOOKBACK + 120)
    closes = read_total_return(tickers, start=read_from, as_of=cutoff, settings=resolved, root=root)
    base = Path(root) if root is not None else resolved.data_dir
    bars = store.read_bars(
        tickers,
        start=read_from,
        as_of=cutoff,
        settings=resolved,
        root=base / "bars" / "daily",
    )
    frame = closes.merge(bars[["ticker", "date", "high", "low"]], on=["ticker", "date"], how="left")
    sessions = ticker_sessions(frame, resolved)
    out = compute_volatility(frame, price=price, sessions=sessions)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_volatility")
