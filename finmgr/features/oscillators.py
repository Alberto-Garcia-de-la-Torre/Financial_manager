"""Oscillators, written by hand: RSI(14), MACD(12,26,9), Stochastic(14,3), Bollinger %B(20,2).

    from finmgr.features.oscillators import read_oscillators

    panel = read_oscillators(as_of="2024-06-28", start="2024-01-01")

Columns, every window counted in venue sessions exactly as
:mod:`finmgr.features.returns` counts them:

    rsi_14            Wilder's relative strength index, 0..100
    macd              EMA(12) - EMA(26) of the close, in price units
    macd_signal       EMA(9) of `macd`, in price units
    macd_hist         macd - macd_signal, in price units
    macd_pct, macd_signal_pct, macd_hist_pct   the three above divided by the close
    stoch_k           100 * (C - lowest low 14) / (highest high 14 - lowest low 14), 0..100
    stoch_d           3-session SMA of `stoch_k`
    bb_pctb           (C - lower) / (upper - lower), bands = SMA(20) -/+ 2 * stdev(20)

What each one *means*, since the model will lean on them:

**RSI(14)** is the share of the recent movement that was upward, on a 0-100
scale. With `U = max(dC, 0)` and `D = max(-dC, 0)` for the daily close changes,
both are averaged with Wilder's smoothing — seeded with the plain mean of the
first 14 changes, then `avg[t] = (13 * avg[t-1] + x[t]) / 14` — and
`RSI = 100 * avgU / (avgU + avgD)` (the same as `100 - 100 / (1 + RS)`,
`RS = avgU / avgD`). 70 and above is "overbought", 30 and below "oversold". All
gains gives 100, all losses 0; a dead-flat stretch (no movement at all) is 50.
The first value is on the 15th price (14 changes). This is the textbook
definition and matches TA-Lib's `RSI` to 1e-9 (see `tests/test_oscillators.py`).

TA-Lib's `MACD` seeds its fast EMA later than ours (so both lines start on the
same bar); the two agree to 1e-4 once that seed has decayed, after ~120 bars.

**MACD(12,26,9)** is the gap between a fast and a slow EMA of the close: positive
when the last few weeks are above the last few months, i.e. short-term momentum.
The signal line is a 9-session EMA of that gap, and the histogram is the gap
minus its signal — MACD's own acceleration; a sign change of the histogram is
the classic "MACD cross". The EMAs are :func:`finmgr.features.trend.ema`
(alpha = 2/(n+1), seeded with an SMA), and the signal is seeded with the SMA of
the first 9 MACD values, so the first signal is on the 34th price. MACD is in
price units, so the `_pct` columns divide by the close to make it comparable
across tickers and through splits; those are the model's columns.

**Stochastic(14,3)** places today's close inside the 14-session high-low range:
100 at the top, 0 at the bottom. `stoch_d` is its 3-session SMA (the "fast"
stochastic; TA-Lib's `STOCHF(14, 3)`). A range of zero gives 50. It needs highs
and lows, back-adjusted with the close's own factor exactly as ATR does.

**Bollinger %B(20,2)** places the close inside bands two (population, ddof=0, as
Bollinger defines them) standard deviations either side of the 20-session SMA:
0 at the lower band, 1 at the upper, 0.5 on the average, and outside [0, 1]
beyond the bands. It is `0.5 + z / 4` where `z` is the close's 20-session
z-score. Zero dispersion gives 0.5.

**Prices.** These are chart indicators, so they are computed on the
listing-currency adjusted bar (`adj_close`, with highs and lows adjusted by
`adj_close / close`), not the euro series: an FX move is not a move in the
stock's own chart. Pass `price="adj_close_eur"` for the euro version of the
close-only columns (the stochastic always uses the listing bar).

**Missing sessions** are NaN on the session grid. A missing close removes the
two changes touching it. Wilder's averages skip those changes (RSI is NaN on
the bar after the gap, exactly as `atr_14` is) and resume from their last
value; the MACD's EMAs carry theirs through. Rolling windows (stochastic, Bollinger) still
produce a value with :data:`MIN_COVERAGE` of their bars present, never before
the window's full span exists.

**Strictly backward-looking.** Every value at t uses bars at t and earlier.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit, store
from finmgr.data.benchmarks import read_total_return
from finmgr.features.returns import session_prices, ticker_sessions
from finmgr.features.trend import ema
from finmgr.features.volatility import wilder_atr

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
STOCH_PERIOD, STOCH_SMOOTH = 14, 3
BB_PERIOD, BB_WIDTH = 20, 2.0

#: The fraction of a rolling window's bars that must be present for a value.
MIN_COVERAGE = 0.9

#: The default close: the listing-currency, back-adjusted one.
DEFAULT_PRICE = "adj_close"

#: What the stochastic needs on top of the close.
BAR_INPUTS: tuple[str, ...] = ("high", "low", "close", "adj_close")

RSI_COLUMN = f"rsi_{RSI_PERIOD}"
MACD_COLUMNS: tuple[str, ...] = ("macd", "macd_signal", "macd_hist")
MACD_PCT_COLUMNS: tuple[str, ...] = tuple(f"{c}_pct" for c in MACD_COLUMNS)
STOCH_COLUMNS: tuple[str, ...] = ("stoch_k", "stoch_d")
PCTB_COLUMN = "bb_pctb"

OSCILLATOR_COLUMNS: tuple[str, ...] = (
    RSI_COLUMN,
    *MACD_COLUMNS,
    *MACD_PCT_COLUMNS,
    *STOCH_COLUMNS,
    PCTB_COLUMN,
)

#: The longest lookback any column needs, in sessions.
MAX_LOOKBACK = MACD_SLOW + MACD_SIGNAL


def _needed(window: int) -> int:
    return math.ceil(MIN_COVERAGE * window)


def _rolling(series: pd.Series, window: int) -> pd.core.window.Rolling:
    return series.rolling(window, min_periods=_needed(window))


def _full_span(out: pd.Series, window: int) -> pd.Series:
    out.iloc[: window - 1] = np.nan
    return out


def rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI on 0..100; NaN until `period` changes have been seen."""
    change = prices - prices.shift(1)
    gain = wilder_atr(change.clip(lower=0.0), period)
    loss = wilder_atr((-change).clip(lower=0.0), period)
    total = gain + loss
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 100.0 * gain / total
    return out.where(total != 0, 50.0).where(total.notna())


def macd(
    prices: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL
) -> pd.DataFrame:
    """`macd`, `macd_signal` and `macd_hist` of `prices`, in price units."""
    line = ema(prices, fast) - ema(prices, slow)
    # Seeded on the first `signal` *defined* MACD values, so the line's own
    # NaN warm-up does not count against the signal's seed.
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = STOCH_PERIOD,
    smooth: int = STOCH_SMOOTH,
) -> pd.DataFrame:
    """Fast stochastic: `stoch_k` over `period` sessions and `stoch_d`, its `smooth`-SMA."""
    hh = _full_span(_rolling(high, period).max(), period)
    ll = _full_span(_rolling(low, period).min(), period)
    span = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        k = 100.0 * (close - ll) / span
    k = k.where(span != 0, 50.0).where(span.notna() & close.notna())
    d = _full_span(k.rolling(smooth, min_periods=smooth).mean(), smooth)
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def percent_b(prices: pd.Series, period: int = BB_PERIOD, width: float = BB_WIDTH) -> pd.Series:
    """Bollinger %B: 0 on the lower band, 1 on the upper, 0.5 on the SMA."""
    mid = _full_span(_rolling(prices, period).mean(), period)
    sd = _rolling(prices, period).std(ddof=0)
    lower = mid - width * sd
    band = 2.0 * width * sd
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (prices - lower) / band
    return out.where(band != 0, 0.5).where(band.notna() & prices.notna())


def _one_ticker(
    frame: pd.DataFrame, price: str, grid: Sequence[date] | None, with_bars: bool
) -> pd.DataFrame:
    prices = session_prices(frame, price, grid)
    out = {RSI_COLUMN: rsi(prices)}
    lines = macd(prices)
    for col in MACD_COLUMNS:
        out[col] = lines[col]
        out[f"{col}_pct"] = lines[col] / prices
    if with_bars:
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
        stoch = stochastic(bars["high"], bars["low"], bars["close"])
        # high * (adj_close / close) and adj_close round differently, so a close
        # at the day's high can land at 100 + 1e-14. The store's quality checks
        # already guarantee low <= close <= high; this only removes the noise.
        out.update({c: stoch[c].clip(0.0, 100.0) for c in STOCH_COLUMNS})
    else:
        out.update({c: pd.Series(np.nan, index=prices.index) for c in STOCH_COLUMNS})
    out[PCTB_COLUMN] = percent_b(prices)
    return pd.DataFrame(out, index=prices.index)[list(OSCILLATOR_COLUMNS)].loc[frame["date"]]


def compute_oscillators(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> pd.DataFrame:
    """Every :data:`OSCILLATOR_COLUMNS` per ticker, backward only.

    Same contract as :func:`~finmgr.features.returns.compute_returns`. The
    stochastic also needs :data:`BAR_INPUTS`; without them its two columns are
    NaN. Returns `ticker, date, <price>` and :data:`OSCILLATOR_COLUMNS`, sorted
    by `(ticker, date)`.
    """
    missing = {"ticker", "date", price} - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    with_bars = set(BAR_INPUTS) <= set(prices.columns)
    keep = ["ticker", "date", price, *(c for c in BAR_INPUTS if with_bars and c != price)]
    base = prices[keep].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in ["ticker", "date", price, *OSCILLATOR_COLUMNS]})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, grid, with_bars).reset_index(drop=True)
        head = frame[["ticker", "date", price]].reset_index(drop=True)
        parts.append(pd.concat([head, part], axis=1))
    return pd.concat(parts, ignore_index=True)


def read_oscillators(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_oscillators` on the store, on each venue's calendar.

    Closes come from :func:`read_total_return`, highs and lows from the bars.
    Nothing after `as_of` is read. As in :func:`~finmgr.features.trend.read_trend`,
    the whole stored history is read whatever `start` is, so the recursive
    columns (RSI, MACD) dated t do not depend on where the panel is cut; `start`
    only trims the output.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    closes = read_total_return(tickers, as_of=cutoff, settings=resolved, root=root)
    base = Path(root) if root is not None else resolved.data_dir
    bars = store.read_bars(tickers, as_of=cutoff, settings=resolved, root=base / "bars" / "daily")
    frame = closes.merge(bars[["ticker", "date", "high", "low"]], on=["ticker", "date"], how="left")
    sessions = ticker_sessions(frame, resolved)
    out = compute_oscillators(frame, price=price, sessions=sessions)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_oscillators")
