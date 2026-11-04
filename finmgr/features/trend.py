"""Trend: SMA and EMA at 20, 50 and 200 sessions, distance, slope and crosses.

    from finmgr.features.trend import read_trend

    panel = read_trend(as_of="2024-06-28", start="2024-01-01")

Columns, for each window n in 20, 50 and 200 and each average kind in `sma`, `ema`,
every window counted in venue sessions exactly as :mod:`finmgr.features.returns`
counts them:

    sma_{n}, ema_{n}              the moving average, in (adjusted) price units
    dist_sma_{n}, dist_ema_{n}    p[t] / ma[t] - 1: how far the price is above it
    slope_sma_{n}, slope_ema_{n}  ma[t] / ma[t-5] - 1: how fast it is rising
    trend_50_200                  +1 while sma_50 > sma_200, -1 otherwise
    golden_cross                  1 on the bar sma_50 closes above sma_200, else 0
    death_cross                   1 on the bar sma_50 closes below sma_200, else 0

**Compare the ratios, not the levels.** `sma_{n}` and `ema_{n}` are in the units
of the adjusted price, which a later split or dividend rescales — like `atr_14`.
Distance, slope and the cross flags are ratios of two adjusted prices and do not
move when the store learns of a later action; they are the model's columns.

**SMA** is the plain mean of the `n` prices ending at t. It needs the window's
full span of history and at least :data:`MIN_COVERAGE` of its prices, so one
missing bar does not blank 200 sessions of `sma_200`.

**EMA** uses `alpha = 2 / (n + 1)`, seeded with the SMA of the first `n` prices
and then `ema[t] = alpha * p[t] + (1 - alpha) * ema[t-1]`. A missing price
leaves the running value as it was (and carries it through that session). The recursion has
infinite memory, so :func:`read_trend` always starts it at the first stored bar,
whatever `start` is: an EMA dated t is the same number however the panel is cut.

**Slope** is the average's own 5-session return (:data:`SLOPE_SESSIONS`), so it
is scale-free and comparable across tickers.

**Crosses** are events on the ticker's stored bars: `golden_cross` is 1 on the
first bar on which the 50-session SMA is above the 200-session one after a bar on
which it was not. Both are NaN until both averages exist, and 0 on the first bar
they do — a regime that is already in place is not a cross. `trend_50_200` is
the regime itself.

**Prices.** `adj_close_eur` by default, the series the returns are on; pass
`price="adj_close"` for the listing-currency trend a chart would show.

**Strictly backward-looking.** Every value at t uses prices at t and earlier.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.data.benchmarks import read_total_return
from finmgr.features.returns import DEFAULT_PRICE, session_prices, ticker_sessions

#: Moving-average windows, in sessions.
MA_WINDOWS: tuple[int, ...] = (20, 50, 200)

#: The two kinds of average.
MA_KINDS: tuple[str, ...] = ("sma", "ema")

#: The span over which a slope is measured, in sessions.
SLOPE_SESSIONS = 5

#: The (fast, slow) SMA pair whose crossings are the golden and death crosses.
CROSS_PAIR: tuple[int, int] = (50, 200)

#: The fraction of an SMA window's prices that must be present for a value.
MIN_COVERAGE = 0.9


def ma_column(kind: str, window: int) -> str:
    """`sma_50` for ("sma", 50)."""
    return f"{kind}_{window}"


def dist_column(kind: str, window: int) -> str:
    """`dist_sma_50` for ("sma", 50)."""
    return f"dist_{kind}_{window}"


def slope_column(kind: str, window: int) -> str:
    """`slope_sma_50` for ("sma", 50)."""
    return f"slope_{kind}_{window}"


REGIME_COLUMN = f"trend_{CROSS_PAIR[0]}_{CROSS_PAIR[1]}"
GOLDEN_COLUMN = "golden_cross"
DEATH_COLUMN = "death_cross"

TREND_COLUMNS: tuple[str, ...] = (
    *(ma_column(k, w) for k in MA_KINDS for w in MA_WINDOWS),
    *(dist_column(k, w) for k in MA_KINDS for w in MA_WINDOWS),
    *(slope_column(k, w) for k in MA_KINDS for w in MA_WINDOWS),
    REGIME_COLUMN,
    GOLDEN_COLUMN,
    DEATH_COLUMN,
)

#: The longest lookback any column needs, in sessions (the EMA's seed aside).
MAX_LOOKBACK = max(MA_WINDOWS) + SLOPE_SESSIONS


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Mean of the `window` prices ending at each row; NaN until the window is full."""
    needed = math.ceil(MIN_COVERAGE * window)
    out = prices.rolling(window, min_periods=needed).mean()
    out.iloc[: window - 1] = np.nan
    return out


def ema(prices: pd.Series, window: int) -> pd.Series:
    """EMA with `alpha = 2 / (window + 1)`, seeded with the SMA of the first `window` prices.

    A NaN price leaves the running value unchanged and carries it on that row,
    so a missing session does not blank the slope five sessions later.
    """
    alpha = 2.0 / (window + 1)
    values = prices.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    seed: list[float] = []
    level = math.nan
    for i, x in enumerate(values):
        if math.isnan(x):
            out[i] = level  # NaN before the seed, the carried value after it
            continue
        if len(seed) < window:
            seed.append(x)
            if len(seed) == window:
                level = sum(seed) / window
                out[i] = level
            continue
        level = alpha * x + (1.0 - alpha) * level
        out[i] = level
    return pd.Series(out, index=prices.index)


def crosses(fast: pd.Series, slow: pd.Series) -> pd.DataFrame:
    """The regime (+1 / -1) and golden / death cross events of `fast` against `slow`.

    Rows are compared with the previous row on which both were defined, so a
    gap of NaN rows between them does not hide a cross.
    """
    defined = fast.notna() & slow.notna()
    above = (fast > slow).astype(float).where(defined)  # 1.0, 0.0 or NaN
    regime = 2.0 * above - 1.0
    before = above.ffill().shift(1)
    golden = ((above == 1) & (before == 0)).astype(float).where(defined)
    death = ((above == 0) & (before == 1)).astype(float).where(defined)
    return pd.DataFrame({REGIME_COLUMN: regime, GOLDEN_COLUMN: golden, DEATH_COLUMN: death})


def _one_ticker(frame: pd.DataFrame, price: str, grid: Sequence[date] | None) -> pd.DataFrame:
    prices = session_prices(frame, price, grid)
    averages = {ma_column("sma", w): sma(prices, w) for w in MA_WINDOWS} | {
        ma_column("ema", w): ema(prices, w) for w in MA_WINDOWS
    }
    out = dict(averages)
    for kind in MA_KINDS:
        for w in MA_WINDOWS:
            out[dist_column(kind, w)] = prices / averages[ma_column(kind, w)] - 1.0
    for kind in MA_KINDS:
        for w in MA_WINDOWS:
            ma = averages[ma_column(kind, w)]
            # shift by a positive count: the average five sessions *earlier*.
            out[slope_column(kind, w)] = ma / ma.shift(SLOPE_SESSIONS) - 1.0
    on_grid = pd.DataFrame(out, index=prices.index).loc[frame["date"]]
    # Crosses on the stored bars, so one that happens over a missing session is
    # flagged on the next bar rather than lost.
    fast, slow = (on_grid[ma_column("sma", w)] for w in CROSS_PAIR)
    return pd.concat([on_grid, crosses(fast, slow)], axis=1)


def compute_trend(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> pd.DataFrame:
    """Every :data:`TREND_COLUMNS` per ticker, backward only.

    Same contract as :func:`~finmgr.features.returns.compute_returns`. Returns
    `ticker, date, <price>` and :data:`TREND_COLUMNS`, sorted by `(ticker, date)`.
    """
    missing = {"ticker", "date", price} - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    base = prices[["ticker", "date", price]].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in ["ticker", "date", price, *TREND_COLUMNS]})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, grid).reset_index(drop=True)
        parts.append(pd.concat([frame.reset_index(drop=True), part], axis=1))
    return pd.concat(parts, ignore_index=True)


def read_trend(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_trend` on :func:`read_total_return`, on each venue's calendar.

    Nothing after `as_of` is read. The whole stored history up to `as_of` is
    read whatever `start` is, so the EMAs' recursion starts at the first bar and
    a value dated t does not depend on where the panel is cut; `start` only
    trims the output.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    frame = read_total_return(tickers, as_of=cutoff, settings=resolved, root=root)
    sessions = ticker_sessions(frame, resolved)
    out = compute_trend(frame, price=price, sessions=sessions)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_trend")
