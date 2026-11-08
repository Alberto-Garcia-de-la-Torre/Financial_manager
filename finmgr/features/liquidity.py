"""Volume and liquidity: euro volume, its 21-session average, a volume z-score, OBV, tradability.

    from finmgr.features.liquidity import read_liquidity

    panel = read_liquidity(as_of="2024-06-28", start="2024-01-01")
    tradable = panel[panel["tradable"]]

Columns, every window counted in venue sessions exactly as
:mod:`finmgr.features.returns` counts them:

    eur_volume      volume * close_eur: the euros that changed hands that session
    adv_eur_21      mean of eur_volume over the last 21 sessions (average daily value)
    volume_z_63     z-score of ln(eur_volume[t]) against the 63 sessions before t
    obv             on-balance volume, in euros: cumulative +-eur_volume by the close's direction
    obv_flow_21     (obv[t] - obv[t-21]) / (sum of eur_volume over those 21 sessions), in [-1, 1]
    tradable        adv_eur_21 >= min_adv_eur (EUR 5M by default); False when unknown

**Euro volume, not shares.** Share counts are not comparable across tickers (a
EUR 4 stock trades more shares than a EUR 400 one) and jump on a split, so every
column here is built on the traded *value*. `volume * close` is what changed
hands whichever way Yahoo has split-adjusted the two, since a split scales one
up and the other down by the same factor; `close_eur` puts every venue in euros
(and already turns pence and other sub-units into their currency).

**adv_eur_21** is the plain mean, not the median: a block-trade day is real
liquidity. It needs :data:`MIN_COVERAGE` of the 21 sessions, and never comes
before the 21st session.

**volume_z_63** is the day's surprise: `(ln ev[t] - mean) / std` with the mean
and std (ddof=1) of `ln ev` over t-63..t-1, so today does not dilute its own
baseline. Logs, because volume is heavy-tailed: a day at twice the usual volume
and one at half are equally far from normal. A zero-volume session has no log
and is NaN.

**obv** is Granville's on-balance volume with euros for shares: the day's euro
volume is added when the adjusted close rose, subtracted when it fell, and
ignored when it was flat or either close is missing. Its level depends on where
it started, so it is not a feature to compare across tickers or reads; the model
column is **obv_flow_21**, the net signed volume of the last 21 sessions as a
share of all their volume: +1 if every euro traded on an up day, -1 if every
one traded on a down day.

**tradable** is the filter: a stock is tradable at t when its average daily euro
volume over the last 21 sessions is at least `min_adv_eur`. Unknown liquidity
(the first 21 sessions of a listing, or too many missing ones) is *not*
tradable: a signal we cannot size against a volume is not one to act on. The
threshold is absolute, so it binds on a stock's own history — a recent IPO, a
pre-index-inclusion decade — rather than marking the bottom few of every day.

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

#: The average-daily-value window, in sessions.
ADV_WINDOW = 21

#: The baseline the volume z-score is measured against, in sessions before t.
Z_WINDOW = 63

#: The window of the normalized OBV flow, in sessions.
OBV_FLOW_WINDOW = 21

#: The tradability threshold: average daily euro volume over :data:`ADV_WINDOW`.
MIN_ADV_EUR = 5_000_000.0

#: The fraction of a window's observations that must be present for a value.
MIN_COVERAGE = 0.9

#: The price whose direction signs OBV: the listing-currency adjusted close, so
#: neither a split nor an FX move flips a day.
DEFAULT_PRICE = "adj_close"

#: What :func:`compute_liquidity` needs besides `ticker`, `date` and the price.
VOLUME_INPUTS: tuple[str, ...] = ("volume", "close_eur")

EUR_VOLUME_COLUMN = "eur_volume"
ADV_COLUMN = f"adv_eur_{ADV_WINDOW}"
VOLUME_Z_COLUMN = f"volume_z_{Z_WINDOW}"
OBV_COLUMN = "obv"
OBV_FLOW_COLUMN = f"obv_flow_{OBV_FLOW_WINDOW}"
TRADABLE_COLUMN = "tradable"

LIQUIDITY_COLUMNS: tuple[str, ...] = (
    EUR_VOLUME_COLUMN,
    ADV_COLUMN,
    VOLUME_Z_COLUMN,
    OBV_COLUMN,
    OBV_FLOW_COLUMN,
    TRADABLE_COLUMN,
)


def _needed(window: int) -> int:
    return math.ceil(MIN_COVERAGE * window)


def _warm(values: pd.Series, span: int) -> pd.Series:
    """NaN for the first `span` positions: the window has not got its history yet."""
    out = values.copy()
    out.iloc[:span] = np.nan
    return out


def average_daily_value(eur_volume: pd.Series, window: int = ADV_WINDOW) -> pd.Series:
    """Mean euro volume of the `window` sessions ending at each row."""
    mean = eur_volume.rolling(window, min_periods=_needed(window)).mean()
    return _warm(mean, window - 1)


def volume_zscore(eur_volume: pd.Series, window: int = Z_WINDOW) -> pd.Series:
    """`ln ev[t]` in standard deviations from its mean over the `window` sessions before t."""
    logs = np.log(eur_volume.where(eur_volume > 0))
    prior = logs.shift(1).rolling(window, min_periods=_needed(window))
    std = prior.std(ddof=1)
    z = (logs - prior.mean()) / std.where(std > 0)
    return _warm(z, window)


def on_balance_volume(prices: pd.Series, eur_volume: pd.Series) -> pd.Series:
    """Cumulative euro volume signed by the close's direction; NaN where the volume is."""
    direction = np.sign(prices - prices.shift(1)).fillna(0.0)
    signed = (direction * eur_volume).fillna(0.0)
    return signed.cumsum().where(eur_volume.notna())


def obv_flow(prices: pd.Series, eur_volume: pd.Series, window: int = OBV_FLOW_WINDOW) -> pd.Series:
    """Net signed euro volume over `window` sessions over their total, in [-1, 1]."""
    direction = np.sign(prices - prices.shift(1))
    signed = (direction * eur_volume).rolling(window, min_periods=_needed(window)).sum()
    total = eur_volume.where(direction.notna()).rolling(window, min_periods=_needed(window)).sum()
    return _warm(signed / total.where(total > 0), window)


def tradable_flag(adv: pd.Series, min_adv_eur: float = MIN_ADV_EUR) -> pd.Series:
    """True where `adv >= min_adv_eur`; an unknown (NaN) average is not tradable."""
    return (adv >= min_adv_eur).fillna(False).astype(bool)


def _one_ticker(
    frame: pd.DataFrame, price: str, grid: Sequence[date] | None, min_adv_eur: float
) -> pd.DataFrame:
    prices = session_prices(frame, price, grid)
    traded = frame["volume"].to_numpy(dtype=float) * frame["close_eur"].to_numpy(dtype=float)
    ev = pd.Series(traded, index=frame["date"].to_numpy()).reindex(prices.index)
    # A negative traded value is bad data; zero is a real (if odd) session.
    ev = ev.where(ev >= 0)
    adv = average_daily_value(ev)
    out = {
        EUR_VOLUME_COLUMN: ev,
        ADV_COLUMN: adv,
        VOLUME_Z_COLUMN: volume_zscore(ev),
        OBV_COLUMN: on_balance_volume(prices, ev),
        OBV_FLOW_COLUMN: obv_flow(prices, ev),
        TRADABLE_COLUMN: tradable_flag(adv, min_adv_eur),
    }
    return pd.DataFrame(out, index=prices.index).loc[frame["date"]]


def compute_liquidity(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    sessions: Mapping[str, Sequence[date]] | None = None,
    min_adv_eur: float = MIN_ADV_EUR,
) -> pd.DataFrame:
    """Every :data:`LIQUIDITY_COLUMNS` per ticker, backward only.

    Same contract as :func:`~finmgr.features.returns.compute_returns`, and
    `prices` also needs :data:`VOLUME_INPUTS` (the raw `volume` and the bar's
    `close_eur`). Returns `ticker, date, <price>` and :data:`LIQUIDITY_COLUMNS`,
    sorted by `(ticker, date)`.
    """
    missing = {"ticker", "date", price, *VOLUME_INPUTS} - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    if not min_adv_eur > 0:
        raise ValueError(f"min_adv_eur must be positive, got {min_adv_eur}")
    keep = ["ticker", "date", price, *(c for c in VOLUME_INPUTS if c != price)]
    base = prices[keep].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        out = pd.DataFrame({c: [] for c in ["ticker", "date", price, *LIQUIDITY_COLUMNS]})
        return out.astype({TRADABLE_COLUMN: bool})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, grid, min_adv_eur).reset_index(drop=True)
        head = frame[["ticker", "date", price]].reset_index(drop=True)
        parts.append(pd.concat([head, part], axis=1))
    return pd.concat(parts, ignore_index=True)


def read_liquidity(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    min_adv_eur: float = MIN_ADV_EUR,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_liquidity` on the store, on each venue's calendar.

    Adjusted closes come from :func:`read_total_return`, volume and `close_eur`
    from the bars. Nothing after `as_of` is read. The whole stored history is
    read whatever `start` is, so `obv` dated t does not depend on where the
    panel is cut; `start` only trims the output.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    closes = read_total_return(tickers, as_of=cutoff, settings=resolved, root=root)
    base = Path(root) if root is not None else resolved.data_dir
    bars = store.read_bars(tickers, as_of=cutoff, settings=resolved, root=base / "bars" / "daily")
    frame = closes.drop(columns=["close_eur"], errors="ignore").merge(
        bars[["ticker", "date", "volume", "close_eur"]], on=["ticker", "date"], how="left"
    )
    sessions = ticker_sessions(frame, resolved)
    out = compute_liquidity(frame, price=price, sessions=sessions, min_adv_eur=min_adv_eur)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_liquidity")


def thinnest(panel: pd.DataFrame, day: date, n: int = 10) -> pd.DataFrame:
    """The `n` names with the lowest `adv_eur_21` on `day`, thinnest first, with their flag."""
    rows = panel[panel["date"] == day]
    columns = ["ticker", ADV_COLUMN, TRADABLE_COLUMN]
    return rows.sort_values(ADV_COLUMN, na_position="first")[columns].head(n).reset_index(drop=True)
