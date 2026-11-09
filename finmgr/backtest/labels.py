"""Labels: what the model is asked to predict, the only forward-looking columns in the project.

    from finmgr.backtest.labels import read_labels

    labels = read_labels(as_of="2026-10-15")                    # all 100, both horizons
    labels[labels["date"] == date(2026, 9, 1)].sort_values("fwd_xs_21d")

For each horizon `N` in :data:`LABEL_HORIZONS` — 5 and 21 sessions, a week and
a month — three columns on the row dated t:

    fwd_ret_{N}d   p[t+N] / p[t] - 1      the return realized from t+1 through t+N
    fwd_xs_{N}d    fwd_ret - that date's equal-weighted universe mean of fwd_ret
    fwd_top_{N}d   1.0 if fwd_ret is in the date's top quintile, else 0.0

**What "from t+1 through t+N" means.** The row dated t is what is known at the
close of t: features dated t are built from prices up to and including t
(day 23). The label is the return earned by buying at that close and holding N
sessions, i.e. the compounded daily returns of sessions t+1, ..., t+N. Day t's
own return is a *feature* (it ends at t) and is never part of the label.
`tests/test_labels.py` checks exactly this against a loop that compounds the
daily returns by hand.

**Explicit `shift(-N)`.** `fwd_ret_{N}d` is the backward `ret_{N}d` of day 23 —
`p[t] / p[t-N] - 1` — moved N sessions *earlier*: `ret_{N}d.shift(-N)`. That is
the one negative shift in the code base, and it lives here, in a module whose
every column is excluded from the feature panel. A label must never become a
model input.

**Sessions, not rows.** As for returns, each ticker's prices are first laid on
its venue's sessions (day 17), so a missing bar does not stretch a 5-session
horizon to six: a label whose end point, or start point, is a missing session
is NaN. The last N sessions of each ticker have no future yet, and stay NaN
until it arrives; nothing is filled. A row the calendar did not expect but the
store holds is still a session, exactly as in :mod:`finmgr.features.returns`.

**The cross-section.** `fwd_xs` and `fwd_top` are computed per date over the
names that have a forward return that date (and, with `mask=`, where that
boolean column is True; the others get NaN in those two columns). A date with
fewer than `min_names` such names (default
:data:`~finmgr.features.crosssection.MIN_NAMES`) gets NaN in both. The mean is
equal-weighted, so the excess returns of a date sum to zero. The top quintile
is the names whose rank `r` (1 = lowest, ties averaged) satisfies `r / n > 0.8`:
20 of 100, 2 of 10. Ranking the excess return would give the same label.

**Point in time.** :func:`read_labels` reads nothing after `as_of`, so the
labels of the last N sessions before `as_of` are NaN: they are not yet known,
and a backtest run as of that date must not see them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.data.benchmarks import read_total_return
from finmgr.features.crosssection import MIN_NAMES
from finmgr.features.returns import DEFAULT_PRICE, session_prices, ticker_sessions

#: Label horizons in sessions: a week and a month.
LABEL_HORIZONS: tuple[int, ...] = (5, 21)

#: The share of a date's names the top-quintile label marks.
TOP_SHARE = 0.2


def fwd_column(horizon: int) -> str:
    """`fwd_ret_5d` for 5."""
    return f"fwd_ret_{horizon}d"


def excess_column(horizon: int) -> str:
    """`fwd_xs_5d` for 5."""
    return f"fwd_xs_{horizon}d"


def top_column(horizon: int) -> str:
    """`fwd_top_5d` for 5."""
    return f"fwd_top_{horizon}d"


def label_columns(horizons: Sequence[int] = LABEL_HORIZONS) -> list[str]:
    """Every label column for `horizons`, grouped by kind."""
    return (
        [fwd_column(h) for h in horizons]
        + [excess_column(h) for h in horizons]
        + [top_column(h) for h in horizons]
    )


def _check_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(h) for h in horizons)
    if not out or any(h < 1 for h in out):
        raise ValueError(f"horizons must be positive session counts, got {list(horizons)}")
    return out


def _one_ticker(
    frame: pd.DataFrame, price: str, horizons: tuple[int, ...], grid: Sequence[date] | None
) -> pd.DataFrame:
    """Forward returns for one ticker's rows, sorted by date, optionally on a session grid."""
    prices = session_prices(frame, price, grid)
    out = {}
    for h in horizons:
        backward = prices / prices.shift(h) - 1.0  # day 23's ret_{h}d: p[t] / p[t-h] - 1
        # shift(-h) moves each value h sessions *earlier*: the row for t gets
        # the return that ends at t+h, i.e. p[t+h] / p[t] - 1.
        out[fwd_column(h)] = backward.shift(-h)
    result = pd.DataFrame(out, index=prices.index)
    return result.loc[frame["date"]]


def _cross_section(
    labels: pd.DataFrame, horizons: tuple[int, ...], mask: str | None, min_names: int
) -> pd.DataFrame:
    dates = labels["date"]
    for h in horizons:
        fwd = labels[fwd_column(h)]
        if mask is not None:
            fwd = fwd.where(labels[mask].fillna(False).astype(bool))
        grouped = fwd.groupby(dates, sort=False)
        n = grouped.transform("count")
        enough = n >= min_names
        excess = fwd - grouped.transform("mean")
        rank = grouped.rank(method="average")
        top = (rank / n > 1.0 - TOP_SHARE).astype(float).where(fwd.notna())
        labels[excess_column(h)] = excess.where(enough)
        labels[top_column(h)] = top.where(enough)
    return labels


def compute_labels(
    prices: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    horizons: Sequence[int] = LABEL_HORIZONS,
    sessions: Mapping[str, Sequence[date]] | None = None,
    mask: str | None = None,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Forward, excess and top-quintile labels over `horizons` sessions.

    `prices` is long: one row per `(ticker, date)` with a `price` column (and
    the boolean `mask` column, if one is named). `sessions` maps a ticker to
    its venue's sessions, as in :func:`~finmgr.features.returns.compute_returns`;
    without it a ticker's own rows are its sessions. Returns `ticker, date`,
    `price`, the `mask` column if any, and :func:`label_columns`, one row per
    input row, sorted by `(ticker, date)`.
    """
    horizons = _check_horizons(horizons)
    keep = ["ticker", "date", price] + ([mask] if mask is not None else [])
    missing = set(keep) - set(prices.columns)
    if missing:
        raise KeyError(f"prices is missing column(s): {', '.join(sorted(missing))}")
    base = prices[keep].sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in [*keep, *label_columns(horizons)]})

    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, price, horizons, grid).reset_index(drop=True)
        parts.append(pd.concat([frame.reset_index(drop=True), part], axis=1))
    labels = pd.concat(parts, ignore_index=True)
    labels = _cross_section(labels, horizons, mask, min_names)
    return labels[[*keep, *label_columns(horizons)]]


def read_labels(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    price: str = DEFAULT_PRICE,
    horizons: Sequence[int] = LABEL_HORIZONS,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_labels` on :func:`read_total_return`, on each venue's calendar.

    Nothing after `as_of` is read, so the last `N` sessions before it have no
    `N`-session label. `start` trims the output (labels need no history before
    their own date, only a little slack for the trim). The cross-section is
    whatever `tickers` names — all of the universe by default — so pass the
    full universe when the excess and top-quintile columns matter.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    horizons = _check_horizons(horizons)
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    read_from = None if first is None else first - timedelta(days=14)
    prices = read_total_return(tickers, start=read_from, as_of=cutoff, settings=resolved, root=root)
    sessions = ticker_sessions(prices, resolved)
    out = compute_labels(prices, price=price, horizons=horizons, sessions=sessions)
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_labels")
