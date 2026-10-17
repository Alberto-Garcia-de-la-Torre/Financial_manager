"""Back-adjusted close, built from raw close, splits and dividends.

    from finmgr.data.adjust import read_adjusted

    read_adjusted("SAN.MC", as_of="2024-06-28")   # ticker, date, close, split_factor,
                                                  # dividend_factor, adj_close

The store keeps raw closes (day 10) and the actions that happened to the share
(day 15) as separate, unmodified facts. This module is the one place they are
combined. A back-adjusted close answers "what would one share bought on day
*t* be worth today, if every payout had been reinvested in the share" — it is
the series returns are computed from, and the latest session is always left
exactly at its raw close; every earlier price is scaled down.

**Dividends.** On an ex-date the price drops by roughly the payout, and that
drop is not a loss. Every close *before* the ex-date is multiplied by

    1 - dividend / close of the last session before the ex-date

and the factors of successive ex-dates multiply. That is the CRSP convention,
and the one Yahoo's `Adj Close` uses; day 16 checks we agree with it to about
one part in a million wherever both have the same inputs. An ex-date that
falls on a day with no session uses the last session before it, and so applies
to the same closes. A dividend before the first stored bar cannot move any
stored close and is ignored, and so is one whose ex-date is after the last
stored session: every stored close is still cum-dividend, and the series is
anchored to the latest raw close.

**Splits are applied only where the bars show them.** Yahoo's daily endpoint
hands back closes already divided by every split up to the day of the request
(see :mod:`finmgr.data.fetch`), so for Yahoo bars the split factor is 1
everywhere and applying the stored 4:1 again would quarter Apple's history.
But a store that grows incrementally will one day hold bars fetched *before* a
split next to bars fetched after it, and then the split does show — a 4:1 is a
75% overnight drop. So each split is checked against the bars on its ex-date:
if the close-to-close move is nearer the split ratio than it is to no move at
all, the earlier closes are divided by the ratio; otherwise the bars already
reflect it. That test only means something for a split bigger than an ordinary
day's move: a 1.5% scrip issue (Santander's 2011-01-17 is one) on a day the
price also fell 1% looks exactly like a visible split. So a split within
:data:`AMBIGUOUS_SPLIT` of 1:1 is never treated as visible — the bars cannot
tell, and Yahoo's convention is that it is already applied. The choice is
reported per split by :func:`split_visibility`.

Dividends are divided by the *split-adjusted* close, because Yahoo states every
historical dividend in today's share count — the same money the
split-adjusted closes are in.

Everything here is pure: frames in, frames out, no network and no disk except
:func:`read_adjusted`, which only reads.

**Point in time.** Back-adjustment is look-ahead by construction: a dividend
paid in March rescales every close before it. Computed over the whole store,
the adjusted close of a January session would change the day a March dividend
went ex — and a backtest standing in January would be using it.
:func:`read_adjusted` therefore takes a required `as_of` and builds the factors
from the bars and actions dated on or before it only, so the series is the one
that could actually have been computed that evening (:mod:`finmgr.data.pit`).
Returns are unaffected by where the series is anchored; the price *level* is
not, which is why the anchor moves with `as_of`. One look-ahead remains that
no `as_of` can undo: Yahoo's raw closes arrive already divided by every split
up to the fetch date (see above), so a price *level* from before a split is in
later shares. Use returns, not levels.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings
from finmgr.data import pit
from finmgr.data.actions import DIVIDEND, SPLIT, read_actions
from finmgr.data.store import read_bars

#: A split whose ratio is within this of 1:1, in log terms (about 0.82 to
#: 1.22), cannot be told from an ordinary day's move and is assumed to be
#: already reflected in the bars. A 5:4 split is just outside it.
AMBIGUOUS_SPLIT = 0.2

#: The columns :func:`adjust` returns, in order.
ADJUSTED_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "close",
    "split_factor",
    "dividend_factor",
    "adj_close",
)


# ---------------------------------------------------------------------------
# One ticker, as plain series
# ---------------------------------------------------------------------------


def _closes(closes: pd.Series) -> pd.Series:
    """A float close series indexed by session date, sorted, NaNs dropped."""
    out = pd.Series(closes, dtype="float64").dropna()
    out.index = pd.Index([pd.Timestamp(d).date() for d in out.index], name="date")
    if out.index.has_duplicates:
        raise ValueError("close series repeats a session date")
    return out.sort_index()


def _events(values: pd.Series | None) -> pd.Series:
    """Action values indexed by ex-date, sorted; empty when there are none."""
    if values is None or len(values) == 0:
        return pd.Series([], dtype="float64", index=pd.Index([], name="date"))
    out = pd.Series(values, dtype="float64")
    out.index = pd.Index([pd.Timestamp(d).date() for d in out.index], name="date")
    return out.groupby(level=0).sum() if out.index.has_duplicates else out.sort_index()


def split_visibility(closes: pd.Series, splits: pd.Series | None) -> pd.DataFrame:
    """For each split, whether the raw closes still show it.

    Returns one row per split with its `ratio`, the close-to-close `move`
    across the ex-date (NaN if the bars do not span it) and `in_bars`: True
    when the move looks like the split (nearer `1 / ratio` than `1` in log
    terms), meaning the earlier closes are in pre-split money and need
    dividing. A split the bars cannot see either side of counts as already
    reflected — there is nothing to divide — and so does one flagged
    `ambiguous`, too close to 1:1 for the bars to tell (see the module
    docstring).
    """
    prices = _closes(closes)
    events = _events(splits)
    rows = []
    for ex_date, ratio in events.items():
        before = prices[prices.index < ex_date]
        on_or_after = prices[prices.index >= ex_date]
        move = np.nan
        visible = False
        ambiguous = abs(np.log(ratio)) < AMBIGUOUS_SPLIT
        if not before.empty and not on_or_after.empty and not ambiguous:
            move = float(on_or_after.iloc[0] / before.iloc[-1])
            visible = abs(np.log(move * ratio)) < abs(np.log(move))
        elif not before.empty and not on_or_after.empty:
            move = float(on_or_after.iloc[0] / before.iloc[-1])
        rows.append(
            {
                "date": ex_date,
                "ratio": float(ratio),
                "move": move,
                "ambiguous": ambiguous,
                "in_bars": visible,
            }
        )
    return pd.DataFrame(rows, columns=["date", "ratio", "move", "ambiguous", "in_bars"])


def dividend_factors(
    closes: pd.Series, dividends: pd.Series | None, splits: pd.Series | None = None
) -> pd.DataFrame:
    """One row per dividend: the close it was divided by and the factor.

    `reference_date` is the last session before the ex-date and
    `reference_close` its split-adjusted close; `factor` is
    `1 - dividend / reference_close`. A dividend with no earlier session, or
    with its ex-date after the last session, has NaN for all three and moves
    nothing. A dividend at or above its reference
    close cannot be right — the share would be worth nothing after it — and
    raises rather than produce a negative price.
    """
    prices = _closes(closes) * split_factor(closes, splits)
    events = _events(dividends)
    rows = []
    for ex_date, amount in events.items():
        before = prices[prices.index < ex_date]
        if before.empty or ex_date > prices.index[-1]:
            rows.append(
                {
                    "date": ex_date,
                    "dividend": float(amount),
                    "reference_date": None,
                    "reference_close": np.nan,
                    "factor": np.nan,
                }
            )
            continue
        reference = float(before.iloc[-1])
        factor = 1.0 - float(amount) / reference
        if factor <= 0:
            raise ValueError(
                f"dividend {amount:g} on {ex_date} is not below the close it applies to "
                f"({reference:g} on {before.index[-1]})"
            )
        rows.append(
            {
                "date": ex_date,
                "dividend": float(amount),
                "reference_date": before.index[-1],
                "reference_close": reference,
                "factor": factor,
            }
        )
    return pd.DataFrame(
        rows, columns=["date", "dividend", "reference_date", "reference_close", "factor"]
    )


def _backward_product(index: pd.Index, events: pd.Series) -> pd.Series:
    """For every date, the product of the event factors strictly after it."""
    out = pd.Series(1.0, index=index)
    for ex_date, factor in events.items():
        out[out.index < ex_date] *= factor
    return out


def split_factor(closes: pd.Series, splits: pd.Series | None) -> pd.Series:
    """Per session, what to multiply the raw close by to undo visible splits."""
    prices = _closes(closes)
    seen = split_visibility(prices, splits)
    seen = seen.loc[seen["in_bars"].astype(bool)]
    return _backward_product(prices.index, pd.Series(1.0 / seen["ratio"].to_numpy(), seen["date"]))


def adjust_series(
    closes: pd.Series, dividends: pd.Series | None = None, splits: pd.Series | None = None
) -> pd.DataFrame:
    """Back-adjust one ticker's closes. Indexed by date; no ticker column.

    `closes` is indexed by session date; `dividends` and `splits` by ex-date,
    with the values the actions store holds (cash per share in the listing
    currency; new shares per old). Returns `close, split_factor,
    dividend_factor, adj_close`, where `adj_close` is the product of the
    other three.
    """
    prices = _closes(closes)
    splits_f = split_factor(prices, splits)
    paid = dividend_factors(prices, dividends, splits).dropna(subset=["factor"])
    dividend_f = _backward_product(prices.index, pd.Series(paid["factor"].to_numpy(), paid["date"]))
    return pd.DataFrame(
        {
            "close": prices,
            "split_factor": splits_f,
            "dividend_factor": dividend_f,
            "adj_close": prices * splits_f * dividend_f,
        }
    )


# ---------------------------------------------------------------------------
# Store frames
# ---------------------------------------------------------------------------


def adjust(bars: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust a bar frame using an action frame, ticker by ticker.

    Both frames are in the store's schemas (:func:`~finmgr.data.store.read_bars`
    and :func:`~finmgr.data.actions.read_actions`). Returns
    :data:`ADJUSTED_COLUMNS`, sorted by `(ticker, date)`.
    """
    pieces = []
    for ticker, group in bars.groupby("ticker", sort=True):
        mine = actions[actions["ticker"] == ticker]
        frame = adjust_series(
            group.set_index("date")["close"],
            mine.loc[mine["action"] == DIVIDEND].set_index("date")["value"],
            mine.loc[mine["action"] == SPLIT].set_index("date")["value"],
        )
        pieces.append(frame.reset_index().assign(ticker=ticker))
    if not pieces:
        return pd.DataFrame({column: [] for column in ADJUSTED_COLUMNS})
    out = pd.concat(pieces, ignore_index=True)
    return out[list(ADJUSTED_COLUMNS)]


def read_adjusted(
    tickers: Iterable[str] | str | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    as_of: pit.AsOf,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Read bars and actions from the store and back-adjust them.

    The factors are computed over the whole history *up to `as_of`* and only
    then cut to `start`/`end`, so a window's adjusted closes are the numbers
    the full series had on those dates as of that evening — and nothing dated
    after `as_of`, bar or action, can reach them. `root` is the data directory
    (the parent of `bars/` and `actions/`), for tests.
    """
    cutoff = pit.resolve_as_of(as_of)
    first = pd.Timestamp(start).date() if start is not None else None
    last = pit.bound_end(pd.Timestamp(end).date() if end is not None else None, cutoff, start=first)
    base = None if root is None else Path(root)
    bars = read_bars(
        tickers,
        as_of=cutoff,
        settings=settings,
        root=None if base is None else base / "bars" / "daily",
    )
    actions = read_actions(
        tickers if tickers is not None else None,
        as_of=cutoff,
        settings=settings,
        root=None if base is None else base / "actions",
    )
    out = adjust(bars, actions)
    if first is not None:
        out = out[out["date"] >= first]
    out = out[out["date"] <= last]
    return pit.guard(out.reset_index(drop=True), cutoff, what="read_adjusted")
