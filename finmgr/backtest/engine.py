"""Backtest engine v1: top-K by score, equal weight, hold N sessions, roll forward.

    from finmgr.backtest.engine import run_backtest
    from finmgr.data.benchmarks import read_total_return

    prices = read_total_return(as_of="2026-10-15", start="2020-01-01")
    result = run_backtest(scores, prices, top_k=10, hold=21)
    result.equity        # date, equity_eur, cash_eur, n_names, rebalance
    result.positions     # date, ticker, shares, price_eur, value_eur, weight
    result.trades        # date, ticker, side, shares, price_eur, value_eur

**Inputs.** `scores` is long — `ticker, date, score` — one score per ticker per
day, higher is better; NaN means "no opinion" and the name cannot be picked
that day. `prices` is long — `ticker, date` and a EUR price column, by default
day 23's `adj_close_eur` total-return close — so the equity curve is in EUR and
dividends are reinvested in the name that paid them.

**The rule.** The date axis is every date `prices` holds from the first scored
date on (or from `start`). Every `hold` dates along that axis, starting with
the first, is a rebalance date t. At the close of t the engine ranks the
eligible names by score (ties broken by ticker, so a run is deterministic) and
keeps the top `top_k`:

* a held name still in the top K is **not traded** — its shares are kept and
  its weight drifts with its price;
* a held name that dropped out is sold at t's close;
* the sale proceeds, plus any idle cash, are split *equally* among the names
  that entered, bought at t's close.

So every purchase is equal-weight, and a name is held for at least `hold`
sessions. `reweight=True` instead resets the whole book to 1/K of equity at
every rebalance (a "rebalanced" equal weight, with the turnover that implies).
Between rebalances nothing trades and the number of shares is constant.

**Why retained names are not touched.** This is what makes the acceptance
criterion an identity: with a constant score the selection never changes,
nothing ever trades after the first day, and the curve *is* equal-weight
buy-and-hold — `capital / K * sum(p[t] / p[0])` — to the last bit, not to a
tolerance. `tests/test_engine.py` checks exactly that.

**Eligibility.** A name can be *bought* on t only if it has a score and a price
dated t (it traded that session). A held name with a score stays eligible on a
day its venue is closed — it is valued at its last close. A held name with *no*
score on t (scores, like features, exist only on a name's own sessions) is
"no opinion": it is neither sold nor re-ranked, and it keeps its slot in the K.
Between sessions on its own venue a name is likewise valued at its last close (prices are carried
forward, never backward), so the multi-venue axis does not punch holes in the
curve. If fewer than K names are eligible the book holds fewer; if none is, it
holds cash, at zero interest. A book with an empty slot fills it only from cash:
with `reweight=False`, no retained name is trimmed to make room.

**What v1 does not do.** No costs, no spread, no slippage, and it trades at the
close of the day the score is dated — the score uses that close too, so this is
optimistic by one session. Day 36 adds the frictions and moves every fill to
the *next* session's open. Treat v1 curves as an upper bound until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from finmgr.features.returns import DEFAULT_PRICE

#: Starting capital, in EUR.
DEFAULT_CAPITAL = 100_000.0

EQUITY_COLUMNS = ("date", "equity_eur", "cash_eur", "n_names", "rebalance")
POSITION_COLUMNS = ("date", "ticker", "shares", "price_eur", "value_eur", "weight")
TRADE_COLUMNS = ("date", "ticker", "side", "shares", "price_eur", "value_eur")


@dataclass(frozen=True)
class BacktestResult:
    """What :func:`run_backtest` returns: three long frames, all in EUR."""

    #: One row per date on the axis: equity, idle cash, names held, rebalance flag.
    equity: pd.DataFrame
    #: One row per (date, held ticker), after that date's trades.
    positions: pd.DataFrame
    #: One row per fill: `side` is "buy" or "sell", `shares` always positive.
    trades: pd.DataFrame

    @property
    def curve(self) -> pd.Series:
        """The equity curve as a Series indexed by date."""
        return self.equity.set_index("date")["equity_eur"]


def _to_date(value: date | str | pd.Timestamp) -> date:
    return pd.Timestamp(value).date()


def _wide(frame: pd.DataFrame, column: str, what: str) -> pd.DataFrame:
    missing = {"ticker", "date", column} - set(frame.columns)
    if missing:
        raise KeyError(f"{what} is missing column(s): {', '.join(sorted(missing))}")
    data = frame[["ticker", "date", column]].copy()
    data["date"] = [_to_date(d) for d in data["date"]]
    if data.duplicated(["ticker", "date"]).any():
        raise ValueError(f"{what} has more than one row for a (ticker, date)")
    return data.pivot(index="date", columns="ticker", values=column).sort_index()


def _select(
    scores: np.ndarray, eligible: np.ndarray, tickers: np.ndarray, top_k: int
) -> np.ndarray:
    """Indices of the top `top_k` eligible names: score descending, then ticker ascending."""
    idx = np.flatnonzero(eligible)
    # lexsort sorts by the last key first: -score, then ticker for ties.
    order = np.lexsort((tickers[idx], -scores[idx]))
    return idx[order[:top_k]]


def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_k: int,
    hold: int,
    capital: float = DEFAULT_CAPITAL,
    price: str = DEFAULT_PRICE,
    score: str = "score",
    start: date | str | None = None,
    end: date | str | None = None,
    reweight: bool = False,
) -> BacktestResult:
    """Pick the top `top_k` by score every `hold` dates, equal weight, and mark to market daily.

    See the module docstring for the rule. `start`/`end` trim the date axis
    (inclusive); by default it runs from the first date with any score to the
    last price.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if hold < 1:
        raise ValueError(f"hold must be a positive session count, got {hold}")
    if not capital > 0:
        raise ValueError(f"capital must be positive, got {capital}")

    px = _wide(prices, price, "prices")
    sc = _wide(scores, score, "scores")
    sc = sc.dropna(how="all")
    if sc.empty:
        raise ValueError("scores has no non-missing score")
    first = _to_date(start) if start is not None else sc.index[0]
    axis = px.index[px.index >= first]
    if end is not None:
        axis = axis[axis <= _to_date(end)]
    if axis.empty:
        raise ValueError("no price dates on the backtest axis")

    tickers = px.columns.to_numpy(dtype=str)
    # Prices on and before the axis feed the carry-forward; scores align to the axis.
    traded = px.loc[axis].notna().to_numpy()
    marks = px.loc[: axis[-1]].ffill().loc[axis].to_numpy(dtype=float)
    score_m = sc.reindex(index=axis, columns=px.columns).to_numpy(dtype=float)

    n_names = len(tickers)
    shares = np.zeros(n_names)
    cash = float(capital)
    equity_rows, position_rows, trade_rows = [], [], []

    for i, day in enumerate(axis):
        mark = marks[i]
        is_rebalance = i % hold == 0
        if is_rebalance:
            held = shares > 0
            has_score = ~np.isnan(score_m[i])
            # A held name with no score today (its venue is shut, its features
            # are missing) is "no opinion": kept as is, and it keeps its slot.
            frozen = held & ~has_score
            eligible = has_score & (traded[i] | (held & ~np.isnan(mark)))
            picked = _select(score_m[i], eligible, tickers, max(top_k - int(frozen.sum()), 0))
            chosen = np.union1d(np.flatnonzero(frozen), picked)
            keep = np.zeros(n_names, dtype=bool)
            keep[chosen] = True

            if reweight:
                target = np.zeros(n_names)
                total = cash + float(np.dot(shares[held], mark[held]))
                if chosen.size:
                    target[chosen] = total / chosen.size / mark[chosen]
                delta = target - shares
                cash = total - float(np.dot(target[keep], mark[keep]))
                shares = target
                changed = np.flatnonzero(np.abs(delta) > 0)
            else:
                exits = np.flatnonzero(held & ~keep)
                entrants = chosen[~held[chosen]]
                delta = np.zeros(n_names)
                cash += float(np.dot(shares[exits], mark[exits]))
                delta[exits] = -shares[exits]
                shares[exits] = 0.0
                if entrants.size and cash > 0:
                    bought = cash / entrants.size / mark[entrants]
                    shares[entrants] = bought
                    delta[entrants] = bought
                    cash = 0.0
                changed = np.flatnonzero(delta != 0)

            for j in changed:
                trade_rows.append(
                    (
                        day,
                        tickers[j],
                        "buy" if delta[j] > 0 else "sell",
                        abs(delta[j]),
                        mark[j],
                        abs(delta[j]) * mark[j],
                    )
                )

        held_idx = np.flatnonzero(shares > 0)
        values = shares[held_idx] * mark[held_idx]
        equity = cash + float(values.sum())
        for j, value in zip(held_idx, values, strict=True):
            position_rows.append((day, tickers[j], shares[j], mark[j], value, value / equity))
        equity_rows.append((day, equity, cash, held_idx.size, is_rebalance))

    return BacktestResult(
        equity=pd.DataFrame(equity_rows, columns=list(EQUITY_COLUMNS)),
        positions=pd.DataFrame(position_rows, columns=list(POSITION_COLUMNS)),
        trades=pd.DataFrame(trade_rows, columns=list(TRADE_COLUMNS)),
    )


def constant_scores(prices: pd.DataFrame, value: float = 1.0) -> pd.DataFrame:
    """The same `value` for every `(ticker, date)` in `prices`: the buy-and-hold control."""
    out = prices[["ticker", "date"]].copy()
    out["score"] = float(value)
    return out.reset_index(drop=True)
