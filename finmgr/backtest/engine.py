"""Backtest engine v1: top-K by score, equal weight, hold N sessions, roll forward.

    from finmgr.backtest.costs import CostModel, read_backtest_prices
    from finmgr.backtest.engine import run_backtest

    prices = read_backtest_prices(as_of="2026-10-15", start="2020-01-01")
    result = run_backtest(scores, prices, top_k=10, hold=21, costs=CostModel.from_settings())
    result.equity        # date, equity_eur, cash_eur, n_names, rebalance, cost_eur
    result.positions     # date, ticker, shares, price_eur, value_eur, weight
    result.trades        # date, ticker, side, shares, price_eur, value_eur, cost_eur, decided

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
* a held name that dropped out is sold;
* the sale proceeds, plus any idle cash, are split *equally* among the names
  that entered, which are bought (fills: see below).

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

**Fills and costs (day 36).** The score dated t uses t's close, so the
engine never fills at that close: every order decided at t is sized at t's
marks and filled at the *next* session's open (`fill="next_open"`, the
default) — per name, so a name whose venue is shut the next day fills at its
own next open. Sells fill before buys. An order still unfilled at the next
rebalance is dropped and the new decision replaces it. Every fill pays the
one-way cost of :class:`~finmgr.backtest.costs.CostModel` — commission, half
the spread and slippage, in bps, the last two optionally scaled by inverse
liquidity — and buy budgets are funded by sale proceeds net of their costs.
Because the budget is sized at t's close and spent at t+1's open, cash can sit
slightly above or below zero by the overnight gap of the names sold; it earns
and costs nothing. `fill="close"` with `costs=NO_COSTS` is v1's frictionless,
same-close engine, kept only as an identity for tests: it is optimistic by a
session and by every cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from finmgr.backtest.costs import LIQUIDITY_COLUMN, OPEN_COLUMN, CostModel
from finmgr.features.returns import DEFAULT_PRICE

#: Starting capital, in EUR.
DEFAULT_CAPITAL = 100_000.0

#: When an order decided at t's close fills.
Fill = Literal["next_open", "close"]
FILLS: tuple[str, ...] = ("next_open", "close")

EQUITY_COLUMNS = ("date", "equity_eur", "cash_eur", "n_names", "rebalance", "cost_eur")
POSITION_COLUMNS = ("date", "ticker", "shares", "price_eur", "value_eur", "weight")
TRADE_COLUMNS = (
    "date",
    "ticker",
    "side",
    "shares",
    "price_eur",
    "value_eur",
    "cost_eur",
    "decided",
)

#: Below this share of capital a cash balance is floating-point dust, not money.
_DUST = 1e-9


@dataclass(frozen=True)
class BacktestResult:
    """What :func:`run_backtest` returns: three long frames, all in EUR."""

    #: One row per date on the axis: equity, idle cash, names held, rebalance flag, costs paid.
    equity: pd.DataFrame
    #: One row per (date, held ticker), after that date's trades.
    positions: pd.DataFrame
    #: One row per fill: `side` is "buy" or "sell", `shares` always positive,
    #: `value_eur` the notional at `price_eur`, `cost_eur` the frictions on top,
    #: `decided` the rebalance date whose close the order was sized at.
    trades: pd.DataFrame

    @property
    def curve(self) -> pd.Series:
        """The equity curve as a Series indexed by date."""
        return self.equity.set_index("date")["equity_eur"]

    @property
    def total_cost(self) -> float:
        """Every euro paid in frictions over the run."""
        return float(self.trades["cost_eur"].sum())


@dataclass
class _Order:
    """One pending order: sell `amount` shares, or buy with a budget of `amount` EUR."""

    j: int
    side: str
    amount: float
    rate: float
    decided: date


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


def _decide(
    shares: np.ndarray,
    cash: float,
    mark: np.ndarray,
    chosen: np.ndarray,
    rates: np.ndarray,
    day: date,
    reweight: bool,
) -> list[_Order]:
    """The orders that take the book to `chosen`, sized at today's marks, sells first.

    Buy budgets are funded by idle cash plus the sales' proceeds *net of their
    costs*, estimated at today's mark, so the book stays (close to) fully
    invested without borrowing for its own frictions.
    """
    n_names = len(shares)
    held = shares > 0
    keep = np.zeros(n_names, dtype=bool)
    keep[chosen] = True
    sells: list[_Order] = []
    wants = np.zeros(n_names)  # EUR each name should receive, before scaling to the cash

    if reweight:
        value = np.where(held, shares * np.nan_to_num(mark), 0.0)
        total = cash + float(value.sum())
        target = np.zeros(n_names)
        if chosen.size:
            target[chosen] = total / chosen.size
        for j in np.flatnonzero(held):
            if not keep[j]:
                sells.append(_Order(j, "sell", shares[j], rates[j], day))
            elif value[j] > target[j]:
                sells.append(_Order(j, "sell", (value[j] - target[j]) / mark[j], rates[j], day))
        under = keep & (value < target)
        wants[under] = target[under] - value[under]
    else:
        for j in np.flatnonzero(held & ~keep):
            sells.append(_Order(j, "sell", shares[j], rates[j], day))
        entrants = chosen[~held[chosen]]
        wants[entrants] = 1.0  # equal shares of whatever there is

    proceeds = sum(o.amount * mark[o.j] * (1 - o.rate) for o in sells)
    available = cash + proceeds
    buys: list[_Order] = []
    wanted = float(wants.sum())
    if wanted > 0 and available > 0:
        scale = available / wanted if not reweight else min(1.0, available / wanted)
        buys = [
            _Order(j, "buy", wants[j] * scale, rates[j], day) for j in np.flatnonzero(wants > 0)
        ]
    return sells + buys


def run_backtest(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    top_k: int,
    hold: int,
    capital: float = DEFAULT_CAPITAL,
    costs: CostModel | None = None,
    fill: Fill = "next_open",
    price: str = DEFAULT_PRICE,
    open_price: str = OPEN_COLUMN,
    liquidity: str = LIQUIDITY_COLUMN,
    score: str = "score",
    start: date | str | None = None,
    end: date | str | None = None,
    reweight: bool = False,
) -> BacktestResult:
    """Pick the top `top_k` by score every `hold` dates, equal weight, and mark to market daily.

    See the module docstring for the rule. `costs` defaults to
    :class:`CostModel`'s defaults (the values in `config/settings.yaml`); pass
    :data:`~finmgr.backtest.costs.NO_COSTS` for free fills. With
    `fill="next_open"` (the default) `prices` needs an `open_price` column;
    with liquidity scaling on it needs a `liquidity` column too.
    :func:`~finmgr.backtest.costs.read_backtest_prices` returns both.
    `start`/`end` trim the date axis (inclusive); by default it runs from the
    first date with any score to the last price.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    if hold < 1:
        raise ValueError(f"hold must be a positive session count, got {hold}")
    if not capital > 0:
        raise ValueError(f"capital must be positive, got {capital}")
    if fill not in FILLS:
        raise ValueError(f"fill must be one of {', '.join(FILLS)}, got {fill!r}")
    model = costs if costs is not None else CostModel()

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
    if fill == "next_open":
        opens = (
            _wide(prices, open_price, "prices")
            .reindex(index=axis, columns=px.columns)
            .to_numpy(dtype=float)
        )
    if model.needs_liquidity:
        advs = (
            _wide(prices, liquidity, "prices")
            .reindex(columns=px.columns)
            .loc[: axis[-1]]
            .ffill()
            .reindex(axis)
            .to_numpy(dtype=float)
        )

    n_names = len(tickers)
    shares = np.zeros(n_names)
    cash = float(capital)
    pending: list[_Order] = []
    equity_rows, position_rows, trade_rows = [], [], []

    def execute(order: _Order, at: float, day: date) -> float:
        """Fill `order` at price `at` on `day`; return the cost paid."""
        nonlocal cash
        j = order.j
        if order.side == "sell":
            qty = min(order.amount, shares[j])
            notional = qty * at
            cost = notional * order.rate
            shares[j] = shares[j] - qty if qty < shares[j] else 0.0
            cash += notional - cost
        else:
            qty = order.amount / (at * (1 + order.rate))
            notional = qty * at
            cost = notional * order.rate
            shares[j] += qty
            cash -= notional + cost
        trade_rows.append((day, tickers[j], order.side, qty, at, notional, cost, order.decided))
        return cost

    for i, day in enumerate(axis):
        mark = marks[i]
        paid = 0.0

        # Yesterday's (or older) orders fill at this session's open, sells first.
        if pending:
            ready = [o for o in pending if not np.isnan(opens[i, o.j])]
            pending = [o for o in pending if np.isnan(opens[i, o.j])]
            for order in sorted(ready, key=lambda o: o.side != "sell"):
                paid += execute(order, opens[i, order.j], day)

        is_rebalance = i % hold == 0
        if is_rebalance:
            # An order that never found an open is stale: the new decision replaces it.
            pending = []
            held = shares > 0
            has_score = ~np.isnan(score_m[i])
            # A held name with no score today (its venue is shut, its features
            # are missing) is "no opinion": kept as is, and it keeps its slot.
            frozen = held & ~has_score
            eligible = has_score & (traded[i] | (held & ~np.isnan(mark)))
            picked = _select(score_m[i], eligible, tickers, max(top_k - int(frozen.sum()), 0))
            chosen = np.union1d(np.flatnonzero(frozen), picked)
            rates = model.rate(advs[i] if model.needs_liquidity else np.full(n_names, np.nan))
            orders = _decide(shares, cash, mark, chosen, rates, day, reweight)
            if fill == "close":
                for order in orders:
                    paid += execute(order, mark[order.j], day)
            else:
                pending = orders

        if abs(cash) < _DUST * capital:
            cash = 0.0
        held_idx = np.flatnonzero(shares > 0)
        values = shares[held_idx] * mark[held_idx]
        equity = cash + float(values.sum())
        for j, value in zip(held_idx, values, strict=True):
            position_rows.append((day, tickers[j], shares[j], mark[j], value, value / equity))
        equity_rows.append((day, equity, cash, held_idx.size, is_rebalance, paid))

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
