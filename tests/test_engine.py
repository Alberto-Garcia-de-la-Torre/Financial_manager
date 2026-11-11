"""Day 35: backtest engine v1.

The acceptance criterion: feeding the engine a constant score reproduces plain
equal-weight buy-and-hold exactly. It is checked against a buy-and-hold curve
built independently in pandas — shares bought for capital / n on the first
day, marked to market every day, never traded again — on one venue and on a
two-venue axis with holidays, for the whole universe and for a top-K subset,
and for every holding period. "Exactly" means: the same trades (n buys on the
first day, nothing after), the same share counts on every day, and the same
equity to floating-point rounding (rtol 1e-12).

Around it: the top-K pick on a known score, the hold period, how entrants are
funded, `reweight`, eligibility on a closed venue, and the input checks.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from finmgr.backtest.engine import (
    EQUITY_COLUMNS,
    POSITION_COLUMNS,
    TRADE_COLUMNS,
    constant_scores,
    run_backtest,
)
from finmgr.features.returns import venue_sessions

D = dt.date
NYSE = venue_sessions("XNYS", D(2023, 1, 1), D(2023, 12, 29))
XETRA = venue_sessions("XETR", D(2023, 1, 1), D(2023, 12, 29))
CAPITAL = 100_000.0


def make_prices(calendars: dict[str, list[dt.date]], seed: int = 35) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [
        pd.DataFrame(
            {
                "ticker": t,
                "date": days,
                "adj_close_eur": 50 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(days))),
            }
        )
        for t, days in calendars.items()
    ]
    return pd.concat(frames, ignore_index=True)


def buy_and_hold(
    prices: pd.DataFrame, tickers: list[str], capital: float, start: dt.date | None = None
) -> pd.Series:
    """Equal-weight buy-and-hold of `tickers` from `start` (default: the first date), by hand."""
    wide = prices.pivot(index="date", columns="ticker", values="adj_close_eur").sort_index()
    if start is not None:
        wide = wide.loc[start:]
    wide = wide[tickers].ffill()
    first = wide.iloc[0]
    assert first.notna().all(), "every name must trade on the first day"
    shares = capital / len(tickers) / first
    return (wide * shares).sum(axis=1)


US = {f"U{i:02d}": NYSE for i in range(6)}
EU = {f"E{i:02d}": XETRA for i in range(4)}
#: The first date both venues trade (XETRA opens 2 Jan 2023, NYSE 3 Jan).
COMMON = min(set(NYSE) & set(XETRA))


# --- the acceptance criterion ------------------------------------------------


@pytest.mark.parametrize("hold", [1, 5, 21, 1000])
@pytest.mark.parametrize("calendars", [US, {**US, **EU}], ids=["one-venue", "two-venues"])
def test_constant_score_is_equal_weight_buy_and_hold(calendars, hold) -> None:
    prices = make_prices(calendars)
    tickers = sorted(calendars)
    result = run_backtest(
        constant_scores(prices),
        prices,
        top_k=len(tickers),
        hold=hold,
        capital=CAPITAL,
        start=COMMON,
    )
    expected = buy_and_hold(prices, tickers, CAPITAL, start=COMMON)

    curve = result.curve
    assert list(curve.index) == list(expected.index)
    np.testing.assert_allclose(curve.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0)
    assert curve.iloc[0] == pytest.approx(CAPITAL, rel=1e-15)

    # Same trades: one buy per name on the first day, nothing ever after.
    first = curve.index[0]
    assert (result.trades["date"] == first).all()
    assert (result.trades["side"] == "buy").all()
    assert sorted(result.trades["ticker"]) == tickers
    np.testing.assert_allclose(result.trades["value_eur"], CAPITAL / len(tickers), rtol=1e-15)

    # Same shares every day, all names held every day, no cash.
    per_name = result.positions.groupby("ticker")["shares"]
    assert (per_name.nunique() == 1).all()
    assert (result.equity["n_names"] == len(tickers)).all()
    assert (result.equity["cash_eur"] == 0).all()


@pytest.mark.parametrize("top_k", [1, 3, 7])
def test_constant_score_top_k_is_buy_and_hold_of_first_k(top_k) -> None:
    calendars = {**US, **EU}
    prices = make_prices(calendars, seed=7)
    result = run_backtest(
        constant_scores(prices), prices, top_k=top_k, hold=5, capital=CAPITAL, start=COMMON
    )
    chosen = sorted(calendars)[:top_k]  # ties broken by ticker
    expected = buy_and_hold(prices, chosen, CAPITAL, start=COMMON)
    np.testing.assert_allclose(result.curve.to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert set(result.positions["ticker"]) == set(chosen)
    assert len(result.trades) == top_k


def test_constant_score_from_first_scored_date_is_buy_and_hold() -> None:
    """Any constant value works, and the axis starts at the first scored date."""
    prices = make_prices(US, seed=99)
    scores = constant_scores(prices, value=-3.5)
    start = NYSE[40]
    scores = scores[scores["date"] >= start]
    result = run_backtest(scores, prices, top_k=10, hold=21, capital=CAPITAL)
    expected = buy_and_hold(prices[prices["date"] >= start], sorted(US), CAPITAL)
    assert result.curve.index[0] == start
    np.testing.assert_allclose(result.curve.to_numpy(), expected.to_numpy(), rtol=1e-12)


# --- the rule ----------------------------------------------------------------


def trending(n_days: int = 30) -> tuple[pd.DataFrame, list[dt.date]]:
    days = NYSE[:n_days]
    prices = pd.DataFrame(
        [(t, d, 10.0 + i) for t in "ABCD" for i, d in enumerate(days)],
        columns=["ticker", "date", "adj_close_eur"],
    )
    return prices, days


def test_picks_top_k_by_score_and_equal_weights() -> None:
    prices, days = trending(10)
    scores = pd.DataFrame(
        [(t, d, s) for d in days for t, s in zip("ABCD", [1.0, 4.0, 3.0, 2.0], strict=True)],
        columns=["ticker", "date", "score"],
    )
    result = run_backtest(scores, prices, top_k=2, hold=5, capital=1000.0)
    day0 = result.positions[result.positions["date"] == days[0]]
    assert sorted(day0["ticker"]) == ["B", "C"]
    np.testing.assert_allclose(day0["weight"], 0.5)
    np.testing.assert_allclose(day0["value_eur"], 500.0)


def test_holds_n_sessions_then_rotates() -> None:
    prices, days = trending(12)
    rows = []
    for i, d in enumerate(days):
        best = "A" if i < 3 else "D"  # the signal flips on day 3
        rows += [(t, d, 1.0 if t == best else 0.0) for t in "ABCD"]
    scores = pd.DataFrame(rows, columns=["ticker", "date", "score"])
    result = run_backtest(scores, prices, top_k=1, hold=5, capital=1000.0)

    held = result.positions.set_index("date")["ticker"]
    assert list(held.loc[days[:5]]) == ["A"] * 5  # flip on day 3 ignored until day 5
    assert list(held.loc[days[5:]]) == ["D"] * 7
    assert list(result.equity["rebalance"]) == [i % 5 == 0 for i in range(12)]

    sale = result.trades[result.trades["side"] == "sell"].iloc[0]
    buy = result.trades[(result.trades["side"] == "buy") & (result.trades["date"] == days[5])]
    assert sale["date"] == days[5] and sale["ticker"] == "A"
    assert buy["value_eur"].iloc[0] == pytest.approx(sale["value_eur"])
    # A was bought at 10, sold at 15: 1000 -> 1500, then D from 15 to 21.
    assert result.curve.iloc[-1] == pytest.approx(1500.0 * 21 / 15)


def test_entrants_split_exit_proceeds_retained_untouched() -> None:
    prices, days = trending(6)
    rows = []
    for i, d in enumerate(days):
        top = ["A", "B"] if i < 3 else ["A", "C"]
        rows += [(t, d, 1.0 if t in top else 0.0) for t in "ABCD"]
    scores = pd.DataFrame(rows, columns=["ticker", "date", "score"])
    result = run_backtest(scores, prices, top_k=2, hold=3, capital=1000.0)
    a = result.positions[result.positions["ticker"] == "A"]["shares"]
    assert a.nunique() == 1  # A never traded
    traded = result.trades[result.trades["date"] == days[3]]
    assert sorted(zip(traded["ticker"], traded["side"], strict=True)) == [
        ("B", "sell"),
        ("C", "buy"),
    ]


def test_reweight_resets_to_equal_weight() -> None:
    days = NYSE[:4]
    prices = pd.DataFrame(
        [("A", d, p) for d, p in zip(days, [10, 20, 20, 20], strict=True)]
        + [("B", d, 10.0) for d in days],
        columns=["ticker", "date", "adj_close_eur"],
    )
    scores = constant_scores(prices)
    drift = run_backtest(scores, prices, top_k=2, hold=2, capital=1000.0)
    reset = run_backtest(scores, prices, top_k=2, hold=2, capital=1000.0, reweight=True)
    on_day2 = reset.positions[reset.positions["date"] == days[2]]
    np.testing.assert_allclose(on_day2["weight"], 0.5)
    drifted = drift.positions[drift.positions["date"] == days[2]].set_index("ticker")["weight"]
    assert drifted["A"] == pytest.approx(1000 / 1500)
    assert reset.curve.iloc[-1] == pytest.approx(1500.0)
    assert (reset.equity["cash_eur"].abs() < 1e-9).all()


def test_closed_venue_held_name_kept_new_name_not_bought() -> None:
    days = NYSE[:3]
    prices = pd.DataFrame(
        [("A", d, 10.0) for d in days] + [("B", days[0], 10.0), ("B", days[2], 10.0)],
        columns=["ticker", "date", "adj_close_eur"],
    )
    # B is held from day 0 and has no bar on day 1, a rebalance date: kept, valued at 10.
    result = run_backtest(constant_scores(prices), prices, top_k=2, hold=1, capital=100.0)
    assert (result.equity["n_names"] == 2).all()
    assert len(result.trades) == 2

    # C has no bar on the first day: it cannot be bought then, and with top_k=1
    # and C the best score it is bought on day 1 when it trades.
    late = pd.DataFrame(
        [("A", d, 10.0) for d in days] + [("C", d, 5.0) for d in days[1:]],
        columns=["ticker", "date", "adj_close_eur"],
    )
    scores = pd.DataFrame(
        [("A", d, 0.0) for d in days] + [("C", d, 1.0) for d in days],
        columns=["ticker", "date", "score"],
    )
    rolled = run_backtest(scores, late, top_k=1, hold=1, capital=100.0)
    assert list(rolled.positions["ticker"]) == ["A", "C", "C"]


def test_no_eligible_names_holds_cash() -> None:
    prices, days = trending(5)
    scores = constant_scores(prices)
    scores.loc[scores["date"] == days[0], "score"] = np.nan
    scores = scores.dropna()
    result = run_backtest(scores, prices, top_k=2, hold=1, capital=1000.0, start=days[0])
    assert result.equity["cash_eur"].iloc[0] == 1000.0
    assert result.equity["n_names"].iloc[0] == 0
    assert result.equity["n_names"].iloc[1] == 2


def test_frames_have_documented_columns() -> None:
    prices, _ = trending(5)
    result = run_backtest(constant_scores(prices), prices, top_k=2, hold=2)
    assert tuple(result.equity.columns) == EQUITY_COLUMNS
    assert tuple(result.positions.columns) == POSITION_COLUMNS
    assert tuple(result.trades.columns) == TRADE_COLUMNS
    sums = result.positions.groupby("date")["weight"].sum()
    np.testing.assert_allclose(sums, 1.0)


@pytest.mark.parametrize(
    "kwargs, error",
    [({"top_k": 0, "hold": 1}, ValueError), ({"top_k": 1, "hold": 0}, ValueError)],
)
def test_rejects_bad_parameters(kwargs, error) -> None:
    prices, _ = trending(3)
    with pytest.raises(error):
        run_backtest(constant_scores(prices), prices, **kwargs)


def test_rejects_duplicate_and_missing_columns() -> None:
    prices, _ = trending(3)
    with pytest.raises(ValueError, match="more than one row"):
        run_backtest(constant_scores(prices), pd.concat([prices, prices]), top_k=1, hold=1)
    with pytest.raises(KeyError, match="score"):
        run_backtest(prices[["ticker", "date"]], prices, top_k=1, hold=1)
