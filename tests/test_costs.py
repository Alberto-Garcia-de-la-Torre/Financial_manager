"""Day 36: costs and frictions.

The acceptance criterion: raising costs to 50bps visibly and plausibly degrades
the equity curve. "Visibly": on a rotating strategy the curve net of 50bps sits
below the frictionless one on every day after the first fill, and final equity
falls monotonically as costs rise through 0, 5, 10, 25, 50bps. "Plausibly":
the euros lost are what the turnover says they should be — every fill pays
exactly 50bps of its notional, a buy-and-hold pays it once (its curve is the
free one divided by 1.005, every day), and the rotating strategy's drag is
within a factor of its traded notional times 50bps.

Around it: the cost formula and its liquidity scaling, the settings section,
and the rule that an order decided at t's close fills at the next open — per
name, on that name's own next session — never at t's close.
"""

from __future__ import annotations

import csv
import datetime as dt
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from finmgr.backtest.costs import (
    BACKTEST_PRICE_COLUMNS,
    NO_COSTS,
    CostModel,
    adjusted_open,
    read_backtest_prices,
)
from finmgr.backtest.engine import constant_scores, run_backtest
from finmgr.config import SETTINGS_PATH_ENV, load_settings
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS
from finmgr.features.returns import venue_sessions

D = dt.date
NYSE = venue_sessions("XNYS", D(2023, 1, 1), D(2023, 12, 29))
XETRA = venue_sessions("XETR", D(2023, 1, 1), D(2023, 12, 29))
CAPITAL = 100_000.0


def make_prices(calendars: dict[str, list[dt.date]], seed: int = 36) -> pd.DataFrame:
    """Closes on a random walk, and opens an overnight gap away from the previous close."""
    rng = np.random.default_rng(seed)
    frames = []
    for ticker, days in calendars.items():
        close = 50 * np.cumprod(1 + rng.normal(0.0004, 0.02, len(days)))
        previous = np.concatenate([[50.0], close[:-1]])
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": days,
                    "adj_close_eur": close,
                    "adj_open_eur": previous * (1 + rng.normal(0, 0.005, len(days))),
                    "adv_eur_21": 100e6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def random_scores(prices: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    out = prices[["ticker", "date"]].copy()
    out["score"] = np.random.default_rng(seed).normal(size=len(out))
    return out


US = {f"U{i:02d}": NYSE for i in range(10)}
EU = {f"E{i:02d}": XETRA for i in range(4)}
COMMON = min(set(NYSE) & set(XETRA))


# --- the acceptance criterion ------------------------------------------------


def test_50bps_degrades_a_rotating_strategy_visibly_and_plausibly() -> None:
    prices = make_prices({**US, **EU})
    scores = random_scores(prices)
    run = {
        bps: run_backtest(
            scores, prices, top_k=4, hold=5, capital=CAPITAL, costs=CostModel.flat(bps)
        )
        for bps in (0, 5, 10, 25, 50)
    }
    free, dear = run[0], run[50]

    # Same decisions, same fill dates: costs change the sizes, not the picks.
    assert list(free.trades["ticker"]) == list(dear.trades["ticker"])
    assert list(free.trades["date"]) == list(dear.trades["date"])

    # Visibly: monotone in cost, and below the free curve after the first fill.
    finals = [run[bps].curve.iloc[-1] for bps in (0, 5, 10, 25, 50)]
    assert all(a > b for a, b in pairwise(finals))
    first_fill = free.trades["date"].min()
    after = free.curve.index >= first_fill
    assert (dear.curve[after] < free.curve[after]).all()
    assert (dear.curve[~after] == free.curve[~after]).all()

    # Plausibly: every fill pays exactly 50bps of its notional...
    np.testing.assert_allclose(
        dear.trades["cost_eur"], 0.005 * dear.trades["value_eur"], rtol=1e-12
    )
    assert free.total_cost == 0.0
    # ... and the drag is the cost of the turnover, give or take what the
    # money not lost would have earned since (about a year at +-20% here).
    traded = float(dear.trades["value_eur"].sum())
    lost = free.curve.iloc[-1] - dear.curve.iloc[-1]
    assert dear.total_cost == pytest.approx(0.005 * traded)
    assert 0.6 < lost / dear.total_cost < 1.6
    # Weekly rotation of 4 names is hundreds of percent of turnover a year:
    # 50bps of it is a several-percent hit, not a rounding error.
    assert lost / CAPITAL > 0.03


def test_50bps_on_buy_and_hold_is_paid_once() -> None:
    prices = make_prices(US)
    kwargs = {"top_k": 10, "hold": 21, "capital": CAPITAL}
    free = run_backtest(constant_scores(prices), prices, costs=NO_COSTS, **kwargs)
    dear = run_backtest(constant_scores(prices), prices, costs=CostModel.flat(50), **kwargs)
    assert len(dear.trades) == 10 and (dear.trades["side"] == "buy").all()
    # The day of the decision is all cash; from the fill on, the book is the
    # free one bought with capital / 1.005.
    ratio = (dear.curve / free.curve).iloc[1:]
    np.testing.assert_allclose(ratio, 1 / 1.005, rtol=1e-12)
    assert dear.total_cost == pytest.approx(CAPITAL * 0.005 / 1.005)


# --- next-open fills -----------------------------------------------------------


def test_constant_score_buys_at_the_next_open_and_then_holds() -> None:
    prices = make_prices({**US, **EU}, seed=7)
    tickers = sorted(prices["ticker"].unique())
    result = run_backtest(
        constant_scores(prices),
        prices,
        top_k=len(tickers),
        hold=5,
        capital=CAPITAL,
        costs=NO_COSTS,
        start=COMMON,
    )
    wide = prices.pivot(index="date", columns="ticker", values="adj_close_eur").loc[COMMON:]
    opens = prices.pivot(index="date", columns="ticker", values="adj_open_eur").loc[COMMON:]
    # Each name fills at its own first session after the decision date.
    fill_day = {t: opens.index[(opens.index > COMMON) & opens[t].notna()][0] for t in tickers}
    shares = pd.Series({t: CAPITAL / len(tickers) / opens.at[fill_day[t], t] for t in tickers})

    trades = result.trades.set_index("ticker")
    assert (trades["decided"] == COMMON).all()
    for t in tickers:
        assert trades.at[t, "date"] == fill_day[t]
        assert trades.at[t, "price_eur"] == opens.at[fill_day[t], t]
    assert len(result.trades) == len(tickers)  # nothing traded after

    # From the day every name has filled, the curve is buy-and-hold at those opens.
    settled = max(fill_day.values())
    expected = (wide.ffill() * shares).sum(axis=1).loc[settled:]
    np.testing.assert_allclose(result.curve.loc[settled:], expected, rtol=1e-12)
    assert result.curve.iloc[0] == CAPITAL  # decision day: still cash


def test_never_fills_at_the_close_the_score_was_dated() -> None:
    days = NYSE[:4]
    # A at 10 every close; it opens at 11 on day 1, so the fill is visibly not day 0's close.
    prices = pd.DataFrame(
        {
            "ticker": "A",
            "date": days,
            "adj_close_eur": [10.0, 12.0, 12.0, 12.0],
            "adj_open_eur": [10.0, 11.0, 12.0, 12.0],
        }
    )
    result = run_backtest(
        constant_scores(prices), prices, top_k=1, hold=10, capital=110.0, costs=NO_COSTS
    )
    (trade,) = result.trades.itertuples()
    assert trade.date == days[1] and trade.decided == days[0]
    assert trade.price_eur == 11.0 and trade.shares == pytest.approx(10.0)
    assert list(result.curve) == pytest.approx([110.0, 120.0, 120.0, 120.0])

    # The same-close mode would have bought 11 shares at 10: the optimism day 36 removes.
    close = run_backtest(
        constant_scores(prices), prices, top_k=1, hold=10, capital=110.0, costs=NO_COSTS,
        fill="close",
    )  # fmt: skip
    assert close.curve.iloc[-1] == pytest.approx(132.0)


def test_a_rotation_sells_and_buys_at_the_next_open() -> None:
    days = NYSE[:4]
    prices = pd.DataFrame(
        [("A", d, 10.0, 10.0) for d in days] + [("B", d, 20.0, 20.0) for d in days],
        columns=["ticker", "date", "adj_close_eur", "adj_open_eur"],
    )
    scores = pd.DataFrame(
        [("A", d, 1.0 if d < days[2] else 0.0) for d in days] + [("B", d, 0.5) for d in days],
        columns=["ticker", "date", "score"],
    )
    result = run_backtest(
        scores, prices, top_k=1, hold=2, capital=1000.0, costs=CostModel.flat(100)
    )
    sides = result.trades[["date", "ticker", "side"]].itertuples(index=False)
    assert [tuple(r) for r in sides] == [
        (days[1], "A", "buy"),
        (days[3], "A", "sell"),
        (days[3], "B", "buy"),
    ]
    # 1000 buys 1000/1.01 of A; its sale returns 99% of that; all of it buys B.
    a_value = 1000 / 1.01
    b_budget = a_value * 0.99
    assert result.curve.iloc[-1] == pytest.approx(b_budget / 1.01)
    assert result.equity["cash_eur"].iloc[-1] == 0.0
    assert result.total_cost == pytest.approx(1000 - b_budget / 1.01)


def test_a_name_on_a_shut_venue_fills_at_its_own_next_open() -> None:
    days = NYSE[:4]
    prices = pd.DataFrame(
        [("A", d, 10.0, 10.0) for d in days]
        + [("B", days[0], 10.0, 10.0), ("B", days[2], 10.0, 9.0), ("B", days[3], 10.0, 10.0)],
        columns=["ticker", "date", "adj_close_eur", "adj_open_eur"],
    )
    result = run_backtest(
        constant_scores(prices), prices, top_k=2, hold=10, capital=100.0, costs=NO_COSTS
    )
    fills = result.trades.set_index("ticker")
    assert fills.at["A", "date"] == days[1]
    assert fills.at["B", "date"] == days[2] and fills.at["B", "price_eur"] == 9.0
    # Between A's fill and B's, B's budget is still cash.
    assert result.equity["cash_eur"].tolist() == [100.0, 50.0, 0.0, 0.0]


def test_next_open_needs_an_open_column() -> None:
    prices = make_prices(US).drop(columns="adj_open_eur")
    with pytest.raises(KeyError, match="adj_open_eur"):
        run_backtest(constant_scores(prices), prices, top_k=1, hold=1, costs=NO_COSTS)
    with pytest.raises(ValueError, match="fill"):
        run_backtest(constant_scores(prices), prices, top_k=1, hold=1, fill="vwap")  # type: ignore[arg-type]


# --- the cost model ------------------------------------------------------------


def test_one_way_cost_is_commission_plus_half_spread_plus_slippage() -> None:
    model = CostModel(commission_bps=3, spread_bps=8, slippage_bps=2, liquidity_scaling=False)
    assert model.one_way_bps() == pytest.approx(3 + 4 + 2)
    assert model.rate() == pytest.approx(9e-4)
    assert CostModel.flat(50).one_way_bps(1.0) == pytest.approx(50)


def test_spread_and_slippage_scale_with_inverse_liquidity() -> None:
    model = CostModel(
        commission_bps=5,
        spread_bps=10,
        slippage_bps=5,
        liquidity_ref_eur=100e6,
        min_liquidity_scale=1,
        max_liquidity_scale=10,
    )
    adv = np.array([400e6, 100e6, 50e6, 25e6, 1e6, np.nan, 0.0])
    # variable part is 10bps at the reference volume, commission never scales.
    np.testing.assert_allclose(model.one_way_bps(adv), [15, 15, 25, 45, 105, 105, 105])
    # A floor below 1 lets the most liquid names pay less than the quote.
    cheap = model.model_copy(update={"min_liquidity_scale": 0.5})
    assert cheap.one_way_bps(400e6) == pytest.approx(5 + 10 * 0.5)


def test_the_thin_name_pays_more_in_a_backtest() -> None:
    days = NYSE[:3]
    prices = pd.DataFrame(
        [("BIG", d, 10.0, 10.0, 200e6) for d in days]
        + [("THIN", d, 10.0, 10.0, 20e6) for d in days],
        columns=["ticker", "date", "adj_close_eur", "adj_open_eur", "adv_eur_21"],
    )
    result = run_backtest(
        constant_scores(prices), prices, top_k=2, hold=10, capital=2000.0, costs=CostModel()
    )
    cost = result.trades.set_index("ticker")
    rate = cost["cost_eur"] / cost["value_eur"]
    assert rate["BIG"] == pytest.approx(15e-4)  # scale floored at 1
    assert rate["THIN"] == pytest.approx(55e-4)  # 5 + 10 * 5

    with pytest.raises(KeyError, match="adv_eur_21"):
        run_backtest(
            constant_scores(prices),
            prices.drop(columns="adv_eur_21"),
            top_k=2,
            hold=10,
            costs=CostModel(),
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"commission_bps": -1},
        {"spread_bps": -0.1},
        {"liquidity_ref_eur": 0},
        {"min_liquidity_scale": 5, "max_liquidity_scale": 2},
        {"rebate_bps": 1},
    ],
)
def test_bad_cost_settings_are_refused(bad) -> None:
    with pytest.raises(ValidationError):
        CostModel(**bad)


def test_every_number_comes_from_the_settings_file(write_settings) -> None:
    values = {
        "commission_bps": 1.5,
        "spread_bps": 7.0,
        "slippage_bps": 2.5,
        "liquidity_scaling": False,
        "liquidity_ref_eur": 5e7,
        "min_liquidity_scale": 0.5,
        "max_liquidity_scale": 4.0,
    }
    settings = load_settings(write_settings({"costs": values}))
    assert CostModel.from_settings(settings).model_dump() == values
    # The project's own file carries a costs section, and it validates.
    assert load_settings().costs == CostModel.from_settings()


# --- reading fill prices from the store -----------------------------------------


def test_adjusted_open_shares_the_close_adjustment() -> None:
    frame = pd.DataFrame({"open": [100.0], "close": [110.0], "adj_close_eur": [55.0]})
    assert adjusted_open(frame).iloc[0] == pytest.approx(50.0)


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dt.date]:
    """One XNYS stock, 60 sessions, opening 1% under each close, EUR at 0.9."""
    data = tmp_path / "data"
    days = NYSE[:60]
    close = 100 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.01, 60))
    bars = pd.DataFrame(
        {
            "ticker": "LIQ",
            "date": days,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1e6,
            "close_eur": close * 0.9,
        }
    )
    write_bars(bars, root=data / "bars" / "daily")
    company = {"exchange": "XNYS", "currency": "USD", "sector": "Industrials", "country": "X"}
    paths = {}
    for name, row in (("universe", {"ticker": "LIQ", "name": "L"}), ("benchmarks", None)):
        path = tmp_path / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
            writer.writeheader()
            if row:
                writer.writerow({**row, **company})
            else:
                writer.writerow({"ticker": "BENCH", "name": "B", **company, "sector": "Benchmark"})
        paths[name] = path
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data),
                "universe_path": str(paths["universe"]),
                "benchmarks_path": str(paths["benchmarks"]),
                "benchmark": "BENCH",
                "log_dir": str(tmp_path / "logs"),
                "history_start": "2018-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return days


def test_read_backtest_prices_gives_the_engine_everything(store: list[dt.date]) -> None:
    out = read_backtest_prices(as_of=store[-1], start=store[10])
    assert tuple(out.columns) == BACKTEST_PRICE_COLUMNS
    assert out["date"].min() == store[10] and out["date"].max() == store[-1]
    # No split or dividend: the adjusted EUR open is the open at 0.9.
    np.testing.assert_allclose(out["adj_open_eur"] / out["adj_close_eur"], 0.99)
    # ADV is known from the 21st session on: 1e6 shares at the EUR close.
    known = out.dropna(subset=["adv_eur_21"])
    assert known["date"].min() == store[20]
    # And it runs straight through the engine with the default costs.
    result = run_backtest(constant_scores(out), out, top_k=1, hold=5, capital=1000.0)
    assert len(result.trades) == 1 and result.total_cost > 0


def test_read_backtest_prices_refuses_the_future(store: list[dt.date]) -> None:
    out = read_backtest_prices(as_of=store[30])
    assert out["date"].max() == store[30]
