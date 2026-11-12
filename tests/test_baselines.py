"""Day 38: the baselines and the bar.

The real numbers live in `docs/baselines.md` (they need the local store); these
tests pin the machinery on a small synthetic universe: random scores are
reproducible per seed and differ across seeds, momentum scores are day 24's
column, every baseline runs on one shared axis, the benchmark run is exactly
the benchmark's own buy-and-hold, and the bar is the highest candidate with the
random runs counted at their 95th percentile.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from finmgr.backtest import baselines as bl
from finmgr.backtest.costs import NO_COSTS, CostModel
from finmgr.features.momentum import compute_momentum

DAYS = list(pd.bdate_range("2021-01-01", "2023-06-30").date)


def make_prices(tickers: list[str], seed: int = 38) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, ticker in enumerate(tickers):
        drift = 0.0002 * (i - len(tickers) / 2)
        close = 50 * np.exp(np.cumsum(drift + 0.012 * rng.standard_normal(len(DAYS))))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": DAYS,
                    "adj_close_eur": close,
                    "adj_open_eur": close * (1 + 0.002 * rng.standard_normal(len(DAYS))),
                    "adv_eur_21": 5e7,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


UNIVERSE = [f"T{i:02d}" for i in range(12)]


@pytest.fixture(scope="module")
def run() -> bl.Baselines:
    prices = make_prices(UNIVERSE)
    bench = make_prices(["BENCH"], seed=1)
    momentum = compute_momentum(prices, price="adj_close_eur")
    return bl.run_baselines(
        prices,
        bench,
        momentum,
        rf=None,
        start=dt.date(2022, 1, 3),
        end=dt.date(2023, 6, 30),
        top_k=3,
        hold=21,
        costs=CostModel.flat(10),
        n_random=12,
    )


def test_random_scores_are_reproducible_and_seed_dependent():
    prices = make_prices(UNIVERSE[:3])
    a, b, c = (bl.random_scores(prices, s) for s in (7, 7, 8))
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["score"], c["score"])
    assert len(a) == len(prices)
    assert a["score"].between(0, 1).all()


def test_momentum_scores_are_the_12_1_column():
    prices = make_prices(UNIVERSE[:2])
    momentum = compute_momentum(prices, price="adj_close_eur")
    scores = bl.momentum_scores(momentum)
    assert list(scores.columns) == ["ticker", "date", "score"]
    assert scores["score"].notna().all()
    merged = scores.merge(momentum, on=["ticker", "date"])
    assert (merged["score"] == merged["mom_12_1"]).all()
    # 12-1 needs 252 sessions of history: nothing before that.
    assert len(scores) == len(prices) - 2 * 252


def test_axis_periods_counts_dates_per_year():
    # 31 Dec 2020 to 31 Dec 2021: exactly one year, 261 weekday returns.
    year = list(pd.bdate_range("2020-12-31", "2021-12-31").date)
    assert bl.axis_periods(year) == 261
    assert bl.axis_periods(year[:131]) == 261  # half the year, same density
    with pytest.raises(ValueError):
        bl.axis_periods([dt.date(2021, 1, 4)])


def test_the_bar_is_the_highest_candidate():
    bar = bl.the_bar({"a": 0.5, "b": 0.9, "c": math.nan}, np.linspace(0, 1, 101))
    assert bar.sharpe == pytest.approx(0.95)
    assert bar.source.startswith(bl.RANDOM)
    assert "c" not in bar.candidates
    bar = bl.the_bar({"a": 0.5, "b": 1.2}, [0.1, 0.2])
    assert (bar.source, bar.sharpe) == ("b", 1.2)
    with pytest.raises(ValueError):
        bl.the_bar({}, [])


def test_every_baseline_runs_on_one_axis(run):
    assert set(run.named) == {
        bl.BENCHMARK,
        bl.EQUAL_WEIGHT,
        bl.EQUAL_WEIGHT_REBALANCED,
        bl.MOMENTUM,
    }
    axes = {tuple(r.curve.index) for r in run.results.values()}
    assert len(axes) == 1
    assert run.periods == 261
    assert len(run.random) == 12 and list(run.random["seed"]) == list(range(12))
    assert run.random["sharpe"].nunique() == 12  # different seeds, different books
    # The bar is above or at every candidate, including each named baseline.
    for metrics in run.named.values():
        assert run.bar.sharpe >= metrics.sharpe
    assert run.bar.sharpe >= run.random["sharpe"].quantile(0.95) - 1e-12


def test_baselines_hold_what_they_claim(run):
    bench = run.results[bl.BENCHMARK].positions
    assert set(bench["ticker"]) == {"BENCH"}
    held = run.results[bl.EQUAL_WEIGHT].positions.groupby("date")["ticker"].nunique()
    assert (held.iloc[1:] == len(UNIVERSE)).all()  # filled at the first next open
    assert run.named[bl.EQUAL_WEIGHT].turnover == 0
    assert run.named[bl.EQUAL_WEIGHT_REBALANCED].turnover > 0
    mom = run.results[bl.MOMENTUM].positions.groupby("date")["ticker"].nunique()
    assert mom.max() == 3


def test_benchmark_run_is_its_own_buy_and_hold():
    """Free fills at the close: the curve is capital * p[t] / p[0], to rounding."""
    prices = make_prices(UNIVERSE[:4])
    bench = make_prices(["BENCH"], seed=1)
    momentum = compute_momentum(prices, price="adj_close_eur")
    out = bl.run_baselines(
        prices,
        bench,
        momentum,
        rf=None,
        start=dt.date(2022, 1, 3),
        end=dt.date(2022, 12, 30),
        top_k=2,
        hold=21,
        costs=NO_COSTS,
        n_random=1,
    )
    curve = out.results[bl.BENCHMARK].curve
    px = bench.set_index("date")["adj_close_eur"]
    # Next-open fill on the second date, then marked at the close.
    first_fill = bench.set_index("date")["adj_open_eur"].loc[curve.index[1]]
    expected = 100_000 * px.loc[curve.index[1:]] / first_fill
    np.testing.assert_allclose(curve.iloc[1:].to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_rejects_a_benchmark_inside_the_universe():
    prices = make_prices(UNIVERSE[:2])
    with pytest.raises(ValueError, match="also in the universe"):
        bl.run_baselines(
            prices,
            prices[prices["ticker"] == "T00"],
            compute_momentum(prices, price="adj_close_eur"),
            rf=None,
            start=dt.date(2022, 1, 3),
            end=dt.date(2022, 6, 30),
            n_random=0,
        )


def test_tables_render(run):
    table = bl.summary_table(run)
    assert "**Sharpe**" in table and "random p95" in table
    assert table.count("\n") == len(bl.TABLE_ROWS) + 1
    assert bl.random_distribution(run).count("|") == 3 * 8
