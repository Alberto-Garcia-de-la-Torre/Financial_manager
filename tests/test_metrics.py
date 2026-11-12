"""Day 37: performance metrics.

The acceptance criterion: metrics for a benchmark buy-and-hold match a
published reference within rounding. The benchmark is SPY. Its daily
dividend-adjusted closes, 2012-12-31 to 2024-12-31, are frozen in
`tests/fixtures/metrics/spy_adj_close.csv` (`tools/spy_reference.py`). The
test feeds them through the backtest engine as a one-name buy-and-hold and
compares the engine's equity curve, via `finmgr.backtest.metrics`, with SPY's
calendar-year total returns as TotalRealReturns publishes them
(https://totalrealreturns.com/n/SPY, nominal, dividends reinvested, two
decimals, read 2026-11-12). "Within rounding" means within half a unit of the
last published digit: |ours - published| <= 0.005 percentage points.

The cross-check: ETFreplay (https://www.etfreplay.com/etf/spy) prints
different second decimals for 2015, 2017, 2018, 2020, 2021, 2022 and 2023,
e.g. 2020 +18.37 against +18.33. Those are SPY's NAV returns, which State
Street reports beside the market-price ones. A curve built from exchange
closes is a market-price curve, so TotalRealReturns is the reference that
applies. `PUBLISHED_NAV` records ETFreplay's figures, and a test pins down
that they differ from ours by more than rounding.

No public source prints a *daily* Sharpe, Sortino or hit rate for SPY, so
those are checked by hand on curves small enough to work out on paper. The
same goes for drawdown, turnover and holding period, which also run on
engine output.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finmgr.backtest import metrics as m
from finmgr.backtest.costs import NO_COSTS
from finmgr.backtest.engine import constant_scores, run_backtest

D = dt.date
FIXTURE = Path(__file__).parent / "fixtures" / "metrics" / "spy_adj_close.csv"

#: SPY calendar-year total return, %, as published by TotalRealReturns
#: (market price, dividends reinvested), read 2026-11-12.
PUBLISHED = {
    2013: 32.31,
    2014: 13.46,
    2015: 1.23,
    2016: 12.00,
    2017: 21.71,
    2018: -4.57,
    2019: 31.22,
    2020: 18.33,
    2021: 28.73,
    2022: -18.18,
    2023: 26.18,
    2024: 24.89,
}
#: The same years as ETFreplay prints them: SPY's NAV returns, not market price.
PUBLISHED_NAV = {2020: 18.37, 2021: 28.75, 2022: -18.17, 2023: 26.19}
#: Half a unit in the last published digit, in percentage points.
ROUNDING = 0.005 + 1e-9
#: First date of the hand-worked curves.
START = D(2024, 1, 1)


# ---------------------------------------------------------------------------
# The acceptance criterion: SPY buy-and-hold against a published reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spy_buy_and_hold():
    frame = pd.read_csv(FIXTURE, comment="#")
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["ticker"] = "SPY"
    result = run_backtest(
        constant_scores(frame),
        frame,
        top_k=1,
        hold=21,
        costs=NO_COSTS,
        fill="close",
        price="adj_close",
    )
    return frame, result


def test_buy_and_hold_calendar_returns_match_published(spy_buy_and_hold):
    _, result = spy_buy_and_hold
    ours = m.calendar_returns(result.curve) * 100
    assert list(ours.index) == list(PUBLISHED)
    for year, published in PUBLISHED.items():
        assert abs(ours[year] - published) <= ROUNDING, (year, ours[year], published)


def test_one_year_cagr_is_the_published_annual_return(spy_buy_and_hold):
    # CAGR is per year of calendar time, a calendar-year return per year-end
    # *session*. They coincide when both year-ends are 31 December, as in
    # 2013-15 and 2019-21; when a year ends on the 29th or 30th the window is
    # not a whole year and CAGR rightly differs.
    _, result = spy_buy_and_hold
    curve = result.curve
    ends = {d.year: d for d in curve.index}  # last session of each year
    checked = []
    for year in PUBLISHED:
        start, end = ends[year - 1], ends[year]
        if (start.month, start.day, end.month, end.day) != (12, 31, 12, 31):
            continue
        got = m.compute_metrics(curve[(curve.index >= start) & (curve.index <= end)], rf=None)
        assert got.years == 1.0
        assert abs(got.cagr * 100 - PUBLISHED[year]) <= ROUNDING, year
        checked.append(year)
    assert checked == [2013, 2014, 2015, 2019, 2020, 2021]


def test_ten_year_cagr_matches_the_compounded_published_years(spy_buy_and_hold):
    _, result = spy_buy_and_hold
    curve = result.curve
    window = curve[curve.index >= D(2014, 12, 31)]
    got = m.compute_metrics(window, rf=None, trades=result.trades, positions=result.positions)
    assert got.years == 10.0
    # Trimmed to the window, the 2012 purchase is outside it: still no turnover.
    assert got.turnover == 0.0
    assert got.avg_holding == len(window)
    growth = np.prod([1 + PUBLISHED[y] / 100 for y in range(2015, 2025)])
    published_cagr = (growth ** (1 / 10) - 1) * 100
    # Each published year is off by at most half a unit; so is their geometric mean.
    assert abs(got.cagr * 100 - published_cagr) <= ROUNDING
    assert round(got.cagr * 100, 2) == 13.00


def test_published_nav_returns_are_not_what_a_close_curve_measures(spy_buy_and_hold):
    _, result = spy_buy_and_hold
    ours = m.calendar_returns(result.curve) * 100
    assert any(abs(ours[y] - nav) > ROUNDING for y, nav in PUBLISHED_NAV.items())


def test_buy_and_hold_covid_drawdown_peak_and_trough(spy_buy_and_hold):
    _, result = spy_buy_and_hold
    curve = result.curve
    got = m.compute_metrics(curve[[d for d in curve.index if d.year == 2020]], rf=None)
    # The 2020 crash: the record close of 19 Feb to the low of 23 Mar, as
    # reported everywhere for the S&P 500 the fund tracks.
    assert (got.max_drawdown_peak, got.max_drawdown_trough) == (D(2020, 2, 19), D(2020, 3, 23))
    # Dividends make the total-return fall a little shallower than the
    # S&P 500 price index's -33.9%.
    assert -0.345 < got.max_drawdown < -0.33
    assert got.max_drawdown_recovered and got.max_drawdown_recovery < D(2020, 8, 18)


def test_buy_and_hold_never_trades_after_the_first_day(spy_buy_and_hold):
    frame, result = spy_buy_and_hold
    got = m.backtest_metrics(result, rf=None)
    assert got.turnover == 0.0
    # One spell, the whole axis.
    assert got.avg_holding == len(frame)
    assert got.start == D(2012, 12, 31) and got.end == D(2024, 12, 31)
    assert got.sessions == len(frame) - 1
    assert 0.5 < got.hit_rate < 0.6  # a sanity range, not a reference
    assert got.avg_loss < 0 < got.avg_win


# ---------------------------------------------------------------------------
# By hand
# ---------------------------------------------------------------------------


def _from_returns(returns, start=START):
    return _curve(100.0 * np.cumprod([1.0, *[1 + r for r in returns]]), start)


def _curve(values, start=START):
    days = [start + dt.timedelta(days=i) for i in range(len(values))]
    return pd.Series(values, index=pd.Index(days, name="date"), dtype="float64")


def test_year_fraction():
    assert m.year_fraction(D(2019, 12, 31), D(2020, 12, 31)) == 1.0
    assert m.year_fraction(D(2014, 12, 31), D(2024, 12, 31)) == 10.0
    assert m.year_fraction(D(2023, 1, 1), D(2023, 7, 2)) == pytest.approx(182 / 365)
    assert m.year_fraction(D(2024, 1, 1), D(2024, 7, 1)) == pytest.approx(182 / 366)
    # 29 Feb has no anniversary: the year back from 28 Feb 2021 starts on the 28th.
    assert m.year_fraction(D(2020, 2, 29), D(2021, 2, 28)) == pytest.approx(365 / 366)
    assert m.year_fraction(D(2024, 3, 3), D(2024, 3, 3)) == 0.0
    with pytest.raises(ValueError):
        m.year_fraction(D(2024, 3, 3), D(2024, 3, 2))


def test_cagr_and_total_return():
    curve = pd.Series(
        [100.0, 150.0, 121.0], index=[D(2020, 12, 31), D(2021, 6, 30), D(2022, 12, 31)]
    )
    assert m.total_return(curve) == pytest.approx(0.21)
    assert m.cagr(curve) == pytest.approx(0.10)


def test_calendar_returns_chain_from_each_year_end():
    curve = pd.Series(
        [100.0, 110.0, 120.0, 90.0],
        index=[D(2020, 12, 31), D(2021, 5, 1), D(2021, 12, 31), D(2022, 3, 1)],
    )
    got = m.calendar_returns(curve)
    assert got.to_dict() == pytest.approx({2021: 0.2, 2022: -0.25})


def test_volatility_sharpe_sortino_by_hand():
    # Returns +2%, -1%, +2%, -1%: mean 0.5%, sample sd = sqrt(4 * 0.015^2 / 3).
    curve = _from_returns([0.02, -0.01, 0.02, -0.01])
    r = m.session_returns(curve)
    assert r.to_numpy() == pytest.approx([0.02, -0.01, 0.02, -0.01])
    sd = math.sqrt(4 * 0.015**2 / 3)
    got = m.compute_metrics(curve, rf=None)
    assert got.volatility == pytest.approx(sd * math.sqrt(252))
    assert got.sharpe == pytest.approx(0.005 / sd * math.sqrt(252))
    # Downside deviation over all four sessions: sqrt((0.01^2 + 0.01^2) / 4).
    assert got.sortino == pytest.approx(0.005 / math.sqrt(0.0002 / 4) * math.sqrt(252))
    assert got.rf == "zero"


def test_sharpe_is_against_the_risk_free_series():
    curve = _from_returns([0.02, -0.01, 0.02, -0.01])
    days = list(curve.index)
    rf = pd.Series([0.0, 0.001, 0.001, 0.001, 0.001], index=days)
    got = m.compute_metrics(curve, rf=rf)
    excess = np.array([0.019, -0.011, 0.019, -0.011])
    assert got.sharpe == pytest.approx(excess.mean() / excess.std(ddof=1) * math.sqrt(252))
    downside = math.sqrt((0.011**2 * 2) / 4)
    assert got.sortino == pytest.approx(excess.mean() / downside * math.sqrt(252))
    assert got.rf == "risk-free series"
    # A session with no rate leaves the ratio undefined rather than guessed.
    gappy = rf.copy()
    gappy.iloc[2] = np.nan
    assert math.isnan(m.compute_metrics(curve, rf=gappy).sharpe)
    # Timestamps index works as well as dates.
    stamped = rf.copy()
    stamped.index = pd.to_datetime(stamped.index)
    assert m.compute_metrics(curve, rf=stamped).sharpe == pytest.approx(got.sharpe)


def test_max_drawdown_and_its_duration():
    #           0     1     2    3    4     5     6    7    8
    values = [100, 120, 90, 60, 100, 120, 130, 117, 125]
    got = m.compute_metrics(_curve(values), rf=None)
    assert got.max_drawdown == pytest.approx(-0.5)
    assert got.max_drawdown_peak == D(2024, 1, 2)
    assert got.max_drawdown_trough == D(2024, 1, 4)
    assert got.max_drawdown_recovery == D(2024, 1, 6)  # back to 120 exactly
    assert got.max_drawdown_sessions == 4 and got.max_drawdown_days == 4
    # The second episode (130 → 117) is unrecovered at the end, 2 sessions long.
    episodes = m.drawdowns(_curve(values))
    assert len(episodes) == 2
    assert not episodes[1].recovered and episodes[1].sessions == 2
    assert episodes[1].depth == pytest.approx(-0.1)
    assert got.longest_drawdown_sessions == 4


def test_unrecovered_max_drawdown_runs_to_the_end_and_a_rising_curve_has_none():
    got = m.compute_metrics(_curve([100, 110, 80, 90, 100]), rf=None)
    assert got.max_drawdown_recovery is None and not got.max_drawdown_recovered
    assert got.max_drawdown_sessions == 3
    flat = m.compute_metrics(_curve([100, 101, 102]), rf=None)
    assert flat.max_drawdown == 0.0 and flat.max_drawdown_sessions == 0
    np.testing.assert_allclose(m.drawdown_series(_curve([100, 110, 99])), [0, 0, -0.1])


def test_hit_rate_and_win_loss():
    # +10%, 0 (flat: ignored), -5%, +2%, -5%
    curve = _from_returns([0.10, 0.0, -0.05, 0.02, -0.05])
    got = m.compute_metrics(curve, rf=None)
    assert got.hit_rate == pytest.approx(0.5)
    assert got.avg_win == pytest.approx(0.06)
    assert got.avg_loss == pytest.approx(-0.05)
    assert got.payoff == pytest.approx(1.2)


def _two_names(days=12):
    dates = [D(2024, 1, 1) + dt.timedelta(days=i) for i in range(days)]
    rows = []
    for i, d in enumerate(dates):
        rows.append(("A", d, 100.0 * 1.01**i))
        rows.append(("B", d, 50.0))
    return dates, pd.DataFrame(rows, columns=["ticker", "date", "adj_close_eur"])


def test_turnover_and_holding_on_a_rotation():
    # Top-1, hold 3: the leader flips every rebalance, so the whole book turns over.
    dates, prices = _two_names()
    flip = [(t, d, float((i // 3) % 2 == (t == "A"))) for i, d in enumerate(dates) for t in "AB"]
    scores = pd.DataFrame(flip, columns=["ticker", "date", "score"])
    result = run_backtest(scores, prices, top_k=1, hold=3, costs=NO_COSTS, fill="close")
    got = m.backtest_metrics(result, rf=None)
    # Four holding spells of 3 sessions each.
    assert m.holding_spells(result.positions, dates).tolist() == [3, 3, 3, 3]
    assert got.avg_holding == 3.0
    # Rebalances at 3, 6, 9 each sell the book and buy it back: 2 * equity
    # traded, i.e. one full one-way turn each.
    trades = result.trades[result.trades["decided"] != dates[0]]
    curve = result.curve
    expected = trades["value_eur"].sum() / 2 / curve.mean() / m.year_fraction(dates[0], dates[-1])
    assert got.turnover == pytest.approx(expected)
    turns = trades["value_eur"].sum() / 2 / curve.mean()
    assert turns == pytest.approx(3.0, rel=0.05)


def test_initial_deployment_is_not_turnover():
    dates, prices = _two_names()
    prices["adj_open_eur"] = prices["adj_close_eur"]
    result = run_backtest(constant_scores(prices), prices, top_k=2, hold=5, costs=NO_COSTS)
    got = m.backtest_metrics(result, rf=None)
    assert len(result.trades) == 2
    assert got.turnover == 0.0
    # Next-open fills: bought on day 1, held to the end.
    assert got.avg_holding == len(dates) - 1


def test_bad_curves_are_rejected():
    with pytest.raises(ValueError):
        m.compute_metrics(_curve([100.0]), rf=None)
    with pytest.raises(ValueError):
        m.compute_metrics(_curve([100.0, 0.0]), rf=None)
    with pytest.raises(ValueError):
        m.compute_metrics(_curve([100.0, 101.0]).iloc[::-1], rf=None)
