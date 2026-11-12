# Metrics (day 37)

`finmgr/backtest/metrics.py` turns an equity curve, plus the backtest's
trades and positions, into a single `Metrics` row: CAGR, annualised
volatility, Sharpe and Sortino against the risk-free series, max drawdown
with its peak, trough, recovery and duration, hit rate, average win and
average loss, one-way turnover, and average holding period. The module
docstring defines each one; the conventions that matter are:

| Metric | Convention |
|---|---|
| Years | Counted by anniversaries: 31 Dec → 31 Dec is exactly 1.0, leap year or not |
| Volatility, Sharpe, Sortino | Daily simple returns, ×√252, ddof=1 |
| Sharpe, Sortino | Use the excess over `risk_free_returns` (€STR, spliced with EONIA); Sortino's target is that rate, averaged over every session |
| Max drawdown duration | Peak to the first session back at the peak, in sessions and in calendar days; runs to the end if not recovered |
| Hit rate, win, loss | Per session, with flat sessions excluded |
| Turnover | (buys + sells) / 2 / mean equity / years; the initial deployment of cash is not turnover |
| Holding period | Mean length, in sessions, of each continuous spell a name is held; a spell still open at the end is censored |

```python
from finmgr.backtest.metrics import backtest_metrics
from finmgr.data.rates import risk_free_returns

m = backtest_metrics(result, rf=risk_free_returns(result.curve.index, as_of=...))
```

## Acceptance: SPY buy-and-hold against a published reference

`tests/test_metrics.py` runs SPY's dividend-adjusted close as a one-name
buy-and-hold through the day-35 engine (constant score, no costs). The
frozen input is `tests/fixtures/metrics/spy_adj_close.csv`, regenerated with
`tools/spy_reference.py`. The test then compares the curve's calendar-year
returns with the figures [TotalRealReturns](https://totalrealreturns.com/n/SPY)
publishes (nominal, dividends reinvested, read 2026-11-12):

| Year | Published % | Ours % | Diff (pp) |
|---|---:|---:|---:|
| 2013 | 32.31 | 32.3078 | −0.0022 |
| 2014 | 13.46 | 13.4638 | +0.0038 |
| 2015 | 1.23 | 1.2343 | +0.0043 |
| 2016 | 12.00 | 11.9979 | −0.0021 |
| 2017 | 21.71 | 21.7054 | −0.0046 |
| 2018 | −4.57 | −4.5690 | +0.0010 |
| 2019 | 31.22 | 31.2239 | +0.0039 |
| 2020 | 18.33 | 18.3316 | +0.0016 |
| 2021 | 28.73 | 28.7288 | −0.0012 |
| 2022 | −18.18 | −18.1754 | +0.0046 |
| 2023 | 26.18 | 26.1758 | −0.0042 |
| 2024 | 24.89 | 24.8865 | −0.0035 |

Every year rounds to the published two decimals: each difference is under
half a unit (0.005 pp). The test also checks two CAGR figures against the same
reference:

* **One-year CAGR.** Over the six years whose year-ends are both 31 Dec
  (2013–15 and 2019–21), CAGR equals the published annual return. When a year
  ends on the 29th or 30th the window is shorter than a year, so CAGR differs
  from the calendar-year return, as it should.
* **Ten-year CAGR.** CAGR from 2014-12-31 to 2024-12-31 is 13.00%/yr, which
  matches the ten compounded published years.

[ETFreplay](https://www.etfreplay.com/etf/spy) prints different second
decimals for some years, e.g. 2020 at +18.37. Those are SPY's NAV returns. A
curve built from exchange closes is a market-price curve and cannot match
them; a test pins down that difference.

No public source prints a daily Sharpe, Sortino or hit rate for SPY. Those
metrics, plus drawdown, turnover and holding period, are checked by hand on
small curves. The drawdown is also checked on the 2020 crash: the peak is
19 Feb 2020 and the trough 23 Mar 2020, the dates reported for the S&P 500.

## Benchmark buy-and-hold, full metric set (local store, as of 2026-11-12)

These figures are for information; nothing tests them. Sharpe and Sortino
are against €STR. That makes them exact for the EUR rows. For the USD row
they are only indicative, because SPY in USD should be measured against a
USD rate.

| | SPY, USD | SPY, EUR | IWDA.AS, EUR |
|---|---:|---:|---:|
| Window | 2012-12-31 → 2024-12-31 | 2014-12-31 → 2024-12-31 | 2014-12-31 → 2024-12-31 |
| CAGR | 14.53% | 14.77% | 11.73% |
| Volatility | 16.71% | 19.18% | 15.89% |
| Sharpe (vs €STR) | 0.88 | 0.79 | 0.74 |
| Sortino | 1.22 | 1.12 | 1.03 |
| Max drawdown | −33.72% | −33.09% | −33.63% |
| Peak → trough | 2020-02-19 → 03-23 | 2020-02-19 → 03-23 | 2020-02-19 → 03-23 |
| Recovered | 2020-08-10 (120 sessions) | 2021-01-08 (225 sessions) | 2021-01-07 (226 sessions) |
| Longest drawdown | 489 sessions | 238 sessions | 436 sessions |
| Hit rate | 55.4% | 54.0% | 55.6% |
| Avg win / avg loss | +0.68% / −0.72% | +0.83% / −0.84% | +0.67% / −0.73% |
| Turnover | 0 | 0 | 0 |
| Avg holding | whole window | whole window | whole window |
