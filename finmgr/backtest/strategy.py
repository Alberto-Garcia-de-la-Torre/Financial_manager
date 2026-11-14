"""Day 40: run one named strategy against the benchmark, the way `finmgr backtest` does.

    from finmgr.backtest.strategy import load_inputs, run_strategy

    inputs = load_inputs(as_of="2026-11-12")
    run = run_strategy("momentum", inputs)
    run.metrics.sharpe, run.interval, run.benchmark_metrics.sharpe

Every strategy here is one of day 38's baselines, run exactly as
:func:`finmgr.backtest.baselines.run_baselines` runs it — same universe prices,
same configured costs, next-open fills, same window start, and the benchmark
bought and held on the universe's own date axis — so the numbers a tearsheet
shows are the numbers in `docs/baselines.md`, not a second opinion on them.

A run is logged in the variant ledger (:mod:`finmgr.backtest.trials`) under the
same family and parameters as day 38's, so `finmgr backtest --strategy momentum`
at the default settings is a *rerun* of that variant and does not add to the
count; changing `--top-k`, `--hold` or `--start` is a new variant and does.
A single random seed is its own family: it is one draw, not the 95th
percentile of 200 that competes for the bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from finmgr.backtest import baselines as bl
from finmgr.backtest.bootstrap import (
    LEVEL,
    SharpeInterval,
    bootstrap_sharpe,
    bootstrap_sharpe_difference,
    familywise_level,
    result_excess,
)
from finmgr.backtest.costs import CostModel, read_backtest_prices
from finmgr.backtest.engine import BacktestResult, constant_scores, run_backtest
from finmgr.backtest.metrics import Metrics, backtest_metrics
from finmgr.backtest.trials import LEDGER, count_variants, record
from finmgr.config import Settings, load_settings
from finmgr.data import pit

#: Where day 38 starts every baseline: the first date with a full year of 12-1 momentum.
DEFAULT_START = "2011-03-01"
#: How far back to read prices: enough history for 12-1 momentum before DEFAULT_START.
HISTORY = "2010-01-01"

#: CLI name -> day 38 label. The label is also the ledger family (prefixed "baseline: ").
STRATEGIES: dict[str, str] = {
    "momentum": bl.MOMENTUM,
    "equal-weight": bl.EQUAL_WEIGHT,
    "equal-weight-rebalanced": bl.EQUAL_WEIGHT_REBALANCED,
    "benchmark": bl.BENCHMARK,
    "random": f"{bl.RANDOM}, one seed",
}


@dataclass(frozen=True)
class Inputs:
    """Everything a strategy run reads from the store, read once."""

    prices: pd.DataFrame  # the universe, as read_backtest_prices returns it
    benchmark_prices: pd.DataFrame  # one ticker
    momentum: pd.DataFrame | None  # ticker, date, mom_12_1; only momentum needs it
    rf: pd.Series | None  # risk-free return into each date; None for zero
    as_of: date
    end: date

    @property
    def benchmark_ticker(self) -> str:
        return str(self.benchmark_prices["ticker"].iloc[0])


def load_inputs(
    *,
    as_of: pit.AsOf,
    end: date | None = None,
    momentum: bool = True,
    settings: Settings | None = None,
) -> Inputs:
    """Read universe, benchmark, 12-1 momentum and the risk-free series as of `as_of`."""
    from finmgr.data.benchmarks import benchmarks_dir, primary_benchmark
    from finmgr.data.rates import risk_free_returns
    from finmgr.features.momentum import read_momentum

    settings = settings or load_settings()
    cutoff = pit.resolve_as_of(as_of)
    bench = primary_benchmark(settings=settings)
    prices = read_backtest_prices(as_of=cutoff, start=HISTORY, settings=settings)
    if prices.empty:
        raise ValueError("no universe prices in the store; run `finmgr ingest` first")
    bench_prices = read_backtest_prices(
        bench, as_of=cutoff, start=HISTORY, settings=settings, root=benchmarks_dir()
    )
    if bench_prices.empty:
        raise ValueError(f"no prices for benchmark {bench}; run `finmgr benchmarks` first")
    last = end or max(prices["date"])
    mom = read_momentum(as_of=cutoff, start=HISTORY, settings=settings) if momentum else None
    axis = sorted(d for d in set(prices["date"]) | set(bench_prices["date"]) if d <= last)
    rf = risk_free_returns(axis, as_of=cutoff, settings=settings)
    return Inputs(prices, bench_prices, mom, rf, as_of=cutoff, end=last)


@dataclass(frozen=True)
class StrategyRun:
    """One strategy and its benchmark over the same window, with every number a tearsheet needs."""

    strategy: str  # the CLI name
    label: str  # day 38's name for it
    result: BacktestResult
    benchmark: BacktestResult
    benchmark_ticker: str
    metrics: Metrics
    benchmark_metrics: Metrics
    excess: pd.Series  # daily excess returns the Sharpe is computed from
    benchmark_excess: pd.Series
    interval: SharpeInterval  # block-bootstrap interval on the strategy's Sharpe
    vs_benchmark: SharpeInterval  # paired interval on Sharpe(strategy) - Sharpe(benchmark)
    periods: int  # annualisation factor of the axis
    params: dict[str, object]  # what defines the variant in the ledger
    costs: CostModel
    top_k: int
    hold: int
    start: date | str
    as_of: date
    n_tickers: int
    rf_used: bool
    #: Ledger family prefix: "baseline" for day 38's strategies, "model" for learned ones.
    kind: str = "baseline"

    @property
    def family(self) -> str:
        return f"{self.kind}: {self.label}"


def strategy_scores(strategy: str, inputs: Inputs, *, seed: int = 0) -> pd.DataFrame:
    """The `ticker, date, score` frame a named strategy ranks on."""
    if strategy == "momentum":
        if inputs.momentum is None:
            raise ValueError("momentum needs the 12-1 momentum frame; load_inputs(momentum=True)")
        return bl.momentum_scores(inputs.momentum)
    if strategy in ("equal-weight", "equal-weight-rebalanced"):
        return constant_scores(inputs.prices)
    if strategy == "random":
        return bl.random_scores(inputs.prices, seed)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")


def _benchmark_run(inputs: Inputs, *, hold: int, common: dict) -> BacktestResult:
    """The benchmark bought and held on the union of its own and the universe's dates."""
    both = pd.concat([inputs.prices, inputs.benchmark_prices], ignore_index=True)
    return run_backtest(
        constant_scores(inputs.benchmark_prices), both, top_k=1, hold=hold, **common
    )


def _measure(
    *,
    strategy: str,
    label: str,
    result: BacktestResult,
    bench: BacktestResult,
    inputs: Inputs,
    params: dict[str, object],
    costs: CostModel,
    top_k: int,
    hold: int,
    start: date | str,
    kind: str = "baseline",
) -> StrategyRun:
    """Score `result` and the benchmark identically: metrics, excess returns, intervals."""
    axis = list(bench.curve.index)
    if list(result.curve.index) != axis:
        raise AssertionError(f"{strategy} ran on a different date axis from the benchmark")
    periods = bl.axis_periods(axis)
    excess = result_excess(result, inputs.rf)
    bench_excess = result_excess(bench, inputs.rf)
    return StrategyRun(
        strategy=strategy,
        label=label,
        result=result,
        benchmark=bench,
        benchmark_ticker=inputs.benchmark_ticker,
        metrics=backtest_metrics(result, rf=inputs.rf, periods=periods),
        benchmark_metrics=backtest_metrics(bench, rf=inputs.rf, periods=periods),
        excess=excess,
        benchmark_excess=bench_excess,
        interval=bootstrap_sharpe(excess, periods=periods),
        vs_benchmark=bootstrap_sharpe_difference(excess, bench_excess, periods=periods),
        periods=periods,
        params=params,
        costs=costs,
        top_k=top_k,
        hold=hold,
        start=start,
        as_of=inputs.as_of,
        n_tickers=inputs.prices["ticker"].nunique(),
        rf_used=inputs.rf is not None,
        kind=kind,
    )


def run_strategy(
    strategy: str,
    inputs: Inputs,
    *,
    start: date | str = DEFAULT_START,
    top_k: int = bl.TOP_K,
    hold: int = bl.HOLD,
    costs: CostModel | None = None,
    seed: int = 0,
) -> StrategyRun:
    """Backtest `strategy` and the benchmark on one axis, at the same costs, and measure both."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
    model = costs if costs is not None else CostModel()
    common = {"costs": model, "start": start, "end": inputs.end}
    prices = inputs.prices
    bench = _benchmark_run(inputs, hold=hold, common=common)
    n_names = prices["ticker"].nunique()

    if strategy == "benchmark":
        result = bench
    elif strategy in ("equal-weight", "equal-weight-rebalanced"):
        result = run_backtest(
            strategy_scores(strategy, inputs),
            prices,
            top_k=n_names,
            hold=hold,
            reweight=strategy == "equal-weight-rebalanced",
            **common,
        )
    else:
        result = run_backtest(
            strategy_scores(strategy, inputs, seed=seed), prices, top_k=top_k, hold=hold, **common
        )
    label = STRATEGIES[strategy]
    if strategy == "random":
        params: dict[str, object] = {
            "start": str(start),
            "hold": hold,
            "costs": model.model_dump(),
            "scores": "uniform noise",
            "top_k": top_k,
            "seed": seed,
        }
    else:
        params = bl.baseline_params(label, start=start, hold=hold, costs=model, top_k=top_k)
    return _measure(
        strategy=strategy,
        label=label,
        result=result,
        bench=bench,
        inputs=inputs,
        params=params,
        costs=model,
        top_k=1 if strategy == "benchmark" else (n_names if "equal" in strategy else top_k),
        hold=hold,
        start=start,
    )


def run_scores(
    scores: pd.DataFrame,
    inputs: Inputs,
    *,
    name: str,
    params: dict[str, object],
    start: date | str = DEFAULT_START,
    top_k: int = bl.TOP_K,
    hold: int = bl.HOLD,
    costs: CostModel | None = None,
    kind: str = "model",
) -> StrategyRun:
    """Backtest any `ticker, date, score` frame exactly as :func:`run_strategy` runs momentum.

    This is how a learned model meets the baselines: its out-of-sample scores
    go through the same engine, window, top-K, holding period, costs, fills
    and benchmark. `params` is what defines the variant in the ledger; the
    window, sizing and costs are added to it here so they cannot be forgotten.
    """
    model = costs if costs is not None else CostModel()
    common = {"costs": model, "start": start, "end": inputs.end}
    bench = _benchmark_run(inputs, hold=hold, common=common)
    result = run_backtest(scores, inputs.prices, top_k=top_k, hold=hold, **common)
    full = {
        **params,
        "start": str(start),
        "hold": hold,
        "top_k": top_k,
        "costs": model.model_dump(),
    }
    return _measure(
        strategy=name,
        label=name,
        result=result,
        bench=bench,
        inputs=inputs,
        params=full,
        costs=model,
        top_k=top_k,
        hold=hold,
        start=start,
        kind=kind,
    )


@dataclass(frozen=True)
class Logged:
    """What logging a run in the variant ledger returned."""

    variant_id: str
    n_variants: int
    familywise: SharpeInterval  # the Sharpe interval at the Bonferroni level for n_variants


def log_run(run: StrategyRun, *, path: Path = LEDGER, source: str = "finmgr backtest") -> Logged:
    """Record the run in the variant ledger and quote its Sharpe at the familywise level."""
    iv = run.interval
    vid = record(
        run.family,
        run.params,
        sharpe=run.metrics.sharpe,
        low=iv.low,
        high=iv.high,
        level=iv.level,
        source=source,
        path=path,
    )
    n = count_variants(path)
    familywise = bootstrap_sharpe(
        run.excess, periods=run.periods, level=familywise_level(LEVEL, max(n, 1))
    )
    return Logged(variant_id=vid, n_variants=n, familywise=familywise)
