"""Day 38: the baselines every model has to beat, and the Sharpe bar they set.

    .venv/bin/python tools/baselines.py --as-of 2026-11-12   # writes docs/baselines.md

Every baseline goes through the same engine, on the same date axis, with the
same costs and next-open fills a model will get, and is scored with the same
metrics against the same risk-free series:

* **benchmark** — the configured benchmark (MSCI World, `IWDA.AS`) bought on
  the first day and held: one name, constant score, `top_k=1`.
* **equal weight, buy-and-hold** — every name alive on the first day, equal
  euros each, never traded again (constant score, `top_k` = the universe).
* **equal weight, rebalanced** — every name with a price, reset to 1/N every
  `hold` sessions (`reweight=True`); late listings join at the next rebalance.
* **random top-K** — `n_random` runs whose score is uniform noise drawn from
  `numpy.random.default_rng(seed)`, seeds `0..n_random-1`: K names picked at
  random, held `hold` sessions, with the same retention rule as any score
  (a held name that is drawn again is kept, not re-bought).
* **12-1 momentum, top-K** — day 24's `mom_12_1` as the score: the K names
  with the best return from t-252 to t-21, held `hold` sessions.

**The bar** is the highest Sharpe among the benchmark, both equal-weight
books, 12-1 momentum and the 95th percentile of the random runs. A model has
to beat all of them on the same splits and costs to be worth keeping: below
the benchmark it is worse than doing nothing, below equal weight it is worse
than not choosing, below the random 95th percentile it is indistinguishable
from luck, and below momentum it adds nothing to a one-line rule.

**Annualisation.** The date axis is the union of every venue's sessions —
every weekday, about 260 a year — so the √periods factor uses the dates the
axis actually has per year (:func:`axis_periods`), not 252. All baselines
share the axis, so the choice moves every Sharpe by the same factor.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from finmgr.backtest.costs import CostModel
from finmgr.backtest.engine import BacktestResult, constant_scores, run_backtest
from finmgr.backtest.metrics import Metrics, backtest_metrics, year_fraction

#: What a model will be: top 10, held a month.
TOP_K = 10
HOLD = 21
#: How many random-pick portfolios, seeds 0..N_RANDOM-1.
N_RANDOM = 200
#: The random runs' percentile that counts towards the bar.
RANDOM_QUANTILE = 0.95
#: The momentum signal: day 24's 12-1.
MOMENTUM_COLUMN = "mom_12_1"

BENCHMARK = "benchmark"
EQUAL_WEIGHT = "equal weight, buy-and-hold"
EQUAL_WEIGHT_REBALANCED = "equal weight, rebalanced"
MOMENTUM = "12-1 momentum, top-K"
RANDOM = "random top-K"


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def random_scores(prices: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Uniform noise on every `(ticker, date)` in `prices`; the same seed gives the same draw."""
    keys = prices[["ticker", "date"]].sort_values(["ticker", "date"], kind="stable")
    out = keys.reset_index(drop=True)
    out["score"] = np.random.default_rng(seed).random(len(out))
    return out


def momentum_scores(momentum: pd.DataFrame, column: str = MOMENTUM_COLUMN) -> pd.DataFrame:
    """`ticker, date, score` from a momentum frame; rows without a value are dropped."""
    out = momentum[["ticker", "date", column]].rename(columns={column: "score"})
    return out.dropna(subset=["score"]).reset_index(drop=True)


def axis_periods(dates: Sequence[date]) -> int:
    """Dates per year on an axis, rounded: the annualisation factor for its daily returns."""
    days = sorted(dates)
    years = year_fraction(days[0], days[-1])
    if years <= 0:
        raise ValueError("the axis must span more than one day")
    return round((len(days) - 1) / years)


# ---------------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """The Sharpe a model must beat, and which baseline set it."""

    sharpe: float
    source: str
    #: Every candidate that was considered, name -> Sharpe.
    candidates: dict[str, float]


def the_bar(
    named: Mapping[str, float], random: Iterable[float], quantile: float = RANDOM_QUANTILE
) -> Bar:
    """The highest of the named Sharpes and the `quantile` of the random ones."""
    draws = np.asarray([s for s in random if not math.isnan(s)], dtype=float)
    candidates = {k: float(v) for k, v in named.items() if not math.isnan(v)}
    if len(draws):
        label = f"{RANDOM}, {round(100 * quantile)}th percentile"
        candidates[label] = float(np.quantile(draws, quantile))
    if not candidates:
        raise ValueError("no baseline has a Sharpe")
    source = max(candidates, key=candidates.__getitem__)
    return Bar(sharpe=candidates[source], source=source, candidates=candidates)


# ---------------------------------------------------------------------------
# Running them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Baselines:
    """Everything :func:`run_baselines` measured."""

    #: Named baseline -> its metrics.
    named: dict[str, Metrics]
    #: One row per random seed: `seed` and every :class:`Metrics` field.
    random: pd.DataFrame
    bar: Bar
    top_k: int
    hold: int
    periods: int
    costs: CostModel
    #: Named baseline -> its full engine result, for plots and tearsheets.
    results: dict[str, BacktestResult] = field(default_factory=dict, repr=False)


# Worker state for the random runs: set once per process, not pickled per task.
_SHARED: dict[str, object] = {}


def _init_random(prices: pd.DataFrame, rf: pd.Series | None, kwargs: dict, periods: int) -> None:
    _SHARED.update(prices=prices, rf=rf, kwargs=kwargs, periods=periods)


def _one_random(seed: int) -> dict[str, object]:
    prices = _SHARED["prices"]
    result = run_backtest(random_scores(prices, seed), prices, **_SHARED["kwargs"])
    metrics = backtest_metrics(result, rf=_SHARED["rf"], periods=_SHARED["periods"])
    return {"seed": seed, **metrics.as_dict()}


def run_baselines(
    prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    momentum: pd.DataFrame,
    *,
    rf: pd.Series | None,
    start: date | str,
    end: date | str,
    top_k: int = TOP_K,
    hold: int = HOLD,
    costs: CostModel | None = None,
    n_random: int = N_RANDOM,
    workers: int = 1,
) -> Baselines:
    """Run every baseline on one axis and return their metrics and the bar.

    `prices` is the universe and `benchmark_prices` the benchmark (one ticker),
    both as :func:`~finmgr.backtest.costs.read_backtest_prices` returns them;
    `momentum` has `ticker, date` and :data:`MOMENTUM_COLUMN`. `rf` is the
    risk-free return into each date (`None` for zero). Every run uses the
    same `start`/`end`, `costs` and next-open fills; the benchmark is run on
    the universe's date axis so every curve has the same dates.
    """
    model = costs if costs is not None else CostModel()
    tickers = benchmark_prices["ticker"].unique()
    if len(tickers) != 1:
        raise ValueError(f"benchmark_prices must hold one ticker, got {len(tickers)}")
    if set(tickers) & set(prices["ticker"]):
        raise ValueError(f"benchmark {tickers[0]} is also in the universe")
    common = {"costs": model, "start": start, "end": end}

    # Benchmark on the universe's axis: the other names are there, unscored.
    both = pd.concat([prices, benchmark_prices], ignore_index=True)
    bench = run_backtest(constant_scores(benchmark_prices), both, top_k=1, hold=hold, **common)
    axis = list(bench.curve.index)
    periods = axis_periods(axis)
    n_names = prices["ticker"].nunique()

    results = {
        BENCHMARK: bench,
        EQUAL_WEIGHT: run_backtest(
            constant_scores(prices), prices, top_k=n_names, hold=hold, **common
        ),
        EQUAL_WEIGHT_REBALANCED: run_backtest(
            constant_scores(prices), prices, top_k=n_names, hold=hold, reweight=True, **common
        ),
        MOMENTUM: run_backtest(momentum_scores(momentum), prices, top_k=top_k, hold=hold, **common),
    }
    for name, result in results.items():
        if list(result.curve.index) != axis:
            raise AssertionError(f"{name} ran on a different date axis")
    named = {name: backtest_metrics(r, rf=rf, periods=periods) for name, r in results.items()}

    kwargs = dict(top_k=top_k, hold=hold, **common)
    seeds = range(n_random)
    if workers > 1 and n_random > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_random, initargs=(prices, rf, kwargs, periods)
        ) as pool:
            rows = list(pool.map(_one_random, seeds))
    else:
        _init_random(prices, rf, kwargs, periods)
        rows = [_one_random(seed) for seed in seeds]
    _SHARED.clear()
    random = pd.DataFrame(rows, columns=["seed", *Metrics.__dataclass_fields__])

    bar = the_bar({k: m.sharpe for k, m in named.items()}, random["sharpe"])
    return Baselines(
        named=named,
        random=random,
        bar=bar,
        top_k=top_k,
        hold=hold,
        periods=periods,
        costs=model,
        results=results,
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

#: `(label, Metrics field, format)` for the summary table.
TABLE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("CAGR", "cagr", "pct"),
    ("Volatility", "volatility", "pct"),
    ("**Sharpe**", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Longest drawdown (sessions)", "longest_drawdown_sessions", "int"),
    ("Hit rate", "hit_rate", "pct"),
    ("Avg win / avg loss", "payoff", "num"),
    ("Turnover (x/yr)", "turnover", "num"),
    ("Avg holding (sessions)", "avg_holding", "int"),
    ("Total return", "total_return", "pct"),
)


def _fmt(value: object, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    v = float(value)
    if kind == "pct":
        return f"{100 * v:.2f}%"
    if kind == "int":
        return f"{v:,.0f}"
    return f"{v:.2f}"


def summary_table(baselines: Baselines) -> str:
    """The named baselines and the random runs' 5th/50th/95th percentiles, as markdown."""
    columns: dict[str, dict[str, object]] = {
        name: m.as_dict() for name, m in baselines.named.items()
    }
    for q in (0.05, 0.5, 0.95):
        columns[f"random p{round(100 * q)}"] = baselines.random.quantile(
            q, numeric_only=True
        ).to_dict()
    names = list(columns)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for label, key, kind in TABLE_ROWS:
        cells = [_fmt(columns[n].get(key), kind) for n in names]
        if key == "sharpe":
            cells = [f"**{c}**" for c in cells]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def random_distribution(baselines: Baselines, column: str = "sharpe", kind: str = "num") -> str:
    """Percentiles of one metric over the random runs, as a one-row markdown table."""
    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    values = baselines.random[column].quantile(list(qs))
    head = ["min", "p5", "p25", "median", "p75", "p95", "max"]
    return "\n".join(
        [
            "| " + " | ".join(head) + " |",
            "|" + "---:|" * len(head),
            "| " + " | ".join(_fmt(v, kind) for v in values) + " |",
        ]
    )
