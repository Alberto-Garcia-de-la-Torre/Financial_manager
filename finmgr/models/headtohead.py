"""Day 43: two (or more) models side by side, on provably identical inputs.

    from finmgr.models.headtohead import head_to_head

    h = head_to_head(["ridge", "lgbm"], data, inputs, start=..., top_k=10, hold=21, costs=costs)
    h.checks            # every fairness check, each with what it compared
    h.pairs[0].sharpe   # Sharpe(lgbm) - Sharpe(ridge), paired block bootstrap
    h.comparisons       # each model against day 38's baselines, as on its own page

A comparison between two models is only worth reading if the *only* thing that
differs between them is the model. This module runs every model through
:func:`~finmgr.models.walkforward.walk_forward` on one
:class:`~finmgr.models.walkforward.Dataset` object (one feature matrix, one
label, one fill rule), then through
:func:`~finmgr.backtest.strategy.run_scores` on one
:class:`~finmgr.backtest.strategy.Inputs` with one window, top-K, holding
period and :class:`~finmgr.backtest.costs.CostModel`; the baselines are run
once and shared. Doing it in one process is what makes it fair by
construction.

It is then also *checked*, not just asserted in a docstring:
:func:`fairness_checks` compares the runs' features, label, fold boundaries
(train and test dates of every fold), the `(ticker, date)` rows each model
scored, the backtest's date axis, and the window, top-K, hold and cost
parameters recorded in each run. :func:`head_to_head` raises
:class:`UnfairComparison` if any of them differ, so an unfair page cannot be
written.

The models are then compared with each other directly, not only each against
the bar:

* **Sharpe difference**: the paired block bootstrap of day 39 on the two
  models' daily excess returns. Both books trade the same universe on the
  same dates, so most of their variance is shared and cancels.
* **IC difference**: per test date, IC(challenger) - IC(incumbent). Its mean, and
  a t-stat that divides the naive one by √horizon because the 21-session
  labels overlap.
* **Per fold**: which model had the higher mean IC in each refit window.
* **Score agreement**: the per-date Spearman correlation between the two
  models' scores, and how much of the top-K they picked in common: whether
  they are the same bet or different ones.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from finmgr.backtest.bootstrap import SharpeInterval, bootstrap_sharpe_difference
from finmgr.backtest.costs import CostModel
from finmgr.backtest.metrics import calendar_returns
from finmgr.backtest.strategy import Inputs, StrategyRun, run_scores
from finmgr.models.compare import Comparison, compare, run_baselines_for
from finmgr.models.walkforward import Dataset, WalkForward, walk_forward


class UnfairComparison(ValueError):
    """The runs differ in something other than the model; the comparison is refused."""


@dataclass(frozen=True)
class Check:
    """One thing that has to be identical across the models, and whether it was."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Pair:
    """A challenger against the incumbent (the first model named)."""

    challenger: str
    incumbent: str
    #: Sharpe(challenger) - Sharpe(incumbent), paired blocks.
    sharpe: SharpeInterval
    #: Per test date: IC(challenger) - IC(incumbent), on dates where both are known.
    ic_diff: pd.Series
    #: Per fold: (test_start, mean IC incumbent, mean IC challenger).
    folds: pd.DataFrame
    #: Per date: Spearman correlation of the two models' scores.
    score_corr: pd.Series
    #: Per date: share of the incumbent's top K that the challenger also ranks in its top K.
    top_overlap: pd.Series

    def ic_summary(self, horizon: int) -> dict[str, float]:
        """Mean IC difference, its naive t, and the t deflated by √horizon for overlap."""
        d = self.ic_diff.dropna()
        n = len(d)
        mean = float(d.mean()) if n else math.nan
        std = float(d.std(ddof=1)) if n > 1 else math.nan
        t = mean / std * math.sqrt(n) if std and std > 0 else math.nan
        return {
            "dates": float(n),
            "mean": mean,
            "t_naive": t,
            "t": t / math.sqrt(horizon),
            "positive": float((d > 0).mean()) if n else math.nan,
        }

    @property
    def fold_wins(self) -> int:
        f = self.folds.dropna()
        return int((f["challenger"] > f["incumbent"]).sum())


@dataclass(frozen=True)
class HeadToHead:
    """Every model's walk-forward, backtest and baseline comparison, plus the pairings."""

    models: list[str]
    walkforwards: dict[str, WalkForward]
    runs: dict[str, StrategyRun]
    baselines: dict[str, StrategyRun]
    comparisons: dict[str, Comparison]
    checks: list[Check]
    pairs: list[Pair]

    @property
    def incumbent(self) -> str:
        return self.models[0]

    def yearly(self) -> pd.DataFrame:
        """Calendar-year return of every model and of the bar, one row per year."""
        cols = {m: calendar_returns(self.runs[m].result.curve) for m in self.models}
        bar = self.comparisons[self.incumbent].reference
        cols[bar] = calendar_returns(self.baselines[bar].result.curve)
        return pd.DataFrame(cols)


def _describe_folds(wf: WalkForward) -> list[tuple[date, date, date, date]]:
    return [(f.train_start, f.train_end, f.test_start, f.test_end) for f in wf.folds]


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[["ticker", "date"]]).sort_values()


def _same(values: Mapping[str, Any]) -> bool:
    items = list(values.values())
    return all(v == items[0] for v in items[1:])


def fairness_checks(
    walkforwards: Mapping[str, WalkForward], runs: Mapping[str, StrategyRun]
) -> list[Check]:
    """Everything that must not differ between the models' runs, compared one by one."""
    names = list(walkforwards)
    first = names[0]
    checks: list[Check] = []

    def add(name: str, values: Mapping[str, Any], shown: str) -> None:
        ok = _same(values)
        detail = shown if ok else "; ".join(f"{k}: {v}" for k, v in values.items())
        checks.append(Check(name, ok, detail))

    wf0 = walkforwards[first]
    add(
        "features",
        {m: wf.features for m, wf in walkforwards.items()},
        f"{len(wf0.features)} per-date ranks, same order",
    )
    add(
        "label and horizon",
        {m: (wf.label, wf.horizon) for m, wf in walkforwards.items()},
        f"{wf0.label}, {wf0.horizon} sessions",
    )
    add(
        "split settings",
        {
            m: (wf.test_size, wf.min_train_size, wf.train_size, wf.horizon)
            for m, wf in walkforwards.items()
        },
        f"test {wf0.test_size}, min train {wf0.min_train_size}, "
        f"{'expanding' if wf0.train_size is None else f'rolling {wf0.train_size}'}, "
        f"embargo {wf0.horizon}",
    )
    folds = {m: _describe_folds(wf) for m, wf in walkforwards.items()}
    add(
        "fold boundaries",
        folds,
        f"{len(folds[first])} folds, train and test dates identical in every one",
    )
    add(
        "rows fitted per fold",
        {m: [f.n_train for f in wf.folds] for m, wf in walkforwards.items()},
        f"{sum(f.n_train for f in wf0.folds):,} training rows over all folds",
    )
    keys = {m: _keys(wf.scores) for m, wf in walkforwards.items()}
    k0 = keys[first]
    same_rows = all(k.equals(k0) for k in keys.values())
    checks.append(
        Check(
            "scored rows",
            same_rows,
            f"{len(k0):,} (ticker, date) predictions each"
            if same_rows
            else "; ".join(f"{m}: {len(k):,}" for m, k in keys.items()),
        )
    )
    axes = {m: run.result.curve.index for m, run in runs.items()}
    a0 = axes[first]
    same_axis = all(a.equals(a0) for a in axes.values())
    checks.append(
        Check(
            "backtest date axis",
            same_axis,
            f"{a0[0]} → {a0[-1]}, {len(a0):,} sessions"
            if same_axis
            else "; ".join(f"{m}: {a[0]} → {a[-1]}" for m, a in axes.items()),
        )
    )
    r0 = runs[first]
    add(
        "window, top-K, hold",
        {m: (str(r.start), r.top_k, r.hold, r.as_of) for m, r in runs.items()},
        f"from {r0.start}, top {r0.top_k}, held {r0.hold}, data as of {r0.as_of}",
    )
    add(
        "costs",
        {m: r.costs.model_dump() for m, r in runs.items()},
        f"{r0.costs.commission_bps:g} bps commission, {r0.costs.spread_bps:g} bps spread, "
        f"{r0.costs.slippage_bps:g} bps slippage"
        + (", liquidity-scaled" if r0.costs.liquidity_scaling else ""),
    )
    add(
        "benchmark and risk-free",
        {m: (r.benchmark_ticker, r.rf_used) for m, r in runs.items()},
        f"{r0.benchmark_ticker}, Sharpe against " + ("€STR/EONIA" if r0.rf_used else "zero"),
    )
    return checks


def _top_overlap(a: pd.DataFrame, b: pd.DataFrame, k: int) -> pd.Series:
    """Per date: |top k of a ∩ top k of b| / k (ties broken by ticker, as stable sorts)."""

    def tops(frame: pd.DataFrame) -> pd.Series:
        ranked = frame.sort_values(["date", "score", "ticker"], ascending=[True, False, True])
        return ranked.groupby("date").head(k).groupby("date")["ticker"].agg(frozenset)

    ta, tb = tops(a), tops(b)
    common = ta.index.intersection(tb.index)
    return pd.Series(
        [len(ta[d] & tb[d]) / min(k, len(ta[d])) for d in common], index=common, name="overlap"
    )


def _score_corr(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    both = a.merge(b, on=["ticker", "date"], suffixes=("_a", "_b"))
    ranks = both.groupby("date")[["score_a", "score_b"]].rank()
    ranks["date"] = both["date"]

    def corr(g: pd.DataFrame) -> float:
        if len(g) < 3 or g["score_a"].nunique() < 2 or g["score_b"].nunique() < 2:
            return math.nan
        return float(np.corrcoef(g["score_a"], g["score_b"])[0, 1])

    out = ranks.groupby("date")[["score_a", "score_b"]].apply(corr)
    out.name = "score_corr"
    return out


def _fold_ic(wf: WalkForward) -> pd.Series:
    ic = wf.ic
    means = {}
    for f in wf.folds:
        window = ic[(ic.index >= f.test_start) & (ic.index <= f.test_end)].dropna()
        means[f.test_start] = float(window.mean()) if len(window) else math.nan
    return pd.Series(means)


def pair(
    incumbent: str,
    challenger: str,
    walkforwards: Mapping[str, WalkForward],
    runs: Mapping[str, StrategyRun],
    *,
    top_k: int,
) -> Pair:
    a, b = walkforwards[incumbent], walkforwards[challenger]
    ra, rb = runs[incumbent], runs[challenger]
    ic = pd.concat([a.ic.rename("a"), b.ic.rename("b")], axis=1).dropna()
    folds = pd.DataFrame({"incumbent": _fold_ic(a), "challenger": _fold_ic(b)})
    folds.index.name = "test_start"
    return Pair(
        challenger=challenger,
        incumbent=incumbent,
        sharpe=bootstrap_sharpe_difference(rb.excess, ra.excess, periods=rb.periods),
        ic_diff=(ic["b"] - ic["a"]).rename("ic_diff"),
        folds=folds,
        score_corr=_score_corr(a.scores, b.scores),
        top_overlap=_top_overlap(a.scores, b.scores, top_k),
    )


def head_to_head(
    models: Sequence[str],
    data: Dataset,
    inputs: Inputs,
    *,
    start: date | str,
    top_k: int,
    hold: int,
    costs: CostModel,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    test_size: int,
    min_train_size: int,
    train_size: int | None = None,
    random: Sequence[float] | None = None,
    random_for: Any = None,
) -> HeadToHead:
    """Train, backtest and compare `models` on one dataset, one input set and one cost model.

    The first model is the incumbent every other one is paired against.
    `params` maps a model name to its hyperparameter overrides. `random_for`,
    if given, is called with a run and returns the random-book Sharpes for its
    window (see :func:`finmgr.models.compare.random_sharpes`); `random` is used
    as is otherwise. Raises :class:`UnfairComparison` if any fairness check fails.
    """
    names = list(dict.fromkeys(models))
    if len(names) < 2:
        raise ValueError("a head-to-head needs at least two different models")
    overrides = dict(params or {})
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(f"parameters given for model(s) not compared: {', '.join(unknown)}")
    walkforwards: dict[str, WalkForward] = {}
    runs: dict[str, StrategyRun] = {}
    for name in names:
        wf = walk_forward(
            name,
            data,
            params=overrides.get(name),
            test_size=test_size,
            min_train_size=min_train_size,
            train_size=train_size,
        )
        walkforwards[name] = wf
        runs[name] = run_scores(
            wf.scores,
            inputs,
            name=f"{name}, walk-forward",
            params=wf.variant_params(),
            start=start,
            top_k=top_k,
            hold=hold,
            costs=costs,
        )
    checks = fairness_checks(walkforwards, runs)
    failed = [c for c in checks if not c.passed]
    if failed:
        raise UnfairComparison(
            "the runs differ in more than the model: "
            + "; ".join(f"{c.name} ({c.detail})" for c in failed)
        )
    baselines = run_baselines_for(inputs, start=start, top_k=top_k, hold=hold, costs=costs)
    first = runs[names[0]]
    draws = list(random_for(first) if random_for is not None else (random or []))
    comparisons = {name: compare(runs[name], baselines, draws) for name in names}
    pairs = [pair(names[0], name, walkforwards, runs, top_k=top_k) for name in names[1:]]
    return HeadToHead(
        models=names,
        walkforwards=walkforwards,
        runs=runs,
        baselines=baselines,
        comparisons=comparisons,
        checks=checks,
        pairs=pairs,
    )
