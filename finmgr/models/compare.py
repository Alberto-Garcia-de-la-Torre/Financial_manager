"""Day 42: a model's backtest next to day 38's baselines, on the very same inputs.

    from finmgr.models.compare import compare, run_baselines_for

    baselines = run_baselines_for(inputs, start=..., top_k=10, hold=21, costs=costs)
    comp = compare(model_run, baselines, random=random_sharpes(start=..., end=...))
    comp.bar.sharpe, comp.vs_bar, comp.beats_bar

The named baselines — benchmark, both equal-weight books, 12-1 momentum — are
re-run here through :func:`finmgr.backtest.strategy.run_strategy` on the same
:class:`~finmgr.backtest.strategy.Inputs`, window, top-K, holding period and
costs as the model, so every number in the comparison comes from one engine on
one date axis. The 200 random books are not re-run (minutes of compute, and
they are luck, not a strategy): their Sharpes are read from
`docs/baselines_random.csv`, written by `tools/baselines.py`, and used only
when that file's window is the run's window.

**The bar** is :func:`finmgr.backtest.baselines.the_bar` over those
candidates: day 38's rule, unchanged. A model *beats* it only if its Sharpe is
higher and the paired block-bootstrap interval on Sharpe(model) - Sharpe(bar)
lies above zero (day 39). When the random 95th percentile sets the bar there is
no single return series to pair with, so the difference is taken against the
best named baseline instead.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from finmgr.backtest import baselines as bl
from finmgr.backtest.bootstrap import SharpeInterval, bootstrap_sharpe_difference
from finmgr.backtest.costs import CostModel
from finmgr.backtest.strategy import Inputs, StrategyRun, run_strategy
from finmgr.config import PROJECT_ROOT

#: The CLI strategies re-run beside every model, in the order the tables list them.
BASELINE_STRATEGIES: tuple[str, ...] = (
    "equal-weight-rebalanced",
    "equal-weight",
    "momentum",
    "benchmark",
)

RANDOM_FILE = PROJECT_ROOT / "docs" / "baselines_random.csv"


def run_baselines_for(
    inputs: Inputs,
    *,
    start: date | str,
    top_k: int,
    hold: int,
    costs: CostModel,
) -> dict[str, StrategyRun]:
    """Day 38's named baselines on `inputs`, keyed by their label."""
    runs = {}
    for strategy in BASELINE_STRATEGIES:
        if strategy == "momentum" and inputs.momentum is None:
            continue
        run = run_strategy(strategy, inputs, start=start, top_k=top_k, hold=hold, costs=costs)
        runs[run.label] = run
    return runs


def random_sharpes(path: Path = RANDOM_FILE, *, start: date | str, end: date | str) -> list[float]:
    """Sharpes of day 38's random books, if `path` holds runs over exactly `start`..`end`.

    `start` is the first date of the run's curve (its metrics' `start`). An
    empty list when the file is missing or covers another window: a percentile
    from another window is not a bar for this one.
    """
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if frame.empty or not {"start", "end", "sharpe"} <= set(frame.columns):
        return []
    same = (frame["start"].astype(str) == str(start)) & (frame["end"].astype(str) == str(end))
    if not same.all():
        return []
    return [float(v) for v in frame["sharpe"] if not math.isnan(v)]


@dataclass(frozen=True)
class Comparison:
    """A model run, the baselines beside it, and whether it clears the bar."""

    model: StrategyRun
    baselines: dict[str, StrategyRun]
    random: list[float]
    bar: bl.Bar
    #: The baseline the paired difference is taken against: the bar, or the best named one.
    reference: str
    #: Sharpe(model) - Sharpe(reference), paired blocks.
    vs_reference: SharpeInterval
    #: Sharpe(model) - Sharpe(each named baseline), paired blocks.
    vs_named: dict[str, SharpeInterval]

    @property
    def beats_bar(self) -> bool:
        """Above the bar's point *and* the paired interval above zero."""
        return self.model.metrics.sharpe > self.bar.sharpe and self.vs_reference.low > 0

    @property
    def verdict(self) -> str:
        sharpe, bar = self.model.metrics.sharpe, self.bar.sharpe
        d = self.vs_reference
        if self.beats_bar:
            return f"beats the bar: Sharpe {sharpe:.2f} > {bar:.2f}, paired interval above zero"
        if sharpe > bar:
            return (
                f"above the bar's point ({sharpe:.2f} > {bar:.2f}) but not distinguishable "
                f"from it: paired interval [{d.low:+.2f}, {d.high:+.2f}] includes zero"
            )
        return f"does not beat the bar: Sharpe {sharpe:.2f} ≤ {bar:.2f}"


def compare(
    model: StrategyRun, baselines: dict[str, StrategyRun], random: list[float] | None = None
) -> Comparison:
    """Put `model` next to `baselines`; every run must share the model's date axis."""
    if not baselines:
        raise ValueError("nothing to compare against: no baselines")
    axis = list(model.result.curve.index)
    for name, run in baselines.items():
        if list(run.result.curve.index) != axis:
            raise ValueError(f"baseline {name!r} ran on a different date axis from the model")
    draws = list(random or [])
    bar = bl.the_bar({name: run.metrics.sharpe for name, run in baselines.items()}, draws)
    named = {name: run.metrics.sharpe for name, run in baselines.items()}
    reference = bar.source if bar.source in baselines else max(named, key=named.__getitem__)
    periods = model.periods
    vs_named = {
        name: bootstrap_sharpe_difference(model.excess, run.excess, periods=periods)
        for name, run in baselines.items()
    }
    return Comparison(
        model=model,
        baselines=baselines,
        random=draws,
        bar=bar,
        reference=reference,
        vs_reference=vs_named[reference],
        vs_named=vs_named,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _rows(c: Comparison) -> list[tuple[str, str, float, float, float, float, float, str]]:
    """(name, role, sharpe, cagr, vol, max_dd, turnover, interval text), Sharpe descending."""
    out = []
    runs = {c.model.label: c.model, **c.baselines}
    for name, run in runs.items():
        m, iv = run.metrics, run.interval
        role = "model" if run is c.model else ("bar" if name == c.bar.source else "baseline")
        out.append(
            (
                name,
                role,
                m.sharpe,
                m.cagr,
                m.volatility,
                m.max_drawdown,
                m.turnover,
                f"[{iv.low:.2f}, {iv.high:.2f}]",
            )
        )
    if c.random:
        label = next(k for k in c.bar.candidates if k.startswith(bl.RANDOM))
        role = "bar" if label == c.bar.source else "baseline"
        p95 = c.bar.candidates[label]
        nan = math.nan
        out.append((label, role, p95, nan, nan, nan, nan, f"{len(c.random)} runs"))
    return sorted(out, key=lambda r: -r[2])


def _f(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if kind == "pct":
        return f"{v:.2%}"
    return f"{v:.2f}"


def comparison_markdown(c: Comparison) -> str:
    """The comparison as Markdown tables, for `docs/models/*.md` and the terminal."""
    lines = [
        "| Candidate | Sharpe | 95% interval | CAGR | Volatility | Max drawdown "
        "| Turnover (x/yr) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, role, sharpe, cagr, vol, mdd, turn, iv in _rows(c):
        tag = " **← model**" if role == "model" else (" **← the bar**" if role == "bar" else "")
        lines.append(
            f"| {name}{tag} | {_f(sharpe, 'num')} | {iv} | {_f(cagr, 'pct')} | "
            f"{_f(vol, 'pct')} | {_f(mdd, 'pct')} | {_f(turn, 'num')} |"
        )
    lines += [
        "",
        "| Paired difference | Sharpe | 95% low | 95% high | P(diff ≤ 0) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, d in c.vs_named.items():
        lines.append(
            f"| {c.model.label} minus {name} | {d.point:+.2f} | {d.low:+.2f} | "
            f"{d.high:+.2f} | {d.p_le_zero:.1%} |"
        )
    return "\n".join(lines) + "\n"


def comparison_html(c: Comparison) -> str:
    """The comparison as an HTML fragment for the tearsheet's "Against the baselines"."""
    head = (
        "<tr><th>Candidate</th><th>Sharpe</th><th>95% interval</th><th>CAGR</th>"
        "<th>Volatility</th><th>Max drawdown</th><th>Turnover (x/yr)</th></tr>"
    )
    body = []
    for name, role, sharpe, cagr, vol, mdd, turn, iv in _rows(c):
        tag = {"model": " ← model", "bar": " ← the bar"}.get(role, "")
        style = ' style="font-weight:600"' if role != "baseline" else ""
        body.append(
            f"<tr{style}><th>{html.escape(name + tag)}</th><td>{_f(sharpe, 'num')}</td>"
            f"<td>{html.escape(iv)}</td><td>{_f(cagr, 'pct')}</td><td>{_f(vol, 'pct')}</td>"
            f"<td>{_f(mdd, 'pct')}</td><td>{_f(turn, 'num')}</td></tr>"
        )
    diffs = "".join(
        f"<tr><th>minus {html.escape(name)}</th><td>{d.point:+.2f}</td>"
        f"<td>[{d.low:+.2f}, {d.high:+.2f}]</td><td>{d.p_le_zero:.1%}</td></tr>"
        for name, d in c.vs_named.items()
    )
    return (
        f'<p class="note"><b>{html.escape(c.verdict)}.</b></p>'
        f'<table class="metrics"><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>'
        '<p class="note">Paired block bootstrap of Sharpe(model) - Sharpe(baseline): both '
        "curves resampled on the same blocks of dates, so what they share cancels.</p>"
        '<table class="metrics"><thead><tr><th>Sharpe difference</th><th>point</th>'
        "<th>95% interval</th><th>P(diff ≤ 0)</th></tr></thead>"
        f"<tbody>{diffs}</tbody></table>"
    )
