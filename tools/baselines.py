"""Day 38: run the baselines on the local store and write docs/baselines.md.

    .venv/bin/python tools/baselines.py --as-of 2026-11-12 --workers 8

Runs :func:`finmgr.backtest.baselines.run_baselines` — benchmark buy-and-hold,
equal weight (held and rebalanced), 200 random top-10 portfolios and 12-1
momentum top-10, all at the configured costs with next-open fills — and writes
every number, plus the bar, to `docs/baselines.md` and the random runs to
`docs/baselines_random.csv`.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from finmgr.backtest import baselines as bl
from finmgr.backtest.bootstrap import (
    BLOCK,
    LEVEL,
    N_BOOT,
    SEED,
    SharpeInterval,
    bootstrap_sharpe,
    familywise_level,
)
from finmgr.backtest.costs import NO_COSTS, CostModel, read_backtest_prices
from finmgr.backtest.engine import run_backtest
from finmgr.backtest.metrics import backtest_metrics
from finmgr.backtest.trials import LEDGER
from finmgr.config import PROJECT_ROOT, load_settings
from finmgr.data import pit
from finmgr.data.benchmarks import benchmarks_dir, primary_benchmark
from finmgr.data.rates import risk_free_returns
from finmgr.features.momentum import read_momentum

DOCS = PROJECT_ROOT / "docs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--start", default="2011-03-01", help="first rebalance date")
    parser.add_argument("--end", default=None, help="last date (default: last universe price)")
    parser.add_argument("--top-k", type=int, default=bl.TOP_K)
    parser.add_argument("--hold", type=int, default=bl.HOLD)
    parser.add_argument("--n-random", type=int, default=bl.N_RANDOM)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=DOCS / "baselines.md")
    parser.add_argument(
        "--ledger", type=Path, default=LEDGER, help="variant ledger to log the candidates in"
    )
    args = parser.parse_args()

    settings = load_settings()
    cutoff = pit.resolve_as_of(args.as_of)
    bench = primary_benchmark(settings=settings)
    history = "2010-01-01"
    prices = read_backtest_prices(as_of=cutoff, start=history, settings=settings)
    bench_prices = read_backtest_prices(
        bench, as_of=cutoff, start=history, settings=settings, root=benchmarks_dir()
    )
    end = date.fromisoformat(args.end) if args.end else max(prices["date"])
    momentum = read_momentum(as_of=cutoff, start=history, settings=settings)
    axis = sorted(d for d in set(prices["date"]) | set(bench_prices["date"]) if d <= end)
    rf = risk_free_returns(axis, as_of=cutoff, settings=settings)
    costs = CostModel.from_settings(settings)

    t0 = time.time()
    result = bl.run_baselines(
        prices,
        bench_prices,
        momentum,
        rf=rf,
        start=args.start,
        end=end,
        top_k=args.top_k,
        hold=args.hold,
        costs=costs,
        n_random=args.n_random,
        workers=args.workers,
    )
    elapsed = time.time() - t0
    gross = gross_check(prices, momentum, rf, result, start=args.start, end=end)
    luck = Luck.measure(result, ledger=args.ledger)

    csv = args.out.with_name(args.out.stem + "_random.csv")
    result.random.round(6).to_csv(csv, index=False, lineterminator="\n")
    args.out.write_text(
        render(
            result,
            bench=bench,
            as_of=cutoff,
            n_tickers=prices["ticker"].nunique(),
            csv=csv,
            gross=gross,
            luck=luck,
            ledger=args.ledger,
        )
    )
    print(bl.summary_table(result))
    print(f"\nbar: Sharpe {result.bar.sharpe:.3f} ({result.bar.source}); {elapsed:.0f}s")
    print(
        f"bar {luck.bar:.0%} interval: {luck.bar_interval}; "
        f"{luck.n_variants} variants tested -> {luck.bar_familywise.level:.2%}: "
        f"{luck.bar_familywise}"
    )
    print(f"wrote {args.out} and {csv}")


#: Block lengths for the sensitivity row: i.i.d., a week, a month, a quarter.
BLOCKS = (1, 5, BLOCK, 63)


@dataclass(frozen=True)
class Luck:
    """Day 39: every bootstrap number the page quotes."""

    bar: float  # the nominal level, 0.95
    intervals: dict[str, SharpeInterval]
    differences: dict[str, SharpeInterval]
    by_block: dict[int, SharpeInterval]
    n_variants: int
    bar_familywise: SharpeInterval

    @property
    def bar_interval(self) -> SharpeInterval:
        return self.by_block[BLOCK]

    @classmethod
    def measure(cls, result: bl.Baselines, *, ledger: Path) -> Luck:
        intervals = bl.sharpe_intervals(result)
        n_variants = bl.record_trials(result, intervals, path=ledger)
        source = result.bar.source
        # Bars set by the random percentile have no one series; use the best named one.
        top = (
            source if source in result.excess else max(intervals, key=lambda k: intervals[k].point)
        )
        excess = result.excess[top]
        by_block = {b: bootstrap_sharpe(excess, periods=result.periods, block=b) for b in BLOCKS}
        familywise = bootstrap_sharpe(
            excess, periods=result.periods, level=familywise_level(LEVEL, n_variants)
        )
        return cls(
            bar=LEVEL,
            intervals=intervals,
            differences=bl.bar_differences(result),
            by_block=by_block,
            n_variants=n_variants,
            bar_familywise=familywise,
        )

    def render(self, result: bl.Baselines, ledger: Path) -> str:
        pct = f"{self.bar:.0%}"
        fw = self.bar_familywise
        rows = "\n".join(
            f"| {name} | {iv.point:.2f} | {iv.low:.2f} | {iv.high:.2f} | {iv.se:.2f} "
            f"| {iv.p_le_zero:.1%} |"
            for name, iv in sorted(self.intervals.items(), key=lambda kv: -kv[1].point)
        )
        blocks = "\n".join(
            f"| {b} | {iv.low:.2f} | {iv.high:.2f} | {iv.se:.2f} |"
            for b, iv in self.by_block.items()
        )
        diffs = "\n".join(
            f"| {result.bar.source} minus {name} | {iv.point:+.2f} | {iv.low:+.2f} "
            f"| {iv.high:+.2f} | {iv.p_le_zero:.1%} |"
            for name, iv in self.differences.items()
        )
        try:
            shown = ledger.relative_to(PROJECT_ROOT)
        except ValueError:
            shown = ledger
        return f"""## How lucky was that? (day 39)

Every Sharpe on this page is one draw. The intervals below come from a
circular block bootstrap of the daily excess returns
(`finmgr/backtest/bootstrap.py`): {N_BOOT:,} resamples, blocks of {BLOCK}
sessions (one holding period, so a rebalance cycle's autocorrelation and
volatility clustering survive the resampling), seed {SEED}, percentile
intervals. The point is exactly the Sharpe in the tables above.

### Sharpe as an interval ({pct})

| Baseline | Sharpe | {pct} low | {pct} high | Bootstrap s.e. | P(Sharpe ≤ 0) |
|---|---:|---:|---:|---:|---:|
{rows}

The random top-K books have no interval of their own here: their 200 runs
*are* the luck distribution, shown in the percentiles above.

### Does the bar actually beat the others?

Paired: both curves resampled on the same blocks of dates, so what they have
in common cancels. An interval that straddles zero means the data cannot tell
the two apart.

| Difference | Sharpe | {pct} low | {pct} high | P(diff ≤ 0) |
|---|---:|---:|---:|---:|
{diffs}

### Block length

The bar's {pct} interval for other block lengths (1 is plain i.i.d.
resampling). Longer blocks keep more dependence and usually widen it.

| Block (sessions) | {pct} low | {pct} high | s.e. |
|---:|---:|---:|---:|
{blocks}

### Variants tested so far: {self.n_variants}

`{shown}` counts every strategy variant ever backtested and compared,
one row per family + defining parameters (`finmgr/backtest/trials.py`).
Rerunning a variant updates its row; changing any parameter adds one, and
rows are never removed. The random runs count as one variant (the
percentile that competes), not 200.

The bar is the best of {self.n_variants} variants, so quoting its {pct} interval
alone would flatter it. Holding all {self.n_variants} intervals jointly at
{pct} (Bonferroni) means quoting each at **{fw.level:.2%}**:

> **Bar Sharpe {fw.point:.2f}, {fw.level:.2%} interval [{fw.low:.2f}, {fw.high:.2f}]**
> ({self.n_variants} variants tested).

Every model trained from here on adds to the count, and the familywise level
rises with it. Quote a model's Sharpe the same way: as an interval at the
familywise level for the count *at the time*, and with its paired difference
against the bar.
"""


def gross_check(prices, momentum, rf, result: bl.Baselines, *, start, end) -> str:
    """Momentum and random seed 0 again with free fills: how much of each gap is costs."""
    runs = {
        bl.MOMENTUM: bl.momentum_scores(momentum),
        f"{bl.RANDOM}, seed 0": bl.random_scores(prices, 0),
    }
    lines = [
        "| | CAGR net | CAGR gross | Sharpe net | Sharpe gross | Costs paid |",
        "|---|" + "---:|" * 5,
    ]
    for name, scores in runs.items():
        out = {}
        for label, costs in (("net", result.costs), ("gross", NO_COSTS)):
            run = run_backtest(
                scores,
                prices,
                top_k=result.top_k,
                hold=result.hold,
                costs=costs,
                start=start,
                end=end,
            )
            out[label] = (backtest_metrics(run, rf=rf, periods=result.periods), run.total_cost)
        (net, paid), (grs, _) = out["net"], out["gross"]
        lines.append(
            f"| {name} | {100 * net.cagr:.2f}% | {100 * grs.cagr:.2f}% | {net.sharpe:.2f} "
            f"| {grs.sharpe:.2f} | €{paid:,.0f} |"
        )
    return "\n".join(lines)


def render(
    result: bl.Baselines,
    *,
    bench: str,
    as_of: date,
    n_tickers: int,
    csv: Path,
    gross: str,
    luck: Luck,
    ledger: Path = LEDGER,
) -> str:
    any_metrics = next(iter(result.named.values()))
    c = result.costs
    beaten = (result.random["sharpe"] > result.bar.sharpe).sum()
    mom = result.named[bl.MOMENTUM].sharpe
    mom_rank = (result.random["sharpe"] < mom).mean()
    candidates = "\n".join(
        f"| {name}{' **← the bar**' if name == result.bar.source else ''} | {sharpe:.2f} |"
        for name, sharpe in sorted(result.bar.candidates.items(), key=lambda kv: -kv[1])
    )
    first, last = any_metrics.start, any_metrics.end
    iv = luck.bar_interval
    fw = luck.bar_familywise
    n_random = len(result.random)
    scale = f"[{c.min_liquidity_scale:g}, {c.max_liquidity_scale:g}]"
    setup = "\n".join(
        f"| {k} | {v} |"
        for k, v in [
            ("", ""),
            ("---", "---"),
            ("Universe", f"the {n_tickers} names in `config/universe.csv`, EUR total return"),
            ("Benchmark", f"{bench} (`benchmark` in settings), on the universe's date axis"),
            (
                "Window",
                f"{first} to {last}, {any_metrics.years:.2f} years, {any_metrics.sessions} dates",
            ),
            ("Capital", "EUR 100,000"),
            ("Selection", f"top {result.top_k}, equal weight, held {result.hold} sessions"),
            ("Fills", "next session's open, sells before buys"),
            (
                "Costs per side",
                f"{c.commission_bps:g}bps commission + ({c.spread_bps:g}/2 spread + "
                f"{c.slippage_bps:g} slippage)bps x liquidity scale in {scale}, "
                f"scale 1 at EUR {c.liquidity_ref_eur / 1e6:,.0f}M ADV",
            ),
            ("Risk-free", "EUR STR, spliced with EONIA before Oct 2019"),
            ("Annualisation", f"sqrt({result.periods}): the axis is every weekday"),
            (
                "Random runs",
                f"{n_random} seeds, 0..{n_random - 1}; per-seed rows in `docs/{csv.name}`",
            ),
        ]
    )
    return f"""# Baselines (days 38 and 39)

Generated by `tools/baselines.py --as-of {as_of}` from the local store; the
code is `finmgr/backtest/baselines.py`. Rerun it after any change to the
engine, costs, metrics or data, and update this page.

## The bar

> **Sharpe {iv.point:.2f}, {iv.level:.0%} interval [{iv.low:.2f}, {iv.high:.2f}]**;
> **[{fw.low:.2f}, {fw.high:.2f}] at {fw.level:.2%}**, the familywise level for the
> {luck.n_variants} variants tested so far (against €STR/EONIA, {any_metrics.start} →
> {any_metrics.end}, top {result.top_k}, held {result.hold} sessions, configured
> costs, next-open fills, block bootstrap of {iv.block} sessions). Set by:
> *{result.bar.source}*.
>
> A model is worth keeping only if, run through the same engine on the same
> window with the same costs, it beats {result.bar.sharpe:.2f} *and* the paired
> bootstrap interval on its Sharpe minus the bar's lies above zero. Clearing
> the point by a hair is not clearing it: the interval is how wide a hair is.

| Candidate | Sharpe |
|---|---:|
{candidates}

The bar is the highest of these: below the benchmark a model is worse than
doing nothing; below equal weight, worse than not choosing; below the random
95th percentile, indistinguishable from a lucky draw; below 12-1 momentum, it
adds nothing to a one-line rule.

## Setup

{setup}

## Every number

{bl.summary_table(result)}

Fractions are per year where that makes sense (CAGR, volatility, turnover).
Turnover is one-way, excluding the initial deployment of cash. The "random"
columns are percentiles of each metric taken separately across the
{len(result.random)} runs, not one particular run.

### Sharpe across the random runs

{bl.random_distribution(result)}

{beaten} of {len(result.random)} random runs beat the bar. 12-1 momentum's
Sharpe ({mom:.2f}) is above {100 * mom_rank:.0f}% of the random runs.

### Max drawdown across the random runs

{bl.random_distribution(result, "max_drawdown", "pct")}

### What costs take

The random books turn over about ten times a year, because a fresh draw
keeps a held name only one time in ten; momentum turns over about three
times. Run again with free fills, on the same window:

{gross}

So most of the gap between the random books and equal weight is costs, not
bad luck in picking: a model that trades as often as a random book starts
several points of CAGR behind.

{luck.render(result, ledger)}
## Caveats

* **Survivorship.** The universe is today's 100 large caps, chosen knowing
  they became large. Every stock-picking baseline here, random included,
  inherits that tailwind and so beats the benchmark more easily than it
  would have in real time. That is why the bar includes the equal-weight and
  random books, which carry the same bias, and not only the benchmark.
* **Equal weight, buy-and-hold** holds only the names priced on the first
  day; names listed later (e.g. AENA.MC, DSFIR.AS) never enter it, since a
  buy-and-hold book has no cash to buy them with. The rebalanced book adds
  them at the next rebalance.
* **Momentum is a baseline, not a model.** It is the plainest known signal,
  run untuned: no volatility scaling, no sector neutralisation, no crash
  protection.
* **One window.** These are full-period numbers, not per-split numbers. The
  bootstrap resamples days within this window; it cannot say how a regime
  the window never saw would have gone.
"""


if __name__ == "__main__":
    main()
