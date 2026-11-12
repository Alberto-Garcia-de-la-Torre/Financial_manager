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
from datetime import date
from pathlib import Path

from finmgr.backtest import baselines as bl
from finmgr.backtest.costs import NO_COSTS, CostModel, read_backtest_prices
from finmgr.backtest.engine import run_backtest
from finmgr.backtest.metrics import backtest_metrics
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
        )
    )
    print(bl.summary_table(result))
    print(f"\nbar: Sharpe {result.bar.sharpe:.3f} ({result.bar.source}); {elapsed:.0f}s")
    print(f"wrote {args.out} and {csv}")


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
    result: bl.Baselines, *, bench: str, as_of: date, n_tickers: int, csv: Path, gross: str
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
    return f"""# Baselines (day 38)

Generated by `tools/baselines.py --as-of {as_of}` from the local store; the
code is `finmgr/backtest/baselines.py`. Rerun it after any change to the
engine, costs, metrics or data, and update this page.

## The bar

> **Sharpe {result.bar.sharpe:.2f}** (against €STR/EONIA, {any_metrics.start} →
> {any_metrics.end}, top {result.top_k}, held {result.hold} sessions, configured
> costs, next-open fills). Set by: *{result.bar.source}*.
>
> A model is worth keeping only if, run through the same engine on the same
> window with the same costs, its Sharpe is above {result.bar.sharpe:.2f}. Day 39
> turns this point into an interval; until then, clearing it by a hair is
> not clearing it.

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
* **One window, one point estimate.** These are full-period numbers, not
  per-split numbers; day 39 adds bootstrap intervals and the count of
  variants tested.
"""


if __name__ == "__main__":
    main()
