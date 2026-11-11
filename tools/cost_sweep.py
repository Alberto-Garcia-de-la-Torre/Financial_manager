"""Day 36: what raising costs does to a real equity curve.

    .venv/bin/python tools/cost_sweep.py --as-of 2026-10-15 --start 2018-01-01

Runs one rotating strategy on the stored universe — top 10 by trailing
63-session EUR return, held 21 sessions, filled at the next open — at a flat
0, 5, 10, 25 and 50bps one way, and at the configured cost model from
`config/settings.yaml`. Prints final equity, CAGR, total costs and turnover,
and saves the curves to `docs/day36_costs.png`. The score is a placeholder
with real turnover, not a strategy anyone should trade.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from finmgr.backtest.costs import CostModel, read_backtest_prices
from finmgr.backtest.engine import run_backtest
from finmgr.config import PROJECT_ROOT

FLAT_BPS = (0, 5, 10, 25, 50)


def momentum_scores(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Trailing `window`-session return per name, on each name's own sessions."""
    out = prices[["ticker", "date"]].copy()
    out["score"] = prices.groupby("ticker")["adj_close_eur"].pct_change(window)
    return out.dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--hold", type=int, default=21)
    parser.add_argument("--plot", type=Path, default=PROJECT_ROOT / "docs" / "day36_costs.png")
    args = parser.parse_args()

    history = str(pd.Timestamp(args.start) - pd.Timedelta(days=150))[:10]
    prices = read_backtest_prices(as_of=args.as_of, start=history)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    scores = momentum_scores(prices)
    models = {f"{bps}bps flat": CostModel.flat(bps) for bps in FLAT_BPS}
    models["configured"] = CostModel.from_settings()

    rows, curves = [], {}
    for label, model in models.items():
        result = run_backtest(
            scores, prices, top_k=args.top_k, hold=args.hold, costs=model, start=args.start
        )
        curve = result.curve
        years = (pd.Timestamp(curve.index[-1]) - pd.Timestamp(curve.index[0])).days / 365.25
        traded = result.trades["value_eur"].sum()
        rows.append(
            {
                "costs": label,
                "final_eur": round(curve.iloc[-1], 2),
                "cagr_%": round(100 * ((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1), 2),
                "costs_eur": round(result.total_cost, 2),
                "fills": len(result.trades),
                "turnover_x_per_yr": round(traded / curve.mean() / years, 2),
                "avg_bps": round(1e4 * result.total_cost / traded, 2) if traded else 0.0,
            }
        )
        curves[label] = curve

    table = pd.DataFrame(rows)
    print(f"top {args.top_k} by 63d return, hold {args.hold}, {args.start} .. {args.as_of}")
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, curve in curves.items():
        style = "--" if label == "configured" else "-"
        ax.plot(pd.to_datetime(curve.index), curve.to_numpy(), style, lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("equity, EUR (log)")
    ax.set_title("Same strategy, rising one-way costs, next-open fills")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot, dpi=110)
    print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
