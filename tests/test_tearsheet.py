"""Day 40: `finmgr backtest --report` and the tearsheet behind it.

The real page needs the local store (`finmgr backtest --strategy momentum
--report`); these tests pin the machinery on a small synthetic universe: a
strategy run is the very same backtest day 38 runs (same metrics, same ledger
variant), the page's numbers are right (monthly returns compound to the total,
the rolling Sharpe's last window is the Sharpe of that window), and the page
holds every section with its charts inline. The CLI is driven end to end with
the store swapped for the synthetic inputs.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from finmgr.backtest import baselines as bl
from finmgr.backtest import strategy as st
from finmgr.backtest.costs import CostModel
from finmgr.backtest.metrics import sharpe_ratio
from finmgr.backtest.trials import count_variants, read_ledger, record
from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.features.momentum import compute_momentum
from finmgr.report import tearsheet as ts

DAYS = list(pd.bdate_range("2021-01-01", "2023-06-30").date)
UNIVERSE = [f"T{i:02d}" for i in range(12)]
START = dt.date(2022, 1, 3)
COSTS = CostModel.flat(10)


def make_prices(tickers: list[str], seed: int = 40) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def inputs() -> st.Inputs:
    prices = make_prices(UNIVERSE)
    return st.Inputs(
        prices=prices,
        benchmark_prices=make_prices(["BENCH"], seed=1),
        momentum=compute_momentum(prices, price="adj_close_eur"),
        rf=None,
        as_of=DAYS[-1],
        end=DAYS[-1],
    )


@pytest.fixture(scope="module")
def run(inputs: st.Inputs) -> st.StrategyRun:
    return st.run_strategy("momentum", inputs, start=START, top_k=3, hold=21, costs=COSTS)


def test_a_strategy_run_is_day_38s_backtest(inputs: st.Inputs, run: st.StrategyRun) -> None:
    base = bl.run_baselines(
        inputs.prices,
        inputs.benchmark_prices,
        inputs.momentum,
        rf=None,
        start=START,
        end=inputs.end,
        top_k=3,
        hold=21,
        costs=COSTS,
        n_random=1,
    )
    assert run.metrics == base.named[bl.MOMENTUM]
    assert run.benchmark_metrics == base.named[bl.BENCHMARK]
    assert run.params == bl.variant_params(base, bl.MOMENTUM)
    assert list(run.result.curve.index) == list(run.benchmark.curve.index)


@pytest.mark.parametrize("strategy", ["equal-weight", "equal-weight-rebalanced", "benchmark"])
def test_every_named_strategy_matches_its_baseline(inputs: st.Inputs, strategy: str) -> None:
    got = st.run_strategy(strategy, inputs, start=START, hold=21, costs=COSTS)
    base = bl.run_baselines(
        inputs.prices,
        inputs.benchmark_prices,
        inputs.momentum,
        rf=None,
        start=START,
        end=inputs.end,
        top_k=3,
        hold=21,
        costs=COSTS,
        n_random=1,
    )
    assert got.metrics == base.named[st.STRATEGIES[strategy]]


def test_random_is_seeded_and_unknown_strategies_fail(inputs: st.Inputs) -> None:
    a, b, c = (
        st.run_strategy("random", inputs, start=START, top_k=3, costs=COSTS, seed=s)
        for s in (5, 5, 6)
    )
    assert a.metrics == b.metrics
    assert a.metrics.sharpe != c.metrics.sharpe
    assert a.params["seed"] == 5
    with pytest.raises(ValueError, match="unknown strategy"):
        st.run_strategy("astrology", inputs)


def test_rerunning_a_variant_does_not_add_to_the_count(
    inputs: st.Inputs, run: st.StrategyRun, tmp_path: Path
) -> None:
    ledger = tmp_path / "trials.csv"
    first = st.log_run(run, path=ledger)
    again = st.log_run(run, path=ledger)
    assert first.variant_id == again.variant_id
    assert again.n_variants == 1
    assert read_ledger(ledger)["runs"].tolist() == [2]

    other = st.run_strategy("momentum", inputs, start=START, top_k=4, hold=21, costs=COSTS)
    assert st.log_run(other, path=ledger).n_variants == 2 == count_variants(ledger)


def test_the_familywise_interval_widens_with_the_count(run: st.StrategyRun, tmp_path) -> None:
    ledger = tmp_path / "trials.csv"
    one = st.log_run(run, path=ledger).familywise
    for i in range(9):
        record(f"other variant {i}", {"i": i}, sharpe=0.0, path=ledger)
    ten = st.log_run(run, path=ledger)
    assert ten.n_variants == 10
    assert ten.familywise.level == pytest.approx(0.995)
    assert ten.familywise.low < one.low and ten.familywise.high > one.high


def test_monthly_returns_compound_to_the_total() -> None:
    dates = pd.bdate_range("2022-01-14", "2023-03-10").date
    curve = pd.Series(np.linspace(100, 150, len(dates)), index=dates)
    table = ts.monthly_returns(curve)
    assert list(table.columns) == list(range(1, 13))
    assert list(table.index) == [2022, 2023]
    assert table.loc[2023, 4:].isna().all()
    assert np.prod(1 + table.stack().dropna().to_numpy()) - 1 == pytest.approx(0.5)
    jan = curve[[d for d in dates if d.month == 1 and d.year == 2022][-1]] / 100 - 1
    assert table.loc[2022, 1] == pytest.approx(jan)


def test_rolling_sharpe_last_window_is_that_windows_sharpe() -> None:
    rng = np.random.default_rng(0)
    excess = pd.Series(0.0004 + 0.01 * rng.standard_normal(600))
    rolled = ts.rolling_sharpe(excess, window=250, periods=252)
    assert rolled.iloc[:249].isna().all()
    assert rolled.iloc[-1] == pytest.approx(sharpe_ratio(excess.iloc[-250:], periods=252))


def test_heat_colours_diverge_around_a_neutral_grey() -> None:
    assert ts.heat_colour(0.0)[0] == "#f0efec"
    up, down = ts.heat_colour(0.05)[0], ts.heat_colour(-0.05)[0]
    assert int(up[5:7], 16) > int(up[1:3], 16)  # blue pole: more blue than red
    assert int(down[1:3], 16) > int(down[5:7], 16)  # red pole: more red than blue
    assert ts.heat_colour(1.0) == ts.heat_colour(ts.HEAT_CAP)  # capped
    assert ts.heat_colour(-0.5)[1] == "#ffffff"  # deep cells carry white text
    assert ts.heat_colour(math.nan)[0] == "transparent"


def test_the_page_has_every_section(run: st.StrategyRun, tmp_path: Path) -> None:
    logged = st.log_run(run, path=tmp_path / "trials.csv")
    page = ts.render_tearsheet(run, logged, generated=dt.datetime(2026, 11, 14, 9, 0))
    for heading in (
        "Equity curve",
        "Drawdown",
        "Rolling 12-month Sharpe",
        "Monthly returns",
        "Metrics",
        "How this was run",
    ):
        assert f"<h2>{heading}</h2>" in page
    assert page.count("<svg") == 3
    assert "<script" not in page and "http" not in page.split("<svg", 1)[0]  # self-contained
    ids = re.findall(r'id="([^"]+)"', page)
    assert len(ids) == len(set(ids)), "SVG ids collide across charts"
    iv = run.interval
    assert f"[{iv.low:.2f}, {iv.high:.2f}]" in page
    assert "1 variants" in page or "1 in <code>docs/trials.csv" in page
    assert "-0.0%" not in page
    assert "2026-11-14 09:00" in page


def test_cli_writes_and_opens_the_tearsheet(
    inputs: st.Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\n"
        "costs:\n  commission_bps: 10\n  spread_bps: 0\n  slippage_bps: 0\n"
        "  liquidity_scaling: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    monkeypatch.setattr(st, "load_inputs", lambda **_: inputs)
    ledger = tmp_path / "trials.csv"
    monkeypatch.setattr(st, "log_run", partial(st.log_run, path=ledger))
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)

    result = CliRunner().invoke(
        app,
        ["backtest", "--strategy", "momentum", "--report", "--start", str(START), "--top-k", "3"],
        terminal_width=200,
    )

    assert result.exit_code == 0, result.output
    page = tmp_path / "data" / "reports" / f"backtest_momentum_{inputs.as_of}.html"
    assert page.exists()
    assert opened == [page.resolve().as_uri()]
    assert "12-1 momentum, top-K vs BENCH" in page.read_text(encoding="utf-8")
    assert count_variants(ledger) == 1
    assert "Sharpe" in result.output and "variants tested" in result.output


def test_cli_rejects_an_unknown_strategy() -> None:
    result = CliRunner().invoke(app, ["backtest", "--strategy", "astrology"])
    assert result.exit_code != 0
    assert "unknown strategy" in result.output
