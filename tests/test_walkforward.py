"""Day 42: ridge trained walk-forward, backtested beside the baselines.

The real run needs the local store (`finmgr train --model ridge --report
--doc docs/models/ridge.md`). These tests pin the machinery on synthetic data:

* the walk-forward never lets a fold see a label that reaches into its test
  window — scrambling every label from a fold's first test date on leaves that
  fold's predictions bit-for-bit unchanged;
* on a panel where one feature carries the signal, ridge finds it (positive
  weight, positive out-of-sample IC) and ignores the noise feature;
* model scores go through the *same* path as the baselines: momentum's scores
  through :func:`run_scores` give exactly :func:`run_strategy`'s metrics;
* the comparison picks day 38's bar and pairs the model against it;
* `finmgr train` runs end to end and writes the tearsheet and the Markdown page.
"""

from __future__ import annotations

import datetime as dt
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from finmgr.backtest import strategy as st
from finmgr.backtest.costs import CostModel
from finmgr.backtest.splits import label_end_dates
from finmgr.backtest.trials import read_ledger
from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.features.momentum import compute_momentum
from finmgr.models import compare as cmp
from finmgr.models import walkforward as wfm
from finmgr.report import model as model_report
from finmgr.report import tearsheet as ts

DAYS = list(pd.bdate_range("2021-01-01", "2023-06-30").date)
TICKERS = [f"T{i:02d}" for i in range(12)]
START = dt.date(2022, 1, 3)
COSTS = CostModel.flat(10)
FEATURES = ["signal_rank", "noise_rank"]
LABEL = "fwd_xs_21d"


def make_dataset(days=DAYS, tickers=TICKERS, seed: int = 42, horizon: int = 21) -> wfm.Dataset:
    """Ranks of a signal and of noise; the label is the signal's rank plus noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        sig = rng.permutation(len(tickers)) / (len(tickers) - 1)
        noise = rng.permutation(len(tickers)) / (len(tickers) - 1)
        label = 0.04 * (sig - 0.5) + 0.03 * rng.standard_normal(len(tickers))
        rows.append(
            pd.DataFrame(
                {
                    "date": d,
                    "ticker": tickers,
                    "signal_rank": sig,
                    "noise_rank": noise,
                    LABEL: label - label.mean(),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    frame.loc[frame["date"] > days[-horizon - 1], LABEL] = np.nan  # no future yet
    return wfm.Dataset(
        frame=frame,
        features=FEATURES,
        label=LABEL,
        horizon=horizon,
        label_end=label_end_dates(days, horizon),
        as_of=days[-1],
    )


@pytest.fixture(scope="module")
def data() -> wfm.Dataset:
    return make_dataset()


@pytest.fixture(scope="module")
def wf(data: wfm.Dataset) -> wfm.WalkForward:
    return wfm.walk_forward("ridge", data, test_size=63, min_train_size=120)


# --- the walk-forward --------------------------------------------------------


def test_scores_are_only_out_of_sample(data: wfm.Dataset, wf: wfm.WalkForward) -> None:
    first = wf.folds[0].test_start
    assert wf.scores["date"].min() == first
    assert wf.scores.duplicated(["ticker", "date"]).sum() == 0
    for fold in wf.folds:
        assert data.label_end[fold.train_end] < fold.test_start
    tested = {d for f in wf.folds for d in DAYS if f.test_start <= d <= f.test_end}
    assert set(wf.scores["date"]) == tested


def test_a_fold_never_sees_its_test_labels(data: wfm.Dataset, wf: wfm.WalkForward) -> None:
    fold = wf.folds[2]
    scrambled = data.frame.copy()
    later = scrambled["date"] >= fold.test_start
    rng = np.random.default_rng(0)
    scrambled.loc[later, LABEL] = rng.standard_normal(int(later.sum()))
    other = wfm.walk_forward(
        "ridge",
        wfm.Dataset(scrambled, FEATURES, LABEL, 21, data.label_end, data.as_of),
        test_size=63,
        min_train_size=120,
    )
    window = (wf.scores["date"] >= fold.test_start) & (wf.scores["date"] <= fold.test_end)
    pd.testing.assert_frame_equal(wf.scores[window], other.scores[window])


def test_ridge_finds_the_signal_and_ignores_the_noise(wf: wfm.WalkForward) -> None:
    coef = wf.coefficients()
    assert list(coef.columns) == FEATURES and len(coef) == len(wf.folds)
    assert (coef["signal_rank"] > 0.02).all()
    assert (coef["noise_rank"].abs() < 0.01).all()
    ic = wf.ic_summary()
    assert ic["mean"] > 0.2 and ic["positive"] > 0.8
    table = model_report.coefficient_table(wf)
    assert table.index[0] == "signal_rank" and table.loc["signal_rank", "positive"] == 1.0


def test_every_fold_gets_a_fresh_model_with_the_overrides(data: wfm.Dataset) -> None:
    strong = wfm.walk_forward(
        "ridge", data, params={"alpha": 1e7}, test_size=63, min_train_size=120
    )
    assert strong.params["alpha"] == 1e7
    assert strong.coefficients().abs().to_numpy().max() < 1e-3  # shrunk to nothing
    assert strong.variant_params()["params"]["alpha"] == 1e7


def test_a_model_without_coefficients_still_trains(data: wfm.Dataset) -> None:
    trees = wfm.walk_forward(
        "lgbm",
        data,
        params={"n_estimators": 20, "min_child_samples": 20},
        test_size=126,
        min_train_size=120,
    )
    assert trees.coefficients().empty and len(trees.scores) > 0
    md = model_report.learned_markdown(trees)
    assert "Mean weight" not in md and "Mean share of gain" in md


def test_too_little_history_is_an_error(data: wfm.Dataset) -> None:
    with pytest.raises(ValueError, match="no walk-forward fold"):
        wfm.walk_forward("ridge", data, test_size=63, min_train_size=len(DAYS))


def test_build_frame_fills_missing_ranks_and_keeps_unlabelled_rows() -> None:
    d1, d2 = dt.date(2024, 1, 2), dt.date(2024, 1, 3)
    panel = pd.DataFrame(
        {
            "date": [d1, d1, d2, d2],
            "ticker": ["A", "B", "A", "B"],
            "x_rank": [0.0, np.nan, 1.0, np.nan],
            "y_rank": [1.0, np.nan, np.nan, 0.0],
        }
    ).set_index(["date", "ticker"])
    labels = pd.DataFrame({"date": [d1], "ticker": ["A"], LABEL: [0.01]})
    out = wfm.build_frame(panel, labels, ["x_rank", "y_rank"], LABEL)
    assert list(zip(out["date"], out["ticker"], strict=True)) == [(d1, "A"), (d2, "A"), (d2, "B")]
    assert out["x_rank"].tolist() == [0.0, 1.0, 0.5]
    assert out["y_rank"].tolist() == [1.0, 0.5, 0.0]
    assert out[LABEL].iloc[0] == 0.01 and out[LABEL].iloc[1:].isna().all()


def test_information_coefficient_is_per_date_spearman() -> None:
    d = dt.date(2024, 1, 2)
    frame = pd.DataFrame(
        {"date": [d] * 4, "score": [1.0, 2.0, 3.0, 4.0], LABEL: [0.1, 0.2, 0.4, 0.3]}
    )
    ic = wfm.information_coefficient(frame, LABEL)
    assert ic[d] == pytest.approx(0.8)


# --- the backtest beside the baselines -----------------------------------------


def make_prices(tickers: list[str], seed: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, ticker in enumerate(tickers):
        close = 50 * np.exp(
            np.cumsum(0.0002 * (i - len(tickers) / 2) + 0.012 * rng.standard_normal(len(DAYS)))
        )
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
    prices = make_prices(TICKERS)
    return st.Inputs(
        prices=prices,
        benchmark_prices=make_prices(["BENCH"], seed=1),
        momentum=compute_momentum(prices, price="adj_close_eur"),
        rf=None,
        as_of=DAYS[-1],
        end=DAYS[-1],
    )


def test_model_scores_take_the_baselines_path(inputs: st.Inputs) -> None:
    base = st.run_strategy("momentum", inputs, start=START, top_k=3, hold=21, costs=COSTS)
    scores = st.strategy_scores("momentum", inputs)
    run = st.run_scores(
        scores,
        inputs,
        name="momentum again",
        params={"x": 1},
        start=START,
        top_k=3,
        hold=21,
        costs=COSTS,
    )
    assert run.metrics == base.metrics
    assert run.benchmark_metrics == base.benchmark_metrics
    assert run.family == "model: momentum again"
    assert run.params["x"] == 1 and run.params["top_k"] == 3 and "costs" in run.params


def test_comparison_uses_day_38s_bar(inputs: st.Inputs, wf: wfm.WalkForward) -> None:
    run = st.run_scores(
        wf.scores,
        inputs,
        name="ridge, walk-forward",
        params=wf.variant_params(),
        start=START,
        top_k=3,
        costs=COSTS,
    )
    baselines = cmp.run_baselines_for(inputs, start=START, top_k=3, hold=21, costs=COSTS)
    assert set(baselines) == {
        "equal weight, rebalanced",
        "equal weight, buy-and-hold",
        "12-1 momentum, top-K",
        "benchmark",
    }
    comp = cmp.compare(run, baselines, random=[0.1, 0.2, 0.3])
    best = max(baselines, key=lambda k: baselines[k].metrics.sharpe)
    assert comp.bar.sharpe == max(
        baselines[best].metrics.sharpe, comp.bar.candidates["random top-K, 95th percentile"]
    )
    assert comp.vs_reference.point == pytest.approx(
        run.metrics.sharpe - baselines[comp.reference].metrics.sharpe
    )
    assert comp.beats_bar == (run.metrics.sharpe > comp.bar.sharpe and comp.vs_reference.low > 0)
    md = cmp.comparison_markdown(comp)
    assert "**← model**" in md and "**← the bar**" in md
    page = ts.render_tearsheet(
        run, sections=[("Against the baselines", cmp.comparison_html(comp))], command="x"
    )
    assert "<h2>Against the baselines</h2>" in page and comp.verdict.split(":")[0] in page


def test_random_sharpes_only_from_the_same_window(tmp_path: Path) -> None:
    path = tmp_path / "random.csv"
    pd.DataFrame(
        {
            "seed": [0, 1],
            "start": ["2011-03-01"] * 2,
            "end": ["2026-10-15"] * 2,
            "sharpe": [0.5, 0.7],
        }
    ).to_csv(path, index=False)
    assert cmp.random_sharpes(path, start="2011-03-01", end="2026-10-15") == [0.5, 0.7]
    assert cmp.random_sharpes(path, start="2012-01-02", end="2026-10-15") == []
    assert cmp.random_sharpes(tmp_path / "missing.csv", start="a", end="b") == []


# --- the command ------------------------------------------------------------


def test_cli_trains_backtests_and_writes_both_reports(
    inputs: st.Inputs, data: wfm.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\n"
        "costs:\n  commission_bps: 10\n  spread_bps: 0\n  slippage_bps: 0\n"
        "  liquidity_scaling: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    monkeypatch.setattr(wfm, "load_dataset", lambda **_: data)
    monkeypatch.setattr(st, "load_inputs", lambda **_: inputs)
    ledger = tmp_path / "trials.csv"
    monkeypatch.setattr(st, "log_run", partial(st.log_run, path=ledger))
    monkeypatch.setattr(cmp, "RANDOM_FILE", tmp_path / "none.csv")
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    doc = tmp_path / "docs" / "models" / "ridge.md"

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--model",
            "ridge",
            "--report",
            "--start",
            str(START),
            "--top-k",
            "3",
            "--min-train",
            "120",
            "--doc",
            str(doc),
        ],
        terminal_width=200,
    )

    assert result.exit_code == 0, result.output
    page = tmp_path / "data" / "reports" / f"train_ridge_{inputs.as_of}.html"
    assert opened == [page.resolve().as_uri()]
    text = page.read_text(encoding="utf-8")
    for heading in ("Against the baselines", "What the model learnt", "Equity curve"):
        assert f"<h2>{heading}</h2>" in text
    assert "ridge, walk-forward vs BENCH" in text and "finmgr train --model ridge" in text
    md = doc.read_text(encoding="utf-8")
    assert md.startswith("# Model: ridge, walk-forward") and "## Against the baselines" in md
    assert "equal weight, rebalanced" in md and "signal" in md
    ledger_rows = read_ledger(ledger)
    assert ledger_rows["family"].tolist() == ["model: ridge, walk-forward"]
    assert ledger_rows["source"].tolist() == ["finmgr train"]
    assert "verdict:" in result.output
