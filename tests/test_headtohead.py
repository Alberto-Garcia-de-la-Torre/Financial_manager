"""Day 43: boosting beside ridge, on provably identical inputs.

The real run needs the local store (`finmgr compare-models ridge lgbm --report
--doc docs/models/ridge_vs_lgbm.md`). These tests pin the machinery on the
synthetic panel of `test_walkforward.py`:

* LightGBM reports gain importances that sum to one and find the signal;
* a head-to-head of ridge and lgbm passes every fairness check, and the
  pairing statistics are what they claim (a model against itself: zero
  difference, perfect agreement);
* each fairness check *catches* the difference it is named for, and
  :func:`head_to_head` refuses to compare runs that differ in more than the model;
* both pages render every section, and `finmgr compare-models` runs end to end.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_walkforward import COSTS, START, make_dataset, make_prices
from typer.testing import CliRunner

from finmgr.backtest import strategy as st
from finmgr.backtest.costs import CostModel
from finmgr.backtest.trials import read_ledger
from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.features.momentum import compute_momentum
from finmgr.models import compare as cmp
from finmgr.models import create
from finmgr.models import headtohead as hth
from finmgr.models import walkforward as wfm
from finmgr.report import headtohead as hreport
from finmgr.report import model as model_report

TICKERS = [f"T{i:02d}" for i in range(12)]
FAST_LGBM = {"n_estimators": 30, "min_child_samples": 20}
SPLITS = {"test_size": 63, "min_train_size": 120}


@pytest.fixture(scope="module")
def data() -> wfm.Dataset:
    return make_dataset()


@pytest.fixture(scope="module")
def inputs() -> st.Inputs:
    prices = make_prices(TICKERS)
    days = sorted(set(prices["date"]))
    return st.Inputs(
        prices=prices,
        benchmark_prices=make_prices(["BENCH"], seed=1),
        momentum=compute_momentum(prices, price="adj_close_eur"),
        rf=None,
        as_of=days[-1],
        end=days[-1],
    )


@pytest.fixture(scope="module")
def h(data: wfm.Dataset, inputs: st.Inputs) -> hth.HeadToHead:
    return hth.head_to_head(
        ["ridge", "lgbm"],
        data,
        inputs,
        start=START,
        top_k=3,
        hold=21,
        costs=COSTS,
        params={"lgbm": FAST_LGBM},
        random=[0.1, 0.2],
        **SPLITS,
    )


# --- importances ---------------------------------------------------------------


def test_lgbm_importances_sum_to_one_and_find_the_signal(data: wfm.Dataset) -> None:
    train = data.frame.dropna(subset=[data.label])
    model = create("lgbm", **FAST_LGBM).fit(train[data.features], train[data.label])
    imp = model.importances()
    assert imp is not None and list(imp.index) == data.features
    assert imp.sum() == pytest.approx(1.0)
    assert imp["signal_rank"] > 0.7
    assert model.coefficients() is None


def test_ridge_has_no_importances(data: wfm.Dataset) -> None:
    train = data.frame.dropna(subset=[data.label])
    model = create("ridge").fit(train[data.features], train[data.label])
    assert model.importances() is None


def test_walk_forward_records_importances_per_fold(h: hth.HeadToHead) -> None:
    trees = h.walkforwards["lgbm"]
    imp = trees.importances()
    assert len(imp) == len(trees.folds)
    assert np.allclose(imp.sum(axis=1), 1.0)
    assert h.walkforwards["ridge"].importances().empty
    md = model_report.learned_markdown(trees)
    assert "Mean share of gain" in md and "Mean weight" not in md
    table = model_report.importance_table(trees)
    assert table.index[0] == "signal_rank"


# --- the head-to-head ------------------------------------------------------------


def test_every_fairness_check_passes(h: hth.HeadToHead) -> None:
    assert [c.name for c in h.checks] == [
        "features",
        "label and horizon",
        "split settings",
        "fold boundaries",
        "rows fitted per fold",
        "scored rows",
        "backtest date axis",
        "window, top-K, hold",
        "costs",
        "benchmark and risk-free",
    ]
    assert all(c.passed for c in h.checks), [c for c in h.checks if not c.passed]
    ridge, lgbm = h.walkforwards["ridge"], h.walkforwards["lgbm"]
    assert [(f.train_end, f.test_start) for f in ridge.folds] == [
        (f.train_end, f.test_start) for f in lgbm.folds
    ]


def test_both_models_meet_the_same_baselines(h: hth.HeadToHead) -> None:
    a, b = h.comparisons["ridge"], h.comparisons["lgbm"]
    assert a.baselines is b.baselines is h.baselines
    assert a.bar == b.bar
    assert h.runs["lgbm"].params["costs"] == h.runs["ridge"].params["costs"]
    assert h.runs["lgbm"].params["params"]["n_estimators"] == 30


def test_the_pair_is_the_paired_sharpe_difference(h: hth.HeadToHead) -> None:
    (p,) = h.pairs
    assert (p.incumbent, p.challenger) == ("ridge", "lgbm")
    assert p.sharpe.point == pytest.approx(
        h.runs["lgbm"].metrics.sharpe - h.runs["ridge"].metrics.sharpe
    )
    assert p.sharpe.low <= p.sharpe.point <= p.sharpe.high
    ic = pd.concat([h.walkforwards["ridge"].ic, h.walkforwards["lgbm"].ic], axis=1).dropna()
    assert p.ic_diff.mean() == pytest.approx((ic.iloc[:, 1] - ic.iloc[:, 0]).mean())
    assert len(p.folds) == len(h.walkforwards["ridge"].folds)
    # Both learn the same single signal: they mostly agree.
    assert p.score_corr.mean() > 0.8
    assert p.top_overlap.min() >= 0 and p.top_overlap.max() <= 1


def test_a_model_against_itself_is_no_different(h: hth.HeadToHead) -> None:
    wf = h.walkforwards["ridge"]
    twin = dataclasses.replace(wf, model="ridge2")
    p = hth.pair(
        "ridge",
        "ridge2",
        {"ridge": wf, "ridge2": twin},
        {"ridge": h.runs["ridge"], "ridge2": h.runs["ridge"]},
        top_k=3,
    )
    assert p.sharpe.point == 0 and p.sharpe.low == 0 and p.sharpe.high == 0
    assert (p.ic_diff == 0).all() and p.fold_wins == 0
    assert p.score_corr.dropna().eq(1.0).all()
    assert p.top_overlap.eq(1.0).all()


def test_top_overlap_counts_shared_picks() -> None:
    d = dt.date(2024, 1, 2)
    a = pd.DataFrame({"ticker": list("ABCD"), "date": d, "score": [4.0, 3.0, 2.0, 1.0]})
    b = pd.DataFrame({"ticker": list("ABCD"), "date": d, "score": [4.0, 1.0, 3.0, 2.0]})
    assert hth._top_overlap(a, b, 2)[d] == 0.5
    assert hth._top_overlap(a, a, 2)[d] == 1.0


# --- fairness is enforced, not assumed -----------------------------------------------


def test_checks_catch_different_splits(h: hth.HeadToHead, data: wfm.Dataset) -> None:
    other = wfm.walk_forward("ridge", data, test_size=42, min_train_size=120)
    checks = {
        c.name: c
        for c in hth.fairness_checks({"ridge": h.walkforwards["ridge"], "x": other}, h.runs)
    }
    assert not checks["split settings"].passed
    assert not checks["fold boundaries"].passed
    assert checks["features"].passed


def test_checks_catch_different_costs(h: hth.HeadToHead, inputs: st.Inputs) -> None:
    wf = h.walkforwards["lgbm"]
    cheap = st.run_scores(
        wf.scores,
        inputs,
        name="lgbm",
        params={},
        start=START,
        top_k=3,
        hold=21,
        costs=CostModel.flat(1),
    )
    checks = {
        c.name: c
        for c in hth.fairness_checks(h.walkforwards, {"ridge": h.runs["ridge"], "lgbm": cheap})
    }
    assert not checks["costs"].passed and "ridge:" in checks["costs"].detail
    assert checks["fold boundaries"].passed


def test_checks_catch_different_windows_and_top_k(h: hth.HeadToHead, inputs: st.Inputs) -> None:
    wf = h.walkforwards["lgbm"]
    later = st.run_scores(
        wf.scores,
        inputs,
        name="lgbm",
        params={},
        start=dt.date(2022, 6, 1),
        top_k=4,
        hold=21,
        costs=COSTS,
    )
    checks = {
        c.name: c
        for c in hth.fairness_checks(h.walkforwards, {"ridge": h.runs["ridge"], "lgbm": later})
    }
    assert not checks["backtest date axis"].passed
    assert not checks["window, top-K, hold"].passed


def test_head_to_head_refuses_runs_that_differ_in_more_than_the_model(
    data: wfm.Dataset, inputs: st.Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = hth.walk_forward

    def leaky(name, data, **kw):  # lgbm "accidentally" sees one feature fewer
        if name == "lgbm":
            data = dataclasses.replace(data, features=data.features[:1])
        return real(name, data, **kw)

    monkeypatch.setattr(hth, "walk_forward", leaky)
    with pytest.raises(hth.UnfairComparison, match="features"):
        hth.head_to_head(
            ["ridge", "lgbm"],
            data,
            inputs,
            start=START,
            top_k=3,
            hold=21,
            costs=COSTS,
            params={"lgbm": FAST_LGBM},
            **SPLITS,
        )


def test_head_to_head_needs_two_models_and_known_params(
    data: wfm.Dataset, inputs: st.Inputs
) -> None:
    common = {"start": START, "top_k": 3, "hold": 21, "costs": COSTS, **SPLITS}
    with pytest.raises(ValueError, match="at least two"):
        hth.head_to_head(["ridge", "ridge"], data, inputs, **common)
    with pytest.raises(ValueError, match="not compared"):
        hth.head_to_head(["ridge", "lgbm"], data, inputs, params={"xgb": {}}, **common)


# --- the pages ---------------------------------------------------------------------


def test_markdown_page_has_every_section(h: hth.HeadToHead) -> None:
    md = hreport.headtohead_markdown(h, command="finmgr compare-models ridge lgbm", n_variants=7)
    assert md.startswith("# Head to head: ridge vs lgbm")
    for heading in (
        "Verdict",
        "Is the comparison fair?",
        "Side by side",
        "Against the baselines",
        "Head to head",
        "Year by year",
        "Setup",
        "What each model learnt",
    ):
        assert f"## {heading}\n" in md
    assert md.count("✅ same") == len(h.checks) and "❌" not in md
    assert "| ridge, walk-forward ← model |" in md and "| lgbm, walk-forward ← model |" in md
    assert "← the bar" in md and "lgbm minus ridge" in md
    assert "Mean weight" in md and "Mean share of gain" in md
    head, rows = hreport.side_by_side(h)
    assert head == ["", "ridge", "lgbm"] and rows[0][0] == "Sharpe"


def test_html_page_draws_both_models_and_the_bar(h: hth.HeadToHead) -> None:
    page = hreport.headtohead_html(h, command="x")
    for heading in ("Is the comparison fair?", "Side by side", "Equity curve", "Head to head"):
        assert f"<h2>{heading}</h2>" in page
    assert page.count("<svg") == 3
    assert "DIFFERS" not in page and "(the bar)" in page
    for colour in hreport.COLOURS:
        assert colour in page


# --- the command ---------------------------------------------------------------------


def test_cli_compares_models_and_writes_both_pages(
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
    doc = tmp_path / "docs" / "models" / "ridge_vs_lgbm.md"

    args = [
        "compare-models",
        "ridge",
        "lgbm",
        "--start",
        str(START),
        "--top-k",
        "3",
        "--min-train",
        "120",
        "-p",
        "lgbm.n_estimators=30",
        "-p",
        "lgbm.min_child_samples=20",
    ]
    result = CliRunner().invoke(app, [*args, "--report", "--doc", str(doc)], terminal_width=200)

    assert result.exit_code == 0, result.output
    page = tmp_path / "data" / "reports" / f"compare_ridge_vs_lgbm_{inputs.as_of}.html"
    assert opened == [page.resolve().as_uri()]
    assert "<h2>Is the comparison fair?</h2>" in page.read_text(encoding="utf-8")
    md = doc.read_text(encoding="utf-8")
    assert md.startswith("# Head to head: ridge vs lgbm")
    assert "finmgr compare-models ridge lgbm -p lgbm.n_estimators=30" in md
    assert "n_estimators=30" in md
    rows = read_ledger(ledger)
    assert sorted(rows["family"]) == ["model: lgbm, walk-forward", "model: ridge, walk-forward"]
    assert "fair: fold boundaries" in result.output and "verdict:" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["compare-models", "ridge"],
        ["compare-models", "ridge", "xgb"],
        ["compare-models", "ridge", "lgbm", "-p", "n_estimators=5"],
        ["compare-models", "ridge", "lgbm", "-p", "lgbm.n_estimatorz=5"],
    ],
)
def test_cli_rejects_bad_arguments(args: list[str]) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
