"""Day 44: P(top quintile), calibrated on a held-out slice, checked out of sample.

The real run needs the local store (`finmgr classify --model logit --slice 63
--slice 0.25 --as-of 2026-11-12 --doc docs/models/classifier_logit.md`). These
tests pin the machinery on synthetic data:

* the target marks exactly each date's top share of known labels;
* classifiers go through the registry, refuse a continuous label, and
  `finmgr train` refuses them;
* calibrators repair a deliberately distorted probability on unseen rows;
* the calibration slice is the end of the training window, purged from the
  classifier's fit, and the test windows are day 42's;
* neither the test window's labels nor the slice's reach the classifier;
* the check passes a calibrated forecast and fails an overconfident one;
* `finmgr classify` runs end to end, writes the page, chart and tearsheet, and
  exits 3 when the curve is off the diagonal.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_walkforward import DAYS, START, TICKERS, make_dataset, make_prices
from typer.testing import CliRunner

from finmgr import models
from finmgr.backtest import strategy as st
from finmgr.backtest.trials import read_ledger
from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.features.momentum import compute_momentum
from finmgr.models import classify as cl
from finmgr.models import compare as cmp
from finmgr.models import walkforward as wfm
from finmgr.report import calibration as cal_report

MIN_TRAIN = 200  # the CLI's default, so the command and these tests see the same folds


@pytest.fixture(scope="module")
def data() -> wfm.Dataset:
    return make_dataset()


@pytest.fixture(scope="module")
def cw(data: wfm.Dataset) -> cl.CalibratedWalkForward:
    return cl.calibrated_walk_forward("logit", data, min_train_size=MIN_TRAIN)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# --- the target --------------------------------------------------------------


def test_target_is_each_dates_top_share_of_known_labels() -> None:
    frame = pd.DataFrame(
        {
            "date": [1] * 10 + [2] * 5,
            "y": [*range(10), 5.0, 1.0, np.nan, 3.0, 2.0],
        }
    )
    top = cl.top_quantile_label(frame, "y", q=0.2)
    assert top.name == "top20_y"
    assert top[:10].tolist() == [0.0] * 8 + [1.0, 1.0]
    # Four known labels on date 2: the top 20% is the single highest; NaN stays NaN.
    assert top[10] == 1.0 and top[11] == 0.0 and np.isnan(top[12])
    assert top[13] == 0.0 and top[14] == 0.0
    with pytest.raises(ValueError, match="quantile"):
        cl.top_quantile_label(frame, "y", q=1.0)


# --- the classifiers ----------------------------------------------------------


@pytest.mark.parametrize("name", ["logit", "lgbm_clf"])
def test_classifiers_learn_a_signal_and_return_probabilities(name: str) -> None:
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.uniform(size=(3000, 2)), columns=["signal", "noise"])
    y = pd.Series((rng.uniform(size=3000) < sigmoid(6 * (X["signal"] - 0.6))).astype(float))
    model = models.create(name, **({"n_estimators": 50} if name == "lgbm_clf" else {}))
    assert isinstance(model, models.Model) and model.kind == "classifier"
    p = model.fit(X[:2000], y[:2000]).predict(X[2000:])
    assert p.min() >= 0 and p.max() <= 1
    assert np.corrcoef(p, X["signal"][2000:])[0, 1] > 0.8


def test_a_classifier_refuses_a_continuous_or_one_class_label() -> None:
    X = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="0/1 label"):
        models.create("logit").fit(X, pd.Series([0.01, -0.02, 0.03, 0.0]))
    with pytest.raises(ValueError, match="0/1 label"):
        models.create("logit").fit(X, pd.Series([0.0, 1.0, np.nan, 1.0]))
    with pytest.raises(ValueError, match="one class"):
        models.create("logit").fit(X, pd.Series([0.0, 0.0, 0.0, 0.0]))


def test_train_refuses_a_classifier(tmp_path, monkeypatch, write_settings) -> None:
    settings = write_settings({"data_dir": str(tmp_path / "d"), "log_dir": str(tmp_path / "l")})
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    result = CliRunner().invoke(app, ["train", "--model", "logit", "--dry-run"])
    assert result.exit_code == 2
    assert "finmgr classify" in " ".join(result.output.split())
    result = CliRunner().invoke(app, ["classify", "--model", "ridge"])
    assert result.exit_code == 2 and "regressor" in result.output


# --- the calibrators -----------------------------------------------------------


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrators_repair_a_distorted_probability_on_unseen_rows(method: str) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=40_000)
    truth = sigmoid(1.2 * x - 1.4)  # base rate near 20%
    y = (rng.uniform(size=x.size) < truth).astype(float)
    raw = sigmoid(3.0 * x)  # right ordering, badly overconfident and centred wrong
    fit, test = slice(0, 20_000), slice(20_000, None)
    before = cl.calibration_check(raw[test], y[test])
    assert not before.passed and before.ece > 0.1
    calibrator = cl.Calibrator(method).fit(raw[fit], y[fit])
    after = cl.calibration_check(calibrator.transform(raw[test]), y[test])
    assert after.passed and after.ece < 0.01
    # Monotone: it may tie, never reorder.
    order = np.argsort(raw[test])
    assert np.all(np.diff(calibrator.transform(raw[test])[order]) >= -1e-12)


def test_no_calibration_is_the_identity_and_bad_methods_fail() -> None:
    raw = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(cl.Calibrator("none").fit(raw, [0, 1, 1]).transform(raw), raw)
    with pytest.raises(ValueError, match="unknown calibration"):
        cl.Calibrator("beta")
    with pytest.raises(RuntimeError, match="before fit"):
        cl.Calibrator("platt").transform(raw)


# --- the folds -------------------------------------------------------------------


def test_slice_length_is_a_share_or_a_count() -> None:
    assert cl.slice_length(200, 0.4) == 80
    assert cl.slice_length(200, 63) == 63
    assert cl.slice_length(200, 63.0) == 63
    for bad in (0, -1, 1.5, 0.0):
        with pytest.raises(ValueError):
            cl.slice_length(200, bad)


def test_the_slice_ends_the_training_window_and_is_purged_from_the_fit(
    data: wfm.Dataset, cw: cl.CalibratedWalkForward
) -> None:
    for _, f in cw.folds.iterrows():
        assert data.label_end[f["fit_end"]] < f["cal_start"]  # purged
        assert f["cal_start"] > f["fit_end"]
        assert data.label_end[f["cal_end"]] < f["test_start"]  # day 42's embargo
    first = cw.folds.iloc[0]
    n_cal = sum(1 for d in DAYS if first["cal_start"] <= d <= first["cal_end"])
    assert n_cal == cl.slice_length(MIN_TRAIN, cl.CAL_SIZE)


def test_test_windows_are_day_42s(data: wfm.Dataset, cw: cl.CalibratedWalkForward) -> None:
    ridge = wfm.walk_forward("ridge", data, min_train_size=MIN_TRAIN)
    assert [f.test_start for f in cw.wf.folds] == [f.test_start for f in ridge.folds]
    assert [f.test_end for f in cw.wf.folds] == [f.test_end for f in ridge.folds]
    pd.testing.assert_frame_equal(
        cw.wf.scores[["ticker", "date"]], ridge.scores[["ticker", "date"]]
    )


def test_the_book_ranks_on_the_raw_probability(cw: cl.CalibratedWalkForward) -> None:
    merged = cw.wf.scores.merge(cw.oos, on=["ticker", "date"])
    assert np.allclose(merged["score"], merged["raw"])
    assert np.array_equal(cw.oos["prob"], cw.oos["prob_platt"])
    # Same ranking whichever calibrator: the variant does not depend on it.
    assert (
        cw.variant_params()
        == cl.calibrated_walk_forward(
            "logit", make_dataset(), method="isotonic", min_train_size=MIN_TRAIN
        ).variant_params()
    )


def test_no_label_from_the_test_window_reaches_the_fold(
    data: wfm.Dataset, cw: cl.CalibratedWalkForward
) -> None:
    fold = cw.folds.iloc[1]
    scrambled = data.frame.copy()
    later = scrambled["date"] >= fold["test_start"]
    scrambled.loc[later, data.label] = np.random.default_rng(0).standard_normal(int(later.sum()))
    other = cl.calibrated_walk_forward(
        "logit", wfm.Dataset(scrambled, *_rest(data)), min_train_size=MIN_TRAIN
    )
    mine = cw.oos[cw.oos["fold"] == fold["fold"]][["raw", "prob_platt", "prob_isotonic"]]
    theirs = other.oos[other.oos["fold"] == fold["fold"]][["raw", "prob_platt", "prob_isotonic"]]
    np.testing.assert_array_equal(mine.to_numpy(), theirs.to_numpy())


def test_the_slice_moves_the_calibrator_but_not_the_classifier(
    data: wfm.Dataset, cw: cl.CalibratedWalkForward
) -> None:
    fold = cw.folds.iloc[1]
    flipped = data.frame.copy()
    inside = (flipped["date"] >= fold["cal_start"]) & (flipped["date"] <= fold["cal_end"])
    flipped.loc[inside, data.label] = -flipped.loc[inside, data.label]
    other = cl.calibrated_walk_forward(
        "logit", wfm.Dataset(flipped, *_rest(data)), min_train_size=MIN_TRAIN
    )
    mine = cw.oos[cw.oos["fold"] == fold["fold"]]
    theirs = other.oos[other.oos["fold"] == fold["fold"]]
    np.testing.assert_array_equal(mine["raw"].to_numpy(), theirs["raw"].to_numpy())
    assert not np.allclose(mine["prob_platt"], theirs["prob_platt"])


def _rest(data: wfm.Dataset) -> tuple:
    return data.features, data.label, data.horizon, data.label_end, data.as_of


def test_a_slice_too_long_to_leave_a_fit_is_an_error(data: wfm.Dataset) -> None:
    with pytest.raises(ValueError, match="dates left to fit on"):
        cl.calibrated_walk_forward("logit", data, cal_size=0.9, min_train_size=MIN_TRAIN)
    with pytest.raises(ValueError, match="regressor"):
        cl.calibrated_walk_forward("ridge", data)


def test_the_synthetic_signal_is_calibrated_out_of_sample(cw: cl.CalibratedWalkForward) -> None:
    check = cl.calibration_check(cw.oos["prob"], cw.oos["target"])
    assert check.passed, check.verdict
    s = cl.scores(cw.oos["prob"], cw.oos["target"], base_rate=0.25)
    assert s["brier_skill"] > 0.05 and s["auc"] > 0.65
    assert cw.oos["target"].mean() == pytest.approx(0.25)  # 3 of 12 names a date


# --- the curve and the check -------------------------------------------------------


def test_reliability_bins_are_equal_count_and_keep_ties_together() -> None:
    p = np.repeat([0.1, 0.2, 0.3, 0.4], 20)
    y = np.tile([1, 0, 0, 0], 20).astype(float)
    curve = cl.reliability(p, y, bins=4)
    assert curve["n"].tolist() == [20] * 4
    assert curve["predicted"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert curve["observed"].tolist() == pytest.approx([0.25] * 4)
    assert cl.expected_calibration_error(curve) == pytest.approx(0.1)
    assert len(cl.reliability(np.full(80, 0.2), y, bins=10)) == 1  # one tie, one bin


def test_the_check_passes_a_calibrated_forecast_and_fails_an_overconfident_one() -> None:
    rng = np.random.default_rng(11)
    p = rng.beta(2, 8, size=50_000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert cl.calibration_check(p, y).passed
    stretched = np.clip(0.2 + 2.5 * (p - 0.2), 0, 1)
    bad = cl.calibration_check(stretched, y)
    assert not bad.passed and "NOT close" in bad.verdict
    # The documented blind spot: a flat forecast at the base rate passes too.
    assert cl.calibration_check(np.full(p.size, y.mean()), y).passed


def test_by_year_reports_each_year(cw: cl.CalibratedWalkForward) -> None:
    years = cl.by_year(cw.oos)
    assert list(years.index) == sorted({d.year for d in cw.oos["date"]})
    assert years["n"].sum() == len(cw.oos)


# --- the command ---------------------------------------------------------------------


@pytest.fixture
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


@pytest.fixture
def wired(inputs, data, tmp_path, monkeypatch) -> Path:
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
    monkeypatch.setattr(st, "log_run", partial(st.log_run, path=tmp_path / "trials.csv"))
    monkeypatch.setattr(cmp, "RANDOM_FILE", tmp_path / "none.csv")
    monkeypatch.setattr("webbrowser.open", lambda _url: True)
    return tmp_path


def test_cli_classifies_checks_backtests_and_writes_everything(wired: Path) -> None:
    doc = wired / "docs" / "models" / "classifier_logit.md"
    out = wired / "sheet.html"
    result = CliRunner().invoke(
        app,
        [
            "classify",
            "--start",
            str(START),
            "--top-k",
            "3",
            "--slice",
            "63",
            "--report",
            "--out",
            str(out),
            "--doc",
            str(doc),
        ],
        terminal_width=200,
    )
    assert result.exit_code == 0, result.output
    assert "calibration curve close to the diagonal" in result.output
    page = doc.read_text(encoding="utf-8")
    assert "![Reliability diagram](classifier_logit.png)" in page
    assert "### Slice length" in page and "63 dates" in page and "40% of training" in page
    assert "## The book it trades, against the baselines" in page
    assert doc.with_suffix(".png").read_bytes()[:4] == b"\x89PNG"
    sheet = out.read_text(encoding="utf-8")
    assert "<h2>Calibration out of sample</h2>" in sheet and "<svg" in sheet
    ledger = read_ledger(wired / "trials.csv")
    assert ledger["family"].tolist() == ["model: logit P(top 20%), walk-forward"]


@pytest.mark.usefixtures("wired")
def test_cli_exits_3_when_the_curve_is_off_the_diagonal(monkeypatch) -> None:
    strict = partial(cl.calibration_check, ece_tol=0.0, gap_tol=0.0)
    monkeypatch.setattr(cl, "calibration_check", strict)
    result = CliRunner().invoke(
        app, ["classify", "--start", str(START), "--top-k", "3"], terminal_width=200
    )
    assert result.exit_code == 3, result.output
    assert "NOT close to the diagonal" in result.output


@pytest.mark.usefixtures("wired")
def test_cli_rejects_bad_options() -> None:
    for args, hint in (
        (["--method", "beta"], "--method"),
        (["--cal-size", "1.5"], "--cal-size"),
        (["--model", "nope"], "unknown model"),
    ):
        result = CliRunner().invoke(app, ["classify", *args])
        assert result.exit_code == 2 and hint in result.output, result.output


def test_report_scores_every_method(cw: cl.CalibratedWalkForward) -> None:
    table = cal_report.method_scores(cw)
    assert list(table.index) == list(cl.METHODS)
    assert {"ece", "max_gap", "brier_skill", "auc"} <= set(table.columns)
