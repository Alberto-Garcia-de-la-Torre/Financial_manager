"""Day 44: predict P(top quintile), calibrate it on a held-out slice, check it out of sample.

    from finmgr.models.classify import calibrated_walk_forward, reliability, calibration_check

    data = load_dataset(as_of="2026-11-12")
    cw = calibrated_walk_forward("logit", data, method="platt")
    rel = reliability(cw.oos["prob"], cw.oos["target"])   # the calibration curve
    calibration_check(cw.oos["prob"], cw.oos["target"]).passed

**The target.** On each date, a row is in the *top quintile* when its
`fwd_xs_21d` ranks in the top ``q = 20%`` of the names with a known label that
date (:func:`top_quantile_label`). The base rate is therefore 20% on every
date by construction, so any skill must come from telling names apart, not
from timing.

**The folds.** Exactly day 42's: :func:`~finmgr.backtest.splits.walk_forward_splits`
with the same test size, minimum training window and embargo, over the same
dates — the test windows are identical to ridge's and LightGBM's. Inside each
fold the training dates are split once more, in time order:

* the **calibration slice** is the last :data:`CAL_SIZE` (40%) of the training dates;
* the classifier is **fitted** on the earlier training dates whose label ends
  before the slice opens (purged like a test window, so the slice's outcomes
  are as unseen to the classifier as the test window's are);
* a monotone **calibrator** — isotonic regression or Platt scaling — maps the
  classifier's raw probability on the slice to the slice's outcomes;
* the **test window** gets the calibrated probability. Nothing in it, label or
  feature, touches either fit.

**What the book trades.** The backtest ranks on the *raw* probability.
Calibration is monotone, so it cannot change who is in the top K — except that
isotonic regression is a step function and turns a strict ranking into ties.
Ranking on the raw score keeps that ordering and makes the book independent
of the calibration method (the ledger records one variant, not two copies).

**The acceptance check** (:func:`calibration_check`) is fixed before looking:
on the out-of-sample rows, the expected calibration error over
:data:`BINS` equal-count bins is at most :data:`ECE_TOL`, and no bin's
observed frequency is more than :data:`GAP_TOL` from its mean prediction. A
flat 20% forecast passes that trivially, so the report also shows the spread of
the predictions, Brier skill and AUC: calibrated *and* informative is the claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from finmgr.backtest.splits import OPEN_END, Split, walk_forward_splits
from finmgr.models.registry import create, resolve
from finmgr.models.walkforward import (
    FILL,
    MIN_TRAIN,
    TEST_SIZE,
    Dataset,
    Fold,
    WalkForward,
    information_coefficient,
)

#: Fraction of each date's names that count as "top": a quintile.
QUANTILE = 0.2
#: Share of each fold's training dates held out at its end to fit the calibrator.
#: A fraction, not a fixed count: a single quarter (63 dates) proved too short —
#: the calibrator learnt that quarter's regime and over-stretched the tails out
#: of sample (see docs/models/classifier_logit.md, "Slice length").
CAL_SIZE = 0.4
#: The default calibrator. Platt's two parameters overfit a slice less than the
#: isotonic step function does when the signal is this weak.
METHOD = "platt"
#: Fewest dates the classifier itself must be fitted on, after the slice and purge.
MIN_FIT = 63
#: Calibration methods: isotonic regression, Platt (logistic) scaling, or none.
METHODS = ("isotonic", "platt", "none")
#: Equal-count bins of the reliability curve.
BINS = 10
#: Acceptance: expected calibration error (weighted mean |observed - predicted|).
ECE_TOL = 0.02
#: Acceptance: the worst single bin's |observed - predicted|.
GAP_TOL = 0.05
#: Probabilities are clipped here before Platt's logit, so 0 and 1 stay finite.
_EPS = 1e-6


def target_column(label: str, q: float = QUANTILE) -> str:
    """``fwd_xs_21d`` -> ``top20_fwd_xs_21d``."""
    return f"top{round(q * 100)}_{label}"


def top_quantile_label(frame: pd.DataFrame, label: str, q: float = QUANTILE) -> pd.Series:
    """1.0 where `label` ranks in the date's top `q` of known labels, 0.0 below, NaN if unknown."""
    if not 0 < q < 1:
        raise ValueError(f"quantile must be in (0, 1), got {q}")
    pct = frame.groupby("date")[label].rank(pct=True, method="average")
    out = (pct > 1 - q).astype(float)
    out[frame[label].isna()] = np.nan
    return out.rename(target_column(label, q))


class Calibrator:
    """A monotone map from raw probability to calibrated probability, fitted on held-out rows."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in METHODS:
            raise ValueError(f"unknown calibration {method!r}; choose from {', '.join(METHODS)}")
        self.method = method
        self._fitted: Any = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: Sequence[float], outcome: Sequence[float]) -> Calibrator:
        raw, y = np.asarray(raw, dtype=float), np.asarray(outcome, dtype=float)
        if len(raw) != len(y) or len(raw) == 0:
            raise ValueError("calibration needs as many outcomes as predictions, and some")
        if self.method == "isotonic":
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._fitted = iso.fit(raw, y)
        elif self.method == "platt":
            # Unpenalised one-feature logistic regression on the raw log-odds.
            self._fitted = LogisticRegression(C=1e6).fit(self._logit(raw), y.astype(int))
        else:
            self._fitted = True
        return self

    def transform(self, raw: Sequence[float]) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("Calibrator.transform called before fit")
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            return np.asarray(self._fitted.predict(raw), dtype=float)
        if self.method == "platt":
            return self._fitted.predict_proba(self._logit(raw))[:, 1]
        return raw.copy()


def slice_length(n_train: int, cal_size: float) -> int:
    """Calibration dates for a fold of `n_train` training dates.

    An integer `cal_size` is a fixed count of dates; a fraction in (0, 1) is
    that share of the fold's training dates, so the slice grows with the
    expanding window instead of staying one quarter forever.
    """
    if 0 < cal_size < 1:
        return max(1, round(cal_size * n_train))
    if cal_size >= 1 and float(cal_size).is_integer():
        return int(cal_size)
    raise ValueError(
        f"cal_size must be a fraction in (0, 1) or a whole number of dates, got {cal_size}"
    )


def calibration_split(
    split: Split, label_end: Mapping[date, date], cal_size: float
) -> tuple[list[date], list[date]]:
    """A fold's training dates as (fit dates, calibration dates), purged between them."""
    train = list(split.train)
    cal_size = slice_length(len(train), cal_size)
    cal = train[-cal_size:]
    if len(train) <= cal_size:
        return [], cal
    fit = [t for t in train[:-cal_size] if label_end.get(t, OPEN_END) < cal[0]]
    return fit, cal


@dataclass(frozen=True)
class CalibratedWalkForward:
    """The walk-forward of a classifier, and its out-of-sample probabilities."""

    #: Scores the book ranks on (raw probability), folds, IC against the return label.
    wf: WalkForward
    method: str
    quantile: float
    cal_size: float
    target: str
    #: Out-of-sample rows with a known outcome: ticker, date, fold, raw, prob (the
    #: chosen `method`), prob_<m> for every method in :data:`METHODS`, target.
    oos: pd.DataFrame
    #: Per fold: fit / calibration date ranges and sizes.
    folds: pd.DataFrame

    def variant_params(self) -> dict[str, object]:
        """The ledger variant: the model's, plus what changes the book (not the calibrator)."""
        return {
            **self.wf.variant_params(),
            "target": {"label": self.target, "quantile": self.quantile},
            "calibration_slice": self.cal_size,
        }


def calibrated_walk_forward(
    model_name: str,
    data: Dataset,
    *,
    method: str = METHOD,
    quantile: float = QUANTILE,
    cal_size: float = CAL_SIZE,
    params: Mapping[str, Any] | None = None,
    test_size: int = TEST_SIZE,
    min_train_size: int = MIN_TRAIN,
    train_size: int | None = None,
    min_fit: int = MIN_FIT,
) -> CalibratedWalkForward:
    """Fit classifier `model_name` and a `method` calibrator in every walk-forward fold.

    Raises if `model_name` is not a classifier, or if a fold leaves fewer than
    `min_fit` dates to fit on once the calibration slice and its purge are taken.
    """
    if getattr(resolve(model_name), "kind", "regressor") != "classifier":
        raise ValueError(f"{model_name!r} is a regressor; use a classifier such as 'logit'")
    if method not in METHODS:
        raise ValueError(f"unknown calibration {method!r}; choose from {', '.join(METHODS)}")
    slice_length(min_train_size, cal_size)  # validates before any fitting
    overrides = dict(params or {})
    feats, label = data.features, data.label
    frame = data.frame.copy()
    target = target_column(label, quantile)
    frame[target] = top_quantile_label(frame, label, quantile)
    by_date = dict(frame.groupby("date", sort=True).indices)
    splits = list(
        walk_forward_splits(
            data.dates,
            horizon=data.horizon,
            test_size=test_size,
            train_size=train_size,
            min_train_size=min_train_size,
            label_end=data.label_end,
        )
    )
    if not splits:
        raise ValueError(
            f"no walk-forward fold fits: {len(data.dates)} dates, need "
            f"{min_train_size} training dates + a {data.horizon}-date embargo first"
        )

    def rows(dates: Sequence[date]) -> pd.DataFrame:
        picked = [by_date[d] for d in dates if d in by_date]
        return frame.iloc[np.concatenate(picked) if picked else np.array([], dtype=int)]

    folds: list[Fold] = []
    info: list[dict[str, object]] = []
    parts: list[pd.DataFrame] = []
    resolved: dict[str, Any] = {}
    for split in splits:
        fit_dates, cal_dates = calibration_split(split, data.label_end, cal_size)
        if len(fit_dates) < min_fit:
            raise ValueError(
                f"fold {split.fold}: {len(fit_dates)} dates left to fit on after a "
                f"calibration slice of {len(cal_dates)} dates (need {min_fit}); "
                "raise --min-train or shrink --cal-size"
            )
        fit = rows(fit_dates)
        fit = fit[fit[target].notna()]
        cal = rows(cal_dates)
        cal = cal[cal[target].notna()]
        test = rows(split.test)
        if fit.empty or cal.empty or test.empty:
            continue
        model = create(model_name, **overrides)
        resolved = model.params
        model.fit(fit[feats], fit[target])
        cal_raw = model.predict(cal[feats])
        pred = test[["ticker", "date", label, target]].copy()
        pred["fold"] = split.fold
        pred["raw"] = model.predict(test[feats])
        # Every method on the same slice: the comparison costs one fit, not three.
        for each in METHODS:
            calibrator = Calibrator(each).fit(cal_raw, cal[target])
            pred[f"prob_{each}"] = calibrator.transform(pred["raw"])
        pred["prob"] = pred[f"prob_{method}"]
        parts.append(pred)
        coef = model.coefficients() if hasattr(model, "coefficients") else None
        imp = model.importances() if hasattr(model, "importances") else None
        folds.append(
            Fold(
                fold=split.fold,
                train_start=fit_dates[0],
                train_end=fit_dates[-1],
                test_start=split.test_start,
                test_end=split.test_end,
                n_train=len(fit),
                n_test=len(test),
                coefficients=coef,
                importances=imp,
            )
        )
        info.append(
            {
                "fold": split.fold,
                "fit_start": fit_dates[0],
                "fit_end": fit_dates[-1],
                "fit_dates": len(fit_dates),
                "cal_start": cal_dates[0],
                "cal_end": cal_dates[-1],
                "n_cal": len(cal),
                "test_start": split.test_start,
                "test_end": split.test_end,
            }
        )
    if not parts:
        raise ValueError("every walk-forward fold was empty")
    scored = pd.concat(parts, ignore_index=True)
    scores = scored[["ticker", "date"]].assign(score=scored["raw"])
    wf = WalkForward(
        model=model_name,
        params=resolved,
        features=feats,
        label=label,
        horizon=data.horizon,
        test_size=test_size,
        min_train_size=min_train_size,
        train_size=train_size,
        folds=folds,
        scores=scores.reset_index(drop=True),
        ic=information_coefficient(scored.assign(score=scored["raw"]), label),
    )
    oos = scored[scored[target].notna()]
    keep = ["ticker", "date", "fold", "raw", "prob", *(f"prob_{m}" for m in METHODS), target]
    oos = oos[keep].rename(columns={target: "target"})
    return CalibratedWalkForward(
        wf=wf,
        method=method,
        quantile=quantile,
        cal_size=cal_size,
        target=target,
        oos=oos.reset_index(drop=True),
        folds=pd.DataFrame(info),
    )


# --- the calibration curve -------------------------------------------------------


def reliability(prob: Sequence[float], outcome: Sequence[float], bins: int = BINS) -> pd.DataFrame:
    """The calibration curve over `bins` equal-count bins of the prediction.

    One row per bin: its prediction range, row count, mean prediction, observed
    frequency, the gap between them, and a naive 95% binomial half-width (rows
    treated as independent; overlapping labels make the true one wider).
    Ties are kept together, so a step-function calibrator can yield fewer bins.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if len(p) != len(y) or len(p) == 0:
        raise ValueError("reliability needs as many outcomes as predictions, and some")
    ranks = pd.Series(p).rank(method="min").to_numpy()
    edges = np.quantile(ranks, np.linspace(0, 1, bins + 1)[1:-1])
    which = np.searchsorted(edges, ranks, side="left")
    frame = pd.DataFrame({"bin": which, "p": p, "y": y})
    grouped = frame.groupby("bin")
    out = pd.DataFrame(
        {
            "low": grouped["p"].min(),
            "high": grouped["p"].max(),
            "n": grouped.size(),
            "predicted": grouped["p"].mean(),
            "observed": grouped["y"].mean(),
        }
    ).reset_index(drop=True)
    out["gap"] = out["observed"] - out["predicted"]
    obs = out["observed"].clip(1e-9, 1 - 1e-9)
    out["half_width"] = 1.96 * np.sqrt(obs * (1 - obs) / out["n"])
    out.index.name = "bin"
    return out


def expected_calibration_error(curve: pd.DataFrame) -> float:
    """Row-weighted mean of |observed - predicted| over a :func:`reliability` curve."""
    return float((curve["gap"].abs() * curve["n"]).sum() / curve["n"].sum())


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def scores(
    prob: Sequence[float], outcome: Sequence[float], *, base_rate: float
) -> dict[str, float]:
    """Brier, Brier skill against a flat `base_rate` forecast, log loss, AUC, spread."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    flat = float(np.mean((base_rate - y) ** 2))
    curve = reliability(p, y)
    return {
        "n": float(len(p)),
        "base_rate": float(y.mean()),
        "mean_prob": float(p.mean()),
        "brier": brier,
        "brier_flat": flat,
        "brier_skill": 1 - brier / flat if flat > 0 else float("nan"),
        "log_loss": _log_loss(p, y),
        "log_loss_flat": _log_loss(np.full_like(p, base_rate), y),
        "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "p05": float(np.quantile(p, 0.05)),
        "p95": float(np.quantile(p, 0.95)),
        "ece": expected_calibration_error(curve),
        "max_gap": float(curve["gap"].abs().max()),
    }


@dataclass(frozen=True)
class CalibrationCheck:
    """The day 44 acceptance test on out-of-sample probabilities."""

    curve: pd.DataFrame
    ece: float
    max_gap: float
    ece_tol: float
    gap_tol: float

    @property
    def passed(self) -> bool:
        return self.ece <= self.ece_tol and self.max_gap <= self.gap_tol

    @property
    def verdict(self) -> str:
        what = "close to the diagonal" if self.passed else "NOT close to the diagonal"
        return (
            f"calibration curve {what}: ECE {self.ece:.4f} (limit {self.ece_tol}), "
            f"worst bin {self.max_gap:.4f} (limit {self.gap_tol})"
        )


def calibration_check(
    prob: Sequence[float],
    outcome: Sequence[float],
    *,
    bins: int = BINS,
    ece_tol: float = ECE_TOL,
    gap_tol: float = GAP_TOL,
) -> CalibrationCheck:
    """Is the reliability curve within the fixed tolerances of the diagonal?"""
    curve = reliability(prob, outcome, bins)
    return CalibrationCheck(
        curve=curve,
        ece=expected_calibration_error(curve),
        max_gap=float(curve["gap"].abs().max()),
        ece_tol=ece_tol,
        gap_tol=gap_tol,
    )


def by_year(oos: pd.DataFrame, column: str = "prob") -> pd.DataFrame:
    """ECE, worst bin and observed rate per calendar year: is it calibrated *every* year?"""
    rows = []
    years = pd.to_datetime(oos["date"]).dt.year
    for year, part in oos.groupby(years.to_numpy()):
        curve = reliability(part[column], part["target"], bins=5)
        rows.append(
            {
                "year": int(year),
                "n": len(part),
                "mean_prob": float(part[column].mean()),
                "observed": float(part["target"].mean()),
                "ece": expected_calibration_error(curve),
                "max_gap": float(curve["gap"].abs().max()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


__all__ = [
    "BINS",
    "CAL_SIZE",
    "ECE_TOL",
    "FILL",
    "GAP_TOL",
    "METHOD",
    "METHODS",
    "QUANTILE",
    "CalibratedWalkForward",
    "CalibrationCheck",
    "Calibrator",
    "by_year",
    "calibrated_walk_forward",
    "calibration_check",
    "calibration_split",
    "expected_calibration_error",
    "reliability",
    "scores",
    "slice_length",
    "target_column",
    "top_quantile_label",
]
