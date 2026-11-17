"""Day 42: train any registered model walk-forward and score the out-of-sample predictions.

    from finmgr.models.walkforward import load_dataset, walk_forward

    data = load_dataset(as_of="2026-11-12")
    wf = walk_forward("ridge", data)
    wf.scores          # ticker, date, score - only ever predicted by a model fit on the past
    wf.coefficients()  # one row per fold: what ridge learnt each quarter
    wf.ic              # per test date: Spearman of score vs the realised label

**The data.** One row per `(date, ticker)` of the feature panel (day 31): the
day 32 model features as per-date ranks (`<feature>_rank`, uniform on [0, 1])
and the day 33 label `fwd_xs_21d` — the next 21 sessions' return minus that
date's equal-weighted universe mean. Ranks make every feature the same scale
and immune to outliers; the excess label takes the market out, so the model is
asked only the cross-sectional question a top-K book cares about.

**Missing features.** A rank is missing where its feature has not warmed up
(12-1 momentum and beta need a year) or the date had too few names. It is
filled with 0.5, the middle of the date's ranks: *no information*, which pulls
a linear prediction neither way. A row with every feature missing is dropped.
Training rows also need a known label; test rows do not.

**The splits.** :func:`~finmgr.backtest.splits.walk_forward_splits` over the
panel's dates: an expanding training window, a test window of
:data:`TEST_SIZE` dates (a quarter: the model is refitted four times a year),
an embargo of the 21-session horizon, and purging on each date's true
label end across every venue's calendar. The first fold needs
:data:`MIN_TRAIN` training dates (about nine months). Every score in :attr:`WalkForward.scores`
therefore comes from a model that had seen no label overlapping its date.

**One model per fold.** The model is created afresh from the registry for every
fold, so nothing learnt in one fold leaks into the next except through the
(earlier) data itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from finmgr.backtest.labels import compute_labels, excess_column
from finmgr.backtest.splits import Split, label_end_dates, walk_forward_splits
from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.features.hygiene import model_columns
from finmgr.models.registry import create

#: The label horizon in sessions: a month, the holding period of the book.
HORIZON = 21
#: Test window in panel dates: refit every quarter.
TEST_SIZE = 63
#: Fewest training dates the first fold needs: about nine months. Chosen so that,
#: on the store's history (panel from January 2010), the first test window opens
#: before day 38's window does (2011-03-01) and the model is invested from the
#: same first rebalance as every baseline; a year would leave it in cash until May.
MIN_TRAIN = 200
#: What a missing rank becomes: the middle of the date's cross-section.
FILL = 0.5


@dataclass(frozen=True)
class Dataset:
    """Features and label on one `(date, ticker)` frame, plus what the splitter needs."""

    frame: pd.DataFrame  # date, ticker, *features, label; sorted by (date, ticker)
    features: list[str]
    label: str
    horizon: int
    #: Panel date -> the last date any venue's `horizon`-session label on it depends on.
    label_end: dict[date, date]
    as_of: date

    @property
    def dates(self) -> list[date]:
        return sorted(set(self.frame["date"]))


def build_frame(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    features: Sequence[str],
    label: str,
    *,
    fill: float = FILL,
) -> pd.DataFrame:
    """Join panel features to labels on `(date, ticker)`, fill missing ranks with `fill`.

    `panel` is indexed by (or has columns) `date, ticker`; `labels` has
    `ticker, date` and the `label` column. Every panel row is kept (left join):
    the label may be unknown, the features are what is known that day.
    """
    feats = list(features)
    frame = panel.reset_index() if "date" not in panel.columns else panel.copy()
    missing = [c for c in ["date", "ticker", *feats] if c not in frame.columns]
    if missing:
        raise KeyError(f"panel is missing column(s): {', '.join(missing)}")
    frame = frame[["date", "ticker", *feats]]
    frame = frame[frame[feats].notna().any(axis=1)]
    frame = frame.merge(labels[["date", "ticker", label]], on=["date", "ticker"], how="left")
    frame[feats] = frame[feats].astype(float).fillna(fill)
    return frame.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def load_dataset(
    *,
    as_of: pit.AsOf,
    horizon: int = HORIZON,
    features: Sequence[str] | None = None,
    settings: Settings | None = None,
) -> Dataset:
    """The stored panel's model features and the `horizon`-session excess label, as of `as_of`."""
    from finmgr.data.benchmarks import read_total_return
    from finmgr.features.panel import read_panel
    from finmgr.features.returns import ticker_sessions

    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    feats = list(features) if features is not None else model_columns()
    panel = read_panel(columns=feats, settings=resolved).reset_index()
    panel = panel[panel["date"] <= cutoff]
    if panel.empty:
        raise ValueError("the feature panel has no rows; run `finmgr features --full`")
    prices = read_total_return(None, as_of=cutoff, settings=resolved)
    sessions = ticker_sessions(prices, resolved)
    labels = compute_labels(prices, horizons=(horizon,), sessions=sessions)
    labels = pit.guard(labels, cutoff, what="load_dataset labels")
    label = excess_column(horizon)
    frame = build_frame(panel, labels, feats, label)
    ends = label_end_dates(sorted(set(frame["date"])), horizon, sessions)
    return Dataset(frame, feats, label, horizon, ends, cutoff)


@dataclass(frozen=True)
class Fold:
    """One refit: when it trained, when it predicted, on how much, and what it learnt."""

    fold: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train: int  # rows with a label actually fitted on
    n_test: int  # rows predicted
    coefficients: pd.Series | None = field(default=None, repr=False)
    #: Normalised feature importances (trees); None for a model that has none.
    importances: pd.Series | None = field(default=None, repr=False)


@dataclass(frozen=True)
class WalkForward:
    """Every out-of-sample prediction of one model, and how it was made."""

    model: str
    params: dict[str, Any]
    features: list[str]
    label: str
    horizon: int
    test_size: int
    min_train_size: int
    train_size: int | None
    folds: list[Fold]
    #: `ticker, date, score`: the frame the backtest engine ranks on.
    scores: pd.DataFrame
    #: Test date -> Spearman rank correlation of score and realised label (where known).
    ic: pd.Series

    def coefficients(self) -> pd.DataFrame:
        """Fold x feature weights; empty when the model has none (trees)."""
        rows = {f.test_start: f.coefficients for f in self.folds if f.coefficients is not None}
        if not rows:
            return pd.DataFrame(columns=self.features)
        out = pd.DataFrame(rows).T
        out.index.name = "test_start"
        return out

    def importances(self) -> pd.DataFrame:
        """Fold x feature importance shares (each row sums to 1); empty for linear models."""
        rows = {f.test_start: f.importances for f in self.folds if f.importances is not None}
        if not rows:
            return pd.DataFrame(columns=self.features)
        out = pd.DataFrame(rows).T
        out.index.name = "test_start"
        return out

    def ic_summary(self) -> dict[str, float]:
        """Mean IC, its std, the IC information ratio and a naive t-stat.

        The t-stat counts every test date as independent, which overlapping
        21-session labels are not; read it as an upper bound.
        """
        ic = self.ic.dropna()
        n = len(ic)
        mean = float(ic.mean()) if n else float("nan")
        std = float(ic.std(ddof=1)) if n > 1 else float("nan")
        ir = mean / std if std and std > 0 else float("nan")
        return {
            "dates": float(n),
            "mean": mean,
            "std": std,
            "ir": ir,
            "t": ir * np.sqrt(n) if n else float("nan"),
            "positive": float((ic > 0).mean()) if n else float("nan"),
        }

    def variant_params(self) -> dict[str, object]:
        """What defines this run as a variant in the ledger (window and costs added later)."""
        return {
            "model": self.model,
            "params": self.params,
            "features": self.features,
            "label": self.label,
            "fill": FILL,
            "splits": {
                "horizon": self.horizon,
                "test_size": self.test_size,
                "min_train_size": self.min_train_size,
                "train_size": self.train_size,
                "embargo": self.horizon,
            },
        }


def information_coefficient(scored: pd.DataFrame, label: str) -> pd.Series:
    """Per date: Spearman correlation of `score` and `label` over the rows with both."""
    known = scored.dropna(subset=["score", label])
    if known.empty:
        return pd.Series(dtype=float, name="ic")
    ranks = known.groupby("date")[["score", label]].rank()
    ranks["date"] = known["date"]

    def corr(group: pd.DataFrame) -> float:
        if len(group) < 3 or group["score"].nunique() < 2 or group[label].nunique() < 2:
            return float("nan")
        return float(np.corrcoef(group["score"], group[label])[0, 1])

    ic = ranks.groupby("date")[["score", label]].apply(corr)
    ic.name = "ic"
    return ic


def walk_forward(
    model_name: str,
    data: Dataset,
    *,
    params: Mapping[str, Any] | None = None,
    test_size: int = TEST_SIZE,
    min_train_size: int = MIN_TRAIN,
    train_size: int | None = None,
) -> WalkForward:
    """Fit `model_name` on each fold's training rows and predict its test rows.

    Raises if the splitter yields no fold (too little history for
    `min_train_size` plus the embargo).
    """
    overrides = dict(params or {})
    frame, feats, label = data.frame, data.features, data.label
    by_date = dict(frame.groupby("date", sort=True).indices)
    splits: list[Split] = list(
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

    def rows(dates: Sequence[date]) -> np.ndarray:
        picked = [by_date[d] for d in dates if d in by_date]
        return np.concatenate(picked) if picked else np.array([], dtype=int)

    folds: list[Fold] = []
    parts: list[pd.DataFrame] = []
    resolved: dict[str, Any] = {}
    for split in splits:
        train = frame.iloc[rows(split.train)]
        train = train[train[label].notna()]
        test = frame.iloc[rows(split.test)]
        if train.empty or test.empty:
            continue
        model = create(model_name, **overrides)
        resolved = model.params
        model.fit(train[feats], train[label])
        pred = test[["ticker", "date", label]].copy()
        pred["score"] = model.predict(test[feats])
        parts.append(pred)
        coef = model.coefficients() if hasattr(model, "coefficients") else None
        imp = model.importances() if hasattr(model, "importances") else None
        folds.append(
            Fold(
                fold=split.fold,
                train_start=split.train_start,
                train_end=split.train_end,
                test_start=split.test_start,
                test_end=split.test_end,
                n_train=len(train),
                n_test=len(test),
                coefficients=coef,
                importances=imp,
            )
        )
    if not parts:
        raise ValueError("every walk-forward fold was empty")
    scored = pd.concat(parts, ignore_index=True)
    return WalkForward(
        model=model_name,
        params=resolved,
        features=feats,
        label=label,
        horizon=data.horizon,
        test_size=test_size,
        min_train_size=min_train_size,
        train_size=train_size,
        folds=folds,
        scores=scored[["ticker", "date", "score"]].reset_index(drop=True),
        ic=information_coefficient(scored, label),
    )
