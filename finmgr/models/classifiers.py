"""Day 44: classifiers that predict P(top quintile) through the same interface.

A classifier is a :class:`~finmgr.models.base.EstimatorModel` like ridge and
LightGBM: same registry, same column checks, same determinism. Two things
differ. :meth:`fit` takes a *binary* label (1 = the row landed in its date's
top quintile, see :func:`finmgr.models.classify.top_quantile_label`) and
refuses anything else, so a classifier cannot be fed the continuous return by
mistake. :meth:`predict` returns the probability of class 1, uncalibrated:
:mod:`finmgr.models.classify` calibrates it on a held-out slice.

Train one with ``finmgr classify --model logit``; ``finmgr train`` (regression on
the excess return) refuses them.
"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from finmgr.models.base import EstimatorModel
from finmgr.models.boosting import LGBMModel
from finmgr.models.registry import register


class ClassifierModel(EstimatorModel):
    """An :class:`EstimatorModel` fitted on a 0/1 label whose scores are P(label = 1)."""

    #: Lets the CLI tell a classifier from a regressor without importing classes.
    kind: ClassVar[str] = "classifier"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        values = np.asarray(y, dtype=float)
        seen = set(np.unique(values[~np.isnan(values)]).tolist())
        if np.isnan(values).any() or not seen <= {0.0, 1.0}:
            raise ValueError(
                f"{self.name}: a classifier needs a 0/1 label (top quintile or not); "
                "train it with `finmgr classify`, not `finmgr train`"
            )
        if seen != {0.0, 1.0}:
            raise ValueError(f"{self.name}: the training label has only one class")
        return super().fit(X, pd.Series(values.astype(int), index=getattr(y, "index", None)))

    def _output(self, fitted: Any, X: pd.DataFrame) -> Any:
        return fitted.predict_proba(X)[:, 1]


@register("logit")
class LogitModel(ClassifierModel):
    """L2-penalised logistic regression: ridge's classification twin."""

    defaults: ClassVar[dict[str, Any]] = {"C": 1.0, "fit_intercept": True, "max_iter": 1000}

    def _estimator(self) -> Any:
        return LogisticRegression(**self.params)


@register("lgbm_clf")
class LGBMClassifierModel(ClassifierModel):
    """LightGBM with a log-loss objective; the same conservative, deterministic defaults."""

    defaults: ClassVar[dict[str, Any]] = dict(LGBMModel.defaults)

    def _estimator(self) -> Any:
        return LGBMClassifier(
            **self.params, n_jobs=1, deterministic=True, verbose=-1, importance_type="gain"
        )
