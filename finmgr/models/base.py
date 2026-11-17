"""Day 41: the one interface every learned model implements.

    from finmgr.models import create

    model = create("ridge", alpha=10.0)
    model.fit(X_train, y_train)  # X: features as columns, y: the label
    scores = model.predict(X_test)  # one float per row of X_test, higher = better

A model is four things — :attr:`~Model.name`, :attr:`~Model.params`,
:meth:`~Model.fit`, :meth:`~Model.predict` — and nothing else. The walk-forward
harness (day 42 on) only ever talks to that surface, so ridge, boosting, the
classifier and the ensemble all go through the same splits, the same costs and
the same ledger without a line of model-specific plumbing.

:class:`EstimatorModel` is the shared implementation for anything with a
scikit-learn style estimator underneath. It adds the two checks every model
needs and none should have to remember: ``predict`` before ``fit`` is an error,
and ``predict`` on a different feature set — a missing column, an extra one, or
the same ones in another order — is an error rather than a silently shuffled
prediction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Model(Protocol):
    """What the harness needs from a model. Structural: no base class required."""

    #: Registry name, e.g. ``"ridge"``; also the family the ledger logs runs under.
    name: str

    @property
    def params(self) -> dict[str, Any]:
        """The full hyperparameters, defaults included: enough to rebuild the model."""
        ...

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Learn from features ``X`` (one column per feature) and label ``y``."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """One score per row of ``X``; only the ordering within a date matters."""
        ...


class NotFittedError(RuntimeError):
    """``predict`` was called on a model that has not been fitted."""


class EstimatorModel(ABC):
    """A :class:`Model` backed by a scikit-learn style estimator.

    Subclasses set :attr:`name` and :attr:`defaults` and build the estimator in
    :meth:`_estimator`. Construction takes keyword overrides of the defaults and
    rejects any name not among them, so a typo in ``--param`` fails loudly
    instead of being ignored.
    """

    name: str = ""
    #: Every hyperparameter the model accepts, with its default.
    defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {', '.join(unknown)}; "
                f"accepted: {', '.join(sorted(self.defaults))}"
            )
        self._params = {**self.defaults, **params}
        self._fitted: Any = None
        self._columns: list[str] | None = None

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    @abstractmethod
    def _estimator(self) -> Any:
        """A fresh, unfitted estimator built from :attr:`params`."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        if len(X) != len(y):
            raise ValueError(f"{self.name}: X has {len(X)} rows but y has {len(y)}")
        if len(X) == 0:
            raise ValueError(f"{self.name}: nothing to fit on")
        estimator = self._estimator()
        estimator.fit(X, np.asarray(y, dtype=float))
        # Only after a successful fit: a failed refit must not leave a stale model usable.
        self._fitted, self._columns = estimator, list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fitted is None or self._columns is None:
            raise NotFittedError(f"{self.name}: predict called before fit")
        if list(X.columns) != self._columns:
            raise ValueError(
                f"{self.name}: predict columns {list(X.columns)} differ from "
                f"the fitted ones {self._columns}"
            )
        return np.asarray(self._output(self._fitted, X), dtype=float)

    def _output(self, fitted: Any, X: pd.DataFrame) -> Any:
        """The score of a fitted estimator; classifiers override it with a probability."""
        return fitted.predict(X)

    def coefficients(self) -> pd.Series | None:
        """Per-feature weights of a fitted linear estimator; None for anything else.

        Not part of :class:`Model`: an optional window into what was learnt,
        which the walk-forward harness records per fold when it exists.
        """
        if self._fitted is None or self._columns is None:
            raise NotFittedError(f"{self.name}: coefficients asked for before fit")
        coef = getattr(self._fitted, "coef_", None)
        if coef is None:
            return None
        return pd.Series(np.ravel(coef).astype(float), index=self._columns, name=self.name)

    def importances(self) -> pd.Series | None:
        """Per-feature importance of a fitted tree estimator, summing to 1; None otherwise.

        Whatever the estimator reports as ``feature_importances_`` (LightGBM is
        built to report total split *gain*), normalised so folds of different
        sizes are comparable. Like :meth:`coefficients`, optional and not part of
        :class:`Model`.
        """
        if self._fitted is None or self._columns is None:
            raise NotFittedError(f"{self.name}: importances asked for before fit")
        raw = getattr(self._fitted, "feature_importances_", None)
        if raw is None:
            return None
        values = np.asarray(raw, dtype=float)
        total = values.sum()
        share = values / total if total > 0 else np.zeros_like(values)
        return pd.Series(share, index=self._columns, name=self.name)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{type(self).__name__}({args})"
