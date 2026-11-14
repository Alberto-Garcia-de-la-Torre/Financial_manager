"""Ridge regression: the linear model every later one has to beat (day 42 trains it)."""

from __future__ import annotations

from typing import Any, ClassVar

from sklearn.linear_model import Ridge

from finmgr.models.base import EstimatorModel
from finmgr.models.registry import register


@register("ridge")
class RidgeModel(EstimatorModel):
    """L2-penalised least squares. Expects finite features: rank or impute first."""

    defaults: ClassVar[dict[str, Any]] = {"alpha": 1.0, "fit_intercept": True}

    def _estimator(self) -> Any:
        return Ridge(**self.params)
