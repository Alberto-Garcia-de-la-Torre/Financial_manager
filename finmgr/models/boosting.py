"""LightGBM gradient boosting, through the same interface as ridge (day 43 trains it).

Defaults are deliberately conservative for a noisy cross-sectional target —
shallow trees, a slow learning rate, large leaves — and fully deterministic:
one thread and a fixed seed, so a rerun of the same variant reproduces the
same numbers and the trials ledger stays honest.
"""

from __future__ import annotations

from typing import Any, ClassVar

from lightgbm import LGBMRegressor

from finmgr.models.base import EstimatorModel
from finmgr.models.registry import register


@register("lgbm")
class LGBMModel(EstimatorModel):
    """LightGBM regressor. Tolerates missing features (NaN goes down a learned branch)."""

    defaults: ClassVar[dict[str, Any]] = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 0,
    }

    def _estimator(self) -> Any:
        return LGBMRegressor(**self.params, n_jobs=1, deterministic=True, verbose=-1)
