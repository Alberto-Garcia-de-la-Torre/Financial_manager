"""Learned models and the registry that makes them swappable.

Importing this package registers every model; use :func:`create` by name rather
than importing a model class.
"""

# Registers "lgbm", "ridge", and the day 44 classifiers "logit" and "lgbm_clf".
from finmgr.models import boosting, classifiers, linear  # noqa: F401
from finmgr.models.base import EstimatorModel, Model, NotFittedError
from finmgr.models.registry import UnknownModel, available, create, register, resolve

__all__ = [
    "EstimatorModel",
    "Model",
    "NotFittedError",
    "UnknownModel",
    "available",
    "create",
    "register",
    "resolve",
]
