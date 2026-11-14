"""Learned models and the registry that makes them swappable.

Importing this package registers every model; use :func:`create` by name rather
than importing a model class.
"""

from finmgr.models import boosting, linear  # noqa: F401  (registers "lgbm", "ridge")
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
