"""Day 41: models by name, so the CLI can swap them with ``--model``.

A model module registers its class with :func:`register`; everything else —
``finmgr train --model``, the walk-forward harness, the ensemble — goes through
:func:`create` and never imports a model class directly. Adding a model is one
new module and one import line in :mod:`finmgr.models`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from finmgr.models.base import Model

_REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


class UnknownModel(KeyError):
    """No model is registered under the requested name."""

    def __str__(self) -> str:  # KeyError would quote the whole message
        return str(self.args[0])


def register(name: str) -> Callable[[T], T]:
    """Class decorator: make ``cls`` available as ``create(name)``."""

    def decorate(cls: T) -> T:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"model name {name!r} is already registered to {_REGISTRY[name]}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorate


def available() -> list[str]:
    """Every registered model name, sorted."""
    return sorted(_REGISTRY)


def resolve(name: str) -> type:
    """The class registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownModel(
            f"unknown model {name!r}; choose from {', '.join(available()) or '(none)'}"
        ) from None


def create(name: str, **params: Any) -> Model:
    """A fresh, unfitted instance of model ``name`` with ``params`` over its defaults."""
    model = resolve(name)(**params)
    if not isinstance(model, Model):  # pragma: no cover - a registration bug, not user error
        raise TypeError(f"{name!r} does not implement the Model protocol")
    return model
