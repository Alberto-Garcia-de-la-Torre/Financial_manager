"""Day 39: how lucky was that? Block-bootstrap intervals on Sharpe.

A Sharpe ratio from one backtest is one draw. Resampling the daily excess
returns says how far that draw could have landed with the same strategy and
different luck. Plain i.i.d. resampling would scramble volatility clustering
and the autocorrelation a 21-session holding period puts into returns, and so
understate the spread; the **circular block bootstrap** (Politis & Romano
1992) resamples whole runs of `block` consecutive sessions instead, wrapping
around the end so every session is equally likely to be drawn.

* :func:`bootstrap_sharpe` — percentile interval on one strategy's Sharpe.
* :func:`bootstrap_sharpe_difference` — the same on the *difference* between
  two strategies, resampling both with the same blocks so the correlation
  between them is kept. That, not two overlapping intervals, is how to ask
  whether a model beats the bar.
* :func:`familywise_level` — the level an interval needs when it is the best
  of `n` variants tried (Bonferroni): with :mod:`finmgr.backtest.trials`
  counting the variants, the winner's interval widens as the count grows.

Annualisation matches :func:`finmgr.backtest.metrics.sharpe_ratio`: mean over
sample standard deviation (ddof=1), times √periods, so the point estimate
here is exactly the one in the metrics table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finmgr.backtest.engine import BacktestResult
from finmgr.backtest.metrics import SESSIONS_PER_YEAR, excess_returns, session_returns

#: One holding period: long enough to keep a rebalance cycle's autocorrelation.
BLOCK = 21
N_BOOT = 5000
LEVEL = 0.95
SEED = 39
#: Resamples per chunk, to bound memory at chunk x len(returns) indices.
_CHUNK = 500


@dataclass(frozen=True)
class SharpeInterval:
    """A Sharpe ratio and the bootstrap interval around it."""

    point: float
    low: float
    high: float
    level: float
    block: int
    n_boot: int
    #: Bootstrap standard error: the standard deviation of the resampled Sharpes.
    se: float
    #: Share of resamples at or below zero: a one-sided bootstrap p-value against 0.
    p_le_zero: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def __str__(self) -> str:
        return f"{self.point:.2f} [{self.low:.2f}, {self.high:.2f}]"


def _as_array(excess: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(excess, dtype="float64")
    if values.ndim != 1:
        raise ValueError("excess returns must be one-dimensional")
    if np.isnan(values).any():
        raise ValueError("excess returns contain NaN; align the risk-free series first")
    if len(values) < 2:
        raise ValueError("need at least two returns")
    return values


def block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """`n_boot` x `n` circular-block resample indices into a series of length `n`.

    Each row is `ceil(n / block)` blocks of `block` consecutive positions,
    each starting at a uniform random position and wrapping past the end,
    truncated to `n`.
    """
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    block = min(block, n)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, n_blocks * block)[:, :n]


def _sharpe_rows(samples: np.ndarray, periods: int) -> np.ndarray:
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std * math.sqrt(periods), np.nan)


def _point(values: np.ndarray, periods: int) -> float:
    return float(_sharpe_rows(values[None, :], periods)[0])


def _resampled(
    columns: list[np.ndarray], *, block: int, n_boot: int, seed: int, periods: int
) -> list[np.ndarray]:
    """Bootstrap Sharpes for each column, every column resampled with the same indices."""
    n = len(columns[0])
    rng = np.random.default_rng(seed)
    out: list[list[np.ndarray]] = [[] for _ in columns]
    for done in range(0, n_boot, _CHUNK):
        idx = block_indices(n, block, min(_CHUNK, n_boot - done), rng)
        for k, values in enumerate(columns):
            out[k].append(_sharpe_rows(values[idx], periods))
    return [np.concatenate(parts) for parts in out]


def _interval(point: float, draws: np.ndarray, *, level: float, block: int) -> SharpeInterval:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    draws = draws[~np.isnan(draws)]
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return SharpeInterval(
        point=point,
        low=float(low),
        high=float(high),
        level=level,
        block=block,
        n_boot=len(draws),
        se=float(draws.std(ddof=1)),
        p_le_zero=float((draws <= 0).mean()),
    )


def bootstrap_sharpe(
    excess: pd.Series | np.ndarray,
    *,
    periods: int = SESSIONS_PER_YEAR,
    block: int = BLOCK,
    n_boot: int = N_BOOT,
    level: float = LEVEL,
    seed: int = SEED,
) -> SharpeInterval:
    """Percentile circular-block-bootstrap interval on the annualised Sharpe of `excess`."""
    values = _as_array(excess)
    (draws,) = _resampled([values], block=block, n_boot=n_boot, seed=seed, periods=periods)
    return _interval(_point(values, periods), draws, level=level, block=block)


def bootstrap_sharpe_difference(
    a: pd.Series,
    b: pd.Series,
    *,
    periods: int = SESSIONS_PER_YEAR,
    block: int = BLOCK,
    n_boot: int = N_BOOT,
    level: float = LEVEL,
    seed: int = SEED,
) -> SharpeInterval:
    """Interval on Sharpe(a) - Sharpe(b), both resampled on the same blocks of dates.

    `a` and `b` must share their index exactly. `p_le_zero` is then the
    bootstrap probability that `a` is no better than `b`.
    """
    if not a.index.equals(b.index):
        raise ValueError("a and b must share one date index")
    va, vb = _as_array(a), _as_array(b)
    da, db = _resampled([va, vb], block=block, n_boot=n_boot, seed=seed, periods=periods)
    return _interval(_point(va, periods) - _point(vb, periods), da - db, level=level, block=block)


def familywise_level(level: float, n_variants: int) -> float:
    """Per-interval level so that `n_variants` intervals jointly hold at `level` (Bonferroni)."""
    if n_variants < 1:
        raise ValueError(f"n_variants must be >= 1, got {n_variants}")
    return 1 - (1 - level) / n_variants


def result_excess(result: BacktestResult, rf: pd.Series | None) -> pd.Series:
    """The daily excess returns a backtest's Sharpe is computed from."""
    return excess_returns(session_returns(result.curve), rf)
