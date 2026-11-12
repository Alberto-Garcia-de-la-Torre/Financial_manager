"""Day 39: block-bootstrap intervals on Sharpe and the variant ledger.

The acceptance criterion — the best baseline's Sharpe stated as an interval —
is met in `docs/baselines.md`, which needs the local store. These tests pin
the machinery: the bootstrap's point is the metrics module's Sharpe, its
blocks are circular runs of consecutive sessions, its 95% interval covers a
known true Sharpe about 95% of the time, blocks widen it on autocorrelated
returns, paired differences cancel what two series share, and the ledger
counts variants, not runs.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from finmgr.backtest import bootstrap as bs
from finmgr.backtest import trials
from finmgr.backtest.metrics import sharpe_ratio


def series(values: np.ndarray) -> pd.Series:
    days = pd.bdate_range("2020-01-01", periods=len(values)).date
    return pd.Series(values, index=pd.Index(days))


def normal(n: int, mean: float = 0.0004, sd: float = 0.01, seed: int = 0) -> pd.Series:
    return series(np.random.default_rng(seed).normal(mean, sd, n))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def test_point_is_the_metrics_sharpe():
    x = normal(1500)
    iv = bs.bootstrap_sharpe(x, periods=260, n_boot=200)
    assert iv.point == pytest.approx(sharpe_ratio(x, 260), rel=1e-12)
    assert iv.low < iv.point < iv.high
    assert iv.contains(iv.point)
    assert str(iv) == f"{iv.point:.2f} [{iv.low:.2f}, {iv.high:.2f}]"


def test_block_indices_are_circular_runs():
    rng = np.random.default_rng(1)
    idx = bs.block_indices(10, 4, 50, rng)
    assert idx.shape == (50, 10)
    assert idx.min() >= 0 and idx.max() <= 9
    for row in idx:
        for start in (0, 4):  # whole blocks; the third is truncated to two
            block = row[start : start + 4]
            assert ((np.diff(block) % 10) == 1).all()
    # Wrapping happens: some block runs past 9 back to 0.
    assert any(((row[:4] == 9) & (np.roll(row[:4], -1) == 0))[:3].any() for row in idx)
    # A block longer than the series is the series, rotated.
    whole = bs.block_indices(5, 50, 3, rng)
    for row in whole:
        assert sorted(row) == list(range(5))
    with pytest.raises(ValueError):
        bs.block_indices(5, 0, 1, rng)


def test_reproducible_by_seed():
    x = normal(800)
    a = bs.bootstrap_sharpe(x, n_boot=300, seed=5)
    b = bs.bootstrap_sharpe(x, n_boot=300, seed=5)
    c = bs.bootstrap_sharpe(x, n_boot=300, seed=6)
    assert a == b
    assert (a.low, a.high) != (c.low, c.high)


def test_95_interval_covers_the_true_sharpe_about_95_percent():
    # i.i.d. normal daily returns: the true annualised Sharpe is known exactly.
    mean, sd, periods = 0.0005, 0.01, 252
    true = mean / sd * math.sqrt(periods)
    hits = 0
    sims = 200
    for k in range(sims):
        x = normal(1000, mean, sd, seed=1000 + k)
        iv = bs.bootstrap_sharpe(x, periods=periods, block=5, n_boot=400, seed=k)
        hits += iv.contains(true)
    assert 0.88 <= hits / sims <= 0.99


def test_blocks_widen_the_interval_on_autocorrelated_returns():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 0.01, 3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):  # AR(1), phi 0.5: positive autocorrelation
        x[t] = 0.5 * x[t - 1] + e[t]
    s = series(x + 0.0003)
    iid = bs.bootstrap_sharpe(s, block=1, n_boot=1000)
    block = bs.bootstrap_sharpe(s, block=21, n_boot=1000)
    assert block.se > 1.3 * iid.se
    assert block.high - block.low > iid.high - iid.low


def test_higher_level_is_wider():
    x = normal(1000)
    narrow = bs.bootstrap_sharpe(x, n_boot=1000, level=0.9)
    wide = bs.bootstrap_sharpe(x, n_boot=1000, level=0.99)
    assert wide.low < narrow.low and wide.high > narrow.high


def test_rejects_nan_and_bad_level():
    x = normal(100)
    x.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        bs.bootstrap_sharpe(x)
    with pytest.raises(ValueError, match="level"):
        bs.bootstrap_sharpe(normal(100), level=1.0, n_boot=10)


def test_difference_with_itself_is_exactly_zero():
    x = normal(1000)
    iv = bs.bootstrap_sharpe_difference(x, x, n_boot=200)
    assert iv.point == 0 and iv.low == 0 and iv.high == 0


def test_paired_difference_is_tighter_than_the_marginals():
    # b is a plus a little noise of its own: they share almost all their luck.
    a = normal(2000, seed=1)
    b = a - 0.0002 + normal(2000, mean=0, sd=0.002, seed=2)
    diff = bs.bootstrap_sharpe_difference(a, b, n_boot=1000)
    ia = bs.bootstrap_sharpe(a, n_boot=1000)
    assert diff.point == pytest.approx(sharpe_ratio(a) - sharpe_ratio(b))
    assert diff.high - diff.low < (ia.high - ia.low) / 3
    assert diff.low > 0  # a's edge is visible once the shared luck cancels
    assert diff.p_le_zero < 0.025
    with pytest.raises(ValueError, match="index"):
        bs.bootstrap_sharpe_difference(a, b.iloc[1:])


def test_familywise_level():
    assert bs.familywise_level(0.95, 1) == pytest.approx(0.95)
    assert bs.familywise_level(0.95, 5) == pytest.approx(0.99)
    assert bs.familywise_level(0.95, 50) == pytest.approx(0.999)
    with pytest.raises(ValueError):
        bs.familywise_level(0.95, 0)


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_variant_id_is_stable_and_order_free():
    a = trials.variant_id("momentum", {"top_k": 10, "hold": 21})
    b = trials.variant_id("momentum", {"hold": 21, "top_k": 10})
    assert a == b and len(a) == 12
    assert trials.variant_id("momentum", {"top_k": 11, "hold": 21}) != a
    assert trials.variant_id("reversal", {"top_k": 10, "hold": 21}) != a


def test_ledger_counts_variants_not_runs(tmp_path):
    path = tmp_path / "trials.csv"
    assert trials.count_variants(path) == 0
    day1, day2 = dt.date(2026, 11, 13), dt.date(2026, 11, 14)
    vid = trials.record("m", {"k": 10}, sharpe=0.9, low=0.4, high=1.4, today=day1, path=path)
    trials.record("m", {"k": 10}, sharpe=0.95, today=day2, path=path)  # rerun
    assert trials.count_variants(path) == 1
    trials.record("m", {"k": 20}, sharpe=0.7, today=day2, path=path)  # new variant
    trials.record("r", {"k": 10}, sharpe=0.1, today=day2, path=path)  # another
    assert trials.count_variants(path) == 3

    ledger = trials.read_ledger(path)
    row = ledger.set_index("id").loc[vid]
    assert row["runs"] == 2
    assert row["first_logged"] == "2026-11-13" and row["last_logged"] == "2026-11-14"
    assert row["sharpe"] == pytest.approx(0.95)
    assert list(ledger.columns) == trials.COLUMNS
