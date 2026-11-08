"""Cross-sectional normalization: each feature as a z-score and a [0, 1] rank across the day.

    from finmgr.features.crosssection import normalize

    panel = read_oscillators(as_of="2024-06-28", start="2024-01-01")
    ranked = normalize(panel, ["rsi_14", "bb_pctb"])
    ranked[ranked["date"] == day].sort_values("rsi_14_rank")   # the most oversold first

An RSI of 30 says little on its own; "the lowest RSI of the 100 today" is
something a portfolio can act on. For every feature `f` this adds two columns,
both computed one date at a time over the names that have a value that day:

    f_rank   (r - 1) / (n - 1), r the 1-based rank of the value among the n names
             (ties share their average rank): 0 for the lowest, 1 for the highest
    f_z      (x - mean) / std across the same n names (std with ddof=0)

**Uniform by construction.** With n distinct values the ranks of a date are
exactly `0, 1/(n-1), ..., 1`: the n-point uniform grid on [0, 1], whatever the
feature's distribution, so a rank means the same on a calm day and in a crash,
for RSI and for volatility. Tied values share the average of the grid points
they would cover, which keeps the date's mean rank at exactly 0.5. A lone name
ranks 0.5. Ranks are the robust column: a monotone transform of the feature
(logs, a sign-preserving squash) does not move them.

**z keeps the magnitude** the rank throws away, so it is also sensitive to
outliers: one name far out moves every other name's z. Pass `clip=` to cap
|z| (winsorizing after standardizing). Features in :data:`LOG_COLUMNS`
(positive and spanning orders of magnitude, like euro volume) are standardized
in logs; their ranks are unaffected. If every name has the same value, z is 0.

**Who is in the cross-section.** Only rows with a value, and, when `mask=` names
a boolean column (say `tradable` from :mod:`finmgr.features.liquidity`), only
rows where it is True; the others get NaN in both columns. A date with fewer
than `min_names` such rows (default :data:`MIN_NAMES`) gets NaN throughout, so
a feature is not ranked among three names in its first year of history.

**No look-ahead, no cross-date mixing.** A value dated t depends only on the
other names' values dated t, so normalization is exactly as backward-looking as
the raw features it reads. Comparing across dates is what the ranks remove; it
is never an input.

Flags (`golden_cross`, `trend_50_200`, `tradable`) and price-unit levels
(`sma_200`, `atr_14`, `macd`, `obv`) are not in :data:`FEATURE_COLUMNS`: a flag
has no meaningful rank and a level is not comparable across tickers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from finmgr.features.liquidity import ADV_COLUMN, OBV_FLOW_COLUMN, VOLUME_Z_COLUMN
from finmgr.features.momentum import MOMENTUM_COLUMNS
from finmgr.features.oscillators import MACD_PCT_COLUMNS, PCTB_COLUMN, RSI_COLUMN, STOCH_COLUMNS
from finmgr.features.returns import HORIZONS, simple_column
from finmgr.features.trend import MA_KINDS, MA_WINDOWS, dist_column, slope_column
from finmgr.features.volatility import (
    ATR_PCT_COLUMN,
    DOWNSIDE_COLUMN,
    DRAWDOWN_COLUMN,
    VOL_WINDOWS,
    vol_column,
)

#: Fewest names a date needs before any of them is ranked.
MIN_NAMES = 10

RANK_SUFFIX = "_rank"
Z_SUFFIX = "_z"

#: The raw features that compare across tickers: unit-free ratios, returns,
#: bounded oscillators and euro liquidity. What :func:`normalize` does by default.
FEATURE_COLUMNS: tuple[str, ...] = (
    *(simple_column(h) for h in HORIZONS),
    *MOMENTUM_COLUMNS,
    *(vol_column(w) for w in VOL_WINDOWS),
    DOWNSIDE_COLUMN,
    ATR_PCT_COLUMN,
    DRAWDOWN_COLUMN,
    *(dist_column(k, w) for k in MA_KINDS for w in MA_WINDOWS),
    *(slope_column(k, w) for k in MA_KINDS for w in MA_WINDOWS),
    RSI_COLUMN,
    *MACD_PCT_COLUMNS,
    *STOCH_COLUMNS,
    PCTB_COLUMN,
    ADV_COLUMN,
    VOLUME_Z_COLUMN,
    OBV_FLOW_COLUMN,
)

#: Positive features spanning orders of magnitude: their z-score is taken in logs.
LOG_COLUMNS: frozenset[str] = frozenset({ADV_COLUMN, "eur_volume"})


def rank_column(feature: str) -> str:
    return f"{feature}{RANK_SUFFIX}"


def z_column(feature: str) -> str:
    return f"{feature}{Z_SUFFIX}"


def _counts(values: pd.Series, dates: pd.Series) -> pd.Series:
    return values.notna().groupby(dates, sort=False).transform("sum")


def cross_rank(values: pd.Series, dates: pd.Series, *, min_names: int = MIN_NAMES) -> pd.Series:
    """`(r - 1) / (n - 1)` per date, average rank for ties, 0.5 for a lone name; NaN stays NaN."""
    ranks = values.groupby(dates, sort=False).rank(method="average")
    n = _counts(values, dates)
    out = (ranks - 1.0) / (n - 1.0).where(n > 1)
    out = out.mask((n == 1) & values.notna(), 0.5)
    return out.where(n >= min_names)


def cross_zscore(
    values: pd.Series,
    dates: pd.Series,
    *,
    min_names: int = MIN_NAMES,
    clip: float | None = None,
) -> pd.Series:
    """`(x - mean) / std` (ddof=0) per date; 0 where every name is equal; NaN stays NaN."""
    groups = values.groupby(dates, sort=False)
    n = _counts(values, dates)
    centred = values - groups.transform("mean")
    # Pandas' grouped std is ddof=1; rescale rather than run a Python lambda per date.
    many = n.where(n > 1)
    std = (groups.transform("std") * np.sqrt((many - 1) / many)).where(n > 1, 0.0)
    z = (centred / std.where(std > 0)).mask((std == 0) & values.notna(), 0.0)
    if clip is not None:
        z = z.clip(-clip, clip)
    return z.where(n >= min_names)


def normalize(
    panel: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    by: str = "date",
    mask: str | None = None,
    min_names: int = MIN_NAMES,
    clip: float | None = None,
    log_columns: Iterable[str] = LOG_COLUMNS,
) -> pd.DataFrame:
    """`panel` with `<f>_rank` and `<f>_z` added for every feature `f` in `columns`.

    `columns` defaults to every :data:`FEATURE_COLUMNS` present in `panel`.
    Each date (`by`) is normalized on its own, over the rows with a value and,
    if `mask` names a boolean column, where it is True. Rows and their order
    are kept; the raw columns are left as they are.
    """
    if by not in panel.columns:
        raise KeyError(f"panel has no {by!r} column")
    if min_names < 1:
        raise ValueError(f"min_names must be at least 1, got {min_names}")
    if clip is not None and not clip > 0:
        raise ValueError(f"clip must be positive, got {clip}")
    if columns is None:
        columns = [c for c in FEATURE_COLUMNS if c in panel.columns]
        if not columns:
            raise ValueError("panel has none of the default FEATURE_COLUMNS; pass columns=")
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing column(s): {', '.join(missing)}")
    if by in columns or (mask is not None and mask in columns):
        raise ValueError("cannot normalize the date or the mask column")
    if mask is not None:
        if mask not in panel.columns:
            raise KeyError(f"panel has no mask column {mask!r}")
        if panel[mask].dtype != bool:
            raise TypeError(f"mask column {mask!r} must be boolean, got {panel[mask].dtype}")
        keep = panel[mask]
    else:
        keep = pd.Series(True, index=panel.index)

    logs = set(log_columns)
    dates = panel[by]
    out = panel.copy()
    for col in columns:
        values = pd.to_numeric(panel[col], errors="raise").astype(float).where(keep)
        values = values.where(np.isfinite(values))
        out[rank_column(col)] = cross_rank(values, dates, min_names=min_names)
        scale = np.log(values.where(values > 0)) if col in logs else values
        out[z_column(col)] = cross_zscore(scale, dates, min_names=min_names, clip=clip)
    return out
