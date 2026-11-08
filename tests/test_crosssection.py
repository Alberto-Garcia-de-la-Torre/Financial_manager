"""Day 29: cross-sectional z-scores and percentile ranks.

The acceptance criterion — each date's ranks are uniform on [0, 1] by
construction — is :func:`test_each_dates_ranks_are_the_uniform_grid`: on a
synthetic panel of 300 dates whose size, missing values and distribution change
from date to date, every date's sorted ranks are exactly the n-point grid
`0, 1/(n-1), ..., 1`. The rest pins down ties, masks, z-scores and the fact
that a date never sees another date's values.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from finmgr.features.crosssection import (
    FEATURE_COLUMNS,
    MIN_NAMES,
    cross_rank,
    cross_zscore,
    normalize,
    rank_column,
    z_column,
)

D = dt.date
DATES = [D(2020, 1, 1) + dt.timedelta(days=i) for i in range(300)]
TICKERS = [f"T{i:03d}" for i in range(100)]


def synthetic_panel(seed: int = 7) -> pd.DataFrame:
    """100 names x 300 dates; each date draws its own distribution and NaN share.

    `rsi_14` is bounded (a logistic, so no ties at 0 or 100), `vol_63d` is
    lognormal, `ret_5d` is fat-tailed and `adv_eur_21` spans five orders of
    magnitude, so uniform ranks cannot come
    from the inputs already being uniform.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for day in DATES:
        n = len(TICKERS)
        frame = pd.DataFrame(
            {
                "date": day,
                "ticker": TICKERS,
                "rsi_14": 100 / (1 + np.exp(-rng.normal(rng.uniform(-1, 1), 0.8, n))),
                "vol_63d": np.exp(rng.normal(np.log(0.25), rng.uniform(0.1, 1.0), n)),
                "ret_5d": rng.standard_t(3, n) * 0.02,
                "adv_eur_21": 10 ** rng.uniform(5, 10, n),
                "tradable": rng.random(n) > 0.2,
            }
        )
        # A date-specific share of missing values, up to 60% of the names.
        for col in ("rsi_14", "vol_63d", "ret_5d"):
            frame.loc[rng.random(n) < rng.uniform(0, 0.6), col] = np.nan
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed)


FEATURES = ["rsi_14", "vol_63d", "ret_5d", "adv_eur_21"]


def test_each_dates_ranks_are_the_uniform_grid():
    panel = synthetic_panel()
    out = normalize(panel, FEATURES)
    checked = 0
    for col in FEATURES:
        for day, rows in out.groupby("date"):
            raw = rows[col].dropna()
            ranks = rows[rank_column(col)].dropna().sort_values().to_numpy()
            assert len(ranks) == len(raw), (col, day)
            if len(raw) < MIN_NAMES:
                continue
            n = len(raw)
            # Uniform on [0, 1] by construction: exactly the n-point grid.
            np.testing.assert_allclose(ranks, np.linspace(0.0, 1.0, n), rtol=0, atol=1e-12)
            assert ranks[0] == 0.0 and ranks[-1] == 1.0
            assert abs(ranks.mean() - 0.5) < 1e-12
            checked += 1
    assert checked == len(FEATURES) * len(DATES)


def test_ranks_pooled_over_dates_are_uniform():
    """Pooled over dates, the ranks fill the deciles exactly as the per-date grids do.

    The grid includes both 0 and 1, so with ~40-100 names a date the end deciles
    hold a point or so more than 10%. That is the discreteness of the grid, not a
    tilt, and it is the same for any feature: the decile counts are a function of
    each date's n alone.
    """
    out = normalize(synthetic_panel(seed=11), ["vol_63d"])
    col = rank_column("vol_63d")
    bins = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(out[col].dropna(), bins=bins)
    sizes = out.groupby("date")[col].count()
    # i / (n - 1), the module's own arithmetic: grid points such as 0.3 sit on a
    # bin edge, where np.linspace's rounding would put them in the other bin.
    grids = (np.arange(n) / (n - 1) for n in sizes if n > 1)
    expected = sum(np.histogram(grid, bins=bins)[0] for grid in grids)
    np.testing.assert_array_equal(counts, expected)
    share = counts / counts.sum()
    assert np.all(np.abs(share - 0.1) < 0.01), share


def test_rank_orders_like_the_feature_and_ignores_monotone_transforms():
    panel = synthetic_panel(seed=3)
    panel["vol_cubed"] = panel["vol_63d"] ** 3
    out = normalize(panel, ["vol_63d", "vol_cubed"])
    pd.testing.assert_series_equal(
        out[rank_column("vol_63d")], out[rank_column("vol_cubed")], check_names=False
    )
    one = out[out["date"] == DATES[0]].dropna(subset=["vol_63d"]).sort_values("vol_63d")
    assert one[rank_column("vol_63d")].is_monotonic_increasing


def test_lowest_rsi_of_the_day_ranks_zero():
    panel = synthetic_panel(seed=5)
    out = normalize(panel, ["rsi_14"])
    for _, rows in out.groupby("date"):
        rows = rows.dropna(subset=["rsi_14"])
        lowest = rows.loc[rows["rsi_14"].idxmin(), rank_column("rsi_14")]
        highest = rows.loc[rows["rsi_14"].idxmax(), rank_column("rsi_14")]
        assert lowest == 0.0 and highest == 1.0


def test_ties_share_the_average_grid_point():
    dates = pd.Series([DATES[0]] * 5)
    values = pd.Series([1.0, 2.0, 2.0, 3.0, np.nan])
    ranks = cross_rank(values, dates, min_names=1)
    np.testing.assert_allclose(ranks[:4], [0.0, 0.5, 0.5, 1.0])
    assert np.isnan(ranks[4])
    flat = cross_rank(pd.Series([4.0, 4.0, 4.0]), dates[:3], min_names=1)
    np.testing.assert_allclose(flat, [0.5, 0.5, 0.5])


def test_a_lone_name_ranks_in_the_middle():
    assert cross_rank(pd.Series([7.0]), pd.Series([DATES[0]]), min_names=1).iloc[0] == 0.5


def test_zscores_have_mean_zero_and_unit_std_per_date():
    out = normalize(synthetic_panel(seed=9), ["ret_5d", "rsi_14"])
    for col in ("ret_5d", "rsi_14"):
        stats = out.groupby("date")[z_column(col)].agg(["mean", lambda s: s.std(ddof=0)])
        np.testing.assert_allclose(stats["mean"], 0.0, atol=1e-12)
        np.testing.assert_allclose(stats.iloc[:, 1], 1.0, atol=1e-12)


def test_zscore_by_hand_and_on_a_flat_day():
    dates = pd.Series([DATES[0]] * 4 + [DATES[1]] * 3)
    values = pd.Series([1.0, 2.0, 3.0, 6.0, 5.0, 5.0, 5.0])
    z = cross_zscore(values, dates, min_names=1)
    x = np.array([1.0, 2.0, 3.0, 6.0])
    np.testing.assert_allclose(z[:4], (x - x.mean()) / x.std())
    np.testing.assert_allclose(z[4:], 0.0)
    clipped = cross_zscore(values, dates, min_names=1, clip=1.0)
    assert clipped.max() == 1.0 and clipped.min() >= -1.0


def test_log_columns_are_standardized_in_logs():
    panel = synthetic_panel(seed=13)
    out = normalize(panel, ["adv_eur_21"])
    day = out[out["date"] == DATES[0]]
    logs = np.log(day["adv_eur_21"])
    np.testing.assert_allclose(day[z_column("adv_eur_21")], (logs - logs.mean()) / logs.std(ddof=0))


def test_a_date_never_sees_another_dates_values():
    panel = synthetic_panel(seed=17)
    base = normalize(panel, FEATURES)
    scrambled = panel.copy()
    later = scrambled["date"] > DATES[150]
    rng = np.random.default_rng(0)
    for col in FEATURES:
        scrambled.loc[later, col] = rng.normal(1e6, 1e6, later.sum())
    moved = normalize(scrambled, FEATURES)
    cols = [rank_column(c) for c in FEATURES] + [z_column(c) for c in FEATURES]
    pd.testing.assert_frame_equal(base.loc[~later, cols], moved.loc[~later, cols])


def test_mask_drops_names_from_the_cross_section():
    panel = synthetic_panel(seed=19)
    out = normalize(panel, ["vol_63d"], mask="tradable")
    assert out.loc[~out["tradable"], rank_column("vol_63d")].isna().all()
    for _, rows in out[out["tradable"]].groupby("date"):
        ranks = rows[rank_column("vol_63d")].dropna().sort_values().to_numpy()
        np.testing.assert_allclose(ranks, np.linspace(0, 1, len(ranks)), atol=1e-12)


def test_thin_dates_are_not_ranked():
    panel = synthetic_panel(seed=23)
    first = panel["date"] == DATES[0]
    keep = panel.index[first][: MIN_NAMES - 1]
    panel.loc[first & ~panel.index.isin(keep), "rsi_14"] = np.nan
    out = normalize(panel, ["rsi_14"])
    assert out.loc[first, [rank_column("rsi_14"), z_column("rsi_14")]].isna().all().all()
    assert out.loc[~first, rank_column("rsi_14")].notna().any()


def test_rows_and_raw_columns_are_kept():
    panel = synthetic_panel(seed=29)
    out = normalize(panel, FEATURES)
    assert out.index.equals(panel.index)
    pd.testing.assert_frame_equal(out[panel.columns], panel)


def test_default_columns_are_the_comparable_features_present():
    panel = synthetic_panel(seed=31)
    out = normalize(panel)
    assert {rank_column(c) for c in FEATURES} <= set(out.columns)
    assert "tradable_rank" not in out.columns
    assert all(c in FEATURE_COLUMNS for c in FEATURES)
    for level in ("sma_200", "atr_14", "macd", "obv", "golden_cross", "tradable"):
        assert level not in FEATURE_COLUMNS


def test_infinite_values_are_left_out():
    dates = pd.Series([DATES[0]] * 4)
    panel = pd.DataFrame({"date": dates, "x": [1.0, np.inf, 2.0, 3.0]})
    out = normalize(panel, ["x"], min_names=1)
    np.testing.assert_allclose(out["x_rank"], [0.0, np.nan, 0.5, 1.0])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"columns": ["nope"]}, KeyError),
        ({"columns": ["date"]}, ValueError),
        ({"columns": ["rsi_14"], "mask": "ticker"}, TypeError),
        ({"columns": ["rsi_14"], "mask": "nope"}, KeyError),
        ({"columns": ["rsi_14"], "min_names": 0}, ValueError),
        ({"columns": ["rsi_14"], "clip": 0.0}, ValueError),
        ({"columns": ["rsi_14"], "by": "day"}, KeyError),
    ],
)
def test_bad_arguments_are_refused(kwargs, error):
    with pytest.raises(error):
        normalize(synthetic_panel(seed=1).head(200), **kwargs)


def test_no_default_feature_present_is_an_error():
    with pytest.raises(ValueError, match="FEATURE_COLUMNS"):
        normalize(pd.DataFrame({"date": [DATES[0]], "foo": [1.0]}))
