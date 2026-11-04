"""Day 26: SMA and EMA at 20/50/200, distance, slope and golden/death crosses.

The acceptance criterion — a chart of one ticker's price with its three moving
averages looks visually correct — is `notebooks/day26_trend.ipynb`. What makes
a chart look right is pinned down here on synthetic prices: the averages are
the definitions by hand, they lag a trend in the right order, they are smoother
the longer they are, and nothing after t is read.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from finmgr.data.pit import LookAheadError
from finmgr.features.returns import venue_sessions
from finmgr.features.trend import (
    MA_WINDOWS,
    TREND_COLUMNS,
    compute_trend,
    crosses,
    ema,
    read_trend,
    sma,
)

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2018, 1, 1), D(2024, 6, 28))
RATIOS = [c for c in TREND_COLUMNS if not c.startswith(("sma_", "ema_"))]


def walk(n: int, seed: int = 26, drift: float = 0.0003, vol: float = 0.015) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def panel(prices: np.ndarray | None = None, tickers: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    frames = []
    for i, t in enumerate(tickers):
        p = walk(600, seed=i) if prices is None else prices
        frames.append(pd.DataFrame({"ticker": t, "date": SESSIONS[: len(p)], "adj_close_eur": p}))
    return pd.concat(frames, ignore_index=True)


def one(frame: pd.DataFrame, ticker: str = "AAA") -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].set_index("date")


# ---------------------------------------------------------------------------
# The definitions, by hand
# ---------------------------------------------------------------------------


def test_sma_is_the_plain_mean_of_the_last_n_prices() -> None:
    p = pd.Series(np.arange(1.0, 31.0))
    out = sma(p, 20)
    assert out.iloc[:19].isna().all()
    assert out.iloc[19] == pytest.approx(10.5)  # mean of 1..20
    assert out.iloc[29] == pytest.approx(20.5)  # mean of 11..30


def test_ema_is_seeded_with_the_sma_then_smoothed_with_two_over_n_plus_one() -> None:
    p = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 6.0])
    out = ema(p, 3)
    alpha = 0.5
    seed = 2.0
    e3 = alpha * 4 + (1 - alpha) * seed
    e4 = alpha * 10 + (1 - alpha) * e3
    e5 = alpha * 6 + (1 - alpha) * e4
    np.testing.assert_allclose(out.to_numpy(), [np.nan, np.nan, seed, e3, e4, e5])


def test_ema_after_its_seed_matches_pandas_recursive_ewm() -> None:
    p = pd.Series(walk(400))
    out = ema(p, 50)
    reference = pd.concat([pd.Series([p.iloc[:50].mean()]), p.iloc[50:]], ignore_index=True)
    expected = reference.ewm(alpha=2 / 51, adjust=False).mean().to_numpy()
    np.testing.assert_allclose(out.iloc[49:].to_numpy(), expected, rtol=1e-12)


def test_ema_steps_over_a_missing_price() -> None:
    p = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    out = ema(p, 3)
    assert out.iloc[3] == pytest.approx(2.0)  # carried, not NaN and not moved
    assert out.iloc[4] == pytest.approx(0.5 * 5 + 0.5 * 2.0)


def test_distance_and_slope_by_hand() -> None:
    out = one(compute_trend(panel()))
    t, back = SESSIONS[400], SESSIONS[395]
    row = out.loc[t]
    for kind in ("sma", "ema"):
        for w in MA_WINDOWS:
            ma = out[f"{kind}_{w}"]
            assert row[f"dist_{kind}_{w}"] == pytest.approx(row["adj_close_eur"] / ma[t] - 1)
            assert row[f"slope_{kind}_{w}"] == pytest.approx(ma[t] / ma[back] - 1)


# ---------------------------------------------------------------------------
# What a correct chart looks like
# ---------------------------------------------------------------------------


def test_in_a_steady_uptrend_the_averages_lag_in_order() -> None:
    # Price rises 0.1% a session: every average sits below it, the longer the
    # further, and every slope and distance is positive.
    p = 100 * 1.001 ** np.arange(600)
    row = one(compute_trend(panel(p))).iloc[-1]
    for kind in ("sma", "ema"):
        assert row["adj_close_eur"] > row[f"{kind}_20"] > row[f"{kind}_50"] > row[f"{kind}_200"]
        for w in MA_WINDOWS:
            assert row[f"dist_{kind}_{w}"] > 0
            assert row[f"slope_{kind}_{w}"] > 0
    # An SMA of a geometric series grows exactly as fast as the series.
    for w in MA_WINDOWS:
        assert row[f"slope_sma_{w}"] == pytest.approx(1.001**5 - 1, rel=1e-9)
    # p / SMA of a geometric series is n / sum(1.001^-k) for k in 0..n-1.
    for w in MA_WINDOWS:
        exact = w / sum(1.001**-k for k in range(w)) - 1
        assert row[f"dist_sma_{w}"] == pytest.approx(exact, rel=1e-9)


def test_longer_averages_are_smoother() -> None:
    out = one(compute_trend(panel())).iloc[200:]
    for kind in ("sma", "ema"):
        wiggle = [out[f"{kind}_{w}"].diff().abs().sum() for w in MA_WINDOWS]
        assert out["adj_close_eur"].diff().abs().sum() > wiggle[0] > wiggle[1] > wiggle[2]


def test_a_constant_price_is_its_own_average() -> None:
    row = one(compute_trend(panel(np.full(300, 42.0)))).iloc[-1]
    for kind in ("sma", "ema"):
        for w in MA_WINDOWS:
            assert row[f"{kind}_{w}"] == pytest.approx(42.0)
            assert row[f"dist_{kind}_{w}"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Crosses
# ---------------------------------------------------------------------------


def test_crosses_flag_the_bar_the_regime_changes_and_nothing_else() -> None:
    fast = pd.Series([np.nan, 1.0, 1.0, 3.0, 3.0, 1.0, np.nan, 3.0])
    slow = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    out = crosses(fast, slow)
    np.testing.assert_array_equal(
        out["trend_50_200"].to_numpy(), [np.nan, -1, -1, 1, 1, -1, np.nan, 1]
    )
    # The first defined bar is not a cross; the one after a NaN gap still is.
    np.testing.assert_array_equal(
        out["golden_cross"].to_numpy(), [np.nan, 0, 0, 1, 0, 0, np.nan, 1]
    )
    np.testing.assert_array_equal(out["death_cross"].to_numpy(), [np.nan, 0, 0, 0, 0, 1, np.nan, 0])


def test_a_down_then_up_market_gives_one_death_cross_then_one_golden_cross() -> None:
    p = np.concatenate(
        [100 * 1.002 ** np.arange(300), 100 * 1.002**299 * 0.998 ** np.arange(1, 301)]
    )
    p = np.concatenate([p, p[-1] * 1.002 ** np.arange(1, 401)])
    out = one(compute_trend(panel(p)))
    golden, death = out.index[out["golden_cross"] == 1], out.index[out["death_cross"] == 1]
    assert len(death) == 1 and len(golden) == 1
    assert SESSIONS[300] < death[0] < SESSIONS[600] < golden[0]
    assert out.loc[death[0], "sma_50"] < out.loc[death[0], "sma_200"]
    assert out.loc[golden[0], "sma_50"] > out.loc[golden[0], "sma_200"]


# ---------------------------------------------------------------------------
# History, look-ahead, gaps, schema
# ---------------------------------------------------------------------------


def test_not_enough_history_is_nan() -> None:
    out = one(compute_trend(panel()))
    assert out["sma_200"].iloc[:199].isna().all()
    assert out["sma_200"].iloc[199:].notna().all()
    assert out["slope_sma_200"].iloc[:204].isna().all()
    assert out["golden_cross"].iloc[:199].isna().all()


@pytest.mark.parametrize("cut", [30, 210, 450, 599])
def test_trend_at_t_uses_no_price_later_than_t(cut: int) -> None:
    prices = panel(tickers=("AAA", "BBB"))
    t = SESSIONS[cut]
    before = compute_trend(prices)
    scrambled = prices.copy()
    later = scrambled["date"] > t
    scrambled.loc[later, "adj_close_eur"] *= np.random.default_rng(0).uniform(0.2, 5, later.sum())
    after = compute_trend(scrambled)
    upto = before["date"] <= t
    pd.testing.assert_frame_equal(before.loc[upto], after.loc[upto])


def test_tickers_never_borrow_each_others_prices() -> None:
    prices = panel(tickers=("AAA", "BBB"))
    together = one(compute_trend(prices), "BBB")[list(TREND_COLUMNS)].to_numpy()
    alone = one(compute_trend(prices[prices["ticker"] == "BBB"]), "BBB")[list(TREND_COLUMNS)]
    np.testing.assert_array_equal(together, alone.to_numpy())


def test_one_missing_session_does_not_blank_the_200_session_average() -> None:
    prices = panel()
    holed = prices[prices["date"] != SESSIONS[400]]
    out = one(compute_trend(holed, sessions={"AAA": SESSIONS[:600]}))
    p = one(prices)["adj_close_eur"].drop(SESSIONS[400])
    t = SESSIONS[450]
    assert out.loc[t, "sma_200"] == pytest.approx(p.loc[SESSIONS[251] : t].mean(), rel=1e-12)
    assert out[list(RATIOS)].loc[SESSIONS[401] :].notna().all().all()
    assert SESSIONS[400] not in out.index


def test_ratios_do_not_move_when_every_price_is_rescaled() -> None:
    # A later split back-adjusts all earlier prices by one factor: the levels
    # scale with it, the ratios and the crosses do not.
    prices = panel()
    scaled = prices.assign(adj_close_eur=prices["adj_close_eur"] / 3)
    a, b = compute_trend(prices), compute_trend(scaled)
    pd.testing.assert_frame_equal(a[RATIOS], b[RATIOS], rtol=1e-10)
    np.testing.assert_allclose(a["sma_200"] / 3, b["sma_200"], rtol=1e-10)


def test_a_duplicated_row_is_refused() -> None:
    prices = panel()
    with pytest.raises(ValueError):
        compute_trend(pd.concat([prices, prices.iloc[:1]]))


def test_empty_input_gives_the_full_schema() -> None:
    out = compute_trend(panel().iloc[:0])
    assert list(out.columns) == ["ticker", "date", "adj_close_eur", *TREND_COLUMNS]


def test_read_trend_needs_as_of_and_refuses_the_future() -> None:
    with pytest.raises(TypeError):
        read_trend("AAA")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_trend("AAA", as_of=SESSIONS[10], start=SESSIONS[20])
