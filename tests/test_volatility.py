"""Day 25: realized volatility, downside deviation, ATR(14) and max drawdown.

The acceptance criterion — a large cap's annualized vol in a plausible 15-35%
band — is checked on the real store in `notebooks/day25_volatility.ipynb`.
Here the same scale check runs on synthetic prices with a known volatility,
next to the definitions by hand and the properties the notebook cannot show:
nothing after t is read, a missing bar does not blank a year, and a split is
not a true range.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data.actions import merge_actions
from finmgr.data.pit import LookAheadError
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS
from finmgr.features.returns import venue_sessions
from finmgr.features.volatility import (
    VOLATILITY_COLUMNS,
    compute_volatility,
    downside_deviation,
    max_drawdown,
    read_volatility,
    true_range,
    wilder_atr,
)

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2018, 1, 1), D(2024, 6, 28))
ROOT252 = math.sqrt(252)
RISK = ["vol_21d", "vol_63d", "vol_252d", "downdev_63d", "mdd_252d"]


def gbm(n: int, annual_vol: float, seed: int = 25, drift: float = 0.0003) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, annual_vol / ROOT252, n)))


def panel(
    n: int = 400, tickers: tuple[str, ...] = ("AAA", "BBB"), vol: float = 0.25
) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"ticker": t, "date": SESSIONS[:n], "adj_close_eur": gbm(n, vol, seed=i)})
        for i, t in enumerate(tickers)
    ]
    return pd.concat(frames, ignore_index=True)


def one(frame: pd.DataFrame, ticker: str = "AAA") -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].set_index("date")


# ---------------------------------------------------------------------------
# Scale: the acceptance criterion's shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("annual", [0.15, 0.25, 0.35])
def test_vol_recovers_the_volatility_the_prices_were_drawn_with(annual: float) -> None:
    # Ten years of a stock drawn at `annual` vol: the 252-session estimate,
    # averaged over its life, is that number — not 1/16th of it (daily, not
    # annualized) nor 16x it (annualized twice).
    out = one(compute_volatility(panel(n=1600, tickers=("AAA",), vol=annual)))
    assert out["vol_252d"].mean() == pytest.approx(annual, rel=0.05)
    assert out["vol_63d"].mean() == pytest.approx(annual, rel=0.07)
    assert out["vol_21d"].mean() == pytest.approx(annual, rel=0.1)


# ---------------------------------------------------------------------------
# Definitions, by hand
# ---------------------------------------------------------------------------


def test_vol_is_the_annualized_sample_std_of_daily_log_returns() -> None:
    prices = panel()
    p = one(prices)["adj_close_eur"].to_numpy()
    r = np.diff(np.log(p))  # r[k] is the return into session k+1
    out = one(compute_volatility(prices))
    for i in (252, 300, 399):
        t = SESSIONS[i]
        for w in (21, 63, 252):
            hand = np.std(r[i - w : i], ddof=1) * ROOT252
            assert out.loc[t, f"vol_{w}d"] == pytest.approx(hand, rel=1e-10)


def test_downside_deviation_counts_only_the_falls_but_averages_over_every_day() -> None:
    up = pd.Series(np.full(100, 0.01))
    assert downside_deviation(up, 63).iloc[-1] == 0.0
    a = 0.01
    zigzag = pd.Series([a, -a] * 50)
    # Half the days square to a^2, half to 0: sqrt(a^2 / 2), annualized.
    assert downside_deviation(zigzag, 63).iloc[-1] == pytest.approx(
        a / math.sqrt(2) * ROOT252, rel=0.02
    )
    prices = panel()
    r = np.diff(np.log(one(prices)["adj_close_eur"].to_numpy()))
    hand = math.sqrt(np.mean(np.minimum(r[300 - 63 : 300], 0) ** 2)) * ROOT252
    assert one(compute_volatility(prices)).loc[SESSIONS[300], "downdev_63d"] == pytest.approx(hand)


def test_true_range_includes_the_gap_from_yesterdays_close() -> None:
    high = pd.Series([10.0, 12.0, 9.0])
    low = pd.Series([9.0, 11.5, 8.0])
    close = pd.Series([9.5, 12.0, 8.5])
    tr = true_range(high, low, close)
    assert math.isnan(tr[0])  # no previous close
    assert tr[1] == pytest.approx(2.5)  # gap up: 12.0 - 9.5
    assert tr[2] == pytest.approx(4.0)  # gap down: 12.0 - 8.0


def test_wilder_atr_by_hand() -> None:
    tr = pd.Series([np.nan, 2.0, 4.0, 6.0, 3.0, 9.0])
    atr = wilder_atr(tr, period=3)
    assert atr.iloc[:3].isna().all()
    assert atr[3] == pytest.approx(4.0)  # mean of 2, 4, 6
    assert atr[4] == pytest.approx((4.0 * 2 + 3.0) / 3)
    assert atr[5] == pytest.approx((atr[4] * 2 + 9.0) / 3)
    # A constant true range is its own ATR.
    assert wilder_atr(pd.Series([1.5] * 40)).iloc[-1] == pytest.approx(1.5)


def test_wilder_atr_steps_over_a_missing_true_range() -> None:
    tr = pd.Series([2.0, 4.0, 6.0, np.nan, 3.0])
    atr = wilder_atr(tr, period=3)
    assert math.isnan(atr[3])
    assert atr[4] == pytest.approx((4.0 * 2 + 3.0) / 3)


def test_max_drawdown_finds_the_worst_peak_to_trough_in_the_window() -> None:
    path = pd.Series([100.0, 120.0, 110.0, 90.0, 115.0, 125.0, 100.0])
    mdd = max_drawdown(path, window=6)
    assert mdd.iloc[:6].isna().all()
    assert mdd.iloc[6] == pytest.approx(90.0 / 120.0 - 1)  # -25%, not 125 -> 100 (-20%)
    assert max_drawdown(pd.Series(np.arange(1.0, 30.0)), window=10).iloc[-1] == 0.0
    # A peak that has left the window no longer counts.
    later = max_drawdown(pd.Series([200.0, 100.0, 105.0, 101.0, 104.0]), window=3)
    assert later.iloc[4] == pytest.approx(101.0 / 105.0 - 1)


def test_not_enough_history_is_nan_not_a_shorter_window() -> None:
    out = one(compute_volatility(panel()))
    for column, span in [("vol_21d", 21), ("vol_63d", 63), ("vol_252d", 252)]:
        assert out[column].iloc[:span].isna().all()
        assert out[column].iloc[span:].notna().all()
    assert out["mdd_252d"].iloc[:252].isna().all()
    assert out["mdd_252d"].iloc[252:].notna().all()
    assert (out["mdd_252d"].dropna() <= 0).all()


def test_without_high_and_low_atr_is_nan_and_the_rest_still_computes() -> None:
    out = compute_volatility(panel())
    assert out["atr_14"].isna().all()
    assert out["vol_63d"].notna().any()


def test_a_duplicated_row_is_refused() -> None:
    prices = panel(n=10, tickers=("AAA",))
    with pytest.raises(ValueError, match="more than one row"):
        compute_volatility(pd.concat([prices, prices.iloc[[3]]]))


def test_empty_input_gives_the_full_schema() -> None:
    out = compute_volatility(pd.DataFrame({"ticker": [], "date": [], "adj_close_eur": []}))
    assert list(out.columns) == ["ticker", "date", "adj_close_eur", *VOLATILITY_COLUMNS]


# ---------------------------------------------------------------------------
# No look-ahead, no borrowed prices, gaps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [30, 100, 260, 399])
def test_risk_at_t_uses_no_observation_later_than_t(cut: int) -> None:
    prices = panel()
    t = SESSIONS[cut]
    before = compute_volatility(prices)
    scrambled = prices.copy()
    later = scrambled["date"] > t
    scrambled.loc[later, "adj_close_eur"] *= np.random.default_rng(0).uniform(0.2, 5, later.sum())
    after = compute_volatility(scrambled)
    upto = before["date"] <= t
    pd.testing.assert_frame_equal(before.loc[upto, RISK], after.loc[upto, RISK])


def test_tickers_never_borrow_each_others_prices() -> None:
    prices = panel()
    together = one(compute_volatility(prices), "BBB")[RISK].to_numpy()
    alone = compute_volatility(prices[prices["ticker"] == "BBB"])[RISK].to_numpy()
    np.testing.assert_array_equal(together, alone)


def test_one_missing_session_does_not_blank_a_year() -> None:
    prices = panel(tickers=("AAA",))
    holed = prices[prices["date"] != SESSIONS[300]]
    out = one(compute_volatility(holed, sessions={"AAA": SESSIONS[:400]}))
    p = one(prices)["adj_close_eur"]
    # The two returns touching the hole are gone; the other 250 still count.
    r = np.log(p).diff()
    r[SESSIONS[300]] = r[SESSIONS[301]] = np.nan
    t = SESSIONS[350]
    hand = r.loc[SESSIONS[350 - 251] : t].std(ddof=1) * ROOT252
    assert out.loc[t, "vol_252d"] == pytest.approx(hand, rel=1e-10)
    assert out["vol_21d"].loc[SESSIONS[301] : SESSIONS[321]].notna().all()
    assert SESSIONS[300] not in out.index


def test_too_many_missing_sessions_is_nan() -> None:
    prices = panel(tickers=("AAA",))
    holed = prices[~prices["date"].isin(SESSIONS[340:345])]  # 5 of 21: 6 returns lost
    out = one(compute_volatility(holed, sessions={"AAA": SESSIONS[:400]}))
    assert math.isnan(out.loc[SESSIONS[350], "vol_21d"])
    assert not math.isnan(out.loc[SESSIONS[350], "vol_252d"])


# ---------------------------------------------------------------------------
# Through the store
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dt.date]:
    """One XNYS stock, 400 sessions at 25% vol, 1% intraday range, 2-for-1 split at 330."""
    data = tmp_path / "data"
    days = SESSIONS[:400]
    closes = gbm(400, 0.25, seed=7)
    closes[330:] /= 2
    write_bars(
        pd.DataFrame(
            {
                "ticker": "AAA",
                "date": days,
                "open": closes,
                "high": closes * 1.005,
                "low": closes * 0.995,
                "close": closes,
                "volume": 1_000.0,
                "close_eur": closes * 0.9,
            }
        ),
        root=data / "bars" / "daily",
    )
    merge_actions(
        pd.DataFrame({"ticker": "AAA", "date": [days[330]], "action": "split", "value": [2.0]}),
        root=data / "actions",
    )
    company = {"exchange": "XNYS", "currency": "USD", "sector": "Industrials", "country": "X"}
    universe = write_csv(tmp_path / "universe.csv", [{"ticker": "AAA", "name": "A", **company}])
    benchmarks = write_csv(
        tmp_path / "benchmarks.csv",
        [{"ticker": "BENCH", "name": "B", **company, "sector": "Benchmark"}],
    )
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data),
                "universe_path": str(universe),
                "benchmarks_path": str(benchmarks),
                "benchmark": "BENCH",
                "log_dir": str(tmp_path / "logs"),
                "history_start": "2018-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return days


def test_a_split_is_neither_a_vol_spike_nor_a_true_range(store: list[dt.date]) -> None:
    out = read_volatility("AAA", as_of=store[-1]).set_index("date")
    after = out.loc[store[340]]
    assert 0.15 < after["vol_21d"] < 0.40  # an unadjusted -50% day would put it near 2.0
    assert after["mdd_252d"] > -0.45
    # A 1% intraday range plus a ~1.6% typical gap: ATR is a couple of percent, not 50%.
    assert out["atr_14_pct"].loc[store[330] :].max() < 0.04


def test_atr_pct_read_before_the_split_equals_atr_pct_read_after(store: list[dt.date]) -> None:
    t = store[300]
    then = read_volatility("AAA", as_of=t)
    later = read_volatility("AAA", as_of=store[-1])
    common = later[later["date"] <= t].reset_index(drop=True)
    columns = [*RISK, "atr_14_pct"]
    np.testing.assert_allclose(
        then[columns].to_numpy(), common[columns].to_numpy(), rtol=1e-10, equal_nan=True
    )
    # atr_14 itself is in adjusted price units: halved by the later split.
    np.testing.assert_allclose(common["atr_14"].dropna(), then["atr_14"].dropna() / 2, rtol=1e-10)


def test_read_volatility_trims_to_start_but_keeps_the_lookback(store: list[dt.date]) -> None:
    out = read_volatility("AAA", as_of=store[-1], start=store[380])
    assert out["date"].min() == store[380]
    assert out[list(VOLATILITY_COLUMNS)].notna().all().all()


def test_read_volatility_needs_as_of_and_refuses_the_future(store: list[dt.date]) -> None:
    with pytest.raises(TypeError):
        read_volatility("AAA")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_volatility("AAA", as_of=store[10], start=store[20])
