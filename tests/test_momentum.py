"""Day 24: 12-1 and 6-1 momentum, 5-day short-term reversal.

The acceptance criterion — 12-1 for a known trending stock matching a hand
calculation — lives in `notebooks/day24_momentum.ipynb`, on the real store.
Here the same hand calculation runs on synthetic prices, where the answer is
known in closed form, together with the properties the notebook cannot show:
the last month really is excluded, nothing after t is read, a missing session
does not stretch a window, and a later split does not change a value already
computed.
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
from finmgr.features.momentum import (
    MOMENTUM_COLUMNS,
    compute_momentum,
    read_momentum,
    skip_return,
)
from finmgr.features.returns import compute_returns, venue_sessions

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2022, 1, 1), D(2024, 6, 28))


def panel(n: int = 400, tickers: tuple[str, ...] = ("AAA", "BBB"), seed: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = SESSIONS[:n]
    frames = [
        pd.DataFrame(
            {
                "ticker": ticker,
                "date": days,
                "adj_close_eur": 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, len(days))),
            }
        )
        for ticker in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def one(frame: pd.DataFrame, ticker: str = "AAA") -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].set_index("date")


# ---------------------------------------------------------------------------
# Definitions, by hand
# ---------------------------------------------------------------------------


def test_12_1_is_the_price_a_month_ago_over_the_price_a_year_ago() -> None:
    prices = panel()
    p = one(prices)["adj_close_eur"].to_numpy()
    out = one(compute_momentum(prices))
    for i in (252, 300, 399):
        t = SESSIONS[i]
        assert out.loc[t, "mom_12_1"] == pytest.approx(p[i - 21] / p[i - 252] - 1, rel=1e-12)
        assert out.loc[t, "mom_6_1"] == pytest.approx(p[i - 21] / p[i - 126] - 1, rel=1e-12)
        assert out.loc[t, "rev_5d"] == pytest.approx(-(p[i] / p[i - 5] - 1), rel=1e-12)


def test_a_steady_trend_gives_the_closed_form_answer() -> None:
    # A stock growing exactly 0.1% a session: 12-1 spans 231 sessions of
    # growth, 6-1 spans 105, and a week's reversal is minus five sessions'.
    g = 0.001
    days = SESSIONS[:300]
    prices = pd.DataFrame(
        {"ticker": "TRD", "date": days, "adj_close_eur": 50 * (1 + g) ** np.arange(300)}
    )
    last = compute_momentum(prices).iloc[-1]
    assert last["mom_12_1"] == pytest.approx((1 + g) ** 231 - 1, rel=1e-10)
    assert last["mom_6_1"] == pytest.approx((1 + g) ** 105 - 1, rel=1e-10)
    assert last["rev_5d"] == pytest.approx(-((1 + g) ** 5 - 1), rel=1e-10)


def test_12_1_is_the_year_return_with_the_last_month_divided_out() -> None:
    prices = panel()
    mom = one(compute_momentum(prices))
    ret = one(compute_returns(prices, horizons=(5, 21, 126, 252)))
    np.testing.assert_allclose(
        mom["mom_12_1"], (1 + ret["ret_252d"]) / (1 + ret["ret_21d"]) - 1, rtol=1e-12
    )
    np.testing.assert_allclose(
        mom["mom_6_1"], (1 + ret["ret_126d"]) / (1 + ret["ret_21d"]) - 1, rtol=1e-12
    )
    np.testing.assert_allclose(mom["rev_5d"], -ret["ret_5d"], rtol=1e-12)


def test_the_last_month_does_not_move_12_1_or_6_1() -> None:
    prices = panel(tickers=("AAA",))
    t_index = 350
    before = one(compute_momentum(prices))
    # Crash the stock over the 20 sessions before t (t-20..t-1) and at t.
    crashed = prices.copy()
    window = crashed["date"].isin(SESSIONS[t_index - 20 : t_index + 1])
    crashed.loc[window, "adj_close_eur"] *= 0.3
    crashed.loc[crashed["date"] == SESSIONS[t_index], "adj_close_eur"] *= 0.5
    after = one(compute_momentum(crashed))
    t = SESSIONS[t_index]
    assert after.loc[t, "mom_12_1"] == before.loc[t, "mom_12_1"]
    assert after.loc[t, "mom_6_1"] == before.loc[t, "mom_6_1"]
    # ...while the 5-day reversal, which is about exactly that, sees it.
    assert after.loc[t, "rev_5d"] != before.loc[t, "rev_5d"]


def test_not_enough_history_is_nan_not_a_shorter_window() -> None:
    out = one(compute_momentum(panel()))
    assert out["mom_12_1"].iloc[:252].isna().all()
    assert out["mom_12_1"].iloc[252:].notna().all()
    assert out["mom_6_1"].iloc[:126].isna().all()
    assert out["mom_6_1"].iloc[126:].notna().all()
    assert out["rev_5d"].iloc[:5].isna().all()


@pytest.mark.parametrize(("lookback", "skip"), [(21, 21), (21, 30), (21, -1)])
def test_skip_must_sit_inside_the_window(lookback: int, skip: int) -> None:
    with pytest.raises(ValueError, match="skip"):
        skip_return(pd.Series([1.0, 2.0]), lookback, skip)


def test_a_duplicated_row_is_refused() -> None:
    prices = panel(n=10, tickers=("AAA",))
    with pytest.raises(ValueError, match="more than one row"):
        compute_momentum(pd.concat([prices, prices.iloc[[3]]]))


def test_empty_input_gives_the_full_schema() -> None:
    out = compute_momentum(pd.DataFrame({"ticker": [], "date": [], "adj_close_eur": []}))
    assert list(out.columns) == ["ticker", "date", "adj_close_eur", *MOMENTUM_COLUMNS]


# ---------------------------------------------------------------------------
# No look-ahead, no borrowed prices, no stretched windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [5, 126, 252, 399])
def test_momentum_at_t_uses_no_observation_later_than_t(cut: int) -> None:
    prices = panel()
    t = SESSIONS[cut]
    before = compute_momentum(prices)
    scrambled = prices.copy()
    later = scrambled["date"] > t
    rng = np.random.default_rng(0)
    scrambled.loc[later, "adj_close_eur"] *= rng.uniform(0.2, 5.0, later.sum())
    after = compute_momentum(scrambled)
    upto = before["date"] <= t
    pd.testing.assert_frame_equal(
        before.loc[upto, list(MOMENTUM_COLUMNS)], after.loc[upto, list(MOMENTUM_COLUMNS)]
    )


def test_tickers_never_borrow_each_others_prices() -> None:
    prices = panel()
    together = compute_momentum(prices)
    alone = compute_momentum(prices[prices["ticker"] == "BBB"])
    np.testing.assert_array_equal(
        one(together, "BBB")[list(MOMENTUM_COLUMNS)].to_numpy(),
        alone.set_index("date")[list(MOMENTUM_COLUMNS)].to_numpy(),
    )


def test_a_missing_session_does_not_stretch_the_window() -> None:
    prices = panel(tickers=("AAA",))
    p = one(prices)["adj_close_eur"]
    gone = SESSIONS[200]
    holed = prices[prices["date"] != gone]
    out = one(compute_momentum(holed, sessions={"AAA": SESSIONS[:400]}))
    # 12-1 at session 300 spans the hole: still exactly sessions 48 and 279.
    t = SESSIONS[300]
    assert out.loc[t, "mom_12_1"] == pytest.approx(p[SESSIONS[279]] / p[SESSIONS[48]] - 1)
    # With the hole as its t-21 end, 12-1 has no answer — it is not filled.
    assert math.isnan(out.loc[SESSIONS[221], "mom_12_1"])
    # Nor has 6-1 with it as the t-126 end, nor the reversal with it as t-5.
    assert math.isnan(out.loc[SESSIONS[326], "mom_6_1"])
    assert math.isnan(out.loc[SESSIONS[205], "rev_5d"])
    # The missing session does not come back as a row.
    assert gone not in out.index


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
    """One XNYS stock, 400 sessions, a 2-for-1 split at session 330."""
    data = tmp_path / "data"
    days = SESSIONS[:400]
    closes = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.001, 0.02, len(days)))
    closes[330:] /= 2
    closes = [float(c) for c in closes]
    write_bars(
        pd.DataFrame(
            {
                "ticker": "AAA",
                "date": days,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": 1_000.0,
                "close_eur": [c * 0.9 for c in closes],
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
                "history_start": "2022-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return days


def test_12_1_read_as_of_t_equals_12_1_read_after_a_split(store: list[dt.date]) -> None:
    t = store[300]
    then = read_momentum("AAA", as_of=t)
    later = read_momentum("AAA", as_of=store[-1])
    assert then["date"].max() == t
    common = later[later["date"] <= t].reset_index(drop=True)
    np.testing.assert_allclose(
        then[list(MOMENTUM_COLUMNS)].to_numpy(),
        common[list(MOMENTUM_COLUMNS)].to_numpy(),
        rtol=1e-12,
        equal_nan=True,
    )


def test_a_split_inside_the_window_is_not_a_crash(store: list[dt.date]) -> None:
    out = read_momentum("AAA", as_of=store[-1]).set_index("date")
    p = out["adj_close_eur"]
    t = store[380]  # 12-1 runs from session 128 to 359, across the split at 330
    assert out.loc[t, "mom_12_1"] == pytest.approx(p[store[359]] / p[store[128]] - 1)
    assert out.loc[t, "mom_12_1"] > -0.3  # a drift of +0.1% a session, not -50%


def test_read_momentum_trims_to_start_but_keeps_the_lookback(store: list[dt.date]) -> None:
    out = read_momentum("AAA", as_of=store[-1], start=store[380])
    assert out["date"].min() == store[380]
    assert out["mom_12_1"].notna().all()


def test_read_momentum_needs_as_of_and_refuses_the_future(store: list[dt.date]) -> None:
    with pytest.raises(TypeError):
        read_momentum("AAA")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_momentum("AAA", as_of=store[10], start=store[20])
