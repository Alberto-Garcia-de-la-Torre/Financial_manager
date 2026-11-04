"""Day 27: RSI(14), MACD(12,26,9), Stochastic(14,3) and Bollinger %B(20,2), by hand.

Two kinds of check. First, tiny fixtures worked out on paper (the arithmetic is
in the comments), so each definition is pinned down exactly. Second — the
acceptance criterion — the hand-written RSI against TA-Lib, an independent
reference implementation, to four decimal places (it actually agrees to
~1e-14). TA-Lib is not a dependency: its output on a 400-bar synthetic series
is frozen in `tests/fixtures/oscillators/talib_reference.csv` by
`tools/oscillator_reference.py`. The other three oscillators are checked against
the same file.
"""

from __future__ import annotations

import csv
import datetime as dt
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
from finmgr.features.oscillators import (
    MACD_PCT_COLUMNS,
    OSCILLATOR_COLUMNS,
    compute_oscillators,
    macd,
    percent_b,
    read_oscillators,
    rsi,
    stochastic,
)
from finmgr.features.returns import venue_sessions

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2018, 1, 1), D(2024, 6, 28))
REFERENCE = Path(__file__).parent / "fixtures" / "oscillators" / "talib_reference.csv"
SCALE_FREE = [c for c in OSCILLATOR_COLUMNS if c not in ("macd", "macd_signal", "macd_hist")]


@pytest.fixture(scope="module")
def ref() -> pd.DataFrame:
    return pd.read_csv(REFERENCE, comment="#")


def bars(n: int = 300, seed: int = 27) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    spread = np.abs(rng.normal(0, 0.01, n))
    return pd.DataFrame({"high": close * (1 + spread), "low": close * (1 - spread), "close": close})


def panel(n: int = 300, tickers: tuple[str, ...] = ("AAA",)) -> pd.DataFrame:
    frames = []
    for i, t in enumerate(tickers):
        b = bars(n, seed=i)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": SESSIONS[:n],
                    "high": b["high"],
                    "low": b["low"],
                    "close": b["close"],
                    "adj_close": b["close"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def one(frame: pd.DataFrame, ticker: str = "AAA") -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].set_index("date")


# ---------------------------------------------------------------------------
# The acceptance criterion: RSI against a reference implementation
# ---------------------------------------------------------------------------


def test_rsi_matches_talib_to_four_decimal_places(ref: pd.DataFrame) -> None:
    mine = rsi(ref["close"], 14)
    # Same warm-up: the first RSI is on the 15th price.
    np.testing.assert_array_equal(mine.isna().to_numpy(), ref["rsi_14"].isna().to_numpy())
    assert mine.first_valid_index() == 14
    defined = ref["rsi_14"].notna()
    assert defined.sum() == len(ref) - 14
    # Four decimal places, literally...
    np.testing.assert_array_equal(mine[defined].round(4), ref["rsi_14"][defined].round(4))
    # ...and in fact to floating-point noise.
    np.testing.assert_allclose(mine[defined], ref["rsi_14"][defined], rtol=0, atol=1e-9)
    # The fixture visits both extremes, so the check is not only of mid-range values.
    assert ref["rsi_14"].max() > 85 and ref["rsi_14"].min() < 15


# ---------------------------------------------------------------------------
# RSI by hand
# ---------------------------------------------------------------------------


def test_rsi_by_hand() -> None:
    # Changes: +1, -0.5, +1, -0.5, +1; period 3.
    # Seed (first 3 changes): avgU = (1 + 0 + 1) / 3 = 2/3, avgD = (0 + 0.5 + 0) / 3 = 1/6
    #   RSI[3] = 100 * (2/3) / (2/3 + 1/6) = 80
    # avgU = (2 * 2/3 + 0) / 3 = 4/9, avgD = (2 * 1/6 + 0.5) / 3 = 5/18
    #   RSI[4] = 100 * (8/18) / (13/18) = 800/13
    # avgU = (2 * 4/9 + 1) / 3 = 17/27, avgD = (2 * 5/18 + 0) / 3 = 5/27
    #   RSI[5] = 100 * 17 / 22
    p = pd.Series([10.0, 11.0, 10.5, 11.5, 11.0, 12.0])
    np.testing.assert_allclose(
        rsi(p, 3).to_numpy(), [np.nan, np.nan, np.nan, 80.0, 800 / 13, 1700 / 22]
    )


def test_rsi_equals_100_minus_100_over_one_plus_rs() -> None:
    p = pd.Series(bars()["close"])
    change = p.diff()
    up = change.clip(lower=0)
    down = (-change).clip(lower=0)
    # Wilder's smoothing is an EMA with alpha = 1/n once seeded with the mean.
    seed_u, seed_d = up.iloc[1:15].mean(), down.iloc[1:15].mean()
    u = pd.concat([pd.Series([seed_u]), up.iloc[15:]]).ewm(alpha=1 / 14, adjust=False).mean()
    d = pd.concat([pd.Series([seed_d]), down.iloc[15:]]).ewm(alpha=1 / 14, adjust=False).mean()
    expected = 100 - 100 / (1 + u.to_numpy() / d.to_numpy())
    np.testing.assert_allclose(rsi(p).iloc[14:].to_numpy(), expected, rtol=1e-12)


def test_rsi_extremes() -> None:
    up = pd.Series(np.arange(1.0, 31.0))
    assert (rsi(up).dropna() == 100).all()
    assert (rsi(up[::-1].reset_index(drop=True)).dropna() == 0).all()
    assert (rsi(pd.Series(np.full(30, 7.0))).dropna() == 50).all()
    out = rsi(pd.Series(bars()["close"])).dropna()
    assert ((out > 0) & (out < 100)).all()


def test_rsi_steps_over_a_missing_close() -> None:
    p = pd.Series(bars()["close"])
    holed = p.copy()
    holed.iloc[100] = np.nan
    out = rsi(holed)
    # The bar after the gap has no change of its own; the next bar does again.
    assert np.isnan(out.iloc[100]) and np.isnan(out.iloc[101])
    assert out.iloc[102:].notna().all()
    assert out.iloc[:100].equals(rsi(p).iloc[:100])


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def test_macd_by_hand() -> None:
    # fast 2 (alpha 2/3), slow 3 (alpha 1/2), signal 2.
    # EMA2: seed mean(1,2) = 1.5 at [1]; 2/3*3 + 1/3*1.5 = 2.5; 2/3*4 + 1/3*2.5 = 3.5;
    #       2/3*10 + 1/3*3.5 = 47/6; 2/3*6 + 1/3*47/6 = 119/18
    # EMA3: seed mean(1,2,3) = 2 at [2]; 3; 6.5; 6.25
    # MACD: [2] 0.5, [3] 0.5, [4] 47/6 - 6.5 = 4/3, [5] 119/18 - 6.25 = 13/36
    # Signal: seed mean(0.5, 0.5) = 0.5 at [3]; 2/3*4/3 + 1/3*0.5 = 19/18;
    #         2/3*13/36 + 1/3*19/18 = 32/54
    p = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 6.0])
    out = macd(p, 2, 3, 2)
    nan = np.nan
    np.testing.assert_allclose(out["macd"], [nan, nan, 0.5, 0.5, 4 / 3, 13 / 36])
    np.testing.assert_allclose(out["macd_signal"], [nan, nan, nan, 0.5, 19 / 18, 32 / 54])
    np.testing.assert_allclose(out["macd_hist"], out["macd"] - out["macd_signal"])


def test_macd_converges_to_talib(ref: pd.DataFrame) -> None:
    # TA-Lib seeds the fast EMA later than we do (so that both lines start on
    # bar 25); the difference decays geometrically and is gone by bar 120.
    out = macd(ref["close"])
    assert out["macd"].first_valid_index() == 25
    assert out["macd_signal"].first_valid_index() == 33 == ref["macd_signal"].first_valid_index()
    for col in ("macd", "macd_signal", "macd_hist"):
        np.testing.assert_allclose(out[col].iloc[120:], ref[col].iloc[120:], rtol=0, atol=1e-4)


def test_macd_is_positive_in_an_uptrend_and_negative_in_a_downtrend() -> None:
    up = pd.Series(100 * 1.01 ** np.arange(80))
    assert (macd(up)["macd"].dropna() > 0).all()
    assert (macd(up[::-1].reset_index(drop=True))["macd"].dropna() < 0).all()


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------


def test_stochastic_by_hand() -> None:
    # period 3, smooth 2.
    # [2]: HH(10,11,12) = 12, LL(8,9,10) = 8, C = 11 -> 100 * 3/4 = 75
    # [3]: HH(11,12,12) = 12, LL(9,10,9) = 9,  C = 9  -> 0
    # [4]: HH(12,12,13) = 13, LL(10,9,11) = 9, C = 13 -> 100
    # %D: [3] (75 + 0) / 2 = 37.5, [4] (0 + 100) / 2 = 50
    high = pd.Series([10.0, 11.0, 12.0, 12.0, 13.0])
    low = pd.Series([8.0, 9.0, 10.0, 9.0, 11.0])
    close = pd.Series([9.0, 10.0, 11.0, 9.0, 13.0])
    out = stochastic(high, low, close, 3, 2)
    nan = np.nan
    np.testing.assert_allclose(out["stoch_k"], [nan, nan, 75.0, 0.0, 100.0])
    np.testing.assert_allclose(out["stoch_d"], [nan, nan, nan, 37.5, 50.0])


def test_stochastic_matches_talib(ref: pd.DataFrame) -> None:
    out = stochastic(ref["high"], ref["low"], ref["close"])
    defined = ref["stoch_k"].notna()  # TA-Lib starts %K on the bar %D starts
    assert out["stoch_k"].first_valid_index() == 13
    np.testing.assert_allclose(out.loc[defined, "stoch_k"], ref.loc[defined, "stoch_k"], atol=1e-9)
    np.testing.assert_allclose(out.loc[defined, "stoch_d"], ref.loc[defined, "stoch_d"], atol=1e-9)


def test_stochastic_with_no_range_is_50() -> None:
    flat = pd.Series(np.full(20, 5.0))
    assert (stochastic(flat, flat, flat)["stoch_k"].dropna() == 50).all()


# ---------------------------------------------------------------------------
# Bollinger %B
# ---------------------------------------------------------------------------


def test_percent_b_by_hand() -> None:
    # period 3, width 2 on 1, 2, 3: mean 2, population sd sqrt(2/3).
    # %B = (3 - (2 - 2 sd)) / (4 sd) = 1 / (4 sd) + 1/2
    sd = np.sqrt(2 / 3)
    out = percent_b(pd.Series([1.0, 2.0, 3.0, 2.0]), 3, 2.0)
    # [3]: 2, 3, 2: mean 7/3, sd sqrt(2/9); %B = (2 - 7/3 + 2 sd') / (4 sd')
    sd3 = np.sqrt(2 / 9)
    expected = [np.nan, np.nan, 0.5 + 1 / (4 * sd), (2 - 7 / 3 + 2 * sd3) / (4 * sd3)]
    np.testing.assert_allclose(out, expected)


def test_percent_b_is_half_plus_a_quarter_of_the_z_score() -> None:
    p = pd.Series(bars()["close"])
    z = (p - p.rolling(20).mean()) / p.rolling(20).std(ddof=0)
    np.testing.assert_allclose(percent_b(p), 0.5 + z / 4, rtol=1e-10)
    assert (percent_b(pd.Series(np.full(30, 3.0))).dropna() == 0.5).all()


def test_percent_b_matches_talib(ref: pd.DataFrame) -> None:
    out = percent_b(ref["close"])
    np.testing.assert_array_equal(out.isna(), ref["bb_pctb"].isna())
    np.testing.assert_allclose(out.dropna(), ref["bb_pctb"].dropna(), atol=1e-9)


# ---------------------------------------------------------------------------
# The panel: history, look-ahead, gaps, scale, schema
# ---------------------------------------------------------------------------


def test_panel_columns_are_the_functions_on_the_adjusted_bar() -> None:
    prices = panel()
    out = one(compute_oscillators(prices))
    p = one(prices)
    np.testing.assert_allclose(out["rsi_14"], rsi(p["adj_close"]).to_numpy(), equal_nan=True)
    np.testing.assert_allclose(
        out["macd_hist_pct"],
        (macd(p["adj_close"])["macd_hist"] / p["adj_close"]).to_numpy(),
        equal_nan=True,
    )
    stoch = stochastic(p["high"], p["low"], p["adj_close"])
    np.testing.assert_allclose(out["stoch_d"], stoch["stoch_d"].to_numpy(), equal_nan=True)
    # Everything is defined once MACD's signal is.
    assert out[list(OSCILLATOR_COLUMNS)].iloc[33:].notna().all().all()


@pytest.mark.parametrize("cut", [20, 40, 150, 299])
def test_oscillators_at_t_use_no_bar_later_than_t(cut: int) -> None:
    prices = panel(tickers=("AAA", "BBB"))
    t = SESSIONS[cut]
    before = compute_oscillators(prices)
    scrambled = prices.copy()
    later = scrambled["date"] > t
    noise = np.random.default_rng(0).uniform(0.2, 5, later.sum())
    for col in ("high", "low", "close", "adj_close"):
        scrambled.loc[later, col] *= noise
    after = compute_oscillators(scrambled)
    upto = before["date"] <= t
    pd.testing.assert_frame_equal(before.loc[upto], after.loc[upto])


def test_tickers_never_borrow_each_others_bars() -> None:
    prices = panel(tickers=("AAA", "BBB"))
    together = one(compute_oscillators(prices), "BBB")[list(OSCILLATOR_COLUMNS)]
    alone = one(compute_oscillators(prices[prices["ticker"] == "BBB"]), "BBB")
    np.testing.assert_array_equal(together.to_numpy(), alone[list(OSCILLATOR_COLUMNS)].to_numpy())


def test_scale_free_columns_do_not_move_when_every_price_is_rescaled() -> None:
    prices = panel()
    scaled = prices.assign(**{c: prices[c] / 3 for c in ("high", "low", "close", "adj_close")})
    a, b = compute_oscillators(prices), compute_oscillators(scaled)
    pd.testing.assert_frame_equal(a[SCALE_FREE], b[SCALE_FREE], rtol=1e-9)
    np.testing.assert_allclose(a["macd"] / 3, b["macd"], rtol=1e-9)


def test_a_missing_session_is_skipped_not_fatal() -> None:
    prices = panel()
    holed = prices[prices["date"] != SESSIONS[200]]
    out = one(compute_oscillators(holed, sessions={"AAA": SESSIONS[:300]}))
    assert SESSIONS[200] not in out.index
    # RSI's change into 201 spans the gap and is skipped; %D needs three %Ks.
    assert np.isnan(out.loc[SESSIONS[201], "rsi_14"])
    assert out[list(OSCILLATOR_COLUMNS)].loc[SESSIONS[203] :].notna().all().all()


def test_without_high_and_low_the_stochastic_is_nan_and_the_rest_computes() -> None:
    prices = panel()[["ticker", "date", "adj_close"]]
    out = compute_oscillators(prices)
    assert out[["stoch_k", "stoch_d"]].isna().all().all()
    assert out[["rsi_14", "bb_pctb", *MACD_PCT_COLUMNS]].iloc[40:].notna().all().all()


def test_a_duplicated_row_is_refused() -> None:
    prices = panel()
    with pytest.raises(ValueError):
        compute_oscillators(pd.concat([prices, prices.iloc[:1]]))


def test_empty_input_gives_the_full_schema() -> None:
    out = compute_oscillators(panel().iloc[:0])
    assert list(out.columns) == ["ticker", "date", "adj_close", *OSCILLATOR_COLUMNS]


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
    """One XNYS stock, 300 sessions, 2-for-1 split at session 250."""
    data = tmp_path / "data"
    days = SESSIONS[:300]
    b = bars(300, seed=3)
    for col in ("high", "low", "close"):
        b.loc[250:, col] /= 2
    write_bars(
        pd.DataFrame(
            {
                "ticker": "AAA",
                "date": days,
                "open": b["close"],
                "high": b["high"],
                "low": b["low"],
                "close": b["close"],
                "volume": 1_000.0,
                "close_eur": b["close"] * 0.9,
            }
        ),
        root=data / "bars" / "daily",
    )
    merge_actions(
        pd.DataFrame({"ticker": "AAA", "date": [days[250]], "action": "split", "value": [2.0]}),
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


def test_a_split_is_not_an_oversold_signal(store: list[dt.date]) -> None:
    out = read_oscillators("AAA", as_of=store[-1]).set_index("date")
    raw = bars(300, seed=3)
    clean = compute_oscillators(
        pd.DataFrame(
            {"ticker": "AAA", "date": store, **{c: raw[c] for c in ("high", "low", "close")}}
        ).assign(adj_close=raw["close"])
    ).set_index("date")
    # Back-adjusted, the split vanishes: the oscillators equal the never-split series.
    np.testing.assert_allclose(
        out[SCALE_FREE].to_numpy(), clean[SCALE_FREE].to_numpy(), rtol=1e-9, equal_nan=True
    )


def test_read_oscillators_does_not_depend_on_start(store: list[dt.date]) -> None:
    full = read_oscillators("AAA", as_of=store[-1])
    cut = read_oscillators("AAA", as_of=store[-1], start=store[200])
    assert cut["date"].min() == store[200]
    pd.testing.assert_frame_equal(cut, full[full["date"] >= store[200]].reset_index(drop=True))


def test_read_oscillators_needs_as_of_and_refuses_the_future(store: list[dt.date]) -> None:
    with pytest.raises(TypeError):
        read_oscillators("AAA")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_oscillators("AAA", as_of=store[10], start=store[20])
