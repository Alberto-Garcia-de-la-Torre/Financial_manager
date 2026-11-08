"""Day 28: euro volume, average daily value, volume z-score, OBV and the tradability flag.

The acceptance criterion — the tradability filter excludes the thinnest names in
the universe — is checked on the real store in `notebooks/day28_liquidity.ipynb`.
Here it runs on a synthetic universe whose liquidity is known by construction,
next to the definitions by hand and the properties the notebook cannot show:
nothing after t is read, a split does not move euro volume, unknown liquidity
is not tradable.
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
from finmgr.data.pit import LookAheadError
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS
from finmgr.features.liquidity import (
    LIQUIDITY_COLUMNS,
    MIN_ADV_EUR,
    average_daily_value,
    compute_liquidity,
    obv_flow,
    on_balance_volume,
    read_liquidity,
    thinnest,
    tradable_flag,
    volume_zscore,
)
from finmgr.features.returns import venue_sessions

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2018, 1, 1), D(2024, 6, 28))
VALUES = ["eur_volume", "adv_eur_21", "volume_z_63", "obv", "obv_flow_21"]


def walk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 50 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))


def ticker_frame(ticker: str, n: int, daily_eur: float, seed: int) -> pd.DataFrame:
    """`n` sessions of a stock whose traded value averages `daily_eur`, noise +-50%."""
    rng = np.random.default_rng(seed + 1000)
    close = walk(n, seed)
    traded = daily_eur * rng.uniform(0.5, 1.5, n)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": SESSIONS[:n],
            "adj_close": close,
            "close_eur": close * 0.9,
            "volume": traded / (close * 0.9),
        }
    )


#: A synthetic 10-name universe, liquidity from EUR 0.5M to EUR 2B a day.
UNIVERSE = {
    "DUST": 0.5e6,
    "THIN": 2e6,
    "EDGE": 4e6,
    "OKAY": 8e6,
    "MID1": 30e6,
    "MID2": 80e6,
    "BIG1": 300e6,
    "BIG2": 600e6,
    "MEGA": 1.2e9,
    "GIGA": 2e9,
}


def universe(n: int = 200) -> pd.DataFrame:
    frames = [ticker_frame(t, n, v, seed=i) for i, (t, v) in enumerate(UNIVERSE.items())]
    return pd.concat(frames, ignore_index=True)


def one(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].set_index("date")


# ---------------------------------------------------------------------------
# The acceptance criterion's shape: the filter drops the thinnest names
# ---------------------------------------------------------------------------


def test_the_filter_excludes_exactly_the_names_below_five_million() -> None:
    out = compute_liquidity(universe())
    last = out[out["date"] == out["date"].max()].set_index("ticker")
    excluded = set(last.index[~last["tradable"]])
    assert excluded == {"DUST", "THIN", "EDGE"}
    assert (last.loc[sorted(excluded), "adv_eur_21"] < MIN_ADV_EUR).all()
    assert (last.loc[~last["tradable"].to_numpy(), "adv_eur_21"] < MIN_ADV_EUR).all()


def test_on_every_date_every_excluded_name_is_thinner_than_every_included_one() -> None:
    out = compute_liquidity(universe()).dropna(subset=["adv_eur_21"])
    for _, day in out.groupby("date"):
        out_names = day.loc[~day["tradable"], "adv_eur_21"]
        in_names = day.loc[day["tradable"], "adv_eur_21"]
        assert not in_names.empty
        if not out_names.empty:
            assert out_names.max() < in_names.min()
    # And the excluded ones are always the thinnest of the universe.
    always_in = {"MID1", "MID2", "BIG1", "BIG2", "MEGA", "GIGA"}
    assert out.loc[out["ticker"].isin(always_in), "tradable"].all()
    assert not out.loc[out["ticker"].isin({"DUST", "THIN"}), "tradable"].any()


def test_thinnest_lists_the_bottom_of_the_universe_first() -> None:
    out = compute_liquidity(universe())
    bottom = thinnest(out, out["date"].max(), n=4)
    assert list(bottom["ticker"]) == ["DUST", "THIN", "EDGE", "OKAY"]
    assert list(bottom["tradable"]) == [False, False, False, True]


def test_the_threshold_is_a_parameter() -> None:
    out = compute_liquidity(universe(), min_adv_eur=100e6)
    last = out[out["date"] == out["date"].max()]
    assert set(last.loc[~last["tradable"], "ticker"]) == {
        "DUST",
        "THIN",
        "EDGE",
        "OKAY",
        "MID1",
        "MID2",
    }
    with pytest.raises(ValueError):
        compute_liquidity(universe(), min_adv_eur=0)


# ---------------------------------------------------------------------------
# Definitions by hand
# ---------------------------------------------------------------------------


def test_eur_volume_is_shares_times_the_euro_close() -> None:
    frame = ticker_frame("AAA", 30, 10e6, seed=1)
    out = one(compute_liquidity(frame), "AAA")
    expected = frame["volume"].to_numpy() * frame["close_eur"].to_numpy()
    np.testing.assert_allclose(out["eur_volume"].to_numpy(), expected, rtol=1e-12)


def test_adv_is_the_mean_of_the_last_21_and_starts_on_the_21st_session() -> None:
    ev = pd.Series(np.arange(1.0, 41.0))
    adv = average_daily_value(ev)
    assert adv.iloc[:20].isna().all()
    assert adv.iloc[20] == pytest.approx(np.mean(np.arange(1.0, 22.0)))
    assert adv.iloc[39] == pytest.approx(np.mean(np.arange(20.0, 41.0)))


def test_volume_z_is_measured_against_the_63_sessions_before_today() -> None:
    rng = np.random.default_rng(3)
    ev = pd.Series(np.exp(rng.normal(16, 0.4, 100)))
    z = volume_zscore(ev)
    assert z.iloc[:63].isna().all()
    t = 80
    base = np.log(ev.iloc[t - 63 : t])
    expected = (np.log(ev.iloc[t]) - base.mean()) / base.std(ddof=1)
    assert z.iloc[t] == pytest.approx(expected, rel=1e-12)
    # A day at e^3 times the usual volume is a ~3/0.4 sigma surprise.
    spiked = ev.copy()
    spiked.iloc[90] = np.exp(19)
    assert volume_zscore(spiked).iloc[90] > 5


def test_a_zero_volume_session_has_no_z_score() -> None:
    ev = pd.Series(np.full(80, 1e6))
    ev.iloc[np.arange(80) % 2 == 0] = 2e6
    ev.iloc[70] = 0.0
    z = volume_zscore(ev)
    assert np.isnan(z.iloc[70])
    assert np.isfinite(z.iloc[71])


def test_obv_by_hand() -> None:
    prices = pd.Series([10.0, 11.0, 10.5, 10.5, 12.0, 11.0])
    ev = pd.Series([100.0, 200.0, 50.0, 70.0, 30.0, 10.0])
    # first day: no direction; then up +200, down -50, flat 0, up +30, down -10.
    expected = [0.0, 200.0, 150.0, 150.0, 180.0, 170.0]
    np.testing.assert_allclose(on_balance_volume(prices, ev).to_numpy(), expected)


def test_obv_flow_is_plus_one_all_up_and_minus_one_all_down() -> None:
    ev = pd.Series(np.full(40, 1e6))
    up = pd.Series(np.arange(1.0, 41.0))
    assert obv_flow(up, ev).iloc[21:].eq(1.0).all()
    assert obv_flow(up[::-1].reset_index(drop=True), ev).iloc[21:].eq(-1.0).all()
    assert obv_flow(up, ev).iloc[:21].isna().all()


def test_obv_flow_stays_in_minus_one_one() -> None:
    flow = compute_liquidity(universe())["obv_flow_21"].dropna()
    assert flow.between(-1, 1).all()
    assert flow.abs().max() < 1  # random walks: neither all up nor all down


def test_unknown_liquidity_is_not_tradable() -> None:
    flag = tradable_flag(pd.Series([np.nan, 4.9e6, 5e6, 6e6]))
    assert flag.dtype == bool
    assert list(flag) == [False, False, True, True]
    # A brand-new listing, however liquid, is not tradable before its 21st session.
    out = one(compute_liquidity(ticker_frame("NEW", 30, 1e9, seed=2)), "NEW")
    assert not out["tradable"].iloc[:20].any()
    assert out["tradable"].iloc[20:].all()


def test_missing_inputs_and_duplicates_are_refused() -> None:
    frame = ticker_frame("AAA", 30, 10e6, seed=1)
    with pytest.raises(KeyError, match="volume"):
        compute_liquidity(frame.drop(columns=["volume"]))
    with pytest.raises(ValueError):
        compute_liquidity(pd.concat([frame, frame.iloc[[3]]]))


def test_empty_input_gives_the_full_schema() -> None:
    out = compute_liquidity(ticker_frame("AAA", 30, 1e6, seed=1).iloc[:0])
    assert list(out.columns) == ["ticker", "date", "adj_close", *LIQUIDITY_COLUMNS]


# ---------------------------------------------------------------------------
# Alignment and robustness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [30, 70, 150])
def test_liquidity_at_t_uses_no_observation_later_than_t(cut: int) -> None:
    full = compute_liquidity(universe())
    data = universe()
    early = compute_liquidity(data[data["date"] <= SESSIONS[cut]])
    common = full[full["date"] <= SESSIONS[cut]].reset_index(drop=True)
    np.testing.assert_allclose(
        early[VALUES].to_numpy(), common[VALUES].to_numpy(), rtol=1e-12, equal_nan=True
    )
    assert (early["tradable"] == common["tradable"]).all()


def test_a_split_does_not_move_euro_volume() -> None:
    frame = ticker_frame("AAA", 120, 10e6, seed=4)
    split = frame.copy()
    # A 4-for-1 split at 60, unadjusted: a quarter of the price, four times the shares.
    split.loc[60:, "close_eur"] /= 4
    split.loc[60:, "volume"] *= 4
    a = one(compute_liquidity(frame), "AAA")
    b = one(compute_liquidity(split), "AAA")
    np.testing.assert_allclose(a[VALUES].to_numpy(), b[VALUES].to_numpy(), rtol=1e-9)


def test_one_missing_session_does_not_blank_the_average() -> None:
    frame = ticker_frame("AAA", 80, 10e6, seed=5).drop(index=50)
    grid = {"AAA": SESSIONS[:80]}
    out = one(compute_liquidity(frame, sessions=grid), "AAA")
    assert SESSIONS[50] not in out.index
    assert out.loc[SESSIONS[51] : SESSIONS[70], "adv_eur_21"].notna().all()
    # Its neighbours have no direction across the gap: obv carries, flow still computes.
    assert out.loc[SESSIONS[51], "obv"] == out.loc[SESSIONS[49], "obv"]
    assert out.loc[SESSIONS[51] : SESSIONS[79], "obv_flow_21"].notna().all()


def test_tickers_never_borrow_each_others_volume() -> None:
    both = compute_liquidity(universe())
    alone = compute_liquidity(ticker_frame("MEGA", 200, UNIVERSE["MEGA"], seed=8))
    np.testing.assert_allclose(
        one(both, "MEGA")[VALUES].to_numpy(), one(alone, "MEGA")[VALUES].to_numpy(), rtol=1e-12
    )


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
    """Two XNYS stocks, 120 sessions: LIQ trades ~EUR 50M a day, ILLQ ~EUR 1M."""
    data = tmp_path / "data"
    days = SESSIONS[:120]
    frames = []
    for ticker, value, seed in (("LIQ", 50e6, 1), ("ILLQ", 1e6, 2)):
        close = walk(120, seed)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": days,
                    "open": close,
                    "high": close * 1.005,
                    "low": close * 0.995,
                    "close": close,
                    "volume": value / (close * 0.9),
                    "close_eur": close * 0.9,
                }
            )
        )
    write_bars(pd.concat(frames, ignore_index=True), root=data / "bars" / "daily")
    company = {"exchange": "XNYS", "currency": "USD", "sector": "Industrials", "country": "X"}
    universe = write_csv(
        tmp_path / "universe.csv",
        [{"ticker": "LIQ", "name": "L", **company}, {"ticker": "ILLQ", "name": "I", **company}],
    )
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


def test_read_liquidity_flags_the_illiquid_name(store: list[dt.date]) -> None:
    out = read_liquidity(as_of=store[-1], start=store[30])
    assert out["date"].min() == store[30]
    by = out.groupby("ticker")
    assert by["tradable"].all()["LIQ"]
    assert not by["tradable"].any()["ILLQ"]
    assert by["adv_eur_21"].mean()["LIQ"] == pytest.approx(50e6, rel=1e-9)
    assert by["adv_eur_21"].mean()["ILLQ"] == pytest.approx(1e6, rel=1e-9)


def test_read_liquidity_obv_does_not_depend_on_start(store: list[dt.date]) -> None:
    full = read_liquidity("LIQ", as_of=store[-1])
    cut = read_liquidity("LIQ", as_of=store[-1], start=store[60])
    tail = full[full["date"] >= store[60]].reset_index(drop=True)
    np.testing.assert_allclose(cut["obv"].to_numpy(), tail["obv"].to_numpy())


def test_read_liquidity_needs_as_of_and_refuses_the_future(store: list[dt.date]) -> None:
    with pytest.raises(TypeError):
        read_liquidity("LIQ")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_liquidity("LIQ", as_of=store[10], start=store[20])
