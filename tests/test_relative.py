"""Day 30: rolling beta, residual momentum and sector-demeaned features.

The acceptance criterion — sector-demeaned momentum no longer simply ranks
whichever sector had a good year — is
`test_sector_demeaned_momentum_no_longer_ranks_the_good_sector`: five sectors
with very different years, where raw 12-1 puts one sector at the top and the
demeaned version spreads the top across sectors, leaves sector explaining none
of the variance, and keeps each sector's own order intact.
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
from finmgr.data.pit import LookAheadError
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS
from finmgr.features.crosssection import FEATURE_COLUMNS, cross_rank
from finmgr.features.momentum import compute_momentum
from finmgr.features.relative import (
    BETA_COLUMN,
    IDIO_VOL_COLUMN,
    RELATIVE_COLUMNS,
    RESID_MOM_COLUMN,
    SECTOR_COLUMNS,
    bench_on_grid,
    compute_relatives,
    read_relatives,
    rolling_beta,
    sector_column,
    sector_demean,
    sector_mean_rank,
    sector_share,
)
from finmgr.features.returns import venue_sessions

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2022, 1, 1), D(2024, 6, 28))
N = 400


def market(n: int = N, seed: int = 30) -> pd.Series:
    """Benchmark daily log returns."""
    return pd.Series(np.random.default_rng(seed).normal(0.0003, 0.01, n))


def to_prices(log_returns: pd.Series | np.ndarray, start: float = 100.0) -> np.ndarray:
    steps = np.asarray(log_returns, dtype=float).copy()
    steps[0] = 0.0
    return start * np.exp(np.cumsum(steps))


def bench_frame(r_m: pd.Series, days: list[dt.date] | None = None) -> pd.DataFrame:
    days = days if days is not None else SESSIONS[: len(r_m)]
    return pd.DataFrame({"ticker": "BENCH", "date": days, "adj_close_eur": to_prices(r_m)})


def stock_frame(ticker: str, r_i: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": ticker, "date": SESSIONS[: len(r_i)], "adj_close_eur": to_prices(r_i)}
    )


def one(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].set_index("date")


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


def test_beta_matches_a_least_squares_fit_over_the_window() -> None:
    r_m = market()
    rng = np.random.default_rng(1)
    r_i = 1.3 * r_m + rng.normal(0, 0.015, N)
    fit = rolling_beta(r_i, r_m)
    for t in (251, 320, 399):
        x, y = r_m[t - 251 : t + 1], r_i[t - 251 : t + 1]
        slope, intercept = np.polyfit(x, y, 1)
        assert fit.loc[t, "beta"] == pytest.approx(slope, rel=1e-9)
        resid = y - (slope * x + intercept)
        assert fit.loc[t, "resid_var"] == pytest.approx(np.mean(resid**2), rel=1e-7)


def test_a_stock_that_is_the_market_levered_has_that_beta_and_no_residual() -> None:
    r_m = market()
    out = one(compute_relatives(stock_frame("LEV", 2.0 * r_m), bench_frame(r_m)), "LEV")
    beta = out[BETA_COLUMN].dropna()
    np.testing.assert_allclose(beta, 2.0, rtol=1e-8)
    np.testing.assert_allclose(out[IDIO_VOL_COLUMN].dropna(), 0.0, atol=1e-6)


def test_beta_needs_a_full_window() -> None:
    r_m = market()
    out = one(compute_relatives(stock_frame("AAA", r_m), bench_frame(r_m)), "AAA")
    # Row 251 is the first with 252 rows behind it (251 return pairs).
    assert out[BETA_COLUMN].iloc[:251].isna().all()
    assert out[BETA_COLUMN].iloc[251:].notna().all()


def test_too_few_pairs_in_the_window_is_nan() -> None:
    r_m = market()
    r_i = r_m.copy()
    r_i[100:200] = np.nan  # 100 missing returns: 152 pairs in the window at row 251
    fit = rolling_beta(r_i, r_m)
    assert math.isnan(fit.loc[251, "beta"])
    assert fit.loc[251, "pairs"] == 152
    assert fit.loc[399, "beta"] == pytest.approx(1.0)


@pytest.mark.parametrize(("min_pairs", "window"), [(1, 252), (300, 252)])
def test_min_pairs_must_fit_the_window(min_pairs: int, window: int) -> None:
    with pytest.raises(ValueError, match="min_pairs"):
        rolling_beta(market(), market(), window=window, min_pairs=min_pairs)


# ---------------------------------------------------------------------------
# Residual momentum
# ---------------------------------------------------------------------------


def test_residual_momentum_is_the_drift_the_market_does_not_explain() -> None:
    # Log return = 1.5 * market + g every session: beta is exactly 1.5, and the
    # 12-1 window holds 231 sessions of g.
    r_m, g = market(), 0.0007
    r_i = 1.5 * r_m + g
    out = one(compute_relatives(stock_frame("DRF", r_i), bench_frame(r_m)), "DRF")
    np.testing.assert_allclose(out[BETA_COLUMN].dropna(), 1.5, rtol=1e-8)
    np.testing.assert_allclose(out[RESID_MOM_COLUMN].dropna(), 231 * g, rtol=1e-6)


def test_a_high_beta_name_in_a_rising_market_has_no_residual_momentum() -> None:
    r_m = market() + 0.002  # a strong year for the market
    prices = pd.concat([stock_frame("HI", 2.0 * r_m), stock_frame("LO", 0.5 * r_m)])
    rel = compute_relatives(prices, bench_frame(r_m))
    mom = compute_momentum(prices)
    t = SESSIONS[N - 1]
    # Raw 12-1 says the high-beta name won; market-neutral, neither did.
    assert one(mom, "HI").loc[t, "mom_12_1"] > one(mom, "LO").loc[t, "mom_12_1"] + 0.5
    assert one(rel, "HI").loc[t, RESID_MOM_COLUMN] == pytest.approx(0.0, abs=1e-9)
    assert one(rel, "LO").loc[t, RESID_MOM_COLUMN] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("cut", [251, 300, 380])
def test_relatives_at_t_use_no_observation_later_than_t(cut: int) -> None:
    r_m = market()
    r_i = 0.8 * r_m + np.random.default_rng(3).normal(0, 0.01, N)
    stock, bench = stock_frame("AAA", r_i), bench_frame(r_m)
    before = compute_relatives(stock, bench)
    t = SESSIONS[cut]
    rng = np.random.default_rng(0)
    for frame in (stock, bench):
        later = frame["date"] > t
        frame.loc[later, "adj_close_eur"] *= rng.uniform(0.2, 5.0, later.sum())
    after = compute_relatives(stock, bench)
    upto = before["date"] <= t
    pd.testing.assert_frame_equal(
        before.loc[upto, list(RELATIVE_COLUMNS)], after.loc[upto, list(RELATIVE_COLUMNS)]
    )


def test_the_benchmark_is_carried_over_its_own_holidays_but_not_further() -> None:
    bench = pd.Series([1.0, 2.0, 3.0], index=[D(2024, 1, 1), D(2024, 1, 2), D(2024, 1, 10)])
    grid = pd.Index([D(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10)])
    out = bench_on_grid(bench, grid, limit=2)
    assert list(out.iloc[:3]) == [2.0, 2.0, 2.0]
    assert out.iloc[3:6].isna().all()
    assert out.iloc[6] == 3.0


def test_a_benchmark_with_two_tickers_is_refused() -> None:
    r_m = market(n=10)
    two = pd.concat([bench_frame(r_m), bench_frame(r_m).assign(ticker="OTHER")])
    with pytest.raises(ValueError, match="more than one"):
        compute_relatives(stock_frame("AAA", r_m), two)


def test_empty_input_gives_the_full_schema() -> None:
    empty = pd.DataFrame({"ticker": [], "date": [], "adj_close_eur": []})
    out = compute_relatives(empty, bench_frame(market(n=5)))
    assert list(out.columns) == ["ticker", "date", "adj_close_eur", *RELATIVE_COLUMNS]


# ---------------------------------------------------------------------------
# Sector demeaning
# ---------------------------------------------------------------------------


def small_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [D(2024, 1, 2)] * 7,
            "ticker": list("ABCDEFG"),
            "sector": ["X", "X", "X", "Y", "Y", "Y", "Z"],
            "mom_12_1": [0.1, 0.2, 0.6, -0.3, -0.1, np.nan, 0.5],
        }
    )


def test_sector_demean_subtracts_the_sector_mean_of_the_day() -> None:
    out = sector_demean(small_panel(), ["mom_12_1"], min_names=2)
    col = out[sector_column("mom_12_1")]
    np.testing.assert_allclose(col.iloc[:3], [-0.2, -0.1, 0.3])
    np.testing.assert_allclose(col.iloc[3:5], [-0.1, 0.1])
    # No value stays NaN; a sector of one has nothing to be relative to.
    assert col.iloc[5:].isna().all()


def test_a_sector_below_min_names_is_nan() -> None:
    out = sector_demean(small_panel(), ["mom_12_1"])  # default 3: Y has only 2 valued
    col = out[sector_column("mom_12_1")]
    assert col.iloc[:3].notna().all()
    assert col.iloc[3:].isna().all()


def test_sectors_can_come_from_a_mapping() -> None:
    panel = small_panel()
    mapping = dict(zip(panel["ticker"], panel["sector"], strict=True))
    by_map = sector_demean(panel.drop(columns="sector"), ["mom_12_1"], sectors=mapping)
    by_col = sector_demean(panel, ["mom_12_1"])
    pd.testing.assert_series_equal(by_map["mom_12_1_sec"], by_col["mom_12_1_sec"])


def test_dates_are_demeaned_separately() -> None:
    a = small_panel()
    b = a.assign(date=D(2024, 1, 3), mom_12_1=a["mom_12_1"] + 10.0)
    out = sector_demean(pd.concat([a, b], ignore_index=True), ["mom_12_1"], min_names=2)
    np.testing.assert_allclose(out["mom_12_1_sec"].iloc[:7], out["mom_12_1_sec"].iloc[7:])


def test_sector_demean_needs_a_sector() -> None:
    with pytest.raises(KeyError, match="sector"):
        sector_demean(small_panel().drop(columns="sector"), ["mom_12_1"])


def sector_universe(seed: int = 30) -> tuple[pd.DataFrame, dict[str, str]]:
    """Five sectors of eight names, 400 sessions; sector years differ far more than names do."""
    rng = np.random.default_rng(seed)
    drifts = {"Energy": 0.003, "Tech": 0.0015, "Banks": 0.0, "Health": -0.001, "Retail": -0.002}
    frames, sectors = [], {}
    for sector, drift in drifts.items():
        common = rng.normal(drift, 0.008, N)
        for k in range(8):
            ticker = f"{sector[:3].upper()}{k}"
            sectors[ticker] = sector
            frames.append(stock_frame(ticker, common + rng.normal(0, 0.006, N)))
    return pd.concat(frames, ignore_index=True), sectors


def test_sector_demeaned_momentum_no_longer_ranks_the_good_sector() -> None:
    prices, sectors = sector_universe()
    mom = compute_momentum(prices)
    mom["sector"] = mom["ticker"].map(sectors)
    panel = sector_demean(mom.dropna(subset=["mom_12_1"]), ["mom_12_1"])
    last = panel[panel["date"] == panel["date"].max()].copy()
    last["raw_rank"] = cross_rank(last["mom_12_1"], last["date"])
    last["sec_rank"] = cross_rank(last["mom_12_1_sec"], last["date"])

    # Raw: the top eight are the energy sector, and sector is most of the story.
    assert set(last.nlargest(8, "raw_rank")["sector"]) == {"Energy"}
    raw_share = sector_share(panel, "mom_12_1")
    assert raw_share.min() > 0.8

    # Demeaned: sector explains nothing on any date, every sector's mean rank
    # is near the middle, and the top eight come from across the universe.
    np.testing.assert_allclose(sector_share(panel, "mom_12_1_sec"), 0.0, atol=1e-12)
    means = sector_mean_rank(panel, "mom_12_1_sec")
    assert (means - 0.5).abs().to_numpy().max() < 0.12
    assert last.nlargest(8, "sec_rank")["sector"].nunique() >= 4

    # ...without scrambling who leads within a sector.
    for _, group in last.groupby("sector"):
        assert list(group.sort_values("mom_12_1")["ticker"]) == list(
            group.sort_values("mom_12_1_sec")["ticker"]
        )


def test_sector_share_is_one_when_the_value_is_the_sector() -> None:
    panel = small_panel().assign(mom_12_1=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    assert sector_share(panel, "mom_12_1").iloc[0] == pytest.approx(1.0)


def test_relatives_are_in_the_cross_section_defaults() -> None:
    for column in (*RELATIVE_COLUMNS, *SECTOR_COLUMNS):
        assert column in FEATURE_COLUMNS
    assert "mom_12_1_sec" in SECTOR_COLUMNS


# ---------------------------------------------------------------------------
# Through the store
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def bars(ticker: str, closes: np.ndarray) -> pd.DataFrame:
    values = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": SESSIONS[: len(values)],
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": 1_000.0,
            "close_eur": values,
        }
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """AAA at 1.5x the benchmark plus drift, on XNYS, with BENCH in its own directory."""
    data = tmp_path / "data"
    r_m = market()
    write_bars(bars("AAA", to_prices(1.5 * r_m + 0.0007)), root=data / "bars" / "daily")
    write_bars(bars("BENCH", to_prices(r_m)), root=data / "benchmarks" / "bars" / "daily")
    company = {"exchange": "XNYS", "currency": "EUR", "country": "X"}
    universe = write_csv(
        tmp_path / "universe.csv", [{"ticker": "AAA", "name": "A", "sector": "Energy", **company}]
    )
    benchmarks = write_csv(
        tmp_path / "benchmarks.csv",
        [{"ticker": "BENCH", "name": "B", "sector": "Benchmark", **company}],
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
    return data


@pytest.mark.usefixtures("store")
def test_read_relatives_reads_the_configured_benchmark() -> None:
    start = SESSIONS[300]
    out = read_relatives(as_of=SESSIONS[N - 1], start=start)
    assert out["date"].min() == start
    assert set(out["sector"]) == {"Energy"}
    np.testing.assert_allclose(out[BETA_COLUMN], 1.5, rtol=1e-6)
    np.testing.assert_allclose(out[RESID_MOM_COLUMN], 231 * 0.0007, rtol=1e-5)


@pytest.mark.usefixtures("store")
def test_read_relatives_stops_at_as_of() -> None:
    as_of = SESSIONS[350]
    out = read_relatives(as_of=as_of, start=SESSIONS[300])
    assert out["date"].max() == as_of
    with pytest.raises(LookAheadError):
        read_relatives(as_of=as_of, start=SESSIONS[360])
