"""Day 23: returns, strictly backward-looking.

The acceptance criterion: `ret_5d` at date t uses no observation later than t.
It is tested two ways that do not trust the implementation — every price after
t is scrambled and `ret_5d` at t must not move, and the panel computed on data
truncated at t must agree with the one computed on all of it — and then
through the store, where the series is read as of t and as of a later date
with a split in between. Around it: a hand calculation per horizon, the log
and simple returns agreeing, and the session alignment that keeps a missing
bar from stretching a window.
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
from finmgr.features.returns import (
    HORIZONS,
    compute_returns,
    log_column,
    read_returns,
    return_columns,
    simple_column,
    venue_sessions,
)

D = dt.date
#: XNYS sessions; 2024-01-15 (MLK day) and 2024-02-19 (Presidents' Day) are out.
SESSIONS = venue_sessions("XNYS", D(2023, 1, 1), D(2024, 6, 28))


def panel(n: int = 300, tickers: tuple[str, ...] = ("AAA", "BBB"), seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = SESSIONS[:n]
    frames = [
        pd.DataFrame(
            {
                "ticker": ticker,
                "date": days,
                "adj_close_eur": 100 * np.cumprod(1 + rng.normal(0.0, 0.02, len(days))),
            }
        )
        for ticker in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def at(frame: pd.DataFrame, ticker: str, day: dt.date) -> pd.Series:
    row = frame[(frame["ticker"] == ticker) & (frame["date"] == day)]
    assert len(row) == 1
    return row.iloc[0]


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [5, 6, 100, 299])
def test_ret_5d_at_t_uses_no_observation_later_than_t(cut: int) -> None:
    prices = panel()
    t = SESSIONS[cut]
    before = compute_returns(prices)

    # Scramble every price after t — by a different random factor per row, so
    # no ratio survives — and recompute.
    rng = np.random.default_rng(0)
    later = prices["date"] > t
    scrambled = prices.copy()
    scrambled.loc[later, "adj_close_eur"] *= rng.uniform(0.1, 10.0, later.sum())
    after = compute_returns(scrambled)

    for ticker in ("AAA", "BBB"):
        assert at(after, ticker, t)["ret_5d"] == at(before, ticker, t)["ret_5d"]
        assert at(after, ticker, t)["logret_5d"] == at(before, ticker, t)["logret_5d"]
        # And the scramble did reach the rows after t: the test can fail.
        if cut < 299:
            assert at(after, ticker, SESSIONS[cut + 5])["ret_5d"] != pytest.approx(
                at(before, ticker, SESSIONS[cut + 5])["ret_5d"]
            )


def test_no_row_changes_when_the_future_is_cut_off() -> None:
    prices = panel()
    full = compute_returns(prices)
    for cut in (0, 4, 5, 62, 251, 252, 299):
        t = SESSIONS[cut]
        truncated = compute_returns(prices[prices["date"] <= t])
        expected = full[full["date"] <= t].reset_index(drop=True)
        pd.testing.assert_frame_equal(truncated, expected)


def test_ret_5d_is_the_price_today_over_the_price_five_sessions_ago() -> None:
    prices = panel(tickers=("AAA",))
    out = compute_returns(prices)
    p = prices.set_index("date")["adj_close_eur"]
    t = SESSIONS[40]
    assert at(out, "AAA", t)["ret_5d"] == pytest.approx(p[t] / p[SESSIONS[35]] - 1)


# ---------------------------------------------------------------------------
# Every horizon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", HORIZONS)
def test_each_horizon_matches_a_hand_calculation(h: int) -> None:
    prices = panel(tickers=("AAA",))
    out = compute_returns(prices).set_index("date")
    p = prices.set_index("date")["adj_close_eur"]
    for i in (h, h + 1, 299):
        t, then = SESSIONS[i], SESSIONS[i - h]
        assert out.loc[t, simple_column(h)] == pytest.approx(p[t] / p[then] - 1, rel=1e-12)
        assert out.loc[t, log_column(h)] == pytest.approx(math.log(p[t] / p[then]), rel=1e-12)
    # The first h sessions have no price h sessions back: NaN, not 0.
    assert out[simple_column(h)].iloc[:h].isna().all()
    assert out[log_column(h)].iloc[:h].isna().all()
    assert out[simple_column(h)].iloc[h:].notna().all()


def test_log_returns_are_log1p_of_simple_ones_and_add_up() -> None:
    out = compute_returns(panel())
    for h in HORIZONS:
        np.testing.assert_allclose(
            out[log_column(h)], np.log1p(out[simple_column(h)]), rtol=1e-10, equal_nan=True
        )
    # Five one-session log returns sum to the five-session one.
    aaa = out[out["ticker"] == "AAA"].reset_index(drop=True)
    summed = aaa["logret_1d"].rolling(5).sum()
    np.testing.assert_allclose(summed.iloc[5:], aaa["logret_5d"].iloc[5:], rtol=1e-10)


def test_tickers_never_borrow_each_others_prices() -> None:
    prices = panel()
    # BBB's first session must be NaN even though AAA's last row sits above it.
    out = compute_returns(prices.sort_values(["ticker", "date"]))
    first_bbb = out[out["ticker"] == "BBB"].iloc[0]
    assert np.isnan(first_bbb["ret_1d"])
    assert list(out.columns) == ["ticker", "date", "adj_close_eur", *return_columns()]
    assert len(out) == len(prices)


def test_shuffled_input_gives_the_same_answer() -> None:
    prices = panel()
    shuffled = prices.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(compute_returns(shuffled), compute_returns(prices))


@pytest.mark.parametrize("bad", [[0], [-5], [], [5, 0]])
def test_a_horizon_that_is_not_a_positive_lookback_is_refused(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_returns(panel(), horizons=bad)


def test_a_duplicated_row_is_refused() -> None:
    prices = panel(tickers=("AAA",))
    with pytest.raises(ValueError, match="more than one row"):
        compute_returns(pd.concat([prices, prices.iloc[[10]]]))


def test_a_nan_price_is_never_filled_from_a_neighbour() -> None:
    prices = panel(tickers=("AAA",))
    prices.loc[50, "adj_close_eur"] = np.nan
    out = compute_returns(prices).set_index("date")
    assert np.isnan(out.loc[SESSIONS[50], "ret_1d"])
    assert np.isnan(out.loc[SESSIONS[51], "ret_1d"])
    assert np.isnan(out.loc[SESSIONS[55], "ret_5d"])
    assert not np.isnan(out.loc[SESSIONS[52], "ret_1d"])


# ---------------------------------------------------------------------------
# Session alignment
# ---------------------------------------------------------------------------


def test_venue_sessions_skip_holidays() -> None:
    assert D(2024, 1, 15) not in SESSIONS  # MLK day
    assert D(2024, 1, 12) in SESSIONS and D(2024, 1, 16) in SESSIONS


def test_a_missing_bar_does_not_stretch_the_window() -> None:
    prices = panel(tickers=("AAA",))
    gone = SESSIONS[100]
    holed = prices[prices["date"] != gone]
    p = prices.set_index("date")["adj_close_eur"]

    aligned = compute_returns(holed, sessions={"AAA": SESSIONS[:300]}).set_index("date")
    by_rows = compute_returns(holed).set_index("date")

    # A window spanning the hole is still exactly five sessions...
    t = SESSIONS[103]
    assert aligned.loc[t, "ret_5d"] == pytest.approx(p[t] / p[SESSIONS[98]] - 1)
    # ...where counting rows would have reached back six.
    assert by_rows.loc[t, "ret_5d"] == pytest.approx(p[t] / p[SESSIONS[97]] - 1)
    # A window with the hole as an endpoint has no answer.
    assert np.isnan(aligned.loc[SESSIONS[105], "ret_5d"])
    assert np.isnan(aligned.loc[SESSIONS[101], "ret_1d"])
    # The missing session does not come back as a row.
    assert gone not in aligned.index
    assert len(aligned) == len(holed)


def test_a_bar_off_the_calendar_is_kept_as_a_session() -> None:
    prices = panel(n=10, tickers=("AAA",))
    out = compute_returns(prices, sessions={"AAA": SESSIONS[:5]})
    assert len(out) == 10
    assert out["ret_1d"].iloc[1:].notna().all()


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
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """One XNYS stock with a 2-for-1 split, in a settings file under tmp_path."""
    data = tmp_path / "data"
    days = SESSIONS[:300]
    split = days[200]
    closes = 100 * np.cumprod(1 + np.random.default_rng(5).normal(0.0, 0.02, len(days)))
    closes[200:] /= 2  # the raw tape halves on the split day
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
        pd.DataFrame({"ticker": "AAA", "date": [split], "action": "split", "value": [2.0]}),
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
                "history_start": "2023-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return {"days": days, "split": split, "closes": closes}


def test_ret_5d_read_as_of_t_equals_ret_5d_read_later(store: dict) -> None:
    days = store["days"]
    t = days[150]  # before the split, which a later read back-adjusts
    then = read_returns("AAA", as_of=t)
    later = read_returns("AAA", as_of=days[-1])

    assert then["date"].max() == t
    assert at(then, "AAA", t)["ret_5d"] == pytest.approx(at(later, "AAA", t)["ret_5d"], rel=1e-12)
    common = later[later["date"] <= t].reset_index(drop=True)
    np.testing.assert_allclose(
        then[return_columns()].to_numpy(), common[return_columns()].to_numpy(), equal_nan=True
    )


def test_the_split_is_not_a_minus_fifty_percent_day(store: dict) -> None:
    days, closes = store["days"], store["closes"]
    out = read_returns("AAA", as_of=days[-1]).set_index("date")
    # Raw tape: p[200] is half the true price. Adjusted, the 1-day return is
    # the stock's own move, and the 5-day window across the split is too.
    true = [c * (2 if i >= 200 else 1) for i, c in enumerate(closes)]
    assert out.loc[days[200], "ret_1d"] == pytest.approx(true[200] / true[199] - 1)
    assert out.loc[days[203], "ret_5d"] == pytest.approx(true[203] / true[198] - 1)


def test_read_returns_trims_to_start_but_keeps_the_lookback(store: dict) -> None:
    days = store["days"]
    out = read_returns("AAA", as_of=days[-1], start=days[280], horizons=(1, 252))
    assert out["date"].min() == days[280]
    assert out["ret_252d"].notna().all()


def test_read_returns_needs_as_of_and_refuses_the_future(store: dict) -> None:
    with pytest.raises(TypeError):
        read_returns("AAA")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_returns("AAA", as_of=store["days"][10], start=store["days"][20])
