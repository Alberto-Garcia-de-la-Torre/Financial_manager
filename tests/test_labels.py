"""Day 33: labels, the forward-looking columns.

The acceptance criterion: the label at t equals the realized return from t+1
through t+N. It is checked against a loop that compounds the daily returns of
sessions t+1, ..., t+N by hand and never calls `shift`, and by perturbation:
moving the price at t+N moves the label at t, moving any price after t+N or
before t does not. Around it: the session grid (a missing bar makes the
labels that start or end on it NaN instead of stretching the horizon), the
cross-sectional excess and top-quintile columns, and point in time.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from finmgr.backtest.labels import (
    LABEL_HORIZONS,
    compute_labels,
    excess_column,
    fwd_column,
    label_columns,
    read_labels,
    top_column,
)
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data.actions import merge_actions
from finmgr.data.pit import LookAheadError
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS
from finmgr.features.returns import compute_returns, simple_column, venue_sessions

D = dt.date
SESSIONS = venue_sessions("XNYS", D(2023, 1, 1), D(2024, 6, 28))


def panel(n: int = 120, n_tickers: int = 3, seed: int = 33) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = SESSIONS[:n]
    frames = [
        pd.DataFrame(
            {
                "ticker": f"T{i:03d}",
                "date": days,
                "adj_close_eur": 100 * np.cumprod(1 + rng.normal(0.0, 0.02, len(days))),
            }
        )
        for i in range(n_tickers)
    ]
    return pd.concat(frames, ignore_index=True)


def realized(prices: list[float], t: int, n: int) -> float:
    """Compound the daily returns of sessions t+1 .. t+n, one at a time."""
    growth = 1.0
    for k in range(t + 1, t + n + 1):
        growth *= prices[k] / prices[k - 1]
    return growth - 1.0


def one(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return frame[frame["ticker"] == ticker].reset_index(drop=True)


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", LABEL_HORIZONS)
def test_label_at_t_is_the_return_realized_from_t_plus_1_through_t_plus_n(n: int) -> None:
    data = panel()
    labels = compute_labels(data, min_names=1)
    for ticker in data["ticker"].unique():
        prices = one(data, ticker)["adj_close_eur"].tolist()
        got = one(labels, ticker)[fwd_column(n)].tolist()
        for t in range(len(prices)):
            if t + n < len(prices):
                assert got[t] == pytest.approx(realized(prices, t, n), rel=1e-12, abs=1e-14)
            else:
                assert np.isnan(got[t]), f"{ticker} row {t} has no future yet"


@pytest.mark.parametrize("n", LABEL_HORIZONS)
def test_label_is_the_backward_return_moved_n_sessions_earlier(n: int) -> None:
    data = panel()
    labels = one(compute_labels(data, min_names=1), "T000")
    backward = one(compute_returns(data, horizons=(n,)), "T000")[simple_column(n)]
    # The label dated t and the feature dated t+N describe the same window.
    np.testing.assert_allclose(labels[fwd_column(n)].iloc[:-n], backward.iloc[n:], rtol=1e-12)


@pytest.mark.parametrize("n", LABEL_HORIZONS)
def test_label_at_t_reads_exactly_the_prices_at_t_and_t_plus_n(n: int) -> None:
    data = one(panel(n_tickers=1), "T000")
    t = 40
    base = compute_labels(data, min_names=1)[fwd_column(n)].iloc[t]

    def label_after(index: int, factor: float) -> float:
        bumped = data.copy()
        bumped.loc[index, "adj_close_eur"] *= factor
        return compute_labels(bumped, min_names=1)[fwd_column(n)].iloc[t]

    # The end point moves it ...
    assert label_after(t + n, 1.1) != pytest.approx(base)
    # ... and so does the start point: the label is the return *after* t's close.
    assert label_after(t, 1.1) != pytest.approx(base)
    # Nothing before t, nothing strictly inside the window, nothing after t+N.
    for index in [0, t - 1, t + 1, t + n - 1, t + n + 1, len(data) - 1]:
        if index not in (t, t + n):
            assert label_after(index, 1.3) == pytest.approx(base, rel=1e-12)


def test_the_label_does_not_include_day_t_own_return() -> None:
    # A +50% jump on day t is a feature of t, not part of t's label.
    days = SESSIONS[:30]
    prices = [100.0] * 10 + [150.0] * 20
    frame = pd.DataFrame({"ticker": "A", "date": days, "adj_close_eur": prices})
    labels = compute_labels(frame, min_names=1)
    assert labels[fwd_column(5)].iloc[10] == pytest.approx(0.0)
    assert labels[fwd_column(5)].iloc[9] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Sessions, not rows
# ---------------------------------------------------------------------------


def test_a_missing_session_makes_labels_touching_it_nan_without_stretching() -> None:
    data = one(panel(n_tickers=1), "T000")
    gap = 50
    holed = data.drop(index=gap).reset_index(drop=True)
    labels = compute_labels(holed, sessions={"T000": SESSIONS[:120]}, min_names=1)
    by_date = labels.set_index("date")[fwd_column(5)]
    prices = data["adj_close_eur"].tolist()
    days = data["date"].tolist()
    # Started 5 sessions before the hole: ends on it, so unknown, not a 6-session return.
    assert np.isnan(by_date[days[gap - 5]])
    # Spans the hole but neither end is on it: still exact.
    assert by_date[days[gap - 2]] == pytest.approx(realized(prices, gap - 2, 5), rel=1e-12)
    # Counting rows instead would have stretched it.
    by_rows = compute_labels(holed, min_names=1).set_index("date")[fwd_column(5)]
    assert by_rows[days[gap - 5]] == pytest.approx(prices[gap + 1] / prices[gap - 5] - 1)


# ---------------------------------------------------------------------------
# The cross-section
# ---------------------------------------------------------------------------


def test_excess_is_return_minus_the_date_mean_and_sums_to_zero() -> None:
    data = panel(n_tickers=100, n=60)
    labels = compute_labels(data)
    for n in LABEL_HORIZONS:
        known = labels.dropna(subset=[fwd_column(n)])
        mean = known.groupby("date")[fwd_column(n)].transform("mean")
        np.testing.assert_allclose(known[excess_column(n)], known[fwd_column(n)] - mean)
        sums = known.groupby("date")[excess_column(n)].sum()
        assert (sums.abs() < 1e-12).all()


def test_top_quintile_marks_the_best_fifth_of_each_date() -> None:
    data = panel(n_tickers=100, n=60)
    labels = compute_labels(data)
    for n in LABEL_HORIZONS:
        known = labels.dropna(subset=[fwd_column(n)])
        assert set(known[top_column(n)].unique()) == {0.0, 1.0}
        for _, day in known.groupby("date"):
            top = day[day[top_column(n)] == 1.0]
            rest = day[day[top_column(n)] == 0.0]
            assert len(top) == 20
            assert top[fwd_column(n)].min() > rest[fwd_column(n)].max()


def test_top_quintile_of_ten_names_is_two() -> None:
    days = SESSIONS[:10]
    frames = [
        pd.DataFrame(
            {
                "ticker": f"T{i}",
                "date": days,
                "adj_close_eur": 100 * (1 + 0.01 * i) ** np.arange(10),
            }
        )
        for i in range(10)
    ]
    labels = compute_labels(pd.concat(frames), horizons=(5,))
    day0 = labels[labels["date"] == days[0]].set_index("ticker")[top_column(5)]
    assert day0[day0 == 1.0].index.tolist() == ["T8", "T9"]


def test_too_few_names_and_the_last_n_sessions_get_nan() -> None:
    data = panel(n_tickers=5)
    labels = compute_labels(data)  # MIN_NAMES is 10
    assert labels[[excess_column(5), top_column(5)]].isna().all().all()
    assert labels[fwd_column(5)].notna().any()
    tail = labels.groupby("ticker").tail(21)
    assert tail[[fwd_column(21), excess_column(21), top_column(21)]].isna().all().all()


def test_mask_takes_names_out_of_the_cross_section() -> None:
    data = panel(n_tickers=20, n=40)
    data["tradable"] = data["ticker"] != "T000"
    labels = compute_labels(data, mask="tradable")
    out = labels[labels["ticker"] == "T000"]
    assert out[[excess_column(5), top_column(5)]].isna().all().all()
    assert out[fwd_column(5)].notna().any()  # its own return is still known
    rest = labels[labels["ticker"] != "T000"].dropna(subset=[fwd_column(5)])
    assert (rest.groupby("date")[excess_column(5)].sum().abs() < 1e-12).all()


# ---------------------------------------------------------------------------
# Shape and guards
# ---------------------------------------------------------------------------


def test_columns_and_order() -> None:
    labels = compute_labels(panel())
    assert list(labels.columns) == ["ticker", "date", "adj_close_eur", *label_columns()]
    assert labels[["ticker", "date"]].equals(
        labels[["ticker", "date"]].sort_values(["ticker", "date"])
    )


def test_input_order_does_not_matter() -> None:
    data = panel()
    shuffled = data.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(compute_labels(data), compute_labels(shuffled))


@pytest.mark.parametrize("bad", [(0,), (-5,), ()])
def test_rejects_non_positive_horizons(bad: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        compute_labels(panel(), horizons=bad)


def test_rejects_duplicates_and_missing_columns() -> None:
    data = panel()
    with pytest.raises(ValueError):
        compute_labels(pd.concat([data, data.head(1)]))
    with pytest.raises(KeyError):
        compute_labels(data.drop(columns="adj_close_eur"))


# ---------------------------------------------------------------------------
# Through the store: point in time
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """One XNYS stock with a 2-for-1 split on session 100, in a settings file under tmp_path."""
    data = tmp_path / "data"
    days = SESSIONS[:150]
    closes = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.0, 0.02, len(days)))
    true = [float(c) for c in closes]
    raw = [c / 2 if i >= 100 else c for i, c in enumerate(true)]
    write_bars(
        pd.DataFrame(
            {
                "ticker": "AAA",
                "date": days,
                "open": raw,
                "high": raw,
                "low": raw,
                "close": raw,
                "volume": 1_000.0,
                "close_eur": [c * 0.9 for c in raw],
            }
        ),
        root=data / "bars" / "daily",
    )
    merge_actions(
        pd.DataFrame({"ticker": "AAA", "date": [days[100]], "action": "split", "value": [2.0]}),
        root=data / "actions",
    )
    company = {"exchange": "XNYS", "currency": "USD", "sector": "Industrials", "country": "X"}
    paths = {}
    for name, row in [
        ("universe", {"ticker": "AAA", "name": "A", **company}),
        ("benchmarks", {"ticker": "BENCH", "name": "B", **company}),
    ]:
        path = tmp_path / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        paths[name] = str(path)
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data),
                "universe_path": paths["universe"],
                "benchmarks_path": paths["benchmarks"],
                "benchmark": "BENCH",
                "log_dir": str(tmp_path / "logs"),
                "history_start": "2023-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return {"days": days, "true": true}


def test_read_labels_across_a_split_is_the_true_forward_return(store: dict) -> None:
    days, true = store["days"], store["true"]
    out = read_labels("AAA", as_of=days[-1]).set_index("date")
    t = 97  # the 5-session window crosses the split on session 100
    assert out.loc[days[t], fwd_column(5)] == pytest.approx(realized(true, t, 5), rel=1e-9)


def test_read_labels_as_of_t_does_not_know_the_last_n_sessions(store: dict) -> None:
    days = store["days"]
    cut = 80
    then = read_labels("AAA", as_of=days[cut]).set_index("date")
    later = read_labels("AAA", as_of=days[-1]).set_index("date")
    assert then.index.max() == days[cut]
    for n in LABEL_HORIZONS:
        # The last n rows wait for their future; the rest already agree with hindsight.
        assert then[fwd_column(n)].iloc[-n:].isna().all()
        known = then[fwd_column(n)].iloc[:-n]
        np.testing.assert_allclose(known, later.loc[known.index, fwd_column(n)], rtol=1e-12)


def test_read_labels_needs_as_of_and_refuses_the_future(store: dict) -> None:
    with pytest.raises(TypeError):
        read_labels("AAA")  # type: ignore[call-arg]
    with pytest.raises(LookAheadError):
        read_labels("AAA", as_of=store["days"][10], start=store["days"][20])
    out = read_labels("AAA", as_of=store["days"][-1], start=store["days"][40])
    assert out["date"].min() == store["days"][40]
