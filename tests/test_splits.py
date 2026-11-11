"""Day 34: purged walk-forward splits.

The acceptance criterion: zero overlap between any training label window and
its test window. It is checked three ways — on the label-end dates the splitter
purges with; by brute force, recomputing every label from scratch after
scrambling every price from the first test date on and requiring that no
training label moves; and on a two-venue axis, where the embargo alone is not
enough and the session-aware label ends have to do the work. Around it: the
walk-forward shape (test strictly after train, windows tiling forward), the
embargo floor, rolling windows and selecting rows of a long frame.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from finmgr.backtest.labels import LABEL_HORIZONS, compute_labels, fwd_column
from finmgr.backtest.splits import (
    OPEN_END,
    Split,
    label_end_dates,
    walk_forward_splits,
)
from finmgr.features.returns import venue_sessions

D = dt.date
NYSE = venue_sessions("XNYS", D(2022, 1, 1), D(2023, 12, 29))
XETRA = venue_sessions("XETR", D(2022, 1, 1), D(2023, 12, 29))


def prices(days: list[dt.date], tickers: list[str], seed: int = 34) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [
        pd.DataFrame(
            {
                "ticker": t,
                "date": days,
                "adj_close_eur": 100 * np.cumprod(1 + rng.normal(0.0, 0.02, len(days))),
            }
        )
        for t in tickers
    ]
    return pd.concat(frames, ignore_index=True)


def label_window(t: dt.date, ends: dict[dt.date, dt.date]) -> tuple[dt.date, dt.date]:
    """The dates whose prices the label on t uses: p[t] through p[end(t)]."""
    return t, ends[t]


def overlaps(a: tuple[dt.date, dt.date], b: tuple[dt.date, dt.date]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


# --- the acceptance criterion ------------------------------------------------


@pytest.mark.parametrize("horizon", LABEL_HORIZONS)
@pytest.mark.parametrize("train_size", [None, 120])
def test_no_training_label_window_overlaps_its_test_window(horizon, train_size):
    ends = label_end_dates(NYSE, horizon)
    splits = list(walk_forward_splits(NYSE, horizon=horizon, test_size=42, train_size=train_size))
    assert len(splits) >= 5
    for split in splits:
        # The test window as the prices it touches: its own dates, plus the
        # horizon its own labels run into.
        test_window = (split.test_start, max(ends[split.test_end], split.test_end))
        for t in split.train:
            window = label_window(t, ends)
            assert window[1] < split.test_start, (split.fold, t)
            assert not overlaps(window, test_window), (split.fold, t)
        # And the embargo really is there: at least `horizon` dates skipped.
        gap = NYSE.index(split.test_start) - NYSE.index(split.train_end) - 1
        assert gap >= horizon


@pytest.mark.parametrize("horizon", LABEL_HORIZONS)
def test_scrambling_the_test_period_leaves_every_training_label_unchanged(horizon):
    """Brute force: the labels themselves, not the dates the splitter reasons with."""
    days = NYSE[:300]
    base = prices(days, ["AAA", "BBB", "CCC"])
    labels = compute_labels(base, horizons=[horizon])
    column = fwd_column(horizon)
    rng = np.random.default_rng(7)
    for split in walk_forward_splits(days, horizon=horizon, test_size=40):
        scrambled = base.copy()
        future = scrambled["date"] >= split.test_start
        scrambled.loc[future, "adj_close_eur"] *= rng.uniform(0.5, 1.5, future.sum())
        again = compute_labels(scrambled, horizons=[horizon])
        train_rows = labels["date"].isin(set(split.train))
        assert train_rows.sum() == 3 * len(split.train)
        before = labels.loc[train_rows, column]
        after = again.loc[train_rows, column]
        assert before.notna().all()
        pd.testing.assert_series_equal(before, after)
        # The control: the dates the purge removed *would* have moved.
        if split.purged:
            purged_rows = labels["date"].isin(set(split.purged))
            assert not np.allclose(labels.loc[purged_rows, column], again.loc[purged_rows, column])


def test_two_venues_need_session_aware_label_ends():
    """On the union axis the embargo alone lets a training label reach the test."""
    horizon = 5
    axis = sorted(set(NYSE) | set(XETRA))
    sessions = {"AAPL": NYSE, "SAP.DE": XETRA}
    true_ends = label_end_dates(axis, horizon, sessions)

    naive = list(walk_forward_splits(axis, horizon=horizon, test_size=21))
    leaky = [
        s for s in naive if any(true_ends[t] >= s.test_start for t in s.train)
    ]  # the embargo counts axis dates, the label counts one venue's sessions
    assert leaky, "expected the naive split to overlap on a holiday boundary"

    purged = list(walk_forward_splits(axis, horizon=horizon, test_size=21, label_end=true_ends))
    assert len(purged) == len(naive)
    for split in purged:
        for t in split.train:
            assert true_ends[t] < split.test_start
    assert any(s.purged for s in purged)

    # And on the labels: scrambling the test period moves no training label.
    base = prices(NYSE, ["AAPL"]).pipe(
        lambda f: pd.concat([f, prices(XETRA, ["SAP.DE"], seed=5)], ignore_index=True)
    )
    labels = compute_labels(base, horizons=[horizon], sessions=sessions, min_names=1)
    column = fwd_column(horizon)
    for split in purged[:6]:
        scrambled = base.copy()
        future = scrambled["date"] >= split.test_start
        scrambled.loc[future, "adj_close_eur"] *= 1.37
        again = compute_labels(scrambled, horizons=[horizon], sessions=sessions, min_names=1)
        rows = labels["date"].isin(set(split.train))
        pd.testing.assert_series_equal(labels.loc[rows, column], again.loc[rows, column])


# --- walk-forward shape ------------------------------------------------------


def test_folds_move_forward_and_test_windows_tile_the_axis():
    splits = list(walk_forward_splits(NYSE, horizon=21, test_size=63))
    tested = [d for s in splits for d in s.test]
    assert tested == NYSE[NYSE.index(splits[0].test_start) :]  # contiguous, no gaps, no repeats
    for i, split in enumerate(splits):
        assert split.fold == i
        assert split.train_end < split.test_start  # never trained on the future
        assert set(split.train).isdisjoint(split.test)
        if i:
            assert split.test_start > splits[i - 1].test_end
            assert split.train_start == NYSE[0]  # expanding
            assert len(split.train) > len(splits[i - 1].train)
    assert len(splits[-1].test) <= 63


def test_rolling_window_keeps_the_last_train_size_dates():
    splits = list(
        walk_forward_splits(NYSE, horizon=5, test_size=20, train_size=100, min_train_size=100)
    )
    assert {len(s.train) for s in splits} == {100}
    assert splits[1].train_start > splits[0].train_start


def test_step_smaller_than_test_size_overlaps_tests_but_still_purges():
    ends = label_end_dates(NYSE, 21)
    splits = list(walk_forward_splits(NYSE, horizon=21, test_size=63, step=21))
    assert splits[1].test_start == NYSE[NYSE.index(splits[0].test_start) + 21]
    assert all(ends[t] < s.test_start for s in splits for t in s.train)


def test_embargo_shorter_than_the_horizon_is_refused():
    with pytest.raises(ValueError, match="embargo"):
        list(walk_forward_splits(NYSE, horizon=21, test_size=63, embargo=20))
    longer = next(walk_forward_splits(NYSE, horizon=5, test_size=20, embargo=10))
    assert NYSE.index(longer.test_start) - NYSE.index(longer.train_end) - 1 == 10


def test_unknown_label_ends_are_purged():
    days = NYSE[:60]
    ends = label_end_dates(days, 5)
    assert ends[days[-1]] == OPEN_END and ends[days[-6]] == days[-1]
    ends = dict(ends)
    victim = days[2]
    ends[victim] = OPEN_END  # say its label is not known yet
    split = next(walk_forward_splits(days, horizon=5, test_size=10, label_end=ends))
    assert victim not in split.train and victim in split.purged


def test_axis_is_sorted_and_deduplicated_and_accepts_timestamps():
    shuffled = list(pd.to_datetime(NYSE[:200])) * 2
    np.random.default_rng(1).shuffle(shuffled)
    a = list(walk_forward_splits(shuffled, horizon=5, test_size=20))
    b = list(walk_forward_splits(NYSE[:200], horizon=5, test_size=20))
    assert a == b


def test_select_splits_a_long_frame_by_date_column_or_index_level():
    days = NYSE[:120]
    frame = prices(days, ["AAA", "BBB"])
    split = next(walk_forward_splits(days, horizon=5, test_size=20))
    train, test = split.select(frame)
    assert set(train["date"]) == set(split.train)
    assert set(test["date"]) == set(split.test)
    assert len(train) == 2 * len(split.train)
    indexed = frame.set_index(["date", "ticker"])
    train_i, test_i = split.select(indexed)
    assert len(train_i) == len(train) and len(test_i) == len(test)


def test_split_is_a_plain_value():
    s = Split(fold=0, train=(D(2023, 1, 3),), test=(D(2023, 1, 10),))
    assert (s.train_start, s.train_end, s.test_start, s.test_end) == (
        D(2023, 1, 3),
        D(2023, 1, 3),
        D(2023, 1, 10),
        D(2023, 1, 10),
    )
