"""Purged walk-forward splits: train on the past, test on what comes after, never the reverse.

    from finmgr.backtest.splits import walk_forward_splits, label_end_dates

    dates = sorted(labels["date"].unique())
    for split in walk_forward_splits(dates, horizon=21, test_size=63):
        train, test = split.select(labels)
        ...

**Why not k-fold.** Random k-fold puts rows from 2024 in the training set of a
fold that tests on 2022: the model is fitted on the future and scored on the
past, and every metric is fiction. Here every test window lies strictly after
its training window, and the windows move forward in time, `step` dates at a
time (by default one test window, so the test windows tile the axis).

**Why purge.** The label on the row dated t is the return realized over
sessions t+1 .. t+N (day 33) — it *ends* N sessions later. A training row dated
just before the test window therefore carries a label built from prices inside
the test window, and the model would learn what the test period did. The rule
enforced here is strict: a training row's label window `[t, end(t)]` (the
start price p[t] included) must end *before* the first test date. Not even the
boundary price may be shared.

**The embargo.** Between the last training date and the first test date the
splitter skips `embargo` dates, default — and minimum — the label `horizon`.
On a single venue's sessions that alone guarantees the rule: the label of the
last training row ends at most `horizon` sessions later, still before the
test. Asking for a smaller embargo is an error, not a warning.

**Several venues.** The panel's date axis is the union of every venue's
sessions, so N sessions on one venue can span *more* than N dates of the axis
(a date on which only another venue traded sits inside the horizon). Pass
`label_end=label_end_dates(dates, horizon, sessions)` and the splitter purges
on each date's true latest label end — the N-th session after t on the
slowest venue — on top of the embargo. Training dates whose label end is still
unknown (no N-th session yet) are treated as ending never, and purged.

**Train window.** `train_size=None` is an expanding window from the first date;
an integer keeps only the last `train_size` dates before the embargo. A fold
whose purged training set has fewer than `min_train_size` dates is skipped.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

#: Stand-in end for a label that has no N-th session yet: later than anything.
OPEN_END = date.max


def _axis(dates: Iterable[date]) -> list[date]:
    """Sorted unique `datetime.date`s; accepts Timestamps, strings, a Series or an Index."""
    values = pd.to_datetime(pd.Series(list(dates)), errors="raise")
    if values.isna().any():
        raise ValueError("dates contains a missing value")
    return sorted({ts.date() for ts in values})


def label_end_dates(
    dates: Iterable[date],
    horizon: int,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> dict[date, date]:
    """The last date an `horizon`-session label on each of `dates` depends on.

    Without `sessions` the axis itself is the calendar: date i's label ends at
    date i + horizon. With `sessions` (a ticker -> sessions mapping, as for
    :func:`~finmgr.backtest.labels.compute_labels`; repeated calendars are
    fine), the end is the `horizon`-th session strictly after t on each
    calendar, and the latest of them is kept. :data:`OPEN_END` where a calendar
    has no such session yet.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive session count, got {horizon}")
    axis = _axis(dates)
    if sessions is None:
        calendars = [axis]
    else:
        unique = {tuple(_axis(cal)) for cal in sessions.values()}
        calendars = [list(cal) for cal in unique] or [axis]
    out: dict[date, date] = {}
    for t in axis:
        latest = t
        for cal in calendars:
            k = bisect_right(cal, t) + horizon - 1  # horizon-th session after t
            latest = max(latest, cal[k] if k < len(cal) else OPEN_END)
        out[t] = latest
    return out


@dataclass(frozen=True)
class Split:
    """One fold: the dates to train on and the later dates to test on."""

    fold: int
    train: tuple[date, ...]
    test: tuple[date, ...]
    #: Dates inside the train window dropped because their label reaches the test.
    purged: tuple[date, ...] = field(default=())

    @property
    def train_start(self) -> date:
        return self.train[0]

    @property
    def train_end(self) -> date:
        return self.train[-1]

    @property
    def test_start(self) -> date:
        return self.test[0]

    @property
    def test_end(self) -> date:
        return self.test[-1]

    def masks(self, dates: pd.Series | pd.Index) -> tuple[pd.Series, pd.Series]:
        """Boolean (train, test) masks over a long frame's `date` column or level."""
        values = pd.Series(pd.to_datetime(pd.Series(list(dates))).dt.date.to_numpy())
        train = values.isin(set(self.train))
        test = values.isin(set(self.test))
        if isinstance(dates, pd.Series):
            train.index = test.index = dates.index
        return train, test

    def select(
        self, frame: pd.DataFrame, column: str = "date"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(train, test) rows of `frame`, by its `column` column or index level."""
        by_column = column in frame.columns
        dates = frame[column] if by_column else frame.index.get_level_values(column)
        train, test = self.masks(dates)
        return frame[train.to_numpy()], frame[test.to_numpy()]


def walk_forward_splits(
    dates: Iterable[date],
    *,
    horizon: int,
    test_size: int,
    train_size: int | None = None,
    min_train_size: int | None = None,
    embargo: int | None = None,
    step: int | None = None,
    label_end: Mapping[date, date] | None = None,
) -> Iterator[Split]:
    """Yield purged, embargoed walk-forward folds over `dates`; see the module docstring.

    Sizes count dates of the (sorted, de-duplicated) axis. The first test
    window starts as soon as `min_train_size` (default `test_size`) training
    dates fit before the embargo; the last may be shorter than `test_size`.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive session count, got {horizon}")
    if test_size < 1:
        raise ValueError(f"test_size must be positive, got {test_size}")
    embargo = horizon if embargo is None else embargo
    if embargo < horizon:
        raise ValueError(
            f"embargo ({embargo}) is shorter than the label horizon ({horizon}): "
            "training labels would overlap the test window"
        )
    step = test_size if step is None else step
    min_train = test_size if min_train_size is None else min_train_size
    if step < 1 or min_train < 1:
        raise ValueError("step and min_train_size must be positive")
    if train_size is not None and train_size < min_train:
        raise ValueError(f"train_size ({train_size}) is below min_train_size ({min_train})")

    axis = _axis(dates)
    ends = label_end if label_end is not None else label_end_dates(axis, horizon)
    fold = 0
    for start in range(min_train + embargo, len(axis), step):
        test = tuple(axis[start : start + test_size])
        stop = start - embargo
        first = 0 if train_size is None else max(0, stop - train_size)
        window = axis[first:stop]
        keep = [t for t in window if ends.get(t, OPEN_END) < test[0]]
        if len(keep) < min_train:
            continue
        purged = tuple(t for t in window if ends.get(t, OPEN_END) >= test[0])
        yield Split(fold=fold, train=tuple(keep), test=test, purged=purged)
        fold += 1
