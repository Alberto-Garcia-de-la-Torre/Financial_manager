"""Point-in-time discipline: no read returns a row dated after `as_of`.

    from finmgr.data.pit import LATEST, LookAheadError
    from finmgr.data.store import read_bars

    read_bars("SAN.MC", as_of="2024-06-28")         # nothing after 28 Jun 2024
    read_bars("SAN.MC", as_of=LATEST)               # "as of now": today

Look-ahead bias is the most common reason a backtest looks brilliant and live
trading does not: somewhere, a decision dated Tuesday was computed from a bar
dated Wednesday. It almost never looks like a bug. It looks like a forgotten
`end=`, a factor computed over "the whole history", a join that pulled in the
next session. So the guard is not left to each caller:

* Every read function in the data layer — :func:`~finmgr.data.store.read_bars`,
  :func:`~finmgr.data.actions.read_actions`, :func:`~finmgr.data.fx.read_fx`,
  :func:`~finmgr.data.adjust.read_adjusted` and
  :func:`~finmgr.data.quarantine.read_quarantine` — takes `as_of` as a
  *required* keyword. There is no default, because a default is exactly the
  forgotten argument this exists to catch. A caller that genuinely wants
  everything on disk says so with :data:`LATEST`.
* The bound goes into the query (`date <= as_of`), so future rows are never
  materialised.
* The result is then checked again by :func:`guard`, which raises
  :class:`LookAheadError` rather than trimming. Trimming would hide the bug —
  a query that leaks is broken, and the run that hit it must stop.
* An `end` later than `as_of` is refused outright, not silently clamped: the
  caller asked for the future, and should hear about it.

`tests/test_pit.py` is the leak test. It plants future rows in every store and
tries to get them out by every route, including one where the query's own
filter is sabotaged. Re-run it after every change to the data layer.

What this does *not* cover: :func:`~finmgr.data.store.coverage` and
:func:`~finmgr.data.store.max_stored_dates` describe the store's extent for
the ingest and return no rows of data; the manifest is a log of runs, dated by
when they happened. None of them feed a feature.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Final

import pandas as pd


class LookAheadError(RuntimeError):
    """A read would have returned data dated after its `as_of`."""


class _Latest:
    """The sentinel for "as of now"; resolves to today's date at call time."""

    _instance: _Latest | None = None

    def __new__(cls) -> _Latest:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "LATEST"


#: Pass as `as_of` to mean "everything up to today". Maintenance code — a merge
#: rewriting a partition, the quality checks — uses it; research code should
#: always name a date. A row on disk dated after today is itself a bug; a read
#: at LATEST never returns it.
LATEST: Final = _Latest()

AsOf = date | datetime | str | _Latest


def resolve_as_of(as_of: AsOf) -> date:
    """Turn an `as_of` argument into a date, refusing None.

    None is refused rather than read as "no bound": the whole point is that an
    unbounded read has to be asked for by name (:data:`LATEST`).
    """
    if as_of is None:
        raise TypeError("as_of is required; pass a date, or pit.LATEST for 'up to today'")
    if isinstance(as_of, _Latest):
        return date.today()
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    if isinstance(as_of, str):
        return date.fromisoformat(as_of)
    raise TypeError(
        f"as_of must be a date, datetime, ISO string or LATEST, got {type(as_of).__name__}"
    )


def bound_end(end: date | None, as_of: date, *, start: date | None = None) -> date:
    """The effective inclusive end of a window: `end`, or `as_of` if none.

    Raises :class:`LookAheadError` when `end` — or `start` — is after `as_of`.
    """
    if start is not None and start > as_of:
        raise LookAheadError(f"start {start} is after as_of {as_of}: that window reads the future")
    if end is None:
        return as_of
    if end > as_of:
        raise LookAheadError(f"end {end} is after as_of {as_of}: that window reads the future")
    return end


def guard(frame: pd.DataFrame, as_of: date, *, what: str, column: str = "date") -> pd.DataFrame:
    """Raise if any row of `frame` is dated after `as_of`; else return it.

    The last line of defence, run on every result after the query has already
    filtered. It should never fire. If it does, the query or something feeding
    it is wrong, and the error says which rows got through.
    """
    if frame.empty:
        return frame
    # Through datetime64 so every spelling of a date column compares the same
    # way: date32[pyarrow], object dates and timestamps alike.
    dates = pd.to_datetime(pd.Series(frame[column]).astype(object))
    late = dates[dates > pd.Timestamp(as_of)]
    if not late.empty:
        sample = ", ".join(str(value.date()) for value in late.head(5))
        raise LookAheadError(
            f"{what}: {len(late)} row(s) dated after as_of {as_of} (e.g. {sample})"
        )
    return frame
