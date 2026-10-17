"""The leak test: no read in the data layer returns a row dated after `as_of`.

Day 21's acceptance criterion, and the test to re-run after *every* change to
the data layer. Every store gets rows on both sides of `AS_OF`, and then each
reader is attacked by every route a future row could take:

* the honest read — the future rows must simply not be there;
* asking for them — an `end` or `start` after `as_of` must raise;
* sabotage — the query's own date bound is broken on purpose, so the future
  rows *do* come back from DuckDB, and the call must still fail rather than
  hand them over;
* forgetting — a reader called without `as_of` must not run at all;
* a new reader — every `read_*` in `finmgr.data` must take a required `as_of`,
  so the next store added (benchmarks, day 22) cannot skip the guard.
"""

from __future__ import annotations

import datetime as dt
import importlib
import inspect
import pkgutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import finmgr.data
from finmgr.data import pit
from finmgr.data.actions import merge_actions, read_actions
from finmgr.data.adjust import adjust, read_adjusted
from finmgr.data.fx import merge_fx, read_fx
from finmgr.data.pit import LATEST, LookAheadError
from finmgr.data.quarantine import _write_records, read_quarantine, record_path
from finmgr.data.store import read_bars, write_bars

D = dt.date
SESSIONS = list(pd.bdate_range("2024-03-01", "2024-03-14").date)
AS_OF = D(2024, 3, 7)
PAST = [day for day in SESSIONS if day <= AS_OF]
FUTURE = [day for day in SESSIONS if day > AS_OF]
PAST_DIVIDEND = D(2024, 3, 5)
FUTURE_DIVIDEND = D(2024, 3, 12)

#: Read functions that are deliberately not point-in-time, and why.
NOT_MARKET_DATA = {
    # A log of ingest runs, dated by when they ran; no feature reads it.
    "finmgr.data.manifest.read_manifest",
}


def bars(days: list[dt.date] = SESSIONS) -> pd.DataFrame:
    close = [20.0 + i for i in range(len(days))]
    return pd.DataFrame(
        {
            "ticker": "SAN.MC",
            "date": days,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1_000.0,
        }
    )


def actions(days: list[dt.date]) -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": "SAN.MC", "date": days, "action": "dividend", "value": [0.5] * len(days)}
    )


def records(days: list[dt.date]) -> pd.DataFrame:
    stamp = pd.Timestamp("2024-03-20", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": "SAN.MC",
            "date": days,
            "check": "high_below_low",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "severity": "error",
            "detail": "",
            "reason": "test",
            "quarantined_at": stamp,
            "released_at": pd.NaT,
        }
    )


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """One data directory with rows before and after AS_OF in every store."""
    write_bars(bars(), root=tmp_path / "bars" / "daily")
    merge_actions(actions([PAST_DIVIDEND, FUTURE_DIVIDEND]), root=tmp_path / "actions")
    merge_fx(pd.DataFrame({"pair": "EURUSD", "date": SESSIONS, "rate": 1.08}), root=tmp_path / "fx")
    _write_records(
        record_path("SAN.MC", root=tmp_path / "quarantine"), records([PAST[1], FUTURE[1]])
    )
    return tmp_path


#: Every reader, as `(name, call)`: `call(data_dir, **kwargs)` reads its store.
READERS: list[tuple[str, Callable[..., pd.DataFrame]]] = [
    ("read_bars", lambda d, **kw: read_bars("SAN.MC", root=d / "bars" / "daily", **kw)),
    ("read_actions", lambda d, **kw: read_actions("SAN.MC", root=d / "actions", **kw)),
    ("read_fx", lambda d, **kw: read_fx("EURUSD", root=d / "fx", **kw)),
    ("read_adjusted", lambda d, **kw: read_adjusted("SAN.MC", root=d, **kw)),
    ("read_quarantine", lambda d, **kw: read_quarantine("SAN.MC", root=d / "quarantine", **kw)),
]
#: The readers that take a `start`/`end` window.
WINDOWED = [reader for reader in READERS if reader[0] != "read_quarantine"]


def ids(readers: list[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in readers]


@pytest.mark.parametrize("read", [r for _, r in READERS], ids=ids(READERS))
def test_the_store_really_holds_future_rows(data: Path, read: Callable) -> None:
    """Without this the other tests could pass against a store with nothing to leak."""
    assert max(read(data, as_of=FUTURE[-1])["date"]) > AS_OF


@pytest.mark.parametrize("read", [r for _, r in READERS], ids=ids(READERS))
def test_a_read_returns_nothing_after_as_of(data: Path, read: Callable) -> None:
    result = read(data, as_of=AS_OF)

    assert not result.empty
    assert max(result["date"]) <= AS_OF


@pytest.mark.parametrize("read", [r for _, r in READERS], ids=ids(READERS))
def test_as_of_accepts_a_date_a_datetime_and_an_iso_string(data: Path, read: Callable) -> None:
    expected = read(data, as_of=AS_OF)
    for spelling in (dt.datetime(2024, 3, 7, 23, 59), "2024-03-07"):
        pd.testing.assert_frame_equal(read(data, as_of=spelling), expected)


@pytest.mark.parametrize("read", [r for _, r in WINDOWED], ids=ids(WINDOWED))
def test_an_end_after_as_of_fails(data: Path, read: Callable) -> None:
    with pytest.raises(LookAheadError, match="after as_of"):
        read(data, end=FUTURE[0], as_of=AS_OF)


@pytest.mark.parametrize("read", [r for _, r in WINDOWED], ids=ids(WINDOWED))
def test_a_start_after_as_of_fails(data: Path, read: Callable) -> None:
    with pytest.raises(LookAheadError, match="after as_of"):
        read(data, start=FUTURE[0], as_of=AS_OF)


@pytest.mark.parametrize("read", [r for _, r in READERS], ids=ids(READERS))
def test_a_leaking_query_fails_instead_of_returning_the_future(
    data: Path, read: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leak test proper.

    The bound every reader puts into its query is replaced with "no bound at
    all", so DuckDB hands the future rows back. A reader that trusted its own
    query would return them; each one must raise instead.
    """
    monkeypatch.setattr(pit, "bound_end", lambda *_, **__: D.max)

    with pytest.raises(LookAheadError, match=f"dated after as_of {AS_OF}"):
        read(data, as_of=AS_OF)


@pytest.mark.parametrize("read", [r for _, r in READERS], ids=ids(READERS))
def test_a_read_without_as_of_does_not_run(data: Path, read: Callable) -> None:
    with pytest.raises(TypeError, match="as_of"):
        read(data)
    with pytest.raises(TypeError, match="as_of"):
        read(data, as_of=None)


@pytest.mark.parametrize("read", [r for _, r in READERS], ids=ids(READERS))
def test_latest_reads_up_to_today(data: Path, read: Callable) -> None:
    pd.testing.assert_frame_equal(read(data, as_of=LATEST), read(data, as_of=dt.date.today()))


def test_a_future_dividend_does_not_move_past_adjusted_closes(data: Path) -> None:
    """Back-adjustment is where look-ahead hides best.

    A dividend that goes ex after AS_OF rescales every earlier close when the
    factors are built over the whole store. As of AS_OF it had not happened,
    so the series must be exactly the one built from what was known then.
    """
    as_known = adjust(bars(PAST), actions([PAST_DIVIDEND]))
    hindsight = adjust(bars(), actions([PAST_DIVIDEND, FUTURE_DIVIDEND]))
    hindsight = hindsight[hindsight["date"] <= AS_OF].reset_index(drop=True)

    result = read_adjusted("SAN.MC", as_of=AS_OF, root=data)

    pd.testing.assert_frame_equal(result, as_known)
    # And the trap is real: with hindsight the same dates adjust differently.
    assert not result["adj_close"].equals(hindsight["adj_close"])


def test_guard_passes_rows_on_as_of_and_refuses_one_after() -> None:
    frame = bars([AS_OF])
    assert pit.guard(frame, AS_OF, what="test") is frame

    with pytest.raises(LookAheadError, match="1 row"):
        pit.guard(bars([AS_OF, FUTURE[0]]), AS_OF, what="test")


def _readers_in_the_data_layer() -> list[tuple[str, Callable]]:
    found = []
    for info in pkgutil.iter_modules(finmgr.data.__path__):
        module = importlib.import_module(f"finmgr.data.{info.name}")
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("read_") and function.__module__ == module.__name__:
                found.append((f"{module.__name__}.{name}", function))
    return found


def test_the_reader_list_above_covers_the_data_layer() -> None:
    """A reader added without a row in READERS is a reader nobody tried to leak."""
    found = {qualified.rsplit(".", 1)[1] for qualified, _ in _readers_in_the_data_layer()}
    tested = set(ids(READERS)) | {name.rsplit(".", 1)[1] for name in NOT_MARKET_DATA}
    assert found == tested


@pytest.mark.parametrize(
    ("qualified", "function"),
    [reader for reader in _readers_in_the_data_layer() if reader[0] not in NOT_MARKET_DATA],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_reader_requires_as_of(qualified: str, function: Callable) -> None:
    parameter = inspect.signature(function).parameters.get("as_of")

    assert parameter is not None, f"{qualified} has no as_of"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty, f"{qualified} defaults as_of"
