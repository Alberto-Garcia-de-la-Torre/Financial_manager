"""Corporate actions: splits and dividends, stored apart from the bars.

    from finmgr.data.actions import ingest_actions, read_actions

    ingest_actions(["AAPL"])                        # fetch, merge into data/actions/
    read_actions("AAPL", action="split")            # back out of the store

A bar says what a share traded at. An action says something happened *to* the
share — it was cut into four, or it paid out €0.10 — and that is a different
kind of fact with a different life: a few rows per ticker per year rather than
one per session, and the raw material day 16 builds the adjusted series from.
So it gets its own store, in one long schema:

    ticker  str                   Yahoo's spelling, suffix included
    date    date32[day][pyarrow]  the ex-date, exchange-local
    action  str                   "split" or "dividend"
    value   float64               split: new shares per old (4.0 is 4:1,
                                  0.1 is 1:10); dividend: cash per share, in
                                  the listing currency

laid out exactly like the bars, one Parquet file per ticker:

    data/actions/ticker=AAPL/part.parquet

The key is `(ticker, date, action)` rather than `(ticker, date)`: a split and a
dividend can share an ex-date, and both have to survive.

**Where they come from.** The same Yahoo call the bars use —
:func:`~finmgr.data.fetch.download_history` — returns `Dividends` and
`Stock Splits` columns that :func:`~finmgr.data.fetch.to_bars` deliberately
drops. :func:`to_actions` is the other half: it keeps only those two columns,
only the rows where they are non-zero, and the same exchange-local date. Values
are stored exactly as Yahoo sent them. In particular Yahoo quotes historical
dividends in *today's* share count — AAPL's August 2020 dividend arrives as
$0.205, not the $0.82 declared before the 4:1 split — which is the same
post-split money its raw bars are in. Day 16 has to know that; this module
does not correct it, because correcting it would be the first adjustment and
adjustments are day 16's.

**Merging, never replacing.** Every run asks for the whole window, because
actions are sparse and an incremental window would mostly be empty. What comes
back is merged on the key: a new action is added, a stored one Yahoo now
reports differently is overwritten, and a stored one Yahoo has stopped
reporting is *kept* — a fact on disk is not deleted because the source forgot
it. An unchanged partition is not rewritten.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data import fetch as fetch_module
from finmgr.data.ingest import (
    DEFAULT_ABORT_AFTER,
    DEFAULT_ATTEMPTS,
    DEFAULT_BACKOFF,
    DEFAULT_PAUSE,
    Sleeper,
    fetch_with_retries,
    resolve_window,
)
from finmgr.data.store import COMPRESSION, PART_FILENAME, _check_ticker, as_date
from finmgr.data.universe import Company, load_universe

#: The columns of an action frame, in order.
ACTION_COLUMNS: tuple[str, ...] = ("ticker", "date", "action", "value")

#: What identifies one action. Not `(ticker, date)` alone — see the docstring.
ACTION_KEY: tuple[str, ...] = ("ticker", "date", "action")

ACTION_DTYPES: dict[str, str] = {
    "ticker": "str",
    "date": "date32[day][pyarrow]",
    "action": "str",
    "value": "float64",
}

ACTION_ARROW_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
)

SPLIT = "split"
DIVIDEND = "dividend"

#: Yahoo's column, lowercased, for each kind of action.
SOURCE_COLUMNS: dict[str, str] = {SPLIT: "stock splits", DIVIDEND: "dividends"}

#: Where the actions live under `settings.data_dir`.
ACTIONS_SUBPATH = ("actions",)

#: A downloader with :func:`~finmgr.data.fetch.fetch_daily`'s inclusive bounds,
#: returning an action frame. Swapped out by the tests.
ActionFetcher = Callable[[str, date, date], pd.DataFrame]

log = runlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def actions_root(*, settings: Settings | None = None, root: Path | str | None = None) -> Path:
    """The root of the action dataset: `<data_dir>/actions`."""
    if root is not None:
        return Path(root).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*ACTIONS_SUBPATH)


def partition_path(
    ticker: str, *, settings: Settings | None = None, root: Path | str | None = None
) -> Path:
    """Where one ticker's actions live: `<root>/ticker=<T>/part.parquet`."""
    base = actions_root(settings=settings, root=root)
    return base / f"ticker={_check_ticker(ticker)}" / PART_FILENAME


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def empty_actions() -> pd.DataFrame:
    """An empty frame carrying the full schema."""
    return pd.DataFrame(
        {column: pd.Series([], dtype=dtype) for column, dtype in ACTION_DTYPES.items()}
    )


def conform_actions(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an action frame and return it sorted by `(ticker, date, action)`.

    Refuses a missing or unexpected column, an unknown action, a timezone-aware
    date, a missing or non-positive value and a repeated key. A zero is
    refused rather than dropped: Yahoo's "nothing happened" is filtered out by
    :func:`to_actions`, so a zero arriving here is a bug upstream.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    present = list(frame.columns)
    if sorted(present) != sorted(ACTION_COLUMNS):
        raise ValueError(f"action frame columns {present}; expected {list(ACTION_COLUMNS)}")
    if frame.empty:
        return empty_actions()

    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        raise ValueError("the 'date' column is timezone-aware; take the exchange-local date first")
    out = pd.DataFrame(
        {
            column: frame[column].reset_index(drop=True).astype(dtype)
            for column, dtype in ACTION_DTYPES.items()
        }
    )
    out["ticker"] = out["ticker"].str.strip()
    for ticker in out["ticker"].unique():
        _check_ticker(ticker)
    if out["date"].isna().any():
        raise ValueError(f"{int(out['date'].isna().sum())} action(s) have no date")

    unknown = sorted(set(out["action"]) - set(SOURCE_COLUMNS))
    if unknown:
        raise ValueError(f"unknown action(s) {unknown}; expected one of {sorted(SOURCE_COLUMNS)}")
    bad = out["value"].isna() | (out["value"] <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} action(s) have a missing or non-positive value")

    repeated = out.duplicated(subset=list(ACTION_KEY), keep=False)
    if repeated.any():
        sample = out.loc[repeated, list(ACTION_KEY)].drop_duplicates().head(5)
        keys = ", ".join(f"{r.ticker}@{r.date}:{r.action}" for r in sample.itertuples())
        raise ValueError(f"{int(repeated.sum())} row(s) repeat an action, e.g. {keys}")

    return out.sort_values(list(ACTION_KEY), kind="stable").reset_index(drop=True)


def describe(action: str, value: float) -> str:
    """How a human writes it: `4:1` for a split, `0.24` for a dividend."""
    if action == SPLIT:
        if value >= 1:
            return f"{value:g}:1"
        return f"1:{1 / value:g}"
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Response -> schema
# ---------------------------------------------------------------------------


def to_actions(ticker: str, frame: Any) -> pd.DataFrame:
    """Pull the splits and dividends out of one yfinance history response.

    Pure: the counterpart of :func:`~finmgr.data.fetch.to_bars` over the same
    frame. Yahoo carries an action as a non-zero cell on the session it took
    effect — its ex-date — so every zero and NaN is "nothing happened" and is
    dropped. A response with neither column has no actions to give, which is
    an empty frame, not an error: some venues never send them.
    """
    if frame is None or len(frame) == 0:
        return empty_actions()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame from yfinance, got {type(frame).__name__}")

    lowered = {str(column).strip().lower(): column for column in frame.columns}
    dates = fetch_module._session_dates(frame)
    pieces = []
    for action, source in SOURCE_COLUMNS.items():
        if source not in lowered:
            continue
        values = pd.Series(frame[lowered[source]].to_numpy(dtype="float64"))
        happened = values.notna() & (values != 0)
        pieces.append(
            pd.DataFrame(
                {
                    "ticker": str(ticker).strip(),
                    "date": dates[happened].to_numpy(),
                    "action": action,
                    "value": values[happened].to_numpy(),
                }
            )
        )
    if not pieces:
        return empty_actions()
    found = pd.concat(pieces, ignore_index=True)
    # Yahoo's repeated-last-session quirk (see to_bars) can repeat an action
    # too. Same remedy: keep the later row.
    found = found.drop_duplicates(subset=list(ACTION_KEY), keep="last")
    return conform_actions(found)


def fetch_actions(
    ticker: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    settings: Settings | None = None,
    download: fetch_module.Downloader | None = None,
) -> pd.DataFrame:
    """Fetch one ticker's splits and dividends between two inclusive dates.

    Defaults mirror :func:`~finmgr.data.fetch.fetch_daily`: `history_start`
    to today. Raises if the request fails; retries are the caller's.
    """
    symbol = str(ticker).strip()
    first, last = resolve_window(start, end, settings)
    # Looked up at call time, so a test that replaces download_history on the
    # fetch module takes the network away from this path too.
    downloader = fetch_module.download_history if download is None else download
    raw = downloader(symbol, first, last + timedelta(days=1))
    found = to_actions(symbol, raw)
    found = found.loc[found["date"].between(first, last)].reset_index(drop=True)
    log.info("%s: %d action(s) between %s and %s", symbol, len(found), first, last)
    return found


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionMerge:
    """What :func:`merge_actions` did to one ticker's partition."""

    ticker: str
    added: int = 0
    updated: int = 0
    stored: int = 0
    path: Path | None = None


def merge_actions(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> list[ActionMerge]:
    """Fold an action frame into the store, keyed on `(ticker, date, action)`.

    New actions are added, stored ones the frame revises are overwritten, and
    stored ones the frame does not mention are kept. A partition with nothing
    added or updated is not rewritten (`path` is None).
    """
    incoming = conform_actions(frame)
    results = []
    for ticker, group in incoming.groupby("ticker", sort=True):
        symbol = str(ticker)
        existing = read_actions(symbol, settings=settings, root=root)
        joined = group.merge(
            existing, on=list(ACTION_KEY), how="left", suffixes=("", "_old"), indicator=True
        )
        added = int((joined["_merge"] == "left_only").sum())
        updated = int(
            ((joined["_merge"] == "both") & (joined["value"] != joined["value_old"])).sum()
        )
        if not added and not updated:
            results.append(ActionMerge(symbol, stored=len(existing)))
            continue

        kept = existing.merge(
            group[list(ACTION_KEY)], on=list(ACTION_KEY), how="left", indicator=True
        )
        kept = kept.loc[kept["_merge"] == "left_only", list(ACTION_COLUMNS)]
        combined = conform_actions(pd.concat([kept, group], ignore_index=True))

        path = partition_path(symbol, settings=settings, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(combined, schema=ACTION_ARROW_SCHEMA, preserve_index=False)
        # Staged and renamed, as for the bars, so an interrupted run never
        # leaves a half-written partition behind.
        staged = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, staged, compression=COMPRESSION)
        staged.replace(path)
        results.append(ActionMerge(symbol, added, updated, len(combined), path))
    return results


def read_actions(
    tickers: Iterable[str] | str | None = None,
    *,
    action: str | None = None,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Read actions out of the store, sorted by `(ticker, date, action)`.

    `tickers` is one symbol, several, or None for everything stored; `action`
    narrows to `"split"` or `"dividend"`; `start` and `end` are inclusive.
    Always returns the full schema, empty when nothing matches.
    """
    if action is not None and action not in SOURCE_COLUMNS:
        raise ValueError(f"unknown action {action!r}; expected one of {sorted(SOURCE_COLUMNS)}")
    base = actions_root(settings=settings, root=root)
    if tickers is None:
        files = sorted(base.glob(f"ticker=*/{PART_FILENAME}"))
    else:
        wanted = [tickers] if isinstance(tickers, str) else list(tickers)
        files = [
            path
            for path in (partition_path(t, settings=settings, root=root) for t in wanted)
            if path.is_file()
        ]
    if not files:
        return empty_actions()

    columns = ", ".join(f'"{column}"' for column in ACTION_COLUMNS)
    sql = f"SELECT {columns} FROM read_parquet(?, hive_partitioning = true)"
    params: list[Any] = [[str(path) for path in files]]
    where = []
    for clause, value in (
        ('"action" = ?', action),
        ('"date" >= ?', as_date(start, "start")),
        ('"date" <= ?', as_date(end, "end")),
    ):
        if value is not None:
            where.append(clause)
            params.append(value)
    if where:
        sql += " WHERE " + " AND ".join(where)
    with duckdb.connect() as con:
        table = con.execute(sql, params).to_arrow_table()
    return conform_actions(table.to_pandas())


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionIngest:
    """One ticker's outcome. `error` is empty when it worked."""

    ticker: str
    splits: int = 0
    dividends: int = 0
    added: int = 0
    updated: int = 0
    attempts: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def ingest_actions(
    tickers: Iterable[Company | str] | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    pause: float = DEFAULT_PAUSE,
    abort_after: int = DEFAULT_ABORT_AFTER,
    settings: Settings | None = None,
    root: Path | str | None = None,
    fetch: ActionFetcher | None = None,
    sleep: Sleeper | None = None,
) -> list[ActionIngest]:
    """Fetch and merge every ticker's actions; one result row each, never raises.

    The same discipline as the bar ingest: a pause between tickers, retries
    with backoff through :func:`~finmgr.data.ingest.fetch_with_retries`, a
    try/except per ticker, and an early stop after `abort_after` consecutive
    failures — the remaining tickers come back with an error saying so.
    """
    sleeper = time.sleep if sleep is None else sleep
    symbols = [
        company.ticker if isinstance(company, Company) else str(company).strip()
        for company in (load_universe(settings=settings) if tickers is None else tickers)
    ]
    first, last = resolve_window(start, end, settings)

    def fetch_one(symbol: str, window_start: date, window_end: date) -> pd.DataFrame:
        if fetch is not None:
            return fetch(symbol, window_start, window_end)
        return fetch_actions(symbol, window_start, window_end, settings=settings)

    results: list[ActionIngest] = []
    consecutive = 0
    for position, symbol in enumerate(symbols):
        if abort_after and consecutive >= abort_after:
            results.append(ActionIngest(symbol, error=f"skipped: {consecutive} failures in a row"))
            continue
        if position and pause:
            sleeper(pause)
        used = 0
        try:
            found, used = fetch_with_retries(
                symbol,
                first,
                last,
                attempts=attempts,
                backoff=backoff,
                fetch=fetch_one,
                sleep=sleep,
            )
            merged = merge_actions(found, settings=settings, root=root)
        except Exception as exc:
            consecutive += 1
            detail = f"{type(exc).__name__}: {exc}"
            log.warning("%s: actions not ingested: %s", symbol, detail)
            results.append(ActionIngest(symbol, attempts=used or attempts, error=detail))
            continue
        consecutive = 0
        counts = found["action"].value_counts()
        result = ActionIngest(
            symbol,
            splits=int(counts.get(SPLIT, 0)),
            dividends=int(counts.get(DIVIDEND, 0)),
            added=sum(m.added for m in merged),
            updated=sum(m.updated for m in merged),
            attempts=used,
        )
        log.info(
            "%s [%d/%d]: %d split(s), %d dividend(s), %d new",
            symbol,
            position + 1,
            len(symbols),
            result.splits,
            result.dividends,
            result.added,
        )
        results.append(result)
    return results
