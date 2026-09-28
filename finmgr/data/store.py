"""The bar store: Parquet on disk, DuckDB on top.

Every daily bar this project ever uses is written here and read back here, in
one fixed schema:

    ticker  str                   Yahoo's spelling, suffix included
    date    date32[day][pyarrow]  the session's calendar date, exchange-local
    open    float64
    high    float64
    low     float64
    close   float64
    volume  float64

The layout is one Parquet file per ticker, in a Hive-style partition:

    data/bars/daily/ticker=AAPL/part.parquet
    data/bars/daily/ticker=SAN.MC/part.parquet

which buys three things. A single ticker's fifteen years is one small file, so
re-ingesting it rewrites nothing else. DuckDB reads the whole set as one table
and skips the partitions a `WHERE ticker IN (...)` cannot match. And when
something looks wrong in eight weeks, the file to open is obvious from the
ticker alone.

    from finmgr.data.store import merge_bars, read_bars, write_bars

    write_bars(frame)                                   # frame in the schema above
    merge_bars(frame)                                   # ... or fold it into what is stored
    read_bars(["AAPL", "SAN.MC"], start="2024-01-01")   # back out again

The schema is fixed *now*, before day 10 downloads anything, because every
feature, label and backtest downstream assumes these column names and these
dtypes. :func:`conform_bars` is the one place that enforces it, and it runs on
both the write and the read path — so what comes out of :func:`read_bars` is
dtype-identical to what went into :func:`write_bars`, whatever the caller
happened to hand over.

`volume` is float64 rather than an integer on purpose: yfinance returns NaN
volume for some sessions on some venues, and an integer column cannot hold
that without inventing a zero that never traded.

Two functions exist so a re-run costs nothing and changes nothing.
:func:`max_stored_dates` says how far each ticker's history already reaches,
which is what day 12's ingest asks Yahoo *from*; :func:`merge_bars` folds a new
frame into a stored partition, keyed on `(ticker, date)`, so a session that is
already on disk is overwritten rather than appended and an unchanged partition
is not rewritten at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finmgr import runlog
from finmgr.config import Settings, load_settings

#: The columns of a bar frame, in order.
BAR_COLUMNS: tuple[str, ...] = ("ticker", "date", "open", "high", "low", "close", "volume")

#: What `(ticker, date)` identifies — a session — and what is compared when
#: deciding whether a bar already on disk has actually changed.
KEY_COLUMNS: tuple[str, ...] = ("ticker", "date")
VALUE_COLUMNS: tuple[str, ...] = tuple(c for c in BAR_COLUMNS if c not in KEY_COLUMNS)

#: The pandas dtype of every column. This is the contract `read_bars()` and
#: `write_bars()` both honour, and what "dtypes identical" means downstream.
BAR_DTYPES: dict[str, str] = {
    "ticker": "str",
    "date": "date32[day][pyarrow]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

#: The physical Parquet schema, stated rather than inferred. Handing this to
#: pyarrow means the file on disk has the same types whichever pandas dtype the
#: caller arrived with — and that a future pandas release changing its default
#: string or date representation cannot quietly rewrite the store.
BAR_ARROW_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
    ]
)

#: Where the daily bars live under `settings.data_dir`.
BARS_SUBPATH = ("bars", "daily")

#: The file inside each `ticker=<T>/` directory.
PART_FILENAME = "part.parquet"

#: zstd beats Parquet's default snappy by a comfortable margin on price
#: columns, and the store is read far more often than it is written.
COMPRESSION = "zstd"

log = runlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def bars_root(*, settings: Settings | None = None, root: Path | str | None = None) -> Path:
    """The root of the daily bar dataset: `<data_dir>/bars/daily`.

    `root` overrides it outright, which is how the tests stay inside
    `tmp_path` and out of the real `data/`.
    """
    if root is not None:
        return Path(root).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*BARS_SUBPATH)


def _check_ticker(ticker: str) -> str:
    """Refuse anything that would escape or confuse the partition layout.

    The ticker becomes a directory name, and it comes from a CSV a human
    edits. A stray slash would write outside the store; an empty string would
    collapse two tickers into one partition.
    """
    cleaned = str(ticker).strip()
    if not cleaned:
        raise ValueError("ticker is empty")
    if cleaned in {".", ".."} or any(bad in cleaned for bad in ("/", "\\", "\x00")):
        raise ValueError(f"ticker {ticker!r} is not usable as a directory name")
    return cleaned


def partition_path(
    ticker: str, *, settings: Settings | None = None, root: Path | str | None = None
) -> Path:
    """Where one ticker's bars live: `<root>/ticker=<T>/part.parquet`."""
    base = bars_root(settings=settings, root=root)
    return base / f"ticker={_check_ticker(ticker)}" / PART_FILENAME


def stored_tickers(
    *, settings: Settings | None = None, root: Path | str | None = None
) -> list[str]:
    """Every ticker that currently has a partition on disk, sorted."""
    return sorted(
        path.parent.name.removeprefix("ticker=") for path in _partition_files(settings, root)
    )


def _partition_files(
    settings: Settings | None,
    root: Path | str | None,
    tickers: Sequence[str] | None = None,
) -> list[Path]:
    """The Parquet files to read, narrowed to `tickers` when one is given.

    Selecting the paths here rather than globbing inside DuckDB is what makes
    a ticker filter free: an unasked-for partition is never opened.
    """
    base = bars_root(settings=settings, root=root)
    if tickers is None:
        return sorted(base.glob(f"ticker=*/{PART_FILENAME}"))
    wanted = [partition_path(ticker, settings=settings, root=root) for ticker in tickers]
    return [path for path in wanted if path.is_file()]


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def empty_bars() -> pd.DataFrame:
    """An empty frame carrying the full schema.

    What a read of an empty store returns, so callers can concatenate and
    check dtypes without special-casing "nothing stored yet".
    """
    return pd.DataFrame(
        {column: pd.Series([], dtype=dtype) for column, dtype in BAR_DTYPES.items()}
    )


def _conform_dates(values: pd.Series) -> pd.Series:
    """Cast one column to `date32[day]`, refusing timezone-aware input.

    A tz-aware column would convert through UTC on its way to a date, and a
    Tokyo close at 15:00 JST would land on the previous calendar day — the bar
    silently dated one session early, every session, forever. Day 10 hands
    over exchange-local dates deliberately; this is the check that keeps it
    honest.
    """
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        raise ValueError(
            "the 'date' column is timezone-aware; take the exchange-local calendar date "
            "first (frame['date'] = frame.index.date) so the session is not shifted by UTC"
        )
    return values.astype(BAR_DTYPES["date"])


def conform_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a frame against the schema and return it in canonical form.

    Canonical means: exactly :data:`BAR_COLUMNS`, in that order, with
    :data:`BAR_DTYPES`, sorted by `(ticker, date)` and a fresh RangeIndex.

    Raises on anything structurally wrong — a missing or unexpected column, a
    missing ticker or date, a repeated `(ticker, date)`. It does *not* judge
    the prices: a negative close or a high below its low is a data quality
    problem, and day 19 is where those get flagged and day 20 quarantines
    them. Dropping them here would lose the evidence.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")

    present = list(frame.columns)
    missing = [column for column in BAR_COLUMNS if column not in present]
    unexpected = [column for column in present if column not in BAR_COLUMNS]
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unexpected:
            detail.append(f"unexpected {unexpected}")
        raise ValueError(
            f"bar frame columns: {', and '.join(detail)}; expected {list(BAR_COLUMNS)}"
        )
    if len(present) != len(set(present)):
        raise ValueError(f"bar frame has duplicate columns: {present}")

    if frame.empty:
        return empty_bars()

    out = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    for column in BAR_COLUMNS:
        values = frame[column].reset_index(drop=True)
        if column == "date":
            out[column] = _conform_dates(values)
        else:
            out[column] = values.astype(BAR_DTYPES[column])

    if out["date"].isna().any():
        raise ValueError(f"{int(out['date'].isna().sum())} row(s) have no date")
    blank = out["ticker"].isna() | (out["ticker"].str.strip() == "")
    if blank.any():
        raise ValueError(f"{int(blank.sum())} row(s) have an empty ticker")
    out["ticker"] = out["ticker"].str.strip()
    for ticker in out["ticker"].unique():
        _check_ticker(ticker)

    duplicated = out.duplicated(subset=["ticker", "date"], keep=False)
    if duplicated.any():
        sample = out.loc[duplicated, ["ticker", "date"]].drop_duplicates().head(5)
        pairs = ", ".join(f"{row.ticker}@{row.date}" for row in sample.itertuples())
        raise ValueError(
            f"{int(duplicated.sum())} row(s) repeat a (ticker, date) pair, e.g. {pairs}"
        )

    return out.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_bars(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> list[Path]:
    """Write a bar frame to the store, one Parquet file per ticker.

    The frame may cover any number of tickers; each one's partition is
    replaced whole. Returns the files written, in ticker order.

    Replacing rather than appending is what makes a re-ingest safe to run
    twice: the partition on disk is always exactly what the caller last handed
    over for that ticker. Merging new bars into the existing history — reading
    what is stored, adding only what is newer — is day 12's job, and it will
    build on this call rather than replace it.
    """
    conformed = conform_bars(frame)
    if conformed.empty:
        log.debug("write_bars: nothing to write")
        return []

    written: list[Path] = []
    for ticker, group in conformed.groupby("ticker", sort=True):
        path = partition_path(str(ticker), settings=settings, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(
            group.reset_index(drop=True), schema=BAR_ARROW_SCHEMA, preserve_index=False
        )
        # Write beside the target and rename over it. os.replace is atomic on
        # the same filesystem, so an interrupted run — and day 11 assumes runs
        # get interrupted — leaves the previous partition intact rather than a
        # half-written file that reads as corrupt.
        staged = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, staged, compression=COMPRESSION)
        staged.replace(path)
        written.append(path)
        log.debug("wrote %d bars for %s to %s", len(group), ticker, path)

    log.info("wrote %d bars across %d ticker(s)", len(conformed), len(written))
    return written


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def as_date(value: date | datetime | str | None, label: str) -> date | None:
    """Accept a date, a datetime, an ISO string or None.

    Public because every caller that bounds a window — `read_bars` here,
    `fetch_daily` on day 10 — has to read the same four spellings of a date
    the same way.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"{label} must be a date, datetime or ISO string, got {type(value).__name__}")


def read_bars(
    tickers: Iterable[str] | str | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Read bars out of the store, sorted by `(ticker, date)`.

    `tickers` is one symbol, several, or None for everything stored. `start`
    and `end` are inclusive bounds on the session date. The result always
    carries the full schema — an empty store, or a filter that matches
    nothing, returns an empty frame with the right dtypes rather than
    something the caller has to test for.

    The read goes through DuckDB rather than `pd.read_parquet` because the
    filtering belongs next to the data: DuckDB pushes the date bound into the
    Parquet row groups and never materialises the rows this call did not ask
    for. Fifteen years of 100 tickers is small enough to fit in memory, and
    day 29 will want exactly that — but most reads are one slice of it.
    """
    if isinstance(tickers, str):
        requested: list[str] | None = [tickers]
    elif tickers is None:
        requested = None
    else:
        requested = [str(ticker) for ticker in tickers]
        if not requested:
            return empty_bars()

    files = _partition_files(settings, root, requested)
    if not files:
        return empty_bars()

    first = as_date(start, "start")
    last = as_date(end, "end")
    if first is not None and last is not None and first > last:
        raise ValueError(f"start {first} is after end {last}")

    columns = ", ".join(f'"{column}"' for column in BAR_COLUMNS)
    # hive_partitioning reads `ticker` back out of the directory name. The
    # column is stored inside the file too — a partition is readable on its own
    # that way — and DuckDB prefers the file's copy when both exist.
    sql = [
        f"SELECT {columns}",
        "FROM read_parquet(?, hive_partitioning = true)",
    ]
    params: list[Any] = [[str(path) for path in files]]
    where: list[str] = []
    if first is not None:
        where.append('"date" >= ?')
        params.append(first)
    if last is not None:
        where.append('"date" <= ?')
        params.append(last)
    if where:
        sql.append("WHERE " + " AND ".join(where))
    sql.append('ORDER BY "ticker", "date"')

    with duckdb.connect() as con:
        table = con.execute("\n".join(sql), params).to_arrow_table()

    # to_pandas() hands back dates as objects; conform_bars is what puts the
    # declared dtypes back, and is the same call the write path went through.
    return conform_bars(table.to_pandas())


def max_stored_dates(
    tickers: Iterable[str] | str | None = None,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> dict[str, date]:
    """The newest stored session per ticker: `{"AAPL": date(2026, 9, 25), ...}`.

    A ticker with no partition, or a partition holding no rows, is simply
    absent from the mapping — there is no "never" date to invent, and a caller
    asking "what do I already have?" wants to tell those apart from a stored
    history that happens to be short.

    This is where an incremental ingest starts: day 12 asks Yahoo from here
    forward instead of re-requesting fifteen years every evening. It reads the
    max out of DuckDB rather than loading the bars because Parquet keeps
    per-row-group statistics, so the answer comes from the file's metadata and
    the price columns are never touched.
    """
    if isinstance(tickers, str):
        requested: list[str] | None = [tickers]
    elif tickers is None:
        requested = None
    else:
        requested = [str(ticker) for ticker in tickers]
        if not requested:
            return {}

    files = _partition_files(settings, root, requested)
    if not files:
        return {}

    sql = (
        'SELECT "ticker", max("date") AS last_date '
        "FROM read_parquet(?, hive_partitioning = true) "
        'GROUP BY "ticker"'
    )
    with duckdb.connect() as con:
        rows = con.execute(sql, [[str(path) for path in files]]).fetchall()

    return {str(ticker): last for ticker, last in rows if last is not None}


#: The columns of a :func:`coverage` frame, in order.
COVERAGE_COLUMNS: tuple[str, ...] = ("ticker", "first_date", "last_date", "rows")


def coverage(
    tickers: Iterable[str] | str | None = None,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """First date, last date and row count per ticker, in the order asked.

    `tickers` None means everything stored, sorted. A ticker that was asked
    about and has nothing on disk is *kept*, with zero rows and no dates —
    after a backfill the question is "what is missing?", and a coverage table
    that silently omits the gaps cannot answer it.

    Like :func:`max_stored_dates` this is one aggregate query; the price
    columns are never loaded.
    """
    if isinstance(tickers, str):
        requested: list[str] | None = [tickers]
    elif tickers is None:
        requested = None
    else:
        requested = [str(ticker).strip() for ticker in tickers]

    files = _partition_files(settings, root, requested)
    found: dict[str, tuple[date, date, int]] = {}
    if files:
        sql = (
            'SELECT "ticker", min("date"), max("date"), count(*) '
            "FROM read_parquet(?, hive_partitioning = true) "
            'GROUP BY "ticker"'
        )
        with duckdb.connect() as con:
            rows = con.execute(sql, [[str(path) for path in files]]).fetchall()
        found = {str(ticker): (first, last, int(n)) for ticker, first, last, n in rows}

    order = sorted(found) if requested is None else requested
    records = [(ticker, *found.get(ticker, (None, None, 0))) for ticker in order]
    frame = pd.DataFrame(records, columns=list(COVERAGE_COLUMNS))
    for column in ("first_date", "last_date"):
        frame[column] = frame[column].astype(BAR_DTYPES["date"])
    frame["rows"] = frame["rows"].astype("int64")
    frame["ticker"] = frame["ticker"].astype(BAR_DTYPES["ticker"])
    return frame


def max_stored_date(
    ticker: str,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> date | None:
    """The newest stored session for one ticker, or None if nothing is stored."""
    return max_stored_dates([ticker], settings=settings, root=root).get(str(ticker).strip())


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MergeResult:
    """What :func:`merge_bars` did to one ticker's partition.

    The three counts are the whole point: `added` is the number the acceptance
    criterion of day 12 cares about — a second identical run has to report
    zero — while `updated` says Yahoo revised a session that was already
    stored, and `unchanged` says it sent the same bar back.
    """

    ticker: str
    #: Sessions that were not in the store before.
    added: int = 0
    #: Sessions that were stored and came back with different values.
    updated: int = 0
    #: Sessions that were stored and came back identical.
    unchanged: int = 0
    #: Rows in the partition after the merge.
    stored: int = 0
    #: The file written, or None when nothing changed and nothing was written.
    path: Path | None = None

    @property
    def changed(self) -> bool:
        """True when the partition on disk was rewritten."""
        return self.path is not None


def _same_values(joined: pd.DataFrame, left: str, right: str) -> pd.Series:
    """Row-wise equality over the price columns of a suffixed join.

    NaN counts as equal to NaN here. `NaN != NaN` is right for arithmetic and
    wrong for this question: a session Yahoo reports with no volume, re-fetched
    and still reported with no volume, has not changed, and calling it a
    revision would rewrite the partition on every single run.
    """
    same = pd.Series(True, index=joined.index)
    for column in VALUE_COLUMNS:
        this, that = joined[column + left], joined[column + right]
        same &= (this == that) | (this.isna() & that.isna())
    return same


def _merge_one(
    ticker: str,
    incoming: pd.DataFrame,
    *,
    settings: Settings | None,
    root: Path | str | None,
) -> MergeResult:
    """Fold one ticker's new bars into its stored partition."""
    existing = read_bars(ticker, settings=settings, root=root)
    if existing.empty:
        (path,) = write_bars(incoming, settings=settings, root=root)
        return MergeResult(ticker, added=len(incoming), stored=len(incoming), path=path)

    is_new = ~incoming["date"].isin(existing["date"])
    added = int(is_new.sum())

    shared = incoming.loc[~is_new]
    updated = unchanged = 0
    if not shared.empty:
        # Joined on the key rather than aligned on an index: the two frames
        # arrive sorted but nothing guarantees the same rows in the same
        # places, and a silent misalignment here would compare one session's
        # close against another's.
        joined = shared.merge(existing, on="date", suffixes=("_new", "_old"))
        unchanged = int(_same_values(joined, "_new", "_old").sum())
        updated = len(joined) - unchanged

    if not added and not updated:
        # Nothing to say that the file does not already say. Leaving it alone
        # keeps a re-run free and, more usefully, keeps the partition's mtime
        # honest: it means "this is when the data last changed".
        log.debug("%s: already current, %d row(s) stored", ticker, len(existing))
        return MergeResult(ticker, unchanged=unchanged, stored=len(existing))

    # De-duplication on (ticker, date): every stored session the new frame also
    # carries is dropped in favour of the new one, so a re-run overwrites where
    # it overlaps instead of appending a second copy. conform_bars inside
    # write_bars re-checks that, and would refuse the frame if a duplicate
    # survived.
    kept = existing.loc[~existing["date"].isin(incoming["date"])]
    combined = pd.concat([kept, incoming], ignore_index=True)
    (path,) = write_bars(combined, settings=settings, root=root)
    log.debug(
        "%s: +%d new, %d updated, %d unchanged, %d stored",
        ticker,
        added,
        updated,
        unchanged,
        len(combined),
    )
    return MergeResult(
        ticker,
        added=added,
        updated=updated,
        unchanged=unchanged,
        stored=len(combined),
        path=path,
    )


def merge_bars(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> list[MergeResult]:
    """Fold a bar frame into the store, de-duplicating on `(ticker, date)`.

    Where :func:`write_bars` replaces a ticker's partition with what it was
    handed, this keeps the history and merges: sessions the store has never
    seen are added, sessions it already holds are *overwritten* by the new
    values, and stored sessions the frame says nothing about are left alone.
    Re-running an ingest therefore converges instead of growing — the same bars
    fetched twice produce one copy, not two.

    Returns one :class:`MergeResult` per ticker in the frame, in ticker order.
    A partition where nothing was added or updated is not rewritten at all, and
    its result carries `path=None`.
    """
    incoming = conform_bars(frame)
    if incoming.empty:
        log.debug("merge_bars: nothing to merge")
        return []

    results = [
        _merge_one(str(ticker), group.reset_index(drop=True), settings=settings, root=root)
        for ticker, group in incoming.groupby("ticker", sort=True)
    ]
    log.info(
        "merged %d bar(s) across %d ticker(s): %d new, %d updated, %d unchanged",
        len(incoming),
        len(results),
        sum(result.added for result in results),
        sum(result.updated for result in results),
        sum(result.unchanged for result in results),
    )
    return results
