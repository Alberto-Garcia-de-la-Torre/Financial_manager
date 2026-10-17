"""Quarantine: flag bad rows out of every read, and never delete them.

    from finmgr.data.quarantine import apply_quarantine, read_quarantine

    apply_quarantine()                  # run the checks, quarantine the bad-data rows
    read_quarantine("EQNR.OL", as_of=LATEST)   # what is quarantined, and why

A row the triage register (`config/quality_triage.csv`, day 19) calls
`bad-data` gets two things:

* its `quarantined` column in the bar store is set, so
  :func:`~finmgr.data.store.read_bars` leaves it out of every read that does
  not ask for it with `include_quarantined=True`;
* a record in `data/quarantine/ticker=<T>/part.parquet` holding the row as it
  stood when it was quarantined, the check that flagged it, the flag's detail,
  and the register's reason.

The bar itself stays in its partition. Silently dropping bad data is how you
end up debugging a model that was never the problem: a hole in a return series
with nothing on disk to say why. Here the bar, the flag and the reason are all
still there to look at.

Only `bad-data` verdicts quarantine. A flag nobody has triaged yet already
fails `finmgr check`, which is the louder signal; quarantining it unseen would
hide exactly the row a human still has to look at. `genuine` and `source-gap`
rows are right, or are not rows at all.

The quarantine is *synchronised* with the register, not accumulated. Each run
re-reads the checks: a row newly called `bad-data` is quarantined, and a row
that no longer is — its rule was removed, or Yahoo re-sent a corrected bar that
no longer trips the check — is released. Released records are kept, with a
`released_at` stamp, so the quarantine directory is itself never deleted from.
A second run with nothing changed writes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data import pit, store
from finmgr.data.quality import Flag, QualityReport, Verdict, run_checks

#: Where the quarantine records live under `settings.data_dir`.
QUARANTINE_SUBPATH = ("quarantine",)

#: The bar values copied into each record: the row as it was quarantined.
ROW_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: The columns of a quarantine record, in order. `(ticker, date, check)` is
#: the key: one bar can be quarantined by more than one check.
QUARANTINE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "check",
    *ROW_COLUMNS,
    "severity",
    "detail",
    "reason",
    "quarantined_at",
    "released_at",
)

QUARANTINE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("check", pa.string(), nullable=False),
        *(pa.field(column, pa.float64()) for column in ROW_COLUMNS),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("detail", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("quarantined_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("released_at", pa.timestamp("us", tz="UTC")),
    ]
)

KEY: tuple[str, ...] = ("date", "check")

log = runlog.get_logger(__name__)


def quarantine_root(*, settings: Settings | None = None, root: Path | str | None = None) -> Path:
    """`<data_dir>/quarantine`, or `root` outright (the tests' `tmp_path`)."""
    if root is not None:
        return Path(root).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*QUARANTINE_SUBPATH)


def record_path(
    ticker: str, *, settings: Settings | None = None, root: Path | str | None = None
) -> Path:
    """One ticker's records: `<root>/ticker=<T>/part.parquet`, as in the bar store."""
    base = quarantine_root(settings=settings, root=root)
    return base / f"ticker={store._check_ticker(ticker)}" / store.PART_FILENAME


def empty_records() -> pd.DataFrame:
    return QUARANTINE_ARROW_SCHEMA.empty_table().to_pandas()


def read_quarantine(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    include_released: bool = False,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Quarantine records, sorted by `(ticker, date, check)`.

    By default only the active ones — the rows :func:`~finmgr.data.store.read_bars`
    is currently hiding. `include_released` adds the history. A record carries
    the bar's prices, so like every read it stops at `as_of`, by bar date.
    """
    cutoff = pit.resolve_as_of(as_of)
    if isinstance(tickers, str):
        tickers = [tickers]
    if tickers is None:
        base = quarantine_root(settings=settings, root=root)
        files = sorted(base.glob(f"ticker=*/{store.PART_FILENAME}"))
    else:
        files = [record_path(t, settings=settings, root=root) for t in tickers]
        files = [path for path in files if path.is_file()]
    if not files:
        return empty_records()

    sql = 'SELECT * FROM read_parquet(?, hive_partitioning = true) WHERE "date" <= ?'
    if not include_released:
        sql += " AND released_at IS NULL"
    sql += ' ORDER BY "ticker", "date", "check"'
    with duckdb.connect() as con:
        bound = pit.bound_end(None, cutoff)
        table = con.execute(sql, [[str(path) for path in files], bound]).to_arrow_table()
    frame = table.select(list(QUARANTINE_COLUMNS)).cast(QUARANTINE_ARROW_SCHEMA).to_pandas()
    return pit.guard(frame, cutoff, what="read_quarantine")


def _write_records(path: Path, records: pd.DataFrame) -> None:
    """Atomically replace one ticker's record file, as the bar store does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = records.sort_values(["date", "check", "quarantined_at"], kind="stable")
    table = pa.Table.from_pandas(
        ordered[list(QUARANTINE_COLUMNS)].reset_index(drop=True),
        schema=QUARANTINE_ARROW_SCHEMA,
        preserve_index=False,
    )
    staged = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, staged, compression=store.COMPRESSION)
    staged.replace(path)


@dataclass(frozen=True, slots=True)
class QuarantineResult:
    """What one run of :func:`apply_quarantine` did to one ticker."""

    ticker: str
    #: Bars hidden from reads after this run.
    quarantined: int = 0
    #: Records opened this run.
    added: int = 0
    #: Records closed this run: the row is back in reads.
    released: int = 0
    #: The bar partition, when its flag column was rewritten.
    path: Path | None = None

    @property
    def changed(self) -> bool:
        return bool(self.added or self.released)


def apply_quarantine(
    report: QualityReport | None = None,
    *,
    tickers: Iterable[str] | None = None,
    settings: Settings | None = None,
    root: Path | str | None = None,
    quarantine_dir: Path | str | None = None,
    now: datetime | None = None,
    **check_options,
) -> list[QuarantineResult]:
    """Bring the store's `quarantined` flags and the records in line with the checks.

    `report` is a :class:`~finmgr.data.quality.QualityReport`; without one,
    :func:`~finmgr.data.quality.run_checks` runs over `tickers` (None: the
    universe) with `check_options` passed through. Only the tickers the report
    covers are touched. `root` is the bar store, `quarantine_dir` the records.
    Returns one result per ticker checked, in report order.
    """
    settings = settings or load_settings()
    if report is None:
        report = run_checks(tickers, settings=settings, root=root, **check_options)
    stamp = pd.Timestamp(now or datetime.now(UTC)).tz_convert("UTC")

    wanted: dict[str, dict[tuple[date, str], Flag]] = {}
    for flag in report.flags:
        if flag.triage is not None and flag.triage.verdict is Verdict.BAD_DATA:
            wanted.setdefault(flag.ticker, {})[(flag.date, flag.check)] = flag

    results = []
    for ticker in report.tickers:
        bars = store.read_bars(
            ticker, as_of=pit.LATEST, include_quarantined=True, settings=settings, root=root
        )
        path = record_path(ticker, settings=settings, root=quarantine_dir)
        records = read_quarantine(
            ticker,
            as_of=pit.LATEST,
            include_released=True,
            settings=settings,
            root=quarantine_dir,
        )
        rows = bars.set_index("date")
        # A flag with no stored bar behind it — a missing session — has
        # nothing to quarantine.
        flags = {key: flag for key, flag in wanted.get(ticker, {}).items() if key[0] in rows.index}

        active = records["released_at"].isna()
        open_keys = set(zip(records.loc[active, "date"], records.loc[active, "check"], strict=True))

        release = active & pd.Series(
            [key not in flags for key in zip(records["date"], records["check"], strict=True)],
            index=records.index,
            dtype=bool,
        )
        records.loc[release, "released_at"] = stamp

        new = [
            {
                "ticker": ticker,
                "date": day,
                "check": check,
                **{column: rows.at[day, column] for column in ROW_COLUMNS},
                "severity": str(flag.severity),
                "detail": flag.detail,
                "reason": flag.triage.reason if flag.triage else "",
                "quarantined_at": stamp,
                "released_at": pd.NaT,
            }
            for (day, check), flag in flags.items()
            if (day, check) not in open_keys
        ]
        if new or release.any():
            frames = [records] if not records.empty else []
            if new:
                frames.append(pd.DataFrame(new, columns=list(QUARANTINE_COLUMNS)))
            _write_records(path, pd.concat(frames, ignore_index=True))

        hidden = bars["date"].isin({day for day, _ in flags})
        written = None
        if not bars.empty and not hidden.equals(bars["quarantined"]):
            (written,) = store.write_bars(
                bars.assign(quarantined=hidden), settings=settings, root=root
            )
        results.append(
            QuarantineResult(
                ticker,
                quarantined=int(hidden.sum()),
                added=len(new),
                released=int(release.sum()),
                path=written,
            )
        )
        if new or release.any():
            log.info(
                "%s: %d quarantined (+%d, -%d released)",
                ticker,
                int(hidden.sum()),
                len(new),
                int(release.sum()),
            )
    return results
