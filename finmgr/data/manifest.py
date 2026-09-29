"""The ingestion manifest: one row per ticker per run, kept forever.

    from finmgr.data.manifest import append_run, last_run, read_manifest

    append_run(report)                  # after `ingest_universe`
    frame = read_manifest()             # every run ever recorded
    report = last_run(frame)            # the newest one, as an IngestReport

Day 11's status rows answer "what happened tonight" and then vanish with the
terminal. This module writes them to `data/meta/ingest_runs.parquet` so the
question can be asked weeks later — "why is Iberdrola stale?" — and answered
from a file rather than from memory: the row says whether the ticker was
asked at all, from which date, how many bars came back and were written, what
dates they spanned and, if it failed, the error Yahoo gave.

The layout is deliberately flat. One row per `(run_id, ticker)`, every run
level fact (the window, when it started and finished, whether it stopped
early) repeated on each row, so a single filter on `ticker` gives a symbol's
complete ingestion history without joining anything. At 100 tickers a day the
file grows by about 36,000 rows a year, which Parquet shrugs at; appending is
a read, a concat and an atomic rewrite.

:func:`last_run` rebuilds an :class:`~finmgr.data.ingest.IngestReport` from the
rows alone, which is what `finmgr ingest --report` prints: the same tables and
the same summary line the run itself printed, with nothing downloaded and the
store never opened.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data.ingest import IngestReport, Status, TickerIngest

#: Where the manifest lives under `settings.data_dir`.
MANIFEST_SUBPATH = ("meta", "ingest_runs.parquet")

#: The physical schema, stated rather than inferred for the same reason as the
#: bar store's: a manifest read in eight weeks must have the types it was
#: written with. `rows_written` is `added + updated` — the sessions this run
#: actually changed on disk — and `min_date`/`max_date` span the bars fetched.
MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("rows_fetched", pa.int64(), nullable=False),
        pa.field("rows_written", pa.int64(), nullable=False),
        pa.field("added", pa.int64(), nullable=False),
        pa.field("updated", pa.int64(), nullable=False),
        pa.field("unchanged", pa.int64(), nullable=False),
        pa.field("min_date", pa.date32()),
        pa.field("max_date", pa.date32()),
        pa.field("asked_from", pa.date32()),
        pa.field("stored_max", pa.date32()),
        pa.field("attempts", pa.int64(), nullable=False),
        pa.field("seconds", pa.float64(), nullable=False),
        #: The reason, for `failed` and `skipped` rows; null for the rest.
        pa.field("error", pa.string()),
        pa.field("window_start", pa.date32()),
        pa.field("window_end", pa.date32()),
        pa.field("incremental", pa.bool_(), nullable=False),
        pa.field("interrupted", pa.bool_(), nullable=False),
        pa.field("stopped_early", pa.string(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC")),
        pa.field("finished_at", pa.timestamp("us", tz="UTC")),
    ]
)

MANIFEST_COLUMNS: tuple[str, ...] = tuple(MANIFEST_SCHEMA.names)

log = runlog.get_logger(__name__)


def manifest_path(*, settings: Settings | None = None, path: Path | str | None = None) -> Path:
    """`<data_dir>/meta/ingest_runs.parquet`, unless `path` says otherwise."""
    if path is not None:
        return Path(path).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*MANIFEST_SUBPATH)


def manifest_rows(report: IngestReport, run_id: str) -> pd.DataFrame:
    """One manifest row per status row in `report`, stamped with `run_id`."""
    records: list[dict[str, Any]] = []
    for result in report.results:
        records.append(
            {
                "run_id": run_id,
                "ticker": result.ticker,
                "status": str(result.status),
                "rows_fetched": result.rows,
                "rows_written": result.added + result.updated,
                "added": result.added,
                "updated": result.updated,
                "unchanged": result.unchanged,
                "min_date": result.first_date,
                "max_date": result.last_date,
                "asked_from": result.asked_from,
                "stored_max": result.stored_max,
                "attempts": result.attempts,
                "seconds": float(result.seconds),
                "error": None if result.succeeded else (result.detail or None),
                "window_start": report.start,
                "window_end": report.end,
                "incremental": report.incremental,
                "interrupted": report.interrupted,
                "stopped_early": report.stopped_early,
                "started_at": report.started_at,
                "finished_at": report.finished_at,
            }
        )
    table = pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA)
    return table.to_pandas(types_mapper=_types_mapper)


def _types_mapper(arrow_type: pa.DataType) -> Any:
    # Dates stay dates rather than becoming datetime64 midnights, exactly as
    # the bar store keeps them.
    if pa.types.is_date32(arrow_type):
        return pd.ArrowDtype(arrow_type)
    return None


def read_manifest(
    *, settings: Settings | None = None, path: Path | str | None = None
) -> pd.DataFrame:
    """Every recorded run, oldest first. Empty (with the columns) if none yet."""
    target = manifest_path(settings=settings, path=path)
    if not target.exists():
        return MANIFEST_SCHEMA.empty_table().to_pandas(types_mapper=_types_mapper)
    table = pq.read_table(target, schema=MANIFEST_SCHEMA)
    return table.to_pandas(types_mapper=_types_mapper)


def append_run(
    report: IngestReport,
    *,
    settings: Settings | None = None,
    path: Path | str | None = None,
) -> Path:
    """Add `report`'s rows to the manifest and return the file written.

    A report without a run id — `ingest_universe` called outside the CLI, with
    no run context — gets one minted here, so every row is attributable.
    Recording the same run twice replaces its rows rather than duplicating
    them. The rewrite is staged beside the file and renamed over it, so an
    interruption leaves the previous manifest whole.
    """
    run_id = report.run_id or runlog.mint_run_id(now=report.started_at)
    target = manifest_path(settings=settings, path=path)
    target.parent.mkdir(parents=True, exist_ok=True)

    new = pa.Table.from_pandas(manifest_rows(report, run_id), schema=MANIFEST_SCHEMA)
    if target.exists():
        existing = pq.read_table(target, schema=MANIFEST_SCHEMA)
        keep = pc.not_equal(existing["run_id"], run_id)
        table = pa.concat_tables([existing.filter(keep), new])
    else:
        table = new

    staged = target.with_suffix(target.suffix + ".tmp")
    pq.write_table(table, staged, compression="zstd")
    staged.replace(target)
    log.info("recorded %d manifest row(s) for run %s in %s", new.num_rows, run_id, target)
    return target


def last_run_id(frame: pd.DataFrame) -> str | None:
    """The run that started most recently, or None for an empty manifest.

    Ordered by `started_at` rather than by position in the file, with the run
    id — which begins with a UTC timestamp — breaking ties.
    """
    if frame.empty:
        return None
    runs = frame[["run_id", "started_at"]].drop_duplicates("run_id")
    runs = runs.sort_values(["started_at", "run_id"], na_position="first")
    return str(runs["run_id"].iloc[-1])


def _optional(value: Any) -> Any:
    return None if pd.isna(value) else value


def _timestamp(value: Any) -> datetime | None:
    return None if pd.isna(value) else pd.Timestamp(value).to_pydatetime()


def report_for(frame: pd.DataFrame, run_id: str) -> IngestReport:
    """Rebuild one run's :class:`IngestReport` from its manifest rows alone."""
    rows = frame.loc[frame["run_id"] == run_id]
    if rows.empty:
        raise KeyError(f"no manifest rows for run {run_id!r}")
    results = [
        TickerIngest(
            ticker=row.ticker,
            status=Status(row.status),
            rows=int(row.rows_fetched),
            first_date=_optional(row.min_date),
            last_date=_optional(row.max_date),
            attempts=int(row.attempts),
            seconds=float(row.seconds),
            written=int(row.rows_written) > 0,
            detail=_optional(row.error) or "",
            asked_from=_optional(row.asked_from),
            stored_max=_optional(row.stored_max),
            added=int(row.added),
            updated=int(row.updated),
            unchanged=int(row.unchanged),
        )
        for row in rows.itertuples(index=False)
    ]
    first = rows.iloc[0]
    return IngestReport(
        results=results,
        start=_optional(first["window_start"]),
        end=_optional(first["window_end"]),
        started_at=_timestamp(first["started_at"]),
        finished_at=_timestamp(first["finished_at"]),
        interrupted=bool(first["interrupted"]),
        stopped_early=str(first["stopped_early"]),
        run_id=run_id,
        incremental=bool(first["incremental"]),
    )


def last_run(frame: pd.DataFrame) -> IngestReport | None:
    """The most recent run in `frame`, or None if nothing was ever recorded."""
    run_id = last_run_id(frame)
    return None if run_id is None else report_for(frame, run_id)
