"""The risk-free rate: the euro overnight rate, from the ECB.

    from finmgr.data.rates import ingest_rates, read_rates, risk_free_returns

    ingest_rates()                                  # ESTR and EONIA into data/rates/
    read_rates("ESTR", as_of="2024-06-28")          # series, date, rate (% a year)
    risk_free_returns(sessions, as_of="2024-06-28") # what cash earned, per session

A Sharpe ratio is excess return over volatility, and "excess" needs a
denominator's worth of cash. This project counts in EUR, so cash is the euro
overnight rate:

* **€STR** (`ESTR`), the ECB's euro short-term rate, published from
  1 October 2019 — ECB series `EST.B.EU000A2X2A25.WT`;
* **EONIA** (`EONIA`), the overnight rate it replaced — `EON.D.EONIA_TO.RATE`,
  which reaches back past `history_start`. From October 2019 to its end in
  January 2022 EONIA was by definition €STR + 8.5 bp, so the two never
  disagree about which one to use: :func:`short_rate` takes €STR wherever it
  exists and EONIA, as published, before it.

Both are stored as the ECB publishes them, in one long schema:

    series  str                   "ESTR" or "EONIA"
    date    date32[day][pyarrow]  the day the rate applies to
    rate    float64               percent a year, ACT/360

    data/rates/series=ESTR/part.parquet

merged on `(series, date)` like the FX store: added, overwritten on revision,
never deleted.

**Accrual.** Overnight money accrues ACT/360: a rate fixed on a Friday is
earned for three calendar days. So cash held from session *t-1* to session *t*
earns `rate(t-1) * days / 360`, with `rate(t-1)` the newest rate dated on or
before *t-1* — known by the morning of *t* (€STR is published at 08:00 the next
TARGET day), never one from after it. That is :func:`risk_free_returns`.

The ECB is not Yahoo, so this module has its own small fetcher; it goes through
`requests` because its CA bundle is the one that verifies the ECB's certificate
on this machine, where the system store does not.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.data.ingest import (
    DEFAULT_ATTEMPTS,
    DEFAULT_BACKOFF,
    DEFAULT_PAUSE,
    Sleeper,
    fetch_with_retries,
    resolve_window,
)
from finmgr.data.store import COMPRESSION, PART_FILENAME, as_date

#: Our name for each series, and the ECB data-portal key it is fetched with.
ECB_SERIES: dict[str, str] = {
    "ESTR": "EST/B.EU000A2X2A25.WT",
    "EONIA": "EON/D.EONIA_TO.RATE",
}

#: In order of preference: the first series with a rate for a day is used.
SPLICE_ORDER: tuple[str, ...] = ("ESTR", "EONIA")

ECB_URL = "https://data-api.ecb.europa.eu/service/data/{key}"

RATE_COLUMNS: tuple[str, ...] = ("series", "date", "rate")
RATE_KEY: tuple[str, ...] = ("series", "date")

RATE_DTYPES: dict[str, str] = {
    "series": "str",
    "date": "date32[day][pyarrow]",
    "rate": "float64",
}

RATE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("series", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("rate", pa.float64()),
    ]
)

#: Where the rates live under `settings.data_dir`.
RATES_SUBPATH = ("rates",)

#: Money-market day count: a year of interest is 360 days of it.
DAY_COUNT = 360

#: Oldest rate, in calendar days, a session may accrue at. Easter's TARGET
#: closure is four days; older than this is a hole in the series.
MAX_RATE_AGE = 7

#: A downloader with inclusive bounds returning a rate frame. Swapped out by
#: the tests.
RateFetcher = Callable[[str, date, date], pd.DataFrame]

log = runlog.get_logger(__name__)


def _check_series(series: str) -> str:
    cleaned = str(series).strip().upper()
    if cleaned not in ECB_SERIES:
        raise ValueError(
            f"{series!r} is not a known rate series; expected one of {list(ECB_SERIES)}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# The schema and the store
# ---------------------------------------------------------------------------


def rates_root(*, settings: Settings | None = None, root: Path | str | None = None) -> Path:
    """The root of the rates dataset: `<data_dir>/rates`."""
    if root is not None:
        return Path(root).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*RATES_SUBPATH)


def partition_path(
    series: str, *, settings: Settings | None = None, root: Path | str | None = None
) -> Path:
    """Where one series lives: `<root>/series=ESTR/part.parquet`."""
    return (
        rates_root(settings=settings, root=root) / f"series={_check_series(series)}" / PART_FILENAME
    )


def empty_rates() -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series([], dtype=dtype) for column, dtype in RATE_DTYPES.items()}
    )


def conform_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a rate frame and return it sorted by `(series, date)`.

    Negative rates are real — €STR spent 2019-2022 below zero — and are kept.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    if sorted(frame.columns) != sorted(RATE_COLUMNS):
        raise ValueError(f"rate frame columns {list(frame.columns)}; expected {list(RATE_COLUMNS)}")
    if frame.empty:
        return empty_rates()
    out = pd.DataFrame(
        {
            column: frame[column].reset_index(drop=True).astype(dtype)
            for column, dtype in RATE_DTYPES.items()
        }
    )
    out["series"] = out["series"].map(_check_series).astype(RATE_DTYPES["series"])
    if out["date"].isna().any():
        raise ValueError(f"{int(out['date'].isna().sum())} rate(s) have no date")
    repeated = out.duplicated(subset=list(RATE_KEY), keep=False)
    if repeated.any():
        raise ValueError(f"{int(repeated.sum())} row(s) repeat a (series, date)")
    return out.sort_values(list(RATE_KEY), kind="stable").reset_index(drop=True)


def parse_ecb_csv(series: str, text: str) -> pd.DataFrame:
    """The ECB's `format=csvdata` answer as a rate frame. Pure.

    Only `TIME_PERIOD` and `OBS_VALUE` are read; an empty body (the ECB's
    answer to a window with no observations) is an empty frame.
    """
    name = _check_series(series)
    if not text.strip():
        return empty_rates()
    raw = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    return conform_rates(
        pd.DataFrame(
            {
                "series": name,
                "date": pd.to_datetime(raw["TIME_PERIOD"]).dt.date,
                "rate": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
            }
        )
    )


def fetch_rates(
    series: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    settings: Settings | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """One series between two inclusive dates, from the ECB data portal.

    Raises if the request fails; retries are the caller's.
    """
    import requests

    name = _check_series(series)
    first, last = resolve_window(start, end, settings)
    response = requests.get(
        ECB_URL.format(key=ECB_SERIES[name]),
        params={
            "format": "csvdata",
            "startPeriod": first.isoformat(),
            "endPeriod": last.isoformat(),
        },
        timeout=timeout,
    )
    # 404 is the ECB's "no observations in that window", not a failure.
    if response.status_code == 404:
        return empty_rates()
    response.raise_for_status()
    return parse_ecb_csv(name, response.text)


@dataclass(frozen=True, slots=True)
class RateMerge:
    """What :func:`merge_rates` did to one series' partition."""

    series: str
    added: int = 0
    updated: int = 0
    stored: int = 0
    path: Path | None = None


def merge_rates(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> list[RateMerge]:
    """Fold rates into the store on `(series, date)`; unchanged partitions are not rewritten."""
    incoming = conform_rates(frame)
    results = []
    for series, group in incoming.groupby("series", sort=True):
        name = str(series)
        existing = read_rates(name, as_of=pit.LATEST, settings=settings, root=root)
        joined = group.merge(
            existing, on=list(RATE_KEY), how="left", suffixes=("", "_old"), indicator=True
        )
        new = joined["_merge"] == "left_only"
        same = (joined["rate"] == joined["rate_old"]) | (
            joined["rate"].isna() & joined["rate_old"].isna()
        )
        added = int(new.sum())
        updated = int((~new & ~same).sum())
        if not added and not updated:
            results.append(RateMerge(name, stored=len(existing)))
            continue

        kept = existing.loc[~existing["date"].isin(group["date"])]
        combined = conform_rates(pd.concat([kept, group], ignore_index=True))
        path = partition_path(name, settings=settings, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(combined, schema=RATE_ARROW_SCHEMA, preserve_index=False)
        staged = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, staged, compression=COMPRESSION)
        staged.replace(path)
        results.append(RateMerge(name, added, updated, len(combined), path))
    return results


def read_rates(
    series: Iterable[str] | str | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    as_of: pit.AsOf,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Read rates out of the store, sorted by `(series, date)`. Empty, never absent.

    `as_of` is required and no rate dated after it is returned
    (:mod:`finmgr.data.pit`).
    """
    cutoff = pit.resolve_as_of(as_of)
    base = rates_root(settings=settings, root=root)
    if series is None:
        files = sorted(base.glob(f"series=*/{PART_FILENAME}"))
    else:
        wanted = [series] if isinstance(series, str) else list(series)
        files = [
            path
            for path in (partition_path(s, settings=settings, root=root) for s in wanted)
            if path.is_file()
        ]
    if not files:
        return empty_rates()

    columns = ", ".join(f'"{column}"' for column in RATE_COLUMNS)
    sql = f"SELECT {columns} FROM read_parquet(?, hive_partitioning = true)"
    params: list[Any] = [[str(path) for path in files]]
    where = []
    for clause, value in (
        ('"date" >= ?', as_date(start, "start")),
        ('"date" <= ?', pit.bound_end(as_date(end, "end"), cutoff, start=as_date(start, "start"))),
    ):
        if value is not None:
            where.append(clause)
            params.append(value)
    sql += " WHERE " + " AND ".join(where)
    with duckdb.connect() as con:
        table = con.execute(sql, params).to_arrow_table()
    return pit.guard(conform_rates(table.to_pandas()), cutoff, what="read_rates")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RateIngest:
    """One series' outcome. `error` is empty when it worked."""

    series: str
    rows: int = 0
    added: int = 0
    updated: int = 0
    attempts: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def ingest_rates(
    series: Sequence[str] | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    pause: float = DEFAULT_PAUSE,
    settings: Settings | None = None,
    root: Path | str | None = None,
    fetch: RateFetcher | None = None,
    sleep: Sleeper | None = None,
) -> list[RateIngest]:
    """Fetch and merge every series; one result each, never raises for a series.

    The whole window every run, like the FX ingest: two requests, and the
    merge makes the overlap free. A series with no observations in the window
    (EONIA, asked only for 2023) is not an error.
    """
    sleeper = time.sleep if sleep is None else sleep
    names = [_check_series(s) for s in (SPLICE_ORDER if series is None else series)]
    first, last = resolve_window(start, end, settings)

    def fetch_one(name: str, window_start: date, window_end: date) -> pd.DataFrame:
        if fetch is not None:
            return fetch(name, window_start, window_end)
        return fetch_rates(name, window_start, window_end, settings=settings)

    results: list[RateIngest] = []
    for position, name in enumerate(names):
        if position and pause:
            sleeper(pause)
        used = 0
        try:
            found, used = fetch_with_retries(
                name, first, last, attempts=attempts, backoff=backoff, fetch=fetch_one, sleep=sleep
            )
            merged = merge_rates(found, settings=settings, root=root) if not found.empty else []
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            log.warning("%s: rates not ingested: %s", name, detail)
            results.append(RateIngest(name, attempts=used or attempts, error=detail))
            continue
        result = RateIngest(
            name,
            rows=len(found),
            added=sum(m.added for m in merged),
            updated=sum(m.updated for m in merged),
            attempts=used,
        )
        log.info(
            "%s: %d rate(s), %d new, %d revised", name, result.rows, result.added, result.updated
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# The short rate, and what cash earned
# ---------------------------------------------------------------------------


def short_rate(rates: pd.DataFrame) -> pd.Series:
    """One overnight rate per day, % a year, indexed by date: €STR, else EONIA. Pure.

    `rates` is a frame in the store schema. Days where every series is NaN
    are dropped.
    """
    usable = rates.dropna(subset=["rate"])
    pieces = []
    covered: set[date] = set()
    for name in SPLICE_ORDER:
        mine = usable.loc[usable["series"] == name]
        mine = mine.loc[~mine["date"].isin(covered)]
        pieces.append(mine)
        covered.update(mine["date"])
    joined = pd.concat(pieces, ignore_index=True) if pieces else usable
    out = pd.Series(
        joined["rate"].to_numpy(dtype="float64"),
        index=pd.Index(list(joined["date"]), name="date"),
        name="rate",
    )
    return out.sort_index()


def risk_free_returns(
    sessions: Iterable[date],
    *,
    as_of: pit.AsOf,
    rates: pd.DataFrame | None = None,
    max_age: int = MAX_RATE_AGE,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.Series:
    """What one euro of overnight cash earned into each session, indexed by session.

    The return into session *t* is `rate(t-1) / 100 * days / 360`, `days` the
    calendar days since the session before, `rate(t-1)` the newest spliced
    rate dated on or before it and no more than `max_age` days old. The first
    session has no "before" and earns 0; a session with no usable rate is NaN.
    `rates` defaults to the store, read at `as_of`.
    """
    cutoff = pit.resolve_as_of(as_of)
    days = sorted({pd.Timestamp(d).date() for d in sessions})
    index = pd.Index(days, name="date")
    if not days:
        return pd.Series([], index=index, dtype="float64", name="rf")
    if days[-1] > cutoff:
        raise pit.LookAheadError(f"session {days[-1]} is after as_of {cutoff}")
    stored = read_rates(as_of=cutoff, settings=settings, root=root) if rates is None else rates
    stored = pit.guard(stored, cutoff, what="risk_free_returns")
    rate = short_rate(stored)
    if rate.empty:
        return pd.Series(np.nan, index=index, dtype="float64", name="rf")

    previous = pd.DataFrame({"when": pd.to_datetime(days[:-1])})
    known = pd.DataFrame({"when": pd.to_datetime(list(rate.index)), "rate": rate.to_numpy()})
    matched = pd.merge_asof(
        previous, known, on="when", direction="backward", tolerance=pd.Timedelta(days=max_age)
    )
    gaps = np.diff(pd.to_datetime(days).to_numpy()).astype("timedelta64[D]").astype("float64")
    accrued = matched["rate"].to_numpy() / 100.0 * gaps / DAY_COUNT
    return pd.Series(np.concatenate([[0.0], accrued]), index=index, dtype="float64", name="rf")
