"""Exchange rates, and the EUR close they buy every bar.

    from finmgr.data.fx import ingest_fx, refresh_close_eur, read_fx

    ingest_fx()                          # EURUSD, EURGBP, ... into data/fx/
    refresh_close_eur()                  # write close_eur into every bar partition
    read_fx("EURUSD", start="2025-01-01", as_of=LATEST)

A US and a Spanish stock are not comparable until both are in one currency, and
this project's currency is EUR (`base_currency` in settings). So every listing
currency in the universe gets a daily EUR cross from Yahoo — the roadmap's four,
EURUSD, EURGBP, EURCHF and EURSEK, plus whatever else `config/universe.csv`
lists (DKK, NOK, JPY, HKD, TWD, CAD, AUD today). They live in their own store,
in one long schema:

    pair    str                   "EURUSD": units of the quote currency per EUR
    date    date32[day][pyarrow]  the date Yahoo stamps the rate with
    rate    float64               Yahoo's daily close for `EURUSD=X`, untouched

    data/fx/pair=EURUSD/part.parquet

Rates are merged on `(pair, date)` exactly as actions are: added, overwritten
when Yahoo revises them, never deleted because Yahoo stopped sending them.

**The conversion.** `close_eur = close * quote_scale / rate`, with the rate of
the session's own date. Two details decide whether that is right:

*Pence.* Yahoo quotes London equities in GBp, not GBP — AZN.L closes around
12,700, which is £127. The universe says GBP because that is the listing
currency; :data:`QUOTE_SCALE` is where a venue's quote unit is corrected to it,
before dividing by EURGBP. Without it a London share is worth a hundred times
too much in EUR.

*Which rate.* FX trades on days some exchanges do not and the other way round —
Tokyo opens on days Yahoo has no EURJPY row. A session without a same-day rate
takes the most recent rate *before* it (never after: a rate from the future is
lookahead), up to :data:`MAX_RATE_AGE` days old. Older than that, or after the
newest stored rate, the session stays NaN: an unconverted price is visible, a
stale one is not. Yahoo's daily FX close is a London-evening snapshot rather
than each exchange's closing auction, so the EUR close is right to the day, not
to the minute — fine for daily research, and worth knowing before anyone
reasons about intraday FX.

EUR listings need no rate: their close_eur is their close.

:func:`refresh_close_eur` recomputes the column for every stored ticker and
rewrites only the partitions where it changed, so it is cheap to run after
every ingest and safe to run twice.
"""

from __future__ import annotations

import math
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
from finmgr.data import fetch as fetch_module
from finmgr.data import pit, store
from finmgr.data.ingest import (
    DEFAULT_ABORT_AFTER,
    DEFAULT_ATTEMPTS,
    DEFAULT_BACKOFF,
    DEFAULT_PAUSE,
    Sleeper,
    fetch_with_retries,
    resolve_window,
)
from finmgr.data.store import COMPRESSION, PART_FILENAME, as_date
from finmgr.data.universe import Company, load_universe

#: The currency everything is compared in. Settings carries it too; the
#: pairs below are EUR crosses, so anything else would need a different store.
BASE_CURRENCY = "EUR"

#: The crosses the roadmap names. Always ingested, whatever the universe says.
REQUIRED_PAIRS: tuple[str, ...] = ("EURUSD", "EURGBP", "EURCHF", "EURSEK")

FX_COLUMNS: tuple[str, ...] = ("pair", "date", "rate")
FX_KEY: tuple[str, ...] = ("pair", "date")

FX_DTYPES: dict[str, str] = {
    "pair": "str",
    "date": "date32[day][pyarrow]",
    "rate": "float64",
}

FX_ARROW_SCHEMA = pa.schema(
    [
        pa.field("pair", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("rate", pa.float64()),
    ]
)

#: Where the rates live under `settings.data_dir`.
FX_SUBPATH = ("fx",)

#: Multiplier taking a venue's Yahoo quote to its listing currency. London
#: equities are quoted in pence (GBp) against a listing currency of GBP.
QUOTE_SCALE: dict[str, float] = {"XLON": 0.01}

#: Oldest rate, in calendar days, a session without a same-day rate may borrow.
#: A long weekend plus a holiday on either side; anything older is a hole in
#: the FX history, not a closed market, and is left unconverted.
MAX_RATE_AGE = 5

#: A downloader with inclusive bounds returning an FX frame. Swapped out by
#: the tests.
FxFetcher = Callable[[str, date, date], pd.DataFrame]

log = runlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def pair_for(currency: str) -> str | None:
    """The EUR cross a currency is converted with, or None for EUR itself."""
    code = str(currency).strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError(f"{currency!r} is not an ISO currency code")
    return None if code == BASE_CURRENCY else f"{BASE_CURRENCY}{code}"


def yahoo_symbol(pair: str) -> str:
    """Yahoo's spelling of a cross: `EURUSD` -> `EURUSD=X`."""
    return f"{_check_pair(pair)}=X"


def _check_pair(pair: str) -> str:
    cleaned = str(pair).strip().upper()
    if len(cleaned) != 6 or not cleaned.isalpha() or not cleaned.startswith(BASE_CURRENCY):
        raise ValueError(f"{pair!r} is not a {BASE_CURRENCY} cross like EURUSD")
    return cleaned


def universe_pairs(companies: Iterable[Company] | None = None, **kwargs: Any) -> list[str]:
    """Every cross the universe needs, the roadmap's four first, then sorted."""
    listed = load_universe(**kwargs) if companies is None else companies
    needed = {pair_for(company.currency) for company in listed} - {None}
    return list(REQUIRED_PAIRS) + sorted(p for p in needed if p not in REQUIRED_PAIRS)


def quote_scale(company: Company) -> float:
    """What one unit of a venue's Yahoo quote is in the listing currency."""
    return QUOTE_SCALE.get(company.exchange, 1.0)


# ---------------------------------------------------------------------------
# The schema and the store
# ---------------------------------------------------------------------------


def fx_root(*, settings: Settings | None = None, root: Path | str | None = None) -> Path:
    """The root of the FX dataset: `<data_dir>/fx`."""
    if root is not None:
        return Path(root).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*FX_SUBPATH)


def partition_path(
    pair: str, *, settings: Settings | None = None, root: Path | str | None = None
) -> Path:
    """Where one cross lives: `<root>/pair=EURUSD/part.parquet`."""
    return fx_root(settings=settings, root=root) / f"pair={_check_pair(pair)}" / PART_FILENAME


def empty_fx() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([], dtype=dtype) for column, dtype in FX_DTYPES.items()})


def conform_fx(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an FX frame and return it sorted by `(pair, date)`.

    A NaN or non-positive rate is *kept*: it is what Yahoo said, and day 19
    is where bad values get flagged. The conversion simply never uses one.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    if sorted(frame.columns) != sorted(FX_COLUMNS):
        raise ValueError(f"fx frame columns {list(frame.columns)}; expected {list(FX_COLUMNS)}")
    if frame.empty:
        return empty_fx()
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        raise ValueError("the 'date' column is timezone-aware; take the calendar date first")
    out = pd.DataFrame(
        {
            column: frame[column].reset_index(drop=True).astype(dtype)
            for column, dtype in FX_DTYPES.items()
        }
    )
    out["pair"] = out["pair"].map(_check_pair).astype(FX_DTYPES["pair"])
    if out["date"].isna().any():
        raise ValueError(f"{int(out['date'].isna().sum())} rate(s) have no date")
    repeated = out.duplicated(subset=list(FX_KEY), keep=False)
    if repeated.any():
        sample = out.loc[repeated, list(FX_KEY)].drop_duplicates().head(5)
        keys = ", ".join(f"{r.pair}@{r.date}" for r in sample.itertuples())
        raise ValueError(f"{int(repeated.sum())} row(s) repeat a rate, e.g. {keys}")
    return out.sort_values(list(FX_KEY), kind="stable").reset_index(drop=True)


def fetch_fx(
    pair: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """One cross's daily closes between two inclusive dates, from Yahoo.

    Through :func:`~finmgr.data.fetch.fetch_daily`, so the window, the date
    handling and Yahoo's quirks are the bars' own; only the close is kept.
    Raises if the request fails; retries are the caller's.
    """
    cross = _check_pair(pair)
    bars = fetch_module.fetch_daily(yahoo_symbol(cross), start, end, settings=settings)
    return conform_fx(pd.DataFrame({"pair": cross, "date": bars["date"], "rate": bars["close"]}))


@dataclass(frozen=True, slots=True)
class FxMerge:
    """What :func:`merge_fx` did to one cross's partition."""

    pair: str
    added: int = 0
    updated: int = 0
    stored: int = 0
    path: Path | None = None


def merge_fx(
    frame: pd.DataFrame,
    *,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> list[FxMerge]:
    """Fold rates into the store on `(pair, date)`; unchanged partitions are not rewritten."""
    incoming = conform_fx(frame)
    results = []
    for pair, group in incoming.groupby("pair", sort=True):
        cross = str(pair)
        existing = read_fx(cross, as_of=pit.LATEST, settings=settings, root=root)
        joined = group.merge(
            existing, on=list(FX_KEY), how="left", suffixes=("", "_old"), indicator=True
        )
        new = joined["_merge"] == "left_only"
        added = int(new.sum())
        updated = int((~new & ~_same(joined["rate"], joined["rate_old"])).sum())
        if not added and not updated:
            results.append(FxMerge(cross, stored=len(existing)))
            continue

        kept = existing.loc[~existing["date"].isin(group["date"])]
        combined = conform_fx(pd.concat([kept, group], ignore_index=True))
        path = partition_path(cross, settings=settings, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(combined, schema=FX_ARROW_SCHEMA, preserve_index=False)
        staged = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, staged, compression=COMPRESSION)
        staged.replace(path)
        results.append(FxMerge(cross, added, updated, len(combined), path))
    return results


def read_fx(
    pairs: Iterable[str] | str | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    as_of: pit.AsOf,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Read rates out of the store, sorted by `(pair, date)`. Empty, never absent.

    `as_of` is required and no rate dated after it is returned
    (:mod:`finmgr.data.pit`).
    """
    cutoff = pit.resolve_as_of(as_of)
    base = fx_root(settings=settings, root=root)
    if pairs is None:
        files = sorted(base.glob(f"pair=*/{PART_FILENAME}"))
    else:
        wanted = [pairs] if isinstance(pairs, str) else list(pairs)
        files = [
            path
            for path in (partition_path(p, settings=settings, root=root) for p in wanted)
            if path.is_file()
        ]
    if not files:
        return empty_fx()

    columns = ", ".join(f'"{column}"' for column in FX_COLUMNS)
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
    return pit.guard(conform_fx(table.to_pandas()), cutoff, what="read_fx")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FxIngest:
    """One cross's outcome. `error` is empty when it worked."""

    pair: str
    rows: int = 0
    added: int = 0
    updated: int = 0
    attempts: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def ingest_fx(
    pairs: Sequence[str] | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF,
    pause: float = DEFAULT_PAUSE,
    abort_after: int = DEFAULT_ABORT_AFTER,
    settings: Settings | None = None,
    root: Path | str | None = None,
    fetch: FxFetcher | None = None,
    sleep: Sleeper | None = None,
) -> list[FxIngest]:
    """Fetch and merge every cross; one result each, never raises for a pair.

    The whole window every run: a dozen crosses is a dozen requests, and the
    merge makes the overlap free. Same discipline as the other ingests —
    pause, retries with backoff, a try/except per pair, an early stop.
    """
    sleeper = time.sleep if sleep is None else sleep
    crosses = [
        _check_pair(p) for p in (universe_pairs(settings=settings) if pairs is None else pairs)
    ]
    first, last = resolve_window(start, end, settings)

    def fetch_one(cross: str, window_start: date, window_end: date) -> pd.DataFrame:
        if fetch is not None:
            return fetch(cross, window_start, window_end)
        return fetch_fx(cross, window_start, window_end, settings=settings)

    results: list[FxIngest] = []
    consecutive = 0
    for position, cross in enumerate(crosses):
        if abort_after and consecutive >= abort_after:
            results.append(FxIngest(cross, error=f"skipped: {consecutive} failures in a row"))
            continue
        if position and pause:
            sleeper(pause)
        used = 0
        try:
            found, used = fetch_with_retries(
                cross, first, last, attempts=attempts, backoff=backoff, fetch=fetch_one, sleep=sleep
            )
            if found.empty:
                raise ValueError(f"Yahoo returned no rates for {yahoo_symbol(cross)}")
            merged = merge_fx(found, settings=settings, root=root)
        except Exception as exc:
            consecutive += 1
            detail = f"{type(exc).__name__}: {exc}"
            log.warning("%s: rates not ingested: %s", cross, detail)
            results.append(FxIngest(cross, attempts=used or attempts, error=detail))
            continue
        consecutive = 0
        result = FxIngest(
            cross,
            rows=len(found),
            added=sum(m.added for m in merged),
            updated=sum(m.updated for m in merged),
            attempts=used,
        )
        log.info(
            "%s: %d rate(s), %d new, %d revised", cross, result.rows, result.added, result.updated
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def to_eur(
    bars: pd.DataFrame,
    currency: str,
    rates: pd.DataFrame,
    *,
    scale: float = 1.0,
    max_age: int = MAX_RATE_AGE,
) -> pd.Series:
    """The EUR close of every bar, aligned with `bars`. Pure.

    `bars` is one ticker's frame in the store schema; `rates` holds the cross
    `currency` needs (other pairs in it are ignored). Each session takes the
    rate of its own date, else the newest rate before it no more than
    `max_age` days old; anything else is NaN — including every session after
    the newest usable rate, so an evening where the bars arrived and the rates
    did not leaves today unconverted rather than priced at yesterday's FX.
    """
    close = bars["close"].astype("float64").reset_index(drop=True) * scale
    pair = pair_for(currency)
    if pair is None:
        return close.rename("close_eur")
    return (close / rate_on(bars["date"], pair, rates, max_age=max_age)).rename("close_eur")


def rate_on(
    dates: pd.Series, pair: str, rates: pd.DataFrame, *, max_age: int = MAX_RATE_AGE
) -> pd.Series:
    """The rate each date converts at, by :func:`to_eur`'s rule; NaN where none.

    Positional: the result has a fresh RangeIndex, one entry per date in order.
    """
    index = pd.RangeIndex(len(dates))
    usable = rates.loc[(rates["pair"] == pair) & (rates["rate"] > 0), ["date", "rate"]]
    if usable.empty or len(dates) == 0:
        return pd.Series(np.nan, index=index, dtype="float64")

    # merge_asof wants datetime64 keys sorted ascending; the store's date32 is
    # neither on the pandas side, so convert both once.
    left = pd.DataFrame(
        {"when": pd.to_datetime(dates.astype("str")).to_numpy(), "row": range(len(dates))}
    ).sort_values("when", kind="stable")
    right = pd.DataFrame(
        {
            "when": pd.to_datetime(usable["date"].astype("str")).to_numpy(),
            "rate": usable["rate"].to_numpy(),
        }
    ).sort_values("when", kind="stable")
    matched = pd.merge_asof(
        left,
        right,
        on="when",
        direction="backward",
        tolerance=pd.Timedelta(days=max_age),
    ).sort_values("row")
    rate = pd.Series(matched["rate"].to_numpy(), index=index, dtype="float64")
    rate[matched["when"].to_numpy() > right["when"].max()] = np.nan
    return rate


@dataclass(frozen=True, slots=True)
class Conversion:
    """What :func:`refresh_close_eur` did for one ticker."""

    ticker: str
    currency: str = ""
    pair: str | None = None
    rows: int = 0
    converted: int = 0
    changed: int = 0
    path: Path | None = None
    error: str = ""

    @property
    def unconverted(self) -> int:
        return self.rows - self.converted


def _same(new: pd.Series, old: pd.Series) -> pd.Series:
    """Equal, with NaN equal to NaN — the merge's notion of "unchanged"."""
    return (new == old) | (new.isna() & old.isna())


def refresh_close_eur(
    tickers: Iterable[str] | None = None,
    *,
    settings: Settings | None = None,
    bars_dir: Path | str | None = None,
    fx_dir: Path | str | None = None,
    companies: Iterable[Company] | None = None,
) -> list[Conversion]:
    """Recompute close_eur for every stored ticker; rewrite only what changed.

    The listing currency comes from the universe (`companies`, default
    `config/universe.csv`). A stored ticker the universe does not list cannot
    be converted, and is reported with an error rather than guessed at.
    """
    listed = {
        c.ticker: c for c in (load_universe(settings=settings) if companies is None else companies)
    }
    symbols = (
        store.stored_tickers(settings=settings, root=bars_dir)
        if tickers is None
        else [str(t).strip() for t in tickers]
    )
    rates = read_fx(as_of=pit.LATEST, settings=settings, root=fx_dir)

    results: list[Conversion] = []
    for symbol in symbols:
        company = listed.get(symbol)
        if company is None:
            results.append(
                Conversion(symbol, error="not in the universe: listing currency unknown")
            )
            continue
        pair = pair_for(company.currency)
        # Quarantined rows included: the partition is rewritten from this
        # frame, and a read that left them out would delete them.
        bars = store.read_bars(
            symbol, as_of=pit.LATEST, include_quarantined=True, settings=settings, root=bars_dir
        )
        if bars.empty:
            results.append(Conversion(symbol, company.currency, pair, error="no bars stored"))
            continue
        eur = to_eur(bars, company.currency, rates, scale=quote_scale(company))
        changed = int((~_same(eur, bars["close_eur"])).sum())
        path = None
        if changed:
            (path,) = store.write_bars(
                bars.assign(close_eur=eur.to_numpy()), settings=settings, root=bars_dir
            )
        error = ""
        if pair is not None and not (rates["pair"] == pair).any():
            error = f"no {pair} rates stored; run `finmgr fx`"
        results.append(
            Conversion(
                symbol,
                company.currency,
                pair,
                rows=len(bars),
                converted=int(eur.notna().sum()),
                changed=changed,
                path=path,
                error=error,
            )
        )
        log.debug(
            "%s: %d of %d converted, %d changed", symbol, eur.notna().sum(), len(bars), changed
        )
    log.info(
        "close_eur: %d ticker(s), %d partition(s) rewritten",
        len(results),
        sum(1 for r in results if r.path is not None),
    )
    return results


# ---------------------------------------------------------------------------
# The acceptance check: does the EUR series move the way FX says it should?
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Divergence:
    """A ticker's local and EUR returns over one window, and the FX move between.

    For a non-EUR listing the identity is
    `(1 + eur_return) = (1 + local_return) * rate_start / rate_end`: when the
    EUR strengthens against the listing currency (`fx_return` > 0) the EUR
    series must underperform the local one, and the other way round.
    """

    ticker: str
    currency: str
    pair: str | None
    start: date
    end: date
    local_return: float
    eur_return: float
    fx_return: float

    @property
    def gap(self) -> float:
        """EUR return minus local return: the part FX added or took away."""
        return self.eur_return - self.local_return

    @property
    def expected_gap(self) -> float:
        """What the identity says the gap must be, from the FX move alone."""
        return (1 + self.local_return) / (1 + self.fx_return) - 1 - self.local_return

    @property
    def consistent(self) -> bool:
        """The gap has the sign FX implies, and the size, to within a basis point."""
        if self.pair is None or not math.isfinite(self.gap):
            return False
        same_sign = np.sign(self.gap) == -np.sign(self.fx_return)
        return bool(same_sign and abs(self.gap - self.expected_gap) < 1e-4)


def divergence(
    ticker: str,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    settings: Settings | None = None,
    bars_dir: Path | str | None = None,
    fx_dir: Path | str | None = None,
    companies: Iterable[Company] | None = None,
) -> tuple[Divergence, pd.DataFrame]:
    """Compare one ticker's local and EUR closes over a window, from the store.

    Returns the summary and the series behind it: date, close, close_eur, the
    stored rate for each date — read from the FX store afresh rather than
    backed out of close_eur, so the summary's consistency check tests the
    column instead of restating it — and both closes rebased to 100 at the
    first converted session.
    """
    listed = {
        c.ticker: c for c in (load_universe(settings=settings) if companies is None else companies)
    }
    company = listed.get(ticker)
    if company is None:
        raise ValueError(f"{ticker} is not in the universe")
    # The window's end is its point in time: neither series reaches past it.
    as_of = end if end is not None else pit.LATEST
    bars = store.read_bars(
        ticker, start=start, end=end, as_of=as_of, settings=settings, root=bars_dir
    )
    bars = bars.loc[bars["close_eur"].notna() & bars["close"].notna()].reset_index(drop=True)
    if len(bars) < 2:
        raise ValueError(f"{ticker}: fewer than two converted sessions in the window")
    pair = pair_for(company.currency)
    if pair is None:
        rate = pd.Series(1.0, index=bars.index)
    else:
        rates = read_fx(pair, as_of=as_of, settings=settings, root=fx_dir)
        rate = rate_on(bars["date"], pair, rates)
    series = pd.DataFrame(
        {
            "date": bars["date"],
            "close": bars["close"],
            "close_eur": bars["close_eur"],
            "rate": rate.to_numpy(),
            "local_index": 100 * bars["close"] / bars["close"].iloc[0],
            "eur_index": 100 * bars["close_eur"] / bars["close_eur"].iloc[0],
        }
    )
    first, last = series.iloc[0], series.iloc[-1]
    summary = Divergence(
        ticker=ticker,
        currency=company.currency,
        pair=pair,
        start=first["date"],
        end=last["date"],
        local_return=last["close"] / first["close"] - 1,
        eur_return=last["close_eur"] / first["close_eur"] - 1,
        fx_return=last["rate"] / first["rate"] - 1,
    )
    return summary, series


def plot_divergence(summary: Divergence, series: pd.DataFrame, path: Path) -> Path:
    """Both closes rebased to 100, with the cross underneath. Written to `path`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    when = pd.to_datetime(series["date"].astype("str"))
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=(3, 1), constrained_layout=True
    )
    top.plot(when, series["local_index"], label=f"{summary.ticker} in {summary.currency}", lw=1.4)
    top.plot(when, series["eur_index"], label=f"{summary.ticker} in EUR", lw=1.4)
    top.set_ylabel(f"rebased, {summary.start} = 100")
    top.legend(loc="upper left", frameon=False)
    top.set_title(
        f"{summary.ticker}: {summary.local_return:+.1%} in {summary.currency}, "
        f"{summary.eur_return:+.1%} in EUR; {summary.pair} {summary.fx_return:+.1%}"
    )
    bottom.plot(when, series["rate"], color="0.35", lw=1.0)
    bottom.set_ylabel(str(summary.pair))
    for axis in (top, bottom):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
