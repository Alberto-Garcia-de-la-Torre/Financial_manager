"""One ticker's daily bars, from Yahoo into the store's schema.

    from finmgr.data.fetch import fetch_daily

    bars = fetch_daily("SAN.MC", "2024-10-28", "2024-11-11")
    write_bars(bars)

:func:`fetch_daily` is the only place in the project that turns a Yahoo
response into bars, and it hands back exactly what :mod:`finmgr.data.store`
stores: `ticker, date, open, high, low, close, volume`, lowercase, in that
order, with the dtypes :data:`~finmgr.data.store.BAR_DTYPES` declares. Day 11
wraps it in retries and a status row per ticker; day 12 makes the window
incremental. Neither will need to know what Yahoo's columns are called.

Three decisions are worth stating, because everything downstream inherits them.

**`auto_adjust=False`.** Yahoo will happily hand back a close already adjusted
for dividends, and it is tempting because it saves work. It is also a moving
number: the whole history shifts every time a dividend is paid, so a bar you
stored last month no longer matches the same bar today and no backtest is
reproducible. So the raw close is what gets stored, `Adj Close` is discarded
along with `Dividends` and `Stock Splits` — day 15 ingests those as facts in
their own right — and day 16 builds the adjusted series from them, ours,
versioned, and checkable against Yahoo's.

What `auto_adjust=False` does *not* buy is a pre-split price: Yahoo's daily
endpoint applies splits upstream, so the AAPL bars for August 2020 come back
in post-split money whatever this flag says. That is a property of the source,
not a choice made here, and day 16's comparison has to account for it.

**Exchange-local dates.** yfinance indexes the frame with timestamps in the
exchange's own timezone — a Madrid session is stamped `00:00:00+01:00`. Taking
the calendar date from that timestamp *before* touching UTC is the difference
between a bar dated the day it traded and one dated the evening before; the
store refuses a timezone-aware date column precisely to keep this mistake from
reaching disk.

**Errors raise, empties do not.** A request that fails — a timeout, a 404, a
rate-limit rejection — raises, so day 11's retry loop gets to see it. A window
with no bars in it is not a failure: it returns an empty frame carrying the
full schema, the same shape :func:`~finmgr.data.store.read_bars` returns for a
miss. Yahoo itself does not draw that line — it raises `YFPricesMissingError`
both for a delisted symbol and for a live one asked about a bank holiday, in
the same words — so :func:`download_history` draws it instead.

Nothing here writes to disk, and nothing here retries. One call, one ticker,
one frame back.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data.store import as_date, conform_bars, empty_bars

#: The bar interval this project deals in. Everything downstream — features,
#: labels, the backtest — is daily; intraday is out of scope for v1.
INTERVAL = "1d"

#: Raw prices, never Yahoo's adjusted ones. See the module docstring.
AUTO_ADJUST = False

#: The columns a response must carry, lowercased. `date` comes from the index.
REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Columns yfinance returns that a bar frame deliberately drops. `Adj Close`
#: is a derived number this project computes itself on day 16; the two action
#: columns are day 15's subject and belong in `data/actions/`, not in a bar.
DROPPED_COLUMNS: tuple[str, ...] = ("adj close", "dividends", "stock splits", "capital gains")

#: A downloader: `(ticker, start, end_exclusive) -> DataFrame`. Every function
#: below takes one as an optional argument and falls back to
#: :func:`download_history` *at call time*, which is how the tests run offline.
Downloader = Callable[[str, date, date], Any]

log = runlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# The network call
# ---------------------------------------------------------------------------


def download_history(ticker: str, start: date, end_exclusive: date) -> pd.DataFrame:
    """Ask Yahoo for one ticker's daily bars. The only call here that is online.

    `end_exclusive` is Yahoo's convention — the day after the last session
    wanted. :func:`fetch_daily` takes an inclusive `end` and does the
    conversion, so callers never have to think about it.

    A request that fails — a timeout, a 404, a rate-limit rejection — raises,
    so day 11 can retry it. "Yahoo has no prices in this window" does not: it
    comes back as an empty frame, because a re-run over a weekend asks for a
    window with no sessions in it all the time, and that is not a failure.

    yfinance is imported lazily: the import is slow, and a module that can be
    imported without it is a module the offline tests can exercise whole.
    """
    import yfinance as yf
    from yfinance.exceptions import YFPricesMissingError

    # yfinance swallows request failures by default and returns an empty frame,
    # which is indistinguishable from a delisted symbol. Turning that off is
    # what lets day 11 retry a timeout and give up on a dead ticker.
    yf.config.debug.hide_exceptions = False
    try:
        return yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            interval=INTERVAL,
            auto_adjust=AUTO_ADJUST,
        )
    # The one error that is not one. Yahoo raises this both for a delisted
    # symbol and for a live symbol whose window happens to hold no sessions,
    # and says the same sentence either way — so it cannot be used to tell
    # them apart, and treating it as a failure would make day 12's "nothing
    # new since Friday" look like a broken ticker every weekend. Day 8's
    # validation is what catches a symbol that is genuinely gone.
    except YFPricesMissingError as exc:
        log.debug("%s: no prices for %s..%s (%s)", ticker, start, end_exclusive, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Response -> schema
# ---------------------------------------------------------------------------


def _session_dates(frame: pd.DataFrame) -> pd.Series:
    """The exchange-local calendar date of every row.

    yfinance puts the session timestamp in the index; a frame that has been
    through `reset_index()` carries it in a `Date` column instead, and both are
    accepted. `.dt.date` is taken on the timestamp as it arrives — in exchange
    time — so no conversion to UTC can move a bar to the previous day.
    """
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    if "date" in lowered:
        stamps = frame[lowered["date"]]
    else:
        stamps = pd.Series(frame.index, index=pd.RangeIndex(len(frame)))

    if not isinstance(stamps.dtype, pd.DatetimeTZDtype) and stamps.dtype.kind != "M":
        raise ValueError(
            "no session timestamps: expected a DatetimeIndex, or a 'Date' column, "
            f"got an index of dtype {frame.index.dtype}"
        )
    return stamps.reset_index(drop=True).dt.date


def to_bars(ticker: str, frame: Any) -> pd.DataFrame:
    """Convert one yfinance response into a bar frame. Pure: no network, no clock.

    This is the function the day 10 test pins against a saved response. It
    lowercases the columns, drops the ones the bar schema does not carry, takes
    the exchange-local date off the index, stamps the ticker on every row and
    hands the result to :func:`~finmgr.data.store.conform_bars`, which is what
    guarantees the dtypes rather than this function's own care.

    Prices are passed through untouched — a NaN close, a zero volume or a high
    below its low all survive. They are Yahoo's answer, and day 19 is where
    they get flagged and day 20 where they get quarantined; cleaning them here
    would destroy the evidence before anyone saw it.
    """
    if frame is None or len(frame) == 0:
        return empty_bars()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame from yfinance, got {type(frame).__name__}")

    lowered: dict[str, Any] = {}
    for column in frame.columns:
        name = str(column).strip().lower()
        if name in DROPPED_COLUMNS or name == "date":
            continue
        if name in lowered:
            raise ValueError(f"{ticker}: response has two '{name}' columns")
        lowered[name] = column

    missing = [name for name in REQUIRED_COLUMNS if name not in lowered]
    if missing:
        raise ValueError(
            f"{ticker}: response is missing {missing}; "
            f"got columns {[str(c) for c in frame.columns]}"
        )
    unexpected = [name for name in lowered if name not in REQUIRED_COLUMNS]
    if unexpected:
        # Not fatal: Yahoo adding a column is not a reason to lose the prices.
        # Worth a line in the log, because a new column may be day 15's
        # business, or a sign the response is not the one we think it is.
        log.debug("%s: ignoring unrecognised column(s) %s", ticker, unexpected)

    bars = pd.DataFrame(
        {
            "ticker": str(ticker).strip(),
            "date": _session_dates(frame),
            **{name: frame[lowered[name]].to_numpy() for name in REQUIRED_COLUMNS},
        }
    )

    # Yahoo occasionally repeats the most recent session — a live partial bar
    # arriving alongside the settled one. conform_bars would refuse the frame
    # outright, so collapse it here, keeping the later row: it is the one with
    # the fuller day in it.
    repeated = bars["date"].duplicated(keep="last")
    if repeated.any():
        log.warning(
            "%s: response repeated %d session date(s); kept the last",
            ticker,
            int(repeated.sum()),
        )
        bars = bars.loc[~repeated]

    return conform_bars(bars)


# ---------------------------------------------------------------------------
# The public call
# ---------------------------------------------------------------------------


def fetch_daily(
    ticker: str,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    *,
    settings: Settings | None = None,
    download: Downloader | None = None,
) -> pd.DataFrame:
    """Fetch one ticker's daily bars, in the store's schema.

    `start` and `end` are inclusive bounds on the session date, given as a
    date, a datetime or an ISO string. They default to the configured
    `history_start` and to today in the configured timezone, so
    `fetch_daily("AAPL")` means "the whole history this project cares about".

    Inclusive on both ends because that is what
    :func:`~finmgr.data.store.read_bars` means by `start` and `end`, and a
    project where the same two words bound a window differently depending on
    which function you called is a project with an off-by-one waiting in it.
    Yahoo's exclusive `end` is converted here, once.

    Returns a frame that is ready for :func:`~finmgr.data.store.write_bars`,
    empty-with-schema if the window holds no sessions. Raises if the request
    fails — retrying is day 11's job, not this function's.
    """
    symbol = str(ticker).strip()
    if not symbol:
        raise ValueError("ticker is empty")

    resolved: Settings | None = settings
    if start is None or end is None:
        resolved = resolved or load_settings()
    first = as_date(start, "start") or _history_start(resolved)
    last = as_date(end, "end") or _today(resolved)
    if first > last:
        raise ValueError(f"start {first} is after end {last}")

    fetcher = download_history if download is None else download
    log.debug("fetching %s from %s to %s", symbol, first, last)
    raw = fetcher(symbol, first, last + timedelta(days=1))

    bars = to_bars(symbol, raw)
    if not bars.empty:
        # Yahoo is generous at the edges of a window — a request that lands on
        # a weekend can come back with the Friday before it. The caller asked
        # for a window; give them that window and nothing else, so two calls
        # with touching ranges cannot both claim the same session.
        bars = bars.loc[bars["date"].between(first, last)].reset_index(drop=True)

    if bars.empty:
        log.info("%s: no bars between %s and %s", symbol, first, last)
    else:
        log.info(
            "%s: %d bars from %s to %s",
            symbol,
            len(bars),
            bars["date"].iloc[0],
            bars["date"].iloc[-1],
        )
    return bars


def _history_start(settings: Settings | None) -> date:
    return (settings or load_settings()).history_start


def _today(settings: Settings | None) -> date:
    """Today where the project lives, not where the server thinks it is."""
    resolved = settings or load_settings()
    return datetime.now(ZoneInfo(resolved.timezone)).date()
