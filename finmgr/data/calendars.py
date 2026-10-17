"""Exchange calendars: which sessions each ticker *should* have.

A bar store with no bar for SAN.MC on 2024-03-29 is not missing anything —
that was Good Friday, and Madrid was shut. A store with no bar on 2024-03-27
is. Telling the two apart needs the venue's own calendar, and that is what
this module adds on top of `exchange_calendars`:

    from finmgr.data.calendars import expected_sessions, session_gaps

    expected_sessions("XMAD", 2024)          # 256 dates, holidays excluded
    for gap in session_gaps(year=2024):      # one row per universe ticker
        print(gap.ticker, gap.venue, gap.missing)

Each ticker's venue is the `exchange` column of `config/universe.csv`, already
an ISO 10383 MIC (XNYS, XNAS, XMAD, XPAR, XETR, XAMS, ...) — which is exactly
the name `exchange_calendars` files its calendars under, so the mapping is the
universe file itself and there is no second table to keep in step with it.

**What counts as missing.** A session is expected for a ticker when its venue
was open, it falls inside the year asked about, and it lies between the
ticker's first stored bar and `as_of` (yesterday by default). The first bound
is there because a company listed in June has no January sessions to miss; the
second because today's session may not have closed yet. Anything left that the
store has no bar for is genuinely missing. Bars dated on a day the venue was
*closed* are reported too, as `off_calendar` — those are not gaps but they are
just as wrong, and day 19's checks will want both.

**Unscheduled closures.** `exchange_calendars` knows the published holiday
schedule, not the typhoon that shut Taipei on 2024-10-31. Such closures are
listed in :data:`UNSCHEDULED_CLOSURES` with their reason, each one confirmed
against the venue rather than inferred from our own gaps, and reported as
excused rather than silently dropped. Add to that table only with a source;
otherwise it becomes a place where real gaps go to hide.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from finmgr.config import Settings
from finmgr.data.store import coverage, read_bars
from finmgr.data.universe import load_universe

#: Closures the venue announced at short notice, which `exchange_calendars`
#: (4.13.2) does not carry: `{MIC: {date: reason}}`. Every entry needs a
#: reason a person can check.
UNSCHEDULED_CLOSURES: dict[str, dict[date, str]] = {
    "XTAI": {
        # Day 19: the three below are absent from TWSE's own daily market
        # statistics (twse.com.tw exchangeReport/FMTQIK for that month).
        date(2011, 5, 2): "Labour Day (Sun 1 May) observed: TWSE closed",
        date(2022, 2, 4): "Lunar New Year: TWSE closed 27 Jan - 6 Feb",
        date(2023, 1, 18): "pre-Lunar New Year settlement-only day: no TWSE trading",
        date(2024, 10, 31): "typhoon Kong-rey: TWSE closed by typhoon-day order",
    },
    "XHKG": {
        date(2023, 9, 1): "typhoon Saola: signal no. 8 past noon, HKEX cancelled all sessions",
        date(2023, 9, 8): "black rainstorm warning all day: HKEX did not open",
    },
}


@cache
def _calendar(venue: str, year: int) -> xcals.ExchangeCalendar:
    """One calendar per venue and year, built once.

    Bounded to the year because the library's default window is the last
    twenty years, and history_start reaches back further than some of that.
    """
    if venue not in xcals.get_calendar_names(include_aliases=True):
        raise ValueError(f"no exchange calendar for venue {venue!r}")
    return xcals.get_calendar(venue, start=f"{year}-01-01", end=f"{year}-12-31")


def expected_sessions(
    venue: str, year: int, *, start: date | None = None, end: date | None = None
) -> list[date]:
    """Every session `venue` held in `year`, optionally clipped to [start, end].

    Public holidays and weekends are excluded by the calendar; the unscheduled
    closures in :data:`UNSCHEDULED_CLOSURES` are *not* — they are excused by
    :func:`session_gaps`, which reports them, rather than vanishing here.
    """
    sessions = [stamp.date() for stamp in _calendar(venue, year).sessions]
    return [
        day for day in sessions if (start is None or day >= start) and (end is None or day <= end)
    ]


def ticker_venues(
    tickers: Iterable[str] | None = None,
    *,
    settings: Settings | None = None,
    path: Path | str | None = None,
) -> dict[str, str]:
    """`{ticker: MIC}` from the universe file, in file order.

    `tickers` narrows it; asking for a symbol the universe does not list is an
    error rather than a guess, since the venue is the whole point.
    """
    venues = {c.ticker: c.exchange for c in load_universe(path, settings=settings)}
    if tickers is None:
        return venues
    wanted = list(tickers)
    unknown = [ticker for ticker in wanted if ticker not in venues]
    if unknown:
        raise ValueError(f"not in the universe, so no venue: {', '.join(unknown)}")
    return {ticker: venues[ticker] for ticker in wanted}


@dataclass(frozen=True, slots=True)
class SessionGaps:
    """One ticker's year, held against its venue's calendar."""

    ticker: str
    venue: str
    year: int
    #: Sessions the store should hold: the venue's, inside the window, with
    #: unscheduled closures already taken out.
    expected: int
    #: Bars the store holds inside the window.
    stored: int
    #: Expected sessions with no bar — the genuinely missing days.
    missing: tuple[date, ...] = ()
    #: Bars on a day the venue was closed.
    off_calendar: tuple[date, ...] = ()
    #: Sessions the calendar lists but the venue did not hold, with the reason.
    excused: tuple[tuple[date, str], ...] = ()
    #: The window the year was checked over; None when nothing is stored.
    window: tuple[date, date] | None = field(default=None)

    @property
    def clean(self) -> bool:
        return not self.missing and not self.off_calendar


def session_gaps(
    tickers: Iterable[str] | None = None,
    *,
    year: int,
    as_of: date | None = None,
    settings: Settings | None = None,
    universe: Path | str | None = None,
    root: Path | str | None = None,
    include_quarantined: bool = False,
) -> list[SessionGaps]:
    """Missing and off-calendar sessions per ticker for one calendar year.

    `tickers` None means the whole universe, in file order. A ticker with no
    bars at all is checked from the start of the year, so every session shows
    up as missing — "never ingested" is the biggest gap there is.

    By default a quarantined bar counts as missing, because to everything
    downstream it is. The data quality checks set `include_quarantined` so
    they judge what is stored, quarantine or not.
    """
    venues = ticker_venues(tickers, settings=settings, path=universe)
    last = min(date(year, 12, 31), as_of or date.today() - timedelta(days=1))
    first_bars = coverage(list(venues), settings=settings, root=root).set_index("ticker")
    bars = read_bars(
        list(venues),
        start=date(year, 1, 1),
        end=last,
        as_of=last,
        include_quarantined=include_quarantined,
        settings=settings,
        root=root,
    )
    stored_by_ticker = {t: set(group["date"]) for t, group in bars.groupby("ticker", sort=False)}

    results = []
    for ticker, venue in venues.items():
        first_bar = first_bars.loc[ticker, "first_date"]
        first = date(year, 1, 1) if pd.isna(first_bar) else max(first_bar, date(year, 1, 1))
        if first > last:
            results.append(SessionGaps(ticker, venue, year, expected=0, stored=0))
            continue

        closures = UNSCHEDULED_CLOSURES.get(venue, {})
        sessions = set(expected_sessions(venue, year, start=first, end=last))
        excused = tuple(sorted((d, why) for d, why in closures.items() if d in sessions))
        sessions -= {d for d, _ in excused}
        stored = {d for d in stored_by_ticker.get(ticker, set()) if first <= d <= last}
        results.append(
            SessionGaps(
                ticker,
                venue,
                year,
                expected=len(sessions),
                stored=len(stored),
                missing=tuple(sorted(sessions - stored)),
                # An excused day is not in `sessions`, so a bar on one lands
                # here — which is right: it means the closure entry is wrong.
                off_calendar=tuple(sorted(stored - sessions)),
                excused=excused,
                window=(first, last),
            )
        )
    return results
