"""Data quality checks over the bar store, and the register that triages them.

    from finmgr.data.quality import run_checks

    report = run_checks()               # every universe ticker, every stored year
    report.untriaged                    # the flags nobody has looked at yet

Eight checks, each with a fixed severity:

    bad_price          error    open/high/low/close is NaN, zero or negative
    high_below_low     error    the day's high is under its low
    duplicate_date     error    a session stored twice for one ticker
    unexplained_move   error    close-to-close move over 40%, no split to explain it
    close_out_of_range warning  close above the high or below the low
    flat_price         warning  the close unchanged five sessions running
    missing_session    warning  the venue was open, the store has no bar
    off_calendar       warning  a bar on a day the venue was closed

An *error* is a row no number downstream should be computed from; a *warning*
is a row that is probably wrong but would not, alone, wreck a return series.
The calendar checks are :func:`finmgr.data.calendars.session_gaps`, run over
every year the store covers.

**Triage.** A flag is not a verdict. NFLX really did rise 42% on 2013-01-24,
and Yahoo really does serve a volume-0 placeholder bar for Tokyo on New Year's
Day. Someone has to look, decide, and write the decision down — that is
`config/quality_triage.csv`, one rule per finding:

    check,scope,start,end,verdict,reason
    unexplained_move,NFLX,2013-01-24,2013-01-24,genuine,Q4 2012 results: ...

`scope` is a ticker or a venue MIC (a rule for `XTKS` covers all three Tokyo
listings); `start`/`end` bound the flag's date, inclusive. `verdict` is one of

    genuine     the row is right and the flag is a false alarm
    bad-data    the row is wrong; `finmgr quarantine` flags it rather than deleting it
    source-gap  the session traded but Yahoo has no bar for it; nothing to fix

A flag matched by a rule is *triaged*: still reported, never hidden, but it no
longer fails the run. `finmgr check` exits non-zero while anything is
untriaged, so a new bad print shows up the evening it arrives instead of in a
backtest weeks later. A rule that matches nothing is reported as unused, so the
register cannot quietly fill up with decisions about data that is gone.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data.actions import SPLIT, read_actions
from finmgr.data.calendars import SessionGaps, session_gaps, ticker_venues
from finmgr.data.store import coverage, read_bars

#: A close-to-close move bigger than this, either way, needs explaining.
MOVE_THRESHOLD = 0.40

#: This many consecutive sessions on one close is a stuck feed, not a quiet market.
FLAT_SESSIONS = 5

#: Relative slack on the range checks, so float noise in Yahoo's numbers
#: (a close stored as 30.000000001 against a high of 30.0) is not a finding.
RANGE_TOLERANCE = 1e-9

#: Where the triage register lives, relative to the repository root.
TRIAGE_PATH = Path("config") / "quality_triage.csv"

TRIAGE_COLUMNS: tuple[str, ...] = ("check", "scope", "start", "end", "verdict", "reason")


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class Verdict(StrEnum):
    GENUINE = "genuine"
    BAD_DATA = "bad-data"
    SOURCE_GAP = "source-gap"


#: Every check, in report order, with its severity.
CHECKS: dict[str, Severity] = {
    "bad_price": Severity.ERROR,
    "high_below_low": Severity.ERROR,
    "duplicate_date": Severity.ERROR,
    "unexplained_move": Severity.ERROR,
    "close_out_of_range": Severity.WARNING,
    "flat_price": Severity.WARNING,
    "missing_session": Severity.WARNING,
    "off_calendar": Severity.WARNING,
}

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class TriageRule:
    """One decision from the register: flags it matches are triaged."""

    check: str
    scope: str
    start: date
    end: date
    verdict: Verdict
    reason: str

    def matches(self, flag: Flag, venue: str | None) -> bool:
        return (
            self.check == flag.check
            and self.scope in (flag.ticker, venue)
            and self.start <= flag.date <= self.end
        )


@dataclass(frozen=True, slots=True)
class Flag:
    """One finding: which check, which ticker and session, and what it saw."""

    check: str
    ticker: str
    date: date
    detail: str
    triage: TriageRule | None = None

    @property
    def severity(self) -> Severity:
        return CHECKS[self.check]

    @property
    def triaged(self) -> bool:
        return self.triage is not None


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Every flag over the tickers checked, with triage applied."""

    tickers: tuple[str, ...]
    flags: tuple[Flag, ...]
    #: Register rules that matched no flag in this run.
    unused_rules: tuple[TriageRule, ...] = field(default=())

    @property
    def untriaged(self) -> list[Flag]:
        return [flag for flag in self.flags if not flag.triaged]

    @property
    def clean(self) -> bool:
        return not self.untriaged


# ---------------------------------------------------------------------------
# The row checks: pure, one frame in, flags out
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    return "NaN" if pd.isna(value) else f"{value:.6g}"


def _splits_by_ticker(splits: pd.DataFrame | None) -> dict[str, pd.Series]:
    """`{ticker: ratio by ex-date}` from an actions frame; split rows only."""
    if splits is None or splits.empty:
        return {}
    rows = splits[splits["action"] == SPLIT] if "action" in splits else splits
    return {
        str(ticker): pd.Series(group["value"].to_numpy(float), index=list(group["date"]))
        for ticker, group in rows.groupby("ticker", sort=False)
    }


def _explained_by_split(move: float, ratios: Iterable[float], threshold: float) -> bool:
    """True when undoing a split on the day brings the move back under the threshold."""
    return any(abs(move * ratio - 1) <= threshold for ratio in ratios)


def check_ticker(
    ticker: str,
    bars: pd.DataFrame,
    splits: pd.Series | None = None,
    *,
    move_threshold: float = MOVE_THRESHOLD,
    flat_sessions: int = FLAT_SESSIONS,
) -> list[Flag]:
    """The row checks for one ticker's bars (every check except the calendar's).

    `bars` is in the store's schema; `splits` maps ex-date to ratio (4.0 for
    a 4:1). A move is explained by a split whose ex-date falls after the
    previous bar and on or before this one — a split on a closed day takes
    effect at the next session.
    """
    flags: list[Flag] = []
    frame = bars.sort_values("date", kind="stable").reset_index(drop=True)

    for day in frame.loc[frame["date"].duplicated(), "date"].drop_duplicates():
        count = int((frame["date"] == day).sum())
        flags.append(Flag("duplicate_date", ticker, day, f"{count} bars for this session"))
    frame = frame.drop_duplicates("date", keep="last").reset_index(drop=True)

    prices = frame[list(PRICE_COLUMNS)].astype(float)
    bad = prices.isna() | (prices <= 0)
    for i in np.flatnonzero(bad.any(axis=1).to_numpy()):
        columns = [c for c in PRICE_COLUMNS if bad.at[i, c]]
        shown = ", ".join(f"{c}={_fmt(prices.at[i, c])}" for c in columns)
        flags.append(Flag("bad_price", ticker, frame.at[i, "date"], shown))

    high, low, close = prices["high"], prices["low"], prices["close"]
    inverted = high < low
    for i in np.flatnonzero(inverted.to_numpy()):
        detail = f"high {_fmt(high[i])} < low {_fmt(low[i])}"
        flags.append(Flag("high_below_low", ticker, frame.at[i, "date"], detail))

    # An inverted bar has no range for the close to sit in; it is already an error.
    above = close > high * (1 + RANGE_TOLERANCE)
    below = close < low * (1 - RANGE_TOLERANCE)
    for i in np.flatnonzero(((above | below) & ~inverted).to_numpy()):
        if above[i]:
            detail = (
                f"close {_fmt(close[i])} > high {_fmt(high[i])} ({close[i] / high[i] - 1:+.2%})"
            )
        else:
            detail = f"close {_fmt(close[i])} < low {_fmt(low[i])} ({close[i] / low[i] - 1:+.2%})"
        flags.append(Flag("close_out_of_range", ticker, frame.at[i, "date"], detail))

    valid = close.where(close > 0)
    moves = valid / valid.shift(1) - 1
    events = splits if splits is not None else pd.Series(dtype=float)
    for i in np.flatnonzero((moves.abs() > move_threshold).to_numpy()):
        previous, day = frame.at[i - 1, "date"], frame.at[i, "date"]
        ratios = [r for d, r in events.items() if previous < d <= day]
        if _explained_by_split(1 + moves[i], ratios, move_threshold):
            continue
        detail = f"{moves[i]:+.1%} from {previous} ({_fmt(valid[i - 1])} -> {_fmt(valid[i])})"
        if ratios:
            detail += f"; split {', '.join(f'{r:g}:1' for r in ratios)} does not explain it"
        flags.append(Flag("unexplained_move", ticker, day, detail))

    # Runs of an identical close: a new run starts wherever the close changes.
    run_id = (close != close.shift(1)).cumsum()
    for _, run in frame.groupby(run_id, sort=False):
        if len(run) >= flat_sessions and not pd.isna(close[run.index[0]]):
            first, last = run["date"].iloc[0], run["date"].iloc[-1]
            detail = f"close {_fmt(close[run.index[0]])} for {len(run)} sessions through {last}"
            flags.append(Flag("flat_price", ticker, first, detail))

    return flags


def calendar_flags(gaps: Iterable[SessionGaps]) -> list[Flag]:
    """Missing and off-calendar sessions, as flags."""
    flags = []
    for gap in gaps:
        for day in gap.missing:
            flags.append(Flag("missing_session", gap.ticker, day, f"{gap.venue} open, no bar"))
        for day in gap.off_calendar:
            flags.append(Flag("off_calendar", gap.ticker, day, f"{gap.venue} closed, bar stored"))
    return flags


# ---------------------------------------------------------------------------
# The triage register
# ---------------------------------------------------------------------------


def triage_path(path: Path | str | None = None, *, settings: Settings | None = None) -> Path:
    if path is not None:
        return Path(path)
    settings = settings or load_settings()
    # universe_path is config/universe.csv; the register sits beside it.
    return settings.universe_path.parent / TRIAGE_PATH.name


def load_triage(
    path: Path | str | None = None, *, settings: Settings | None = None
) -> list[TriageRule]:
    """The register's rules, in file order. Lines starting with `#` are comments.

    A missing file is an empty register. A malformed row is an error, never
    skipped: a rule that silently fails to load un-triages its flags, which is
    loud, but a rule that loads wrong could triage flags it was never meant to.
    """
    source = triage_path(path, settings=settings)
    if not source.is_file():
        return []
    with source.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if tuple(reader.fieldnames or ()) != TRIAGE_COLUMNS:
        raise ValueError(f"{source}: header must be {','.join(TRIAGE_COLUMNS)}")
    rules = []
    for number, row in enumerate(reader, start=2):
        where = f"{source} rule {number - 1}"
        if row["check"] not in CHECKS:
            raise ValueError(f"{where}: unknown check {row['check']!r}")
        try:
            verdict = Verdict(row["verdict"])
        except ValueError:
            known = ", ".join(v.value for v in Verdict)
            raise ValueError(f"{where}: verdict {row['verdict']!r} is not one of {known}") from None
        start, end = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
        if start > end:
            raise ValueError(f"{where}: start {start} is after end {end}")
        if not (row["scope"] or "").strip() or not (row["reason"] or "").strip():
            raise ValueError(f"{where}: scope and reason are required")
        rules.append(
            TriageRule(
                row["check"], row["scope"].strip(), start, end, verdict, row["reason"].strip()
            )
        )
    return rules


def apply_triage(
    flags: Iterable[Flag], rules: Sequence[TriageRule], venues: dict[str, str]
) -> tuple[list[Flag], list[TriageRule]]:
    """Attach the first matching rule to each flag; also return the rules nothing matched.

    `venues` is `{ticker: MIC}` for the tickers checked. Only a rule scoped to
    one of those, or to their venues, can be unused — a rule about a ticker
    this run never looked at has nothing to say about it.
    """
    used: set[int] = set()
    out = []
    for flag in flags:
        venue = venues.get(flag.ticker)
        for index, rule in enumerate(rules):
            if rule.matches(flag, venue):
                used.add(index)
                flag = replace(flag, triage=rule)
                break
        out.append(flag)
    in_scope = set(venues) | set(venues.values())
    unused = [r for i, r in enumerate(rules) if i not in used and r.scope in in_scope]
    return out, unused


# ---------------------------------------------------------------------------
# The whole store
# ---------------------------------------------------------------------------


def _sort_key(flag: Flag) -> tuple[int, int, str, date]:
    severity = list(Severity).index(flag.severity)
    return (severity, list(CHECKS).index(flag.check), flag.ticker, flag.date)


def run_checks(
    tickers: Iterable[str] | None = None,
    *,
    as_of: date | None = None,
    settings: Settings | None = None,
    universe: Path | str | None = None,
    triage: Path | str | None = None,
    root: Path | str | None = None,
    actions_root: Path | str | None = None,
) -> QualityReport:
    """Every check over the stored bars of `tickers` (None: the universe).

    The calendar checks cover each year from the ticker's first stored bar up
    to `as_of` (yesterday by default, as in :func:`session_gaps`). Reads the
    store, the actions and the register; writes nothing.
    """
    settings = settings or load_settings()
    venues = ticker_venues(tickers, settings=settings, path=universe)
    symbols = list(venues)
    # Quarantined rows included: the checks judge what is on disk, so a
    # quarantined bar keeps raising the flag that put it there, its triage rule
    # stays in use, and quarantining it does not turn it into a missing session.
    bars = read_bars(symbols, include_quarantined=True, settings=settings, root=root)
    splits = _splits_by_ticker(
        read_actions(symbols, action=SPLIT, settings=settings, root=actions_root)
    )

    flags: list[Flag] = []
    for ticker, group in bars.groupby("ticker", sort=False):
        flags.extend(check_ticker(str(ticker), group, splits.get(str(ticker))))

    last = as_of or date.today() - timedelta(days=1)
    first_dates = coverage(symbols, settings=settings, root=root)["first_date"].dropna()
    if not first_dates.empty:
        for year in range(min(first_dates).year, last.year + 1):
            gaps = session_gaps(
                symbols,
                year=year,
                as_of=last,
                settings=settings,
                universe=universe,
                root=root,
                include_quarantined=True,
            )
            flags.extend(calendar_flags(gaps))

    rules = load_triage(triage, settings=settings)
    flags, unused = apply_triage(flags, rules, venues)
    return QualityReport(tuple(symbols), tuple(sorted(flags, key=_sort_key)), tuple(unused))
