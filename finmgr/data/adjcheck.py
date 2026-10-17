"""Check our adjusted close against Yahoo's, and say why they differ.

    finmgr adjusted --compare --report docs/adjusted_divergences.md

:mod:`finmgr.data.adjust` builds a back-adjusted close from the store. This
module asks Yahoo for its own — the `Close` that `auto_adjust=True` returns —
and lists every ticker where the two are more than
:data:`DIVERGENCE_THRESHOLD` apart on any session. Observing a gap is half the
job; the other half is naming its cause, so every divergence is split into
three parts, each measured on its own:

1. **Raw closes.** Our adjustment run on our stored closes versus the same
   adjustment run on Yahoo's *current* raw closes, our dividends both times.
   Anything here is a difference in the prices themselves.
2. **Dividends.** Yahoo's closes both times; our stored dividends versus the
   dividends Yahoo reports today. Anything here is a dividend that was added,
   dropped or restated since day 15 stored it.
3. **Yahoo's method.** Yahoo's closes and dividends run through our
   adjustment, versus Yahoo's own adjusted close. Anything here is Yahoo
   adjusting differently from what its own inputs imply.

The three multiply back to the total, so whichever one carries the gap is the
cause, and the evidence rows behind it (which sessions, which dividends, which
ex-dates) say exactly where. A divergence none of them accounts for is
reported as unexplained and makes ``finmgr adjusted --compare`` exit non-zero.

**One request, not two.** `auto_adjust=True` gives the adjusted close but
throws the raw close away, and part 1 needs both. yfinance's `auto_adjust`
computes `Close * (Adj Close / Close)` — that is, it returns `Adj Close` —
so a single `auto_adjust=False` request carries both columns and its
`Adj Close` *is* the `auto_adjust=True` close (checked: identical to the bit
over SAN.MC's full history on 2026-10-11).

**Same anchor.** A back-adjusted series is pinned to the latest close, and
Yahoo's latest session is newer than ours. Yahoo's series is therefore divided
by its own adjustment factor on our last stored session, which removes any
dividend paid after it, before the two are compared.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data import fetch as fetch_module
from finmgr.data import pit
from finmgr.data.actions import DIVIDEND, SPLIT, read_actions
from finmgr.data.adjust import adjust_series, dividend_factors
from finmgr.data.ingest import DEFAULT_ATTEMPTS, DEFAULT_PAUSE, Sleeper, fetch_with_retries
from finmgr.data.manifest import read_manifest
from finmgr.data.store import read_bars
from finmgr.data.universe import Company, load_universe

#: A ticker is listed when the two adjusted closes differ by more than this on
#: any session. The roadmap's 0.5%.
DIVERGENCE_THRESHOLD = 0.005

#: A component smaller than this is not named as a cause. Below it the
#: differences are Yahoo's float32 storage, not a fact about the data.
CAUSE_FLOOR = 0.001

#: Two raw closes further apart than this, relatively, are a mismatch.
CLOSE_TOLERANCE = 1e-4

#: Two dividends further apart than this, relatively, are a mismatch.
DIVIDEND_TOLERANCE = 1e-6

#: Yahoo's implied factor on an ex-date further than this from ours is a
#: mismatch. Yahoo's adjusted close is stored to about seven digits, so its
#: day-to-day factor wobbles by ~1e-7; real disagreements are 1e-4 and up.
FACTOR_TOLERANCE = 2e-5

#: `(ticker, start, end_inclusive) -> reference frame`; swapped by the tests.
ReferenceFetcher = Callable[[str, date, date], pd.DataFrame]

REFERENCE_COLUMNS: tuple[str, ...] = ("close", "adj_close", "dividend", "split")

log = runlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Yahoo's side
# ---------------------------------------------------------------------------


def download_reference(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Ask Yahoo for raw close, adjusted close and actions. Online.

    `end` is inclusive, like everywhere else in the project. Uses the same
    error policy as :func:`~finmgr.data.fetch.download_history`: failures
    raise so the retry loop sees them.
    """
    import yfinance as yf

    yf.config.debug.hide_exceptions = False
    raw = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval=fetch_module.INTERVAL,
        auto_adjust=False,
        actions=True,
    )
    return to_reference(raw)


def to_reference(frame: Any) -> pd.DataFrame:
    """A yfinance response as `close, adj_close, dividend, split` by date."""
    if frame is None or len(frame) == 0:
        return pd.DataFrame(
            {c: pd.Series([], dtype="float64") for c in REFERENCE_COLUMNS},
            index=pd.Index([], name="date"),
        )
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for needed in ("close", "adj close"):
        if needed not in lowered:
            raise ValueError(f"reference response has no '{needed}' column")

    def column(name: str) -> np.ndarray:
        if name not in lowered:
            return np.zeros(len(frame))
        return frame[lowered[name]].to_numpy(dtype="float64")

    out = pd.DataFrame(
        {
            "close": column("close"),
            "adj_close": column("adj close"),
            "dividend": np.nan_to_num(column("dividends")),
            "split": np.nan_to_num(column("stock splits")),
        },
        index=pd.Index(fetch_module._session_dates(frame), name="date"),
    )
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(subset=["close", "adj_close"]).sort_index()


# ---------------------------------------------------------------------------
# One ticker
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """Our adjusted close against Yahoo's for one ticker.

    The four divergences are the largest `|a / b - 1|` over the common
    sessions: the total and its three parts (see the module docstring). The
    frames are the evidence for each part.
    """

    ticker: str
    sessions: int = 0
    last_date: date | None = None
    divergence: float = 0.0
    worst_date: date | None = None
    days_over: int = 0
    from_closes: float = 0.0
    from_dividends: float = 0.0
    from_method: float = 0.0
    close_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    dividend_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    method_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    causes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def diverges(self) -> bool:
        return self.divergence > DIVERGENCE_THRESHOLD

    @property
    def explained(self) -> bool:
        return not self.diverges or bool(self.causes)


def _gap(a: pd.Series, b: pd.Series, dates: pd.Index) -> pd.Series:
    return (a.reindex(dates) / b.reindex(dates) - 1.0).abs()


def _close_mismatches(ours: pd.Series, yahoo: pd.Series) -> pd.DataFrame:
    joined = pd.concat([ours.rename("ours"), yahoo.rename("yahoo")], axis=1)
    rel = joined["ours"] / joined["yahoo"] - 1.0
    bad = (rel.abs() > CLOSE_TOLERANCE) | joined["ours"].isna() | joined["yahoo"].isna()
    out = joined[bad].assign(relative=rel[bad])
    out.index.name = "date"
    return out.reset_index()


def _dividend_mismatches(ours: pd.Series, yahoo: pd.Series) -> pd.DataFrame:
    joined = pd.concat([ours.rename("ours"), yahoo.rename("yahoo")], axis=1)
    same = np.isclose(joined["ours"], joined["yahoo"], rtol=DIVIDEND_TOLERANCE, atol=0)
    out = joined[~same]
    out.index.name = "date"
    return out.reset_index()


def _method_mismatches(reference: pd.DataFrame) -> pd.DataFrame:
    """Ex-dates where Yahoo's adjusted close moves by a different factor.

    Yahoo's factor on a session is `adj_close / close`; the step from the
    session before to this one is the factor Yahoo applied for whatever
    happened on this date. Ours comes from Yahoo's own close and dividend.
    A step on a date with no dividend at all shows up too, with dividend 0.
    """
    if len(reference) < 2:
        return pd.DataFrame(
            columns=["date", "dividend", "reference_close", "ours", "yahoo", "applied"]
        )
    k = reference["adj_close"] / reference["close"]
    step = (k.shift(1) / k).iloc[1:]
    dividends = reference.loc[reference["dividend"] > 0, "dividend"]
    expected = dividend_factors(reference["close"], dividends).set_index("date")
    ours = expected["factor"].reindex(step.index).fillna(1.0)
    bad = (step - ours).abs() > FACTOR_TOLERANCE
    rows = []
    for when in step.index[bad]:
        previous = reference["close"][reference.index < when]
        reference_close = float(previous.iloc[-1]) if len(previous) else np.nan
        paid = float(dividends.get(when, 0.0))
        rows.append(
            {
                "date": when,
                "dividend": paid,
                "reference_close": reference_close,
                "ours": float(ours[when]),
                "yahoo": float(step[when]),
                "applied": (1 - float(step[when])) / (1 - float(ours[when])) if paid else np.nan,
            }
        )
    return pd.DataFrame(
        rows, columns=["date", "dividend", "reference_close", "ours", "yahoo", "applied"]
    )


def compare(
    ticker: str,
    closes: pd.Series,
    dividends: pd.Series,
    splits: pd.Series,
    reference: pd.DataFrame,
    *,
    stored_at: tuple[date, datetime] | None = None,
) -> Comparison:
    """Compare one ticker's adjusted close with Yahoo's and attribute the gap.

    `closes`, `dividends` and `splits` are ours, indexed by date; `reference`
    is :func:`to_reference`'s frame. `stored_at` is `(max_date, started_at)`
    of the last ingest that wrote this ticker, used only to word a cause.
    """
    ours_closes = pd.Series(closes, dtype="float64").dropna().sort_index()
    if ours_closes.empty or reference.empty:
        return Comparison(ticker, error="no stored bars" if ours_closes.empty else "no reference")
    last = ours_closes.index[-1]
    upto = reference[reference.index <= last]
    if upto.empty:
        return Comparison(ticker, error=f"Yahoo has nothing on or before {last}")

    anchor = float(upto["adj_close"].iloc[-1] / upto["close"].iloc[-1])
    yahoo = upto["adj_close"] / anchor
    yahoo_dividends = upto.loc[upto["dividend"] > 0, "dividend"]
    yahoo_splits = upto.loc[upto["split"] > 0, "split"]

    ours = adjust_series(ours_closes, dividends, splits)["adj_close"]
    yahoo_closes_our_dividends = adjust_series(upto["close"], dividends, splits)["adj_close"]
    yahoo_inputs = adjust_series(upto["close"], yahoo_dividends, yahoo_splits)["adj_close"]

    common = ours.index.intersection(yahoo.index)
    total = _gap(ours, yahoo, common)
    result = Comparison(
        ticker,
        sessions=len(common),
        last_date=last,
        divergence=float(total.max()),
        worst_date=total.idxmax(),
        days_over=int((total > DIVERGENCE_THRESHOLD).sum()),
        from_closes=float(_gap(ours, yahoo_closes_our_dividends, common).max()),
        from_dividends=float(_gap(yahoo_closes_our_dividends, yahoo_inputs, common).max()),
        from_method=float(_gap(yahoo_inputs, yahoo, common).max()),
        close_mismatches=_close_mismatches(ours_closes, upto["close"]),
        dividend_mismatches=_dividend_mismatches(
            pd.Series(dividends, dtype="float64"), yahoo_dividends
        ),
        method_mismatches=_method_mismatches(upto),
    )
    result.causes = diagnose(result, stored_at=stored_at)
    return result


# ---------------------------------------------------------------------------
# Naming the cause
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _describe_closes(result: Comparison, stored_at: tuple[date, datetime] | None) -> str:
    rows = result.close_mismatches
    priced = rows.dropna(subset=["ours", "yahoo"])
    parts = []
    if not priced.empty:
        worst = priced.loc[priced["relative"].abs().idxmax()]
        parts.append(
            f"{len(priced)} session(s) where our stored raw close differs from Yahoo's "
            f"current one (worst {worst['date']}: {worst['ours']:g} vs {worst['yahoo']:g}, "
            f"{_pct(worst['relative'])})"
        )
        if stored_at is not None and set(priced["date"]) == {result.last_date}:
            max_date, started = stored_at
            if max_date == result.last_date and started.date() == max_date:
                parts.append(
                    f"that is our newest bar, stored by an ingest that ran at "
                    f"{started:%H:%M} UTC on the same day — before that market closed — "
                    f"so it holds an intraday price, not the close; the next incremental "
                    f"ingest re-fetches and overwrites it"
                )
    only_ours = rows["yahoo"].isna().sum()
    only_yahoo = rows["ours"].isna().sum()
    if only_ours:
        parts.append(f"{only_ours} session(s) we store that Yahoo no longer has")
    if only_yahoo:
        parts.append(f"{only_yahoo} session(s) Yahoo has that we do not store")
    return "Raw closes: " + "; ".join(parts) + "."


def _describe_dividends(result: Comparison) -> str:
    rows = result.dividend_mismatches
    restated = rows.dropna()
    parts = []
    if not restated.empty:
        ratios = restated["yahoo"] / restated["ours"]
        sample = ", ".join(
            f"{r.date} {r.ours:g}→{r.yahoo:g}" for r in restated.head(3).itertuples()
        )
        if np.allclose(ratios, ratios.iloc[0], rtol=1e-3):
            parts.append(
                f"Yahoo has restated {len(restated)} dividend(s) since they were stored, "
                f"all by the same x{ratios.iloc[0]:.5f} ({sample})"
            )
        else:
            parts.append(
                f"Yahoo has restated {len(restated)} dividend(s) since they were stored ({sample})"
            )
    added = rows[rows["ours"].isna()]
    dropped = rows[rows["yahoo"].isna()]
    if not added.empty:
        listed = ", ".join(f"{r.date} {r.yahoo:g}" for r in added.head(3).itertuples())
        parts.append(f"Yahoo now reports {len(added)} dividend(s) we never stored ({listed})")
    if not dropped.empty:
        listed = ", ".join(f"{r.date} {r.ours:g}" for r in dropped.head(3).itertuples())
        parts.append(f"{len(dropped)} stored dividend(s) Yahoo no longer reports ({listed})")
    return "Dividends: " + "; ".join(parts) + "."


def _describe_method(result: Comparison) -> str:
    rows = result.method_mismatches
    paid = rows[rows["dividend"] > 0]
    parts = []
    if not paid.empty:
        share = float(paid["applied"].median())
        constant = bool(np.allclose(paid["applied"], share, rtol=1e-3))
        worst = paid.loc[(paid["yahoo"] - paid["ours"]).abs().idxmax()]
        implied = (1 - worst["yahoo"]) * worst["reference_close"]
        if abs(share - 0.01) < 0.001:
            parts.append(
                f"on {len(paid)} ex-date(s) Yahoo's Adj Close moves by 1/100 of the dividend "
                f"it reports itself — price and dividend are both in pence (GBp) and Yahoo's "
                f"adjustment divides the dividend by 100 a second time, so it under-adjusts "
                f"and its history sits too high"
            )
        elif constant:
            parts.append(
                f"on {len(paid)} ex-date(s) Yahoo's Adj Close applies x{share:.4f} of the "
                f"dividend in its own Dividends column, the same ratio every time — an "
                f"exchange rate, not noise: the dividend is declared in another currency, "
                f"Dividends shows it converted into the listing currency at one rate, and "
                f"Adj Close applies the unconverted amount (worst {worst['date']}: Yahoo's "
                f"factor implies {implied:.4g} where Dividends says {worst['dividend']:g})"
            )
        else:
            parts.append(
                f"on {len(paid)} ex-date(s) Yahoo's Adj Close applies a different share of "
                f"its own dividend (median {share:.3f}; worst {worst['date']}: factor "
                f"{worst['yahoo']:.6f} vs {worst['ours']:.6f})"
            )
    unpaid = rows[rows["dividend"] == 0]
    if not unpaid.empty:
        listed = ", ".join(f"{r.date} x{r.yahoo:.5f}" for r in unpaid.head(3).itertuples())
        parts.append(f"Yahoo adjusts on {len(unpaid)} date(s) with no dividend ({listed})")
    return "Yahoo's method: " + "; ".join(parts) + "."


def diagnose(result: Comparison, *, stored_at: tuple[date, datetime] | None = None) -> list[str]:
    """One sentence per part that carries the gap, built from its evidence.

    A part is named only if it moves the series by at least
    :data:`CAUSE_FLOOR` *and* has evidence rows behind it — a part with no
    rows has nothing to point at, and would be a guess.
    """
    causes = []
    if result.from_closes >= CAUSE_FLOOR and not result.close_mismatches.empty:
        causes.append(_describe_closes(result, stored_at))
    if result.from_dividends >= CAUSE_FLOOR and not result.dividend_mismatches.empty:
        causes.append(_describe_dividends(result))
    if result.from_method >= CAUSE_FLOOR and not result.method_mismatches.empty:
        causes.append(_describe_method(result))
    return causes


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _last_writes(settings: Settings | None) -> dict[str, tuple[date, datetime]]:
    """Per ticker, `(max_date, started_at)` of the last run that wrote it."""
    manifest = read_manifest(settings=settings)
    if manifest.empty:
        return {}
    wrote = manifest[(manifest["rows_written"].fillna(0) > 0) & manifest["max_date"].notna()]
    latest = wrote.sort_values("started_at").groupby("ticker").tail(1)
    return {
        row.ticker: (
            pd.Timestamp(row.max_date).date(),
            pd.Timestamp(row.started_at).to_pydatetime(),
        )
        for row in latest.itertuples()
    }


def compare_universe(
    tickers: Iterable[Company | str] | None = None,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    pause: float = DEFAULT_PAUSE,
    settings: Settings | None = None,
    fetch: ReferenceFetcher | None = None,
    sleep: Sleeper | None = None,
) -> list[Comparison]:
    """Compare every ticker's stored adjusted close with Yahoo's.

    Reads bars and actions from the store, asks Yahoo for the window the store
    covers, and returns one :class:`Comparison` per ticker, in order. A ticker
    whose request fails comes back with `error` set; nothing raises.
    """
    resolved = settings or load_settings()
    sleeper = time.sleep if sleep is None else sleep
    fetcher = download_reference if fetch is None else fetch
    symbols = [
        company.ticker if isinstance(company, Company) else str(company).strip()
        for company in (load_universe(settings=resolved) if tickers is None else tickers)
    ]
    # Up to today: this compares the whole stored history with Yahoo's, which
    # is itself computed from everything Yahoo knows now.
    bars = read_bars(symbols, as_of=pit.LATEST, settings=resolved)
    actions = read_actions(symbols, as_of=pit.LATEST, settings=resolved)
    writes = _last_writes(resolved)

    results = []
    for position, symbol in enumerate(symbols):
        mine = bars[bars["ticker"] == symbol]
        if mine.empty:
            results.append(Comparison(symbol, error="no stored bars"))
            continue
        if position and pause:
            sleeper(pause)
        first, last = mine["date"].iloc[0], mine["date"].iloc[-1]
        try:
            reference, _ = fetch_with_retries(
                symbol, first, last, attempts=attempts, fetch=fetcher, sleep=sleep
            )
        except Exception as exc:
            log.warning("%s: no reference from Yahoo: %s", symbol, exc)
            results.append(Comparison(symbol, error=f"{type(exc).__name__}: {exc}"))
            continue
        acts = actions[actions["ticker"] == symbol]
        result = compare(
            symbol,
            mine.set_index("date")["close"],
            acts.loc[acts["action"] == DIVIDEND].set_index("date")["value"],
            acts.loc[acts["action"] == SPLIT].set_index("date")["value"],
            reference,
            stored_at=writes.get(symbol),
        )
        log.info(
            "%s [%d/%d]: max divergence %.4f%%%s",
            symbol,
            position + 1,
            len(symbols),
            result.divergence * 100,
            "" if result.explained else " UNEXPLAINED",
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

#: What each kind of cause means, printed under the table so the report stands
#: on its own when regenerated.
CAUSE_NOTES = """\
## What the causes mean

- **Raw closes.** Our stored raw close and Yahoo's current one disagree on
  some session, so the two adjusted closes disagree on that session too —
  the adjustment factors themselves can be identical. The commonest case is
  the newest stored bar: an ingest that runs while a market is open stores the
  live price under today's date, and Yahoo's settled close differs from it.
  It is transient — every incremental ingest re-asks for the last few stored
  sessions and overwrites them (day 12) — but until then the store holds a
  price that was never a close. Day 19's checks are where it gets flagged.
- **Dividends.** Yahoo's dividend history is not frozen: a dividend can be
  added, dropped or restated after day 15 stored it, and then every adjusted
  close before it moves. Our store keeps what it was told on the day it
  asked; the comparison shows the drift since.
- **Yahoo's method.** Yahoo's `Adj Close` disagrees with its own close and
  dividend columns. Two variants have turned up:
  - *Pence.* On the London listings (`.L`) both columns are in pence (GBp)
    but the adjustment treats the dividend as pounds and divides it by 100
    once more, so Yahoo applies 1% of every dividend and its back history
    sits far too high — 45-57% over fifteen years. Our series uses the
    dividend against the price in the same unit.
  - *Declared in another currency.* UBS (CHF listing) has declared its
    dividend in USD since 2020, and Equinor (NOK listing) has some dividends
    recorded in USD. Yahoo's Dividends column converts them into the listing
    currency at *one* rate for every year — the ratio is identical to five
    digits from 2020 to 2026, so it is today's rate, not the rate on each
    ex-date — and its Adj Close applies the unconverted USD amount against a
    CHF or NOK price. For UBS the implied amounts are exactly the USD
    dividends ($0.365, $0.37, $0.50, $0.55, $0.70, $0.90). Ours uses the
    converted figure, which is the right currency but still at today's rate
    rather than the ex-date's, so it is closer to the truth without being
    exact; and because Yahoo re-converts at each fetch, those stored dividends
    drift a little (Equinor's moved x0.994 between day 15 and day 16).
"""


def render_markdown(
    results: Sequence[Comparison],
    *,
    checked_at: datetime,
    run_id: str | None = None,
) -> str:
    """The report: the divergent tickers with their causes, then everyone."""
    diverging = sorted((r for r in results if r.diverges), key=lambda r: r.divergence, reverse=True)
    unexplained = [r for r in diverging if not r.explained]
    failed = [r for r in results if r.error]
    lines = [
        "# Adjusted close: ours against Yahoo's",
        "",
        "Generated by `finmgr adjusted --compare --report` (roadmap day 16).",
        "Our back-adjusted close (`finmgr.data.adjust`, built from the stored raw",
        "closes and dividends) is compared with Yahoo's `auto_adjust=True` close",
        "on every common session; Yahoo's is re-anchored to our last stored session.",
        "Each divergence is split into raw closes, dividends and Yahoo's own",
        "method — see `finmgr/data/adjcheck.py` for exactly how.",
        "",
        f"- Checked at: {checked_at.isoformat(timespec='seconds')}",
        f"- Threshold: {DIVERGENCE_THRESHOLD:.1%} on any session",
        f"- Result: **{len(diverging)} of {len(results)} tickers diverge; "
        f"{len(unexplained)} unexplained**",
    ]
    if run_id:
        lines.append(f"- Run id: `{run_id}`")
    lines += ["", "## Divergences", ""]
    if not diverging:
        lines.append("None. Every ticker is within the threshold on every session.")
    else:
        lines += [
            "| ticker | max | worst session | sessions over | closes | dividends | Yahoo method |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for r in diverging:
            lines.append(
                f"| {r.ticker} | {r.divergence:.2%} | {r.worst_date} | {r.days_over} "
                f"| {r.from_closes:.2%} | {r.from_dividends:.2%} | {r.from_method:.2%} |"
            )
        lines += ["", "### Cause, ticker by ticker", ""]
        for r in diverging:
            lines.append(f"- **{r.ticker}**")
            for cause in r.causes or ["**Unexplained** — no part carries the gap."]:
                lines.append(f"  - {cause}")
    if failed:
        lines += ["", "## Not compared", ""]
        lines += [f"- {r.ticker}: {r.error}" for r in failed]
    lines += ["", CAUSE_NOTES.rstrip(), "", "## Every ticker", ""]
    lines += [
        "| ticker | sessions | last stored | max divergence | worst session |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.ticker} | - | - | - | {r.error} |")
        else:
            lines.append(
                f"| {r.ticker} | {r.sessions} | {r.last_date} | {r.divergence:.4%} "
                f"| {r.worst_date} |"
            )
    return "\n".join(lines) + "\n"
