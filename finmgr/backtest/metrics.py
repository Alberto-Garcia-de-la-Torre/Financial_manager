"""Performance metrics for an equity curve and the backtest that produced it.

    from finmgr.backtest.metrics import backtest_metrics
    from finmgr.data.rates import risk_free_returns

    rf = risk_free_returns(result.curve.index, as_of="2026-10-17")
    m = backtest_metrics(result, rf=rf)
    m.sharpe, m.max_drawdown, m.turnover, m.avg_holding
    m.as_dict()          # one flat row, for a table

Every function here is pure: a curve (a Series of equity indexed by date, in
any currency) in, numbers out. The definitions, all in one place:

* **Returns** are simple session returns, `curve.pct_change()`; the first
  date is the starting point and has none.
* **Years** are counted by anniversaries (:func:`year_fraction`): whole
  years back from the last date, plus a stub of actual days over 365 or 366.
  So the window from one 31 December close to the next is exactly 1.0 years
  (leap year or not) and CAGR over it
  *is* that calendar year's return — which is how published annual returns
  are quoted, and what `tests/test_metrics.py` checks against.
* **CAGR** = `(end / start) ** (1 / years) - 1`.
* **Volatility** is the sample standard deviation (ddof=1) of the session
  returns times `sqrt(252)`.
* **Sharpe** uses the excess return over the risk-free series — what one euro
  of overnight cash earned into the same session
  (:func:`finmgr.data.rates.risk_free_returns`): the mean excess return over
  its standard deviation, times `sqrt(252)`. NaN if any session lacks a rate.
  Passing `rf=None` measures against zero, and says so in `Metrics.rf`.
* **Sortino** = annualised mean excess return over the annualised *downside
  deviation*, `sqrt(mean(min(excess, 0) ** 2))` over every session (target =
  the risk-free rate; up sessions count as zero, they are not dropped).
* **Max drawdown** is the worst fall from a running peak, `min(curve /
  cummax(curve) - 1)`, as a negative fraction. Its **duration** runs from the
  peak to the first session back at or above that peak, in sessions on the
  curve's own axis and in calendar days. A drawdown not recovered by the
  last date runs to the last date and `max_drawdown_recovered` is False.
  `longest_drawdown` is the longest underwater stretch of any depth, measured
  the same way.
* **Hit rate** is the share of sessions with a positive return among the
  sessions whose return is not exactly zero (a flat session — a holiday on a
  multi-venue axis, a day in cash — is neither a win nor a loss). **Average
  win / average loss** are the mean positive and the mean negative session
  return (the loss is negative); `payoff` is their ratio in absolute value.
* **Turnover** is one-way and annualised: the notional traded, buys plus
  sells, halved, over mean equity, per year. The fills that first deploy the
  starting cash (those decided at the first date) are not turnover — so
  buy-and-hold has a turnover of zero and a strategy that replaces its whole
  book once a month about 12.
* **Average holding period** is the mean length, in sessions on the axis, of
  every spell in which a name was continuously held. A spell still open at
  the end is counted as it stands (it is censored, so this is a lower bound
  for a long-hold strategy); a trim under `reweight` does not end a spell.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

from finmgr.backtest.engine import BacktestResult

#: Sessions per year, for annualising session statistics.
SESSIONS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _as_date(value: object) -> date:
    return pd.Timestamp(value).date()


def _check_curve(curve: pd.Series) -> pd.Series:
    clean = curve.dropna()
    if len(clean) < 2:
        raise ValueError("an equity curve needs at least two valued dates")
    if not (clean > 0).all():
        raise ValueError("an equity curve must be strictly positive")
    if not clean.index.is_monotonic_increasing or clean.index.has_duplicates:
        raise ValueError("an equity curve must be indexed by strictly increasing dates")
    return clean.astype("float64")


def _years_before(day: date, n: int) -> date:
    """The same calendar date `n` years earlier; 29 February falls back to the 28th."""
    try:
        return day.replace(year=day.year - n)
    except ValueError:
        return day.replace(year=day.year - n, day=28)


def year_fraction(start: date | str, end: date | str) -> float:
    """Years from `start` to `end`, by anniversaries.

    Whole years are counted back from `end` (31 Dec 2024 → 31 Dec 2014 is
    exactly 10); the remaining stub is its calendar days over the length of
    the year that contains it (365 or 366), so a leap day weighs 1/366.
    """
    a, b = _as_date(start), _as_date(end)
    if b < a:
        raise ValueError(f"end {b} is before start {a}")
    whole = b.year - a.year
    anchor = _years_before(b, whole)
    if anchor < a:
        whole -= 1
        anchor = _years_before(b, whole)
    year_before = _years_before(anchor, 1)
    return whole + (anchor - a).days / (anchor - year_before).days


def session_returns(curve: pd.Series) -> pd.Series:
    """Simple returns into every date but the first."""
    return _check_curve(curve).pct_change().iloc[1:]


def total_return(curve: pd.Series) -> float:
    clean = _check_curve(curve)
    return float(clean.iloc[-1] / clean.iloc[0] - 1)


def cagr(curve: pd.Series) -> float:
    """Compound annual growth rate over the curve's first to last date (actual/actual)."""
    clean = _check_curve(curve)
    years = year_fraction(clean.index[0], clean.index[-1])
    if years <= 0:
        return math.nan
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1)


def calendar_returns(curve: pd.Series) -> pd.Series:
    """Total return in each calendar year, last close to last close, indexed by year.

    The first year is measured from the curve's first date, so it is a full
    year only if the curve starts on the previous year's last session; a
    first year holding only that starting date has no return and is left out.
    """
    clean = _check_curve(curve)
    years = pd.Index([_as_date(d).year for d in clean.index])
    last = clean.groupby(years).last()
    base = last.shift(1)
    base.iloc[0] = clean.iloc[0]
    out = last / base - 1
    if (years == years[0]).sum() == 1:
        out = out.iloc[1:]
    out.index.name = "year"
    return out.rename("return")


def annualised_volatility(returns: pd.Series, periods: int = SESSIONS_PER_YEAR) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods))


def excess_returns(returns: pd.Series, rf: pd.Series | None) -> pd.Series:
    """Returns minus what cash earned into the same date; NaN where there is no rate."""
    if rf is None:
        return returns.copy()
    aligned = rf.copy()
    aligned.index = pd.Index([_as_date(d) for d in aligned.index])
    keys = [_as_date(d) for d in returns.index]
    return returns - aligned.reindex(keys).to_numpy(dtype="float64")


def sharpe_ratio(excess: pd.Series, periods: int = SESSIONS_PER_YEAR) -> float:
    spread = float(excess.std(ddof=1)) if len(excess) > 1 else math.nan
    if excess.isna().any() or not spread or math.isnan(spread):
        return math.nan
    return float(excess.mean() / spread * math.sqrt(periods))


def sortino_ratio(excess: pd.Series, periods: int = SESSIONS_PER_YEAR) -> float:
    if excess.isna().any() or excess.empty:
        return math.nan
    downside = math.sqrt(float((np.minimum(excess.to_numpy(), 0.0) ** 2).mean()))
    if not downside:
        return math.nan
    return float(excess.mean() / downside * math.sqrt(periods))


@dataclass(frozen=True)
class Drawdown:
    """One peak-to-recovery episode under water."""

    depth: float  # worst fall from the peak, a negative fraction
    peak: date
    trough: date
    recovery: date | None  # first date back at or above the peak; None if never
    sessions: int  # peak to recovery (or to the last date), on the curve's axis
    days: int  # the same, in calendar days

    @property
    def recovered(self) -> bool:
        return self.recovery is not None


def drawdown_series(curve: pd.Series) -> pd.Series:
    """Fall from the running peak on every date: 0 at a new high, negative below it."""
    clean = _check_curve(curve)
    return clean / clean.cummax() - 1


def drawdowns(curve: pd.Series) -> list[Drawdown]:
    """Every underwater episode, in date order."""
    clean = _check_curve(curve)
    dates = [_as_date(d) for d in clean.index]
    values = clean.to_numpy()
    episodes: list[Drawdown] = []
    peak_i = 0
    i = 1
    n = len(values)
    while i < n:
        if values[i] >= values[peak_i]:
            peak_i = i
            i += 1
            continue
        # Under water from peak_i: run to the first date back at the peak.
        trough_i = i
        j = i
        while j < n and values[j] < values[peak_i]:
            if values[j] < values[trough_i]:
                trough_i = j
            j += 1
        end_i = j if j < n else n - 1
        episodes.append(
            Drawdown(
                depth=float(values[trough_i] / values[peak_i] - 1),
                peak=dates[peak_i],
                trough=dates[trough_i],
                recovery=dates[j] if j < n else None,
                sessions=end_i - peak_i,
                days=(dates[end_i] - dates[peak_i]).days,
            )
        )
        peak_i = j if j < n else peak_i
        i = j + 1
    return episodes


def hit_rate(returns: pd.Series) -> tuple[float, float, float]:
    """(hit rate, average win, average loss) over the non-flat sessions."""
    moved = returns.dropna()
    moved = moved[moved != 0]
    if moved.empty:
        return math.nan, math.nan, math.nan
    wins, losses = moved[moved > 0], moved[moved < 0]
    return (
        float(len(wins) / len(moved)),
        float(wins.mean()) if len(wins) else math.nan,
        float(losses.mean()) if len(losses) else math.nan,
    )


def turnover(
    trades: pd.DataFrame, curve: pd.Series, *, first_decision: date | None = None
) -> float:
    """One-way annualised turnover: (buys + sells) / 2 / mean equity / years.

    Only fills dated on or after the curve's first date and up to its last
    count. Fills decided at `first_decision` (default: the curve's first
    date) fund the book from cash and are left out.
    """
    clean = _check_curve(curve)
    lo, hi = _as_date(clean.index[0]), _as_date(clean.index[-1])
    years = year_fraction(lo, hi)
    if trades.empty or years <= 0:
        return 0.0
    first = first_decision if first_decision is not None else lo
    filled = trades["date"].map(_as_date)
    decided = trades["decided"].map(_as_date)
    keep = (filled >= lo) & (filled <= hi) & (decided != first)
    traded = float(trades.loc[keep, "value_eur"].abs().sum())
    return traded / 2 / float(clean.mean()) / years


def holding_spells(positions: pd.DataFrame, axis: pd.Index | list) -> pd.Series:
    """Length in sessions of every continuous holding spell on `axis`, one entry per spell."""
    days = [_as_date(d) for d in axis]
    if positions.empty or not days:
        return pd.Series([], dtype="int64", name="sessions")
    where = {d: i for i, d in enumerate(days)}
    tickers = sorted(positions["ticker"].unique())
    held = np.zeros((len(days), len(tickers)), dtype=bool)
    col = {t: k for k, t in enumerate(tickers)}
    rows = positions.loc[positions["shares"] > 0, ["date", "ticker"]]
    for d, t in zip(rows["date"], rows["ticker"], strict=True):
        i = where.get(_as_date(d))
        if i is not None:  # positions outside the axis (a trimmed curve) are ignored
            held[i, col[t]] = True
    lengths: list[int] = []
    for k in range(len(tickers)):
        padded = np.concatenate([[False], held[:, k], [False]]).astype(np.int8)
        edges = np.diff(padded)
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        lengths.extend((stops - starts).tolist())
    return pd.Series(lengths, dtype="int64", name="sessions")


# ---------------------------------------------------------------------------
# The full set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Every headline number for one equity curve. Fractions, not percent."""

    start: date
    end: date
    years: float
    sessions: int
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    rf: str  # "risk-free series" or "zero"
    max_drawdown: float
    max_drawdown_peak: date
    max_drawdown_trough: date
    max_drawdown_recovery: date | None
    max_drawdown_sessions: int
    max_drawdown_days: int
    longest_drawdown_sessions: int
    longest_drawdown_days: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    payoff: float
    turnover: float = math.nan  # needs the trades
    avg_holding: float = math.nan  # needs the positions; in sessions

    @property
    def max_drawdown_recovered(self) -> bool:
        return self.max_drawdown_recovery is not None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_metrics(
    curve: pd.Series,
    *,
    rf: pd.Series | None,
    trades: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    periods: int = SESSIONS_PER_YEAR,
) -> Metrics:
    """All metrics for `curve`; turnover and holding period only if trades/positions are given.

    `rf` is the risk-free return into each date (indexed by date, as
    :func:`finmgr.data.rates.risk_free_returns` returns it); `None` means zero.
    """
    clean = _check_curve(curve)
    returns = session_returns(clean)
    excess = excess_returns(returns, rf)
    episodes = drawdowns(clean)
    first, last = _as_date(clean.index[0]), _as_date(clean.index[-1])
    if episodes:
        worst = min(episodes, key=lambda e: e.depth)
        longest = max(episodes, key=lambda e: (e.sessions, e.days))
    else:  # never below a previous high
        worst = Drawdown(0.0, first, first, first, 0, 0)
        longest = worst
    hits, win, loss = hit_rate(returns)
    payoff = abs(win / loss) if loss and not math.isnan(loss) and not math.isnan(win) else math.nan
    avg_holding = math.nan
    if positions is not None:
        spells = holding_spells(positions, list(clean.index))
        avg_holding = float(spells.mean()) if len(spells) else math.nan
    return Metrics(
        start=first,
        end=last,
        years=year_fraction(first, last),
        sessions=len(returns),
        total_return=total_return(clean),
        cagr=cagr(clean),
        volatility=annualised_volatility(returns, periods),
        sharpe=sharpe_ratio(excess, periods),
        sortino=sortino_ratio(excess, periods),
        rf="zero" if rf is None else "risk-free series",
        max_drawdown=worst.depth,
        max_drawdown_peak=worst.peak,
        max_drawdown_trough=worst.trough,
        max_drawdown_recovery=worst.recovery,
        max_drawdown_sessions=worst.sessions,
        max_drawdown_days=worst.days,
        longest_drawdown_sessions=longest.sessions,
        longest_drawdown_days=longest.days,
        hit_rate=hits,
        avg_win=win,
        avg_loss=loss,
        payoff=payoff,
        turnover=turnover(trades, clean) if trades is not None else math.nan,
        avg_holding=avg_holding,
    )


def backtest_metrics(
    result: BacktestResult, *, rf: pd.Series | None, periods: int = SESSIONS_PER_YEAR
) -> Metrics:
    """:func:`compute_metrics` for a :class:`BacktestResult`, turnover and holding included."""
    return compute_metrics(
        result.curve, rf=rf, trades=result.trades, positions=result.positions, periods=periods
    )
