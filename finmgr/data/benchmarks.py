"""Benchmarks: what the 100 have to beat, and the chart that says whether they did.

    from finmgr.data.benchmarks import ingest_benchmarks, read_benchmarks, compare

    ingest_benchmarks()                                  # bars, actions, close_eur
    read_benchmarks("IWDA.AS", as_of="2024-06-28")       # total return in EUR
    comparison = compare(start="2015-01-01", as_of="2024-06-28")
    plot_comparison(comparison, Path("logs/vs_benchmark.png"))

Without a benchmark there is no way to say whether a 12% year was skill or just
the market. `config/benchmarks.csv` lists four ETFs, in the universe file's
columns so the existing ingest reads them unchanged:

    IWDA.AS   iShares Core MSCI World (Acc), Amsterdam, EUR — *the* benchmark
    SPY       SPDR S&P 500, New York, USD
    EXSA.DE   iShares STOXX Europe 600, Xetra, EUR
    LYXIB.MC  Amundi IBEX 35 (Dist), Madrid, EUR — the IBEX proxy

ETFs rather than the indices (`^IBEX`, `^GSPC`) because an index level is a
price return: it drops every dividend, about 4% a year for the IBEX, and the
100 are measured on total return. Which one the 100 are held against is
`benchmark` in settings: MSCI World, because the universe is global.

**Storage.** Benchmarks are not in the universe, so they live in their own data
directory, laid out exactly like the main one:

    data/benchmarks/bars/daily/ticker=SPY/part.parquet
    data/benchmarks/actions/ticker=SPY/part.parquet

and go through the same code — :func:`~finmgr.data.ingest.ingest_universe`,
:func:`~finmgr.data.actions.ingest_actions`,
:func:`~finmgr.data.fx.refresh_close_eur` and
:func:`~finmgr.data.adjust.read_adjusted`, each pointed at this root. Kept
apart, `finmgr check`, `quarantine` and the cross-section never mistake SPY for
the 101st company.

**Total return in EUR.** :func:`read_total_return` is the series every
comparison here is made in: the back-adjusted close (dividends reinvested,
day 16) times the session's EUR conversion, `close_eur / close` (day 18). It
works on either data directory, and returns are all it is good for — the
level is anchored to the latest close as of `as_of`.

**The comparison.** :func:`compare` puts the 100 and the benchmark on one
calendar — the union of every session, each series carried forward over days
its own market was shut (backward-looking, so no look-ahead) — and rebases each
to 1 at its first session in the window. A company that lists after `start`
starts at 1 on its own first day, and is judged against what the benchmark did
*from that day*, not from `start`. Beside them: the equal-weighted 100,
rebalanced daily, and overnight cash (:mod:`finmgr.data.rates`), which is also
the denominator of every Sharpe ratio in :func:`summarise`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data import actions as actions_module
from finmgr.data import fx as fx_module
from finmgr.data import ingest as ingest_module
from finmgr.data import pit, rates, store
from finmgr.data.adjust import read_adjusted
from finmgr.data.universe import Company, load_universe

#: Where the benchmarks' own data directory lives under `settings.data_dir`.
BENCHMARKS_SUBPATH = ("benchmarks",)

#: The columns :func:`read_total_return` returns, in order.
TOTAL_RETURN_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "close",
    "adj_close",
    "close_eur",
    "adj_close_eur",
)

#: Sessions in a year, for annualising daily statistics.
SESSIONS_PER_YEAR = 252

log = runlog.get_logger(__name__)


def benchmarks_dir(*, settings: Settings | None = None, root: Path | str | None = None) -> Path:
    """The benchmarks' data directory: `<data_dir>/benchmarks`."""
    if root is not None:
        return Path(root).expanduser()
    return (settings or load_settings()).data_dir.joinpath(*BENCHMARKS_SUBPATH)


def load_benchmarks(
    path: Path | str | None = None, *, settings: Settings | None = None
) -> list[Company]:
    """The benchmark list, from `benchmarks_path` in settings unless given."""
    resolved = path if path is not None else (settings or load_settings()).benchmarks_path
    return load_universe(resolved)


def primary_benchmark(*, settings: Settings | None = None) -> str:
    """The ticker the 100 are measured against; must be in the benchmark list."""
    resolved = settings or load_settings()
    listed = [c.ticker for c in load_benchmarks(settings=resolved)]
    if resolved.benchmark not in listed:
        raise ValueError(f"benchmark {resolved.benchmark!r} is not in {resolved.benchmarks_path}")
    return resolved.benchmark


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkIngest:
    """Everything one :func:`ingest_benchmarks` run did."""

    bars: ingest_module.IngestReport
    actions: list[actions_module.ActionIngest]
    conversions: list[fx_module.Conversion]

    @property
    def failed(self) -> list[str]:
        """Every benchmark something went wrong for, by name."""
        broken = {r.ticker for r in self.bars.results if not r.succeeded}
        broken |= {r.ticker for r in self.actions if r.error}
        broken |= {c.ticker for c in self.conversions if c.error}
        return sorted(broken)


def ingest_benchmarks(
    tickers: Sequence[str] | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    settings: Settings | None = None,
    root: Path | str | None = None,
    fx_dir: Path | str | None = None,
    **ingest_options: object,
) -> BenchmarkIngest:
    """Bars, then actions, then close_eur, for every benchmark; never raises per ticker.

    The universe's own ingest functions, pointed at the benchmarks' data
    directory, so retries, incremental windows and the merge behave exactly as
    they do for the 100. Run `finmgr fx` first: the EUR conversion reads the
    rates already stored.
    """
    resolved = settings or load_settings()
    listed = load_benchmarks(settings=resolved)
    if tickers is not None:
        wanted = {str(t).strip() for t in tickers}
        unknown = sorted(wanted - {c.ticker for c in listed})
        if unknown:
            raise ValueError(f"not in {resolved.benchmarks_path}: {', '.join(unknown)}")
        listed = [c for c in listed if c.ticker in wanted]
    base = benchmarks_dir(settings=resolved, root=root)
    bars = ingest_module.ingest_universe(
        listed,
        start=start,
        end=end,
        settings=resolved,
        root=base / "bars" / "daily",
        **ingest_options,  # type: ignore[arg-type]
    )
    paid = actions_module.ingest_actions(
        listed, start=start, end=end, settings=resolved, root=base / "actions"
    )
    conversions = fx_module.refresh_close_eur(
        [c.ticker for c in listed],
        settings=resolved,
        bars_dir=base / "bars" / "daily",
        fx_dir=fx_dir,
        companies=listed,
    )
    return BenchmarkIngest(bars, paid, conversions)


# ---------------------------------------------------------------------------
# Total return in EUR
# ---------------------------------------------------------------------------


def read_total_return(
    tickers: Iterable[str] | str | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    as_of: pit.AsOf,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """Back-adjusted closes, in the listing currency and in EUR. Sorted by `(ticker, date)`.

    `root` is a data directory holding `bars/daily/` and `actions/` — the main
    one by default, :func:`benchmarks_dir` for the benchmarks. `adj_close_eur`
    is NaN wherever the session has no EUR close yet. Nothing dated after
    `as_of` is read, and the adjustment is the one known that evening
    (:func:`~finmgr.data.adjust.read_adjusted`).
    """
    cutoff = pit.resolve_as_of(as_of)
    base = Path(root) if root is not None else (settings or load_settings()).data_dir
    adjusted = read_adjusted(
        tickers, start=start, end=end, as_of=cutoff, settings=settings, root=base
    )
    bars = store.read_bars(
        tickers,
        start=start,
        end=end,
        as_of=cutoff,
        settings=settings,
        root=base / "bars" / "daily",
    )
    if adjusted.empty:
        return pd.DataFrame({column: [] for column in TOTAL_RETURN_COLUMNS})
    out = adjusted[["ticker", "date", "close", "adj_close"]].merge(
        bars[["ticker", "date", "close_eur"]], on=["ticker", "date"], how="left"
    )
    out["adj_close_eur"] = out["adj_close"] * out["close_eur"] / out["close"]
    out = out[list(TOTAL_RETURN_COLUMNS)].sort_values(["ticker", "date"], kind="stable")
    return pit.guard(out.reset_index(drop=True), cutoff, what="read_total_return")


def read_benchmarks(
    tickers: Iterable[str] | str | None = None,
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    as_of: pit.AsOf,
    settings: Settings | None = None,
    root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`read_total_return` over the benchmarks' data directory."""
    return read_total_return(
        tickers,
        start=start,
        end=end,
        as_of=as_of,
        settings=settings,
        root=benchmarks_dir(settings=settings, root=root),
    )


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def growth(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide prices (date x ticker, NaN before a ticker starts) to growth of 1. Pure.

    Each column is carried forward over its gaps and divided by its first
    value, so it starts at exactly 1 on its own first session.
    """
    filled = prices.sort_index().ffill()
    first = filled.apply(lambda column: column.dropna().iloc[0] if column.notna().any() else np.nan)
    return filled / first


def equal_weight(prices: pd.DataFrame) -> pd.Series:
    """Growth of 1 in the equal-weighted portfolio, rebalanced daily. Pure.

    Each day's return is the mean over the tickers that have a return that
    day — a ticker joins the day after its first session.
    """
    returns = prices.sort_index().ffill().pct_change(fill_method=None)
    daily = returns.mean(axis=1, skipna=True).fillna(0.0)
    return (1 + daily).cumprod().rename("equal_weight")


@dataclass(frozen=True, slots=True)
class Comparison:
    """The 100, the benchmark, the equal-weighted 100 and cash, on one calendar.

    Every series is growth of 1 EUR, indexed by date. `stocks` has one column
    per ticker, NaN before the ticker's first session in the window.
    """

    benchmark: str
    start: date
    end: date
    stocks: pd.DataFrame
    benchmark_growth: pd.Series
    equal_weight: pd.Series
    cash: pd.Series
    #: Daily total returns in EUR of the benchmark on its own sessions, and
    #: what cash earned into each — the inputs to its Sharpe ratio.
    benchmark_returns: pd.Series
    benchmark_rf: pd.Series

    def outperformers(self) -> pd.Series:
        """Per ticker: did it beat the benchmark from its own first session?"""
        verdict = {}
        for ticker, column in self.stocks.items():
            valid = column.dropna()
            if valid.empty:
                continue
            bench = self.benchmark_growth.reindex(valid.index).ffill()
            if bench.dropna().empty:
                continue
            verdict[ticker] = bool(valid.iloc[-1] > bench.iloc[-1] / bench.dropna().iloc[0])
        return pd.Series(verdict, dtype="bool", name="beat_benchmark")


def _wide(frame: pd.DataFrame, column: str = "adj_close_eur") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    wide = frame.pivot(index="date", columns="ticker", values=column)
    wide.index = pd.Index([pd.Timestamp(d).date() for d in wide.index], name="date")
    return wide.sort_index()


def compare(
    *,
    start: date | datetime | str | None = None,
    end: date | datetime | str | None = None,
    as_of: pit.AsOf,
    benchmark: str | None = None,
    tickers: Iterable[str] | None = None,
    settings: Settings | None = None,
    data_dir: Path | str | None = None,
    benchmarks_root: Path | str | None = None,
    rates_root: Path | str | None = None,
) -> Comparison:
    """Build the :class:`Comparison` from the stores, as of `as_of`.

    `tickers` defaults to the universe. Directories default to settings; the
    `*_root`/`data_dir` arguments are for tests.
    """
    resolved = settings or load_settings()
    cutoff = pit.resolve_as_of(as_of)
    first = store.as_date(start, "start") or resolved.history_start
    last = pit.bound_end(store.as_date(end, "end"), cutoff, start=first)
    name = benchmark or primary_benchmark(settings=resolved)
    symbols = [c.ticker for c in load_universe(settings=resolved)] if tickers is None else tickers

    stocks = _wide(
        read_total_return(
            list(symbols), start=first, end=last, as_of=cutoff, settings=resolved, root=data_dir
        )
    )
    bench = _wide(
        read_benchmarks(
            name, start=first, end=last, as_of=cutoff, settings=resolved, root=benchmarks_root
        )
    )
    if stocks.empty or name not in bench.columns or bench[name].dropna().empty:
        missing = "the universe" if stocks.empty else name
        raise ValueError(
            f"no EUR total-return series for {missing} between {first} and {last}; "
            "run `finmgr ingest`, `finmgr fx` and `finmgr benchmarks`"
        )
    stocks = stocks.dropna(axis=1, how="all")
    bench_prices = bench[name].dropna()
    # The benchmark's first session anchors the chart: a stock's history
    # before there is no benchmark to compare with.
    stocks = stocks.loc[stocks.index >= bench_prices.index[0]]
    calendar = stocks.index.union(bench_prices.index)

    on_calendar = stocks.reindex(calendar)
    bench_growth = growth(bench_prices.reindex(calendar).to_frame())[name].rename(name)

    rf_all = rates.risk_free_returns(
        list(calendar), as_of=cutoff, settings=resolved, root=rates_root
    )
    cash = (1 + rf_all.fillna(0.0)).cumprod().rename("cash")

    bench_returns = bench_prices.pct_change(fill_method=None).iloc[1:]
    bench_rf = rates.risk_free_returns(
        list(bench_prices.index), as_of=cutoff, settings=resolved, root=rates_root
    ).iloc[1:]
    return Comparison(
        benchmark=name,
        start=calendar[0],
        end=calendar[-1],
        stocks=growth(on_calendar),
        benchmark_growth=bench_growth,
        equal_weight=equal_weight(on_calendar),
        cash=cash,
        benchmark_returns=bench_returns.rename(name),
        benchmark_rf=bench_rf,
    )


@dataclass(frozen=True, slots=True)
class Performance:
    """Headline numbers for one growth series over the comparison window."""

    name: str
    total_return: float
    cagr: float
    volatility: float
    sharpe: float


def performance(
    name: str, returns: pd.Series, rf: pd.Series, *, start: date, end: date
) -> Performance:
    """Total return, CAGR, annualised volatility and Sharpe ratio. Pure.

    `returns` are simple daily returns on the series' own sessions, `rf` what
    cash earned into the same sessions. Sharpe is the annualised mean excess
    return over the annualised volatility of the *excess* return; NaN when
    there is no rate for some session or no variance at all.
    """
    clean = returns.dropna()
    total = float((1 + clean).prod() - 1)
    years = max((end - start).days, 1) / 365.25
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    volatility = float(clean.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR))
    excess = clean - rf.reindex(clean.index)
    spread = float(excess.std(ddof=1))
    if excess.isna().any() or not spread:
        sharpe = math.nan
    else:
        sharpe = float(excess.mean() / spread * math.sqrt(SESSIONS_PER_YEAR))
    return Performance(name, total, cagr, volatility, sharpe)


def summarise(comparison: Comparison) -> list[Performance]:
    """The benchmark, the equal-weighted 100 and cash, as :class:`Performance` rows."""
    c = comparison
    ew_returns = c.equal_weight.pct_change().iloc[1:]
    # Both on the comparison's calendar, and cash's own return *is* the
    # risk-free return — which is why its Sharpe is NaN.
    cash_returns = c.cash.pct_change().iloc[1:]
    return [
        performance(c.benchmark, c.benchmark_returns, c.benchmark_rf, start=c.start, end=c.end),
        performance("equal-weight 100", ew_returns, cash_returns, start=c.start, end=c.end),
        performance("cash (EUR overnight)", cash_returns, cash_returns, start=c.start, end=c.end),
    ]


# ---------------------------------------------------------------------------
# The chart
# ---------------------------------------------------------------------------

#: The reference categorical palette's first two slots, its light surface
#: and its text inks. The 100 are context, not identities, so they get a
#: recessive neutral rather than a hue each.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
BENCHMARK_COLOR = "#2a78d6"
EQUAL_WEIGHT_COLOR = "#eb6834"
STOCK_COLOR = "#a8a7a1"
CASH_COLOR = "#52514e"


def plot_comparison(
    comparison: Comparison, path: Path, *, names: dict[str, str] | None = None
) -> Path:
    """Every stock's cumulative return against the benchmark on one log-scale chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    c = comparison
    when = pd.to_datetime([str(d) for d in c.stocks.index])
    beat = c.outperformers()
    label = (
        f"{c.benchmark}: {names[c.benchmark]}" if names and c.benchmark in names else c.benchmark
    )

    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True, facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    for _, column in c.stocks.items():
        axis.plot(when, column.to_numpy(), color=STOCK_COLOR, lw=0.6, alpha=0.55, zorder=1)
    # One proxy handle so the hundred thin lines get a single legend entry.
    axis.plot([], [], color=STOCK_COLOR, lw=1.2, label=f"each of the {c.stocks.shape[1]} stocks")
    axis.plot(
        when,
        c.cash.to_numpy(),
        color=CASH_COLOR,
        lw=1.5,
        ls=(0, (4, 3)),
        label="cash (€STR/EONIA)",
        zorder=3,
    )
    axis.plot(
        when,
        c.equal_weight.to_numpy(),
        color=EQUAL_WEIGHT_COLOR,
        lw=2.2,
        label="equal-weight 100",
        zorder=4,
    )
    axis.plot(
        when, c.benchmark_growth.to_numpy(), color=BENCHMARK_COLOR, lw=2.6, label=label, zorder=5
    )
    for series, text in (
        (c.benchmark_growth, c.benchmark),
        (c.equal_weight, "equal-weight 100"),
    ):
        end_value = float(series.dropna().iloc[-1])
        axis.annotate(
            f"{text}  {end_value - 1:+.0%}",
            xy=(when[-1], end_value),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )

    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    axis.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    axis.axhline(1.0, color=INK_SECONDARY, lw=0.8, zorder=2)
    axis.set_ylabel("growth of €1, total return in EUR (log scale)", color=INK_SECONDARY)
    axis.set_title(
        f"The {c.stocks.shape[1]} against {c.benchmark}, {c.start} to {c.end}: "
        f"{int(beat.sum())} of {len(beat)} beat it",
        color=INK,
        loc="left",
    )
    axis.legend(loc="upper left", frameon=False, labelcolor=INK)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK_SECONDARY)
    axis.tick_params(colors=INK_SECONDARY)
    axis.grid(alpha=0.2, which="major")
    axis.margins(x=0.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return path
