"""Market and sector relatives: beta, market-neutral momentum, sector-demeaned features.

    from finmgr.features.relative import read_relatives, sector_demean, sector_share

    rel = read_relatives(as_of="2024-06-28", start="2024-01-01")   # beta, residual momentum
    panel = mom.merge(rel, on=["ticker", "date", "adj_close_eur"])
    panel = sector_demean(panel)            # mom_12_1_sec, ... for SECTOR_FEATURES
    sector_share(panel, "mom_12_1").mean()  # how much of 12-1 is just "which sector"

Day 29 ranks every name against the other 99. That is only half the answer: in
a year when energy rallied, the top of the 12-1 ranking is simply the energy
sector, and a model reading it learns a sector bet, not a stock signal. This
module removes the two common components a stock's return carries.

**The market: beta.** Per ticker, on its venue's sessions, `beta_252` is the
OLS slope of the stock's daily log return on the benchmark's over the trailing
252 sessions (t included), from `cov / var` of the pairs both sides have:

    beta_252        cov(r_i, r_m) / var(r_m)                    over (t-252, t]
    idio_vol_252    sqrt((var(r_i) - cov² / var(r_m)) * 252)    the residual's vol, annualized
    mom_resid_12_1  (l_i[t-21] - l_i[t-252]) - beta_252 * (l_m[t-21] - l_m[t-252])

`l` is the log price, so `mom_resid_12_1` is the sum of the stock's daily
market-neutral returns `r_i - beta * r_m` over the 12-1 window: the 12-1
momentum with the part the market explains taken out. A high-beta name in a
rising year no longer tops the list for being high-beta. It is in logs (a sum
of residuals), unlike `mom_12_1`, and has no intercept, so a steady
outperformer keeps its outperformance. A window needs a full 252 sessions of
history and at least :data:`BETA_MIN_PAIRS` return pairs, otherwise NaN.

**The benchmark on the stock's calendar.** The benchmark (`benchmark` in
settings, MSCI World) is laid on each stock's own sessions, carrying its last
close forward over a day only its market was shut (at most
:data:`BENCH_FILL_LIMIT` sessions): backward-looking, so no look-ahead. MSCI
World trades in Amsterdam and closes at 17:30 CET, before New York does, so a
US stock's same-day return partly reacts to news the benchmark sees the next
day. Daily betas of US names to a European-listed benchmark are biased low by
that non-synchrony; it shifts every US beta the same way, so ranks within the
US names are less affected than levels.

**The sector: demeaning.** :func:`sector_demean` subtracts, one date at a time,
the mean of the feature over the names in the same sector (`sector` from the
universe file, day 7) that have a value. A `<f>_sec` value says how far a name
is above its own sector that day; a sector's `_sec` values average zero on
every date, so by construction no sector sits higher than another. A sector
with fewer than :data:`MIN_SECTOR_NAMES` valued names that date gets NaN — a
lone name minus itself is 0, which is not information. :func:`sector_share`
measures what is removed: the share of a feature's cross-sectional variance
that sector membership explains (η², 0 to 1) per date.

**No look-ahead.** Every relative at t reads prices at t and earlier
(positive shifts and trailing windows only), and the sector mean at t reads
other names' values at t only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from finmgr.config import Settings, load_settings
from finmgr.data import pit
from finmgr.data.benchmarks import benchmarks_dir, primary_benchmark, read_total_return
from finmgr.data.universe import load_universe
from finmgr.features.momentum import WINDOWS
from finmgr.features.returns import DEFAULT_PRICE, session_prices, ticker_sessions
from finmgr.features.trend import dist_column
from finmgr.features.volatility import vol_column

#: Sessions in the beta regression window.
BETA_WINDOW = 252
#: Fewest (stock, benchmark) return pairs a window needs: three quarters of it.
BETA_MIN_PAIRS = 189
#: Longest run of stock sessions the benchmark's last close is carried over.
BENCH_FILL_LIMIT = 5
#: Sessions in a year, for annualizing the residual volatility.
SESSIONS_PER_YEAR = 252

BETA_COLUMN = "beta_252"
IDIO_VOL_COLUMN = "idio_vol_252"
RESID_MOM_COLUMN = "mom_resid_12_1"

#: What :func:`compute_relatives` adds, in order.
RELATIVE_COLUMNS: tuple[str, ...] = (BETA_COLUMN, IDIO_VOL_COLUMN, RESID_MOM_COLUMN)

#: The 12-1 window residual momentum is taken over, as in :mod:`.momentum`.
RESID_LOOKBACK, RESID_SKIP = WINDOWS["mom_12_1"]

SECTOR_SUFFIX = "_sec"
#: Fewest valued names a sector needs on a date before it is demeaned.
MIN_SECTOR_NAMES = 3

#: The features :func:`sector_demean` does by default: the momentum family —
#: where sector years dominate most — plus the risk and trend levels that
#: differ by sector by nature (utilities are calm, tech is far from its 200d).
SECTOR_FEATURES: tuple[str, ...] = (
    "mom_12_1",
    "mom_6_1",
    RESID_MOM_COLUMN,
    "rev_5d",
    vol_column(63),
    dist_column("sma", 200),
)


def sector_column(feature: str) -> str:
    """`mom_12_1_sec` for `mom_12_1`."""
    return f"{feature}{SECTOR_SUFFIX}"


#: Every `_sec` column :func:`sector_demean` adds by default.
SECTOR_COLUMNS: tuple[str, ...] = tuple(sector_column(f) for f in SECTOR_FEATURES)


# ---------------------------------------------------------------------------
# Beta and residual momentum
# ---------------------------------------------------------------------------


def bench_on_grid(bench: pd.Series, grid: pd.Index, *, limit: int = BENCH_FILL_LIMIT) -> pd.Series:
    """The benchmark's price on `grid`, its last close carried over at most `limit` sessions.

    `bench` is indexed by its own dates. Carrying forward only ever reads an
    earlier close, so the value on a grid date is known on that date.
    """
    union = bench.index.union(grid)
    return bench.reindex(union).ffill(limit=limit).reindex(grid)


def rolling_beta(
    stock: pd.Series,
    market: pd.Series,
    *,
    window: int = BETA_WINDOW,
    min_pairs: int = BETA_MIN_PAIRS,
) -> pd.DataFrame:
    """Trailing OLS of `stock` on `market` returns: `beta`, `resid_var`, `pairs`.

    Both are return series on one index. Only rows where both have a value
    count; a window shorter than `window` rows, with fewer than `min_pairs`
    pairs, or with a flat market, is NaN. Population moments (ddof=0).
    """
    if not 2 <= min_pairs <= window:
        raise ValueError(f"need 2 <= min_pairs <= window, got {min_pairs=}, {window=}")
    both = stock.notna() & market.notna()
    y = stock.where(both, 0.0)
    x = market.where(both, 0.0)

    def total(values: pd.Series) -> pd.Series:
        return values.rolling(window, min_periods=window).sum()

    n = total(both.astype(float))
    mx, my = total(x) / n, total(y) / n
    var_x = total(x * x) / n - mx * mx
    var_y = total(y * y) / n - my * my
    cov = total(x * y) / n - mx * my
    ok = (n >= min_pairs) & (var_x > 1e-16)
    beta = (cov / var_x).where(ok)
    resid = (var_y - cov * cov / var_x).clip(lower=0.0).where(ok)
    return pd.DataFrame({"beta": beta, "resid_var": resid, "pairs": n})


def _one_ticker(
    frame: pd.DataFrame, bench: pd.Series, price: str, grid: Sequence[date] | None
) -> pd.DataFrame:
    prices = session_prices(frame, price, grid)
    market = bench_on_grid(bench, prices.index)
    log_i, log_m = np.log(prices), np.log(market)
    fit = rolling_beta(log_i.diff(), log_m.diff())
    beta = fit["beta"]

    def window(logs: pd.Series) -> pd.Series:
        return logs.shift(RESID_SKIP) - logs.shift(RESID_LOOKBACK)

    out = pd.DataFrame(
        {
            BETA_COLUMN: beta,
            IDIO_VOL_COLUMN: np.sqrt(fit["resid_var"] * SESSIONS_PER_YEAR),
            RESID_MOM_COLUMN: window(log_i) - beta * window(log_m),
        },
        index=prices.index,
    )
    return out.loc[frame["date"]]


def compute_relatives(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    price: str = DEFAULT_PRICE,
    sessions: Mapping[str, Sequence[date]] | None = None,
) -> pd.DataFrame:
    """:data:`RELATIVE_COLUMNS` per ticker against one benchmark, backward only.

    `prices` is long, one row per `(ticker, date)`, as for
    :func:`~finmgr.features.returns.compute_returns`; `benchmark` has `date`
    and `price` (one ticker). `sessions` maps a ticker to its venue's sessions.
    Returns `ticker, date, <price>` and :data:`RELATIVE_COLUMNS`, sorted by
    `(ticker, date)`.
    """
    for name, frame, needed in (
        ("prices", prices, {"ticker", "date", price}),
        ("benchmark", benchmark, {"date", price}),
    ):
        missing = needed - set(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing column(s): {', '.join(sorted(missing))}")
    if "ticker" in benchmark.columns and benchmark["ticker"].nunique() > 1:
        raise ValueError("benchmark holds more than one ticker")
    if benchmark["date"].duplicated().any():
        raise ValueError("benchmark has more than one row for a date")
    base = prices[["ticker", "date", price]].sort_values(["ticker", "date"], kind="stable")
    if base.duplicated(["ticker", "date"]).any():
        raise ValueError("prices has more than one row for a (ticker, date)")
    if base.empty:
        return pd.DataFrame({c: [] for c in ["ticker", "date", price, *RELATIVE_COLUMNS]})

    bench = benchmark.set_index("date")[price].astype(float).sort_index()
    bench = bench.where(bench > 0)
    parts = []
    for ticker, frame in base.groupby("ticker", sort=False):
        grid = sessions.get(ticker) if sessions is not None else None
        part = _one_ticker(frame, bench, price, grid).reset_index(drop=True)
        parts.append(pd.concat([frame.reset_index(drop=True), part], axis=1))
    return pd.concat(parts, ignore_index=True)


def read_sectors(*, settings: Settings | None = None) -> dict[str, str]:
    """`{ticker: sector}` from the universe file."""
    return {c.ticker: c.sector for c in load_universe(settings=settings)}


def read_relatives(
    tickers: Iterable[str] | str | None = None,
    *,
    as_of: pit.AsOf,
    start: date | str | None = None,
    benchmark: str | None = None,
    price: str = DEFAULT_PRICE,
    settings: Settings | None = None,
    root: Path | str | None = None,
    benchmark_root: Path | str | None = None,
) -> pd.DataFrame:
    """:func:`compute_relatives` on the store, plus each ticker's `sector`.

    `benchmark` defaults to the one in settings, read from the benchmarks'
    data directory. Nothing after `as_of` is read; `start` trims the output,
    and enough history before it is read for every column to be defined.
    """
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    first = date.fromisoformat(start) if isinstance(start, str) else start
    pit.bound_end(None, cutoff, start=first)
    bench_ticker = benchmark or primary_benchmark(settings=resolved)
    read_from = None if first is None else first - timedelta(days=2 * BETA_WINDOW + 14)
    prices = read_total_return(tickers, start=read_from, as_of=cutoff, settings=resolved, root=root)
    bench = read_total_return(
        bench_ticker,
        start=read_from,
        as_of=cutoff,
        settings=resolved,
        root=benchmark_root or benchmarks_dir(settings=resolved),
    )
    if bench.empty:
        raise ValueError(f"no stored prices for benchmark {bench_ticker!r} as of {cutoff}")
    sessions = ticker_sessions(prices, resolved)
    out = compute_relatives(prices, bench, price=price, sessions=sessions)
    out.insert(2, "sector", out["ticker"].map(read_sectors(settings=resolved)))
    if first is not None:
        out = out[out["date"] >= first].reset_index(drop=True)
    return pit.guard(out, cutoff, what="read_relatives")


# ---------------------------------------------------------------------------
# Sector demeaning
# ---------------------------------------------------------------------------


def _sector_keys(panel: pd.DataFrame, sectors: Mapping[str, str] | None, sector: str) -> pd.Series:
    if sectors is not None:
        if "ticker" not in panel.columns:
            raise KeyError("panel has no 'ticker' column to look sectors up by")
        keys = panel["ticker"].map(sectors)
    elif sector in panel.columns:
        keys = panel[sector]
    else:
        raise KeyError(f"panel has no {sector!r} column; pass sectors=")
    return keys.where(keys.notna() & (keys.astype(str).str.strip() != ""))


def sector_demean(
    panel: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    sectors: Mapping[str, str] | None = None,
    sector: str = "sector",
    by: str = "date",
    min_names: int = MIN_SECTOR_NAMES,
) -> pd.DataFrame:
    """`panel` with `<f>_sec = f - mean(f over the same sector and date)` for each `f`.

    `columns` defaults to the :data:`SECTOR_FEATURES` present. Sectors come
    from the `sector` column, or from `sectors` (`{ticker: sector}`) when
    given. A row with no sector, no value, or whose sector has fewer than
    `min_names` valued names that date gets NaN. Rows and order are kept.
    """
    if by not in panel.columns:
        raise KeyError(f"panel has no {by!r} column")
    if min_names < 2:
        raise ValueError(f"min_names must be at least 2, got {min_names}")
    if columns is None:
        columns = [c for c in SECTOR_FEATURES if c in panel.columns]
        if not columns:
            raise ValueError("panel has none of the default SECTOR_FEATURES; pass columns=")
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing column(s): {', '.join(missing)}")

    keys = _sector_keys(panel, sectors, sector)
    groups = [panel[by], keys]
    out = panel.copy()
    for col in columns:
        values = pd.to_numeric(panel[col], errors="raise").astype(float)
        values = values.where(np.isfinite(values) & keys.notna())
        grouped = values.groupby(groups, sort=False, dropna=False)
        count = grouped.transform("count")
        out[sector_column(col)] = (values - grouped.transform("mean")).where(count >= min_names)
    return out


def sector_share(
    panel: pd.DataFrame,
    column: str,
    *,
    sectors: Mapping[str, str] | None = None,
    sector: str = "sector",
    by: str = "date",
) -> pd.Series:
    """η² per date: the share of `column`'s cross-sectional variance sector explains.

    Between-sector sum of squares over total, over the names with a value and
    a sector. 0 means the sector says nothing about the value; 1 means the
    value is the sector. NaN on a date with no dispersion.
    """
    keys = _sector_keys(panel, sectors, sector)
    values = pd.to_numeric(panel[column], errors="raise").astype(float)
    keep = values.notna() & np.isfinite(values) & keys.notna()
    frame = pd.DataFrame({"date": panel[by][keep], "sector": keys[keep], "x": values[keep]})
    frame["grand"] = frame.groupby("date")["x"].transform("mean")
    frame["group"] = frame.groupby(["date", "sector"])["x"].transform("mean")
    frame["total"] = (frame["x"] - frame["grand"]) ** 2
    frame["between"] = (frame["group"] - frame["grand"]) ** 2
    sums = frame.groupby("date")[["total", "between"]].sum()
    share = sums["between"] / sums["total"].where(sums["total"] > 0)
    return share.rename(f"{column}_sector_share")


def sector_mean_rank(
    panel: pd.DataFrame,
    column: str,
    *,
    sectors: Mapping[str, str] | None = None,
    sector: str = "sector",
    by: str = "date",
) -> pd.DataFrame:
    """Each sector's mean cross-sectional rank of `column`, per date (dates by sectors).

    Ranks are day 29's `(r - 1) / (n - 1)` over every valued name; a sector
    that ranks no differently from the rest sits at 0.5.
    """
    keys = _sector_keys(panel, sectors, sector)
    values = pd.to_numeric(panel[column], errors="raise").astype(float)
    values = values.where(np.isfinite(values) & keys.notna())
    dates = panel[by]
    n = values.notna().groupby(dates).transform("sum")
    ranks = (values.groupby(dates).rank(method="average") - 1.0) / (n - 1.0).where(n > 1)
    frame = pd.DataFrame({"date": dates, "sector": keys, "rank": ranks}).dropna()
    return frame.pivot_table(index="date", columns="sector", values="rank", aggfunc="mean")
