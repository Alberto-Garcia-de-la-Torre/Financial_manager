"""The feature panel: every feature, every name, every date, in one parquet file.

    from finmgr.features.panel import build_panel, read_panel

    build_panel(as_of="2026-10-15", full=True)     # what `finmgr features --full` does
    panel = read_panel()                            # indexed by (date, ticker)
    panel.xs(date(2026, 10, 15), level="date")      # the 100 on one day

`finmgr features` writes `data/features/panel.parquet`, indexed by
`(date, ticker)`, holding everything days 23-30 built:

* the raw features that compare across tickers — returns, momentum, risk,
  trend ratios, oscillators, euro liquidity, beta and residual momentum, and
  the sector-demeaned `_sec` columns;
* each ticker's `sector` and the `tradable` flag;
* for every feature in :data:`~finmgr.features.crosssection.FEATURE_COLUMNS`,
  its per-date `_rank` (uniform on [0, 1]) and `_z`.

**The registry is the schema.** `finmgr/features/registry.yaml` documents every
column of the panel — name, kind, formula, window in sessions and the expected
direction of its relation to forward returns — in the panel's column order.
Every build checks that the computed columns are exactly the registry's
(:func:`check_registry`), and refuses to write a panel that does not match: a
feature added to a module without a registry entry, or an entry for a column no
module makes, fails the build. The columns the modules compute but the panel
deliberately leaves out — levels in price units that a later split rescales, and
inputs — are listed in :data:`EXCLUDED` with the reason, so that list is checked
too.

**Built incrementally.** Without `--full`, a build appends to the existing
panel: the dates after its last one, plus the last :data:`OVERLAP_DAYS`
calendar days again, so a bar ingested late (a US close that arrived after the
European run) is picked up on the next build. Those dates are replaced whole,
since ranks and sector means are per date. The features themselves are always
computed on each ticker's full stored history — the EMAs, RSI, MACD and ATR are
recursions that start at the first bar — so an incrementally built row is
identical to the same row of a full rebuild (`tests/test_panel.py` checks it).
A panel written under another registry, or one dated after `as_of`, is not
appended to; `--full` rebuilds from scratch.

**Point in time.** Nothing after `as_of` is read; the parquet metadata records
the `as_of`, the build time, the build mode and the registry's hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from finmgr.config import Settings, load_settings
from finmgr.data import pit, store
from finmgr.data.benchmarks import benchmarks_dir, primary_benchmark, read_total_return
from finmgr.features.crosssection import FEATURE_COLUMNS, RANK_SUFFIX, Z_SUFFIX, normalize
from finmgr.features.liquidity import compute_liquidity
from finmgr.features.momentum import compute_momentum
from finmgr.features.oscillators import compute_oscillators
from finmgr.features.relative import compute_relatives, read_sectors, sector_demean
from finmgr.features.returns import compute_returns, ticker_sessions
from finmgr.features.trend import compute_trend
from finmgr.features.volatility import compute_volatility

INDEX: tuple[str, str] = ("date", "ticker")

PANEL_FILE = "panel.parquet"

#: The registry ships inside the package, beside the code that computes the columns.
REGISTRY_PATH = Path(__file__).with_name("registry.yaml")

#: Calendar days at the tail of an existing panel an incremental build recomputes.
OVERLAP_DAYS = 10

KINDS = frozenset({"meta", "flag", "feature", "rank", "zscore"})
DIRECTIONS = frozenset({"positive", "negative", "neutral"})
REGISTRY_FIELDS = ("name", "kind", "source", "formula", "window", "direction")

#: Columns the feature modules compute that the panel leaves out, and why.
EXCLUDED: dict[str, str] = {
    "close": "input: raw close in the listing currency",
    "adj_close": "input: back-adjusted close, price units",
    "close_eur": "input: raw close in EUR",
    "adj_close_eur": "input: back-adjusted close in EUR, price units",
    "high": "input",
    "low": "input",
    "volume": "input: share count, jumps on a split",
    "atr_14": "price units; atr_14_pct is the comparable column",
    "sma_20": "price units; dist_sma_20 / slope_sma_20 are the comparable columns",
    "sma_50": "price units; dist_sma_50 / slope_sma_50 are the comparable columns",
    "sma_200": "price units; dist_sma_200 / slope_sma_200 are the comparable columns",
    "ema_20": "price units; dist_ema_20 / slope_ema_20 are the comparable columns",
    "ema_50": "price units; dist_ema_50 / slope_ema_50 are the comparable columns",
    "ema_200": "price units; dist_ema_200 / slope_ema_200 are the comparable columns",
    "macd": "price units; macd_pct is the comparable column",
    "macd_signal": "price units; macd_signal_pct is the comparable column",
    "macd_hist": "price units; macd_hist_pct is the comparable column",
    "eur_volume": "one session's turnover, noisy; adv_eur_21 and volume_z_63 carry it",
    "obv": "level depends on where the sum started; obv_flow_21 is the comparable column",
}


class RegistryMismatch(ValueError):
    """The panel's columns and the registry's entries differ."""


@dataclass(frozen=True)
class BuildResult:
    path: Path
    mode: str  # "full" or "incremental"
    rows: int
    columns: int
    first: date | None
    last: date | None
    written_from: date | None  # first date (re)computed by this build


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def load_registry(path: Path | str = REGISTRY_PATH) -> list[dict[str, Any]]:
    """The registry's entries, in panel column order, each validated."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("columns") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a non-empty 'columns' list")
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        where = f"{path}: entry {i}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} is not a mapping")
        missing = [f for f in REGISTRY_FIELDS if f not in entry]
        extra = set(entry) - set(REGISTRY_FIELDS)
        if missing or extra:
            raise ValueError(f"{where} ({entry.get('name')}): missing {missing}, unknown {extra}")
        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"{where}: name must be a non-empty string")
        if name in seen:
            raise ValueError(f"{where}: duplicate column {name!r}")
        seen.add(name)
        if entry["kind"] not in KINDS:
            raise ValueError(f"{where} ({name}): kind {entry['kind']!r} not in {sorted(KINDS)}")
        if entry["direction"] not in DIRECTIONS:
            raise ValueError(
                f"{where} ({name}): direction {entry['direction']!r} not in {sorted(DIRECTIONS)}"
            )
        window = entry["window"]
        if window is not None and not (isinstance(window, int) and window > 0):
            raise ValueError(f"{where} ({name}): window must be a positive integer or null")
        if not isinstance(entry["formula"], str) or not entry["formula"].strip():
            raise ValueError(f"{where} ({name}): formula must be a non-empty string")
    _check_derived(entries, path)
    return entries


def _check_derived(entries: list[dict[str, Any]], path: Path | str) -> None:
    """Every `_rank` / `_z` entry names a registered base and inherits its direction and window."""
    by_name = {e["name"]: e for e in entries}
    for entry in entries:
        suffix = {"rank": RANK_SUFFIX, "zscore": Z_SUFFIX}.get(entry["kind"])
        if suffix is None:
            continue
        name = entry["name"]
        if not name.endswith(suffix):
            raise ValueError(f"{path}: {entry['kind']} column {name!r} must end in {suffix!r}")
        base = by_name.get(name[: -len(suffix)])
        if base is None or base["kind"] != "feature":
            raise ValueError(f"{path}: {name!r} has no registered base feature")
        for field in ("direction", "window"):
            if entry[field] != base[field]:
                raise ValueError(f"{path}: {name!r} {field} differs from {base['name']!r}")


def registry_columns(path: Path | str = REGISTRY_PATH) -> list[str]:
    """Every column the panel holds, index included, in order."""
    return [e["name"] for e in load_registry(path)]


def registry_hash(path: Path | str = REGISTRY_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def check_registry(columns: Sequence[str], path: Path | str = REGISTRY_PATH) -> None:
    """Raise :class:`RegistryMismatch` unless `columns` are exactly the registry's, in order."""
    expected = registry_columns(path)
    actual = list(columns)
    if actual == expected:
        return
    missing = [c for c in expected if c not in actual]
    unregistered = [c for c in actual if c not in expected]
    lines = []
    if unregistered:
        lines.append(f"not in the registry: {', '.join(unregistered)}")
    if missing:
        lines.append(f"registered but absent: {', '.join(missing)}")
    if not lines:
        lines.append("same columns, different order")
    raise RegistryMismatch("panel and registry differ — " + "; ".join(lines))


# ---------------------------------------------------------------------------
# Computing
# ---------------------------------------------------------------------------


def compute_panel(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    sectors: dict[str, str],
    sessions: dict[str, Sequence[date]] | None = None,
    start: date | None = None,
    registry: Path | str = REGISTRY_PATH,
) -> pd.DataFrame:
    """Every registered column for the rows of `prices` dated `start` or later.

    `prices` is long, one row per `(ticker, date)`, with the columns of
    :func:`~finmgr.data.benchmarks.read_total_return` plus the bars' `high`,
    `low` and `volume`, and the whole history each ticker has: the recursive
    features start at its first row. `benchmark` is the benchmark's total
    return frame. Returns a frame indexed by `(date, ticker)`, sorted, with the
    registry's columns in its order; raises :class:`RegistryMismatch` if the
    modules compute anything the registry and :data:`EXCLUDED` do not account for.
    """
    keys = ["ticker", "date"]
    parts = [
        compute_returns(prices, sessions=sessions),
        compute_momentum(prices, sessions=sessions),
        compute_volatility(prices, sessions=sessions),
        compute_trend(prices, sessions=sessions),
        compute_oscillators(prices, sessions=sessions),
        compute_liquidity(prices, sessions=sessions),
        compute_relatives(prices, benchmark, sessions=sessions),
    ]
    merged = prices.sort_values(keys, kind="stable").reset_index(drop=True)
    for part in parts:
        new = [c for c in part.columns if c not in merged.columns]
        merged = merged.merge(part[[*keys, *new]], on=keys, how="left", validate="1:1")
    if start is not None:
        merged = merged[merged["date"] >= start].reset_index(drop=True)
    merged.insert(2, "sector", merged["ticker"].map(sectors))
    merged = sector_demean(merged)
    merged = normalize(merged, [c for c in FEATURE_COLUMNS if c in merged.columns])

    order = registry_columns(registry)
    computed = [c for c in merged.columns if c not in EXCLUDED]
    unaccounted = [c for c in computed if c not in order]
    absent = [c for c in order if c not in merged.columns]
    if unaccounted or absent:
        check_registry(computed, registry)  # raises, with the difference spelled out
    out = merged[order].set_index(list(INDEX)).sort_index()
    return out


def _read_inputs(
    as_of: date, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[date]]]:
    closes = read_total_return(None, as_of=as_of, settings=settings)
    bars = store.read_bars(
        None, as_of=as_of, settings=settings, root=settings.data_dir / "bars" / "daily"
    )
    prices = closes.merge(
        bars[["ticker", "date", "high", "low", "volume"]], on=["ticker", "date"], how="left"
    )
    bench_ticker = primary_benchmark(settings=settings)
    bench = read_total_return(
        bench_ticker, as_of=as_of, settings=settings, root=benchmarks_dir(settings=settings)
    )
    if bench.empty:
        raise ValueError(f"no stored prices for benchmark {bench_ticker!r} as of {as_of}")
    return prices, bench, ticker_sessions(prices, settings)


# ---------------------------------------------------------------------------
# Storing
# ---------------------------------------------------------------------------


def panel_path(*, settings: Settings | None = None) -> Path:
    return (settings or load_settings()).data_dir / "features" / PANEL_FILE


def panel_metadata(path: Path) -> dict[str, str]:
    """The `finmgr.*` keys of a written panel's parquet metadata."""
    meta = pq.read_schema(path).metadata or {}
    return {
        k.decode().removeprefix("finmgr."): v.decode()
        for k, v in meta.items()
        if k.startswith(b"finmgr.")
    }


def read_panel(
    *,
    columns: Sequence[str] | None = None,
    settings: Settings | None = None,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """The stored panel, indexed by `(date, ticker)`."""
    target = Path(path) if path is not None else panel_path(settings=settings)
    if not target.exists():
        raise FileNotFoundError(f"no feature panel at {target}; run `finmgr features --full`")
    frame = pd.read_parquet(target, columns=list(columns) if columns is not None else None)
    return frame


def _write(frame: pd.DataFrame, path: Path, meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=True)
    merged = dict(table.schema.metadata or {})
    merged.update({f"finmgr.{k}".encode(): v.encode() for k, v in meta.items()})
    table = table.replace_schema_metadata(merged)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def build_panel(
    *,
    as_of: pit.AsOf,
    full: bool = False,
    settings: Settings | None = None,
    path: Path | str | None = None,
    registry: Path | str = REGISTRY_PATH,
) -> BuildResult:
    """Build (`full=True`) or extend the panel on disk; see the module docstring."""
    cutoff = pit.resolve_as_of(as_of)
    resolved = settings or load_settings()
    target = Path(path) if path is not None else panel_path(settings=resolved)
    digest = registry_hash(registry)

    existing: pd.DataFrame | None = None
    start: date | None = None
    if not full and target.exists():
        meta = panel_metadata(target)
        if meta.get("registry") != digest:
            raise RegistryMismatch(
                f"{target} was built under another registry; rebuild with `finmgr features --full`"
            )
        existing = read_panel(path=target)
        check_registry([*existing.index.names, *existing.columns], registry)
        last = existing.index.get_level_values("date").max()
        if last > cutoff:
            raise ValueError(
                f"{target} runs to {last}, after as_of {cutoff}; "
                "rebuild with `finmgr features --full`"
            )
        start = last - timedelta(days=OVERLAP_DAYS)

    prices, bench, sessions = _read_inputs(cutoff, resolved)
    fresh = compute_panel(
        prices,
        bench,
        sectors=read_sectors(settings=resolved),
        sessions=sessions,
        start=start,
        registry=registry,
    )
    if existing is not None:
        kept = existing[existing.index.get_level_values("date") < start]
        fresh = pd.concat([kept, fresh]).sort_index()
    check_registry([*fresh.index.names, *fresh.columns], registry)
    pit.guard(fresh.reset_index()[["date"]], cutoff, what="build_panel")

    mode = "incremental" if existing is not None else "full"
    _write(
        fresh,
        target,
        {
            "as_of": cutoff.isoformat(),
            "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "build": mode,
            "registry": digest,
            "excluded": json.dumps(sorted(EXCLUDED)),
        },
    )
    dates = fresh.index.get_level_values("date")
    return BuildResult(
        path=target,
        mode=mode,
        rows=len(fresh),
        columns=len(fresh.columns) + len(INDEX),
        first=dates.min() if len(fresh) else None,
        last=dates.max() if len(fresh) else None,
        written_from=start,
    )
