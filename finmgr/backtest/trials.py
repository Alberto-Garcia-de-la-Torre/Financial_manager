"""Day 39: the running count of strategy variants tested.

Every variant tried and compared is another lottery ticket: the best of `n`
backtests looks better than any one of them would on fresh data, and the
more tickets, the more it flatters. The ledger is what keeps that count.

`docs/trials.csv` holds one row per **variant** — a strategy family plus the
parameters that define it (scores, top-K, holding period, costs, window
start). A variant's id is a hash of exactly those, so:

* rerunning the same variant (new data, a fixed bug) updates its row —
  latest Sharpe and interval, `runs` + 1 — and does not add to the count;
* changing *any* defining parameter is a new variant and a new row, even if
  it is thrown away a minute later. That is the point: throwing it away does
  not un-try it.

Rows are never deleted. The window's end date is deliberately *not* part of
the id: extending the data is rerunning, not choosing.

The count feeds :func:`finmgr.backtest.bootstrap.familywise_level`: the best
variant's Sharpe is quoted at the Bonferroni level for `n` variants, so its
interval widens as the count grows.

Tests and scratch runs pass their own `path`; only the project's own tools
write to the default ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd

from finmgr.config import PROJECT_ROOT

LEDGER = PROJECT_ROOT / "docs" / "trials.csv"
COLUMNS = [
    "id",
    "family",
    "params",
    "first_logged",
    "last_logged",
    "runs",
    "sharpe",
    "sharpe_low",
    "sharpe_high",
    "level",
    "source",
]


def _canonical(params: Mapping[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def variant_id(family: str, params: Mapping[str, object]) -> str:
    """Stable 12-hex id for a family plus its defining parameters."""
    payload = f"{family}\n{_canonical(params)}".encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def read_ledger(path: Path = LEDGER) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    text = {"id": str, "family": str, "params": str, "first_logged": str, "last_logged": str}
    numbers = dict.fromkeys(("sharpe", "sharpe_low", "sharpe_high", "level"), "float64")
    frame = pd.read_csv(path, dtype={**text, **numbers, "source": str, "runs": "int64"})
    frame["source"] = frame["source"].fillna("")
    return frame.reindex(columns=COLUMNS)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def record(
    family: str,
    params: Mapping[str, object],
    *,
    sharpe: float,
    low: float = math.nan,
    high: float = math.nan,
    level: float = math.nan,
    source: str = "",
    today: date | None = None,
    path: Path = LEDGER,
) -> str:
    """Log one run of a variant; returns its id. New variants add a row, reruns update theirs."""
    day = (today or date.today()).isoformat()
    vid = variant_id(family, params)
    ledger = read_ledger(path)
    latest = {
        "last_logged": day,
        "sharpe": round(float(sharpe), 6),
        "sharpe_low": round(float(low), 6),
        "sharpe_high": round(float(high), 6),
        "level": level,
        "source": source,
    }
    hit = ledger["id"] == vid
    if hit.any():
        row = ledger.index[hit][0]
        for key, value in latest.items():
            ledger.at[row, key] = value
        ledger.at[row, "runs"] = int(ledger.at[row, "runs"]) + 1
    else:
        new = {
            "id": vid,
            "family": family,
            "params": _canonical(params),
            "first_logged": day,
            "runs": 1,
            **latest,
        }
        ledger = pd.concat([ledger, pd.DataFrame([new], columns=COLUMNS)], ignore_index=True)
    _write(ledger, path)
    return vid


def count_variants(path: Path = LEDGER) -> int:
    """How many distinct variants have ever been tested."""
    return int(read_ledger(path)["id"].nunique())
