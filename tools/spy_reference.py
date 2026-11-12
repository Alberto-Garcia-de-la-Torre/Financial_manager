"""Freeze the SPY closes the day-37 metrics tests check against a published reference.

    .venv/bin/python tools/spy_reference.py      # needs SPY in the local benchmark store

The acceptance check for day 37 runs SPY buy-and-hold through the backtest
engine and :mod:`finmgr.backtest.metrics`, and compares the result with SPY's
*published* calendar-year total returns (see ``PUBLISHED`` in
``tests/test_metrics.py`` for the figures and their source). ``data/`` is not
in git, so the test reads a frozen copy of the one input it needs: SPY's
dividend-adjusted close in USD (Yahoo's ``Adj Close``, as day 22 stored it),
from the last session of 2012 to the last session of 2024. This script
rewrites ``tests/fixtures/metrics/spy_adj_close.csv`` from the store.
"""

from __future__ import annotations

from pathlib import Path

from finmgr.data.benchmarks import read_benchmarks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "metrics" / "spy_adj_close.csv"
START, END = "2012-12-31", "2024-12-31"


def main() -> None:
    frame = read_benchmarks("SPY", start=START, end=END, as_of=END)
    frame = frame[["date", "adj_close"]].dropna().sort_values("date")
    if str(frame["date"].iloc[0]) != START or str(frame["date"].iloc[-1]) != END:
        raise SystemExit(f"store does not cover {START}..{END} for SPY")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        fh.write(
            f"# SPY dividend-adjusted close, USD, {START}..{END}, from the day-22 benchmark "
            "store (Yahoo Adj Close); regenerate with tools/spy_reference.py\n"
        )
        frame.to_csv(fh, index=False, float_format="%.6f", lineterminator="\n")
    print(f"wrote {len(frame)} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
