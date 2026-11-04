"""Regenerate the TA-Lib reference values the day-27 oscillator tests check against.

    python tools/oscillator_reference.py        # needs `pip install ta-lib numpy`

TA-Lib is *not* a dependency of finmgr: the oscillators in
:mod:`finmgr.features.oscillators` are written by hand, and this script is the
independent implementation they are checked against. It imports nothing from
finmgr. Run it in any throwaway environment with TA-Lib installed; it rewrites
``tests/fixtures/oscillators/talib_reference.csv``, which is committed so the
test suite needs no TA-Lib.

The input is a deterministic synthetic OHLC series (a seeded random walk with
a flat stretch, a one-way rally and a one-way sell-off, so RSI visits 0 and 100
territory and the stochastic's range collapses), 400 bars long.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import talib

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "oscillators" / "talib_reference.csv"
N = 400


def synthetic_bars(n: int = N, seed: int = 27) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.015, n)
    steps[120:135] = 0.01  # fifteen straight up days: RSI near 100
    steps[220:235] = -0.01  # fifteen straight down days: RSI near 0
    steps[300:320] = 0.0  # twenty unchanged closes: no change at all to average
    close = 100 * np.exp(np.cumsum(steps))
    close = np.round(close, 4)  # prices as a feed would print them
    spread = np.abs(rng.normal(0.0, 0.008, n))
    high = np.round(close * (1 + spread + np.abs(rng.normal(0.0, 0.004, n))), 4)
    low = np.round(close * (1 - spread - np.abs(rng.normal(0.0, 0.004, n))), 4)
    return {"high": high, "low": low, "close": close}


def main() -> int:
    bars = synthetic_bars()
    h, lo, c = bars["high"], bars["low"], bars["close"]
    rsi = talib.RSI(c, timeperiod=14)
    macd, signal, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    fastk, fastd = talib.STOCHF(h, lo, c, fastk_period=14, fastd_period=3, fastd_matype=0)
    upper, _middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    pctb = (c - lower) / (upper - lower)
    columns = {
        "high": h,
        "low": lo,
        "close": c,
        "rsi_14": rsi,
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": hist,
        "stoch_k": fastk,
        "stoch_d": fastd,
        "bb_pctb": pctb,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        handle.write(
            f"# TA-Lib {talib.__ta_version__.decode().split()[0]} via ta-lib {talib.__version__}; "
        )
        handle.write("regenerate with tools/oscillator_reference.py\n")
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(N):
            writer.writerow("" if np.isnan(v[i]) else repr(float(v[i])) for v in columns.values())
    print(f"wrote {OUT.relative_to(ROOT)} ({N} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
