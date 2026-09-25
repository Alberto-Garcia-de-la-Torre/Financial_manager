"""Saved yfinance responses, and the script that captured them.

`tests/test_fetch.py` must pass without touching the network, so the raw
material it feeds :func:`finmgr.data.fetch.to_bars` is kept on disk under
`tests/fixtures/yahoo/`: one CSV holding exactly what
`yf.Ticker(t).history(..., auto_adjust=False)` returned, plus a JSON sidecar
recording where it came from.

The CSV is `DataFrame.to_csv()` output and nothing else — readable in a diff,
so a reviewer can see the prices the tests assert on. The sidecar carries what
a CSV cannot: the exchange timezone of the index, its resolution, the column
dtypes, and when and with which yfinance version the capture happened.
:func:`load_raw` puts those back, and the capture script below refuses to write
a fixture that does not reload identical to the response it saw.

To refresh a fixture, or add one:

    python tests/yahoo_fixtures.py AAPL 2020-08-24 2020-09-04

Both dates are inclusive, the same way :func:`finmgr.data.fetch.fetch_daily`
reads them. Capturing hits the network on purpose; running the tests does not.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "yahoo"

#: The two fixtures the tests use, and why each one is worth keeping.
#:
#: AAPL spans the 4:1 split of 2020-08-31, and SAN.MC spans a €0.10 dividend
#: on 2024-10-30 — so between them the saved responses carry both kinds of
#: corporate action, and a `Close` that differs from `Adj Close` proves the
#: capture really was made with `auto_adjust=False`. SAN.MC also trades in
#: Europe/Madrid, where every bar is stamped midnight +01:00: read through UTC
#: it would be dated the previous day, which is the mistake the store's date
#: rule exists to prevent.
FIXTURES: dict[str, tuple[str, str, str]] = {
    "AAPL_2020-08-24_2020-09-04": ("AAPL", "2020-08-24", "2020-09-04"),
    "SAN.MC_2024-10-28_2024-11-11": ("SAN.MC", "2024-10-28", "2024-11-11"),
}


def fixture_name(ticker: str, start: str | date, end: str | date) -> str:
    """The stem the two files of one fixture share."""
    return f"{ticker}_{start}_{end}"


def load_raw(name: str) -> pd.DataFrame:
    """Reload a saved response as the frame yfinance handed over.

    Same columns, same dtypes, same timezone-aware index — so a test that
    feeds this to the conversion code is exercising it on real Yahoo output
    rather than on a tidied-up imitation of it.
    """
    meta = load_meta(name)
    frame = pd.read_csv(FIXTURE_DIR / f"{name}.csv", index_col=0)
    stamps = pd.to_datetime(frame.index, utc=True, format="ISO8601")
    frame.index = pd.DatetimeIndex(
        stamps.tz_convert(meta["timezone"]).as_unit(meta["index_unit"]),
        name=meta["index_name"],
    )
    return frame.astype(meta["dtypes"])


def load_meta(name: str) -> dict:
    """The sidecar: provenance, timezone, dtypes."""
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Capturing (the only part of this file that uses the network)
# ---------------------------------------------------------------------------


def capture(ticker: str, start: str, end: str) -> Path:
    """Download one window and save it as a fixture. Returns the CSV path."""
    import yfinance as yf

    # `end` is exclusive at Yahoo and inclusive here, exactly as fetch_daily
    # treats it, so the fixture window matches the name on the file.
    exclusive = date.fromisoformat(end) + timedelta(days=1)
    yf.config.debug.hide_exceptions = False
    frame = yf.Ticker(ticker).history(
        start=start,
        end=exclusive.isoformat(),
        interval="1d",
        auto_adjust=False,
    )
    if frame.empty:
        raise SystemExit(f"{ticker}: Yahoo returned no bars for {start}..{end}")

    name = fixture_name(ticker, start, end)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIXTURE_DIR / f"{name}.csv"
    frame.to_csv(csv_path)
    (FIXTURE_DIR / f"{name}.json").write_text(
        json.dumps(
            {
                "ticker": ticker,
                "start": start,
                "end": end,
                "call": (
                    f"yf.Ticker({ticker!r}).history(start={start!r}, "
                    f"end={exclusive.isoformat()!r}, interval='1d', auto_adjust=False)"
                ),
                "captured_at": datetime.now(ZoneInfo("Europe/Madrid")).isoformat(
                    timespec="seconds"
                ),
                "yfinance": yf.__version__,
                "pandas": pd.__version__,
                "rows": len(frame),
                "timezone": str(frame.index.tz),
                "index_name": frame.index.name,
                "index_unit": frame.index.unit,
                "dtypes": {str(c): str(dtype) for c, dtype in frame.dtypes.items()},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # A fixture that does not survive its own round trip is worse than no
    # fixture: the tests would assert on numbers Yahoo never sent.
    pd.testing.assert_frame_equal(load_raw(name), frame)
    return csv_path


def main(argv: list[str]) -> int:
    if len(argv) not in (0, 3):
        print(__doc__)
        return 2
    wanted = [tuple(argv)] if argv else list(FIXTURES.values())
    for ticker, start, end in wanted:
        path = capture(ticker, start, end)
        print(f"captured {path} ({len(load_raw(path.stem))} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
