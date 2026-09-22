"""Reading `config/universe.csv` into typed rows.

The universe file is the one list the whole project turns around, and from
here on several stages loop it — validation today, ingestion on day 11,
cross-sectional ranking later. They all read it through this module rather
than each opening the CSV themselves, so the column names live in one place.

    from finmgr.data.universe import load_universe

    for company in load_universe():
        print(company.ticker, company.exchange)

`tests/test_universe.py` asserts the file's contents; this module only cares
about its shape.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from finmgr.config import Settings, load_settings

#: The columns of `config/universe.csv`, in order.
UNIVERSE_COLUMNS = ("ticker", "name", "exchange", "currency", "sector", "country")


@dataclass(frozen=True, slots=True)
class Company:
    """One row of the universe file."""

    ticker: str
    name: str
    exchange: str
    currency: str
    sector: str
    country: str


def universe_path(path: Path | str | None = None, *, settings: Settings | None = None) -> Path:
    """Resolve which universe file to read: argument, then settings."""
    if path is not None:
        return Path(path).expanduser()
    return (settings or load_settings()).universe_path


def load_universe(
    path: Path | str | None = None, *, settings: Settings | None = None
) -> list[Company]:
    """Load the universe in file order.

    Raises rather than returning a half-usable list: a missing column or a
    duplicated ticker breaks something downstream in a way that is far harder
    to diagnose once it has been ingested, ranked and backtested.
    """
    resolved = universe_path(path, settings=settings)
    if not resolved.is_file():
        raise FileNotFoundError(f"universe file not found: {resolved}")

    with resolved.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header != UNIVERSE_COLUMNS:
            raise ValueError(
                f"{resolved}: expected columns {list(UNIVERSE_COLUMNS)}, got {list(header)}"
            )
        companies = [
            Company(**{column: (row[column] or "").strip() for column in UNIVERSE_COLUMNS})
            for row in reader
        ]

    blank = [company for company in companies if not company.ticker]
    if blank:
        raise ValueError(f"{resolved}: {len(blank)} row(s) have an empty ticker")

    duplicates = sorted(
        ticker for ticker, count in Counter(c.ticker for c in companies).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{resolved}: duplicate tickers {duplicates}")

    return companies


def tickers(path: Path | str | None = None, *, settings: Settings | None = None) -> list[str]:
    """Just the Yahoo symbols, in file order."""
    return [company.ticker for company in load_universe(path, settings=settings)]
