"""The universe file: exactly 100 companies, and the shape everything assumes.

`config/universe.csv` is the one list the whole project turns around. Ingestion
loops it, the cross-sectional ranking on day 29 compares each name against the
other 99, and the sector column is what day 30 demeans against. A duplicate
ticker would silently double-weight a company in every ranking; a typo in a
sector name would split one sector into two thin ones. Neither is visible by
eye in a hundred-line file, so both are asserted here.

These tests read the real file the project ships — unlike the settings tests,
there is nothing to fake, because the file itself is the deliverable.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pytest

from finmgr.config import DEFAULT_SETTINGS_PATH, load_settings

#: The columns, in order. Downstream code indexes by name, but a reordered or
#: renamed header still means someone rewrote the file by hand.
EXPECTED_HEADER = ["ticker", "name", "exchange", "currency", "sector", "country"]

#: The eleven GICS sectors. Anything outside this set is a typo, not a sector:
#: "Healthcare" and "Health Care" would rank as two separate groups.
GICS_SECTORS = {
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
}

#: Venues carrying the ~40 European names the roadmap asks for: Madrid, Paris,
#: Xetra and Amsterdam — IBEX 35, CAC 40, DAX and AEX.
CORE_EUROPEAN_VENUES = {"XMAD", "XPAR", "XETR", "XAMS"}

US_VENUES = {"XNYS", "XNAS"}


def _universe_path() -> Path:
    """Where the settings file says the universe lives."""
    return load_settings(DEFAULT_SETTINGS_PATH).universe_path


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with _universe_path().open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# The acceptance criterion for day 7
# ---------------------------------------------------------------------------


def test_the_universe_has_exactly_one_hundred_rows(rows: list[dict[str, str]]) -> None:
    """Not "about a hundred": the count is fixed and everything downstream says 100."""
    assert len(rows) == 100


def test_no_ticker_appears_twice(rows: list[dict[str, str]]) -> None:
    """A duplicate would be ingested twice and weighted twice in every ranking."""
    tickers = [row["ticker"] for row in rows]
    duplicates = sorted(ticker for ticker, count in Counter(tickers).items() if count > 1)

    assert duplicates == []
    assert len(set(tickers)) == 100


def test_every_sector_is_represented_more_than_once(rows: list[dict[str, str]]) -> None:
    """A sector of one cannot be demeaned against itself (day 30)."""
    counts = Counter(row["sector"] for row in rows)
    singletons = sorted(sector for sector, count in counts.items() if count < 2)

    assert singletons == []


def test_at_least_eight_sectors_are_present(rows: list[dict[str, str]]) -> None:
    """Cross-sectional ranking needs something to compare across."""
    assert len({row["sector"] for row in rows}) >= 8


# ---------------------------------------------------------------------------
# The shape downstream code relies on
# ---------------------------------------------------------------------------


def test_the_settings_file_points_at_a_file_that_exists() -> None:
    assert _universe_path().is_file()


def test_the_header_is_exactly_the_six_expected_columns() -> None:
    with _universe_path().open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))

    assert header == EXPECTED_HEADER


def test_no_field_is_blank_or_padded(rows: list[dict[str, str]]) -> None:
    """Leading whitespace in a ticker is a Yahoo request that fails for no reason."""
    offenders = [
        (row["ticker"], column)
        for row in rows
        for column, value in row.items()
        if value is None or value != value.strip() or not value
    ]

    assert offenders == []


def test_sector_names_are_the_official_gics_ones(rows: list[dict[str, str]]) -> None:
    unknown = sorted({row["sector"] for row in rows} - GICS_SECTORS)

    assert unknown == []


def test_currencies_are_three_letter_iso_codes(rows: list[dict[str, str]]) -> None:
    """Day 18 converts every price to EUR and looks the pair up by these codes."""
    bad = sorted(
        {row["currency"] for row in rows if not re.fullmatch(r"[A-Z]{3}", row["currency"])}
    )

    assert bad == []


def test_exchanges_are_four_letter_mic_codes(rows: list[dict[str, str]]) -> None:
    """Day 17 maps the venue to an exchange calendar by its MIC."""
    bad = sorted(
        {row["exchange"] for row in rows if not re.fullmatch(r"X[A-Z]{3}", row["exchange"])}
    )

    assert bad == []


def test_tickers_look_like_yahoo_symbols(rows: list[dict[str, str]]) -> None:
    """Yahoo's own spelling: no spaces, uppercase, optional `.MC`-style suffix.

    Day 8 is what proves each one actually returns bars; this only catches the
    mechanical mistakes — a lowercase suffix, a stray space, a comma.
    """
    bad = sorted(
        row["ticker"]
        for row in rows
        if not re.fullmatch(r"[0-9A-Z]+(-[A-Z])?(\.[A-Z]{1,2})?", row["ticker"])
    )

    assert bad == []


# ---------------------------------------------------------------------------
# The mix the roadmap asked for
# ---------------------------------------------------------------------------


def test_the_regional_mix_is_roughly_forty_forty_twenty(rows: list[dict[str, str]]) -> None:
    """~40 US large caps, ~40 from IBEX/CAC/DAX/AEX, ~20 elsewhere."""
    venues = Counter(row["exchange"] for row in rows)
    american = sum(count for venue, count in venues.items() if venue in US_VENUES)
    european = sum(count for venue, count in venues.items() if venue in CORE_EUROPEAN_VENUES)

    assert american == 40
    assert european == 40
    assert len(rows) - american - european == 20


def test_the_rest_of_the_world_is_actually_spread_out(rows: list[dict[str, str]]) -> None:
    """Otherwise "elsewhere" quietly becomes twenty more names on one exchange."""
    elsewhere = [row for row in rows if row["exchange"] not in US_VENUES | CORE_EUROPEAN_VENUES]

    assert len({row["country"] for row in elsewhere}) >= 8
