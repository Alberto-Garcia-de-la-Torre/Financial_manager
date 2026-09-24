"""The bar store: schema, partition layout, and the write/read round trip.

The one that matters is `test_round_trip_preserves_dtypes` — day 9's
acceptance criterion. Everything else exists because a storage layer that is
only correct for the frame it was handed last is not a storage layer.

Nothing here touches `data/`: every call passes `root=tmp_path`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from finmgr.config import Settings
from finmgr.data.store import (
    BAR_ARROW_SCHEMA,
    BAR_COLUMNS,
    BAR_DTYPES,
    bars_root,
    conform_bars,
    empty_bars,
    partition_path,
    read_bars,
    stored_tickers,
    write_bars,
)

TICKERS = ("AAPL", "SAN.MC", "BRK-B")


def synthetic_bars(
    tickers: tuple[str, ...] = TICKERS,
    sessions: int = 40,
    start: str = "2024-01-02",
    seed: int = 20260919,
) -> pd.DataFrame:
    """A plausible OHLCV panel in the canonical schema.

    Prices follow a small random walk and the bars obey low <= open/close <=
    high, so the frame is not just dtype-correct but shaped like real data.

    Built already in canonical order — tickers sorted, dates ascending — so a
    round trip can be compared against it without sorting either side through
    the code under test.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=sessions)
    rows = []
    for position, ticker in enumerate(sorted(tickers)):
        close = 100.0 * (1.0 + position) * np.cumprod(1.0 + rng.normal(0, 0.012, sessions))
        open_ = close * (1.0 + rng.normal(0, 0.004, sessions))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.005, sessions)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.005, sessions)))
        rows.append(
            pd.DataFrame(
                {
                    "ticker": pd.Series([ticker] * sessions, dtype=BAR_DTYPES["ticker"]),
                    "date": pd.Series(dates.date, dtype=BAR_DTYPES["date"]),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": rng.integers(1e5, 1e7, sessions).astype("float64"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A throwaway bar store, laid out exactly as the real one."""
    return tmp_path / "bars" / "daily"


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def test_round_trip_preserves_dtypes(root: Path) -> None:
    """Day 9: a synthetic frame survives write/read with dtypes identical."""
    original = synthetic_bars()

    write_bars(original, root=root)
    restored = read_bars(root=root)

    assert list(restored.columns) == list(BAR_COLUMNS)
    assert restored.dtypes.to_dict() == original.dtypes.to_dict()
    pd.testing.assert_frame_equal(restored, original)


def test_round_trip_dtypes_match_the_declared_schema(root: Path) -> None:
    """And those dtypes are the ones the module promises, not merely stable."""
    write_bars(synthetic_bars(), root=root)
    restored = read_bars(root=root)

    assert {column: str(dtype) for column, dtype in restored.dtypes.items()} == BAR_DTYPES


def test_round_trip_normalises_sloppy_input(root: Path) -> None:
    """A caller's float32 prices and integer volume come back canonical.

    yfinance hands over datetime-indexed frames with integer volume; the
    schema is enforced on the way in rather than trusted on the way out.
    """
    original = synthetic_bars(sessions=10)
    sloppy = original.assign(
        date=original["date"].astype("object"),
        open=original["open"].astype("float32"),
        volume=original["volume"].astype("int64"),
        ticker=original["ticker"].astype("object"),
    )

    write_bars(sloppy, root=root)
    restored = read_bars(root=root)

    assert {column: str(dtype) for column, dtype in restored.dtypes.items()} == BAR_DTYPES
    # float32 -> float64 is not lossless, so compare on the columns that were
    # handed over intact plus the round-tripped date and ticker keys.
    pd.testing.assert_frame_equal(
        restored[["ticker", "date", "high", "low", "close", "volume"]],
        original[["ticker", "date", "high", "low", "close", "volume"]],
    )


# ---------------------------------------------------------------------------
# Layout and the file on disk
# ---------------------------------------------------------------------------


def test_layout_is_one_hive_partition_per_ticker(root: Path) -> None:
    write_bars(synthetic_bars(), root=root)

    assert sorted(path.name for path in root.iterdir()) == [f"ticker={t}" for t in sorted(TICKERS)]
    for ticker in TICKERS:
        assert partition_path(ticker, root=root) == root / f"ticker={ticker}" / "part.parquet"
        assert partition_path(ticker, root=root).is_file()
    assert stored_tickers(root=root) == sorted(TICKERS)


def test_parquet_files_carry_the_declared_arrow_schema(root: Path) -> None:
    """The physical types are stated, not inferred from whatever pandas sent."""
    write_bars(synthetic_bars(), root=root)

    schema = pq.read_schema(partition_path("AAPL", root=root))
    assert schema.names == list(BAR_COLUMNS)
    assert [schema.field(name).type for name in schema.names] == [
        BAR_ARROW_SCHEMA.field(name).type for name in schema.names
    ]


def test_ticker_column_is_stored_in_the_file_too(root: Path) -> None:
    """So one partition is readable on its own, without its directory name."""
    write_bars(synthetic_bars(), root=root)

    alone = pd.read_parquet(partition_path("SAN.MC", root=root))
    assert alone["ticker"].unique().tolist() == ["SAN.MC"]


def test_write_leaves_no_temporary_files_behind(root: Path) -> None:
    write_bars(synthetic_bars(), root=root)

    assert [path.name for path in root.rglob("*.tmp")] == []


def test_rewriting_a_ticker_replaces_its_partition(root: Path) -> None:
    """A partition is the whole truth for its ticker, never an append log."""
    write_bars(synthetic_bars(sessions=40), root=root)
    shorter = synthetic_bars(tickers=("AAPL",), sessions=5, seed=7)
    write_bars(shorter, root=root)

    assert len(read_bars("AAPL", root=root)) == 5
    assert len(read_bars("SAN.MC", root=root)) == 40
    pd.testing.assert_frame_equal(read_bars("AAPL", root=root), shorter)


def test_write_bars_returns_the_files_it_wrote(root: Path) -> None:
    written = write_bars(synthetic_bars(sessions=3), root=root)

    assert written == [partition_path(ticker, root=root) for ticker in sorted(TICKERS)]


# ---------------------------------------------------------------------------
# Reading: filters and empties
# ---------------------------------------------------------------------------


def test_read_filters_by_ticker(root: Path) -> None:
    write_bars(synthetic_bars(), root=root)

    one = read_bars("SAN.MC", root=root)
    several = read_bars(["AAPL", "BRK-B"], root=root)

    assert one["ticker"].unique().tolist() == ["SAN.MC"]
    assert several["ticker"].unique().tolist() == ["AAPL", "BRK-B"]


def test_read_filters_by_date_inclusively(root: Path) -> None:
    original = synthetic_bars()
    write_bars(original, root=root)

    first, last = dt.date(2024, 1, 10), dt.date(2024, 1, 19)
    window = read_bars(root=root, start=first, end=last)
    expected = original[original["date"].between(first, last)].reset_index(drop=True)

    pd.testing.assert_frame_equal(window, expected)
    assert window["date"].min() == first
    assert window["date"].max() == last


def test_read_accepts_iso_date_strings(root: Path) -> None:
    write_bars(synthetic_bars(), root=root)

    by_string = read_bars(root=root, start="2024-01-10", end="2024-01-19")
    by_date = read_bars(root=root, start=dt.date(2024, 1, 10), end=dt.date(2024, 1, 19))

    pd.testing.assert_frame_equal(by_string, by_date)


def test_read_is_sorted_by_ticker_then_date(root: Path) -> None:
    write_bars(synthetic_bars(), root=root)

    restored = read_bars(root=root)
    assert restored.equals(restored.sort_values(["ticker", "date"]).reset_index(drop=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"tickers": "NOPE"}, id="unknown-ticker"),
        pytest.param({"tickers": []}, id="no-tickers-asked-for"),
        pytest.param({"start": "2030-01-01"}, id="date-window-with-no-sessions"),
    ],
)
def test_reads_that_match_nothing_still_carry_the_schema(root: Path, kwargs: dict) -> None:
    """Callers should never have to special-case "this matched nothing"."""
    write_bars(synthetic_bars(tickers=("AAPL",), sessions=2), root=root)

    result = read_bars(root=root, **kwargs)

    assert result.empty
    assert {column: str(dtype) for column, dtype in result.dtypes.items()} == BAR_DTYPES


def test_reading_an_empty_store_carries_the_schema(root: Path) -> None:
    result = read_bars(root=root)

    assert result.empty
    assert {column: str(dtype) for column, dtype in result.dtypes.items()} == BAR_DTYPES


def test_empty_bars_matches_a_read_of_an_empty_store(root: Path) -> None:
    pd.testing.assert_frame_equal(read_bars(root=root), empty_bars())


def test_read_rejects_a_reversed_date_range(root: Path) -> None:
    write_bars(synthetic_bars(sessions=5), root=root)

    with pytest.raises(ValueError, match="after end"):
        read_bars(root=root, start="2024-02-01", end="2024-01-01")


def test_writing_nothing_writes_nothing(root: Path) -> None:
    assert write_bars(empty_bars(), root=root) == []
    assert not root.exists()


# ---------------------------------------------------------------------------
# What the schema refuses
# ---------------------------------------------------------------------------


def test_conform_orders_columns_and_rows() -> None:
    frame = synthetic_bars(sessions=4)
    shuffled = frame.sample(frac=1.0, random_state=1)[list(reversed(BAR_COLUMNS))]

    pd.testing.assert_frame_equal(conform_bars(shuffled), frame)


def test_conform_rejects_a_missing_column() -> None:
    with pytest.raises(ValueError, match=r"missing \['volume'\]"):
        conform_bars(synthetic_bars(sessions=2).drop(columns=["volume"]))


def test_conform_rejects_an_unexpected_column() -> None:
    """Day 18 adds `close_eur` by widening the schema here, not by sneaking it in."""
    with pytest.raises(ValueError, match=r"unexpected \['close_eur'\]"):
        conform_bars(synthetic_bars(sessions=2).assign(close_eur=1.0))


def test_conform_rejects_a_repeated_ticker_and_date() -> None:
    frame = synthetic_bars(tickers=("AAPL",), sessions=3)
    doubled = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)

    with pytest.raises(ValueError, match="repeat a"):
        conform_bars(doubled)


def test_conform_rejects_timezone_aware_dates() -> None:
    """The trap: converting a Tokyo close through UTC dates it a day early."""
    frame = synthetic_bars(tickers=("7203.T",), sessions=3)
    tz_aware = frame.assign(
        date=pd.to_datetime(frame["date"].astype("str")).dt.tz_localize("Asia/Tokyo")
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        conform_bars(tz_aware)


def test_conform_rejects_an_empty_ticker() -> None:
    frame = synthetic_bars(tickers=("AAPL",), sessions=2)

    with pytest.raises(ValueError, match="empty ticker"):
        conform_bars(frame.assign(ticker=pd.Series(["AAPL", "  "], dtype=BAR_DTYPES["ticker"])))


def test_conform_rejects_a_missing_date() -> None:
    frame = synthetic_bars(tickers=("AAPL",), sessions=2)
    holed = frame.assign(date=pd.Series([dt.date(2024, 1, 2), None], dtype=BAR_DTYPES["date"]))

    with pytest.raises(ValueError, match="no date"):
        conform_bars(holed)


def test_conform_keeps_suspicious_prices() -> None:
    """Quality is day 19's job; the store must not lose the evidence."""
    frame = synthetic_bars(tickers=("AAPL",), sessions=3)
    broken = frame.assign(low=frame["high"] * 2.0, close=[np.nan, -1.0, 0.0])

    conformed = conform_bars(broken)
    assert len(conformed) == 3
    assert conformed["close"].isna().sum() == 1


def test_write_rejects_a_ticker_that_would_escape_the_store(root: Path) -> None:
    frame = synthetic_bars(tickers=("AAPL",), sessions=2)

    with pytest.raises(ValueError, match="directory name"):
        write_bars(frame.assign(ticker="../escaped"), root=root)
    assert not root.exists()


# ---------------------------------------------------------------------------
# Where the store lives
# ---------------------------------------------------------------------------


def test_bars_root_hangs_off_the_configured_data_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "store")

    assert bars_root(settings=settings) == tmp_path / "store" / "bars" / "daily"
    assert (
        partition_path("AAPL", settings=settings)
        == tmp_path / "store" / "bars" / "daily" / "ticker=AAPL" / "part.parquet"
    )


def test_settings_round_trip_writes_under_the_data_dir(tmp_path: Path) -> None:
    """The path everything else will use: no explicit root, just settings."""
    settings = Settings(data_dir=tmp_path / "store")
    original = synthetic_bars(tickers=("AAPL",), sessions=6)

    write_bars(original, settings=settings)
    restored = read_bars("AAPL", settings=settings)

    assert (tmp_path / "store" / "bars" / "daily" / "ticker=AAPL" / "part.parquet").is_file()
    pd.testing.assert_frame_equal(restored, original)
