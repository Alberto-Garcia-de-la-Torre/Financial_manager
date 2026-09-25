"""Day 10: one ticker's bars, from a Yahoo response into the store's schema.

The acceptance criterion is `test_fetch_daily_from_a_saved_response`, and the
"without touching the network" half of it is enforced structurally: the
`no_network` fixture below is autouse, and it replaces the one function in
:mod:`finmgr.data.fetch` that can reach Yahoo with a stub that fails the test.
Every test here therefore runs offline whether it remembers to or not, and a
future edit that sneaks a live call into the conversion path fails loudly
instead of quietly depending on Yahoo being up.

The raw material is real: `tests/fixtures/yahoo/` holds exactly what yfinance
returned for two windows, captured by `tests/yahoo_fixtures.py`. AAPL spans its
4:1 split, SAN.MC spans a dividend and trades in Europe/Madrid.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yahoo_fixtures

from finmgr.config import Settings
from finmgr.data import fetch
from finmgr.data.fetch import fetch_daily, to_bars
from finmgr.data.store import BAR_COLUMNS, BAR_DTYPES, read_bars, write_bars

AAPL = "AAPL_2020-08-24_2020-09-04"
SANTANDER = "SAN.MC_2024-10-28_2024-11-11"

#: The real downloader, taken at import time — `no_network` below replaces the
#: module attribute, and the three tests that exercise the download path itself
#: need the function rather than the guard standing in for it.
download_history = fetch.download_history


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take Yahoo away from every test in this module."""

    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(fetch, "download_history", refuse)


def responder(name: str, *, calls: list[tuple] | None = None):
    """A downloader handing back a saved response instead of calling Yahoo."""
    raw = yahoo_fixtures.load_raw(name)

    def download(ticker: str, start: dt.date, end_exclusive: dt.date) -> pd.DataFrame:
        if calls is not None:
            calls.append((ticker, start, end_exclusive))
        return raw.copy()

    return download


def expected_bars(name: str) -> pd.DataFrame:
    """What the fixture says the bars should be, built without the code under test.

    Deliberately naive — read the CSV, take the date out of the timestamp
    string, rename the five price columns — so that a bug in `to_bars` cannot
    also be in the expectation it is compared against.
    """
    meta = yahoo_fixtures.load_meta(name)
    raw = pd.read_csv(yahoo_fixtures.FIXTURE_DIR / f"{name}.csv")
    return pd.DataFrame(
        {
            "ticker": pd.Series([meta["ticker"]] * len(raw), dtype=BAR_DTYPES["ticker"]),
            "date": pd.Series(
                [dt.date.fromisoformat(stamp[:10]) for stamp in raw["Date"]],
                dtype=BAR_DTYPES["date"],
            ),
            "open": raw["Open"].astype("float64"),
            "high": raw["High"].astype("float64"),
            "low": raw["Low"].astype("float64"),
            "close": raw["Close"].astype("float64"),
            "volume": raw["Volume"].astype("float64"),
        }
    )


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [AAPL, SANTANDER])
def test_fetch_daily_from_a_saved_response(name: str) -> None:
    """Day 10: a saved yfinance response becomes bars in the fixed schema."""
    meta = yahoo_fixtures.load_meta(name)

    bars = fetch_daily(meta["ticker"], meta["start"], meta["end"], download=responder(name))

    assert list(bars.columns) == list(BAR_COLUMNS)
    assert {column: str(dtype) for column, dtype in bars.dtypes.items()} == BAR_DTYPES
    assert len(bars) == meta["rows"]
    pd.testing.assert_frame_equal(bars, expected_bars(name))


def test_the_saved_response_was_captured_with_auto_adjust_false() -> None:
    """The fixture is only worth testing against if it is raw.

    `auto_adjust=True` folds dividends into the close and drops `Adj Close`
    entirely. Both columns present, and differing, is the fingerprint of the
    unadjusted request this project depends on.
    """
    raw = yahoo_fixtures.load_raw(SANTANDER)

    assert {"Close", "Adj Close", "Dividends", "Stock Splits"} <= set(raw.columns)
    assert not raw["Close"].equals(raw["Adj Close"])
    assert "auto_adjust=False" in yahoo_fixtures.load_meta(SANTANDER)["call"]


def test_close_is_yahoos_raw_close_not_its_adjusted_one() -> None:
    """The number stored is the one that was printed on the day.

    Day 16 rebuilds the adjusted series from splits and dividends; it can only
    do that if what is on disk never moved.
    """
    raw = yahoo_fixtures.load_raw(SANTANDER)
    bars = fetch_daily("SAN.MC", "2024-10-28", "2024-11-11", download=responder(SANTANDER))

    assert bars["close"].to_numpy() == pytest.approx(raw["Close"].to_numpy())
    # The ex-dividend day in the window: Yahoo's adjusted close is 4% lower,
    # so a fetch that quietly used it would be off by that much forever.
    ex_day = bars.loc[bars["date"] == dt.date(2024, 10, 30), "close"].item()
    assert ex_day == pytest.approx(4.41)
    assert raw.loc["2024-10-30", "Dividends"] == pytest.approx(0.10)


def test_a_split_day_survives_untouched() -> None:
    """AAPL's 4:1 split day is a bar like any other; the split is day 15's fact."""
    raw = yahoo_fixtures.load_raw(AAPL)
    bars = fetch_daily("AAPL", "2020-08-24", "2020-09-04", download=responder(AAPL))

    split_day = bars.loc[bars["date"] == dt.date(2020, 8, 31)]
    assert raw.loc["2020-08-31", "Stock Splits"] == 4.0
    assert split_day["close"].item() == pytest.approx(129.0399932861328)
    assert len(bars) == len(raw)


# ---------------------------------------------------------------------------
# Dates: exchange-local, inclusive, trimmed
# ---------------------------------------------------------------------------


def test_madrid_sessions_keep_the_date_they_traded_on() -> None:
    """The bug this guards: Madrid stamps midnight +01:00, which is 23:00 UTC
    the day before. Dating bars through UTC would shift every one of them."""
    raw = yahoo_fixtures.load_raw(SANTANDER)
    bars = fetch_daily("SAN.MC", "2024-10-28", "2024-11-11", download=responder(SANTANDER))

    assert bars["date"].tolist() == list(raw.index.date)
    shifted = list(raw.index.tz_convert("UTC").date)
    assert shifted != bars["date"].tolist()
    assert bars["date"].iloc[0] == dt.date(2024, 10, 28)


def test_end_is_inclusive_and_yahoo_is_asked_for_the_day_after() -> None:
    calls: list[tuple] = []

    bars = fetch_daily("AAPL", "2020-08-24", "2020-09-04", download=responder(AAPL, calls=calls))

    assert calls == [("AAPL", dt.date(2020, 8, 24), dt.date(2020, 9, 5))]
    assert bars["date"].iloc[-1] == dt.date(2020, 9, 4)


def test_bars_outside_the_requested_window_are_trimmed() -> None:
    """Yahoo is generous at the edges; the caller gets what they asked for."""
    bars = fetch_daily("AAPL", "2020-08-27", "2020-09-01", download=responder(AAPL))

    assert bars["date"].tolist() == [
        dt.date(2020, 8, 27),
        dt.date(2020, 8, 28),
        dt.date(2020, 8, 31),
        dt.date(2020, 9, 1),
    ]


def test_dates_may_be_given_as_dates_or_iso_strings() -> None:
    by_string = fetch_daily("AAPL", "2020-08-24", "2020-09-04", download=responder(AAPL))
    by_date = fetch_daily(
        "AAPL", dt.date(2020, 8, 24), dt.datetime(2020, 9, 4, 17, 30), download=responder(AAPL)
    )

    pd.testing.assert_frame_equal(by_string, by_date)


def test_the_default_window_comes_from_settings(tmp_path: Path) -> None:
    """`fetch_daily(t)` means the whole history the project cares about."""
    settings = Settings(
        data_dir=tmp_path, history_start=dt.date(2010, 1, 4), timezone="Europe/Madrid"
    )
    calls: list[tuple] = []

    fetch_daily("AAPL", settings=settings, download=responder(AAPL, calls=calls))

    (_, start, end_exclusive) = calls[0]
    assert start == dt.date(2010, 1, 4)
    # Today in Madrid, taken either side of the call so a midnight rollover
    # between the two cannot fail the test.
    today = {
        dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=offset))).date()
        for offset in (1, 2)
    }
    assert end_exclusive - dt.timedelta(days=1) in today


def test_a_reversed_window_is_refused() -> None:
    with pytest.raises(ValueError, match="after end"):
        fetch_daily("AAPL", "2020-09-04", "2020-08-24", download=responder(AAPL))


def test_an_empty_ticker_is_refused() -> None:
    with pytest.raises(ValueError, match="ticker is empty"):
        fetch_daily("  ", "2020-08-24", "2020-09-04", download=responder(AAPL))


# ---------------------------------------------------------------------------
# Empties, oddities and what the conversion refuses
# ---------------------------------------------------------------------------


def test_a_response_with_no_bars_returns_the_schema() -> None:
    """A dead or too-young listing is not an error; it is zero rows."""
    empty = yahoo_fixtures.load_raw(AAPL).iloc[:0]

    bars = fetch_daily("AAPL", "2020-08-24", "2020-09-04", download=lambda *_: empty)

    assert bars.empty
    assert {column: str(dtype) for column, dtype in bars.dtypes.items()} == BAR_DTYPES


def test_a_window_with_no_sessions_returns_the_schema() -> None:
    """The saved response covers August; ask for the Christmas week."""
    bars = fetch_daily("AAPL", "2020-12-24", "2020-12-28", download=responder(AAPL))

    assert bars.empty
    assert {column: str(dtype) for column, dtype in bars.dtypes.items()} == BAR_DTYPES


def test_to_bars_accepts_a_frame_that_has_been_reset() -> None:
    """Some callers hand over `history().reset_index()`; the Date column works too."""
    raw = yahoo_fixtures.load_raw(AAPL)

    pd.testing.assert_frame_equal(to_bars("AAPL", raw.reset_index()), to_bars("AAPL", raw))


def test_to_bars_rejects_a_response_missing_a_price_column() -> None:
    raw = yahoo_fixtures.load_raw(AAPL).drop(columns=["Volume"])

    with pytest.raises(ValueError, match=r"missing \['volume'\]"):
        to_bars("AAPL", raw)


def test_to_bars_rejects_a_frame_with_no_session_timestamps() -> None:
    raw = yahoo_fixtures.load_raw(AAPL).reset_index(drop=True)

    with pytest.raises(ValueError, match="no session timestamps"):
        to_bars("AAPL", raw)


def test_a_repeated_session_keeps_the_later_row() -> None:
    """Yahoo sometimes returns a live partial bar beside the settled one."""
    raw = yahoo_fixtures.load_raw(AAPL)
    doubled = pd.concat([raw, raw.iloc[[-1]].assign(Close=999.0)])

    bars = to_bars("AAPL", doubled)

    assert len(bars) == len(raw)
    assert bars["close"].iloc[-1] == pytest.approx(999.0)


def test_the_action_and_adjusted_columns_never_reach_the_bars() -> None:
    bars = to_bars("AAPL", yahoo_fixtures.load_raw(AAPL))

    assert list(bars.columns) == list(BAR_COLUMNS)


def test_an_unrecognised_column_is_ignored_rather_than_fatal() -> None:
    raw = yahoo_fixtures.load_raw(AAPL).assign(**{"Capital Gains": 0.0, "Something New": 1.0})

    pd.testing.assert_frame_equal(
        to_bars("AAPL", raw), to_bars("AAPL", yahoo_fixtures.load_raw(AAPL))
    )


# ---------------------------------------------------------------------------
# The download path, with yfinance's own exceptions but no yfinance server
# ---------------------------------------------------------------------------


def stub_yfinance(monkeypatch: pytest.MonkeyPatch, answer: Callable[[], Any]):
    """Replace `yf.Ticker` so `download_history` runs without a request.

    The module itself is the real one — its exception classes have to be, or
    the `except` clause under test is not the one that would fire in
    production — but nothing in it reaches the network.
    """
    import yfinance as yf

    calls: list[dict] = []

    class Ticker:
        def __init__(self, ticker: str) -> None:
            self.ticker = ticker

        def history(self, **kwargs: Any) -> Any:
            calls.append({"ticker": self.ticker, **kwargs})
            return answer()

    monkeypatch.setattr(yf, "Ticker", Ticker)
    return calls


def test_download_asks_for_daily_unadjusted_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = yahoo_fixtures.load_raw(AAPL)
    calls = stub_yfinance(monkeypatch, lambda: raw)

    frame = download_history("AAPL", dt.date(2020, 8, 24), dt.date(2020, 9, 5))

    assert frame is raw
    assert calls == [
        {
            "ticker": "AAPL",
            "start": "2020-08-24",
            "end": "2020-09-05",
            "interval": "1d",
            "auto_adjust": False,
        }
    ]


def test_yahoos_missing_prices_error_is_an_empty_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A holiday window and a delisted symbol raise the same thing at Yahoo.

    Day 12 re-runs over weekends; if that raised, every Saturday would look
    like a broken ticker.
    """
    from yfinance.exceptions import YFPricesMissingError

    def missing() -> Any:
        raise YFPricesMissingError("AAPL", "no price data found")

    stub_yfinance(monkeypatch, missing)

    frame = download_history("AAPL", dt.date(2020, 12, 25), dt.date(2020, 12, 26))

    assert len(frame) == 0
    assert to_bars("AAPL", frame).empty


def test_a_failed_request_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rate limiting is day 11's problem to retry, not this function's to hide."""
    from yfinance.exceptions import YFRateLimitError

    def throttled() -> Any:
        raise YFRateLimitError()

    stub_yfinance(monkeypatch, throttled)

    with pytest.raises(YFRateLimitError):
        download_history("AAPL", dt.date(2020, 8, 24), dt.date(2020, 9, 5))


# ---------------------------------------------------------------------------
# And into the store
# ---------------------------------------------------------------------------


def test_fetched_bars_go_straight_into_the_store(tmp_path: Path) -> None:
    """Day 9 and day 10 meet: no conversion step between fetch and write."""
    bars = fetch_daily("SAN.MC", "2024-10-28", "2024-11-11", download=responder(SANTANDER))

    write_bars(bars, root=tmp_path)
    restored = read_bars("SAN.MC", root=tmp_path)

    pd.testing.assert_frame_equal(restored, bars)


def test_the_network_guard_is_really_armed() -> None:
    """Proof that `no_network` would catch a live call, not just claim to."""
    with pytest.raises(AssertionError, match="tried to reach Yahoo"):
        fetch_daily("AAPL", "2020-08-24", "2020-09-04")
