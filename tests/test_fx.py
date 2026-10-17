"""Day 18: EUR exchange rates, and close_eur in the bar store.

The acceptance criterion is that a US stock's EUR series diverges from its USD
series in the direction FX moved. So the cases that matter are the conversion
itself (a division by the session's EURUSD, never a later one), the two ways
it can go quietly wrong — London's pence, and a session with no same-day rate
— and the end-to-end check: a stock flat in USD while the euro strengthens
must lose money in EUR. Nothing here touches the network; the stores live in
`tmp_path`.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data import fetch
from finmgr.data.fx import (
    REQUIRED_PAIRS,
    divergence,
    fetch_fx,
    ingest_fx,
    merge_fx,
    pair_for,
    read_fx,
    refresh_close_eur,
    to_eur,
    universe_pairs,
    yahoo_symbol,
)
from finmgr.data.pit import LATEST
from finmgr.data.store import read_bars, write_bars
from finmgr.data.universe import Company

D = dt.date

COMPANIES = [
    Company("AAPL", "Apple Inc.", "XNAS", "USD", "Information Technology", "United States"),
    Company("SAN.MC", "Banco Santander", "XMAD", "EUR", "Financials", "Spain"),
    Company("AZN.L", "AstraZeneca PLC", "XLON", "GBP", "Health Care", "United Kingdom"),
    Company("7203.T", "Toyota", "XTKS", "JPY", "Consumer Discretionary", "Japan"),
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(fetch, "download_history", refuse)


def weekdays(start: dt.date, count: int) -> list[dt.date]:
    return [d.date() for d in pd.bdate_range(start, periods=count)]


def bars(ticker: str, days: Iterable[dt.date], closes: Iterable[float]) -> pd.DataFrame:
    closes = list(closes)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": list(days),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000.0,
        }
    )


def rates(pair: str, days: Iterable[dt.date], values: Iterable[float]) -> pd.DataFrame:
    return pd.DataFrame({"pair": pair, "date": list(days), "rate": list(values)})


@pytest.fixture
def bars_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "bars" / "daily"


@pytest.fixture
def fx_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "fx"


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_pairs_are_eur_crosses_and_eur_needs_none() -> None:
    assert pair_for("usd") == "EURUSD"
    assert pair_for("EUR") is None
    assert yahoo_symbol("EURSEK") == "EURSEK=X"
    with pytest.raises(ValueError):
        pair_for("US$")
    with pytest.raises(ValueError):
        yahoo_symbol("USDJPY")


def test_the_roadmaps_four_are_always_ingested_first() -> None:
    pairs = universe_pairs(COMPANIES)

    assert tuple(pairs[:4]) == REQUIRED_PAIRS == ("EURUSD", "EURGBP", "EURCHF", "EURSEK")
    assert pairs[4:] == ["EURJPY"]


# ---------------------------------------------------------------------------
# The conversion
# ---------------------------------------------------------------------------


def test_a_dollar_close_is_divided_by_that_sessions_eurusd() -> None:
    days = weekdays(D(2025, 1, 6), 3)
    frame = bars("AAPL", days, [110.0, 110.0, 110.0])

    eur = to_eur(frame, "USD", rates("EURUSD", days, [1.0, 1.1, 1.25]))

    assert eur.tolist() == pytest.approx([110.0, 100.0, 88.0])


def test_a_euro_close_is_its_own_eur_close() -> None:
    frame = bars("SAN.MC", weekdays(D(2025, 1, 6), 2), [4.5, 4.6])

    assert to_eur(frame, "EUR", rates("EURUSD", [], [])).tolist() == [4.5, 4.6]


def test_london_pence_are_pounds_before_they_are_euros() -> None:
    day = D(2025, 1, 6)
    frame = bars("AZN.L", [day], [12_500.0])  # 12,500 GBp is £125

    eur = to_eur(frame, "GBP", rates("EURGBP", [day], [0.8]), scale=0.01)

    assert eur.tolist() == pytest.approx([156.25])


def test_a_session_without_a_same_day_rate_borrows_the_previous_one_only() -> None:
    """Tokyo trades on a day with no EURJPY row: take the rate before, never after."""
    monday, tuesday, wednesday = D(2025, 1, 6), D(2025, 1, 7), D(2025, 1, 8)
    frame = bars("7203.T", [monday, tuesday, wednesday], [2_000.0] * 3)

    eur = to_eur(frame, "JPY", rates("EURJPY", [monday, wednesday], [160.0, 100.0]))

    assert eur.tolist() == pytest.approx([12.5, 12.5, 20.0])


def test_no_rate_before_the_bar_or_too_old_or_not_yet_fetched_is_nan() -> None:
    days = [D(2025, 1, 2), D(2025, 1, 6), D(2025, 1, 20), D(2025, 1, 21)]
    frame = bars("AAPL", days, [100.0] * 4)
    stored = rates("EURUSD", [D(2025, 1, 3), D(2025, 1, 6), D(2025, 1, 10)], [1.0, 1.0, 1.0])

    eur = to_eur(frame, "USD", stored)

    # Jan 2 precedes every rate; Jan 20 would borrow a 10-day-old rate; Jan 21
    # is after the newest rate stored. Only Jan 6 is honestly convertible.
    assert [not math.isnan(v) for v in eur] == [False, True, False, False]


def test_a_bad_rate_is_kept_in_the_store_but_never_used() -> None:
    day = D(2025, 1, 6)
    stored = rates("EURUSD", [day], [0.0])

    assert math.isnan(to_eur(bars("AAPL", [day], [100.0]), "USD", stored).iloc[0])


# ---------------------------------------------------------------------------
# The FX store and its ingest
# ---------------------------------------------------------------------------


def test_merging_rates_twice_adds_nothing_and_revisions_overwrite(fx_dir: Path) -> None:
    days = weekdays(D(2025, 1, 6), 3)
    (first,) = merge_fx(rates("EURUSD", days, [1.03, 1.04, 1.05]), root=fx_dir)
    (again,) = merge_fx(rates("EURUSD", days, [1.03, 1.04, 1.05]), root=fx_dir)
    (revised,) = merge_fx(rates("EURUSD", days[1:], [1.04, 1.06]), root=fx_dir)

    assert (first.added, again.added, again.path) == (3, 0, None)
    assert (revised.added, revised.updated, revised.stored) == (0, 1, 3)
    assert read_fx("EURUSD", root=fx_dir, as_of=LATEST)["rate"].tolist() == [1.03, 1.04, 1.06]


def test_fetch_fx_keeps_yahoos_close_as_the_rate() -> None:
    days = pd.DatetimeIndex(weekdays(D(2025, 1, 6), 2), name="Date").tz_localize("Europe/London")
    raw = pd.DataFrame(
        {"Open": [1.0, 1.1], "High": [1.0, 1.1], "Low": [1.0, 1.1], "Close": [1.03, 1.04]},
        index=days,
    ).assign(Volume=0)
    asked: list[str] = []

    def download(symbol: str, *_: object) -> pd.DataFrame:
        asked.append(symbol)
        return raw

    original = fetch.download_history
    fetch.download_history = download
    try:
        found = fetch_fx("EURUSD", D(2025, 1, 6), D(2025, 1, 7))
    finally:
        fetch.download_history = original

    assert asked == ["EURUSD=X"]
    assert found.to_dict("records") == [
        {"pair": "EURUSD", "date": D(2025, 1, 6), "rate": 1.03},
        {"pair": "EURUSD", "date": D(2025, 1, 7), "rate": 1.04},
    ]


def test_one_failing_pair_does_not_stop_the_others(fx_dir: Path) -> None:
    def fake(pair: str, start: dt.date, _end: dt.date) -> pd.DataFrame:
        if pair == "EURCHF":
            raise ConnectionError("reset by peer")
        return rates(pair, [start], [1.5])

    results = ingest_fx(
        ["EURUSD", "EURCHF", "EURSEK"],
        start=D(2025, 1, 6),
        end=D(2025, 1, 6),
        root=fx_dir,
        fetch=fake,
        sleep=lambda _: None,
    )

    assert [(r.pair, r.ok, r.added) for r in results] == [
        ("EURUSD", True, 1),
        ("EURCHF", False, 0),
        ("EURSEK", True, 1),
    ]
    assert sorted(read_fx(root=fx_dir, as_of=LATEST)["pair"]) == ["EURSEK", "EURUSD"]


# ---------------------------------------------------------------------------
# close_eur in the bar store
# ---------------------------------------------------------------------------


def test_refresh_writes_close_eur_once_and_then_leaves_the_store_alone(
    bars_dir: Path, fx_dir: Path
) -> None:
    days = weekdays(D(2025, 1, 6), 3)
    write_bars(bars("AAPL", days, [110.0] * 3), root=bars_dir)
    write_bars(bars("SAN.MC", days, [4.0] * 3), root=bars_dir)
    write_bars(bars("ZZZ", days, [1.0] * 3), root=bars_dir)
    merge_fx(rates("EURUSD", days, [1.0, 1.1, 1.25]), root=fx_dir)

    first = {
        r.ticker: r
        for r in refresh_close_eur(bars_dir=bars_dir, fx_dir=fx_dir, companies=COMPANIES)
    }
    second = refresh_close_eur(bars_dir=bars_dir, fx_dir=fx_dir, companies=COMPANIES)

    assert read_bars("AAPL", root=bars_dir, as_of=LATEST)["close_eur"].tolist() == pytest.approx(
        [110, 100, 88]
    )
    assert read_bars("SAN.MC", root=bars_dir, as_of=LATEST)["close_eur"].tolist() == [4.0, 4.0, 4.0]
    assert first["AAPL"].converted == 3 and first["AAPL"].path is not None
    assert "not in the universe" in first["ZZZ"].error
    assert all(r.path is None for r in second)


def test_a_missing_cross_is_reported_not_guessed(bars_dir: Path, fx_dir: Path) -> None:
    write_bars(bars("AZN.L", weekdays(D(2025, 1, 6), 2), [12_000.0] * 2), root=bars_dir)

    (result,) = refresh_close_eur(bars_dir=bars_dir, fx_dir=fx_dir, companies=COMPANIES)

    assert result.converted == 0
    assert "no EURGBP rates" in result.error
    assert read_bars("AZN.L", root=bars_dir, as_of=LATEST)["close_eur"].isna().all()


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def stock_flat_in_dollars_while_the_euro_rises(bars_dir: Path, fx_dir: Path) -> None:
    days = weekdays(D(2025, 1, 6), 20)
    write_bars(bars("AAPL", days, [200.0] * 20), root=bars_dir)
    merge_fx(rates("EURUSD", days, [1.00 + 0.01 * i for i in range(20)]), root=fx_dir)
    refresh_close_eur(bars_dir=bars_dir, fx_dir=fx_dir, companies=COMPANIES)


def test_the_eur_series_diverges_in_the_direction_fx_moved(bars_dir: Path, fx_dir: Path) -> None:
    stock_flat_in_dollars_while_the_euro_rises(bars_dir, fx_dir)

    summary, series = divergence("AAPL", bars_dir=bars_dir, fx_dir=fx_dir, companies=COMPANIES)

    assert summary.local_return == 0.0
    assert summary.fx_return == pytest.approx(0.19)
    # Flat in USD, euro up 19%: an EUR investor lost 1 - 1/1.19 of their money.
    assert summary.eur_return == pytest.approx(1 / 1.19 - 1)
    assert summary.gap < 0
    assert summary.consistent
    assert series["eur_index"].is_monotonic_decreasing
    assert (series["local_index"] == 100).all()


def test_a_conversion_the_wrong_way_round_is_caught(bars_dir: Path, fx_dir: Path) -> None:
    """Multiplying by EURUSD instead of dividing is the classic slip; the check must see it."""
    stock_flat_in_dollars_while_the_euro_rises(bars_dir, fx_dir)
    stored = read_bars("AAPL", root=bars_dir, as_of=LATEST)
    write_bars(stored.assign(close_eur=stored["close"] ** 2 / stored["close_eur"]), root=bars_dir)

    summary, _ = divergence("AAPL", bars_dir=bars_dir, fx_dir=fx_dir, companies=COMPANIES)

    assert summary.gap > 0
    assert not summary.consistent


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        + "".join(
            f"{c.ticker},{c.name},{c.exchange},{c.currency},{c.sector},{c.country}\n"
            for c in COMPANIES
        ),
        encoding="utf-8",
    )
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\nuniverse_path: {universe}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    return CliRunner()


def test_the_cli_compares_and_draws_the_two_series(
    cli: CliRunner, tmp_path: Path, bars_dir: Path, fx_dir: Path
) -> None:
    stock_flat_in_dollars_while_the_euro_rises(bars_dir, fx_dir)
    png = tmp_path / "aapl.png"

    result = cli.invoke(app, ["fx", "--compare", "AAPL", "--plot", str(png)], terminal_width=200)

    assert result.exit_code == 0, result.output
    said = " ".join(result.output.split())  # rich wraps at its own width
    assert "EURUSD +19.00% (the euro strengthened)" in said
    assert "consistent: the EUR series diverges in the direction FX moved" in said
    assert png.stat().st_size > 0


def test_the_cli_refreshes_close_eur_without_fetching(
    cli: CliRunner, bars_dir: Path, fx_dir: Path
) -> None:
    days = weekdays(D(2025, 1, 6), 2)
    write_bars(bars("AAPL", days, [110.0, 110.0]), root=bars_dir)
    merge_fx(rates("EURUSD", days, [1.1, 1.1]), root=fx_dir)

    result = cli.invoke(app, ["fx", "--no-fetch"], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert "1 of 1 tickers converted" in result.output
    assert read_bars("AAPL", root=bars_dir, as_of=LATEST)["close_eur"].tolist() == pytest.approx(
        [100.0, 100.0]
    )
