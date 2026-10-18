"""Day 22: benchmarks and the risk-free rate.

The acceptance criterion is a chart: the cumulative return of all 100 against
the benchmark on one set of axes. So the end-to-end case builds a hundred
synthetic stocks and a benchmark in `tmp_path`, runs `finmgr benchmarks
--plot`, and checks the PNG. Around it, the pieces that can be quietly wrong:
the ECB parser, the €STR/EONIA splice, ACT/360 accrual that never uses a rate
from the future, total return in EUR, and the rebasing of a stock that lists
mid-window. Nothing here touches the network.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV, load_settings
from finmgr.data import fetch
from finmgr.data.actions import merge_actions
from finmgr.data.benchmarks import (
    BENCHMARK_COLOR,
    STOCK_COLOR,
    compare,
    equal_weight,
    growth,
    performance,
    plot_comparison,
    primary_benchmark,
    read_benchmarks,
    read_total_return,
)
from finmgr.data.fx import merge_fx
from finmgr.data.pit import LATEST, LookAheadError
from finmgr.data.rates import (
    ingest_rates,
    merge_rates,
    parse_ecb_csv,
    read_rates,
    risk_free_returns,
    short_rate,
)
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS

D = dt.date
SESSIONS = list(pd.bdate_range("2024-01-01", "2024-03-29").date)

ECB_CSV = """KEY,FREQ,BENCHMARK_ITEM,DATA_TYPE_EST,TIME_PERIOD,OBS_VALUE,OBS_STATUS
EST.B.EU000A2X2A25.WT,B,EU000A2X2A25,WT,2024-01-02,3.905,A
EST.B.EU000A2X2A25.WT,B,EU000A2X2A25,WT,2024-01-03,3.902,A
"""


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(fetch, "download_history", refuse)


def rates_frame(series: str, days: list[dt.date], rate: float) -> pd.DataFrame:
    return pd.DataFrame({"series": series, "date": days, "rate": rate})


def bars(ticker: str, days: list[dt.date], closes: list[float] | np.ndarray) -> pd.DataFrame:
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": days,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000.0,
            "close_eur": closes,
        }
    )


# ---------------------------------------------------------------------------
# The rate store
# ---------------------------------------------------------------------------


def test_the_ecb_csv_parses_to_percent_a_year_by_date() -> None:
    frame = parse_ecb_csv("ESTR", ECB_CSV)

    assert list(frame["date"]) == [D(2024, 1, 2), D(2024, 1, 3)]
    assert list(frame["rate"]) == [3.905, 3.902]
    assert set(frame["series"]) == {"ESTR"}
    assert parse_ecb_csv("ESTR", "").empty


def test_an_unknown_series_is_refused() -> None:
    with pytest.raises(ValueError, match="not a known rate series"):
        parse_ecb_csv("SOFR", ECB_CSV)


def test_merging_the_same_rates_twice_writes_nothing_the_second_time(tmp_path: Path) -> None:
    frame = parse_ecb_csv("ESTR", ECB_CSV)

    (first,) = merge_rates(frame, root=tmp_path)
    (second,) = merge_rates(frame, root=tmp_path)

    assert (first.added, second.added, second.updated, second.path) == (2, 0, 0, None)
    assert len(read_rates("ESTR", as_of=LATEST, root=tmp_path)) == 2


def test_ingest_reports_a_failing_series_and_keeps_going(tmp_path: Path) -> None:
    def fake(series: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if series == "EONIA":
            raise ConnectionError("ECB down")
        return parse_ecb_csv(series, ECB_CSV)

    results = ingest_rates(
        start="2024-01-01", end="2024-01-31", root=tmp_path, fetch=fake, sleep=lambda _: None
    )

    assert [(r.series, r.ok, r.added) for r in results] == [("ESTR", True, 2), ("EONIA", False, 0)]
    assert "ECB down" in results[1].error


def test_estr_wins_where_it_exists_and_eonia_fills_in_before() -> None:
    stored = pd.concat(
        [
            rates_frame("EONIA", [D(2019, 9, 30), D(2019, 10, 1)], -0.45),
            rates_frame("ESTR", [D(2019, 10, 1)], -0.55),
        ]
    )

    rate = short_rate(stored)

    assert rate.to_dict() == {D(2019, 9, 30): -0.45, D(2019, 10, 1): -0.55}


# ---------------------------------------------------------------------------
# What cash earned
# ---------------------------------------------------------------------------


def test_cash_accrues_act_360_over_the_weekend_at_fridays_rate() -> None:
    friday, monday, tuesday = D(2024, 1, 5), D(2024, 1, 8), D(2024, 1, 9)
    stored = pd.concat(
        [rates_frame("ESTR", [friday], 3.6), rates_frame("ESTR", [monday], 7.2)],
        ignore_index=True,
    )

    rf = risk_free_returns([friday, monday, tuesday], as_of=tuesday, rates=stored)

    assert rf[friday] == 0.0
    assert rf[monday] == pytest.approx(0.036 * 3 / 360)
    assert rf[tuesday] == pytest.approx(0.072 * 1 / 360)


def test_cash_never_accrues_at_a_rate_fixed_after_the_session_before() -> None:
    """Monday's return uses Friday's rate even when Monday's own is on file."""
    friday, monday = D(2024, 1, 5), D(2024, 1, 8)
    stored = rates_frame("ESTR", [monday], 3.6)

    rf = risk_free_returns([friday, monday], as_of=monday, rates=stored)

    assert math.isnan(rf[monday])


def test_cash_refuses_sessions_or_rates_after_as_of() -> None:
    day = D(2024, 1, 5)
    with pytest.raises(LookAheadError):
        risk_free_returns([day, D(2024, 1, 8)], as_of=day, rates=rates_frame("ESTR", [day], 3.0))
    with pytest.raises(LookAheadError):
        risk_free_returns([day], as_of=day, rates=rates_frame("ESTR", [D(2024, 1, 8)], 3.0))


# ---------------------------------------------------------------------------
# Total return, growth, statistics
# ---------------------------------------------------------------------------


def test_total_return_in_eur_reinvests_the_dividend_and_converts(tmp_path: Path) -> None:
    days = SESSIONS[:3]
    frame = bars("AAPL", days, [100.0, 100.0, 98.0]).assign(close_eur=[50.0, 50.0, 49.0])
    write_bars(frame, root=tmp_path / "bars" / "daily")
    merge_actions(
        pd.DataFrame({"ticker": "AAPL", "date": [days[2]], "action": "dividend", "value": [2.0]}),
        root=tmp_path / "actions",
    )

    out = read_total_return("AAPL", as_of=days[-1], root=tmp_path)

    # The 2.0 dividend on a 100.0 close makes the 2% drop a flat total return.
    assert out["adj_close_eur"].iloc[-1] / out["adj_close_eur"].iloc[0] == pytest.approx(1.0)
    assert list(out["adj_close_eur"] / out["adj_close"]) == pytest.approx([0.5, 0.5, 0.5])


def test_growth_rebases_a_late_listing_at_its_own_first_session() -> None:
    days = SESSIONS[:4]
    prices = pd.DataFrame(
        {"OLD": [10.0, 11.0, np.nan, 12.1], "NEW": [np.nan, np.nan, 5.0, 10.0]}, index=days
    )

    out = growth(prices)

    assert list(out["OLD"]) == pytest.approx([1.0, 1.1, 1.1, 1.21])
    assert out["NEW"].iloc[:2].isna().all()
    assert list(out["NEW"].iloc[2:]) == pytest.approx([1.0, 2.0])


def test_equal_weight_averages_the_daily_returns_of_whoever_is_listed() -> None:
    days = SESSIONS[:3]
    prices = pd.DataFrame({"A": [10.0, 11.0, 11.0], "B": [np.nan, 20.0, 30.0]}, index=days)

    out = equal_weight(prices)

    # Day 2: only A has a return (+10%). Day 3: A 0%, B +50%, mean +25%.
    assert list(out) == pytest.approx([1.0, 1.1, 1.1 * 1.25])


def test_sharpe_is_excess_return_over_its_volatility() -> None:
    returns = pd.Series([0.01, -0.005, 0.02, 0.0], index=SESSIONS[1:5])
    rf = pd.Series(0.0001, index=returns.index)

    result = performance("X", returns, rf, start=SESSIONS[0], end=SESSIONS[4])

    excess = returns - rf
    assert result.sharpe == pytest.approx(excess.mean() / excess.std() * math.sqrt(252))
    assert result.total_return == pytest.approx((1 + returns).prod() - 1)


# ---------------------------------------------------------------------------
# The acceptance criterion: all 100 against the benchmark, on one chart
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A hundred stocks, one benchmark and €STR, in a settings file under tmp_path."""
    rng = np.random.default_rng(22)
    data = tmp_path / "data"
    tickers = [f"S{i:03d}" for i in range(100)]
    for i, ticker in enumerate(tickers):
        # The last ten list a month into the window.
        days = SESSIONS[20:] if i >= 90 else SESSIONS
        path = 100 * np.cumprod(1 + rng.normal(0.0005 * (i - 50) / 50, 0.01, len(days)))
        write_bars(bars(ticker, days, path), root=data / "bars" / "daily")
    write_bars(
        bars("BENCH", SESSIONS, 100 * 1.001 ** np.arange(len(SESSIONS))),
        root=data / "benchmarks" / "bars" / "daily",
    )
    merge_rates(rates_frame("ESTR", SESSIONS, 3.6), root=data / "rates")
    merge_fx(pd.DataFrame({"pair": "EURUSD", "date": SESSIONS, "rate": 1.0}), root=data / "fx")

    company = {"exchange": "XNYS", "currency": "EUR", "sector": "Industrials", "country": "X"}
    universe = write_csv(
        tmp_path / "universe.csv", [{"ticker": t, "name": t, **company} for t in tickers]
    )
    benchmarks = write_csv(
        tmp_path / "benchmarks.csv",
        [{"ticker": "BENCH", "name": "The Benchmark", **company, "sector": "Benchmark"}],
    )
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        yaml.safe_dump(
            {
                "data_dir": str(data),
                "universe_path": str(universe),
                "benchmarks_path": str(benchmarks),
                "benchmark": "BENCH",
                "log_dir": str(tmp_path / "logs"),
                "history_start": "2024-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return tmp_path


@pytest.mark.usefixtures("world")
def test_compare_puts_all_100_and_the_benchmark_on_one_calendar() -> None:
    settings = load_settings()
    comparison = compare(as_of=SESSIONS[-1], settings=settings)

    assert comparison.stocks.shape[1] == 100
    assert comparison.benchmark == primary_benchmark(settings=settings) == "BENCH"
    assert comparison.stocks.index.equals(comparison.benchmark_growth.index)
    assert comparison.benchmark_growth.iloc[-1] == pytest.approx(1.001 ** (len(SESSIONS) - 1))
    # Every stock starts at 1, the late listings on their own first day.
    firsts = comparison.stocks.apply(lambda column: column.dropna().iloc[0])
    assert np.allclose(firsts, 1.0)
    assert comparison.stocks["S095"].first_valid_index() == SESSIONS[20]
    assert comparison.cash.iloc[-1] > 1.0
    assert len(comparison.outperformers()) == 100


@pytest.mark.usefixtures("world")
def test_compare_reads_nothing_after_as_of() -> None:
    cutoff = SESSIONS[30]
    comparison = compare(as_of=cutoff, settings=load_settings())

    assert comparison.end == cutoff
    with pytest.raises(LookAheadError):
        compare(end=SESSIONS[31], as_of=cutoff, settings=load_settings())


def test_finmgr_benchmarks_plots_all_100_against_the_benchmark(world: Path) -> None:
    out = world / "chart.png"

    result = CliRunner().invoke(app, ["benchmarks", "--no-fetch", "--plot", str(out)])

    assert result.exit_code == 0, result.output
    assert "of 100 stocks beat BENCH" in result.output
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_chart_draws_one_line_per_stock_plus_the_benchmark(
    world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    drawn: list[Any] = []
    monkeypatch.setattr(plt, "close", drawn.append)

    plot_comparison(compare(as_of=SESSIONS[-1], settings=load_settings()), world / "c.png")

    (figure,) = drawn
    lines = figure.axes[0].get_lines()
    # 100 stock lines in the neutral, then the legend proxy, cash, equal weight, benchmark.
    stocks = [line for line in lines if line.get_color() == STOCK_COLOR and len(line.get_xdata())]
    assert len(stocks) == 100
    assert "BENCH: The Benchmark" not in [line.get_label() for line in lines]
    assert [line.get_label() for line in lines if line.get_color() == BENCHMARK_COLOR] == ["BENCH"]
    plt.close("all")


@pytest.mark.usefixtures("world")
def test_read_benchmarks_reads_the_benchmark_directory() -> None:
    out = read_benchmarks(as_of=SESSIONS[-1], settings=load_settings())

    assert set(out["ticker"]) == {"BENCH"}
    assert out["adj_close_eur"].notna().all()
