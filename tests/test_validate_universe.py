"""The ticker validator, exercised without touching the network.

The day-8 script's real job is a hundred HTTP requests, and none of those
belong in a test suite. What is testable — and what would actually go wrong —
is everything around them: whether an empty frame is recognised as a dead
symbol, whether a frame of old bars is recognised as stale, whether a
throttled first answer is retried, and whether the script's exit code really
follows the failure list.

Every test here injects a fake fetcher, so the suite stays offline and fast.
`tests/test_universe.py` covers the contents of `config/universe.csv` itself.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from finmgr.config import load_settings
from finmgr.data import validate
from finmgr.data.universe import Company, load_universe
from finmgr.data.validate import Status, TickerCheck

TODAY = date(2026, 9, 22)


def bars(*days: date) -> pd.DataFrame:
    """A frame shaped like `yf.Ticker(t).history()`: tz-aware daily index."""
    index = pd.DatetimeIndex(
        [datetime(d.year, d.month, d.day, tzinfo=ZoneInfo("America/New_York")) for d in days],
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [10.0] * len(days),
            "High": [11.0] * len(days),
            "Low": [9.0] * len(days),
            "Close": [10.5] * len(days),
            "Volume": [1_000] * len(days),
        },
        index=index,
    )


def recent_bars(count: int = 5, *, last: date = TODAY) -> pd.DataFrame:
    return bars(*[last - timedelta(days=offset) for offset in reversed(range(count))])


def empty_frame() -> pd.DataFrame:
    """What yfinance returns for an unknown or delisted symbol."""
    return bars()


def fetcher(answers: dict[str, object]) -> Callable[[str, str], object]:
    """A fake `fetch_history`. A callable answer is invoked per attempt."""

    def _fetch(ticker: str, period: str) -> object:  # noqa: ARG001 - signature must match
        answer = answers[ticker]
        return answer() if callable(answer) else answer

    return _fetch


# ---------------------------------------------------------------------------
# classify(): the verdict on one response
# ---------------------------------------------------------------------------


def test_a_frame_of_recent_bars_is_ok() -> None:
    result = validate.classify("AAPL", recent_bars(5), today=TODAY)

    assert result.status is Status.OK
    assert result.ok
    assert result.rows == 5
    assert result.last_date == TODAY
    assert result.first_date == TODAY - timedelta(days=4)
    assert result.age_days == 0


def test_an_empty_frame_is_a_failure_not_an_exception() -> None:
    """Yahoo's answer to a wrong suffix: no rows, no error. This is the case."""
    result = validate.classify("AIR", empty_frame(), today=TODAY)

    assert result.status is Status.EMPTY
    assert not result.ok
    assert result.rows == 0
    assert result.last_date is None


def test_none_is_treated_like_an_empty_frame() -> None:
    assert validate.classify("NOPE", None, today=TODAY).status is Status.EMPTY


def test_old_bars_are_stale_rather_than_ok() -> None:
    """A suspended listing keeps answering — with the same last bar forever."""
    last = TODAY - timedelta(days=30)
    result = validate.classify("GRF.MC", bars(last), today=TODAY, max_age_days=7)

    assert result.status is Status.STALE
    assert result.age_days == 30
    assert "30 days old" in result.detail


def test_a_weekend_gap_is_not_stale() -> None:
    """Run on a Monday, the newest US bar is Friday's. That is normal."""
    result = validate.classify("MSFT", bars(TODAY - timedelta(days=3)), today=TODAY)

    assert result.status is Status.OK


def test_the_staleness_threshold_is_the_boundary_not_a_range() -> None:
    at_limit = validate.classify("KO", bars(TODAY - timedelta(days=7)), today=TODAY)
    past_limit = validate.classify("KO", bars(TODAY - timedelta(days=8)), today=TODAY)

    assert at_limit.status is Status.OK
    assert past_limit.status is Status.STALE


# ---------------------------------------------------------------------------
# check_ticker(): retries around one symbol
# ---------------------------------------------------------------------------


def test_an_exception_is_reported_as_an_error_not_raised() -> None:
    def explode(ticker: str, period: str) -> object:  # noqa: ARG001 - signature must match
        raise TimeoutError("read timed out")

    result = validate.check_ticker(
        "SAP.DE", today=TODAY, attempts=2, fetch=explode, sleep=lambda _: None
    )

    assert result.status is Status.ERROR
    assert result.detail == "TimeoutError: read timed out"
    assert result.attempts == 2


def test_a_throttled_first_answer_is_retried_before_condemning_a_ticker() -> None:
    """One empty frame means "ask again", not "delete this row from the CSV"."""
    answers = iter([empty_frame(), recent_bars()])
    result = validate.check_ticker(
        "ASML.AS",
        today=TODAY,
        attempts=3,
        fetch=fetcher({"ASML.AS": lambda: next(answers)}),
        sleep=lambda _: None,
    )

    assert result.status is Status.OK
    assert result.attempts == 2


def test_a_symbol_that_is_always_empty_exhausts_its_attempts() -> None:
    calls = 0

    def count_calls(ticker: str, period: str) -> object:  # noqa: ARG001 - signature must match
        nonlocal calls
        calls += 1
        return empty_frame()

    result = validate.check_ticker(
        "DEAD.XX", today=TODAY, attempts=3, fetch=count_calls, sleep=lambda _: None
    )

    assert result.status is Status.EMPTY
    assert calls == 3


def test_a_successful_first_answer_costs_one_request() -> None:
    calls = 0

    def count_calls(ticker: str, period: str) -> object:  # noqa: ARG001 - signature must match
        nonlocal calls
        calls += 1
        return recent_bars()

    validate.check_ticker("V", today=TODAY, attempts=3, fetch=count_calls, sleep=lambda _: None)

    assert calls == 1


# ---------------------------------------------------------------------------
# check_universe(): the loop, and the failure list
# ---------------------------------------------------------------------------


def test_every_ticker_is_checked_and_one_bad_symbol_does_not_stop_the_run() -> None:
    answers = {
        "AAPL": recent_bars(),
        "AIR": empty_frame(),
        "SAN.MC": recent_bars(),
    }
    results = validate.check_universe(
        list(answers),
        today=TODAY,
        attempts=1,
        pause=0,
        fetch=fetcher(answers),
        sleep=lambda _: None,
    )

    assert [result.ticker for result in results] == ["AAPL", "AIR", "SAN.MC"]
    assert [result.ticker for result in validate.failures(results)] == ["AIR"]


def test_the_failure_list_is_empty_when_everything_returns_bars() -> None:
    """The day-8 acceptance criterion, expressed against a fake Yahoo."""
    answers = {ticker: recent_bars() for ticker in ("AAPL", "SAN.MC", "SAP.DE", "ASML.AS")}
    results = validate.check_universe(
        list(answers), today=TODAY, pause=0, fetch=fetcher(answers), sleep=lambda _: None
    )

    assert validate.failures(results) == []
    assert all(result.ok for result in results)


def test_company_rows_are_accepted_as_well_as_plain_symbols() -> None:
    company = Company("IBE.MC", "Iberdrola SA", "XMAD", "EUR", "Utilities", "Spain")
    results = validate.check_universe(
        [company],
        today=TODAY,
        pause=0,
        fetch=fetcher({"IBE.MC": recent_bars()}),
        sleep=lambda _: None,
    )

    assert [result.ticker for result in results] == ["IBE.MC"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_markdown_report_names_every_failure() -> None:
    results = [
        TickerCheck("AAPL", Status.OK, rows=5, last_date=TODAY, age_days=0),
        TickerCheck("AIR", Status.EMPTY, detail="no bars returned"),
    ]
    report = validate.render_markdown(
        results,
        checked_at=datetime(2026, 9, 22, 9, 0, tzinfo=ZoneInfo("Europe/Madrid")),
        period="5d",
        max_age_days=7,
    )

    assert "1 of 2 returned recent bars" in report
    assert "| AIR | empty |" in report
    assert "## Failures" in report


def test_a_clean_report_says_so_explicitly() -> None:
    results = [TickerCheck("AAPL", Status.OK, rows=5, last_date=TODAY, age_days=0)]
    report = validate.render_markdown(
        results,
        checked_at=datetime(2026, 9, 22, 9, 0, tzinfo=ZoneInfo("Europe/Madrid")),
        period="5d",
        max_age_days=7,
    )

    assert "None. All tickers returned recent bars." in report


# ---------------------------------------------------------------------------
# The script's exit code
# ---------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, object]], None]:
    """Replace the module's network call and its sleeps for a `main()` run."""

    def _install(answers: dict[str, object]) -> None:
        monkeypatch.setattr(validate, "fetch_history", fetcher(answers))
        monkeypatch.setattr(validate.time, "sleep", lambda _: None)

    return _install


def test_main_exits_zero_when_every_ticker_returns_bars(
    offline: Callable[[dict[str, object]], None],
    write_settings: Callable[..., Path],
    tmp_path: Path,
) -> None:
    offline({"AAPL": recent_bars(), "SAN.MC": recent_bars()})
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "AAPL,Apple Inc.,XNAS,USD,Information Technology,United States\n"
        "SAN.MC,Banco Santander SA,XMAD,EUR,Financials,Spain\n",
        encoding="utf-8",
    )
    settings = write_settings({"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path)})

    code = validate.main(
        ["--universe", str(universe), "--pause", "0"],
        settings=load_settings(settings),
    )

    assert code == 0


def test_main_exits_non_zero_when_a_ticker_is_dead(
    offline: Callable[[dict[str, object]], None],
    write_settings: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """A dead ticker must fail the run, not be mentioned in passing."""
    offline({"AAPL": recent_bars(), "AIR": empty_frame()})
    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "AAPL,Apple Inc.,XNAS,USD,Information Technology,United States\n"
        "AIR,Wrong suffix,XPAR,EUR,Industrials,France\n",
        encoding="utf-8",
    )
    settings = write_settings({"data_dir": str(tmp_path / "data"), "log_dir": str(tmp_path)})
    report = tmp_path / "report.md"

    code = validate.main(
        ["--universe", str(universe), "--pause", "0", "--attempts", "1", "--report", str(report)],
        settings=load_settings(settings),
    )

    assert code == 1
    assert "| AIR | empty |" in report.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The loader the script reads the universe through
# ---------------------------------------------------------------------------


def test_the_real_universe_loads_as_one_hundred_companies() -> None:
    companies = load_universe()

    assert len(companies) == 100
    assert all(isinstance(company, Company) for company in companies)
    assert companies[0].ticker == "AAPL"


def test_a_duplicate_ticker_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "universe.csv"
    path.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "AAPL,Apple Inc.,XNAS,USD,Information Technology,United States\n"
        "AAPL,Apple again,XNAS,USD,Information Technology,United States\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate tickers"):
        load_universe(path)


def test_a_renamed_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "universe.csv"
    path.write_text("symbol,name,exchange,currency,sector,country\nAAPL,,,,,\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected columns"):
        load_universe(path)


def test_a_missing_universe_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_universe(tmp_path / "nope.csv")
