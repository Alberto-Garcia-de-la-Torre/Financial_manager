"""Day 16: the back-adjusted close, and the check against Yahoo's.

The acceptance criterion — every divergence from Yahoo comes with its cause —
is exercised here on synthetic references built to carry each kind of cause
the real run found (a stale newest bar, a restated dividend, Yahoo's pence
error, a dividend declared in another currency), plus the saved SAN.MC
response, on which ours and Yahoo's must agree. The live run and its report
are `docs/adjusted_divergences.md`. Nothing here reaches the network.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yahoo_fixtures
from typer.testing import CliRunner

from finmgr.data import adjcheck, fetch
from finmgr.data.actions import merge_actions, to_actions
from finmgr.data.adjust import (
    adjust,
    adjust_series,
    dividend_factors,
    read_adjusted,
    split_visibility,
)
from finmgr.data.fetch import to_bars
from finmgr.data.pit import LATEST
from finmgr.data.store import write_bars

AAPL = "AAPL_2020-08-24_2020-09-04"
SANTANDER = "SAN.MC_2024-10-28_2024-11-11"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(fetch, "download_history", refuse)
    monkeypatch.setattr(adjcheck, "download_reference", refuse)


def days(n: int, start: str = "2024-01-01") -> list[dt.date]:
    """`n` consecutive business days."""
    return [d.date() for d in pd.bdate_range(start, periods=n)]


def series(values: list[float], dates: list[dt.date] | None = None) -> pd.Series:
    return pd.Series(values, index=dates or days(len(values)), dtype="float64")


# ---------------------------------------------------------------------------
# The adjustment
# ---------------------------------------------------------------------------


def test_a_dividend_scales_every_earlier_close_by_one_minus_yield() -> None:
    d = days(4)
    out = adjust_series(series([10.0, 10.0, 9.5, 9.6], d), series([0.5], [d[2]]))

    assert out["dividend_factor"].tolist() == pytest.approx([0.95, 0.95, 1.0, 1.0])
    assert out["adj_close"].tolist() == pytest.approx([9.5, 9.5, 9.5, 9.6])
    # The newest session is always its raw close.
    assert out["adj_close"].iloc[-1] == out["close"].iloc[-1]


def test_successive_dividends_multiply() -> None:
    d = days(5)
    out = adjust_series(series([10.0, 9.0, 9.0, 8.0, 8.0], d), series([1.0, 1.0], [d[1], d[3]]))

    first, second = 1 - 1 / 10, 1 - 1 / 9
    assert out["dividend_factor"].tolist() == pytest.approx([first * second, second, second, 1, 1])


def test_an_ex_date_with_no_session_uses_the_last_session_before_it() -> None:
    d = days(3, "2024-01-04")  # Thu, Fri, Mon
    saturday = dt.date(2024, 1, 6)
    paid = dividend_factors(series([10.0, 8.0, 7.9], d), series([0.4], [saturday]))

    assert paid["reference_date"].iloc[0] == d[1]
    assert paid["factor"].iloc[0] == pytest.approx(1 - 0.4 / 8.0)


def test_dividends_outside_the_stored_history_move_nothing() -> None:
    d = days(3, "2024-02-01")
    out = adjust_series(
        series([10.0, 10.0, 10.0], d),
        series([0.5, 0.5], [dt.date(2023, 12, 1), dt.date(2024, 3, 1)]),
    )

    assert (out["dividend_factor"] == 1.0).all()


def test_a_dividend_bigger_than_the_price_is_refused() -> None:
    d = days(2)
    with pytest.raises(ValueError, match="not below the close"):
        adjust_series(series([1.0, 1.0], d), series([1.5], [d[1]]))


def test_a_split_yahoo_already_applied_is_not_applied_again() -> None:
    bars = to_bars("AAPL", yahoo_fixtures.load_raw(AAPL))
    found = to_actions("AAPL", yahoo_fixtures.load_raw(AAPL))

    out = adjust(bars, found)

    seen = split_visibility(bars.set_index("date")["close"], found.set_index("date")["value"])
    assert not seen["in_bars"].iloc[0]
    assert (out["split_factor"] == 1.0).all()
    assert out["adj_close"].tolist() == pytest.approx(bars["close"].tolist())


def test_a_split_the_bars_still_show_divides_the_earlier_closes() -> None:
    d = days(4)
    out = adjust_series(series([400.0, 404.0, 101.0, 102.0], d), splits=series([4.0], [d[2]]))

    assert out["split_factor"].tolist() == [0.25, 0.25, 1.0, 1.0]
    assert out["adj_close"].tolist() == pytest.approx([100.0, 101.0, 101.0, 102.0])


def test_a_small_scrip_issue_on_a_down_day_is_not_mistaken_for_a_visible_split() -> None:
    # Santander's 2011-01-17 scrip issue: 1.54% more shares on a day the price
    # also fell about 1.1% — closer to the split than to no move at all.
    d = days(3)
    closes = series([8.0, 8.0, 7.915], d)
    seen = split_visibility(closes, series([1.015385], [d[2]]))

    assert seen["ambiguous"].iloc[0]
    assert not seen["in_bars"].iloc[0]
    assert (adjust_series(closes, splits=series([1.015385], [d[2]]))["split_factor"] == 1).all()


def test_read_adjusted_combines_the_two_stores(tmp_path: Path) -> None:
    raw = yahoo_fixtures.load_raw(SANTANDER)
    write_bars(to_bars("SAN.MC", raw), root=tmp_path / "bars" / "daily")
    merge_actions(to_actions("SAN.MC", raw), root=tmp_path / "actions")

    out = read_adjusted("SAN.MC", root=tmp_path, as_of=LATEST).set_index("date")

    factor = 1 - 0.1 / out.loc[dt.date(2024, 10, 29), "close"]
    assert out.loc[dt.date(2024, 10, 28), "dividend_factor"] == pytest.approx(factor)
    assert out.loc[dt.date(2024, 10, 30), "dividend_factor"] == 1.0
    assert len(read_adjusted("SAN.MC", start="2024-11-01", root=tmp_path, as_of=LATEST)) == 7


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def yahoo_like(closes: pd.Series, dividends: pd.Series, *, applied: float = 1.0) -> pd.DataFrame:
    """A reference frame whose Adj Close applies `applied` x each dividend."""
    adj = adjust_series(closes, dividends * applied)["adj_close"]
    out = pd.DataFrame({"close": closes, "adj_close": adj, "dividend": 0.0, "split": 0.0})
    out.loc[dividends.index, "dividend"] = dividends.to_numpy()
    out.index.name = "date"
    return out


def run(
    closes: pd.Series, dividends: pd.Series, reference: pd.DataFrame, **kwargs: Any
) -> adjcheck.Comparison:
    return adjcheck.compare("X", closes, dividends, series([]), reference, **kwargs)


def sample() -> tuple[pd.Series, pd.Series]:
    d = days(30)
    closes = series(list(np.linspace(100, 120, 30)), d)
    return closes, series([3.0, 3.5, 4.0], [d[5], d[15], d[25]])


def test_ours_matches_yahoo_on_the_saved_santander_response() -> None:
    raw = yahoo_fixtures.load_raw(SANTANDER)
    bars = to_bars("SAN.MC", raw).set_index("date")["close"]
    found = to_actions("SAN.MC", raw).set_index("date")["value"]

    result = adjcheck.compare("SAN.MC", bars, found, series([]), adjcheck.to_reference(raw))

    assert result.divergence < 1e-6
    assert not result.diverges and result.causes == []


def test_identical_inputs_do_not_diverge() -> None:
    closes, divs = sample()
    result = run(closes, divs, yahoo_like(closes, divs))

    assert result.divergence < 1e-12
    assert result.explained and not result.diverges


def test_a_dividend_paid_after_our_last_bar_is_anchored_away() -> None:
    closes, divs = sample()
    later = days(32)[30:]
    reference = yahoo_like(
        pd.concat([closes, series([121.0, 118.0], later)]),
        pd.concat([divs, series([3.0], [later[1]])]),
    )

    assert run(closes, divs, reference).divergence < 1e-12


def test_a_stale_newest_bar_is_named_as_an_intraday_price() -> None:
    closes, divs = sample()
    reference = yahoo_like(closes, divs)
    stale = closes.copy()
    stale.iloc[-1] *= 1.02
    last = stale.index[-1]

    result = run(
        stale, divs, reference, stored_at=(last, dt.datetime(*last.timetuple()[:3], 9, 48))
    )

    assert result.diverges and result.days_over == 1 and result.worst_date == last
    assert result.from_closes == pytest.approx(0.02)
    [cause] = result.causes
    assert cause.startswith("Raw closes: 1 session(s)")
    assert "09:48 UTC on the same day" in cause


def test_a_restated_dividend_is_named() -> None:
    closes, divs = sample()
    restated = divs.copy()
    restated.iloc[0] = 4.5

    result = run(closes, divs, yahoo_like(closes, restated))

    assert result.diverges
    [cause] = result.causes
    assert cause.startswith("Dividends: Yahoo has restated 1 dividend(s)")


def test_yahoos_pence_error_is_named() -> None:
    closes, divs = sample()
    result = run(closes, divs, yahoo_like(closes, divs, applied=0.01))

    assert result.diverges
    [cause] = result.causes
    assert "1/100 of the dividend" in cause and "pence" in cause


def test_a_dividend_declared_in_another_currency_is_named() -> None:
    closes, divs = sample()
    result = run(closes, divs, yahoo_like(closes, divs, applied=1.2048))

    assert result.diverges
    [cause] = result.causes
    assert "x1.2048" in cause and "exchange rate" in cause


def test_a_divergence_nobody_accounts_for_is_unexplained() -> None:
    result = adjcheck.Comparison("X", divergence=0.02)

    assert result.diverges and not result.explained


def test_cli_compare_explains_and_reports(
    tmp_path: Path, write_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finmgr import cli
    from finmgr.config import SETTINGS_PATH_ENV

    universe = tmp_path / "universe.csv"
    universe.write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "SAN.MC,Banco Santander,XMAD,EUR,Financials,Spain\n",
        encoding="utf-8",
    )
    values = {
        "data_dir": str(tmp_path),
        "log_dir": str(tmp_path / "logs"),
        "universe_path": str(universe),
    }
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(write_settings(values)))
    raw = yahoo_fixtures.load_raw(SANTANDER)
    write_bars(to_bars("SAN.MC", raw), root=tmp_path / "bars" / "daily")
    merge_actions(to_actions("SAN.MC", raw), root=tmp_path / "actions")

    # Yahoo's side applies only 1% of the dividend: one divergence, explained.
    pence = raw.copy()
    reference = adjcheck.to_reference(raw)
    pence["Adj Close"] = yahoo_like(
        reference["close"], reference.loc[reference["dividend"] > 0, "dividend"], applied=0.01
    )["adj_close"].to_numpy()
    monkeypatch.setattr(adjcheck, "download_reference", lambda *_: adjcheck.to_reference(pence))
    report = tmp_path / "report.md"

    result = CliRunner().invoke(cli.app, ["adjusted", "--compare", "--report", str(report)])

    assert result.exit_code == 0, result.output
    assert "1 of 1 tickers diverge; 0 unexplained" in result.output
    text = report.read_text(encoding="utf-8")
    assert "| SAN.MC |" in text and "pence" in text
