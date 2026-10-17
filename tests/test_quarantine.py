"""Day 20: quarantine, never delete.

The acceptance test is the first one: a row the register calls `bad-data` is
gone from `read_bars(as_of=LATEST)` and still on disk — in its bar partition, read with
pyarrow straight off the file, and in `data/quarantine/` with its reason. The
rest pin down what keeps that true: a merge or an EUR refresh rewriting the
partition must not drop the row, a second run changes nothing, and taking the
rule out releases the row without deleting its record.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data.calendars import expected_sessions
from finmgr.data.pit import LATEST
from finmgr.data.quarantine import apply_quarantine, read_quarantine, record_path
from finmgr.data.store import merge_bars, partition_path, read_bars, write_bars

D = dt.date
SESSIONS = expected_sessions("XMAD", 2024, start=D(2024, 3, 1), end=D(2024, 3, 28))
BAD_DAY = SESSIONS[4]
AS_OF = SESSIONS[-1]
REASON = "impossible bar: high under low"


def bars(ticker: str = "SAN.MC") -> pd.DataFrame:
    close = [10.0 + 0.1 * i for i in range(len(SESSIONS))]
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "date": SESSIONS,
            "open": close,
            "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close],
            "close": close,
            "volume": 1_000.0,
        }
    )
    frame.loc[4, ["high", "low"]] = [9.0, 11.0]  # high < low: the bad row
    return frame


def register(project: Path, *rows: str) -> None:
    (project / "quality_triage.csv").write_text(
        "check,scope,start,end,verdict,reason\n" + "".join(row + "\n" for row in rows),
        encoding="utf-8",
    )


BAD_RULE = f"high_below_low,SAN.MC,{BAD_DAY},{BAD_DAY},bad-data,{REASON}"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "universe.csv").write_text(
        "ticker,name,exchange,currency,sector,country\n"
        "SAN.MC,Banco Santander,XMAD,EUR,Financials,Spain\n",
        encoding="utf-8",
    )
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        f"data_dir: {tmp_path / 'data'}\n"
        f"universe_path: {tmp_path / 'universe.csv'}\n"
        f"log_dir: {tmp_path / 'logs'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    write_bars(bars())
    register(tmp_path, BAD_RULE)
    return tmp_path


@pytest.mark.usefixtures("project")
def test_a_quarantined_row_is_absent_from_read_bars_but_still_on_disk() -> None:
    assert BAD_DAY in set(read_bars("SAN.MC", as_of=LATEST)["date"])  # before: an ordinary row

    (result,) = apply_quarantine(as_of=AS_OF)
    assert (result.quarantined, result.added) == (1, 1)

    # Absent from the default read...
    read = read_bars("SAN.MC", as_of=LATEST)
    assert BAD_DAY not in set(read["date"])
    assert len(read) == len(SESSIONS) - 1

    # ...still in the bar partition, read straight off the file...
    on_disk = pq.read_table(partition_path("SAN.MC")).to_pandas()
    row = on_disk[on_disk["date"] == BAD_DAY]
    assert len(on_disk) == len(SESSIONS)
    assert row["quarantined"].tolist() == [True]
    assert row[["high", "low"]].values.tolist() == [[9.0, 11.0]]

    # ...and recorded in data/quarantine/ with its reason.
    path = record_path("SAN.MC")
    assert path.parent.parent.name == "quarantine" and path.is_file()
    (record,) = pq.read_table(path).to_pylist()
    assert record["date"] == BAD_DAY
    assert record["check"] == "high_below_low"
    assert record["reason"] == REASON
    assert (record["high"], record["low"]) == (9.0, 11.0)
    assert record["released_at"] is None

    # Asking for it explicitly gets it back.
    every = read_bars("SAN.MC", include_quarantined=True, as_of=LATEST)
    assert every.loc[every["date"] == BAD_DAY, "quarantined"].tolist() == [True]


@pytest.mark.usefixtures("project")
def test_a_second_run_writes_nothing() -> None:
    apply_quarantine(as_of=AS_OF)
    bars_mtime = partition_path("SAN.MC").stat().st_mtime_ns
    record_mtime = record_path("SAN.MC").stat().st_mtime_ns

    (again,) = apply_quarantine(as_of=AS_OF)

    assert not again.changed and again.path is None and again.quarantined == 1
    assert partition_path("SAN.MC").stat().st_mtime_ns == bars_mtime
    assert record_path("SAN.MC").stat().st_mtime_ns == record_mtime


@pytest.mark.usefixtures("project")
def test_a_merge_keeps_the_quarantined_row_and_its_flag() -> None:
    apply_quarantine(as_of=AS_OF)
    # Yahoo sends the whole month back, bad bar included, plus a new session.
    later = bars()
    extra = later.iloc[[-1]].assign(date=D(2024, 4, 1))
    merge_bars(pd.concat([later, extra]))

    every = read_bars("SAN.MC", include_quarantined=True, as_of=LATEST)
    assert len(every) == len(SESSIONS) + 1
    assert every.loc[every["date"] == BAD_DAY, "quarantined"].tolist() == [True]
    assert BAD_DAY not in set(read_bars("SAN.MC", as_of=LATEST)["date"])


@pytest.mark.usefixtures("project")
def test_the_eur_refresh_does_not_drop_quarantined_rows() -> None:
    from finmgr.data.fx import refresh_close_eur

    apply_quarantine(as_of=AS_OF)
    refresh_close_eur()  # SAN.MC is EUR-listed: close_eur == close, partition rewritten

    on_disk = pq.read_table(partition_path("SAN.MC")).to_pandas()
    assert len(on_disk) == len(SESSIONS)
    assert on_disk.loc[on_disk["date"] == BAD_DAY, "quarantined"].tolist() == [True]


@pytest.mark.usefixtures("project")
def test_the_checks_still_see_a_quarantined_row() -> None:
    """Quarantine hides the row from analysis, not from `finmgr check`.

    Otherwise its triage rule would turn up unused and the session it left
    behind would be flagged as missing.
    """
    apply_quarantine(as_of=AS_OF)
    runner = CliRunner()
    result = runner.invoke(app, ["check", "--as-of", AS_OF.isoformat()], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "1 triaged, 0 untriaged" in result.output


def test_removing_the_rule_releases_the_row_and_keeps_the_record(project: Path) -> None:
    apply_quarantine(as_of=AS_OF)
    register(project, f"high_below_low,SAN.MC,{BAD_DAY},{BAD_DAY},genuine,looked again")

    (result,) = apply_quarantine(as_of=AS_OF)

    assert (result.quarantined, result.released) == (0, 1)
    assert BAD_DAY in set(read_bars("SAN.MC", as_of=LATEST)["date"])
    assert read_quarantine("SAN.MC", as_of=LATEST).empty
    (history,) = read_quarantine("SAN.MC", include_released=True, as_of=LATEST).itertuples()
    assert history.reason == REASON and pd.notna(history.released_at)


def test_only_bad_data_verdicts_quarantine(project: Path) -> None:
    register(project)  # the flag is untriaged: `finmgr check` fails, nothing is hidden
    (result,) = apply_quarantine(as_of=AS_OF)
    assert result.quarantined == 0 and not record_path("SAN.MC").exists()
    assert len(read_bars("SAN.MC", as_of=LATEST)) == len(SESSIONS)


def test_a_partition_from_before_day_20_reads_as_unquarantined(tmp_path: Path) -> None:
    write_bars(bars(), root=tmp_path / "b")
    path = partition_path("SAN.MC", root=tmp_path / "b")
    table = pq.read_table(path).drop(["quarantined"])
    pq.write_table(table, path)

    read = read_bars("SAN.MC", root=tmp_path / "b", as_of=LATEST)
    assert len(read) == len(SESSIONS)
    assert not read["quarantined"].any() and read["quarantined"].dtype == "bool"


@pytest.mark.usefixtures("project")
def test_the_cli_quarantines_and_lists() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["quarantine", "--as-of", AS_OF.isoformat()], env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0, result.output
    assert "high_below_low" in result.output and REASON in result.output
    assert "1 row(s) quarantined" in result.output
    assert BAD_DAY not in set(read_bars("SAN.MC", as_of=LATEST)["date"])
