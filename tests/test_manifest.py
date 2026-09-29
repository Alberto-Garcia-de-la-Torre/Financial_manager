"""Day 14: the ingestion manifest, and `finmgr ingest --report`.

The acceptance criterion is `test_report_summarises_the_last_run_from_the_manifest_alone`:
two runs go through the real CLI, the store and the network are then taken
away, and `--report` still prints the second run — its tickers, its failure
and its summary line — because everything it needs is in
`data/meta/ingest_runs.parquet`.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yahoo_fixtures
from requests.exceptions import ConnectionError as RequestsConnectionError
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV, load_settings
from finmgr.data import ingest
from finmgr.data.fetch import to_bars
from finmgr.data.ingest import IngestReport, Status, TickerIngest, ingest_universe
from finmgr.data.manifest import (
    MANIFEST_COLUMNS,
    append_run,
    last_run,
    last_run_id,
    manifest_path,
    read_manifest,
)

AAPL = "AAPL_2020-08-24_2020-09-04"
WINDOW = (dt.date(2020, 8, 24), dt.date(2020, 9, 4))


def bars_for(ticker: str, *_: object, **__: object) -> pd.DataFrame:
    return to_bars(ticker, yahoo_fixtures.load_raw(AAPL))


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError(f"a test tried to reach Yahoo: {args} {kwargs}")

    monkeypatch.setattr(ingest, "fetch_daily", refuse)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the CLI's settings at a throwaway data directory."""
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"data_dir: {tmp_path / 'data'}\nlog_dir: {tmp_path / 'logs'}\n", encoding="utf-8"
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings_file))
    # Wide enough that rich does not fold an error message across lines.
    monkeypatch.setenv("COLUMNS", "200")
    return tmp_path / "data"


def run_report(
    tickers: list[str], *, fail: frozenset[str] = frozenset(), **kwargs: Any
) -> IngestReport:
    def fetch(ticker: str, _start: dt.date, _end: dt.date) -> pd.DataFrame:
        if ticker in fail:
            raise RequestsConnectionError("[Errno 101] Network is unreachable")
        return bars_for(ticker)

    report = ingest_universe(
        tickers,
        start=WINDOW[0],
        end=WINDOW[1],
        fetch=fetch,
        sleep=lambda _: None,
        **kwargs,
    )
    # Outside the CLI there is no run of its own; a context left behind by an
    # earlier CLI test would otherwise lend every report the same id.
    return dataclasses.replace(report, run_id=None)


# ---------------------------------------------------------------------------
# The acceptance criterion
# ---------------------------------------------------------------------------


def test_report_summarises_the_last_run_from_the_manifest_alone(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real runs, then `--report` with the store deleted and Yahoo gone."""
    monkeypatch.setattr(ingest, "fetch_daily", bars_for)
    common = "--start 2020-08-24 --end 2020-09-04 --pause 0 --backoff 0 --abort-after 0"
    first = CliRunner().invoke(app, f"ingest -t AAA -t BBB {common}".split())
    assert first.exit_code == 0, first.output

    def flaky(ticker: str, *_: object, **__: object) -> pd.DataFrame:
        if ticker == "IBE.MC":
            raise RequestsConnectionError("[Errno 101] Network is unreachable")
        return bars_for(ticker)

    monkeypatch.setattr(ingest, "fetch_daily", flaky)
    second = CliRunner().invoke(app, f"ingest -t AAA -t IBE.MC -t CCC --full {common}".split())
    assert second.exit_code == 1, second.output

    # Nothing but the manifest survives, and nothing can be downloaded.
    shutil.rmtree(data_dir / "bars")
    assert sorted(p.name for p in data_dir.iterdir()) == ["meta"]

    def refuse(*_: object, **__: object) -> Any:
        raise AssertionError("--report tried to reach Yahoo")

    monkeypatch.setattr(ingest, "fetch_daily", refuse)
    result = CliRunner().invoke(app, ["ingest", "--report"])

    # The run was incomplete, and the report's exit code says so.
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    output = result.output
    assert "Last ingest run" in output and "full" in output
    assert "window 2020-08-24 to 2020-09-04" in output
    # The second run's tickers, not the first's.
    assert "CCC" in output and "BBB" not in output
    assert "IBE.MC" in output and "Network is unreachable" in output
    rows = len(bars_for("AAA"))
    assert f"2 of 3 tickers ingested (failed=1, ok=2); {2 * rows} bars fetched" in output


def test_the_manifest_lives_under_data_meta(data_dir: Path) -> None:
    assert manifest_path(settings=load_settings()) == data_dir / "meta" / "ingest_runs.parquet"


def test_the_manifest_holds_one_row_per_ticker_per_run(data_dir: Path) -> None:
    path = data_dir / "meta" / "ingest_runs.parquet"
    append_run(run_report(["AAA", "BBB"], root=data_dir / "bars"), path=path)
    append_run(
        run_report(["AAA", "IBE.MC"], fail=frozenset({"IBE.MC"}), root=data_dir / "bars"), path=path
    )

    frame = read_manifest(path=path)

    assert tuple(frame.columns) == MANIFEST_COLUMNS
    assert len(frame) == 4
    assert frame["run_id"].nunique() == 2
    assert not frame.duplicated(["run_id", "ticker"]).any()

    rows = len(bars_for("AAA"))
    first_aaa = frame.iloc[0]
    assert first_aaa["ticker"] == "AAA" and first_aaa["status"] == "ok"
    assert first_aaa["rows_fetched"] == rows
    assert first_aaa["rows_written"] == rows
    assert first_aaa["min_date"] == dt.date(2020, 8, 24)
    assert first_aaa["max_date"] == dt.date(2020, 9, 4)
    assert pd.isna(first_aaa["error"])

    # The second run re-fetched AAA's overlap and wrote none of it.
    second_aaa = frame.iloc[2]
    assert second_aaa["ticker"] == "AAA" and second_aaa["rows_written"] == 0

    ibe = frame.iloc[3]
    assert ibe["status"] == "failed"
    assert ibe["rows_written"] == 0
    assert pd.isna(ibe["min_date"]) and pd.isna(ibe["max_date"])
    assert "Network is unreachable" in ibe["error"]


def test_recording_the_same_run_twice_does_not_duplicate_it(tmp_path: Path) -> None:
    path = tmp_path / "ingest_runs.parquet"
    report = run_report(["AAA"], root=tmp_path / "bars", write=False)
    report = dataclasses.replace(report, run_id="20200904T000000-abc-0001")

    append_run(report, path=path)
    append_run(report, path=path)

    assert len(read_manifest(path=path)) == 1


def test_a_report_without_a_run_id_gets_one(tmp_path: Path) -> None:
    path = tmp_path / "ingest_runs.parquet"
    report = run_report(["AAA"], root=tmp_path / "bars")

    append_run(report, path=path)

    run_id = read_manifest(path=path)["run_id"].iloc[0]
    assert run_id and run_id.startswith(report.started_at.strftime("%Y%m%dT"))


def test_the_last_run_round_trips_through_the_manifest(tmp_path: Path) -> None:
    path = tmp_path / "ingest_runs.parquet"
    append_run(run_report(["AAA"], root=tmp_path / "bars"), path=path)
    original = run_report(
        ["AAA", "IBE.MC", "BBB", "CCC"],
        fail=frozenset({"IBE.MC", "BBB"}),
        abort_after=2,
        root=tmp_path / "bars",
    )
    append_run(original, path=path)

    rebuilt = last_run(read_manifest(path=path))

    assert rebuilt is not None
    assert rebuilt.summary() == original.summary()
    assert [r.as_dict() | {"seconds": 0} for r in rebuilt.results] == [
        r.as_dict() | {"seconds": 0} for r in original.results
    ]
    assert rebuilt.stopped_early == "2 consecutive failures"
    assert [r.status for r in rebuilt.skipped] == [Status.SKIPPED]


def test_the_last_run_is_the_latest_started_not_the_last_written(tmp_path: Path) -> None:
    path = tmp_path / "ingest_runs.parquet"
    later = dt.datetime(2026, 9, 29, 20, tzinfo=dt.UTC)
    earlier = dt.datetime(2026, 9, 28, 20, tzinfo=dt.UTC)
    for run_id, started in (("B-later", later), ("A-earlier", earlier)):
        append_run(
            IngestReport(
                results=[TickerIngest(ticker="AAA", status=Status.OK)],
                started_at=started,
                finished_at=started,
                run_id=run_id,
            ),
            path=path,
        )

    assert last_run_id(read_manifest(path=path)) == "B-later"


def test_an_empty_manifest_has_no_last_run(tmp_path: Path) -> None:
    frame = read_manifest(path=tmp_path / "missing.parquet")

    assert frame.empty and tuple(frame.columns) == MANIFEST_COLUMNS
    assert last_run(frame) is None


@pytest.mark.usefixtures("data_dir")
def test_report_with_nothing_recorded_fails_clearly() -> None:
    result = CliRunner().invoke(app, ["ingest", "--report"])

    assert result.exit_code == 1
    assert "no runs recorded" in result.output


def test_a_dry_run_is_not_recorded(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "fetch_daily", bars_for)

    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "-t",
            "AAA",
            "--start",
            "2020-08-24",
            "--end",
            "2020-09-04",
            "--pause",
            "0",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (data_dir / "meta" / "ingest_runs.parquet").exists()


@pytest.mark.usefixtures("data_dir")
def test_a_complete_last_run_reports_with_exit_code_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "fetch_daily", bars_for)
    argv = "ingest -t AAA --start 2020-08-24 --end 2020-09-04 --pause 0"
    assert CliRunner().invoke(app, argv.split()).exit_code == 0

    result = CliRunner().invoke(app, ["ingest", "--report"])

    assert result.exit_code == 0, result.output
    assert "1 of 1 tickers ingested (ok=1)" in result.output
