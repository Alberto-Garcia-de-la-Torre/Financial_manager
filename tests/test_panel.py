"""Day 31: the feature panel, its registry, and full vs incremental builds."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.data.store import write_bars
from finmgr.data.universe import UNIVERSE_COLUMNS
from finmgr.features import panel as P
from finmgr.features.crosssection import FEATURE_COLUMNS, rank_column, z_column
from finmgr.features.returns import venue_sessions

SESSIONS = venue_sessions("XNYS", dt.date(2022, 1, 1), dt.date(2024, 6, 28))
N = 400
SECTORS = ("Energy", "Tech", "Utilities")
TICKERS = tuple(f"T{i:02d}" for i in range(12))


# ---------------------------------------------------------------------------
# The registry on its own
# ---------------------------------------------------------------------------


def test_the_registry_loads_and_every_entry_is_complete() -> None:
    entries = P.load_registry()
    assert [e["name"] for e in entries[:2]] == list(P.INDEX)
    for entry in entries:
        assert set(entry) == set(P.REGISTRY_FIELDS)
        assert entry["direction"] in P.DIRECTIONS


def test_every_normalized_feature_has_its_rank_and_z_registered() -> None:
    names = set(P.registry_columns())
    for feature in FEATURE_COLUMNS:
        assert {feature, rank_column(feature), z_column(feature)} <= names


def test_excluded_columns_are_not_registered() -> None:
    assert not set(P.EXCLUDED) & set(P.registry_columns())


def test_check_registry_names_the_difference() -> None:
    columns = P.registry_columns()
    P.check_registry(columns)
    with pytest.raises(P.RegistryMismatch, match="not in the registry: extra"):
        P.check_registry([*columns, "extra"])
    with pytest.raises(P.RegistryMismatch, match="registered but absent: rsi_14"):
        P.check_registry([c for c in columns if c != "rsi_14"])
    with pytest.raises(P.RegistryMismatch, match="different order"):
        P.check_registry([columns[1], columns[0], *columns[2:]])


def write_registry(tmp_path: Path, drop: str | None = None, add: dict | None = None) -> Path:
    raw = yaml.safe_load(P.REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = [e for e in raw["columns"] if e["name"] not in {drop, f"{drop}_rank", f"{drop}_z"}]
    if add is not None:
        entries.append(add)
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump({"columns": entries}, sort_keys=False), encoding="utf-8")
    return path


def test_a_bad_registry_entry_is_refused(tmp_path: Path) -> None:
    bad = {
        "name": "x",
        "kind": "feature",
        "source": "s",
        "formula": "f",
        "window": 0,
        "direction": "neutral",
    }
    with pytest.raises(ValueError, match="window"):
        P.load_registry(write_registry(tmp_path, add=bad))
    orphan = {**bad, "name": "nothing_rank", "kind": "rank", "window": None}
    with pytest.raises(ValueError, match="no registered base"):
        P.load_registry(write_registry(tmp_path, add=orphan))
    duplicate = {**bad, "name": "rsi_14", "window": 14}
    with pytest.raises(ValueError, match="duplicate"):
        P.load_registry(write_registry(tmp_path, add=duplicate))


# ---------------------------------------------------------------------------
# Through a small store
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=UNIVERSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def bars(ticker: str, closes: np.ndarray, volume: np.ndarray, rng: np.random.Generator):
    spread = 1 + np.abs(rng.normal(0, 0.01, len(closes)))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": SESSIONS[: len(closes)],
            "open": closes,
            "high": closes * spread,
            "low": closes / spread,
            "close": closes,
            "volume": volume,
            "close_eur": closes,
        }
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Twelve tickers in three sectors on XNYS, plus a benchmark."""
    rng = np.random.default_rng(31)
    data = tmp_path / "data"
    r_m = rng.normal(0.0003, 0.01, N)
    for i, ticker in enumerate(TICKERS):
        r = (0.5 + 0.1 * i) * r_m + rng.normal(0.0001 * (i - 6), 0.012, N)
        r[0] = 0.0
        closes = (20 + 10 * i) * np.exp(np.cumsum(r))
        volume = rng.lognormal(11 + 0.3 * i, 0.4, N)
        write_bars(bars(ticker, closes, volume, rng), root=data / "bars" / "daily")
    bench = 100 * np.exp(np.cumsum(np.r_[0.0, r_m[1:]]))
    write_bars(
        bars("BENCH", bench, np.full(N, 1e6), rng), root=data / "benchmarks" / "bars" / "daily"
    )
    company = {"exchange": "XNYS", "currency": "EUR", "country": "X"}
    universe = write_csv(
        tmp_path / "universe.csv",
        [
            {"ticker": t, "name": t, "sector": SECTORS[i % 3], **company}
            for i, t in enumerate(TICKERS)
        ],
    )
    benchmarks = write_csv(
        tmp_path / "benchmarks.csv",
        [{"ticker": "BENCH", "name": "B", "sector": "Benchmark", **company}],
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
                "history_start": "2022-01-01",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return data


LAST = SESSIONS[N - 1]
MIDDLE = SESSIONS[N - 40]


@pytest.mark.usefixtures("store")
def test_a_full_build_has_exactly_the_registry_columns() -> None:
    result = P.build_panel(as_of=LAST, full=True)
    assert result.mode == "full"
    assert result.path == P.panel_path()
    panel = P.read_panel()
    assert list(panel.index.names) == list(P.INDEX)
    assert [*panel.index.names, *panel.columns] == P.registry_columns()
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    assert len(panel) == len(TICKERS) * N
    assert set(panel["sector"]) == set(SECTORS)
    assert P.panel_metadata(result.path)["registry"] == P.registry_hash()
    # The panel has real values, not just the right headers.
    late = panel.xs(LAST, level="date")
    for column in ("mom_12_1", "beta_252", "rsi_14", "mom_12_1_sec", "mom_12_1_rank"):
        assert late[column].notna().all(), column
    np.testing.assert_allclose(late["mom_12_1_rank"].mean(), 0.5)


@pytest.mark.usefixtures("store")
def test_an_incremental_build_equals_a_full_one(tmp_path: Path) -> None:
    full_path = tmp_path / "full.parquet"
    P.build_panel(as_of=LAST, full=True, path=full_path)

    first = P.build_panel(as_of=MIDDLE)  # no panel yet: a full build
    assert first.mode == "full" and first.last == MIDDLE
    second = P.build_panel(as_of=LAST)
    assert second.mode == "incremental"
    assert second.written_from == MIDDLE - dt.timedelta(days=P.OVERLAP_DAYS)
    assert second.last == LAST

    pd.testing.assert_frame_equal(P.read_panel(), P.read_panel(path=full_path))


@pytest.mark.usefixtures("store")
def test_an_incremental_build_keeps_rows_before_the_overlap() -> None:
    P.build_panel(as_of=MIDDLE, full=True)
    before = P.read_panel()
    P.build_panel(as_of=LAST)
    after = P.read_panel()
    cut = MIDDLE - dt.timedelta(days=P.OVERLAP_DAYS)
    old = before[before.index.get_level_values("date") < cut]
    pd.testing.assert_frame_equal(after.loc[old.index], old)


@pytest.mark.usefixtures("store")
def test_an_incremental_build_refuses_a_panel_from_the_future() -> None:
    P.build_panel(as_of=LAST, full=True)
    with pytest.raises(ValueError, match="after as_of"):
        P.build_panel(as_of=MIDDLE)
    assert P.build_panel(as_of=MIDDLE, full=True).last == MIDDLE


@pytest.mark.usefixtures("store")
def test_an_incremental_build_refuses_a_panel_under_another_registry(tmp_path: Path) -> None:
    P.build_panel(as_of=MIDDLE, full=True)
    other = write_registry(tmp_path, drop="rsi_14")
    with pytest.raises(P.RegistryMismatch, match="another registry"):
        P.build_panel(as_of=LAST, registry=other)


@pytest.mark.usefixtures("store")
def test_a_computed_column_missing_from_the_registry_fails_the_build(tmp_path: Path) -> None:
    other = write_registry(tmp_path, drop="rsi_14")
    with pytest.raises(P.RegistryMismatch, match="not in the registry: rsi_14"):
        P.build_panel(as_of=LAST, full=True, registry=other)
    assert not P.panel_path().exists()


@pytest.mark.usefixtures("store")
def test_a_registered_column_nothing_computes_fails_the_build(tmp_path: Path) -> None:
    ghost = {
        "name": "ghost",
        "kind": "feature",
        "source": "nowhere",
        "formula": "nothing",
        "window": None,
        "direction": "neutral",
    }
    other = write_registry(tmp_path, add=ghost)
    with pytest.raises(P.RegistryMismatch, match="registered but absent: ghost"):
        P.build_panel(as_of=LAST, full=True, registry=other)


@pytest.mark.usefixtures("store")
def test_the_cli_rebuilds_from_scratch_and_checks_the_registry() -> None:
    runner = CliRunner()
    built = runner.invoke(app, ["features", "--full", "--as-of", LAST.isoformat()])
    assert built.exit_code == 0, built.output
    assert "full build" in built.output
    checked = runner.invoke(app, ["features", "--check"])
    assert checked.exit_code == 0, checked.output
    assert "match the registry" in checked.output
    again = runner.invoke(app, ["features", "--as-of", LAST.isoformat()])
    assert again.exit_code == 0, again.output
    assert "incremental build" in again.output


@pytest.mark.usefixtures("store")
def test_the_cli_check_fails_without_a_panel() -> None:
    result = CliRunner().invoke(app, ["features", "--check"])
    assert result.exit_code == 1
