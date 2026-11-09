"""Day 32: NaN coverage, rank correlations, the 0.95 rule and the model feature set."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from finmgr.cli import app
from finmgr.config import SETTINGS_PATH_ENV
from finmgr.features import hygiene as H
from finmgr.features import panel as P
from finmgr.features.crosssection import FEATURE_COLUMNS, RANK_SUFFIX, Z_SUFFIX

DATES = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(0, 730, 7)]
TICKERS = [f"T{i:02d}" for i in range(30)]


def frame(columns: dict[str, np.ndarray]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([DATES, TICKERS], names=list(P.INDEX))
    return pd.DataFrame(columns, index=index)


def noise(seed: int, n: int = len(DATES) * len(TICKERS)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


# ---------------------------------------------------------------------------
# The selection file as shipped
# ---------------------------------------------------------------------------


def test_the_shipped_selection_covers_every_registry_feature_once() -> None:
    selection = H.load_selection()
    assert [e["name"] for e in selection.entries] == H.candidate_columns()
    assert selection.threshold == 0.95
    assert all(e["why"].strip() for e in selection.entries)


def test_the_model_set_is_small_and_reads_normalized_columns() -> None:
    model = H.model_features()
    assert 8 <= len(model) <= 15, "a set small enough to defend one by one"
    registered = set(P.registry_columns())
    for name in model:
        assert name in FEATURE_COLUMNS, f"{name} has no per-date rank to feed the model"
    assert set(H.model_columns()) <= registered
    assert set(H.model_columns(Z_SUFFIX)) <= registered
    assert all(c.endswith(RANK_SUFFIX) for c in H.model_columns())


def test_tradable_filters_and_is_never_a_model_input() -> None:
    assert H.filter_columns() == ["tradable"]
    assert "tradable" not in H.model_features()


def test_the_baseline_and_sizing_features_are_kept() -> None:
    # Day 36 benchmarks plain 12-1 momentum; day 52 sizes by volatility.
    assert {"mom_12_1", "vol_63d"} <= set(H.model_features())


def write_selection(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(H.SELECTION_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_a_feature_left_out_of_the_selection_is_refused(tmp_path: Path) -> None:
    path = write_selection(tmp_path, lambda r: r["features"].pop(0))
    with pytest.raises(ValueError, match=r"missing \['ret_1d'\]"):
        H.load_selection(path)


def test_an_unknown_or_duplicated_feature_is_refused(tmp_path: Path) -> None:
    def add(raw):
        raw["features"].append({"name": "nope", "use": "drop", "why": "x"})
        raw["features"].append(dict(raw["features"][0]))

    with pytest.raises(ValueError, match=r"duplicated \['ret_1d'\].*\['nope'\]"):
        H.load_selection(write_selection(tmp_path, add))


def test_an_entry_without_a_reason_or_a_bad_use_is_refused(tmp_path: Path) -> None:
    def blank(raw):
        raw["features"][0]["why"] = " "

    with pytest.raises(ValueError, match="needs a reason"):
        H.load_selection(write_selection(tmp_path, blank))

    def bad_use(raw):
        raw["features"][0]["use"] = "maybe"

    with pytest.raises(ValueError, match="use 'maybe'"):
        H.load_selection(write_selection(tmp_path, bad_use))


def test_entries_out_of_registry_order_are_refused(tmp_path: Path) -> None:
    def swap(raw):
        f = raw["features"]
        f[0], f[1] = f[1], f[0]

    with pytest.raises(ValueError, match="registry's order"):
        H.load_selection(write_selection(tmp_path, swap))


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def test_nan_coverage_is_the_share_of_rows_with_a_value_per_year() -> None:
    a = noise(1)
    years = np.array([d.year for d in DATES for _ in TICKERS])
    n2020 = int((years == 2020).sum())
    a[years == 2020] = np.where(np.arange(n2020) % 4 == 0, np.nan, 1.0)
    cover = H.nan_coverage(frame({"a": a, "b": noise(2)}), ["a", "b"])
    assert list(cover.columns) == [2020, 2021]
    assert cover.loc["a", 2020] == pytest.approx(1 - np.ceil(n2020 / 4) / n2020)
    assert cover.loc["a", 2021] == 1.0
    assert (cover.loc["b"] == 1.0).all()


def test_rank_correlation_ignores_monotone_transforms_and_scale() -> None:
    a = 0.1 * noise(3)  # simple returns, all above -100%
    panel = frame({"ret": a, "logret": np.log1p(a), "neg": -3 * a})
    corr = H.rank_correlation(panel, ["ret", "logret", "neg"])
    assert corr.loc["ret", "logret"] == pytest.approx(1.0, abs=1e-12)
    assert corr.loc["ret", "neg"] == pytest.approx(-1.0, abs=1e-12)


def test_pooled_rank_correlation_is_the_mean_per_date_spearman() -> None:
    a, e = noise(4), noise(5)
    panel = frame({"a": a, "b": a + 2 * e})
    pooled = H.rank_correlation(panel, ["a", "b"]).loc["a", "b"]
    per_date = panel.groupby(level="date").apply(lambda g: g["a"].corr(g["b"], method="spearman"))
    assert pooled == pytest.approx(per_date.mean(), abs=1e-12)


def test_a_level_shared_across_dates_does_not_create_correlation() -> None:
    # Both columns trend with the date but are independent within each date:
    # a pooled correlation of raw values would be near 1, of per-date ranks near 0.
    level = np.repeat(np.arange(len(DATES), dtype=float), len(TICKERS))
    panel = frame({"a": level * 10 + noise(6), "b": level * 10 + noise(7)})
    assert panel["a"].corr(panel["b"]) > 0.99
    assert abs(H.rank_correlation(panel, ["a", "b"]).loc["a", "b"]) < 0.05


# ---------------------------------------------------------------------------
# The 0.95 rule and the check
# ---------------------------------------------------------------------------


def toy() -> pd.DataFrame:
    a, c = noise(8), noise(9)
    return frame(
        {
            "a": a,
            "a_twin": a + 0.05 * noise(10),  # rho ≈ 0.998 with a
            "a_cousin": a + 0.8 * noise(11),  # rho ≈ 0.78 with a
            "c": c,
        }
    )


def test_the_prune_drops_the_later_twin_and_names_the_simpler_one() -> None:
    order = ["a", "a_twin", "a_cousin", "c"]
    corr = H.rank_correlation(toy(), order)
    kept, dropped = H.correlation_prune(corr, order, 0.95)
    assert kept == ["a", "a_cousin", "c"]
    assert [(d.feature, d.partner) for d in dropped] == [("a_twin", "a")]
    assert dropped[0].rho > 0.95
    # Order is simplicity: listing the twin first keeps it instead.
    kept, dropped = H.correlation_prune(corr, ["a_twin", "a", "a_cousin", "c"], 0.95)
    assert kept[0] == "a_twin" and dropped[0].feature == "a"


def test_negative_correlation_counts_as_redundant() -> None:
    a = noise(12)
    corr = H.rank_correlation(frame({"ret": a, "rev": -a}), ["ret", "rev"])
    kept, dropped = H.correlation_prune(corr, ["ret", "rev"], 0.95)
    assert kept == ["ret"] and dropped[0].rho == pytest.approx(-1.0)


def selection(model: list[str], names: list[str]) -> H.Selection:
    return H.Selection(
        threshold=0.95,
        entries=tuple(
            {"name": n, "use": "model" if n in model else "drop", "why": "x"} for n in names
        ),
    )


def test_check_passes_a_selection_of_distinct_features() -> None:
    names = ["a", "a_twin", "a_cousin", "c"]
    panel = toy()
    sel = selection(["a", "c"], names)
    corr = H.rank_correlation(panel, names)
    assert H.check_selection(sel, corr, H.nan_coverage(panel, names)) == []


def test_check_refuses_a_model_feature_the_rule_drops() -> None:
    names = ["a", "a_twin", "a_cousin", "c"]
    panel = toy()
    corr = H.rank_correlation(panel, names)
    problems = H.check_selection(
        selection(["a_twin", "c"], names), corr, H.nan_coverage(panel, names)
    )
    assert problems and "a_twin is a model feature" in problems[0]
    problems = H.check_selection(
        selection(["a", "a_twin"], names), corr, H.nan_coverage(panel, names)
    )
    assert any("model features a and a_twin" in p for p in problems)


def test_check_refuses_a_thin_model_feature_but_forgives_the_warm_up_year() -> None:
    panel = toy()
    years = np.array([d.year for d in panel.index.get_level_values("date")])
    panel.loc[years == 2020, "c"] = np.nan  # warm-up: fine
    names = list(panel.columns)
    corr = H.rank_correlation(panel, names)
    sel = selection(["a", "c"], names)
    assert H.check_selection(sel, corr, H.nan_coverage(panel, names)) == []
    panel.loc[(years == 2021) & (np.arange(len(panel)) % 2 == 0), "c"] = np.nan
    problems = H.check_selection(sel, corr, H.nan_coverage(panel, names))
    assert problems == ["c has values on only 50% of 2021 rows"]


def test_explained_share_sees_a_feature_spread_across_two_others() -> None:
    b, c = noise(13), noise(14)
    panel = frame({"b": b, "c": c, "sum": b + c, "free": noise(15)})
    ranks = H.per_date_ranks(panel, list(panel.columns))
    r2 = H.explained_share(ranks)
    assert r2["sum"] > 0.9  # no pairwise rho above ~0.7, yet almost fully explained
    assert H.rank_correlation(panel, ["b", "sum"]).loc["b", "sum"] < 0.75
    assert r2["free"] < 0.01


# ---------------------------------------------------------------------------
# The CLI on a synthetic panel with the registry's columns
# ---------------------------------------------------------------------------


@pytest.fixture
def stored_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        yaml.safe_dump({"data_dir": str(data), "log_dir": str(tmp_path / "logs")}),
        encoding="utf-8",
    )
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(settings))
    return data / "features" / P.PANEL_FILE


def write_panel(path: Path, override: dict[str, str] | None = None) -> None:
    names = H.candidate_columns()
    columns = {n: noise(100 + i) for i, n in enumerate(names)}
    for target, source in (override or {}).items():
        columns[target] = columns[source]
    panel = frame(columns)
    panel["tradable"] = panel["tradable"] > 0
    P._write(panel, path, {"as_of": "2021-12-31"})


def test_cli_hygiene_writes_the_report_and_the_matrix(stored_panel: Path, tmp_path: Path) -> None:
    write_panel(stored_panel)
    report = tmp_path / "hygiene.md"
    result = CliRunner().invoke(app, ["features", "--hygiene", "--report", str(report)])
    assert result.exit_code == 0, result.output
    text = report.read_text(encoding="utf-8")
    assert "PASS" in text
    for name in H.model_features():
        assert f"| `{name}` |" in text
    matrix = pd.read_csv(stored_panel.with_name("rank_correlation.csv"), index_col=0)
    assert list(matrix.columns) == H.candidate_columns()


def test_cli_hygiene_fails_when_two_model_features_collapse(
    stored_panel: Path, tmp_path: Path
) -> None:
    model = H.model_features()
    write_panel(stored_panel, {model[1]: model[0]})
    report = tmp_path / "hygiene.md"
    result = CliRunner().invoke(app, ["features", "--hygiene", "--report", str(report)])
    assert result.exit_code == 1
    assert f"FAIL: {model[1]} is a model feature" in report.read_text(encoding="utf-8")


@pytest.mark.usefixtures("stored_panel")
def test_cli_hygiene_without_a_panel_fails() -> None:
    result = CliRunner().invoke(app, ["features", "--hygiene"])
    assert result.exit_code == 1
