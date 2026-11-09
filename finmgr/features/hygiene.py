"""Feature hygiene: coverage, redundancy, and the short list the model reads.

    from finmgr.features.hygiene import model_features, model_columns

    model_features()            # ['ret_21d', 'mom_12_1', ...] - the base names
    model_columns()             # their per-date `_rank` columns in the panel

The panel (day 31) keeps every column days 23-30 built. Most of them are
near-copies of each other - EMA and SMA distances, a 20-day slope and a 21-day
return, log and simple returns - and a model given forty correlated inputs
spreads its weight across them and overfits the differences.
`finmgr/features/selection.yaml` decides, for every `feature` and `flag` column
of the registry, whether the model reads it (`model`), uses it to restrict the
universe (`filter`) or ignores it (`drop`), with a written reason for each.

`finmgr features --hygiene` measures the stored panel and checks that decision:

* :func:`nan_coverage` - the share of rows with a value, per feature per year;
* :func:`rank_correlation` - pooled Pearson on per-date ranks, i.e. the
  n-weighted average of each date's Spearman correlation (every date's ranks
  have the same mean and variance, so there is no between-date component);
* :func:`correlation_prune` - the roadmap's mechanical rule: walking the
  registry from simple to derived, drop any feature whose |rho| with a simpler
  surviving one exceeds the threshold (0.95);
* :func:`check_selection` - no `model` feature is one the rule drops, no two
  `model` features exceed the threshold, every `model` feature is covered after
  its warm-up year;
* :func:`explained_share` - how much of each `model` feature the others explain
  (R² of a pooled regression on ranks), the multi-feature version of a pairwise rho.

The registry's order is the simplicity order: it runs from returns to the
transforms built on them and ends with the market and sector relatives, so
between a near-duplicate pair the one listed first is the simpler.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from finmgr.features.crosssection import RANK_SUFFIX, cross_rank
from finmgr.features.panel import REGISTRY_PATH, load_registry

SELECTION_PATH = Path(__file__).with_name("selection.yaml")

USES = frozenset({"model", "filter", "drop"})

#: Share of rows a `model` feature must have a value for in every year but the panel's first.
MIN_COVERAGE = 0.95

#: Pairs at least this correlated are listed in the report.
REPORT_RHO = 0.8


@dataclass(frozen=True)
class Selection:
    threshold: float
    entries: tuple[dict[str, str], ...]

    def names(self, use: str) -> list[str]:
        return [e["name"] for e in self.entries if e["use"] == use]

    @property
    def model(self) -> list[str]:
        return self.names("model")

    @property
    def why(self) -> dict[str, str]:
        return {e["name"]: e["why"] for e in self.entries}


@dataclass(frozen=True)
class Drop:
    feature: str
    partner: str
    rho: float


# ---------------------------------------------------------------------------
# The selection file
# ---------------------------------------------------------------------------


def candidate_columns(registry: Path | str = REGISTRY_PATH) -> list[str]:
    """The registry's `feature` and `flag` columns, in its (simplicity) order."""
    return [e["name"] for e in load_registry(registry) if e["kind"] in ("feature", "flag")]


def load_selection(
    path: Path | str = SELECTION_PATH, *, registry: Path | str = REGISTRY_PATH
) -> Selection:
    """The selection, validated: every registry feature and flag exactly once, in registry order."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("features"), list):
        raise ValueError(f"{path}: expected a mapping with a 'features' list")
    threshold = raw.get("threshold")
    if not isinstance(threshold, int | float) or not 0 < threshold < 1:
        raise ValueError(f"{path}: threshold must be a number in (0, 1), got {threshold!r}")
    entries: list[dict[str, str]] = []
    for i, entry in enumerate(raw["features"]):
        if not isinstance(entry, dict) or set(entry) != {"name", "use", "why"}:
            raise ValueError(f"{path}: entry {i} must have exactly name, use and why")
        if entry["use"] not in USES:
            raise ValueError(f"{path}: {entry['name']}: use {entry['use']!r} not in {sorted(USES)}")
        if not isinstance(entry["why"], str) or not entry["why"].strip():
            raise ValueError(f"{path}: {entry['name']}: every entry needs a reason")
        entries.append({k: str(entry[k]) for k in ("name", "use", "why")})

    names = [e["name"] for e in entries]
    expected = candidate_columns(registry)
    dupes = sorted({n for n in names if names.count(n) > 1})
    missing = [n for n in expected if n not in names]
    unknown = [n for n in names if n not in expected]
    if dupes or missing or unknown:
        raise ValueError(
            f"{path} must list every registry feature and flag once: "
            f"duplicated {dupes}, missing {missing}, not features in the registry {unknown}"
        )
    if names != expected:
        raise ValueError(f"{path}: entries must follow the registry's order")
    if not any(e["use"] == "model" for e in entries):
        raise ValueError(f"{path}: no feature has use: model")
    return Selection(threshold=float(threshold), entries=tuple(entries))


def model_features(path: Path | str = SELECTION_PATH) -> list[str]:
    """The base names of the features the model reads, in registry order."""
    return load_selection(path).model


def model_columns(suffix: str = RANK_SUFFIX, path: Path | str = SELECTION_PATH) -> list[str]:
    """The panel columns the model reads: each model feature with `suffix` (`_rank` or `_z`)."""
    return [f"{name}{suffix}" for name in model_features(path)]


def filter_columns(path: Path | str = SELECTION_PATH) -> list[str]:
    return load_selection(path).names("filter")


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _dates(panel: pd.DataFrame) -> pd.Series:
    if "date" in panel.columns:
        return pd.Series(panel["date"].to_numpy(), index=panel.index)
    return pd.Series(panel.index.get_level_values("date"), index=panel.index)


def nan_coverage(panel: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Share of rows with a value: one row per column, one column per calendar year."""
    years = pd.to_datetime(_dates(panel)).dt.year.rename("year")
    cover = panel[list(columns)].notna().groupby(years.to_numpy()).mean()
    cover.index.name = "year"
    return cover.T


def per_date_ranks(panel: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Each column ranked on [0, 1] within its date (flags as 0/1)."""
    dates = _dates(panel)
    return pd.DataFrame(
        {c: cross_rank(panel[c].astype(float), dates) for c in columns}, index=panel.index
    )


def rank_correlation(panel: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Pooled Pearson correlation of per-date ranks over pairwise-complete rows."""
    return per_date_ranks(panel, columns).corr()


def correlation_prune(
    corr: pd.DataFrame, order: Sequence[str], threshold: float
) -> tuple[list[str], list[Drop]]:
    """Walk `order`, dropping each feature whose |rho| with a kept earlier one exceeds `threshold`.

    Returns the survivors and, for every drop, the simpler feature it
    duplicates and their correlation. A feature with no defined correlation
    against a kept one (a constant, or no overlap) is kept.
    """
    kept: list[str] = []
    dropped: list[Drop] = []
    for feature in order:
        rhos = corr.loc[feature, kept].abs() if kept else pd.Series(dtype=float)
        rhos = rhos.dropna()
        if not rhos.empty and rhos.max() > threshold:
            partner = str(rhos.idxmax())
            dropped.append(Drop(feature, partner, float(corr.loc[feature, partner])))
        else:
            kept.append(feature)
    return kept, dropped


def explained_share(ranks: pd.DataFrame) -> pd.Series:
    """R² of each column regressed on all the others, on rows where every column has a value."""
    complete = ranks.dropna()
    if len(complete) <= ranks.shape[1]:
        raise ValueError("too few complete rows to regress the features on each other")
    centred = complete - complete.mean()
    out = {}
    for column in centred.columns:
        y = centred[column].to_numpy()
        x = centred.drop(columns=column).to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        total = float(y @ y)
        out[column] = 1.0 - float(((y - x @ beta) ** 2).sum()) / total if total > 0 else np.nan
    return pd.Series(out, name="r2")


def high_pairs(corr: pd.DataFrame, floor: float = REPORT_RHO) -> list[tuple[str, str, float]]:
    """Every pair with |rho| ≥ `floor`, strongest first."""
    names = list(corr.columns)
    pairs = [
        (a, b, float(corr.loc[a, b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= floor
    ]
    return sorted(pairs, key=lambda p: -abs(p[2]))


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def check_selection(
    selection: Selection,
    corr: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    order: Sequence[str] | None = None,
    min_coverage: float = MIN_COVERAGE,
) -> list[str]:
    """Every way the measured panel contradicts the selection; empty when it holds."""
    problems: list[str] = []
    model = selection.model
    order = list(order) if order is not None else [e["name"] for e in selection.entries]
    _, dropped = correlation_prune(corr, order, selection.threshold)
    for drop in dropped:
        if drop.feature in model:
            problems.append(
                f"{drop.feature} is a model feature but |rho| {abs(drop.rho):.3f} > "
                f"{selection.threshold} with the simpler {drop.partner}"
            )
    for i, a in enumerate(model):
        for b in model[i + 1 :]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(rho) > selection.threshold:
                problems.append(f"model features {a} and {b} have |rho| {abs(rho):.3f}")
    later = coverage.columns[1:]  # the first year is every long window's warm-up
    for name in model:
        if name not in coverage.index or len(later) == 0:
            continue
        thin = coverage.loc[name, later]
        thin = thin[thin < min_coverage]
        for year, share in thin.items():
            problems.append(f"{name} has values on only {share:.0%} of {year} rows")
    return problems


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Hygiene:
    selection: Selection
    coverage: pd.DataFrame
    corr: pd.DataFrame
    survivors: list[str]
    dropped: list[Drop]
    explained: pd.Series
    problems: list[str]


def run_hygiene(
    panel: pd.DataFrame,
    *,
    selection: Selection | None = None,
    registry: Path | str = REGISTRY_PATH,
) -> Hygiene:
    """Measure `panel` (indexed by `(date, ticker)`) and check the selection against it."""
    chosen = selection or load_selection(registry=registry)
    order = [e["name"] for e in chosen.entries]
    ranks = per_date_ranks(panel, order)
    corr = ranks.corr()
    coverage = nan_coverage(panel, order)
    survivors, dropped = correlation_prune(corr, order, chosen.threshold)
    explained = explained_share(ranks[chosen.model])
    problems = check_selection(chosen, corr, coverage, order=order)
    return Hygiene(chosen, coverage, corr, survivors, dropped, explained, problems)


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def render_report(h: Hygiene, *, as_of: str | None = None) -> str:
    """The hygiene measurements as markdown (what `docs/feature_hygiene.md` holds)."""
    sel = h.selection
    model = sel.model
    uses = {e["name"]: e["use"] for e in sel.entries}
    out = [
        "# Feature hygiene",
        "",
        "Generated by `finmgr features --hygiene`"
        + (f" from the panel as of {as_of}" if as_of else "")
        + ". Do not edit by hand: change `finmgr/features/selection.yaml` and rerun.",
        "",
        f"**{len(h.corr)} candidate columns -> {len(model)} model features"
        f" + {len(sel.names('filter'))} filter.** "
        f"The mechanical {sel.threshold} rule alone keeps {len(h.survivors)}; "
        "the rest of the cut is judgement, written down per feature below.",
        "",
        "Correlations are pooled Pearson on per-date ranks (= the n-weighted mean "
        "of each date's Spearman rho). Nothing here looks at forward returns.",
        "",
        "## Check",
        "",
    ]
    if h.problems:
        out += [f"- FAIL: {p}" for p in h.problems]
    else:
        out.append(
            f"PASS - no model feature is dropped by the {sel.threshold} rule, no two exceed "
            f"it, and every model feature has ≥ {MIN_COVERAGE:.0%} coverage after "
            "its warm-up year."
        )

    out += ["", "## The model set", ""]
    others = {
        m: h.corr.loc[m, [x for x in model if x != m]].abs().sort_values(ascending=False)
        for m in model
    }
    out += _table(
        ["feature", "max abs rho with another", "R² on the others", "why"],
        [
            (
                f"`{m}`",
                f"{others[m].iloc[0]:.2f} (`{others[m].index[0]}`)" if len(others[m]) else "-",
                f"{h.explained[m]:.2f}",
                sel.why[m],
            )
            for m in model
        ],
    )
    out += ["", "Rank correlations within the model set:", ""]
    sub = h.corr.loc[model, model]
    out += _table(
        ["", *(f"`{m}`" for m in model)],
        [(f"`{a}`", *(f"{sub.loc[a, b]:+.2f}" for b in model)) for a in model],
    )
    for name in sel.names("filter"):
        out += ["", f"Filter: `{name}` - {sel.why[name]}"]

    out += ["", f"## Mechanical rule: |rho| > {sel.threshold} with a simpler feature", ""]
    out += _table(
        ["dropped", "simpler twin", "rho"],
        [(f"`{d.feature}`", f"`{d.partner}`", f"{d.rho:+.3f}") for d in h.dropped],
    )

    out += ["", "## Every candidate", ""]
    out += _table(
        ["column", "use", "reason"],
        [(f"`{e['name']}`", e["use"], e["why"]) for e in sel.entries],
    )

    out += ["", f"## Pairs with |rho| ≥ {REPORT_RHO}", ""]
    out += _table(
        ["a", "b", "rho", "uses"],
        [
            (f"`{a}`", f"`{b}`", f"{r:+.3f}", f"{uses[a]} / {uses[b]}")
            for a, b, r in high_pairs(h.corr)
        ],
    )

    out += ["", "## NaN coverage per year (share of rows with a value)", ""]
    years = [str(y) for y in h.coverage.columns]
    out += _table(
        ["column", *years],
        [(f"`{name}`", *(f"{v:.2f}" for v in h.coverage.loc[name])) for name in h.coverage.index],
    )
    out.append("")
    return "\n".join(out)
