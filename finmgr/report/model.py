"""Day 42: the pieces of a model's report that a baseline's does not have.

* :func:`learned_html` / :func:`learned_markdown` — what the model learnt: its
  coefficients across the walk-forward folds (for a linear model) and the
  out-of-sample information coefficient of its scores.
* :func:`model_page` — the Markdown page `finmgr train --doc` writes under
  `docs/models/`, so the comparison with the baselines is in the repository
  next to `docs/baselines.md` and not only in a local HTML file.
"""

from __future__ import annotations

import html
import math
from datetime import date

import pandas as pd

from finmgr.models.compare import Comparison, comparison_markdown
from finmgr.models.walkforward import WalkForward


def coefficient_table(wf: WalkForward) -> pd.DataFrame:
    """Per feature: mean, last fold, min, max weight and the share of folds it was positive.

    Sorted by |mean|. Empty for a model without coefficients.
    """
    coef = wf.coefficients()
    if coef.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "mean": coef.mean(),
            "last": coef.iloc[-1],
            "min": coef.min(),
            "max": coef.max(),
            "positive": (coef > 0).mean(),
        }
    )
    return out.reindex(out["mean"].abs().sort_values(ascending=False).index)


def importance_table(wf: WalkForward) -> pd.DataFrame:
    """Per feature: mean, last fold, min and max share of the total gain. Sorted by mean.

    Empty for a model without importances (ridge reports coefficients instead).
    """
    imp = wf.importances()
    if imp.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {"mean": imp.mean(), "last": imp.iloc[-1], "min": imp.min(), "max": imp.max()}
    )
    return out.sort_values("mean", ascending=False)


IMPORTANCE_NOTE = (
    "Share of the total split gain per feature: how much of the loss reduction the "
    "trees owe to it. It says which features the model used, not in which direction, "
    "and a share is not a weight: it cannot be compared with a linear coefficient."
)


def _feature(column: str) -> str:
    return column.removesuffix("_rank")


def _ic_line(wf: WalkForward) -> str:
    s = wf.ic_summary()
    if not s["dates"]:
        return "No test date has a realised label yet."
    return (
        f"Out-of-sample rank IC (Spearman of score vs realised {wf.label}, per date): "
        f"mean {s['mean']:+.3f}, std {s['std']:.3f}, IC/std {s['ir']:+.2f}, positive on "
        f"{s['positive']:.0%} of {int(s['dates']):,} test dates. "
        f"Naive t {s['t']:.1f} treats the dates as independent; the 21-session labels "
        f"overlap, so divide by roughly √21 ≈ 4.6 for an honest one ({s['t'] / 21**0.5:.1f})."
    )


def _fold_line(wf: WalkForward) -> str:
    first, last = wf.folds[0], wf.folds[-1]
    return (
        f"{len(wf.folds)} folds, refitted every {wf.test_size} dates; expanding window from "
        f"{first.train_start}; first test {first.test_start} (trained on "
        f"{first.n_train:,} rows to {first.train_end}), last test ending {last.test_end} "
        f"(trained on {last.n_train:,} rows). Embargo {wf.horizon} dates, purged on each "
        f"date's latest label end across venues."
    )


def learned_html(wf: WalkForward) -> str:
    """The "What the model learnt" section body."""
    parts = [f'<p class="note">{html.escape(_fold_line(wf))}</p>']
    parts.append(f'<p class="note">{html.escape(_ic_line(wf))}</p>')
    table = coefficient_table(wf)
    if not table.empty:
        rows = "".join(
            f"<tr><th>{html.escape(_feature(name))}</th><td>{r['mean']:+.4f}</td>"
            f"<td>{r['last']:+.4f}</td><td>[{r['min']:+.4f}, {r['max']:+.4f}]</td>"
            f"<td>{r['positive']:.0%}</td></tr>"
            for name, r in table.iterrows()
        )
        parts.append(
            '<p class="note">Weights on per-date ranks (0 = lowest, 1 = highest that day): '
            "a weight of +0.01 means going from the bottom to the top of that feature adds "
            "one percentage point to the predicted 21-session excess return.</p>"
            '<table class="metrics"><thead><tr><th>Feature</th><th>mean weight</th>'
            "<th>last fold</th><th>range over folds</th><th>folds positive</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    imp = importance_table(wf)
    if not imp.empty:
        rows = "".join(
            f"<tr><th>{html.escape(_feature(name))}</th><td>{r['mean']:.1%}</td>"
            f"<td>{r['last']:.1%}</td><td>[{r['min']:.1%}, {r['max']:.1%}]</td></tr>"
            for name, r in imp.iterrows()
        )
        parts.append(
            f'<p class="note">{html.escape(IMPORTANCE_NOTE)}</p>'
            '<table class="metrics"><thead><tr><th>Feature</th><th>mean share of gain</th>'
            "<th>last fold</th><th>range over folds</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "".join(parts)


def learned_markdown(wf: WalkForward) -> str:
    lines = [_fold_line(wf), "", _ic_line(wf), ""]
    table = coefficient_table(wf)
    if not table.empty:
        lines += [
            "| Feature | Mean weight | Last fold | Range over folds | Folds positive |",
            "|---|---:|---:|---|---:|",
        ]
        lines += [
            f"| {_feature(name)} | {r['mean']:+.4f} | {r['last']:+.4f} | "
            f"[{r['min']:+.4f}, {r['max']:+.4f}] | {r['positive']:.0%} |"
            for name, r in table.iterrows()
        ]
    imp = importance_table(wf)
    if not imp.empty:
        lines += [
            IMPORTANCE_NOTE,
            "",
            "| Feature | Mean share of gain | Last fold | Range over folds |",
            "|---|---:|---:|---|",
        ]
        lines += [
            f"| {_feature(name)} | {r['mean']:.1%} | {r['last']:.1%} | "
            f"[{r['min']:.1%}, {r['max']:.1%}] |"
            for name, r in imp.iterrows()
        ]
    return "\n".join(lines) + "\n"


def model_page(
    wf: WalkForward,
    comp: Comparison,
    *,
    as_of: date,
    command: str,
    tearsheet: str | None = None,
    n_variants: int | None = None,
    familywise: tuple[float, float, float] | None = None,
) -> str:
    """The Markdown page for one trained model: setup, the comparison, what it learnt."""
    run = comp.model
    m = run.metrics
    params = ", ".join(f"{k}={v!r}" for k, v in wf.params.items()) or "defaults"
    fw = ""
    if n_variants is not None and familywise is not None:
        level, low, high = familywise
        fw = (
            f"\n\nWith {n_variants} variants now in `docs/trials.csv`, the familywise "
            f"({level:.2%}) interval on its Sharpe is **[{low:.2f}, {high:.2f}]**."
        )
    sheet = (
        f"\n\nThe HTML tearsheet (`{command} --report`) is written to `{tearsheet}` "
        "(under the git-ignored `data/`)."
        if tearsheet
        else ""
    )
    random_note = (
        f"The random row is the 95th percentile of the {len(comp.random)} seeds in "
        "`docs/baselines_random.csv` (same window)."
        if comp.random
        else "No random-book percentile: `docs/baselines_random.csv` covers another window."
    )
    nan = math.isnan
    window = "expanding" if wf.train_size is None else f"rolling {wf.train_size} dates"
    splits = (
        f"walk-forward, {window}, test {wf.test_size} dates, "
        f"min train {wf.min_train_size} dates, embargo {wf.horizon}"
    )
    turnover = "-" if nan(m.turnover) else f"{m.turnover:.2f}"
    return f"""# Model: {run.label}

Generated by `{command}` from the local store, data as of {as_of}. The code is
`finmgr/models/walkforward.py` (training) and `finmgr/models/compare.py` (this
comparison); rerun the command after any change to the model, the features, the
engine, costs or data, and commit the regenerated page.

## Verdict

**{comp.verdict}.** Sharpe {m.sharpe:.2f}, 95% interval
[{run.interval.low:.2f}, {run.interval.high:.2f}]; paired difference against
*{comp.reference}* {comp.vs_reference.point:+.2f}
[{comp.vs_reference.low:+.2f}, {comp.vs_reference.high:+.2f}].{fw}

## Against the baselines

Every row below went through the same engine on the same date axis
({m.start} → {m.end}), top {run.top_k}, held {run.hold} sessions, configured
costs, next-open fills, Sharpe against €STR/EONIA. {random_note}

{comparison_markdown(comp)}
The model turns over {turnover}x a year; compare the costs section of
`docs/baselines.md`: a book that trades this often starts behind equal weight.

## Setup

|  |  |
| --- | --- |
| Model | `{wf.model}` ({params}) |
| Features | {", ".join(f"`{_feature(f)}`" for f in wf.features)} — per-date ranks, missing → 0.5 |
| Label | `{wf.label}`: next {wf.horizon} sessions' return minus the universe mean |
| Splits | {splits} |
| Scores | the model's prediction on each test date; top {run.top_k} bought at the next open |

## What it learnt

{learned_markdown(wf)}{sheet}
"""
