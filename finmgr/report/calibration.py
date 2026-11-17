"""Day 44: the calibration curve of a classifier, as a chart, tables and a Markdown page.

* :func:`reliability_figure` — predicted vs observed P(top quintile) for the
  raw classifier and every calibrator, with the diagonal and the tolerance
  band of the acceptance check. :func:`reliability_svg` inlines it in the HTML
  tearsheet; :func:`save_reliability_png` writes it next to the doc page.
* :func:`calibration_markdown` / :func:`calibration_html` — the out-of-sample
  scores of each method, the curve of the chosen one bin by bin, and the
  per-year check.
* :func:`classifier_page` — the page `finmgr classify --doc` writes under
  `docs/models/`.
"""

from __future__ import annotations

import html
import io
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure

from finmgr.models import classify as cl
from finmgr.models.compare import Comparison, comparison_markdown
from finmgr.report import model as model_report
from finmgr.report import tearsheet as ts

#: Line colours: raw is muted grey, the calibrators categorical slots 1-2.
COLOURS = {"none": ts.MUTED, "platt": ts.STRATEGY_COLOUR, "isotonic": ts.BENCHMARK_COLOUR}
LABELS = {"none": "raw (uncalibrated)", "platt": "Platt", "isotonic": "isotonic"}


def method_scores(cw: cl.CalibratedWalkForward) -> pd.DataFrame:
    """One row per calibration method: the out-of-sample scores of :func:`classify.scores`."""
    y = cw.oos["target"]
    rows = {m: cl.scores(cw.oos[f"prob_{m}"], y, base_rate=cw.quantile) for m in cl.METHODS}
    return pd.DataFrame(rows).T


def reliability_figure(cw: cl.CalibratedWalkForward, *, gap_tol: float = cl.GAP_TOL) -> Figure:
    """The reliability diagram of every method on the out-of-sample rows."""
    curves = {m: cl.reliability(cw.oos[f"prob_{m}"], cw.oos["target"]) for m in cl.METHODS}
    top = max(float(c["predicted"].max()) for c in curves.values())
    top = max(top, float(max(c["observed"].max() for c in curves.values()))) * 1.08
    low = min(min(float(c["predicted"].min()), float(c["observed"].min())) for c in curves.values())
    low = max(0.0, low - 0.03)
    with rc_context(ts._RC):
        fig = Figure(figsize=(6.2, 5.2))
        ax = fig.add_subplot()
        grid = np.array([low, top])
        ax.fill_between(grid, grid - gap_tol, grid + gap_tol, color=ts.GRID, alpha=0.6, lw=0)
        ax.plot(grid, grid, color=ts.AXIS, lw=1, ls="--", label="perfect calibration")
        for m in ("none", "isotonic", "platt"):
            c = curves[m]
            chosen = m == cw.method
            ax.errorbar(
                c["predicted"],
                c["observed"],
                yerr=c["half_width"],
                color=COLOURS[m],
                lw=2.2 if chosen else 1.2,
                marker="o",
                ms=5 if chosen else 3.5,
                capsize=0,
                label=LABELS[m] + (" (chosen)" if chosen else ""),
                zorder=3 if chosen else 2,
            )
        ax.axhline(cw.quantile, color=ts.AXIS, lw=0.8)
        ax.set_xlim(low, top)
        ax.set_ylim(low, top)
        ax.set_aspect("equal")
        ax.set_xlabel("predicted P(top quintile), bin mean")
        ax.set_ylabel("observed share in the top quintile")
        ax.set_title(
            f"{cw.wf.model}: out-of-sample calibration, {len(cw.oos):,} rows, "
            f"{cl.BINS} equal-count bins\nshaded band: ±{gap_tol:.0%} acceptance limit per bin",
            fontsize=9,
            color=ts.INK_2,
            loc="left",
        )
        ax.legend(loc="upper left", fontsize=8.5, labelcolor=ts.INK_2)
    return fig


def reliability_svg(cw: cl.CalibratedWalkForward) -> str:
    return ts._namespace(ts._svg(reliability_figure(cw)), "reliability")


def save_reliability_png(cw: cl.CalibratedWalkForward, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    reliability_figure(cw).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    path.write_bytes(buffer.getvalue())
    return path


def _scores_rows(table: pd.DataFrame, chosen: str) -> list[list[str]]:
    rows = []
    for m, r in table.iterrows():
        rows.append(
            [
                LABELS[str(m)] + (" **(chosen)**" if m == chosen else ""),
                f"{r['ece']:.4f}",
                f"{r['max_gap']:.4f}",
                f"{r['brier']:.4f}",
                f"{r['brier_skill']:+.2%}",
                f"{r['log_loss']:.4f}",
                f"{r['auc']:.3f}",
                f"[{r['p05']:.3f}, {r['p95']:.3f}]",
            ]
        )
    return rows


SCORE_HEADER = [
    "Method",
    "ECE",
    "Worst bin",
    "Brier",
    "Brier skill vs flat 20%",
    "Log loss",
    "AUC",
    "5th to 95th pct of P",
]


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def calibration_markdown(
    cw: cl.CalibratedWalkForward,
    check: cl.CalibrationCheck,
    *,
    chart: str | None = None,
    slices: Mapping[float, pd.DataFrame] | None = None,
) -> str:
    """Scores per method, the chosen curve bin by bin, per-year, and (optionally) slice lengths."""
    table = method_scores(cw)
    flat = table.iloc[0]
    parts = [
        f"**{check.verdict}.**",
        "",
        f"Out-of-sample rows: {len(cw.oos):,} (every test date with a known outcome, "
        f"{cw.oos['date'].nunique():,} dates). Observed top-quintile share "
        f"{flat['base_rate']:.2%}; a flat {cw.quantile:.0%} forecast scores Brier "
        f"{flat['brier_flat']:.4f} and log loss {flat['log_loss_flat']:.4f}.",
        "",
    ]
    if chart:
        parts += [f"![Reliability diagram]({chart})", ""]
    parts += [
        _md_table(SCORE_HEADER, _scores_rows(table, cw.method), ["---"] + ["---:"] * 7),
        "",
        "ECE is the row-weighted mean |observed - predicted| over "
        f"{cl.BINS} equal-count bins; *worst bin* the largest such gap. The acceptance "
        f"limits, fixed before the run, are ECE ≤ {check.ece_tol} and worst bin ≤ "
        f"{check.gap_tol}. A flat 20% forecast would pass them too, which is why Brier "
        "skill, AUC and the spread of the predictions sit beside them: the claim is "
        "*calibrated and informative*, and the honest size of *informative* is the "
        f"Brier skill: {table.loc[cw.method, 'brier_skill']:+.2%} over the flat forecast.",
        "",
        f"### The {LABELS[cw.method]} curve, bin by bin",
        "",
    ]
    curve = check.curve
    rows = [
        [
            str(i + 1),
            f"[{r['low']:.3f}, {r['high']:.3f}]",
            f"{int(r['n']):,}",
            f"{r['predicted']:.3f}",
            f"{r['observed']:.3f}",
            f"{r['gap']:+.3f}",
            f"±{r['half_width']:.3f}",
        ]
        for i, (_, r) in enumerate(curve.iterrows())
    ]
    parts += [
        _md_table(
            ["Bin", "P range", "Rows", "Mean P", "Observed", "Gap", "Naive 95%"],
            rows,
            ["---:", "---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        "The naive interval treats rows as independent; neighbouring dates share 20 of "
        "their 21 label sessions, so the real uncertainty is several times wider.",
        "",
        "### Year by year",
        "",
    ]
    years = cl.by_year(cw.oos)
    rows = [
        [
            str(y),
            f"{int(r['n']):,}",
            f"{r['mean_prob']:.3f}",
            f"{r['observed']:.3f}",
            f"{r['ece']:.4f}",
            f"{r['max_gap']:.4f}",
        ]
        for y, r in years.iterrows()
    ]
    within = int((years["ece"] <= check.ece_tol).sum())
    parts += [
        _md_table(
            ["Year", "Rows", "Mean P", "Observed", "ECE (5 bins)", "Worst bin"],
            rows,
            ["---", "---:", "---:", "---:", "---:", "---:"],
        ),
        "",
        f"{within} of {len(years)} years are within the ECE limit on their own. The "
        "pooled curve is close to the diagonal; single years drift around it, because "
        "the slice a calibrator learns from is one regime and the test window the next.",
        "",
    ]
    if slices:
        parts += [
            "### Slice length",
            "",
            "The same classifier and folds, with the calibration slice set to each "
            "length below (a whole number is dates, a fraction is a share of the "
            "fold's training dates). Shown so the choice of slice is visible, not hidden:",
            "",
        ]
        rows = []
        for size, t in slices.items():
            for m, r in t.iterrows():
                rows.append(
                    [
                        _slice(size),
                        LABELS[str(m)],
                        f"{r['ece']:.4f}",
                        f"{r['max_gap']:.4f}",
                        f"{r['brier_skill']:+.2%}",
                        f"{r['auc']:.3f}",
                        "yes"
                        if r["ece"] <= check.ece_tol and r["max_gap"] <= check.gap_tol
                        else "no",
                    ]
                )
        parts += [
            _md_table(
                ["Slice", "Method", "ECE", "Worst bin", "Brier skill", "AUC", "Passes"],
                rows,
                ["---", "---", "---:", "---:", "---:", "---:", "---"],
            ),
            "",
        ]
    return "\n".join(parts)


def _slice(size: float) -> str:
    return f"{size:.0%} of training" if 0 < size < 1 else f"{int(size)} dates"


def calibration_html(cw: cl.CalibratedWalkForward, check: cl.CalibrationCheck) -> str:
    """The "Calibration" section of the HTML tearsheet."""
    table = method_scores(cw)
    rows = "".join(
        "<tr>"
        + f"<th>{html.escape(r[0].replace('**', ''))}</th>"
        + "".join(f"<td>{html.escape(v)}</td>" for v in r[1:])
        + "</tr>"
        for r in _scores_rows(table, cw.method)
    )
    head = "".join(f"<th>{html.escape(h)}</th>" for h in SCORE_HEADER)
    return (
        f'<p class="note"><strong>{html.escape(check.verdict)}.</strong> '
        f"{len(cw.oos):,} out-of-sample rows; the dashed line is perfect calibration, the "
        f"shaded band the ±{check.gap_tol:.0%} per-bin limit, whiskers naive 95% intervals."
        f"</p>{reliability_svg(cw)}"
        f'<table class="metrics"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
    )


def classifier_page(
    cw: cl.CalibratedWalkForward,
    check: cl.CalibrationCheck,
    comp: Comparison,
    *,
    as_of: date,
    command: str,
    chart: str | None = None,
    slices: Mapping[float, pd.DataFrame] | None = None,
    tearsheet: str | None = None,
    n_variants: int | None = None,
    familywise: tuple[float, float, float] | None = None,
) -> str:
    """The Markdown page for one calibrated classifier: calibration first, then the book."""
    wf, run = cw.wf, comp.model
    m = run.metrics
    params = ", ".join(f"{k}={v!r}" for k, v in wf.params.items()) or "defaults"
    window = "expanding" if wf.train_size is None else f"rolling {wf.train_size} dates"
    folds = cw.folds
    first = folds.iloc[0]
    last = folds.iloc[-1]
    feats = ", ".join(f"`{model_report._feature(f)}`" for f in wf.features)
    setup = _md_table(
        ["", ""],
        [
            ["Classifier", f"`{wf.model}` ({params})"],
            [
                "Target",
                f"`{cw.target}`: 1 if `{wf.label}` ranks in the date's top "
                f"{cw.quantile:.0%} of known labels",
            ],
            ["Features", f"{feats} — per-date ranks, missing → 0.5"],
            [
                "Folds",
                f"day 42's: walk-forward, {window}, test {wf.test_size} dates, min train "
                f"{wf.min_train_size}, embargo {wf.horizon} — the same test windows as "
                "ridge and LightGBM",
            ],
            [
                "Calibration slice",
                f"the last {_slice(cw.cal_size)} dates of each fold's training window, "
                "held out of the classifier's fit; training dates whose label reaches "
                "into the slice are purged",
            ],
            [
                "Calibrator",
                f"{LABELS[cw.method]}, fitted on the slice only; isotonic and Platt are "
                "fitted on the same slice for the comparison above",
            ],
            [
                "Fold 0",
                f"fit {first['fit_start']} → {first['fit_end']} ({first['fit_dates']} "
                f"dates), calibrate {first['cal_start']} → {first['cal_end']}, test "
                f"{first['test_start']} → {first['test_end']}",
            ],
            [
                "Folds",
                f"{len(folds)}, last test window {last['test_start']} → {last['test_end']}",
            ],
        ],
        ["---", "---"],
    )
    fw = ""
    if n_variants is not None and familywise is not None:
        level, low, high = familywise
        fw = (
            f" With {n_variants} variants now in `docs/trials.csv`, the familywise "
            f"({level:.2%}) interval is [{low:.2f}, {high:.2f}]."
        )
    sheet = (
        f"\n\nThe HTML tearsheet (`{command} --report`) is written to `{tearsheet}` "
        "(under the git-ignored `data/`)."
        if tearsheet
        else ""
    )
    return f"""# Classifier: {wf.model}, P(top quintile), {LABELS[cw.method]}-calibrated

Generated by `{command}` from the local store, data as of {as_of}. The code is
`finmgr/models/classify.py` (target, folds, calibration, the check) and
`finmgr/report/calibration.py` (this page); rerun the command after any change
and commit the regenerated page and chart.

## The question

Instead of predicting the 21-session excess return, predict the probability
that a name finishes its date in the **top {cw.quantile:.0%}** of the universe
by that return — the names a top-K book wants. Then check that the probability
means what it says: of all the rows given ~30%, about 30% should land in the
top quintile, out of sample.

## Calibration (the acceptance criterion)

{calibration_markdown(cw, check, chart=chart, slices=slices)}
## How each probability was made

{setup}

## The book it trades, against the baselines

The book ranks each test date on the classifier's raw probability: calibration
is monotone, so it cannot change who is in the top {run.top_k}, and ranking on
the raw score keeps the ordering isotonic's steps would tie. Same engine,
window ({m.start} → {m.end}), top {run.top_k}, held {run.hold} sessions,
configured costs and next-open fills as every baseline.

**{comp.verdict}.** Sharpe {m.sharpe:.2f}, 95% interval
[{run.interval.low:.2f}, {run.interval.high:.2f}].{fw}

{comparison_markdown(comp)}
## What it learnt

Logistic weights are on the log-odds of a top-quintile finish, per unit of a
feature's rank (bottom → top of the day = 1).

{model_report.learned_markdown(wf)}{sheet}
"""
