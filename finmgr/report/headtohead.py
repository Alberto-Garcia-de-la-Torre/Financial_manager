"""Day 43: one page with two models side by side — Markdown for `docs/`, HTML for the browser.

    from finmgr.report.headtohead import headtohead_markdown, headtohead_html

Top to bottom, both forms carry the same content:

1. **Verdict**: each model against day 38's bar, and the challenger against
   the incumbent (paired Sharpe interval, IC difference, folds won).
2. **Fairness**: every check of :func:`finmgr.models.headtohead.fairness_checks`
   with what it compared. The page only exists if they all passed.
3. **Side by side**: one column per model — Sharpe and interval, CAGR,
   volatility, drawdown, turnover, distance to the bar, out-of-sample IC.
4. **Against the baselines**: one table holding both models and every
   baseline, all on the same engine and date axis.
5. **Head to head**: the paired statistics and how similar the two books are.
6. **Year by year**: calendar returns of both models and the bar.
7. **What each learnt**: ridge's weights, the trees' gain shares.

The HTML adds the equity curves, drawdowns and rolling Sharpe of both models
and the bar on shared axes.
"""

from __future__ import annotations

import html
import math
from datetime import datetime

import pandas as pd

from finmgr.backtest.baselines import RANDOM
from finmgr.backtest.metrics import drawdown_series
from finmgr.models.headtohead import HeadToHead, Pair
from finmgr.report import model as model_report
from finmgr.report import tearsheet as ts

#: Fixed categorical order (dataviz reference palette, light): incumbent, challenger, bar.
#: Validated together: CVD ΔE ≥ 9.2; the aqua is below 3:1 on the surface, so every
#: chart also carries a legend and end labels and every number is in a table.
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")


def _nan(v: float | None) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(v: float | None, fmt: str = ".2f") -> str:
    return "-" if _nan(v) else format(v, fmt)


def _pct(v: float | None, digits: int = 1) -> str:
    return "-" if _nan(v) else f"{v:.{digits}%}"


def title(h: HeadToHead) -> str:
    return " vs ".join(h.models)


# ---------------------------------------------------------------------------
# Tables, as (header, rows) so Markdown and HTML render the same content
# ---------------------------------------------------------------------------


def side_by_side(h: HeadToHead) -> tuple[list[str], list[list[str]]]:
    """One row per measure, one column per model."""
    head = ["", *h.models]
    rows: list[list[str]] = []

    def row(label: str, values: list[str]) -> None:
        rows.append([label, *values])

    runs = [h.runs[m] for m in h.models]
    comps = [h.comparisons[m] for m in h.models]
    wfs = [h.walkforwards[m] for m in h.models]
    ics = [wf.ic_summary() for wf in wfs]
    bar = comps[0].bar
    row("Sharpe", [_num(r.metrics.sharpe) for r in runs])
    row(
        "95% interval",
        [f"[{r.interval.low:.2f}, {r.interval.high:.2f}]" for r in runs],
    )
    row("CAGR", [_pct(r.metrics.cagr, 2) for r in runs])
    row("Volatility", [_pct(r.metrics.volatility, 2) for r in runs])
    row("Max drawdown", [_pct(r.metrics.max_drawdown, 2) for r in runs])
    row("Turnover (x/yr)", [_num(r.metrics.turnover) for r in runs])
    row("Costs paid (% of start)", [_cost_share(r) for r in runs])
    row(
        f"Sharpe minus the bar ({bar.sharpe:.2f}), paired",
        [
            f"{c.vs_reference.point:+.2f} [{c.vs_reference.low:+.2f}, {c.vs_reference.high:+.2f}]"
            for c in comps
        ],
    )
    row("Beats the bar", ["yes" if c.beats_bar else "no" for c in comps])
    row("Mean rank IC", [_num(s["mean"], "+.3f") for s in ics])
    row("IC / std", [_num(s["ir"], "+.2f") for s in ics])
    row(
        f"IC t, overlap-adjusted (÷√{wfs[0].horizon})",
        [_num(s["t"] / math.sqrt(wfs[0].horizon), ".1f") for s in ics],
    )
    row("IC positive on", [_pct(s["positive"], 0) for s in ics])
    return head, rows


def _cost_share(run) -> str:
    """Total costs paid over the run as a share of the starting capital."""
    start = float(run.result.equity["equity_eur"].iloc[0])
    return _pct(run.result.total_cost / start, 1) if start else "-"


def candidates(h: HeadToHead) -> tuple[list[str], list[list[str]]]:
    """Both models and every baseline in one table, Sharpe descending, the bar marked."""
    comp = h.comparisons[h.incumbent]
    head = [
        "Candidate",
        "Sharpe",
        "95% interval",
        "CAGR",
        "Volatility",
        "Max drawdown",
        "Turnover (x/yr)",
    ]
    entries = []
    for name, run in [*((h.runs[m].label, h.runs[m]) for m in h.models), *h.baselines.items()]:
        m, iv = run.metrics, run.interval
        tag = (
            " ← model" if run.kind == "model" else (" ← the bar" if name == comp.bar.source else "")
        )
        entries.append(
            (
                m.sharpe,
                [
                    name + tag,
                    _num(m.sharpe),
                    f"[{iv.low:.2f}, {iv.high:.2f}]",
                    _pct(m.cagr, 2),
                    _pct(m.volatility, 2),
                    _pct(m.max_drawdown, 2),
                    _num(m.turnover),
                ],
            )
        )
    if comp.random:
        label = next(k for k in comp.bar.candidates if k.startswith(RANDOM))
        p95 = comp.bar.candidates[label]
        tag = " ← the bar" if label == comp.bar.source else ""
        entries.append((p95, [label + tag, _num(p95), f"{len(comp.random)} runs", *["-"] * 4]))
    entries.sort(key=lambda e: -e[0])
    return head, [e[1] for e in entries]


def pair_table(h: HeadToHead, p: Pair) -> tuple[list[str], list[list[str]]]:
    horizon = h.walkforwards[p.incumbent].horizon
    ic = p.ic_summary(horizon)
    f = p.folds.dropna()
    s = p.sharpe
    head = ["", f"{p.challenger} minus {p.incumbent}"]
    rows = [
        ["Sharpe difference, paired", f"{s.point:+.2f} [{s.low:+.2f}, {s.high:+.2f}]"],
        [f"P({p.challenger} no better)", f"{s.p_le_zero:.1%}"],
        ["Mean IC difference", f"{ic['mean']:+.4f}"],
        [
            f"IC difference t, overlap-adjusted (÷√{horizon})",
            f"{ic['t']:+.1f} (naive {ic['t_naive']:+.1f})",
        ],
        [f"Dates where {p.challenger}'s IC is higher", _pct(ic["positive"], 0)],
        [
            f"Folds where {p.challenger}'s mean IC is higher",
            f"{p.fold_wins} of {len(f)}",
        ],
        ["Score correlation (per-date Spearman, mean)", _num(p.score_corr.mean(), ".2f")],
        ["Top-K picked in common (mean per date)", _pct(p.top_overlap.mean(), 0)],
    ]
    return head, rows


def yearly(h: HeadToHead) -> tuple[list[str], list[list[str]]]:
    table = h.yearly()
    bar = h.comparisons[h.incumbent].reference
    head = ["Year", *h.models, f"{bar} (the bar)"]
    first = h.runs[h.incumbent].metrics.start
    rows = []
    for year, r in table.iterrows():
        label = f"{year}*" if year == pd.Timestamp(str(first)).year else str(year)
        best = r[h.models].idxmax() if r[h.models].notna().any() else None
        cells = [(f"**{_pct(r[m])}**" if m == best else _pct(r[m])) for m in h.models]
        rows.append([label, *cells, _pct(r[bar])])
    last = h.runs[h.incumbent].metrics.end
    if pd.Timestamp(str(last)).month < 12:
        rows[-1][0] += "*"
    return head, rows


def _md_table(head: list[str], rows: list[list[str]], align: str = "r") -> str:
    sep = ["---"] + [("---:" if align == "r" else "---")] * (len(head) - 1)
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _html_table(head: list[str], rows: list[list[str]]) -> str:
    def cell(text: str) -> str:
        if text.startswith("**") and text.endswith("**"):
            return f"<b>{html.escape(text[2:-2])}</b>"
        return html.escape(text)

    thead = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    body = "".join(
        f"<tr><th>{cell(r[0])}</th>" + "".join(f"<td>{cell(c)}</td>" for c in r[1:]) + "</tr>"
        for r in rows
    )
    return f'<table class="metrics"><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------


def verdict_lines(h: HeadToHead) -> list[str]:
    lines = [f"**{m}**: {h.comparisons[m].verdict}." for m in h.models]
    for p in h.pairs:
        s = p.sharpe
        if s.low > 0:
            word = f"{p.challenger} is better than {p.incumbent}"
        elif s.high < 0:
            word = f"{p.challenger} is worse than {p.incumbent}"
        else:
            word = f"{p.challenger} and {p.incumbent} cannot be told apart"
        lines.append(
            f"**{p.challenger} against {p.incumbent}**: {word}. Sharpe difference "
            f"{s.point:+.2f}, paired 95% interval [{s.low:+.2f}, {s.high:+.2f}]; "
            f"{p.challenger} has the higher mean IC in {p.fold_wins} of "
            f"{len(p.folds.dropna())} folds."
        )
    return lines


def setup_rows(h: HeadToHead) -> list[list[str]]:
    wf = h.walkforwards[h.incumbent]
    run = h.runs[h.incumbent]
    rows = []
    for m in h.models:
        params = ", ".join(f"{k}={v!r}" for k, v in h.walkforwards[m].params.items())
        rows.append([f"Model `{m}`", params or "defaults"])
    rows += [
        [
            "Features (both)",
            ", ".join(f"`{f.removesuffix('_rank')}`" for f in wf.features)
            + " — per-date ranks, missing → 0.5",
        ],
        [
            "Label (both)",
            f"`{wf.label}`: next {wf.horizon} sessions' return minus the universe mean",
        ],
        [
            "Backtest (both)",
            f"from {run.start}, top {run.top_k}, held {run.hold} sessions, next-open fills, "
            "configured costs",
        ],
    ]
    return rows


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def headtohead_markdown(
    h: HeadToHead,
    *,
    command: str,
    page: str | None = None,
    n_variants: int | None = None,
) -> str:
    """The page `finmgr compare-models --doc` writes under `docs/models/`."""
    run = h.runs[h.incumbent]
    m = run.metrics
    comp = h.comparisons[h.incumbent]
    checks = [[c.name, "✅ same" if c.passed else "❌ differs", c.detail] for c in h.checks]
    random_note = (
        f"The random row is the 95th percentile of the {len(comp.random)} seeds in "
        "`docs/baselines_random.csv` (same window)."
        if comp.random
        else "No random-book percentile: `docs/baselines_random.csv` covers another window."
    )
    learnt = "\n".join(
        f"### {mname}\n\n{model_report.learned_markdown(h.walkforwards[mname])}"
        for mname in h.models
    )
    pairs = "\n".join(_md_table(*pair_table(h, p)) for p in h.pairs)
    ledger = (
        f"\n\nEach model is logged as a variant in `docs/trials.csv` ({n_variants} variants "
        "in all); `finmgr train --model <name> --doc docs/models/<name>.md` writes a "
        "model's own page, with its Sharpe at the familywise level for that count."
        if n_variants is not None
        else ""
    )
    html_note = (
        f"\n\nThe HTML version, with both equity curves, drawdowns and rolling Sharpes on "
        f"shared axes, is written to `{page}` (under the git-ignored `data/`)."
        if page
        else ""
    )
    return f"""# Head to head: {title(h)}

Generated by `{command}` from the local store, data as of {run.as_of}. The code is
`finmgr/models/headtohead.py` (the runs and the checks) and
`finmgr/report/headtohead.py` (this page). Rerun the command after any change to
either model, the features, the engine, costs or data, and commit the regenerated page.

## Verdict

{chr(10).join(f"* {line}" for line in verdict_lines(h))}

## Is the comparison fair?

All the models were trained and backtested in one process, on one `Dataset`
(one feature matrix, one label) and one set of backtest inputs and costs. The
checks below compare the finished runs field by field: the command refuses to
write this page if any of them differ.

{_md_table(["Check", "Result", "What was compared"], checks, align="l")}
## Side by side

{_md_table(*side_by_side(h))}
Intervals are 95% circular block bootstraps of daily excess returns over €STR/EONIA.
"Minus the bar" pairs each model with *{comp.reference}*, resampling both on the
same blocks of dates. IC is the per-date Spearman correlation of score with the
realised label; the 21-session labels overlap, hence the deflated t.

## Against the baselines

Same engine, same date axis ({m.start} → {m.end}), top {run.top_k}, held
{run.hold} sessions, configured costs, next-open fills. {random_note}

{_md_table(*candidates(h))}
## Head to head

Paired: both books' daily excess returns are resampled on the same blocks of
dates, so the market moves they share cancel, and the interval is far narrower than
the gap between two separate intervals would suggest.

{pairs}
## Year by year

Calendar-year total return after costs; the better model each year in bold.
\\* marks a partial year.

{_md_table(*yearly(h))}
## Setup

{_md_table(["", ""], setup_rows(h), align="l")}
## What each model learnt

{learnt}{ledger}{html_note}
"""


def headtohead_html(h: HeadToHead, *, command: str, generated: datetime | None = None) -> str:
    """A self-contained HTML page, same content as the Markdown plus the charts."""
    names = [*h.models]
    bar = h.comparisons[h.incumbent].reference
    curves = [(m, h.runs[m].result.curve, COLOURS[i]) for i, m in enumerate(names)]
    curves.append((f"{bar} (the bar)", h.baselines[bar].result.curve, COLOURS[2]))
    run = h.runs[h.incumbent]
    window = run.periods
    equity_svg = ts._line_chart(
        curves, name="equity", ylabel="Equity (EUR, log scale)", fmt=ts._eur, log=True, height=3.8
    )
    dd_svg = ts._line_chart(
        [(n, drawdown_series(c), col) for n, c, col in curves],
        name="drawdown",
        ylabel="From running peak",
        fmt=ts._pct0,
        zero=True,
        height=2.6,
    )
    excess = [(m, h.runs[m].excess, COLOURS[i]) for i, m in enumerate(names)]
    excess.append((f"{bar} (the bar)", h.baselines[bar].excess, COLOURS[2]))
    sharpe_svg = ts._line_chart(
        [(n, ts.rolling_sharpe(e, window=window, periods=window), c) for n, e, c in excess],
        name="sharpe",
        ylabel="Sharpe, trailing 12 months",
        fmt=ts._num1,
        zero=True,
        height=2.6,
    )
    checks = _html_table(
        ["Check", "Result", "What was compared"],
        [[c.name, "same" if c.passed else "DIFFERS", c.detail] for c in h.checks],
    )
    verdict = "".join(f"<li>{_strong(line)}</li>" for line in verdict_lines(h))
    pairs = "".join(_html_table(*pair_table(h, p)) for p in h.pairs)
    learnt = "".join(
        f"<h3>{html.escape(m)}</h3>{model_report.learned_html(h.walkforwards[m])}" for m in names
    )
    m = run.metrics
    stamp = (generated or datetime.now()).strftime("%Y-%m-%d %H:%M")
    css = ts._CSS + "h3 { font-size: 14px; margin: 14px 0 4px; } ul { margin: 6px 0; }"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title(h))} — finmgr head to head</title>
<style>{css}</style></head>
<body><main>
<header>
<h1>Head to head: {html.escape(title(h))}</h1>
<p>{m.start} to {m.end} · top {run.top_k}, held {run.hold} sessions · data as of
{run.as_of} · identical features, splits, window and costs</p>
</header>
<section><h2>Verdict</h2><ul>{verdict}</ul></section>
<section><h2>Is the comparison fair?</h2>
<p class="note">Trained and backtested in one process on one dataset and one cost
model, then compared field by field. The page is not written if any check fails.</p>
{checks}</section>
<section><h2>Side by side</h2>{_html_table(*side_by_side(h))}</section>
<section><h2>Equity curve</h2>
<p class="note">Both models and the bar, same capital on the same first date, after
every cost. Log scale.</p>{equity_svg}</section>
<section><h2>Drawdown</h2>{dd_svg}</section>
<section><h2>Rolling 12-month Sharpe</h2>{sharpe_svg}</section>
<section><h2>Against the baselines</h2>{_html_table(*candidates(h))}</section>
<section><h2>Head to head</h2>
<p class="note">Paired block bootstrap: both books resampled on the same blocks of
dates, so what they share cancels.</p>{pairs}</section>
<section><h2>Year by year</h2>
<p class="note">Calendar-year return after costs; the better model in bold;
* a partial year.</p>{_html_table(*yearly(h))}</section>
<section><h2>What each model learnt</h2>{learnt}</section>
<footer>Generated by <code>{html.escape(command)}</code> on {stamp}. Past
performance in a backtest is an upper bound on what to expect, not an estimate of it.
</footer>
</main></body></html>
"""


def _strong(line: str) -> str:
    """`**x**: rest` -> `<b>x</b>: rest`, everything escaped."""
    if line.startswith("**") and "**" in line[2:]:
        end = line.index("**", 2)
        return f"<b>{html.escape(line[2:end])}</b>{html.escape(line[end + 2 :])}"
    return html.escape(line)
