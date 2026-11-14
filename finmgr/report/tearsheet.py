"""Day 40: one backtest as one self-contained HTML page.

    from finmgr.report.tearsheet import render_tearsheet
    html = render_tearsheet(run, logged)      # a StrategyRun and its ledger entry
    Path("tearsheet.html").write_text(html, encoding="utf-8")

The page, top to bottom:

1. **Headline tiles** — CAGR, Sharpe *as an interval* (day 39: never a single
   number), max drawdown and the paired Sharpe difference against the
   benchmark, each with the benchmark's figure beside it.
2. **Equity curve** of the strategy against the benchmark, both starting at
   the same capital on the same date, log scale (equal vertical distances are
   equal percentage moves, so 2012 and 2024 are comparable).
3. **Drawdown** of both from their running peaks.
4. **Rolling 12-month Sharpe** of both: the mean over the std of daily excess
   returns over the last year of sessions on the axis, annualised the same way
   as the headline Sharpe. Starts one year in.
5. **Monthly returns heatmap**: one row per year, one cell per month, a
   diverging blue (up) / red (down) scale symmetric around zero, plus the
   year's return beside the benchmark's. Partial first and last years are
   marked.
6. **Metrics table**: every day 37 metric for both, plus the costs paid.
7. **How to read this**: window, costs, fills, the variant count and the
   Sharpe at the familywise level for that count.

Everything is inline — the charts are SVG drawn by matplotlib, the styles a
`<style>` block — so the file can be mailed or opened offline. No JavaScript;
heatmap cells carry their exact value as a native tooltip and as text.
"""

from __future__ import annotations

import html
import io
import math
import re
from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from finmgr.backtest.metrics import calendar_returns, drawdown_series

# Palette: the dataviz reference instance (light mode). Strategy is categorical
# slot 1 (blue), benchmark slot 2 (orange); the heatmap is the blue <-> red
# diverging pair around a neutral grey, so "up" and "down" never share a hue.
STRATEGY_COLOUR = "#2a78d6"
BENCHMARK_COLOUR = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
NEUTRAL = (0xF0, 0xEF, 0xEC)
UP_POLE = (0x10, 0x42, 0x81)  # blue 650
DOWN_POLE = (0xB0, 0x24, 0x24)  # a red deep enough to carry white text
#: A monthly return of this size or more gets the full pole colour.
HEAT_CAP = 0.08

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


def _dated(curve: pd.Series) -> pd.Series:
    out = curve.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(list(curve.index)))
    return out


def monthly_returns(curve: pd.Series) -> pd.DataFrame:
    """Total return per calendar month, as a year x month (1..12) table; NaN where no data.

    Each month runs from the previous month's last value (or the curve's first
    value, for the first month) to its own last value.
    """
    dated = _dated(curve)
    last = dated.groupby(dated.index.to_period("M")).last()
    base = last.shift(1)
    base.iloc[0] = dated.iloc[0]
    rets = last / base - 1
    if (dated.index.to_period("M") == dated.index[0].to_period("M")).sum() == 1:
        rets = rets.iloc[1:]  # the first month holds only the starting point
    table = pd.DataFrame(
        {"year": rets.index.year, "month": rets.index.month, "ret": rets.to_numpy()}
    ).pivot(index="year", columns="month", values="ret")
    return table.reindex(columns=range(1, 13))


def rolling_sharpe(excess: pd.Series, *, window: int, periods: int) -> pd.Series:
    """Annualised Sharpe of daily excess returns over a trailing `window`; NaN until full."""
    mean = excess.rolling(window, min_periods=window).mean()
    std = excess.rolling(window, min_periods=window).std(ddof=1)
    return (mean / std * math.sqrt(periods)).where(std > 0)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

_RC = {
    "svg.fonttype": "none",  # keep text as text: sharp, selectable, page font
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}


def _svg(fig: Figure) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=True)
    text = buffer.getvalue()
    text = text[text.index("<svg") :]  # drop the XML prolog and doctype
    text = re.sub(r'(<svg[^>]*?)\swidth="[^"]+"\sheight="[^"]+"', r"\1", text, count=1)
    # Matplotlib names every element with a per-figure id; several figures on
    # one page would collide on clip paths, so namespace them.
    return text


def _namespace(svg: str, prefix: str) -> str:
    ids = set(re.findall(r'id="([^"]+)"', svg))
    for old in sorted(ids, key=len, reverse=True):
        svg = svg.replace(f'id="{old}"', f'id="{prefix}-{old}"')
        svg = svg.replace(f"#{old})", f"#{prefix}-{old})")
        svg = svg.replace(f'"#{old}"', f'"#{prefix}-{old}"')
    return svg


def _end_label(ax, series: pd.Series, text: str, colour: str, fmt) -> None:
    clean = series.dropna()
    if clean.empty:
        return
    ax.annotate(
        f"{text} {fmt(clean.iloc[-1])}",
        xy=(clean.index[-1], clean.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=INK_2,
        bbox={"boxstyle": "round,pad=0.25", "fc": SURFACE, "ec": colour, "lw": 1},
    )


def _line_chart(
    series: Sequence[tuple[str, pd.Series, str]],
    *,
    name: str,
    ylabel: str,
    fmt,
    log: bool = False,
    zero: bool = False,
    fill: bool = False,
    height: float = 3.2,
) -> str:
    with rc_context(_RC):
        fig = Figure(figsize=(10, height))
        ax = fig.add_subplot()
        for label, s, colour in series:
            dated = _dated(s)
            ax.plot(dated.index, dated.to_numpy(), color=colour, lw=1.6, label=label)
            if fill:
                ax.fill_between(dated.index, dated.to_numpy(), 0, color=colour, alpha=0.08, lw=0)
            _end_label(ax, dated, "", colour, fmt)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_formatter(NullFormatter())
        if zero:
            ax.axhline(0, color=AXIS, lw=1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
        ax.set_ylabel(ylabel)
        ax.margins(x=0.01)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncols=len(series),
            fontsize=8.5,
            labelcolor=INK_2,
            borderaxespad=0.2,
        )
        return _namespace(_svg(fig), name)


def _eur(v: float) -> str:
    if abs(v) >= 1e6:
        return f"€{v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"€{v / 1e3:.0f}k"
    return f"€{v:.0f}"


def _pct0(v: float) -> str:
    return f"{100 * v:.0f}%"


def _num1(v: float) -> str:
    return f"{v:.1f}"


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------


def _pct(v: float | None, digits: int = 1, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    shown = round(100 * v, digits) + 0.0  # + 0.0 turns -0.0 into 0.0: no "-0.0%"
    return f"{shown:{'+' if sign else ''}.{digits}f}%"


def _num(v: float | None, digits: int = 2, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:{'+' if sign else ''}.{digits}f}"


def _int(v: float | None) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.0f}"


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b, strict=True))  # type: ignore[return-value]


def heat_colour(value: float) -> tuple[str, str]:
    """Background and text colour of a heatmap cell: diverging, symmetric, capped at HEAT_CAP."""
    if not np.isfinite(value):
        return "transparent", MUTED
    t = min(abs(value) / HEAT_CAP, 1.0) ** 0.8
    rgb = _mix(NEUTRAL, UP_POLE if value >= 0 else DOWN_POLE, t)
    luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    return "#{:02x}{:02x}{:02x}".format(*rgb), "#ffffff" if luminance < 0.55 else INK


def _heatmap(curve: pd.Series, bench: pd.Series) -> str:
    table = monthly_returns(curve)
    years = calendar_returns(curve)
    bench_years = calendar_returns(bench)
    first, last = _dated(curve).index[[0, -1]]
    head = "".join(f"<th>{m}</th>" for m in MONTHS)
    rows = []
    for year, row in table.iterrows():
        cells = []
        for month, value in zip(MONTHS, row.to_numpy(), strict=True):
            bg, fg = heat_colour(value)
            tip = f"{month} {year}: {_pct(value, 2, sign=True)}" if np.isfinite(value) else ""
            cells.append(
                f'<td class="heat" style="background:{bg};color:{fg}" title="{tip}">'
                f"{_pct(value, 1) if np.isfinite(value) else ''}</td>"
            )
        partial = (year == first.year and (first.month, first.day) > (1, 3)) or (
            year == last.year and (last.month, last.day) < (12, 24)
        )
        mark = '<span class="partial" title="partial year">*</span>' if partial else ""
        y = years.get(year, math.nan)
        b = bench_years.get(year, math.nan)
        bg, fg = heat_colour(y / 3 if np.isfinite(y) else y)  # a year is ~12 months of moves
        rows.append(
            f"<tr><th>{year}{mark}</th>{''.join(cells)}"
            f'<td class="heat year" style="background:{bg};color:{fg}">{_pct(y, 1)}</td>'
            f'<td class="bench">{_pct(b, 1)}</td></tr>'
        )
    return (
        '<table class="heatmap"><thead><tr><th></th>'
        f'{head}<th class="year">Year</th><th class="bench">Bench.</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


#: `(label, Metrics field, formatter)` for the metrics table.
METRIC_ROWS: tuple[tuple[str, str, object], ...] = (
    ("Total return", "total_return", lambda v: _pct(v, 1)),
    ("CAGR", "cagr", lambda v: _pct(v, 2)),
    ("Volatility (ann.)", "volatility", lambda v: _pct(v, 2)),
    ("Sharpe", "sharpe", _num),
    ("Sortino", "sortino", _num),
    ("Max drawdown", "max_drawdown", lambda v: _pct(v, 1)),
    ("Max drawdown peak", "max_drawdown_peak", str),
    ("Max drawdown trough", "max_drawdown_trough", str),
    ("Max drawdown recovered", "max_drawdown_recovery", lambda v: str(v) if v else "not yet"),
    ("Longest drawdown (sessions)", "longest_drawdown_sessions", _int),
    ("Hit rate (sessions up)", "hit_rate", lambda v: _pct(v, 1)),
    ("Average win", "avg_win", lambda v: _pct(v, 2)),
    ("Average loss", "avg_loss", lambda v: _pct(v, 2)),
    ("Avg win / avg loss", "payoff", _num),
    ("Turnover (one-way, x/yr)", "turnover", _num),
    ("Avg holding (sessions)", "avg_holding", _int),
)


def _metrics_table(run, logged) -> str:
    m, b = run.metrics, run.benchmark_metrics
    rows = [
        f"<tr><th>{label}</th><td>{fmt(getattr(m, field))}</td>"
        f"<td>{fmt(getattr(b, field))}</td></tr>"
        for label, field, fmt in METRIC_ROWS
    ]
    iv = run.interval
    rows.insert(
        4,
        f"<tr><th>Sharpe {iv.level:.0%} interval</th>"
        f"<td>[{_num(iv.low)}, {_num(iv.high)}]</td><td></td></tr>",
    )
    if logged is not None:
        fw = logged.familywise
        rows.insert(
            5,
            f"<tr><th>Sharpe {fw.level:.2%} interval ({logged.n_variants} variants)</th>"
            f"<td>[{_num(fw.low)}, {_num(fw.high)}]</td><td></td></tr>",
        )
    rows.append(
        f"<tr><th>Frictions paid</th><td>{_eur(run.result.total_cost)}</td>"
        f"<td>{_eur(run.benchmark.total_cost)}</td></tr>"
    )
    return (
        '<table class="metrics"><thead><tr><th></th>'
        f'<th><span class="key" style="background:{STRATEGY_COLOUR}"></span>'
        f"{html.escape(run.label)}</th>"
        f'<th><span class="key" style="background:{BENCHMARK_COLOUR}"></span>'
        f"{html.escape(run.benchmark_ticker)}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _tile(title: str, value: str, sub: str) -> str:
    return (
        f'<div class="tile"><div class="tile-title">{title}</div>'
        f'<div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>'
    )


def _tiles(run) -> str:
    m, b, iv, d = run.metrics, run.benchmark_metrics, run.interval, run.vs_benchmark
    return "".join(
        [
            _tile("CAGR", _pct(m.cagr, 1), f"benchmark {_pct(b.cagr, 1)}"),
            _tile(
                f"Sharpe, {iv.level:.0%} interval",
                f"{_num(iv.point)} <small>[{_num(iv.low)}, {_num(iv.high)}]</small>",
                f"benchmark {_num(b.sharpe)}",
            ),
            _tile("Max drawdown", _pct(m.max_drawdown, 1), f"benchmark {_pct(b.max_drawdown, 1)}"),
            _tile(
                "Sharpe minus benchmark",
                f"{_num(d.point, sign=True)} <small>[{_num(d.low, sign=True)}, "
                f"{_num(d.high, sign=True)}]</small>",
                f"P(no better than benchmark) {d.p_le_zero:.0%}",
            ),
        ]
    )


_CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: {PLANE}; color: {INK};
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
header h1 {{ font-size: 24px; margin: 0 0 4px; font-weight: 650; }}
header p {{ margin: 0; color: {INK_2}; }}
section {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
  padding: 18px 20px; margin-top: 18px; }}
h2 {{ font-size: 15px; margin: 0 0 2px; font-weight: 650; }}
.note {{ color: {INK_2}; margin: 0 0 10px; font-size: 13px; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 20px; }}
.tile {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10); border-radius: 10px;
  padding: 14px 16px; }}
.tile-title {{ color: {INK_2}; font-size: 12px; }}
.tile-value {{ font-size: 26px; font-weight: 650; margin: 2px 0; }}
.tile-value small {{ font-size: 13px; font-weight: 500; color: {INK_2}; }}
.tile-sub {{ color: {MUTED}; font-size: 12px; }}
svg {{ width: 100%; height: auto; display: block; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
.metrics {{ width: 100%; }}
.metrics th, .metrics td {{ padding: 5px 10px; border-bottom: 1px solid {GRID}; }}
.metrics tbody th {{ text-align: left; font-weight: 500; color: {INK_2}; }}
.metrics td {{ text-align: right; }}
.metrics thead th {{ text-align: right; font-weight: 650; }}
.key {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }}
.heatmap {{ width: 100%; font-size: 12px; border-collapse: separate; border-spacing: 2px; }}
.heatmap th {{ color: {INK_2}; font-weight: 500; padding: 2px 6px; }}
.heatmap tbody th {{ text-align: left; }}
.heatmap td {{ text-align: center; padding: 5px 2px; border-radius: 3px; }}
.heatmap td.bench {{ color: {INK_2}; }}
.heatmap .year {{ font-weight: 650; }}
.partial {{ color: {MUTED}; }}
.legend-strip {{ display: flex; gap: 2px; align-items: center; font-size: 12px; color: {INK_2};
  margin-top: 8px; }}
.legend-strip span.sw {{ width: 28px; height: 12px; border-radius: 2px; display: inline-block; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 0; }}
dt {{ color: {INK_2}; }} dd {{ margin: 0; }}
footer {{ color: {MUTED}; font-size: 12px; margin-top: 18px; }}
@media (max-width: 760px) {{ .tiles {{ grid-template-columns: repeat(2, 1fr); }} }}
"""


def _legend_strip() -> str:
    steps = (-HEAT_CAP, -HEAT_CAP / 2, -HEAT_CAP / 8, 0.0, HEAT_CAP / 8, HEAT_CAP / 2, HEAT_CAP)
    swatches = "".join(
        f'<span class="sw" style="background:{heat_colour(v)[0]}" '
        f'title="{_pct(v, 0, sign=True)}"></span>'
        for v in steps
    )
    return (
        f'<div class="legend-strip">{_pct(-HEAT_CAP, 0)} or worse {swatches} '
        f"{_pct(HEAT_CAP, 0, sign=True)} or better · year column on a third of the scale · "
        "* partial year</div>"
    )


def render_tearsheet(run, logged=None, *, generated: datetime | None = None) -> str:
    """The whole page for a :class:`~finmgr.backtest.strategy.StrategyRun` as an HTML string.

    `logged` is what :func:`~finmgr.backtest.strategy.log_run` returned; without
    it the page leaves out the variant count and the familywise interval.
    """
    curve, bench = run.result.curve, run.benchmark.curve
    m = run.metrics
    window = run.periods
    strat_name, bench_name = run.label, run.benchmark_ticker
    equity_svg = _line_chart(
        [(strat_name, curve, STRATEGY_COLOUR), (bench_name, bench, BENCHMARK_COLOUR)],
        name="equity",
        ylabel="Equity (EUR, log scale)",
        fmt=_eur,
        log=True,
        height=3.8,
    )
    drawdown_svg = _line_chart(
        [
            (strat_name, drawdown_series(curve), STRATEGY_COLOUR),
            (bench_name, drawdown_series(bench), BENCHMARK_COLOUR),
        ],
        name="drawdown",
        ylabel="From running peak",
        fmt=_pct0,
        zero=True,
        fill=True,
        height=2.6,
    )
    sharpe_svg = _line_chart(
        [
            (
                strat_name,
                rolling_sharpe(run.excess, window=window, periods=window),
                STRATEGY_COLOUR,
            ),
            (
                bench_name,
                rolling_sharpe(run.benchmark_excess, window=window, periods=window),
                BENCHMARK_COLOUR,
            ),
        ],
        name="sharpe",
        ylabel="Sharpe, trailing 12 months",
        fmt=_num1,
        zero=True,
        height=2.6,
    )
    costs = run.costs
    scaling = (
        f", scaled by inverse liquidity ({costs.min_liquidity_scale:g}x to "
        f"{costs.max_liquidity_scale:g}x around €{costs.liquidity_ref_eur / 1e6:.0f}M ADV)"
        if costs.liquidity_scaling
        else ""
    )
    variants = ""
    if logged is not None:
        fw = logged.familywise
        variants = (
            f"<dt>Variants tested</dt><dd>{logged.n_variants} in <code>docs/trials.csv</code> "
            f"(this one: <code>{logged.variant_id}</code>). Holding all {logged.n_variants} "
            f"intervals jointly at 95% means quoting this Sharpe at {fw.level:.2%}: "
            f"<b>[{_num(fw.low)}, {_num(fw.high)}]</b>.</dd>"
        )
    iv = run.interval
    stamp = (generated or datetime.now()).strftime("%Y-%m-%d %H:%M")
    title = f"{strat_name} vs {bench_name}"
    sizing = f"top {run.top_k} of {run.n_tickers}, equal weight" if run.top_k > 1 else "one name"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — finmgr tearsheet</title>
<style>{_CSS}</style></head>
<body><main>
<header>
<h1>{html.escape(title)}</h1>
<p>{m.start} to {m.end} ({m.years:.1f} years, {m.sessions:,} sessions) · {sizing},
held {run.hold} sessions · data as of {run.as_of} · costs and next-open fills included</p>
</header>
<div class="tiles">{_tiles(run)}</div>

<section><h2>Equity curve</h2>
<p class="note">€{run.result.equity["equity_eur"].iloc[0]:,.0f} invested in each on
{m.start}, total return reinvested, after every cost. Log scale: equal heights are
equal percentage moves.</p>
{equity_svg}</section>

<section><h2>Drawdown</h2>
<p class="note">Fall from the running peak. Worst for the strategy:
{_pct(m.max_drawdown, 1)}, {m.max_drawdown_peak} to {m.max_drawdown_trough},
{"recovered " + str(m.max_drawdown_recovery) if m.max_drawdown_recovery else "not yet recovered"}.
</p>
{drawdown_svg}</section>

<section><h2>Rolling 12-month Sharpe</h2>
<p class="note">Excess return over the euro overnight rate, trailing {window} sessions
(one year on this axis), annualised. Full-period Sharpe {_num(iv.point)},
{iv.level:.0%} interval [{_num(iv.low)}, {_num(iv.high)}].</p>
{sharpe_svg}</section>

<section><h2>Monthly returns</h2>
<p class="note">Strategy, month by month; the last two columns are the calendar year
for the strategy and for {html.escape(bench_name)}.</p>
{_heatmap(curve, bench)}
{_legend_strip()}</section>

<section><h2>Metrics</h2>
<p class="note">Day 37 definitions (<code>finmgr/backtest/metrics.py</code>),
annualised at {run.periods} sessions a year, Sharpe and Sortino against
{"the euro overnight rate" if run.rf_used else "zero"}.</p>
{_metrics_table(run, logged)}</section>

<section><h2>How this was run</h2>
<dl>
<dt>Strategy</dt><dd>{html.escape(strat_name)}: {sizing}, rebalanced every {run.hold}
sessions; a held name still in the top K is kept, not re-bought.</dd>
<dt>Window</dt><dd>first decision {run.start}, last date {m.end}; data as of {run.as_of}
(nothing later was read).</dd>
<dt>Fills</dt><dd>decided at a session's close, traded at the next session's open.</dd>
<dt>Costs</dt><dd>per side: {costs.commission_bps:g} bps commission + half of a
{costs.spread_bps:g} bps spread + {costs.slippage_bps:g} bps slippage{scaling}; the
configured <code>costs:</code> in <code>config/settings.yaml</code>.</dd>
<dt>Benchmark</dt><dd>{html.escape(bench_name)} bought on the first date and held, same
costs, same date axis.</dd>
<dt>Sharpe interval</dt><dd>circular block bootstrap of daily excess returns,
{iv.n_boot:,} resamples, blocks of {iv.block} sessions.</dd>
{variants}
</dl></section>

<footer>Generated by <code>finmgr backtest --strategy {run.strategy} --report</code> on
{stamp}. Past performance in a backtest is an upper bound on what to expect,
not an estimate of it.</footer>
</main></body></html>
"""


def default_name(strategy: str, as_of: date) -> str:
    return f"backtest_{strategy}_{as_of.isoformat()}.html"
