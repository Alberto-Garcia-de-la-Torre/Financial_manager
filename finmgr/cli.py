"""The `finmgr` command line.

Six subcommands make up the daily pipeline, in the order they run:

    ingest -> check -> features -> train -> rank -> report

`ingest` is real from day 11, and from day 14 records every run in
`data/meta/ingest_runs.parquet` — `ingest --report` reads the last one back.
`features`, `train`, `rank` and `report` are still stubs. Beside the pipeline, `coverage` (day 13)
reports what the store holds per ticker, and `actions` (day 15) ingests splits
and dividends into `data/actions/` — `actions --show` prints them back out of
the store — and `adjusted` (day 16) builds the back-adjusted close from both,
with `--compare` checking it against Yahoo's. `sessions` (day 17) holds the
store against each venue's exchange calendar and lists the sessions genuinely
missing in a year, holidays excluded. `fx` (day 18) ingests the EUR crosses
into `data/fx/` and writes `close_eur` into every bar partition; `fx --compare`
shows one ticker's local and EUR series diverging as FX says they must.
`check` (day 19) runs the data quality checks and fails on any flag not yet
triaged in `config/quality_triage.csv`, and `quarantine` (day 20) hides every
row that register calls bad-data from reads — flagged in the store, recorded
with its reason in `data/quarantine/`, never deleted. Each stub prints what it will do and
exits non-zero, so nothing downstream can mistake "not written yet" for "ran
successfully".

Every invocation is bracketed by a RUN START / RUN END pair in
`logs/finmgr.log`, both carrying the run id minted in `finmgr.runlog`. The
`run()` wrapper below is the console entry point rather than the Typer app
itself, so the closing line gets written even when a command fails or raises.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import typer
from rich.console import Console
from rich.table import Table

from finmgr import runlog
from finmgr.config import Settings, load_settings
from finmgr.data import actions as actions_module
from finmgr.data import adjcheck, calendars, pit, quality, store
from finmgr.data import adjust as adjust_module
from finmgr.data import fx as fx_module
from finmgr.data import ingest as ingest_module
from finmgr.data import manifest as manifest_module
from finmgr.data import quarantine as quarantine_module
from finmgr.data import universe as universe_module

app = typer.Typer(
    name="finmgr",
    help="Personal quantitative research tool for a 100-company universe.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
log = runlog.get_logger(__name__)

# Which roadmap day turns each stub into a real command. Printed with the
# stub message so it is obvious that nothing is missing by accident.
_PLANNED = {
    "features": "day 31",
    "train": "day 42",
    "rank": "day 49",
    "report": "day 57",
}


def _not_implemented(command: str) -> None:
    """Report a stub clearly and fail, rather than pretending to have worked."""
    # No rich markup in the message itself: the same string goes to the file
    # log, and tags there would be noise. RichHandler colours by level anyway.
    log.warning(
        "finmgr %s: not implemented yet (planned for %s).",
        command,
        _PLANNED[command],
    )
    raise typer.Exit(code=1)


@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Console verbosity. The file log at logs/ always keeps DEBUG.",
        case_sensitive=False,
        metavar="LEVEL",
    ),
) -> None:
    """Personal quantitative research tool for a 100-company universe.

    Run the pipeline stages in order: ingest, check, features, train, rank,
    report.
    """
    level = log_level.upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise typer.BadParameter(f"unknown log level {log_level!r}", param_hint="--log-level")

    settings = load_settings()
    runlog.start_run(
        settings,
        level=level,
        command=ctx.invoked_subcommand or "-",
        console=console,
    )


@app.command()
def ingest(
    # Spelled with Annotated rather than as a default like the options below
    # it: a repeatable option is a list, and a call in the default of a
    # mutably-annotated argument is the thing bugbear's B008 exists to catch.
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Ingest only this symbol; repeatable. Skips the universe file.",
            metavar="SYMBOL",
        ),
    ] = None,
    start: str = typer.Option(
        None, "--start", help="First session date (default: history_start from settings)."
    ),
    end: str = typer.Option(None, "--end", help="Last session date, inclusive (default: today)."),
    attempts: int = typer.Option(
        ingest_module.DEFAULT_ATTEMPTS, "--attempts", help="Tries per ticker before giving up."
    ),
    pause: float = typer.Option(
        ingest_module.DEFAULT_PAUSE, "--pause", help="Seconds between tickers."
    ),
    backoff: float = typer.Option(
        ingest_module.DEFAULT_BACKOFF,
        "--backoff",
        help="Seconds before the first retry; doubles after each failed attempt.",
    ),
    abort_after: int = typer.Option(
        ingest_module.DEFAULT_ABORT_AFTER,
        "--abort-after",
        help="Give up after this many consecutive failures (0 never gives up).",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Re-request the whole window instead of only what the store is missing.",
    ),
    refetch_days: int = typer.Option(
        ingest_module.DEFAULT_REFETCH_DAYS,
        "--refetch-days",
        help="Days of overlap re-requested before each ticker's newest stored bar.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Fetch and report, but write nothing to the store."
    ),
    show_all: bool = typer.Option(
        False, "--all", "-a", help="List every ingested ticker, not just the first twenty."
    ),
    report_only: bool = typer.Option(
        False,
        "--report",
        help="Download nothing: summarise the last recorded run from the manifest.",
    ),
) -> None:
    """Download daily bars for the universe into the data store.

    Incremental by default: each ticker is asked for only what the store is
    missing, and what comes back is merged on `(ticker, date)` — so running
    this twice in a row writes nothing the second time. `--full` re-requests
    the window whole.

    Every run that writes is recorded in `data/meta/ingest_runs.parquet`, one
    row per ticker. `--report` prints the last recorded run from that file
    alone — nothing is fetched, the store is not opened, and the fetch options
    are ignored.

    One bad symbol cannot stop the run: every ticker gets its own retries and
    its own status row, and the report at the end says what made it onto disk.
    Exits non-zero if anything failed or was skipped — with the report printed
    either way.
    """
    settings = load_settings()
    if report_only:
        _report_last_run(settings, show_all=show_all)
        return

    report = ingest_module.ingest_universe(
        ticker or None,
        start=start,
        end=end,
        attempts=attempts,
        backoff=backoff,
        pause=pause,
        abort_after=abort_after,
        write=not dry_run,
        incremental=not full,
        refetch_days=refetch_days,
        settings=settings,
    )
    if not dry_run:
        # After the run rather than per ticker: the loop never raises, so the
        # report is complete by the time it returns, interrupted runs included.
        manifest_module.append_run(report, settings=settings)
    ingest_module.render_console(report, console, show_all=show_all)
    if not report.complete:
        log.warning(
            "finmgr ingest: %d failed, %d skipped",
            len(report.failed),
            len(report.skipped),
        )
        raise typer.Exit(code=1)


def _report_last_run(settings: Settings, *, show_all: bool) -> None:
    """Print the newest run in the manifest; fail if there is none, or it failed.

    The exit code mirrors the run being reported, so `finmgr ingest --report`
    in a script says whether last night's ingest was complete.
    """
    path = manifest_module.manifest_path(settings=settings)
    report = manifest_module.last_run(manifest_module.read_manifest(path=path))
    if report is None:
        log.warning("finmgr ingest --report: no runs recorded in %s", path)
        raise typer.Exit(code=1)

    started = report.started_at.isoformat(timespec="seconds") if report.started_at else "?"
    mode = "incremental" if report.incremental else "full"
    console.print(
        f"Last ingest run [bold]{report.run_id}[/bold], started {started}, "
        f"{mode}, window {report.start} to {report.end}"
    )
    ingest_module.render_console(report, console, show_all=show_all)
    if not report.complete:
        raise typer.Exit(code=1)


@app.command()
def coverage(
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Report only this symbol; repeatable. Skips the universe file.",
            metavar="SYMBOL",
        ),
    ] = None,
) -> None:
    """Print first date, last date and row count for every ticker.

    Reads the store only — nothing is downloaded. The tickers are the
    universe's, in file order, so a symbol the backfill never got onto disk
    shows up as a row of dashes rather than going missing from the table.
    Exits non-zero if any of them has no bars at all.
    """
    settings = load_settings()
    symbols = ticker or universe_module.tickers(settings=settings)
    frame = store.coverage(symbols, settings=settings)

    table = Table(title="Stored daily bars")
    table.add_column("ticker", style="bold")
    table.add_column("first date")
    table.add_column("last date")
    table.add_column("rows", justify="right")
    table.add_column("years", justify="right")
    for row in frame.itertuples(index=False):
        stored = row.rows > 0
        years = (row.last_date - row.first_date).days / 365.25 if stored else None
        table.add_row(
            row.ticker,
            row.first_date.isoformat() if stored else "-",
            row.last_date.isoformat() if stored else "-",
            str(row.rows) if stored else "[red]0[/red]",
            f"{years:.1f}" if years is not None else "-",
        )
    console.print(table)

    missing = frame.loc[frame["rows"] == 0, "ticker"].tolist()
    held = frame.loc[frame["rows"] > 0]
    summary = f"{len(held)} of {len(frame)} tickers stored; {int(frame['rows'].sum())} bars"
    if not held.empty:
        summary += f", {held['first_date'].min()} to {held['last_date'].max()}"
    console.print(summary)
    if missing:
        log.warning("finmgr coverage: no bars stored for %s", ", ".join(missing))
        raise typer.Exit(code=1)


@app.command()
def sessions(
    year: int = typer.Option(..., "--year", "-y", help="Calendar year to check, e.g. 2024."),
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Only this universe symbol; repeatable.",
            metavar="SYMBOL",
        ),
    ] = None,
    show_all: bool = typer.Option(
        False, "--all", "-a", help="List every ticker, not just those with gaps or closures."
    ),
) -> None:
    """List genuinely missing sessions per ticker for one year, holidays excluded.

    Each ticker is held against its venue's exchange calendar (the universe's
    `exchange` column), from its first stored bar or 1 January to 31 December
    or yesterday. Reads the store only. Exits non-zero if any session is
    missing or any bar sits on a day the venue was closed.
    """
    settings = load_settings()
    try:
        gaps = calendars.session_gaps(ticker or None, year=year, settings=settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Sessions against the exchange calendar, {year}")
    table.add_column("ticker", style="bold")
    table.add_column("venue")
    table.add_column("expected", justify="right")
    table.add_column("stored", justify="right")
    table.add_column("missing", overflow="fold")
    table.add_column("off calendar", overflow="fold")
    table.add_column("excused closures", overflow="fold")
    for gap in gaps:
        if not (show_all or gap.missing or gap.off_calendar or gap.excused):
            continue
        table.add_row(
            gap.ticker,
            gap.venue,
            str(gap.expected),
            str(gap.stored),
            f"[red]{_dates(gap.missing)}[/red]" if gap.missing else "-",
            f"[red]{_dates(gap.off_calendar)}[/red]" if gap.off_calendar else "-",
            "\n".join(f"{day} {why}" for day, why in gap.excused) or "-",
        )
    console.print(table)

    dirty = [gap for gap in gaps if not gap.clean]
    console.print(
        f"{len(gaps) - len(dirty)} of {len(gaps)} tickers complete for {year}; "
        f"{sum(len(g.missing) for g in gaps)} missing sessions, "
        f"{sum(len(g.off_calendar) for g in gaps)} off-calendar bars, "
        f"{sum(len(g.excused) for g in gaps)} excused closures"
    )
    if dirty:
        log.warning(
            "finmgr sessions: gaps against the calendar for %s",
            ", ".join(gap.ticker for gap in dirty),
        )
        raise typer.Exit(code=1)


def _dates(days: tuple[date, ...], limit: int = 12) -> str:
    """ISO dates, comma-separated, cut short with a count past `limit`."""
    shown = ", ".join(day.isoformat() for day in days[:limit])
    return shown if len(days) <= limit else f"{shown} (+{len(days) - limit} more)"


@app.command()
def adjusted(
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Only this symbol; repeatable. Skips the universe file.",
            metavar="SYMBOL",
        ),
    ] = None,
    compare: bool = typer.Option(
        False, "--compare", help="Check against Yahoo's auto_adjust=True close (online)."
    ),
    report: Annotated[
        Path | None,
        typer.Option(
            "--report", help="With --compare: also write a Markdown report here.", metavar="PATH"
        ),
    ] = None,
    rows: int = typer.Option(5, "--rows", help="Without --compare: sessions to print per ticker."),
    as_of: str | None = typer.Option(
        None,
        "--as-of",
        help="Without --compare: adjust with only what was known by this date (default: today).",
    ),
) -> None:
    """Print our back-adjusted close, or check it against Yahoo's with --compare.

    Without --compare the store alone is read and the last few sessions of
    each --ticker are printed with their split and dividend factors. With
    --compare every ticker is checked against Yahoo and each one diverging by
    more than 0.5% is listed with its cause; exits non-zero if any divergence
    has no cause or any ticker could not be compared.
    """
    settings = load_settings()
    if not compare:
        if not ticker:
            raise typer.BadParameter("name at least one --ticker, or pass --compare")
        try:
            point = store.as_date(as_of, "--as-of") or pit.LATEST
            frame = adjust_module.read_adjusted(ticker, as_of=point, settings=settings)
        except (ValueError, pit.LookAheadError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        if frame.empty:
            log.warning("finmgr adjusted: nothing stored for %s", ", ".join(ticker))
            raise typer.Exit(code=1)
        table = Table(title="Back-adjusted close")
        for column in adjust_module.ADJUSTED_COLUMNS:
            table.add_column(column, justify="left" if column in ("ticker", "date") else "right")
        for _, group in frame.groupby("ticker", sort=False):
            for row in group.tail(rows).itertuples(index=False):
                table.add_row(
                    row.ticker,
                    row.date.isoformat(),
                    f"{row.close:.4f}",
                    f"{row.split_factor:.6f}",
                    f"{row.dividend_factor:.6f}",
                    f"{row.adj_close:.4f}",
                )
        console.print(table)
        return

    results = adjcheck.compare_universe(ticker or None, settings=settings)
    diverging = sorted((r for r in results if r.diverges), key=lambda r: r.divergence, reverse=True)
    table = Table(title=f"Divergence over {adjcheck.DIVERGENCE_THRESHOLD:.1%} from Yahoo")
    table.add_column("ticker", style="bold")
    table.add_column("max", justify="right")
    table.add_column("worst", justify="left")
    table.add_column("closes", justify="right")
    table.add_column("dividends", justify="right")
    table.add_column("method", justify="right")
    table.add_column("cause", overflow="fold")
    for r in diverging:
        table.add_row(
            r.ticker,
            f"{r.divergence:.2%}",
            str(r.worst_date),
            f"{r.from_closes:.2%}",
            f"{r.from_dividends:.2%}",
            f"{r.from_method:.2%}",
            "\n".join(r.causes) if r.causes else "[red]unexplained[/red]",
        )
    console.print(table)
    failed = [r for r in results if r.error]
    unexplained = [r for r in diverging if not r.explained]
    for r in failed:
        console.print(f"[red]{r.ticker}: {r.error}[/red]")
    console.print(
        f"{len(diverging)} of {len(results)} tickers diverge; "
        f"{len(unexplained)} unexplained; {len(failed)} not compared"
    )
    if report is not None:
        run = runlog.current_run()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            adjcheck.render_markdown(
                results,
                checked_at=datetime.now(ZoneInfo(settings.timezone)),
                run_id=run.run_id if run is not None else None,
            ),
            encoding="utf-8",
        )
        console.print(f"wrote {report}")
    if unexplained or failed:
        log.warning(
            "finmgr adjusted --compare: unexplained %s; not compared %s",
            ", ".join(r.ticker for r in unexplained) or "-",
            ", ".join(r.ticker for r in failed) or "-",
        )
        raise typer.Exit(code=1)


@app.command()
def actions(
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Only this symbol; repeatable. Skips the universe file.",
            metavar="SYMBOL",
        ),
    ] = None,
    start: str = typer.Option(
        None, "--start", help="First date (default: history_start from settings)."
    ),
    end: str = typer.Option(None, "--end", help="Last date, inclusive (default: today)."),
    show: bool = typer.Option(
        False, "--show", help="Download nothing: print the stored actions instead."
    ),
    kind: str = typer.Option(
        None, "--kind", help="With --show: only 'split' or 'dividend'.", metavar="KIND"
    ),
) -> None:
    """Ingest splits and dividends into data/actions/, or print them with --show.

    Without --show every ticker's actions are fetched for the whole window and
    merged into the store on (ticker, date, action); exits non-zero if any
    ticker failed. With --show the store alone is read and printed.
    """
    settings = load_settings()
    if show:
        _show_actions(settings, ticker, kind=kind, start=start, end=end)
        return

    results = actions_module.ingest_actions(ticker or None, start=start, end=end, settings=settings)
    table = Table(title="Corporate actions")
    table.add_column("ticker", style="bold")
    table.add_column("splits", justify="right")
    table.add_column("dividends", justify="right")
    table.add_column("new", justify="right")
    table.add_column("error", overflow="fold")
    for result in results:
        table.add_row(
            result.ticker,
            str(result.splits),
            str(result.dividends),
            str(result.added),
            f"[red]{result.error}[/red]" if result.error else "",
        )
    console.print(table)
    failed = [result.ticker for result in results if not result.ok]
    console.print(
        f"{len(results) - len(failed)} of {len(results)} tickers; "
        f"{sum(r.splits for r in results)} splits, {sum(r.dividends for r in results)} dividends, "
        f"{sum(r.added for r in results)} new"
    )
    if failed:
        log.warning("finmgr actions: failed for %s", ", ".join(failed))
        raise typer.Exit(code=1)


def _show_actions(
    settings: Settings,
    tickers: list[str] | None,
    *,
    kind: str | None,
    start: str | None,
    end: str | None,
) -> None:
    """Print stored actions; fail if there are none to print."""
    try:
        frame = actions_module.read_actions(
            tickers or None, action=kind, start=start, end=end, as_of=pit.LATEST, settings=settings
        )
    except (ValueError, pit.LookAheadError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if frame.empty:
        log.warning("finmgr actions --show: nothing stored for that selection")
        raise typer.Exit(code=1)

    table = Table(title="Stored corporate actions")
    table.add_column("ticker", style="bold")
    table.add_column("date")
    table.add_column("action")
    table.add_column("value", justify="right")
    for row in frame.itertuples(index=False):
        table.add_row(
            row.ticker,
            row.date.isoformat(),
            row.action,
            actions_module.describe(row.action, row.value),
        )
    console.print(table)


@app.command()
def fx(
    pair: Annotated[
        list[str] | None,
        typer.Option(
            "--pair",
            "-p",
            help="Only this EUR cross, e.g. EURUSD; repeatable (default: every one the "
            "universe needs).",
            metavar="PAIR",
        ),
    ] = None,
    start: str = typer.Option(
        None, "--start", help="First date (default: history_start from settings)."
    ),
    end: str = typer.Option(None, "--end", help="Last date, inclusive (default: today)."),
    no_fetch: bool = typer.Option(
        False, "--no-fetch", help="Download nothing: recompute close_eur from the stored rates."
    ),
    compare: str = typer.Option(
        None,
        "--compare",
        help="Download nothing: compare one ticker's local and EUR closes over "
        "--start..--end, from the store.",
        metavar="SYMBOL",
    ),
    plot: Annotated[
        Path | None,
        typer.Option(
            "--plot", help="With --compare: also draw both series to this PNG.", metavar="PATH"
        ),
    ] = None,
    rows: int = typer.Option(12, "--rows", help="With --compare: sessions sampled into the table."),
) -> None:
    """Ingest EUR exchange rates into data/fx/ and write close_eur into the bar store.

    By default every cross the universe needs (EURUSD, EURGBP, EURCHF, EURSEK
    and the rest) is fetched for the whole window and merged on (pair, date);
    then close_eur is recomputed for every stored ticker and the partitions
    where it changed are rewritten. Exits non-zero if a cross failed or a
    ticker could not be converted. Run it after `finmgr ingest`.

    With --compare the store alone is read: one ticker's local and EUR series
    over the window, and whether the gap between them is the one FX implies.
    Exits non-zero if it is not.
    """
    settings = load_settings()
    if compare:
        _compare_currencies(settings, compare, start=start, end=end, plot=plot, rows=rows)
        return

    failed: list[str] = []
    if not no_fetch:
        try:
            results = fx_module.ingest_fx(pair or None, start=start, end=end, settings=settings)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        stored = fx_module.read_fx(as_of=pit.LATEST, settings=settings)
        table = Table(title="EUR exchange rates")
        table.add_column("pair", style="bold")
        table.add_column("fetched", justify="right")
        table.add_column("new", justify="right")
        table.add_column("stored", justify="right")
        table.add_column("from")
        table.add_column("to")
        table.add_column("last rate", justify="right")
        table.add_column("error", overflow="fold")
        for result in results:
            held = stored.loc[stored["pair"] == result.pair]
            table.add_row(
                result.pair,
                str(result.rows),
                str(result.added),
                str(len(held)),
                str(held["date"].iloc[0]) if len(held) else "-",
                str(held["date"].iloc[-1]) if len(held) else "-",
                f"{held['rate'].iloc[-1]:.4f}" if len(held) else "-",
                f"[red]{result.error}[/red]" if result.error else "",
            )
        console.print(table)
        failed = [result.pair for result in results if not result.ok]

    conversions = fx_module.refresh_close_eur(settings=settings)
    broken = [c for c in conversions if c.error]
    for c in broken:
        console.print(f"[red]{c.ticker}: {c.error}[/red]")
    console.print(
        f"close_eur: {len(conversions) - len(broken)} of {len(conversions)} tickers converted, "
        f"{sum(c.converted for c in conversions)} sessions, "
        f"{sum(c.unconverted for c in conversions)} left unconverted, "
        f"{sum(1 for c in conversions if c.path is not None)} partitions rewritten"
    )
    if failed or broken:
        log.warning(
            "finmgr fx: rates failed for %s; not converted %s",
            ", ".join(failed) or "-",
            ", ".join(c.ticker for c in broken) or "-",
        )
        raise typer.Exit(code=1)


def _compare_currencies(
    settings: Settings,
    ticker: str,
    *,
    start: str | None,
    end: str | None,
    plot: Path | None,
    rows: int,
) -> None:
    """Print a ticker's local vs EUR series and say whether FX explains the gap."""
    try:
        summary, series = fx_module.divergence(ticker, start=start, end=end, settings=settings)
    except ValueError as exc:
        log.warning("finmgr fx --compare: %s", exc)
        raise typer.Exit(code=1) from exc
    if summary.pair is None:
        log.warning("finmgr fx --compare: %s is listed in EUR; there is no FX to compare", ticker)
        raise typer.Exit(code=1)

    table = Table(title=f"{ticker}: {summary.currency} against EUR")
    table.add_column("date")
    table.add_column("close as quoted", justify="right")
    table.add_column("close EUR", justify="right")
    table.add_column(summary.pair, justify="right")
    table.add_column(f"{summary.currency} =100", justify="right")
    table.add_column("EUR =100", justify="right")
    picks = sorted({round(i * (len(series) - 1) / max(rows - 1, 1)) for i in range(max(rows, 2))})
    for row in series.iloc[picks].itertuples(index=False):
        table.add_row(
            row.date.isoformat(),
            f"{row.close:,.2f}",
            f"{row.close_eur:,.2f}",
            f"{row.rate:.4f}",
            f"{row.local_index:.1f}",
            f"{row.eur_index:.1f}",
        )
    console.print(table)

    euro = "strengthened" if summary.fx_return > 0 else "weakened"
    console.print(
        f"{summary.start} to {summary.end}: {ticker} {summary.local_return:+.2%} in "
        f"{summary.currency}, {summary.eur_return:+.2%} in EUR. {summary.pair} "
        f"{summary.fx_return:+.2%} (the euro {euro}), so the EUR series should "
        f"{'lag' if summary.fx_return > 0 else 'lead'} by {abs(summary.expected_gap):.2%}; "
        f"it {'lags' if summary.gap < 0 else 'leads'} by {abs(summary.gap):.2%}."
    )
    if plot is not None:
        console.print(f"wrote {fx_module.plot_divergence(summary, series, plot)}")
    if not summary.consistent:
        log.warning("finmgr fx --compare: %s's EUR series does not move with FX", ticker)
        raise typer.Exit(code=1)
    console.print("[green]consistent: the EUR series diverges in the direction FX moved[/green]")


@app.command()
def check(
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Only this universe symbol; repeatable.",
            metavar="SYMBOL",
        ),
    ] = None,
    show_all: bool = typer.Option(
        False, "--all", "-a", help="List triaged flags one by one, not just their counts."
    ),
    as_of: str | None = typer.Option(
        None, "--as-of", help="Last session the calendar checks expect (default: yesterday)."
    ),
) -> None:
    """Run the data quality checks over the stored bars, severity-tagged.

    Flags bad prices, high below low, closes outside the day's range,
    duplicate sessions, unexplained moves over 40%, a close stuck five
    sessions running, and gaps against the exchange calendar. Flags matched by
    `config/quality_triage.csv` are triaged and summarised; the rest are listed.
    Reads the store only. Exits non-zero while any flag is untriaged.
    """
    settings = load_settings()
    try:
        last = store.as_date(as_of, "--as-of")
        result = quality.run_checks(ticker or None, as_of=last, settings=settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    shown = result.flags if show_all else tuple(result.untriaged)
    if shown:
        table = Table(title="Data quality flags" + ("" if show_all else ", untriaged"))
        table.add_column("severity")
        table.add_column("check")
        table.add_column("ticker", style="bold")
        table.add_column("date")
        table.add_column("detail", overflow="fold")
        table.add_column("triage", overflow="fold")
        for flag in shown:
            colour = "red" if flag.severity is quality.Severity.ERROR else "yellow"
            triage = f"{flag.triage.verdict}: {flag.triage.reason}" if flag.triage else "-"
            table.add_row(
                f"[{colour}]{flag.severity}[/{colour}]",
                flag.check,
                flag.ticker,
                flag.date.isoformat(),
                flag.detail,
                triage,
            )
        console.print(table)

    triaged = [flag for flag in result.flags if flag.triaged]
    if triaged and not show_all:
        summary = Table(title="Triaged flags (--all lists them)")
        summary.add_column("severity")
        summary.add_column("check")
        summary.add_column("verdict")
        summary.add_column("flags", justify="right")
        summary.add_column("tickers", justify="right")
        groups: dict[tuple[str, str, str], list[quality.Flag]] = {}
        for flag in triaged:
            groups.setdefault((flag.severity, flag.check, flag.triage.verdict), []).append(flag)
        for (severity, name, verdict), flags in groups.items():
            summary.add_row(
                severity, name, verdict, str(len(flags)), str(len({f.ticker for f in flags}))
            )
        console.print(summary)

    for rule in result.unused_rules:
        log.warning(
            "finmgr check: triage rule matched nothing: %s %s %s..%s",
            rule.check,
            rule.scope,
            rule.start,
            rule.end,
        )

    untriaged = result.untriaged
    errors = sum(flag.severity is quality.Severity.ERROR for flag in untriaged)
    console.print(
        f"{len(result.tickers)} tickers checked; {len(result.flags)} flags, "
        f"{len(triaged)} triaged, {len(untriaged)} untriaged "
        f"({errors} errors, {len(untriaged) - errors} warnings)"
    )
    if untriaged:
        log.warning(
            "finmgr check: untriaged flags for %s",
            ", ".join(sorted({flag.ticker for flag in untriaged})),
        )
        raise typer.Exit(code=1)


@app.command()
def quarantine(
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Only this universe symbol; repeatable.",
            metavar="SYMBOL",
        ),
    ] = None,
    as_of: str | None = typer.Option(
        None, "--as-of", help="Last session the calendar checks expect (default: yesterday)."
    ),
) -> None:
    """Hide every row the triage register calls bad-data from reads; delete nothing.

    Runs the same checks as `check`, sets the store's `quarantined` flag on each
    bad-data row and records it, with its reason, under `data/quarantine/`.
    A row no longer called bad-data is released, its record kept. Re-running
    with nothing changed writes nothing.
    """
    settings = load_settings()
    try:
        last = store.as_date(as_of, "--as-of")
        results = quarantine_module.apply_quarantine(
            tickers=ticker or None, as_of=last, settings=settings
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    records = quarantine_module.read_quarantine(
        [r.ticker for r in results], as_of=pit.LATEST, settings=settings
    )
    if not records.empty:
        table = Table(title="Quarantined rows")
        table.add_column("ticker", style="bold")
        table.add_column("date")
        table.add_column("check")
        table.add_column("detail", overflow="fold")
        table.add_column("reason", overflow="fold")
        for record in records.itertuples():
            table.add_row(
                record.ticker, record.date.isoformat(), record.check, record.detail, record.reason
            )
        console.print(table)

    console.print(
        f"{sum(r.quarantined for r in results)} row(s) quarantined across "
        f"{sum(bool(r.quarantined) for r in results)} ticker(s); this run "
        f"+{sum(r.added for r in results)} added, {sum(r.released for r in results)} released, "
        f"{sum(r.path is not None for r in results)} partition(s) rewritten"
    )


@app.command()
def features() -> None:
    """Build the feature panel from the stored bars."""
    _not_implemented("features")


@app.command()
def train() -> None:
    """Train a model walk-forward on the feature panel."""
    _not_implemented("train")


@app.command()
def rank() -> None:
    """Score and rank the universe as of a given date."""
    _not_implemented("rank")


@app.command()
def report() -> None:
    """Render the daily HTML report."""
    _not_implemented("report")


def run() -> int:
    """Console entry point: run the app and always close the run out.

    Typer/click signal completion by raising SystemExit, so the exit code is
    caught here and written to the log rather than escaping silently.
    """
    status, exit_code = "ok", 0
    try:
        app()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
        status = "ok" if exit_code == 0 else "failed"
    except KeyboardInterrupt:
        exit_code, status = 130, "interrupted"
    except BaseException:
        exit_code, status = 1, "crashed"
        log.exception("unhandled exception")
    finally:
        runlog.end_run(status=status, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
