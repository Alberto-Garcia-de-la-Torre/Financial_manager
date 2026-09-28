"""The `finmgr` command line.

Six subcommands make up the daily pipeline, in the order they run:

    ingest -> check -> features -> train -> rank -> report

`ingest` is real from day 11; the other five are still stubs. Beside the
pipeline, `coverage` (day 13) reports what the store holds per ticker. Each stub prints
what it will do and exits non-zero, so nothing downstream can mistake "not
written yet" for "ran successfully".

Every invocation is bracketed by a RUN START / RUN END pair in
`logs/finmgr.log`, both carrying the run id minted in `finmgr.runlog`. The
`run()` wrapper below is the console entry point rather than the Typer app
itself, so the closing line gets written even when a command fails or raises.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from finmgr import runlog
from finmgr.config import load_settings
from finmgr.data import ingest as ingest_module
from finmgr.data import store
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
    "check": "day 19",
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
) -> None:
    """Download daily bars for the universe into the data store.

    Incremental by default: each ticker is asked for only what the store is
    missing, and what comes back is merged on `(ticker, date)` — so running
    this twice in a row writes nothing the second time. `--full` re-requests
    the window whole.

    One bad symbol cannot stop the run: every ticker gets its own retries and
    its own status row, and the report at the end says what made it onto disk.
    Exits non-zero if anything failed or was skipped — with the report printed
    either way.
    """
    settings = load_settings()
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
    ingest_module.render_console(report, console, show_all=show_all)
    if not report.complete:
        log.warning(
            "finmgr ingest: %d failed, %d skipped",
            len(report.failed),
            len(report.skipped),
        )
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
def check() -> None:
    """Run data quality checks over the stored bars."""
    _not_implemented("check")


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
