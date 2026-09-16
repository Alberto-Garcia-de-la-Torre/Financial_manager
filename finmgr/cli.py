"""The `finmgr` command line.

Six subcommands make up the daily pipeline, in the order they run:

    ingest -> check -> features -> train -> rank -> report

All six are stubs today. Each one prints what it will do and exits non-zero,
so nothing downstream can mistake "not written yet" for "ran successfully".

Every invocation is bracketed by a RUN START / RUN END pair in
`logs/finmgr.log`, both carrying the run id minted in `finmgr.runlog`. The
`run()` wrapper below is the console entry point rather than the Typer app
itself, so the closing line gets written even when a command fails or raises.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console

from finmgr import runlog
from finmgr.config import load_settings

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
    "ingest": "day 11",
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
def ingest() -> None:
    """Download daily bars for the universe into the data store."""
    _not_implemented("ingest")


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
