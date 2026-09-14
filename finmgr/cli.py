"""The `finmgr` command line.

Six subcommands make up the daily pipeline, in the order they run:

    ingest -> check -> features -> train -> rank -> report

All six are stubs today. Each one prints what it will do and exits non-zero,
so nothing downstream can mistake "not written yet" for "ran successfully".
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="finmgr",
    help="Personal quantitative research tool for a 100-company universe.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()

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
    console.print(
        f"[bold yellow]finmgr {command}: not implemented yet[/bold yellow] "
        f"(planned for {_PLANNED[command]})."
    )
    raise typer.Exit(code=1)


@app.callback()
def main() -> None:
    """Personal quantitative research tool for a 100-company universe.

    Run the pipeline stages in order: ingest, check, features, train, rank,
    report.
    """


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


if __name__ == "__main__":
    app()
