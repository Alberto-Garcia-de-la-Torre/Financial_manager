"""Run identity and logging.

Every invocation of `finmgr` mints one `run_id` — a UTC timestamp, the short
git SHA of the checkout it ran from, and a random token so two runs started in
the same second can never collide. That id is stamped on every log line, which
makes the first and last line of a run trivially greppable:

    grep 20260916T083012-1895961-4f2a logs/finmgr.log

Two handlers are installed: a Rich one on the console at the level the user
asked for, and a rotating file handler at `logs/finmgr.log` that always records
DEBUG. The file is the record you read in eight weeks; the console is for now.

    from finmgr import runlog

    run = runlog.start_run(settings, level="INFO", command="ingest")
    runlog.get_logger(__name__).info("...")
    runlog.end_run(status="ok", exit_code=0)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from finmgr.config import PROJECT_ROOT, Settings, settings_path

#: Name of the logger everything in the package hangs off.
ROOT_LOGGER = "finmgr"

#: The rotating log file, relative to the configured log directory.
LOG_FILENAME = "finmgr.log"

#: Rotate at 5 MB and keep five generations. A daily run writes kilobytes, so
#: this holds months of history without anyone pruning it.
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

#: The file line format. `run_id` comes from :class:`_RunIdFilter`, so a line
#: written by any library logger still carries the run it belongs to.
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(run_id)s %(name)s %(message)s"
FILE_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

#: Placeholder for records emitted outside a run (import-time warnings, say).
NO_RUN = "-"


@dataclass(frozen=True)
class RunContext:
    """Everything that identifies one invocation."""

    run_id: str
    started_at: datetime
    started_monotonic: float
    git_sha: str
    git_dirty: bool
    command: str
    argv: list[str] = field(default_factory=list)
    pid: int = 0
    log_file: Path | None = None
    settings_file: Path | None = None

    def elapsed_seconds(self) -> float:
        """Wall-clock seconds since the run started."""
        return time.monotonic() - self.started_monotonic


# The run currently in progress. Module-level because a process is one run.
_current_run: RunContext | None = None

# Handlers this module installed, so a second start_run() in the same process
# (tests, notebooks) replaces them instead of duplicating every line.
_installed_handlers: list[logging.Handler] = []


class _RunIdFilter(logging.Filter):
    """Attach the active run id to every record that passes through."""

    def filter(self, record: logging.LogRecord) -> bool:
        run = _current_run
        record.run_id = run.run_id if run is not None else NO_RUN
        return True


def git_sha(short: int = 7) -> str | None:
    """Short SHA of HEAD, or ``None`` outside a git checkout."""
    return _git(["rev-parse", f"--short={short}", "HEAD"])


def git_is_dirty() -> bool:
    """True when the working tree has uncommitted changes.

    Worth recording: a run from a dirty tree is not reproducible from its SHA
    alone, and that is exactly the sort of thing you forget two months later.
    """
    return bool(_git(["status", "--porcelain"]))


def _git(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def mint_run_id(now: datetime | None = None, sha: str | None = None) -> str:
    """Build a run id: ``<utc timestamp>-<short sha>-<random token>``.

    The timestamp orders runs and the SHA says which code ran. Neither is
    unique on its own — two runs can start in the same second from the same
    commit — so four random hex characters make the id collision-proof.
    """
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    revision = sha if sha is not None else (git_sha() or "nogit")
    token = os.urandom(2).hex()
    return f"{stamp}-{revision}-{token}"


def current_run() -> RunContext | None:
    """The run in progress, or ``None`` if logging was never set up."""
    return _current_run


def get_logger(name: str | None = None) -> logging.Logger:
    """A logger under the `finmgr` root, so it inherits the handlers."""
    if not name or name == ROOT_LOGGER:
        return logging.getLogger(ROOT_LOGGER)
    if name.startswith(f"{ROOT_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def setup_logging(
    settings: Settings,
    level: str | int = "INFO",
    *,
    console: Console | None = None,
) -> Path:
    """Install the console and rotating file handlers. Returns the log path."""
    global _installed_handlers

    logger = logging.getLogger(ROOT_LOGGER)
    for handler in _installed_handlers:
        logger.removeHandler(handler)
        handler.close()
    _installed_handlers = []

    console_level = logging.getLevelName(level) if isinstance(level, str) else level
    if not isinstance(console_level, int):
        raise ValueError(f"unknown log level {level!r}")

    # The logger passes everything to the handlers; each handler decides. The
    # file keeps DEBUG whatever the console was asked for, because the detail
    # you need is always the detail you did not ask to see.
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    run_filter = _RunIdFilter()

    rich_handler = RichHandler(
        console=console or Console(),
        level=console_level,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
        log_time_format="%H:%M:%S",
    )
    rich_handler.addFilter(run_filter)
    logger.addHandler(rich_handler)
    _installed_handlers.append(rich_handler)

    log_dir = settings.ensure_log_dir()
    log_file = log_dir / LOG_FILENAME
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=FILE_DATEFMT))
    file_handler.addFilter(run_filter)
    logger.addHandler(file_handler)
    _installed_handlers.append(file_handler)

    return log_file


def start_run(
    settings: Settings,
    *,
    level: str | int = "INFO",
    command: str = "",
    argv: list[str] | None = None,
    console: Console | None = None,
    config_file: Path | None = None,
) -> RunContext:
    """Mint a run id, set logging up and write the opening lines of the run."""
    global _current_run

    sha = git_sha()
    log_file = setup_logging(settings, level=level, console=console)
    run = RunContext(
        run_id=mint_run_id(sha=sha),
        started_at=datetime.now(UTC),
        started_monotonic=time.monotonic(),
        git_sha=sha or "nogit",
        git_dirty=git_is_dirty(),
        command=command or "-",
        argv=list(argv if argv is not None else sys.argv[1:]),
        pid=os.getpid(),
        log_file=log_file,
        settings_file=config_file if config_file is not None else settings_path(),
    )
    _current_run = run

    log = get_logger(__name__)
    log.info(
        "RUN START run_id=%s command=%s git=%s%s python=%s pid=%d argv=%s",
        run.run_id,
        run.command,
        run.git_sha,
        "+dirty" if run.git_dirty else "",
        sys.version.split()[0],
        run.pid,
        json.dumps(run.argv),
    )
    log.debug(
        "logging to %s (rotating at %d bytes, %d kept); console level %s",
        log_file,
        MAX_BYTES,
        BACKUP_COUNT,
        level,
    )
    # The config actually used, not the config on disk: a run that read an
    # override through $FINMGR_SETTINGS should say so in its own log.
    log.info(
        "RUN CONFIG run_id=%s source=%s values=%s",
        run.run_id,
        run.settings_file,
        json.dumps(settings.model_dump(mode="json"), sort_keys=True),
    )
    return run


def end_run(status: str = "ok", exit_code: int = 0, **extra: Any) -> RunContext | None:
    """Write the closing line of the run and clear the context.

    Safe to call when no run started — the CLI wrapper calls it from a
    `finally`, and `finmgr --help` never gets as far as starting one.
    """
    global _current_run

    run = _current_run
    if run is None:
        return None

    trailer = "".join(f" {key}={value}" for key, value in sorted(extra.items()))
    get_logger(__name__).log(
        logging.INFO if exit_code == 0 else logging.WARNING,
        "RUN END   run_id=%s command=%s status=%s exit_code=%d duration=%.3fs%s",
        run.run_id,
        run.command,
        status,
        exit_code,
        run.elapsed_seconds(),
        trailer,
    )
    for handler in _installed_handlers:
        handler.flush()
    _current_run = None
    return run
