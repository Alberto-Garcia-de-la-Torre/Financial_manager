#!/usr/bin/env python3
"""Day 11's acceptance criterion, run for real against live Yahoo.

    python tools/pull_the_cable.py                  # 8 tickers, cut after 3
    python tools/pull_the_cable.py --alive 1 --ticker AAPL --ticker SAN.MC

`tests/test_ingest.py` proves the ingestion loop survives a dead network using
a fetcher that raises what a dead network raises. This script proves the same
thing without the simulation: it starts a genuine ingest making genuine Yahoo
requests, and then *actually takes the network away from the running process*
part-way through, with `unshare(CLONE_NEWNET)` plus a shutdown of every socket
that was already open. From that point the process lives in a fresh network
namespace holding nothing but loopback — no interface, no route, no reachable
resolver — and every later request fails the way it fails when the cable comes
out: `connect()` gets ENETUNREACH, which yfinance's curl stack reports as
`curl: (7) Failed to connect to query2.finance.yahoo.com:443`.

It needs no privileges (Fedora enables unprivileged user namespaces) and it
cannot affect anything outside this process: the namespace belongs to this PID
alone and disappears when it exits. Nothing is restored afterwards because
nothing else was touched.

The bars go to a throwaway directory under `/tmp`, never to `data/` — this is a
demonstration, not an ingest, and a half-finished download is not something the
real store should inherit.

What it should print: the tickers fetched before the cut reported `ok` and
present on disk, the rest reported `failed` or `skipped` with the reason, one
summary line, and exit code 1 — with no traceback anywhere, because nothing
escaped the loop.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import socket
import stat
import sys
import tempfile
from pathlib import Path

#: From linux/sched.h.
CLONE_NEWUSER = 0x10000000
CLONE_NEWNET = 0x40000000


def _unshare(flags: int, what: str) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.unshare(flags) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"unshare({what}) failed with errno {errno}")


def claim_a_user_namespace() -> None:
    """Become root of our own user namespace, while still single-threaded.

    Two kernel rules shape when this can happen. Dropping the network needs
    CAP_SYS_ADMIN, which an ordinary user only gets by owning a user namespace;
    and `unshare(CLONE_NEWUSER)` is refused with EINVAL once the process has
    more than one thread. Importing pandas starts eight, so this has to run
    before the imports below — which is why it is called at module level rather
    than from `main()`.

    Mapping our own uid and gid to themselves is what keeps files readable and
    writable afterwards; without a mapping every file on disk would belong to
    `nobody` as far as this process is concerned.
    """
    uid, gid = os.geteuid(), os.getegid()
    _unshare(CLONE_NEWUSER, "CLONE_NEWUSER")
    Path("/proc/self/setgroups").write_text("deny")
    Path("/proc/self/uid_map").write_text(f"{uid} {uid} 1")
    Path("/proc/self/gid_map").write_text(f"{gid} {gid} 1")


def cut_the_network() -> int:
    """Take the network away from this process. Local, and final.

    Two things have to happen, and the second is the one that is easy to miss.

    `unshare(CLONE_NEWNET)` moves the process into a namespace holding nothing
    but loopback, so every *new* connection fails with `[Errno 101] Network is
    unreachable`. It does not touch the sockets that are already open: those
    stay attached to the namespace they were created in, and yfinance keeps one
    alive between requests. Run with the namespace switch alone, this script
    quietly ingests all eight tickers over a connection that should have died —
    which is the opposite of what it exists to show.

    A cable coming out of a socket breaks the established connections too, so
    the second step shuts every open socket down. What the ingest sees after
    that is what it would see on a real machine: the pooled connection fails,
    and the reconnect has nowhere to go.

    Returns how many sockets were shut down.
    """
    _unshare(CLONE_NEWNET, "CLONE_NEWNET")

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    shut = 0
    for entry in Path("/proc/self/fd").iterdir():
        try:
            if not stat.S_ISSOCK(entry.stat().st_mode):
                continue
        except OSError:
            continue
        # SHUT_RDWR through libc rather than through a socket object, which
        # would take ownership of a file descriptor it does not own.
        if libc.shutdown(int(entry.name), socket.SHUT_RDWR) == 0:
            shut += 1
    return shut


if __name__ == "__main__":
    claim_a_user_namespace()

# The repo root, so the script runs from a checkout without being installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from finmgr import runlog  # noqa: E402
from finmgr.config import load_settings  # noqa: E402
from finmgr.data.ingest import TickerIngest, ingest_universe, render_console  # noqa: E402

#: Enough tickers to have some before the cut and plenty after it.
DEFAULT_TICKERS = ["AAPL", "MSFT", "SAN.MC", "SAP.DE", "ASML.AS", "AIR.PA", "NESN.SW", "7203.T"]

#: A short window: this is about failure handling, not about history.
DEFAULT_START = "2024-11-01"
DEFAULT_END = "2024-11-08"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/pull_the_cable.py",
        description="Ingest for real, sever the network mid-run, and print the report.",
    )
    parser.add_argument("--ticker", action="append", default=None, metavar="SYMBOL")
    parser.add_argument(
        "--alive",
        type=int,
        default=3,
        help="How many tickers to fetch before the network is cut (default: 3).",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Where to write the bars (default: a fresh directory under /tmp).",
    )
    parser.add_argument("--abort-after", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--backoff", type=float, default=1.0)
    args = parser.parse_args(argv)

    symbols = args.ticker or DEFAULT_TICKERS
    root = args.data_dir or Path(tempfile.mkdtemp(prefix="finmgr-pull-the-cable-"))
    console = Console()
    settings = load_settings()
    runlog.start_run(settings, level="INFO", command="pull-the-cable", console=console)

    console.print(
        f"Ingesting {len(symbols)} tickers into [bold]{root}[/bold]; "
        f"the network goes away after {args.alive} of them."
    )

    done = 0

    def cut_after_enough(result: TickerIngest) -> None:
        """Called by the loop after each ticker — the cable comes out here."""
        nonlocal done
        done += 1
        if done == args.alive:
            shut = cut_the_network()
            console.print(
                f"[bold red]>>> network severed after {result.ticker}: "
                f"empty netns, {shut} open socket(s) cut <<<[/bold red]"
            )

    report = ingest_universe(
        symbols,
        start=args.start,
        end=args.end,
        root=root / "bars" / "daily",
        pause=args.pause,
        backoff=args.backoff,
        abort_after=args.abort_after,
        on_result=cut_after_enough,
    )

    render_console(report, console, show_all=True)
    exit_code = 0 if report.complete else 1
    runlog.end_run(status="ok" if exit_code == 0 else "failed", exit_code=exit_code)
    console.print(f"\nBars on disk under {root}, exit code {exit_code}.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
