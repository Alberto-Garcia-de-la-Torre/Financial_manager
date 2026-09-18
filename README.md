# Financial_manager

A personal quantitative research tool. It ingests fifteen years of daily bars
for a fixed universe of 100 companies, turns them into a feature panel, scores
every company against the other 99 on the same day, and backtests the result
honestly enough to say whether the edge is real.

The question it exists to answer is not "is this stock good?" — that has no
answer — but **"of these 100, which ones today, and why?"** Everything is
cross-sectional: an RSI of 30 means nothing on its own, while "lowest RSI of
the 100 today" is something you can act on.

Two commitments shape the design, and both cost more work than ignoring them
would:

- **No look-ahead.** Every read is `as_of` a date and refuses rows dated after
  it. A backtest that quietly sees the future looks brilliant and is worthless.
- **Beat a real bar, or say so.** Buy-and-hold, equal weight, random portfolios
  and plain momentum are measured first, on identical splits and identical
  costs. A model that loses to random picks is a finding, not a failure — and
  it gets written down either way.

Known limitations, stated up front: the universe is chosen today and backtested
backwards, which is **survivorship bias**; fundamentals from Yahoo are not
point-in-time and are out of scope for v1; and yfinance is an unofficial,
rate-limited scraper that will fail sometimes.

This is a research tool for personal decisions. It is not investment advice.

`ROADMAP.md` is the plan — sixty sessions, one per day — and says which parts
are built and which are still stubs.

## The daily workflow

Once per clone:

```bash
make install    # .venv with pinned dependencies
make hooks      # git pre-commit hooks (see "Commit discipline")
```

Then each working session:

```bash
make check      # lint + tests, the state you start and finish in
# ... make the day's change, with its test ...
make fmt        # apply formatting and the safe lint fixes
make check
git add -p && git commit
```

The hooks run again on commit, so a tree that passes `make check` commits
without argument and a tree that does not is stopped before it lands.

The pipeline the project runs for its own sake — ingest, check, features,
train, rank, report — is described under "Command line" and is still stubbed
out; `finmgr daily` chains it on a timer from day 55.

## Setup

Requires Python 3.11+ and `make`.

```bash
make install
```

That creates `.venv` and installs the package editable with its dev extras.
The equivalent by hand:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

With the environment active, `python -c "import finmgr"` works from any
directory. Dependency versions are pinned in `pyproject.toml`.

## Tests and lint

Every target bootstraps `.venv` first, so a fresh clone needs nothing but
`make test`:

```bash
make test    # pytest
make lint    # ruff check + ruff format --check, changes nothing
make fmt     # ruff format + ruff check --fix
make check   # lint, then test
```

Both tools are configured in `pyproject.toml`. Tests live in `tests/` and
never touch the network, `data/` or the real `config/settings.yaml` — the
autouse fixture in `tests/conftest.py` clears `FINMGR_SETTINGS` and everything
writes under pytest's `tmp_path`. `automation/`, `scratch/` and Markdown files
are outside ruff's scope.

## Commit discipline

```bash
make hooks      # once per clone: writes .git/hooks/pre-commit
make hooks-all  # run every hook over the whole tree
```

The hooks are declared in `tools/pre-commit-config.yaml` — not at the usual
`.pre-commit-config.yaml`, so every entry point names the file explicitly, and
`pre-commit` invoked by hand needs `--config tools/pre-commit-config.yaml`.
There are three:

| hook | what it does |
| --- | --- |
| `no-data-in-git` | refuses any staged path under `data/`, and any Parquet, Feather, HDF5, pickle or database file wherever it sits |
| `ruff-check` | lints the staged Python, applying the safe fixes |
| `ruff-format` | formats the staged Python |

`no-data-in-git` is the one that matters. `.gitignore` hides `data/`, but a
`git add -f` walks straight past it, and a 15-year daily panel for 100 tickers
committed once is in the history of every clone forever — `git rm` tomorrow
does not get it back out. So the hook refuses the commit outright:

```
$ git add -f data/bars/ticker=AAPL/part.parquet && git commit -m "oops"
refuse market data (data/, Parquet and friends)..........................Failed
- hook id: no-data-in-git
- exit code: 1

Market data does not belong in git. These staged paths are refused:

data/bars/ticker=AAPL/part.parquet

ruff (lint, with safe fixes applied).................(no files to check)Skipped
ruff-format..........................................(no files to check)Skipped
```

Nothing is committed: the working tree still has the file, the index still has
it staged, and `git log` is exactly where it was.

`finmgr/data/` is the storage layer's source code and stays committable — the
rule is anchored at the repository root. `.csv` is deliberately not on the
blocked list, because `config/universe.csv` is source too.

`tests/test_commit_hooks.py` proves all of this the only way that means
anything: it builds a throwaway git repository under `tmp_path`, installs this
same configuration into it and attempts real commits. Every hook is a *local*
hook, so nothing is cloned or downloaded at commit time and the ruff that runs
here is the pinned one in `.venv` — the same binary `make lint` uses, rather
than a second version pin free to drift.

`git commit --no-verify` skips all of it, and `SKIP=ruff-format git commit`
skips one. Both leave a trace in the diff; neither is a habit.

## Configuration

`config/settings.yaml` holds the data directory, the log directory, the
universe path, the base currency (EUR), the timezone (Europe/Madrid) and the
history start date. It is
loaded and validated through the pydantic `Settings` model in
`finmgr/config.py`:

```python
from finmgr.config import load_settings

settings = load_settings()
settings.data_dir        # absolute Path, created on demand via ensure_data_dir()
settings.history_start   # datetime.date
```

Relative paths in the YAML are resolved against the repository root, so a run
started from any working directory reads and writes the same files. Set
`FINMGR_SETTINGS` to point at a different file.

## Command line

Installing the package puts a `finmgr` command on the path. The six
subcommands are the daily pipeline, in the order they run:

```
finmgr --help

  ingest     Download daily bars for the universe into the data store.
  check      Run data quality checks over the stored bars.
  features   Build the feature panel from the stored bars.
  train      Train a model walk-forward on the feature panel.
  rank       Score and rank the universe as of a given date.
  report     Render the daily HTML report.
```

All six are stubs at this point. Each prints `not implemented yet` along with
the roadmap day that fills it in, and exits non-zero so nothing mistakes an
unwritten stage for a successful one.

`--log-level` is a global option, so it goes before the subcommand:

```bash
finmgr --log-level DEBUG ingest
```

## Logging and run ids

Every invocation mints a `run_id` — UTC timestamp, short git SHA, and a random
token so two runs started in the same second can never collide:

```
20260916T090020-1895961-5f74
```

That id is stamped on every line of the rotating log at `logs/finmgr.log`
(5 MB × 5 generations, never committed), and a run is bracketed by three lines
you can grep for:

```
RUN START  run_id=... command=ingest git=1895961+dirty python=3.11.9 pid=8086 argv=[...]
RUN CONFIG run_id=... source=config/settings.yaml values={...}
RUN END    run_id=... command=ingest status=failed exit_code=1 duration=0.013s
```

`RUN CONFIG` records the settings the run actually used — including an
override loaded through `FINMGR_SETTINGS` — so a surprising result can be
traced back to the configuration that produced it. `RUN END` is written from a
`finally` block, so it appears even when a command fails or raises, with
`status` one of `ok`, `failed`, `crashed` or `interrupted`.

To read back a single run:

```bash
grep 20260916T090020-1895961-5f74 logs/finmgr.log
```

The console gets the same records through Rich at whatever `--log-level` asks
for; the file always keeps `DEBUG`.
