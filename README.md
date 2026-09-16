# Financial_manager
This is a personal financial manager to make quick research through the market.

## Setup

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

With the environment active, `python -c "import finmgr"` works from any
directory. Dependency versions are pinned in `pyproject.toml`.

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
