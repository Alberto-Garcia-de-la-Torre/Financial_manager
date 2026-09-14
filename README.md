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

`config/settings.yaml` holds the data directory, the universe path, the base
currency (EUR), the timezone (Europe/Madrid) and the history start date. It is
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
