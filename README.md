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

## The universe

`config/universe.csv` is the fixed list of 100 companies everything else turns
around — one row per company, six columns:

```
ticker,name,exchange,currency,sector,country
AAPL,Apple Inc.,XNAS,USD,Information Technology,United States
SAN.MC,Banco Santander SA,XMAD,EUR,Financials,Spain
```

`ticker` is Yahoo's own spelling, suffix included (`SAN.MC`, `AIR.PA`,
`SAP.DE`, `ASML.AS`, `NOVO-B.CO`, `7203.T`) — day 8 checks every one of them
returns bars. `exchange` is the venue's MIC, which is what day 17 hands to
`exchange_calendars` to tell a missing session apart from a Madrid holiday.
`currency` is the ISO code the venue quotes in; note that Yahoo reports London
prices in pence rather than pounds, so the EUR conversion on day 18 has that
one special case to handle. `sector` is a GICS sector, spelled exactly as GICS
spells it, and is what day 30 demeans features against.

The mix is 40 US large caps, 40 European from the IBEX 35, CAC 40, DAX and AEX,
and 20 from the UK, Switzerland, the Nordics, Italy, Japan, Taiwan, Hong Kong,
Canada and Australia. All eleven GICS sectors appear, the smallest — Real
Estate — three times; Financials is the largest at 17. Sector counts are
deliberately uneven because the listed universes are, and a ranking that
pretends otherwise is not ranking the market it claims to.

The list is chosen today and backtested backwards, which is the
**survivorship bias** named at the top of this README: companies that failed or
were acquired over the last fifteen years are missing, so historical returns
here are flattering by an amount this project does not attempt to estimate.

`tests/test_universe.py` asserts the invariants downstream code assumes —
exactly 100 rows, no duplicate ticker, no sector appearing only once, GICS
sector names, ISO currency codes, MIC exchange codes — so a hand edit that
breaks one of them fails `make test` rather than surfacing as a strange
ranking six weeks later.

Code reads the file through `finmgr.data.universe`, never by opening the CSV
itself:

```python
from finmgr.data.universe import load_universe

for company in load_universe():
    print(company.ticker, company.exchange, company.sector)
```

### Checking the tickers are alive

A ticker that is real but misspelled does not raise: yfinance answers an
unknown or delisted symbol with an *empty* DataFrame, so the company simply
never appears in any ranking. `AIR.PA` is Airbus in Paris; `AIR` on its own is
AAR Corp in New York, and both "work".

```bash
python -m finmgr.data.validate                                    # all 100
python -m finmgr.data.validate --ticker ROG.SW --ticker RO.SW     # just these
python -m finmgr.data.validate --report docs/universe_validation.md
```

Each ticker gets `yf.Ticker(t).history(period="5d")` and one of four verdicts —
`ok`, `stale` (bars, but the newest is over `--max-age-days` old), `empty` or
`error`. An empty or failed answer is retried up to `--attempts` times, because
Yahoo answers a throttled request exactly the way it answers a dead symbol and
one empty frame is not grounds for editing the universe file. The command
prints the failure list and exits non-zero if it is not empty. It fetches five
days, keeps nothing and writes nothing under `data/` — day 11 is where
downloading becomes a real pipeline.

The last run is recorded in [`docs/universe_validation.md`](docs/universe_validation.md):
**100 of 100 returned recent bars on 22 Sep 2026.** Getting there took one
replacement. Yahoo no longer has `ROG.SW`, the Roche participation certificate
that carries almost all of Roche's volume — it 404s with "Quote not found".
The bearer share `RO.SW` does return bars, but at roughly CHF 9M of daily
turnover against Novartis's CHF 356M it would be excluded by day 28's
liquidity filter anyway, leaving a dead slot in the universe. So the row is now
`NOVN.SW`, Novartis AG: same country, same GICS sector, same venue and
currency, full history back to 2010.

## The bar store

Every daily bar the project uses lives in one place, `finmgr/data/store.py`,
in one schema fixed before anything was downloaded:

| column | dtype | |
| --- | --- | --- |
| `ticker` | `str` | Yahoo's spelling, suffix included |
| `date` | `date32[day][pyarrow]` | the session's calendar date, exchange-local |
| `open`, `high`, `low`, `close` | `float64` | raw prices, unadjusted |
| `volume` | `float64` | |

On disk it is one Parquet file per ticker, in a Hive-style partition:

```
data/bars/daily/ticker=AAPL/part.parquet
data/bars/daily/ticker=SAN.MC/part.parquet
```

A single ticker's fifteen years is one small file, so re-ingesting it rewrites
nothing else; DuckDB reads the whole set as one table and skips the partitions
a ticker filter cannot match; and when something looks wrong in eight weeks,
the file to open follows from the ticker alone. The `ticker` column is stored
*inside* each file as well as in its directory name, so a partition is
readable on its own.

```python
from finmgr.data.store import read_bars, write_bars

write_bars(frame)                                    # frame in the schema above
read_bars()                                          # everything stored
read_bars("AAPL")                                    # one ticker
read_bars(["AAPL", "SAN.MC"], start="2024-01-01")    # bounds are inclusive
```

`write_bars` replaces each ticker's partition whole — the file is always
exactly what was last handed over for that symbol — writing to a temporary
name and renaming over the target, so an interrupted run leaves the previous
partition intact rather than a truncated file. Merging new bars into stored
history is day 12's job and builds on this. Reads go through DuckDB rather
than `pd.read_parquet` because the filtering belongs next to the data: the
date bound is pushed into the Parquet row groups and the rows the call did not
ask for are never materialised.

`conform_bars()` is the single place the schema is enforced, and it runs on
both the write and the read path — so what comes out is dtype-identical to
what went in, whatever the caller arrived with. It refuses a missing or
unexpected column, an empty ticker, a missing date, a repeated
`(ticker, date)`, and a ticker that would not work as a directory name. It
also refuses a **timezone-aware** `date` column, which is the trap worth
naming: casting one to a date converts through UTC first, so a Tokyo close at
15:00 JST silently lands on the previous calendar day — every session,
forever. Day 10 hands over exchange-local dates deliberately.

What it does *not* do is judge the prices. A negative close, a high below its
low or a NaN goes into the store unchanged: day 19 flags those and day 20
quarantines them, and dropping them here would lose the evidence. An empty
store, or a filter matching nothing, returns an empty frame with the full
schema rather than something every caller has to test for.

`tests/test_store.py` covers it, including day 9's acceptance criterion — a
synthetic frame survives a write/read round trip with dtypes identical.

## Fetching one ticker

`finmgr/data/fetch.py` is the only place a Yahoo response becomes bars. It
hands back exactly the schema above, ready for `write_bars` with nothing in
between:

```python
from finmgr.data.fetch import fetch_daily
from finmgr.data.store import write_bars

write_bars(fetch_daily("SAN.MC", "2024-10-28", "2024-11-11"))
write_bars(fetch_daily("AAPL"))   # history_start .. today, from settings
```

`start` and `end` are **inclusive**, the same way `read_bars` means them —
Yahoo's exclusive `end` is converted here, once, so the same two words cannot
bound a window differently depending on which function was called. Bars
outside the requested window are trimmed, so two calls with touching ranges
cannot both claim the same session.

Three decisions everything downstream inherits:

- **`auto_adjust=False`.** Yahoo's dividend-adjusted close is a moving number:
  the whole history shifts every time a dividend is paid, so a bar stored last
  month no longer matches the same bar today and no backtest is reproducible.
  The raw close is what gets stored; `Adj Close`, `Dividends` and
  `Stock Splits` are dropped, day 15 ingests the actions as facts in their own
  right, and day 16 builds the adjusted series from them. Note what the flag
  does *not* buy: Yahoo applies **splits** upstream, so August 2020 AAPL comes
  back in post-split money whatever is asked for — a property of the source,
  which day 16's comparison has to account for.
- **Exchange-local dates.** yfinance stamps a Madrid session `00:00:00+01:00`,
  which is 23:00 UTC the day *before*. The calendar date is taken from the
  timestamp as it arrives, and the store refuses a timezone-aware `date`
  column to keep that mistake off disk.
- **Errors raise, empties do not.** A failed request — timeout, 404,
  rate-limit rejection — raises, so day 11's retry loop gets to see it. A
  window with no bars in it is not a failure: it returns an empty frame
  carrying the full schema. Yahoo does not draw that line itself, raising
  `YFPricesMissingError` for a delisted symbol and for a live one asked about
  a bank holiday in the same words, so `download_history` draws it — otherwise
  day 12's "nothing new since Friday" would look like a broken ticker every
  weekend.

Prices pass through untouched — a NaN close or a high below its low is Yahoo's
answer, and day 19 is where it gets flagged. Nothing here writes to disk and
nothing here retries: one call, one ticker, one frame.

`tests/test_fetch.py` is day 10's acceptance criterion — the conversion is
pinned against **saved yfinance responses**, never the network. The fixtures
live in `tests/fixtures/yahoo/` as the raw CSV plus a JSON sidecar recording
the ticker, window, exchange timezone, dtypes and yfinance version; AAPL spans
its 4:1 split of 2020-08-31 and SAN.MC spans a €0.10 dividend in Madrid. The
test module replaces the one function that can reach Yahoo with a stub that
fails the test, so the suite is offline by construction rather than by
convention. To refresh or add a fixture — the only part that uses the network:

```bash
python tests/yahoo_fixtures.py AAPL 2020-08-24 2020-09-04
```

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
