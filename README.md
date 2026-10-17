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
train, rank, report — is described under "Command line". Only `ingest` is built
so far; `finmgr daily` chains the lot on a timer from day 55.

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
make leak    # the look-ahead leak test alone; after every change to finmgr/data/
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
| `close_eur` | `float64` | the close in EUR, derived (day 18); NaN until converted |

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
from finmgr.data.pit import LATEST
from finmgr.data.store import max_stored_dates, merge_bars, read_bars, write_bars

write_bars(frame)                                    # replace a ticker's partition
merge_bars(frame)                                    # ... or fold it into what is there
read_bars(as_of=LATEST)                              # everything stored, up to today
read_bars("AAPL", as_of="2024-06-28")                # one ticker, as known that evening
read_bars(["AAPL", "SAN.MC"], start="2024-01-01", as_of="2024-06-28")  # inclusive bounds
max_stored_dates()                                   # {"AAPL": date(2026, 9, 25), ...}
```

`write_bars` replaces each ticker's partition whole — the file is always
exactly what was last handed over for that symbol — writing to a temporary
name and renaming over the target, so an interrupted run leaves the previous
partition intact rather than a truncated file. Reads go through DuckDB rather
than `pd.read_parquet` because the filtering belongs next to the data: the
date bound is pushed into the Parquet row groups and the rows the call did not
ask for are never materialised.

`merge_bars` is the write path an ingest actually uses. It keeps the stored
history and folds the new frame into it, keyed on `(ticker, date)`: unseen
sessions are added, sessions already on disk are **overwritten** by the new
values, and stored sessions the frame says nothing about are left alone. It
returns one `MergeResult` per ticker — `added`, `updated`, `unchanged`,
`stored` and the path written — and a partition where nothing was added or
updated is not rewritten at all, so its `path` is `None` and its mtime still
means "this is when the data last changed". NaN counts as equal to NaN when
comparing, or a session Yahoo reports with no volume would look like a
revision on every run.

`max_stored_dates` answers "how far does each ticker reach?" out of Parquet's
own row-group statistics rather than by loading any bars, and is what makes an
ingest incremental. A ticker with nothing stored is absent from the mapping
rather than dated in 1970.

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

## Ingesting the universe

`finmgr ingest` is the loop around `fetch_daily`, and its design assumption is
that it will partly fail:

```bash
finmgr ingest                                  # every ticker in the universe
finmgr ingest -t AAPL -t SAN.MC                # just these
finmgr ingest --start 2024-01-01 --dry-run     # fetch and report, write nothing
finmgr ingest --full                           # re-request the whole window
```

- **A small delay between calls** — `--pause`, half a second by default.
  yfinance scrapes a rate-limited endpoint, and a hundred requests as fast as
  the socket allows is the quickest way to be throttled.
- **Three attempts with exponential backoff** — `--attempts` and `--backoff`,
  waiting 2s then 4s. A timeout, a 502 or a rate-limit rejection is a statement
  about this second, not about the symbol. Every exception is retried rather
  than a chosen few: the failure modes of a scraper are not enumerable, and
  treating a retryable error as fatal loses a company's history for the day
  while treating a fatal one as retryable costs four seconds.
- **A try/except per ticker**, so one bad symbol cannot end the run. Whatever
  goes wrong — the request, the conversion, the write — becomes that ticker's
  status row and the loop moves on.

Nothing raises for a ticker-level problem. What comes back is one status row
per symbol — `ok`, `empty` (the request worked and the window holds no
sessions, which is normal over a weekend), `current` (the store already runs
past the window, so nothing was requested), `failed` or `skipped` — carrying
the rows fetched, how many of them were new, the window actually asked for,
the attempts spent and the error. They are printed, recorded in the ingestion
manifest (below), and the command exits non-zero if anything failed or was
skipped.

Each ticker is written to the store as soon as it arrives, rather than
collected and written at the end, so a run that dies halfway keeps everything
it had already downloaded. And a run that is failing everywhere stops after
`--abort-after` consecutive failures (ten by default, `0` disables it) and
marks the rest `skipped`: 100 symbols at 3 attempts and 6 seconds of backoff
each is half an hour of pointless waiting when the cause is that the cable is
out. Ctrl-C does the same thing — stop, keep what arrived, print the report.

### Only what is missing, and only once

`ingest` is **incremental** and **idempotent**. Before the loop starts it asks
the store how far each ticker already reaches, and each symbol is then
requested from its own newest stored bar rather than from `history_start` —
less five days of deliberate overlap (`--refetch-days`), because the most
recent bar may have been fetched with the market still open and Yahoo revises
volume after the close. `--start` remains a floor, and a ticker with nothing
stored still gets the full backfill.

The overlap is free because the write path merges rather than appends: what
comes back goes through `merge_bars`, keyed on `(ticker, date)`, so a session
already on disk is overwritten in place. Re-running therefore converges
instead of growing, and the status row separates **bars fetched** from bars
that were actually **new**:

```
$ finmgr ingest -t AAPL -t SAN.MC --start 2026-06-01
│ AAPL   │ ok │ 82 │ 82 │ 0 │ 2026-06-01 │ ... │     ← first run, nothing stored
│ SAN.MC │ ok │ 85 │ 85 │ 0 │ 2026-06-01 │ ... │
2 of 2 tickers ingested (ok=2); 167 bars fetched, 167 new, 0 updated in 1.2s

$ finmgr ingest -t AAPL -t SAN.MC --start 2026-06-01
│ AAPL   │ ok │  5 │  0 │ 0 │ 2026-09-20 │ ... │     ← only the overlap is asked for
│ SAN.MC │ ok │  5 │  0 │ 0 │ 2026-09-20 │ ... │
2 of 2 tickers ingested (ok=2); 10 bars fetched, 0 new, 0 updated in 1.3s
```

That second line is day 12's acceptance criterion: **zero new rows the second
time**, with both Parquet files byte-for-byte what they were — an unchanged
partition is not rewritten at all. `--full` turns the narrowing off and
re-requests the window whole, which is what the day 13 backfill and any repair
of a suspect partition want; it stays idempotent, because idempotence comes
from the merge and not from the shorter window. A ticker whose store already
runs past the requested `--end` is not asked at all and reports `current`.

`tests/test_incremental.py` pins all of it offline, the criterion itself three
ways: the report says zero, `read_bars()` returns a frame identical to the one
before, and no partition's mtime moved.

### The full backfill, and what it left behind

Fifteen years for all 100 tickers is one command, and what it produced is
another:

```bash
finmgr ingest --full --all      # every ticker, from history_start (2010-01-01)
finmgr coverage                 # first date, last date and rows per ticker
```

`--full` asks for the whole window whether or not anything is stored, so the
backfill doubles as a repair; because the write merges on `(ticker, date)`, a
second backfill changes nothing. The run on 2026-09-28 took 85 seconds for
100 requests of about 4,200 sessions each, half a second apart: 100 of 100
ok, 415,876 bars. A ticker that fails after its three attempts is simply
re-run with `-t`.

`finmgr coverage` reads the store and nothing else. It lists the universe in
file order, one row per ticker, so a symbol that never made it onto disk shows
up as a row of dashes with `0` rows instead of quietly dropping out of the
table; the command then exits non-zero. `-t` narrows it to given symbols.
Tickers that start later than 2010 do so because that is when they listed —
Meta in 2012, AbbVie in 2013, Aena in 2015, DSM-Firmenich in 2023 — not
because the download fell short.

### The ingestion manifest

Every `finmgr ingest` that writes (anything but `--dry-run`) appends its status
rows to `data/meta/ingest_runs.parquet`: one row per ticker per run, never
rewritten, so "why is Iberdrola stale?" is answered from a file instead of from
memory. Each row carries the `run_id` from the run log, the `status`,
`rows_fetched` and `rows_written` (sessions the run actually changed on disk:
`added + updated`), `min_date` and `max_date` of the bars fetched, the
`asked_from` date and the `stored_max` the store held beforehand, the attempts,
and an `error` message for `failed` and `skipped` rows. The run-level facts —
window, start and finish times, whether it was incremental or stopped early —
are repeated on each row, so filtering on `ticker` gives that symbol's whole
ingestion history without a join.

```bash
finmgr ingest --report          # the last recorded run, from the manifest alone
```

`--report` downloads nothing and never opens the bar store. It prints the same
tables and summary line the run printed, headed by its run id and window, and
exits non-zero if that run was incomplete — so it doubles as a check of last
night's ingest. With nothing recorded yet it says so and exits 1. For anything
deeper, the file is plain Parquet:

```python
from finmgr.data.manifest import read_manifest
runs = read_manifest()
runs[runs.ticker == "IBE.MC"].tail()     # Iberdrola, run by run
```

### Pulling the cable

Day 11's acceptance criterion is that pulling the network cable mid-run still
exits cleanly with a report of what succeeded, and
`tools/pull_the_cable.py` does exactly that, for real, against live Yahoo:

```bash
python tools/pull_the_cable.py --alive 2
```

It ingests into a throwaway directory under `/tmp` and, after the second
ticker, moves itself into an empty network namespace and shuts every open
socket down — no interface, no route, no resolver, and the pooled connection
broken, which is what the kernel and the socket see when the cable comes out.
Unprivileged, and confined to that one process. The run then ends like this:

```
>>> network severed after MSFT: empty netns, 3 open socket(s) cut <<<
  AAPL     ok        6  2024-11-01  2024-11-08   1
  MSFT     ok        6  2024-11-01  2024-11-08   1
  SAN.MC   failed    3  ConnectionError: Failed to perform, curl: (7) Failed to
                        connect to query2.finance.yahoo.com:443
  AIR.PA   skipped   0  3 consecutive failures
2 of 8 tickers ingested (failed=3, ok=2, skipped=3); 12 bars in 12.0s
  — stopped early: 3 consecutive failures
```

Exit code 1, no traceback, and AAPL's and MSFT's bars are on disk. The same
failure is pinned offline in `tests/test_ingest.py`, so it is checked by
`make test` on a machine with no network at all.

### Adjusted close

The store keeps raw closes and, in `data/actions/` (day 15), the splits and
dividends as separate facts. `finmgr.data.adjust` combines them into a
back-adjusted close: every close before a dividend's ex-date is scaled by
`1 - dividend / previous close`, and the newest session stays at its raw close.
Yahoo's raw bars already have splits applied, so a split is only applied where
the bars still show it.

```bash
finmgr adjusted -t SAN.MC                                     # from the store, offline
finmgr adjusted --compare --report docs/adjusted_divergences.md   # against Yahoo
```

`--compare` checks every ticker against Yahoo's `auto_adjust=True` close and
lists each one that diverges by more than 0.5% on any session. It splits each
gap into three parts: raw closes, dividends, and Yahoo's own method. It also
writes out the cause, and exits non-zero if any divergence has none. The last
run is in [`docs/adjusted_divergences.md`](docs/adjusted_divergences.md). US
and Asian listings agree to about one part in a million. The divergences come
from three causes:

- a stale newest bar stored mid-session;
- Yahoo's pence error on the London listings;
- Yahoo applying UBS's and Equinor's USD-declared dividends against a CHF or
  NOK price.

### Exchange calendars

A missing bar on 2024-03-29 for SAN.MC is not a gap: that was Good Friday and
Madrid was shut. `finmgr.data.calendars` checks each ticker against its venue's
calendar from [`exchange_calendars`](https://github.com/gerrymanoim/exchange_calendars).
The venue is the universe's `exchange` column, already a MIC such as XNYS,
XNAS, XMAD, XPAR, XETR or XAMS. The check then lists the sessions the store
genuinely lacks, with holidays excluded.

```bash
finmgr sessions --year 2024              # tickers with gaps, off-calendar bars or closures
finmgr sessions -y 2024 -t SAN.MC --all  # expected vs stored for one ticker
```

A ticker is checked from its first stored bar (or 1 January) to 31 December
(or yesterday). So a mid-year listing is not penalised for the months before
it existed. Bars dated on a day the venue was closed are listed as
`off calendar`. The command exits non-zero if either list is non-empty.
Closures announced at short notice, which the library does not carry, go in
`UNSCHEDULED_CLOSURES` with a reason. One example is the typhoon that shut
Taipei on 2024-10-31. These closures are shown as excused rather than dropped.
With that one exclusion, all 100 tickers are complete for 2024. Day 19 added
five more closures there, each confirmed against the venue: three days on
which TWSE did not trade, and Hong Kong's two weather closures of September
2023. The remaining disagreements in older years are triaged by `finmgr check`
(below).

### Data quality checks

`finmgr check` runs eight checks over every stored bar and prints a
severity-tagged table:

| check | severity | flags |
|---|---|---|
| `bad_price` | error | open, high, low or close is NaN, zero or negative |
| `high_below_low` | error | the day's high is under its low |
| `duplicate_date` | error | the same session is stored twice |
| `unexplained_move` | error | close-to-close move over 40% with no split on the day to explain it |
| `close_out_of_range` | warning | close above the high or below the low |
| `flat_price` | warning | close unchanged five sessions running |
| `missing_session` | warning | the venue was open and the store has no bar (every year, via `calendars.session_gaps`) |
| `off_calendar` | warning | a bar on a day the venue was closed |

```bash
finmgr check                  # untriaged flags in full, triaged ones counted
finmgr check --all            # every flag with its triage verdict and reason
finmgr check -t EQNR.OL --all # one ticker
```

A flag is not a verdict. Each finding is looked at, and the decision is written
to `config/quality_triage.csv`. A rule there names a check, a ticker or a whole
venue (MIC), and a date range. Its verdict is one of three:

- `genuine`: the row is right.
- `bad-data`: the row is wrong. `finmgr quarantine` hides it (below).
- `source-gap`: Yahoo has no bar, and there is nothing local to fix.

Triaged flags are still reported but no longer fail the run. The command exits
non-zero while any flag is untriaged, so a new bad print fails the evening it
arrives. A rule that matches nothing is logged, so the register does not go
stale.

The first run over all 100 tickers raised 359 flags. Seven of them were real
closures and went into `UNSCHEDULED_CLOSURES`. The other 352 are triaged:

- **182 off-calendar bars, `bad-data`.** Yahoo serves placeholder bars on days
  the venue was shut: volume 0, with every price equal to the previous close.
  Examples are Tokyo through most of 2017–18, Xetra on 2017-10-03, and Paris
  and Madrid on Christmas Day 2019. One is worse: NOVO-B.CO on 2010-05-14 is
  stored at double the price either side of it.
- **10 range breaches, `bad-data`.** Seven EQNR.OL closes sit under the day's
  low on the session before an ex-dividend date, short by about the dividend.
  Yahoo removes the payout one day early, which shows up as a fake drop. The
  other three are truncated London bars on 2024-05-01.
- **1 impossible bar, `bad-data`.** 0700.HK on 2010-01-15 has a high below
  its low.
- **19 range breaches, `genuine`.** Small breaches (up to 1.7%) on venues with
  a closing auction. Each close is consistent with the sessions around it.
- **2 big moves, `genuine`.** NFLX rose 42% on its Q4 2012 results. NOVO-B.CO's
  -49.6% on 2010-05-17 is measured from the bad bar the session before.
- **137 missing sessions, `source-gap`.** Each date was researched against the
  venue: Euronext's half days, Xetra on German Unity Day 2011–13, London on
  2012-05-28 and others. The register says which dates were confirmed open and
  which were left `UNCONFIRMED`, meaning no closure was found but there is no
  proof the venue traded either.

### Quarantine, never delete

A `bad-data` row stays in its partition. `finmgr quarantine` runs the checks
and sets that row's `quarantined` column in the bar store. It also writes a
record to `data/quarantine/ticker=<T>/part.parquet` with the row's values, the
check that caught it, the flag's detail and the register's reason.
`read_bars()` leaves quarantined rows out by default, so nothing downstream
computes from them. `read_bars(..., include_quarantined=True)` returns them.

```bash
finmgr quarantine             # sync the flags with the register; list what is quarantined
finmgr quarantine -t EQNR.OL  # one ticker
```

The quarantine follows the register. Each run quarantines rows newly called
`bad-data` and releases rows that no longer are: the rule was removed, or Yahoo
sent a bar that no longer trips the check. A released record is kept, stamped
`released_at`. Running again with nothing changed writes nothing. Untriaged
flags are not quarantined. They already fail `finmgr check`, and hiding them
would hide the row someone still has to look at.

Some code has to see every row:

- `finmgr check` judges what is stored. A quarantined row still raises its flag
  and still counts against its triage rule, and it does not show up as a
  missing session.
- `merge_bars` and the `close_eur` refresh rewrite whole partitions. If they
  read without `include_quarantined=True`, the quarantined rows would be lost
  on the rewrite. A merge also keeps the flag on a session Yahoo re-sends.

The first run quarantined 193 rows across 48 tickers, from 194 records (one
bar is caught by two checks).

### Point in time: every read takes `as_of`

Every read function in the data layer (`read_bars`, `read_actions`, `read_fx`,
`read_adjusted`, `read_quarantine`) takes a required `as_of` and never returns
a row dated after it. There is no default: a forgotten bound is how look-ahead
gets in, so "everything up to today" has to be asked for by name, as
`as_of=LATEST`. The date bound goes into the query. Then `finmgr.data.pit.guard`
checks the result again and raises `LookAheadError` if a later row got
through; it never trims. An `end` or `start` after `as_of` also raises.

`read_adjusted` builds its split and dividend factors only from bars and
actions dated on or before `as_of`. A dividend that goes ex next month cannot
rescale this month's closes. `finmgr adjusted --as-of DATE` prints the series
as it was known on that date.

`tests/test_pit.py` is the leak test. It puts rows after `as_of` in every store
and tries to read them back: asking for them, calling without `as_of`, and
breaking the query's own date bound on purpose. Every attempt has to fail. It
also fails if a new `read_*` function appears in `finmgr.data` without a
required `as_of`. Run it with `make leak`. The `leak-test` pre-commit hook
runs it on any commit that touches `finmgr/data/`.

### One currency: `close_eur`

A US stock and a Spanish one cannot be compared until both are in the same
currency, and here that currency is EUR. `finmgr.data.fx` ingests a daily EUR
cross from Yahoo (`EURUSD=X` and so on) into `data/fx/pair=EURUSD/part.parquet`.
It fetches the four the roadmap names (EURUSD, EURGBP, EURCHF and EURSEK) plus
every other listing currency in the universe: DKK, NOK, JPY, HKD, TWD, CAD and
AUD. It then writes `close_eur = close × quote scale / rate` into every bar
partition.

```bash
finmgr fx                     # fetch every cross, then recompute close_eur everywhere
finmgr fx --no-fetch          # recompute close_eur from the rates already stored
finmgr fx --compare AAPL --start 2025-01-01 --end 2025-12-31 --plot logs/aapl.png
```

Run `finmgr fx` after `finmgr ingest`. A bar the ingest adds stays NaN in
`close_eur` until then. A session the ingest gets back unchanged keeps its EUR
close.

Three rules decide the number:

- **London is quoted in pence.** Yahoo gives AZN.L at about 12,700 GBp, which
  is £127. `QUOTE_SCALE` maps XLON to 0.01 before the division by EURGBP.
- **Same-day rate, else the one before.** A session with no same-day rate
  (Tokyo trades on some days Yahoo has no EURJPY row) takes the newest earlier
  rate, at most `MAX_RATE_AGE` (5) days old. A later rate is never used.
  Sessions after the newest stored rate stay NaN, so an evening when the rates
  failed to arrive cannot quietly price today at yesterday's FX.
- **EUR listings need no rate.** Their `close_eur` is their `close`.

`--compare` reads the rates from the FX store separately from `close_eur` and
checks the identity `(1 + r_EUR) = (1 + r_local) / (1 + r_FX)`. It exits
non-zero if the gap between the two series has the wrong sign, or is off by
more than a basis point. For AAPL over 2025, the euro rose 13.5% against the
dollar, so a +11.5% year in USD was −1.8% in EUR. Over June 2021 to September
2022 the euro fell 19.7%, and +11.2% in USD became +38.4% in EUR. Yahoo's
daily FX close is a London-evening snapshot rather than each exchange's
closing auction, so `close_eur` is correct to the day but not to the minute.

## Command line

Installing the package puts a `finmgr` command on the path. The six
subcommands are the daily pipeline, in the order they run:

```
finmgr --help

  ingest     Download daily bars for the universe into the data store.
  check      Run the data quality checks over the stored bars, severity-tagged.
  features   Build the feature panel from the stored bars.
  train      Train a model walk-forward on the feature panel.
  rank       Score and rank the universe as of a given date.
  report     Render the daily HTML report.
```

`ingest` is real from day 11 — see "Ingesting the universe" above, and
`finmgr ingest --help` for its options. `check` is real from day 19 (see "Data
quality checks" above), and `quarantine` (day 20) sits right after it. The other four are stubs: each prints
`not implemented yet` along with the roadmap day that fills it in, and exits
non-zero so nothing mistakes an unwritten stage for a successful one.
`finmgr coverage` sits beside the pipeline rather than in it: it reports what
the store holds per ticker (see "The full backfill" above). Beside it,
`finmgr actions` ingests splits and dividends (day 15), and `finmgr adjusted`
builds and checks the adjusted close (day 16, see "Adjusted close" above).
`finmgr sessions` (day 17) lists the sessions missing against each venue's
exchange calendar (see "Exchange calendars" above). `finmgr fx` (day 18)
ingests the EUR crosses and writes `close_eur` (see "One currency" above).

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
