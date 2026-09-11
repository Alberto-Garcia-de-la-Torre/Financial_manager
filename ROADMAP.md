# Sixty Sessions — Financial_manager roadmap

From two yfinance scripts to a system that ingests 100 companies every evening, scores each one, and proves whether the edge is real.

**Day 1 = 11 Sep 2026 · Day 60 = 9 Nov 2026.** One step per day, each sized for a single working session.

Interactive tracker: see the artifact link in the project notes.

## Read this before day 1

The engineering below is entirely achievable in sixty sessions. Finding a signal that survives costs is not guaranteed, and the plan is built to tell you that honestly rather than hide it. Days 38-39 establish the baselines — buy-and-hold, equal-weight, random picks, plain momentum — that everything afterwards has to beat. If the model loses to a random portfolio, the project has still succeeded: you will know, with numbers.

Three biases are baked in and worth naming now. Picking today's 100 companies and backtesting fifteen years is **survivorship bias** — the losers are missing. Yahoo's fundamentals are **not point-in-time**, so they are deliberately out of scope for v1; day 21 builds the guard that keeps future data out of past decisions. And yfinance is an **unofficial, rate-limited scraper** — day 11 assumes it will fail and plans around it.

This is a research tool for your own decisions, not investment advice.


---

## Phase 1 — Foundations

**Days 1–6 · 11 Sep – 16 Sep**

Six sessions turning a folder of experiments into a package you can build on for two months without fighting it.

### Day 01 — Clear the decks

*11 Sep*

Delete `Other/ajs_lexer.py`, `Other/data_lexer.py` and `.ipynb_checkpoints/` — that's PLY compiler-course code with nothing to do with finance. Untrack `__pycache__/`, write a real `.gitignore` (`__pycache__/`, `*.pyc`, `.venv/`, `data/`, `.env`), and move `yahoo/geeks.py` and `yahoo/chat.py` into `scratch/` as reference.

**Done when:** `git status` is clean and nothing in the repo is unrelated to the project.

### Day 02 — Package skeleton and environment

*12 Sep*

Create `.venv`. Write `pyproject.toml` declaring a `finmgr` package with pinned versions of yfinance, pandas, numpy, pyarrow, duckdb, scikit-learn, pydantic, pydantic-settings, typer, rich, matplotlib and pytest. Add `finmgr/{data,features,models,backtest,report}/__init__.py`, then `pip install -e .`

**Done when:** `python -c "import finmgr"` works from any directory on the machine.

### Day 03 — Config and CLI skeleton

*13 Sep*

A `config/settings.yaml` holding data directory, universe path, base currency (EUR), timezone (Europe/Madrid) and history start date, loaded through a pydantic `Settings` model. Build a Typer app with six stub subcommands: `ingest`, `check`, `features`, `train`, `rank`, `report`.

**Done when:** `finmgr --help` lists all six and each stub prints a clear "not implemented yet".

### Day 04 — Logging and run metadata

*14 Sep*

Rich console logging plus a rotating file log at `logs/`. Mint a `run_id` per invocation (timestamp + short git SHA) and stamp it on the first and last line of every run, along with the config actually used.

**Done when:** Two runs back to back are unambiguously distinguishable in the log file.

### Day 05 — Tests and lint

*15 Sep*

Configure pytest and ruff in `pyproject.toml`. Add a `Makefile` with `test`, `lint` and `fmt` targets. Write the first real test: settings load from YAML, defaults apply, and the data directory gets created on demand.

**Done when:** `make test` and `make lint` are both green from a cold clone.

### Day 06 — Commit discipline

*16 Sep*

Install pre-commit with ruff, ruff-format, and a hook that refuses any commit touching `data/` — market data does not belong in git and a 15-year panel will ruin the repo. Rewrite the README's opening with the project's actual goal and the daily workflow.

**Done when:** A deliberate attempt to commit a Parquet file is rejected by the hook.


---

## Phase 2 — The universe and the raw tape

**Days 7–16 · 17 Sep – 26 Sep**

Ten sessions ending with fifteen years of daily bars for exactly 100 companies, re-runnable at will and honest about what failed.

### Day 07 — Define the 100

*17 Sep*

Write `config/universe.csv` with columns `ticker,name,exchange,currency,sector,country`. A defensible mix: ~40 US large caps, ~40 European (IBEX 35, CAC 40, DAX, AEX) and ~20 elsewhere. Spread across at least eight GICS sectors so cross-sectional ranking has something to compare.

**Done when:** Exactly 100 rows, no duplicate tickers, every sector represented more than once.

### Day 08 — Validate every ticker

*18 Sep*

A script calling `yf.Ticker(t).history(period="5d")` for all 100, reporting which return empty frames. Yahoo suffixes trip people up: `AIR.PA`, `SAN.MC`, `SAP.DE`, `ASML.AS`. Replace anything dead and re-run.

**Done when:** 100 of 100 return recent bars, and the failure list is empty.

### Day 09 — Storage layer

*19 Sep*

`finmgr/data/store.py`: Parquet at `data/bars/daily/ticker=<T>/part.parquet` with DuckDB querying the dataset. Fix the schema now — `ticker` (str), `date` (date), `open/high/low/close` (float64), `volume` (float64) — and expose `read_bars()` and `write_bars()`.

**Done when:** A synthetic frame survives a write/read round trip with dtypes identical.

### Day 10 — Single-ticker downloader

*20 Sep*

`fetch_daily(ticker, start, end)` returning that exact schema: lowercase columns, exchange-local dates, and `auto_adjust=False` so you keep raw prices — you will build your own adjustment on day 16 and want the unmodified source.

**Done when:** A unit test against a saved fixture passes without touching the network.

### Day 11 — Batch ingestion that survives failure

*21 Sep*

Loop the universe with a small delay between calls, three retries with exponential backoff, and a per-ticker try/except so one bad symbol can't kill the run. Collect a status row per ticker instead of raising.

**Done when:** Pulling the network cable mid-run still exits cleanly with a report of what succeeded.

### Day 12 — Incremental and idempotent updates

*22 Sep*

Read the max stored date per ticker, request only from there forward, and de-duplicate on `(ticker, date)` when writing so a re-run overwrites rather than appends.

**Done when:** Running `finmgr ingest` twice in a row writes zero new rows the second time.

### Day 13 — The full backfill

*23 Sep*

Pull fifteen years, or the maximum available, for all 100. Budget for this to take a while and to partially fail — that's what yesterday's retry logic is for. Roughly 3,800 sessions × 100 tickers.

**Done when:** A coverage table prints first date, last date and row count for every ticker.

### Day 14 — Ingestion manifest

*24 Sep*

`data/meta/ingest_runs.parquet`: one row per ticker per run with run_id, rows written, min and max date, status and error message. This is how you'll answer "why is Iberdrola stale?" in eight weeks without guessing.

**Done when:** `finmgr ingest --report` summarises the last run from the manifest alone.

### Day 15 — Corporate actions

*25 Sep*

Ingest splits and dividends into `data/actions/`, keyed by ticker and date. Store them separately from bars — they're a different kind of fact and you'll want them unmodified.

**Done when:** You can print Apple's 4:1 split on 2020-08-31 out of your own store.

### Day 16 — Build your own adjusted series

*26 Sep*

Compute back-adjusted close from raw close plus splits plus dividends. Then compare it against yfinance's `auto_adjust=True` output and list every ticker where the two diverge by more than 0.5%.

**Done when:** You can explain the cause of every divergence you found, not just observe it.


---

## Phase 3 — Making the data trustworthy

**Days 17–22 · 27 Sep – 2 Oct**

Six sessions on the unglamorous work that decides whether everything downstream is real. Day 21 in particular is the most valuable session in the plan.

### Day 17 — Exchange calendars

*27 Sep*

Add `exchange_calendars` and map each ticker to its venue — XNYS, XMAD, XPAR, XETR, XAMS. Compute expected sessions per ticker per year so a "missing day" can be distinguished from a public holiday in Madrid.

**Done when:** You can list genuinely missing sessions per ticker for 2024, holidays excluded.

### Day 18 — One currency for comparison

*28 Sep*

Ingest EURUSD, EURGBP, EURCHF and EURSEK daily, and add a `close_eur` column to the bar store. Comparing a US and a Spanish stock without this is comparing two different things.

**Done when:** A US stock's EUR series visibly diverges from its USD series in the direction FX moved.

### Day 19 — Data quality checks

*29 Sep*

`finmgr check` flags: zero, negative or NaN prices; `high < low`; close outside the day's range; duplicate dates; single-day moves over 40% not explained by a split; a price unchanged five sessions running; and gaps against the exchange calendar. Output a severity-tagged table.

**Done when:** The command runs clean over all 100 and you've triaged every flag it raised.

### Day 20 — Quarantine, never delete

*30 Sep*

Flagged rows are written to `data/quarantine/` with their reason and excluded from reads via a flag column. Silently dropping bad data is how you end up debugging a model that was never the problem.

**Done when:** A test proves a quarantined row is absent from `read_bars()` but still present on disk.

### Day 21 — Point-in-time discipline

*1 Oct*

Every read function takes an `as_of` date and refuses to return any row dated after it. Then write the test that tries to leak a future row and asserts the call fails. Look-ahead bias is the single most common reason a backtest looks brilliant and live trading doesn't.

**Done when:** The leak test passes — and you re-run it after every change to the data layer.

### Day 22 — Benchmarks and the risk-free rate

*2 Oct*

Ingest SPY, a STOXX 600 or MSCI World ETF, an IBEX proxy, and a short-rate series for Sharpe denominators. Without a benchmark you have no way to say whether a 12% year was skill or just the market.

**Done when:** You can plot the cumulative return of all 100 against the benchmark on one chart.


---

## Phase 4 — Features

**Days 23–32 · 3 Oct – 12 Oct**

Ten sessions building the panel the model reads. Day 29 is the one that turns "is this stock good?" into the answerable "is it better than the other 99 today?"

### Day 23 — Returns

*3 Oct*

A returns module producing log and simple returns over 1, 5, 21, 63, 126 and 252 sessions, all strictly backward-looking. Every feature from here on is built on this, so get the alignment right once.

**Done when:** A test asserts `ret_5d` at date t uses no observation later than t.

### Day 24 — Momentum

*4 Oct*

12-1 momentum (the 252-session return excluding the most recent 21), 6-1 momentum, and 5-day short-term reversal. The 12-1 exclusion isn't arbitrary — the most recent month tends to reverse, and including it dilutes the effect.

**Done when:** 12-1 momentum for a known trending stock matches a hand calculation in a notebook.

### Day 25 — Volatility and risk

*5 Oct*

Realized volatility annualized over 21, 63 and 252 sessions; ATR(14); downside deviation; and rolling 252-session max drawdown per ticker.

**Done when:** Annualized vol for a large cap lands in a plausible 15–35% band, not 3% or 300%.

### Day 26 — Trend

*6 Oct*

SMA and EMA at 20, 50 and 200 sessions, percentage distance from each, the slope of each, and golden/death cross flags.

**Done when:** A chart of one ticker's price with its three moving averages looks visually correct.

### Day 27 — Oscillators, written by hand

*7 Oct*

RSI(14), MACD(12,26,9), Stochastic(14,3) and Bollinger %B(20,2) — implemented yourself rather than pulled from a library, with unit tests against a small hand-computed fixture. You need to know exactly what these mean when the model leans on them.

**Done when:** Your RSI matches a reference implementation to four decimal places.

### Day 28 — Volume and liquidity

*8 Oct*

21-session average euro volume, volume z-score, OBV, and a tradability flag — say, €5M average daily euro volume. A signal you can't fill at a sane price isn't a signal.

**Done when:** The tradability filter correctly excludes the thinnest names in your universe.

### Day 29 — Cross-sectional normalization

*9 Oct*

For every date, convert each raw feature into a z-score and a percentile rank across the 100. This is the hinge of the whole project: an RSI of 30 means nothing absolute, but "lowest RSI of the 100 today" is a decision you can act on.

**Done when:** Each date's ranks are uniform on [0,1] by construction, verified with a test.

### Day 30 — Market and sector relatives

*10 Oct*

252-session rolling beta to the benchmark, residual (market-neutral) momentum, and sector-demeaned versions of your strongest features using the sector column from day 7.

**Done when:** Sector-demeaned momentum no longer simply ranks whichever sector had a good year.

### Day 31 — The feature panel

*11 Oct*

Assemble everything into one panel at `data/features/panel.parquet` indexed by `(date, ticker)`, built incrementally. Alongside it, `features/registry.yaml` documenting every column: name, formula, window, expected direction.

**Done when:** The panel rebuilds from scratch in one command and the registry matches its columns exactly.

### Day 32 — Feature hygiene

*12 Oct*

NaN coverage per feature per year, a pairwise correlation matrix, and removal of any feature above 0.95 correlation with a simpler one. Forty correlated features are worse than twelve independent ones.

**Done when:** You've cut the count to a set where you can defend each feature individually.


---

## Phase 5 — Labels, backtest and the bar to beat

**Days 33–40 · 13 Oct – 20 Oct**

Eight sessions building the machinery that decides whether anything works — and, at day 38, the honest baselines the rest of the project has to clear.

### Day 33 — Labels

*13 Oct*

Forward 5- and 21-session returns, a cross-sectional excess version (return minus that day's universe mean) and a binary top-quintile label. Explicit `shift(-N)`, with a test for alignment.

**Done when:** A test shows the label at t equals the realized return from t+1 through t+N.

### Day 34 — Purged walk-forward splits

*14 Oct*

A splitter yielding (train, test) windows moving forward in time, with an embargo gap of at least the label horizon so no training row's label overlaps the test window. Never use random k-fold on time series — it leaks the future into the past and every metric becomes fiction.

**Done when:** A test asserts zero overlap between any training label window and its test window.

### Day 35 — Backtest engine v1

*15 Oct*

Input: one score per ticker per day. Output: pick top-K, equal weight, hold N sessions, roll forward — producing a daily positions frame and an equity curve in EUR.

**Done when:** Feeding it a constant score reproduces plain equal-weight buy-and-hold exactly.

### Day 36 — Costs and frictions

*16 Oct*

Commission in basis points, spread and slippage in basis points optionally scaled by inverse liquidity, and the rule that you trade at the *next* session's open, never today's close. Make every number configurable.

**Done when:** Raising costs to 50bps visibly and plausibly degrades the equity curve.

### Day 37 — Metrics

*17 Oct*

CAGR, annualized volatility, Sharpe against your risk-free series, Sortino, max drawdown and its duration, hit rate, average win versus average loss, turnover, and average holding period.

**Done when:** Metrics for a benchmark buy-and-hold match a published reference within rounding.

### Day 38 — The baselines to beat

*18 Oct*

Run the backtest on: benchmark buy-and-hold; equal-weight all 100; 200 random-pick portfolios with different seeds; and plain 12-1 momentum. Write every number into `docs/baselines.md`.

**Done when:** You have a Sharpe figure that any model must beat to be worth keeping. This is the bar.

### Day 39 — How lucky was that?

*19 Oct*

Block bootstrap confidence intervals on Sharpe, plus a running count of how many strategy variants you've tested. Every variant you try inflates the best-looking result; the count is what keeps you honest about it.

**Done when:** You can state your best baseline's Sharpe as an interval, never as a single number.

### Day 40 — Tearsheet

*20 Oct*

One command turning any backtest into an HTML page: equity curve against benchmark, drawdown chart, rolling 12-month Sharpe, monthly return heatmap and the metrics table.

**Done when:** `finmgr backtest --strategy momentum --report` opens a page you'd show someone.


---

## Phase 6 — Models

**Days 41–48 · 21 Oct – 28 Oct**

Eight sessions adding learned models on top of the harness — with every one measured against day 38's baselines on identical splits and identical costs.

### Day 41 — Model interface

*21 Oct*

A small protocol — `fit`, `predict`, `name`, `params` — plus a registry so strategies are swappable by name from the CLI. Doing this before the first model saves rewriting all of them later.

**Done when:** `finmgr train --model ridge` and `--model lgbm` both resolve through the registry.

### Day 42 — Linear baseline

*22 Oct*

Ridge regression on cross-sectional ranks predicting forward 21-session excess return, trained walk-forward. Start linear: it's fast, it's interpretable, and it's surprisingly hard to beat.

**Done when:** It runs end to end and its tearsheet sits alongside the baselines for comparison.

### Day 43 — Gradient boosting

*23 Oct*

Add a LightGBM or XGBoost regressor through the same walk-forward harness on identical features. Same splits, same costs — otherwise the comparison means nothing.

**Done when:** You can put boosting and ridge side by side on one page and the comparison is fair.

### Day 44 — Classification variant

*24 Oct*

Predict P(top quintile) instead of a return, then calibrate the probabilities with isotonic or Platt scaling on a held-out slice. Ranking problems are often easier to learn as classification.

**Done when:** The calibration curve sits close to the diagonal on out-of-sample data.

### Day 45 — Hyperparameter search

*25 Oct*

Optuna over the boosting model, scored by out-of-sample Sharpe averaged across walk-forward folds, with a fixed trial budget. Log every trial, including the bad ones.

**Done when:** You can show the search picked a genuinely good configuration, not the luckiest fold.

### Day 46 — What is the model actually using?

*26 Oct*

Permutation importance and SHAP values on the boosting model. Drop what contributes nothing and retrain. A model whose top features make no economic sense is usually fitting noise.

**Done when:** You can name the top five drivers and give a plausible reason for each.

### Day 47 — Ensemble

*27 Oct*

Average the cross-sectional ranks of ridge, boosting and plain momentum, then check whether the ensemble beats each part on out-of-sample Sharpe *and* turnover.

**Done when:** You've decided with numbers whether the ensemble earns its extra complexity.

### Day 48 — Model versioning

*28 Oct*

Save each trained model with its feature list, training window, hyperparameters, git SHA and metrics to `models/<name>/<timestamp>/`, with `models/registry.json` naming the current production model.

**Done when:** You can load a model trained a week ago and reproduce its scores exactly.


---

## Phase 7 — From scores to a decision

**Days 49–54 · 29 Oct – 3 Nov**

Six sessions turning a column of numbers into an answer: which company today, at what size, and under what circumstances none of them.

### Day 49 — Daily scoring pipeline

*29 Oct*

`finmgr rank --as-of YYYY-MM-DD`: features for that date, through the production model, out to a ranked table of all 100 with score, rank and percentile. The `as_of` guard from day 21 does the heavy lifting here.

**Done when:** Running it with a past date returns exactly what you'd have seen on that date.

### Day 50 — The recommendation

*30 Oct*

Convert scores to a 0–100 conviction number, apply the liquidity filter, break ties deterministically, and surface the top 10 with the three features that pushed each name up. "Why" matters as much as "which" — it's how you'll catch the model going wrong.

**Done when:** The output answers "which one today, and why" on a single screen.

### Day 51 — Position sizing

*31 Oct*

Backtest equal weight against inverse-volatility and volatility-targeted sizing, with a maximum weight per name (say 10%) and per sector (say 30%). Then pick one on the evidence.

**Done when:** The sizing choice is written down in the repo with the numbers that justified it.

### Day 52 — Risk rules

*1 Nov*

A regime filter (move partly to cash when the benchmark is below its 200-session average), a per-position stop, and a portfolio drawdown circuit breaker. Backtest each one in isolation.

**Done when:** You know which rules improved risk-adjusted return and which merely cost you return.

### Day 53 — Paper-trading ledger

*2 Nov*

`data/portfolio/` holding positions, a trade log and daily mark-to-market P&L in EUR, updated by the daily run. Start with a notional €10,000 and no real money anywhere near it.

**Done when:** The ledger reconciles to the cent against the prices in your own store.

### Day 54 — Prediction log

*3 Nov*

Append every day's full ranked table and top pick to `data/predictions/`, plus a job that scores past predictions once their horizon has elapsed. This is the only record that tells you whether live performance matches the backtest.

**Done when:** You can plot realized rank information coefficient of live predictions over time.


---

## Phase 8 — Running it every day

**Days 55–60 · 4 Nov – 9 Nov**

Six sessions making it run without you: one command, on a timer, reporting to your phone, and loud when something breaks.

### Day 55 — One command

*4 Nov*

`finmgr daily` runs ingest → check → features → rank → ledger → report, idempotently, exiting non-zero with a clear message when any stage fails.

**Done when:** Running it twice in the same day is harmless and produces identical output.

### Day 56 — Scheduling

*5 Nov*

A systemd user timer (this is Fedora — `systemd --user` beats cron for logging and dependencies) firing after each market close, handling Europe/Madrid versus US closes, weekends and holidays via the exchange calendars from day 17.

**Done when:** It has run unattended through a full trading day without you touching it.

### Day 57 — The daily report

*6 Nov*

An HTML page with today's top 10 and conviction scores, what changed since yesterday, portfolio state and P&L, risk flags, data quality status and the model's recent hit rate. Archive every day's copy.

**Done when:** The report tells you everything you need without opening a terminal.

### Day 58 — Delivery

*7 Nov*

Push the report to yourself — SMTP email or a Telegram bot — with the top three names in the subject line, plus a separate alert when any pipeline stage fails.

**Done when:** The message arrives on your phone after market close without you doing anything.

### Day 59 — Monitoring and drift

*8 Nov*

Alert on stale data (more than one session old), ingestion failures, score distribution shift against the training period, feature NaN spikes, and rolling 60-day IC turning negative. Add a monthly retraining job.

**Done when:** You deliberately break the ingest and get alerted within one cycle.

### Day 60 — Close the loop

*9 Nov*

Full test suite green. README rewritten with an architecture diagram and an honest limitations section — survivorship bias, no point-in-time fundamentals, assumed costs. A written retrospective of what beat the baselines and what didn't. Three priorities for the next quarter.

**Done when:** Someone else could clone the repo and get `finmgr daily` running from the README alone.
