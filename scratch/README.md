# scratch

Early throwaway experiments, kept only as reference for how the yfinance API
behaves. Nothing here is imported by the `finmgr` package and none of it is
tested or maintained.

- `geeks.py` — daily `yf.download` of SPY plus a matplotlib plot.
- `chat.py` — hourly candles for `AIR.PA` via `yf.Ticker(...).history(interval="60m")`.
- `html_csv.py` — scraping a Yahoo quote table out of saved HTML with `pandas.read_html`.
