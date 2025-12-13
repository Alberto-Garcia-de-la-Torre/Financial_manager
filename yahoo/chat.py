import yfinance as yf

ticker = yf.Ticker("AIR.PA")

# Request hourly data
data = ticker.history(
    period="7d",      # last 7 days (adjust as needed)
    interval="60m"    # hourly candles
)

print(data)
