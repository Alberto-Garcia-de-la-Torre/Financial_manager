from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

ticker = yf.Ticker("AIR.PA")

# initialize parameters
start_date = datetime(2025, 11, 13)
end_date = datetime(2025, 12, 12)

# Request hourly data
data = ticker.history(
    start = start_date,
    end = end_date,
    interval="60m"    # hourly candles
)

print(data)


# display
plt.figure(figsize = (20,10))
# plt.title('Opening Prices from {} to {}'.format(data[0]["Datetime"],
#                                                 data[-1]["Datetime"]))
plt.plot(data['Open'])
plt.show()

print(len(data['Open']))

for value in data.index:
    print(value)