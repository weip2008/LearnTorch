import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
import os

ticker_name = "AAPL"
#ticker_name = "SPY"
#symbol = "MES=F"
# start = "2024-04-12"
# end = "2024-04-15"
# start = "2024-04-15"
# end = "2024-04-21"
# start = "2024-04-22"
# end = "2024-04-28"
# start = "2024-04-29"
# end = "2024-05-05"
# start = "2024-05-06"
# end = "2024-05-12"
start = "2024-05-20"
end = "2024-05-26"
interval = "1m"

# inverval could be 1m, 1h, 1d, 1wk, 1mo, 3mo
ohlcv = yf.download(tickers=ticker_name, start=start, end=end, interval=interval)

print(ohlcv.head(10))
print(ohlcv.tail(10))
print("Length of origianl dataFrame:", len(ohlcv))

# Define the file path
#file_path = "data/{}_{}.csv".format(ticker_name, dt.datetime.now().strftime("%Y%m%d"))
#file_path = "stockdata/{}_{}_{}_{}.csv".format(ticker_name, start, end, interval)
directory = "stockdata"
file_path = os.path.join(directory, f"{ticker_name}_{start}_{end}_{interval}.csv")

# Save the data to CSV
if not os.path.exists(directory):
    os.makedirs(directory)

ohlcv.to_csv(file_path)

print("Data saved to:", file_path)

