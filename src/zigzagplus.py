import pandas as pd
import numpy as np
from zigzag import peak_valley_pivots
import matplotlib.pyplot as plt

# Sample data
data = {
    'high': [1, 2, 3, 4, 5, 4.5, 3, 3.5, 4, 3],
    'low': [0.5, 1, 1.5, 2, 2.5, 2, 1.5, 2, 2.5, 1.5],
    'close': [0.8, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 3, 3.5, 2]
}
df = pd.DataFrame(data)

# ZigZag parameters
depth = 12
deviation = 5 / 100.0  # Percentage
backstep = 2

# Calculate the ZigZag indicator
pivots = peak_valley_pivots(df['close'].values, deviation, -deviation)
zigzag = df['close'][pivots != 0]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['close'], label='Close Price')
plt.scatter(zigzag.index, zigzag, color='red', label='ZigZag')
plt.legend()
plt.show()

# Detecting Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), Lower Lows (LL)
zigzag_points = df[pivots != 0]
for i in range(1, len(zigzag_points)):
    current_point = zigzag_points.iloc[i]
    previous_point = zigzag_points.iloc[i-1]
    if current_point['close'] > previous_point['close']:
        if previous_point['close'] > zigzag_points.iloc[i-2]['close']:
            label = "HH"  # Higher High
        else:
            label = "HL"  # Higher Low
    else:
        if previous_point['close'] < zigzag_points.iloc[i-2]['close']:
            label = "LL"  # Lower Low
        else:
            label = "LH"  # Lower High
    print(f"Point: {current_point['close']}, Label: {label}")

# Add alerts (conceptual representation)
alerts = []
for i in range(1, len(zigzag_points)):
    current_point = zigzag_points.iloc[i]
    previous_point = zigzag_points.iloc[i-1]
    if current_point['close'] > previous_point['close'] and previous_point['close'] > zigzag_points.iloc[i-2]['close']:
        alerts.append("New Higher High detected")
    elif current_point['close'] < previous_point['close'] and previous_point['close'] < zigzag_points.iloc[i-2]['close']:
        alerts.append("New Lower Low detected")
    # Add more conditions based on requirements

for alert in alerts:
    print(alert)

# Example usage:
ticker_symbols = [ "MES=F" ]
# Data interval
t_interval="1m"

# Fetch the historical data from the first day it started trading
stock_data = yf.Ticker("MES=F")
#stock_hist = stock_data.history(period="max", auto_adjust=False)
df = stock_data.history(period="max", interval=t_interval, auto_adjust=False)
# Drop the 'Dividends' and 'Stock Splits' columns
if 'Dividends' in df.columns:
    df = df.drop(columns=['Dividends'])
if 'Stock Splits' in df.columns:
    df = df.drop(columns=['Stock Splits'])
if 'Adj Close' in df.columns:
    df = df.drop(columns=['Adj Close'])


print(df)


df['TickSize'] = 0.0082 # Set a constant tick size for example

zz_df = calculate_zigzag(df)
print(zz_df[['High', 'Low', 'Close', 'ZZ']])

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.scatter(zz_df.index, zz_df['ZZ'], label='ZigZag', color='red')
plt.title('Close Price and ZigZag Indicator')
plt.xlabel('Datetime')
plt.ylabel('Price')
plt.legend()
plt.show()
