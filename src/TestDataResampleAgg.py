# 1. Resampling:
# The resample() method in pandas is used to change the frequency of a time-series data.
# It allows you to aggregate data over different time intervals (e.g., from minutes to hours, 
# or from days to months).
# You specify the desired frequency (e.g., ‘5T’ for 5-minute intervals) as an argument to resample().
# The “T” in “5T” stands for minutes. When you use resample('5T'), it means you’re resampling the data into 5-minute intervals. 
# Similarly, you can use other letters like “H” for hours, “D” for days, or “M” for months.
#
# 2. Aggregation:
# The agg() method (short for “aggregate”) is used to apply one or more aggregation functions to the resampled data.
# It computes summary statistics (such as mean, sum, max, min, etc.) for each interval.
# You provide a dictionary where keys are column names, and values are the aggregation functions you want to apply to those columns.

import pandas as pd
import matplotlib.pyplot as plt

# Read the 1-minute data
nq_1min_df = pd.read_csv('stockdata/NQ_1min_sample.csv', parse_dates=['timestamp'], index_col='timestamp')
print(nq_1min_df.head(27))

# Read the 5-minute data
nq_5min_df = pd.read_csv('stockdata/NQ_5min_sample.csv', parse_dates=['timestamp'], index_col='timestamp')
print(nq_5min_df.head(6))

# Resample 1-minute data to 5-minute intervals
# '5T' specifies 5-minute intervals.
# agg() applies the specified aggregation functions ('first', 'max', 'min', 'last', and 'sum') 
# to each column.
nq_5min_resampled = nq_1min_df.resample('5T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
print(nq_5min_resampled.head(6))

# Plot all three dataframes
plt.figure(figsize=(12, 6))

# 1-minute data
plt.plot(nq_1min_df['close'], label='NQ 1-Minute Close', linewidth=2)

# 5-minute data
plt.plot(nq_5min_df['close'], label='NQ 5-Minute Close', linewidth=2)

# 5-minute resampled data
plt.plot(nq_5min_resampled['close'], label='NQ 5-Minute Resampled Close', linestyle='--', linewidth=2)

plt.title('NQ Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()
