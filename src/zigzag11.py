import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

df = pd.read_csv("stockdata/NQ1!_2024-06-08_1M.csv")
print(df)
print(df.shape)

def filter(values, percentage):
    previous = values[0]
    mask = [True]
    for value in values[1:]:
        relative_difference = np.abs(value - previous)/previous
        if relative_difference > percentage:
            previous = value
            mask.append(True)
        else:
            mask.append(False)
    return mask

# Assuming 'df' is your DataFrame
data_y = df['close'].values

# Find peaks(max).
peak_indexes = signal.argrelextrema(data_y, np.greater)[0]

# Find valleys(min).
valley_indexes = signal.argrelextrema(data_y, np.less)[0]

# Merge peaks and valleys data points using pandas.
df_peaks = pd.DataFrame({'date': df.index[peak_indexes], 'zigzag_y': data_y[peak_indexes]})
df_valleys = pd.DataFrame({'date': df.index[valley_indexes], 'zigzag_y': data_y[valley_indexes]})
df_peaks_valleys = pd.concat([df_peaks, df_valleys], axis=0, ignore_index=True, sort=True)

# Sort peak and valley datapoints by date.
df_peaks_valleys = df_peaks_valleys.sort_values(by=['date'])

# Filter the peaks and valleys based on a percentage change threshold.
p = 0.001 # 10%
filter_mask = filter(df_peaks_valleys.zigzag_y, p)
filtered = df_peaks_valleys[filter_mask]

# Plot original line and zigzag trendline.
plt.figure(figsize=(10,10))
plt.plot(df.index.values, data_y, linestyle='dashed', color='black', label="Original line", linewidth=1)
plt.plot(filtered['date'].values, filtered['zigzag_y'].values, color='blue', label="ZigZag")
plt.legend()
plt.show()

# df.rename(columns=lambda x: x.lower(), inplace=True)
# # convert string datetime to datetime type and set 'datetime' as index
# df['datetime'] = pd.to_datetime(df['datetime']) 
# df.set_index('datetime', inplace=True)
# print(df.high.median())
# print(df.describe())

#                open          high           low         close        volume
# count   6899.000000   6899.000000   6899.000000   6899.000000   6899.000000
# mean   18848.989527  18851.710502  18846.221083  18849.051783    427.749819
# std      210.776388    210.478187    211.045352    210.772513    695.632556
# min    18440.000000  18449.250000  18435.750000  18440.250000      1.000000
# 25%    18652.125000  18654.000000  18650.250000  18652.125000     44.000000
# 50%    18768.750000  18770.750000  18766.500000  18768.750000    114.000000
# 75%    19078.250000  19079.750000  19076.500000  19078.250000    577.500000
# max    19150.750000  19155.000000  19147.500000  19151.000000  10828.000000

# print(df.head(10))
# print(df.info())
# print(df.at["2024-06-02 17:03:00", 'high'])

# df['high'].plot()
# df1 = df.resample('5T').agg({'close' : 'mean', 'high' : 'max', 'low' : 'min', 'volume' : 'sum'})
# print(df1.tail())
# print(df1.shape)

# print(df[['symbol','close', 'volume','high', 'low', 'close', 'volume']])
# print(df.columns)
# print(df.iloc[[1,2]])