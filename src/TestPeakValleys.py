import pandas as pd
import numpy as np
from zigzag import peak_valley_pivots
import matplotlib.pyplot as plt

# Sample DataFrame (using the data provided earlier)
data = {
    "Datetime": [
        "2024-06-02 18:00:00-04:00", "2024-06-02 18:01:00-04:00", "2024-06-02 18:02:00-04:00", 
        "2024-06-02 18:03:00-04:00", "2024-06-02 18:04:00-04:00", "2024-06-02 18:05:00-04:00", 
        "2024-06-02 18:06:00-04:00", "2024-06-02 18:07:00-04:00", "2024-06-02 18:08:00-04:00", 
        "2024-06-02 18:09:00-04:00", "2024-06-02 18:10:00-04:00", "2024-06-02 18:11:00-04:00", 
        "2024-06-02 18:12:00-04:00", "2024-06-02 18:13:00-04:00", "2024-06-02 18:14:00-04:00", 
        "2024-06-02 18:15:00-04:00", "2024-06-02 18:16:00-04:00", "2024-06-02 18:17:00-04:00", 
        "2024-06-02 18:18:00-04:00", "2024-06-02 18:19:00-04:00"
    ],
    "Close": [
        5302.75, 5301.50, 5294.25, 5294.00, 5292.75, 5291.00, 5292.00, 5292.50, 
        5292.75, 5292.00, 5293.00, 5292.50, 5293.25, 5294.25, 5294.50, 5297.00, 
        5297.25, 5299.00, 5298.75, 5299.25
    ]
}

df = pd.DataFrame(data)
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)

# Try different deviation values
deviation_values = [0.001, 0.0005, 0.0002, 0.0001]

for deviation in deviation_values:
    pivots = peak_valley_pivots(df['Close'].values, deviation, -deviation)
    df[f'Pivots_{deviation}'] = pivots

# Print the DataFrame to verify pivots
print(df)

# Plotting to visualize each deviation result
fig, ax = plt.subplots(len(deviation_values), 1, figsize=(14, 14), sharex=True)

for i, deviation in enumerate(deviation_values):
    ax[i].plot(df.index, df['Close'], label='Close Price')
    ax[i].scatter(df.index[df[f'Pivots_{deviation}'] == 1], df['Close'][df[f'Pivots_{deviation}'] == 1], color='green', label='Peak', marker='^', alpha=1)
    ax[i].scatter(df.index[df[f'Pivots_{deviation}'] == -1], df['Close'][df[f'Pivots_{deviation}'] == -1], color='red', label='Valley', marker='v', alpha=1)
    ax[i].set_title(f'Close Price with Peaks and Valleys (Deviation = {deviation})')
    ax[i].set_ylabel('Close Price')
    ax[i].legend()

plt.xlabel('Datetime')
plt.show()
