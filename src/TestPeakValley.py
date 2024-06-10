import pandas as pd
import matplotlib.pyplot as plt
from zigzag import peak_valley_pivots


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

# Define the single deviation value
deviation = 0.0001

# Calculate pivots for the given deviation
pivots = peak_valley_pivots(df['Close'].values, deviation, -deviation)
df['Pivots'] = pivots

# Print the DataFrame to verify pivots
print(df)

# Plotting to visualize the result
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(df.index, df['Close'], label='Close Price')
ax.scatter(df.index[df['Pivots'] == 1], df['Close'][df['Pivots'] == 1], color='green', label='Peak', marker='^', alpha=1)
ax.scatter(df.index[df['Pivots'] == -1], df['Close'][df['Pivots'] == -1], color='red', label='Valley', marker='v', alpha=1)
ax.set_title(f'Close Price with Peaks and Valleys (Deviation = {deviation})')
ax.set_xlabel('Datetime')
ax.set_ylabel('Close Price')
ax.legend()

plt.show()
