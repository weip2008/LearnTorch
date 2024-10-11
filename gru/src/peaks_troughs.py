import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 1: Read data from a CSV file
df = pd.read_csv('data/LULU_max_2024-05-28_2024-05-31_1m.csv')

# Step 2: Assuming the CSV has 'Datetime' and 'Close' columns
price_data = df['Close']
# df['Datetime'] = pd.to_datetime(df['Datetime'])

# Step 3: Adjust parameters for peak detection
prominence_value = 0.8  # Adjust this to change peak prominence
width_value = 1.5     # Adjust this to change peak width
height_value = None   # Set a value or None for no height restriction
distance_value = 10    # Minimum distance between peaks

# Detect peaks with adjusted parameters
peaks, properties = find_peaks(price_data, prominence=prominence_value, width=width_value, height=height_value, distance=distance_value)

# Step 4: Detect troughs (inverted peaks)
troughs, trough_properties = find_peaks(-price_data, prominence=prominence_value, width=width_value, height=height_value, distance=distance_value)

# Combine and sort peaks and troughs by time
extrema_indices = np.sort(np.concatenate((peaks, troughs)))

# Step 5: Plot the data
plt.plot(df['Datetime'], price_data, label="Price", color='blue')
plt.plot(df['Datetime'][peaks], price_data[peaks], "x", label="Peaks", color="green", markersize=10)
plt.plot(df['Datetime'][troughs], price_data[troughs], "o", label="Troughs", color="red", markersize=10)

# Step 6: Draw lines connecting the peaks and troughs
plt.plot(df['Datetime'][extrema_indices], price_data[extrema_indices], label="Connection", color="black", linestyle='--')

# Add labels and legend
plt.xticks(rotation=45)
plt.title("Price Data with Peaks and Troughs from CSV")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

# Show the plot
plt.show()

# Print widths of peaks and troughs (optional)
print("Peak widths:", properties['widths'])
print("Trough widths:", trough_properties['widths'])