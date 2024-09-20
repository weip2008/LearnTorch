import numpy as np
import pandas as pd
import zigzag

# Example time series data
data = {
    'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
    'price': [100, 102, 105, 103, 107, 108, 104, 110, 113, 112]
}
df = pd.DataFrame(data)

# Extract the price series
price_series = df['price'].values

# Apply zigzag peak-valley pivot algorithm with two thresholds: one for upward and one for downward movement
pivots = zigzag.peak_valley_pivots(price_series, 0.02, -0.02)

# Convert pivot indices to a more readable format
pivot_dates = df['date'].values[pivots != 0]

# Print the results
print("Pivot Indices:", pivots)
print("Pivot Dates:", pivot_dates)
print("Pivot Prices:", price_series[pivots != 0])
