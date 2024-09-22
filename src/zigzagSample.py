import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Create chart
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['price'], label="Price", marker="o", color="b")

# Mark pivots
pivot_indices = np.where(pivots != 0)[0]
plt.scatter(df['date'][pivot_indices], df['price'][pivot_indices], color='r', label="Pivots", zorder=5)

# Add labels and legend
plt.title('Price Series with ZigZag Pivots')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

# Show chart
plt.tight_layout()
plt.show()
