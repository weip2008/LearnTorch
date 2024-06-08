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

# Calculate the ZigZag indicator
pivots = peak_valley_pivots(df['close'], 0.05, -0.05)
zigzag = df['close'][pivots != 0]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['close'], label='Close Price')
plt.scatter(zigzag.index, zigzag, color='red', label='ZigZag')
plt.legend()
plt.show()
