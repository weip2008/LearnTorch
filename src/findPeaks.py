import numpy as np
from scipy.signal import find_peaks

# Example time series data
data = [1, 3, 7, 6, 4, 2, 5, 9, 8, 3, 1, 6]
x = np.array(data)

# Find local maxima (tops)
peaks, _ = find_peaks(x)
print(f"Local maxima indices: {peaks}")

# Find local minima (bottoms) by inverting the data
min_peaks, _ = find_peaks(-x)
print(f"Local minima indices: {min_peaks}")
