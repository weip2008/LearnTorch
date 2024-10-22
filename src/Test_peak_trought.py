import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Sample data: a sine wave with noise
x = np.linspace(0, 6 * np.pi, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, 1000)

# Find peaks with minimum height of 0.5 and a minimum distance of 20 samples
peaks, peak_properties = find_peaks(y, height=0.5, distance=20)

# Find troughs (peaks in the negative signal)
troughs, trough_properties = find_peaks(-y, height=0.5, distance=20)

# Plotting the results
plt.plot(x, y, label='Signal')
plt.plot(x[peaks], y[peaks], "x", label='Peaks')  # Mark peaks
plt.plot(x[troughs], y[troughs], "o", label='Troughs')  # Mark troughs
plt.legend()
plt.show()
