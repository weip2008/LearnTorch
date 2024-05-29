import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x / 3)

fig, ax1 = plt.subplots()

# Plotting the first dataset
ax1.plot(x, y1, 'b-')
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data (sin)', color='b')
ax1.tick_params('y', colors='b')

# Creating a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-')
ax2.set_ylabel('Y2 data (exp)', color='r')
ax2.tick_params('y', colors='r')

# Show the plot
plt.title('Plot with Different Scales')
plt.show()
