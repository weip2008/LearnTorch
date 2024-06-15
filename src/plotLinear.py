import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('data/linear_data_with_deviation.csv')

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(data['x'], data['y'], color='blue', label='Data points with noise')
# plt.plot(data['x'], 3 * data['x'] - 7, color='red', label='y = 3x - 7')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of linear data')
plt.legend()
plt.grid(True)
plt.show()
