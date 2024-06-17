import numpy as np
import matplotlib.pyplot as plt
import csv

# Generate some data points
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2.5 * x**2 + np.random.normal(size=x.size)

# Define the file name
file_name = 'data.csv'

# Write the data to the CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y'])  # Write header
    for i in range(len(x)):
        writer.writerow([x[i], y[i]])

# Fit a polynomial curve
degree = 2
coefficients = np.polyfit(x, y, degree)
poly = np.poly1d(coefficients)

# Plot the data and the fitted curve
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data points')
plt.plot(x, poly(x), color='red', label='Fitted polynomial curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
