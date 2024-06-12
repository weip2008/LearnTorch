import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(0)

# Generate 100 random x values
x_values = np.random.rand(100) * 100  # x values between 0 and 100

# Calculate y values based on the equation y = 3x - 7
# Add some random noise to y values
noise = np.random.normal(0, 10, 100)  # Gaussian noise with mean 0 and standard deviation 10
y_values = 3 * x_values - 7 + noise

# Create a DataFrame
data = pd.DataFrame({'x': x_values, 'y': y_values})

# Save to CSV file
data.to_csv('data/linear_data_with_deviation.csv', index=False)

print('Data with deviation has been generated and saved to linear_data_with_deviation.csv')
