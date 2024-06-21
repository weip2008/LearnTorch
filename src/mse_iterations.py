import numpy as np

# Generate some random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Define the learning rate and number of iterations
learning_rate = 0.15
iterations = 100

# Initialize weights and biases
w1, b1 = np.random.randn(), np.random.randn()
w2, b2 = np.random.randn(), np.random.randn()
w3, b3 = np.random.randn(), np.random.randn()

# Store the MSE values for each iteration
mse_y1_values, mse_y2_values, mse_y3_values = [], [], []

# Gradient descent optimization
for i in range(iterations):
    # Compute predictions
    y1_pred = w1 * X + b1
    y2_pred = w2 * X + b2
    y3_pred = w3 * X + b3
    
    # Compute the MSE for each dataset
    mse_y1 = np.mean((y - y1_pred) ** 2)
    mse_y2 = np.mean((y - y2_pred) ** 2)
    mse_y3 = np.mean((y - y3_pred) ** 2)
    
    # Store the MSE values
    mse_y1_values.append(mse_y1)
    mse_y2_values.append(mse_y2)
    mse_y3_values.append(mse_y3)
    
    # Compute the gradients
    gradient_w1 = (2 / len(X)) * np.dot(X.T, (y1_pred - y))
    gradient_b1 = (2 / len(X)) * np.sum(y1_pred - y)
    
    gradient_w2 = (2 / len(X)) * np.dot(X.T, (y2_pred - y))
    gradient_b2 = (2 / len(X)) * np.sum(y2_pred - y)
    
    gradient_w3 = (2 / len(X)) * np.dot(X.T, (y3_pred - y))
    gradient_b3 = (2 / len(X)) * np.sum(y3_pred - y)
    
    # Update weights and biases
    w1 -= learning_rate * gradient_w1
    b1 -= learning_rate * gradient_b1
    
    w2 -= learning_rate * gradient_w2
    b2 -= learning_rate * gradient_b2
    
    w3 -= learning_rate * gradient_w3
    b3 -= learning_rate * gradient_b3

# Print the MSE values for each iteration
for i in range(iterations):
    print(f"Iteration {i+1}: MSE of y1 = {mse_y1_values[i]:.4f}, MSE of y2 = {mse_y2_values[i]:.4f}, MSE of y3 = {mse_y3_values[i]:.4f}")

print(f'[{w1},{b1}]')
print(f'[{w2},{b2}]')
print(f'[{w3},{b3}]')