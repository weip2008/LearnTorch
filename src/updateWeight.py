import numpy as np
import matplotlib.pyplot as plt

# Create an array of 50 values for x ranging from 0 to 10
np.random.seed(42)
x = np.linspace(0, 10, 50)
# Target output
noise = np.random.randn(50)*0.2
y = 2 * x - 1 + noise # Assuming a linear relationship y = 2x + 1 with some noise

# Initialize weights and bias
w = np.random.randn()
b = np.random.randn()

# Learning rate
lr = 0.01 # Learning Rate

# Number of epochs
epochs = 1000

# Training loop
for _ in range(epochs):
    # Forward pass forward propagation
    z = w * x + b
    # Calculate MSE loss
    loss = np.mean((z - y)**2)
    print(f"MSE: {loss}")
    # Backward pass backpropagation
    # Compute gradients
    dL_dw = 2 * np.mean(x * (z - y))
    dL_db = 2 * np.mean(z - y)

    # Update weights and bias
    w -= lr * dL_dw
    b -= lr * dL_db

print("Final weights:", w)
print("Final bias:", b)

# Plot input-output pairs and regression line
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, w*x + b, color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with Gradient Descent')
plt.show()
