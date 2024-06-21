"""
create a model for boolean AND.
"""
import numpy as np

# Define the input data and target output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target output for a simple binary classification (e.g., AND operation)
y = np.array([[0], 
              [0], 
              [0], 
              [1]])

# Initialize weights and biases randomly
np.random.seed(42)
W = np.random.rand(X.shape[1], 1)  # 2 input features, 1 output
b = np.random.rand(1)

# Define the learning rate
learning_rate = 0.1
epoch = 1000
# Training the linear model
for i in range(epoch):
    # Forward pass: compute the predicted output
    y_pred = np.dot(X, W) + b
    
    # Compute the error
    error = y - y_pred
    
    # Compute the gradient for weights and biases
    W_grad = -2 * np.dot(X.T, error) / X.shape[0]
    b_grad = -2 * np.sum(error) / X.shape[0]
    
    # Update weights and biases
    W -= learning_rate * W_grad
    b -= learning_rate * b_grad
    # print(f"{i}: w={W}\n{b}") # demo w,b changes when epoch<20

# Print final weights, biases, and outputs
print("Final weights:", W)
print("Final biases:", b)
print("Final outputs after training:", np.dot(X, W) + b)
