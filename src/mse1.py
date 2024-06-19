import numpy as np

# Generate some random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Set learning rate and number of iterations
eta = 0.1
n_iterations = 100
m = 100  # number of instances

# Random initialization of theta
theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # MSE 的 一阶导数
    theta = theta - eta * gradients
    
    # Calculate MSE
    mse = np.mean((X_b.dot(theta) - y)**2)
    print(f"Iteration {iteration}: MSE = {mse}")

# Final theta values
print("Final theta values:", theta.ravel())
