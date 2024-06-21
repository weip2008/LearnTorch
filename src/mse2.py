import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3*X
y1 = 4 + 3 * X + np.random.randn(100, 1)*0.2
y2 = 4.3 + 3.5 * X + np.random.randn(100, 1)*0.2
y3 = 3.5 + 1 * X + np.random.randn(100, 1)*0.2

# Calculate the Mean Squared Error for y1, y2, and y3
mse_y1 = np.mean((y - y1) ** 2)
mse_y2 = np.mean((y - y2) ** 2)
mse_y3 = np.mean((y - y3) ** 2)

# Plot the data
plt.figure(figsize=(8, 6))
# plt.plot(X, y, c='blue', marker='o', label='Data points')
plt.plot(X, y, 'k-', label='y=3x+4 model')
plt.scatter(X, y1, c='blue', marker='o', label=f'y1=3x+4+noise; mse={mse_y1:.4f}')
plt.scatter(X, y2, c='red', marker='^', label=f'y2=3.5x+4.3+noise; mse={mse_y2:.4f}')
plt.scatter(X, y3, c='black', marker='*', label=f'y3=x+3.5+noise; mse={mse_y3:.4f}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Different w,b give different MSE')
plt.legend()
plt.grid(True)
plt.show()
