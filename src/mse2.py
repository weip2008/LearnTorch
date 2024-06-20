import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y1 = 4 + 3 * X + np.random.randn(100, 1)*0.2
y2 = 4.3 + 3.5 * X + np.random.randn(100, 1)*0.2
y3 = 3.5 + 2.5 * X + np.random.randn(100, 1)*0.2

# Plot the data
plt.figure(figsize=(8, 6))
# plt.plot(X, y, c='blue', marker='o', label='Data points')
plt.scatter(X, y1, c='blue', marker='o', label='y1=3x+4')
plt.scatter(X, y2, c='red', marker='^', label='y2=3.5x+4.3')
plt.scatter(X, y3, c='black', marker='*', label='y3=2.5x+3.5')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Different w,b give different MSE')
plt.legend()
plt.grid(True)
plt.show()
