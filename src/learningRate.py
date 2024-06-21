import numpy as np
import matplotlib.pyplot as plt

# Define the function we want to minimize (in this case, a simple quadratic function)
def f(x):
    return x**2 + 10*np.sin(x)

# Define the derivative of the function
def df(x):
    return 2*x + 10*np.cos(x)

# Define the gradient descent function
def gradient_descent(x0, alpha, eps, max_iter):
    x = x0
    x_history = [x]
    for i in range(max_iter):
        grad = df(x)
        x -= alpha * grad
        x_history.append(x)
        if np.abs(grad) < eps:
            break
    return x, np.array(x_history)

# Set the initial point, learning rate, and stopping criteria
x0 = -5
alpha = 0.1
eps = 1e-8
max_iter = 1000

# Run gradient descent
x_min, x_history = gradient_descent(x0, alpha, eps, max_iter)

# Plot the function and the path taken by gradient descent
x_vals = np.linspace(-10, 10, 1000)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals)
plt.plot(x_history, f(x_history), 'r.')
plt.xlabel('x')
plt.ylabel('Mean Square Error')
plt.show()
