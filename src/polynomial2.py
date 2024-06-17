import matplotlib.pyplot as plt
import numpy as np

# Generate N=10 random points
N = 10
np.random.seed(0)
# x_train = np.sort(np.random.rand(N))
x_train = np.linspace(0, 1, N)
t_train = np.sin(2 * np.pi * x_train) + np.random.randn(N) * 0.1

# True function curve
x_true = np.linspace(0, 1, 100)
y_true = np.sin(2 * np.pi * x_true)

if __name__ == "__main__":
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_true, y_true, 'g-', label='sin(2πx)')
    plt.scatter(x_train, t_train, color='blue', label='Training Data')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Training Data and True Function')
    plt.legend()
    plt.show()
