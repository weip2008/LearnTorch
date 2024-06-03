import numpy as np
import matplotlib.pyplot as plt

num_samples = 200
x = np.linspace(0.1, 1.0, num_samples)

# Generate linear weights
linearWeights = np.linspace(0.1, 1.0, num_samples)

# Generate exponential weights
base = 1.01  # Adjust the base to control the rate of increase
exponentialWeights = np.exp(np.linspace(0, num_samples-1, num_samples) * np.log(base))

plt.plot(x, linearWeights,label="Linear Weights")
plt.plot(x, exponentialWeights,label="Exponential Weights")

plt.title("Comparison between Linear and exponential Weights")
plt.legend()
plt.show()