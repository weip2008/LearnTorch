import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic time series data
def generate_synthetic_data(seq_len, num_features, batch_size):
    x = np.linspace(0, np.pi * 2, seq_len)
    data = np.sin(x) + np.random.normal(0, 0.1, (batch_size, seq_len, num_features))
    return torch.tensor(data, dtype=torch.float32)

# Parameters
seq_len = 10  # Length of the time series
x = np.linspace(0, np.pi * 2, seq_len)

num_features = 1  # Number of features in each time step
batch_size = 4  # Number of sequences in a batch

# Generate synthetic data
src_data = generate_synthetic_data(seq_len, num_features, batch_size)
tgt_data = generate_synthetic_data(seq_len, num_features, batch_size)

print("Source Data:", src_data)
print("Target Data:", tgt_data)

plt.plot(x,src_data[0])
plt.title("Source data [0]")
plt.show()

plt.plot(x,tgt_data[0])
plt.title("Target data [0]")
plt.show()