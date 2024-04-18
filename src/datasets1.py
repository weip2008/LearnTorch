import torch
from torch.utils.data import TensorDataset

actions = ["short","long"]
# Assuming you have input_data and output_data as NumPy arrays
input_data = [[1,2,3,4,5],[6,7,8,9,10]]  # Your input NumPy array with shape [num_samples, 5]
output_data = [[1,0],[0,1]]  # Your output NumPy array with shape [num_samples, 2]

# Convert NumPy arrays to PyTorch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)

print(f'Input Shape: {input_tensor.shape}; Output Shape: {output_tensor.shape}')

# Create a TensorDataset
dataset = TensorDataset(input_tensor, output_tensor)