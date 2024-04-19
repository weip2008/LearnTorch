import torch

# Number of data groups
num_groups = 2000

# Create random input data with shape (num_groups, 1000, 5)
input_data = torch.rand(num_groups, 1000, 5)

# Create random output data with shape (num_groups, 2)
output_data = torch.rand(num_groups, 2)

# Create tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32)

print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)