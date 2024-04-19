import torch

# Assuming input_tensor and output_tensor are already defined as tensors of the correct shape

# Example training data (replace with your actual data)
new_input_data = torch.rand(1000, 5)
new_output_data = torch.rand(1000, 2)

# Insert the new data into the input and output tensors
input_tensor[:1000] = new_input_data
output_tensor[:1000] = new_output_data

# Verify the shape of the tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)
