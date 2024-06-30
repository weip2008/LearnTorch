import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Step 1: Load Testing Data
def load_testing_data(file_path):
    low_data = []
    high_data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            if line[0] == '0':
                low_data.append([float(value) for value in line[1:]])
            elif line[0] == '1':
                high_data.append([float(value) for value in line[1:]])
    
    return low_data, high_data


# Example of BasicTransformerEncoder
class BasicTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(BasicTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    
    def forward(self, src):
        # src shape: (seq_len, batch_size, d_model)
        output = self.transformer_encoder(src)
        return output


# Example usage within your model class
class YourModel(nn.Module):
    def __init__(self, input_size, dim_feedforward, output_size):
        super(YourModel, self).__init__()
        
        # Example initialization of BasicTransformerEncoder
        d_model = 512     # Example dimensionality
        nhead = 8         # Example number of attention heads
        num_layers = 6    # Example number of layers
        self.encoder = BasicTransformerEncoder(d_model, nhead, num_layers)
        
        # Other parts of your model initialization
        # Initialize other layers as needed
        
    def forward(self, low_data, high_data):
        # Example forward pass
        encoded_low = self.encoder(low_data)
        encoded_high = self.encoder(high_data)
        
        # Process further or return as needed
        return encoded_low, encoded_high

# Assuming load_testing_data is defined and returns low_data and high_data as lists
low_data, high_data = load_testing_data('data/SPX_TestingData_200.csv')

# Convert data to PyTorch tensors
low_data_tensor = torch.tensor(low_data, dtype=torch.float32)
high_data_tensor = torch.tensor(high_data, dtype=torch.float32)

# Assuming model is defined and already trained
model.eval()  # Set the model to evaluation mode

# Perform inference
with torch.no_grad():
    encoded_low = model(low_data_tensor)
    encoded_high = model(high_data_tensor)

# Now you can use encoded_low and encoded_high for further analysis or evaluation
