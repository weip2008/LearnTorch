import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset

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

# Step 2: Prepare Testing DataLoader
class VariableLengthTimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32)

# Step 3: Load Saved Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, input_size, dim_feedforward, output_size):
        super(TimeSeriesTransformer, self).__init__()
        # Define your model architecture here
        # Example:
        self.encoder = nn.TransformerEncoder(...)
        self.decoder = nn.TransformerDecoder(...)
        # Replace ... with appropriate arguments depending on your model's implementation for the inference step.
    
    def forward(self, src, tgt):
        # Implement your forward pass logic here
        # Example:
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 4: Evaluate on Testing Data
def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch_idx, src in enumerate(dataloader, 1):
            src = src.to(device)
            # Process input and obtain predictions
            # Example:
            output = model(src, ...)
            # Process output as needed

# Load testing data
low_data, high_data = load_testing_data('data/SPX_TestingData_200.csv')

# Create datasets and dataloaders
low_dataset = VariableLengthTimeSeriesDataset(low_data)
high_dataset = VariableLengthTimeSeriesDataset(high_data)

batch_size = 32
low_dataloader = DataLoader(low_dataset, batch_size=batch_size, shuffle=False)
high_dataloader = DataLoader(high_dataset, batch_size=batch_size, shuffle=False)

# Load saved model
model = TimeSeriesTransformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, input_size=your_input_size, dim_feedforward=your_dim_feedforward, output_size=your_output_size).to(device)
model.load_state_dict(torch.load('timeseries_transformer.pth'))
model.eval()

# Evaluate on low_data
print("Testing on low_data:")
evaluate(model, low_dataloader)

# Evaluate on high_data
print("Testing on high_data:")
evaluate(model, high_dataloader)

