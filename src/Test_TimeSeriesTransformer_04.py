import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time

# Assuming your TimeSeriesTransformer and Dataset classes are defined here

# Define your dataset class if not already defined
class VariableLengthTimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
# Define your DataLoader preparation function
def prepare_dataloader(data, batch_size):
    dataset = VariableLengthTimeSeriesDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to load data from CSV
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, header=None, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        print(f"Error parsing the file: {e}")
        return [], []

    low_data = []
    high_data = []
    
    for row in data.values:
        prefix = row[:2]
        sequence = row[2:]
        
        if prefix.tolist() == [1, 0]:
            low_data.append(sequence[~pd.isna(sequence)].tolist())
        elif prefix.tolist() == [0, 1]:
            high_data.append(sequence[~pd.isna(sequence)].tolist())
    
    return low_data, high_data




# Assuming your model definition here
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, input_size, dim_feedforward, output_size):
        super().__init__()
        # Initialize your model components here
        
    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Implement your forward pass logic here
        pass

# Load data from CSV
print("1. Load data")
low_data, high_data = load_data('data/SPX_TrainingData_200.csv')
    
# Initialize model
print("Initializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TimeSeriesTransformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                              input_size=..., dim_feedforward=..., output_size=...).to(device)

# Prepare data loaders
batch_size = 32
low_dataloader = prepare_dataloader(low_data, batch_size)
high_dataloader = prepare_dataloader(high_data, batch_size)

# Initialize optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}...")
    epoch_start_time = time.time()

    model.train()  # Set model to training mode
    total_loss = 0

    for i, (src, tgt_mask) in enumerate(low_dataloader):
        try:
            print(f"Processing batch {i+1}/{len(low_dataloader)}...")
            src = src.to(device)
            tgt_mask = tgt_mask.to(device)
            src_key_padding_mask = torch.zeros_like(src).bool()  # Adjust as needed
            tgt_key_padding_mask = torch.zeros_like(tgt_mask).bool()  # Adjust as needed

            # Prepare tgt_input for the decoder
            optimizer.zero_grad()
            tgt_input = torch.cat((torch.zeros(tgt_mask.size(0), 1, device=device), tgt_mask[:, :-1]), dim=1)
            print(f"tgt_input shape: {tgt_input.shape}")

            # Time profiling for transformer decoder
            start_time = time.time()
            output = model(src, tgt_input, tgt_mask=None, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            print(f"Model output shape: {output.shape}")
            decoder_duration = time.time() - start_time
            print(f"Decoder computation time: {decoder_duration:.2f} seconds")

            # Compute loss, backpropagate, and update parameters
            loss = criterion(output, tgt_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        except Exception as e:
            print(f"Error in batch {i+1}: {e}")
            raise e

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} completed. Total loss: {total_loss:.4f}, Duration: {epoch_duration / 60:.2f} minutes")

print("Training completed.")
