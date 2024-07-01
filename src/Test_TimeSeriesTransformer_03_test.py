import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time

# Transformer model definition
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(
            src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        output = self.fc_out(output)
        return output

# Hyperparameters
input_size = 1
d_model = 64
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 128
output_size = 1
dropout = 0.1

# Load the saved model
model = TimeSeriesTransformer(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout)
model.load_state_dict(torch.load('timeseries_transformer.pth'))
model.eval()

# Step 1: Load and process the CSV data
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

# Load test data from CSV
print("1. Load test data")
low_test_data, high_test_data = load_data('data/SPX_TestingData_201.csv')

# Step 2: Create a custom dataset for variable-length sequences
class VariableLengthTimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index]
        return torch.tensor(sequence, dtype=torch.float32)

# Step 3: Collate function to pad sequences and create masks
def collate_fn(batch):
    sequences = [item for item in batch]
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    mask = torch.zeros(padded_sequences.size(0), padded_sequences.size(1), dtype=torch.bool)

    for i, seq_len in enumerate([len(seq) for seq in sequences]):
        mask[i, seq_len:] = True

    return padded_sequences, mask

# Create datasets and dataloaders
print("2. Create test datasets and dataloaders")
low_test_dataset = VariableLengthTimeSeriesDataset(low_test_data)
high_test_dataset = VariableLengthTimeSeriesDataset(high_test_data)

low_test_dataloader = DataLoader(low_test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)
high_test_dataloader = DataLoader(high_test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

# Function to create subsequent mask
def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Evaluation function
def evaluate_model(dataloader, model, threshold=0.1):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch, mask in dataloader:
            batch = batch.unsqueeze(-1)  # Adding feature dimension
            tgt_input = batch[:, :-1, :]
            tgt_output = batch[:, 1:, :]

            tgt_subsequent_mask = create_subsequent_mask(tgt_input.size(1)).to(tgt_input.device)

            output = model(
                batch, tgt_input, tgt_mask=tgt_subsequent_mask, 
                src_key_padding_mask=mask, tgt_key_padding_mask=mask[:, :-1]
            )

            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1, tgt_output.size(-1))
            loss = criterion(output, tgt_output)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = output.cpu().numpy()
            actuals = tgt_output.cpu().numpy()
            correct = (abs(predictions - actuals) < threshold).sum()
            total_correct += correct
            total_predictions += predictions.size

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_predictions
    return avg_loss, accuracy

# Evaluate the model on test data
print("3. Evaluate the model on test data")
low_test_loss, low_test_accuracy = evaluate_model(low_test_dataloader, model)
high_test_loss, high_test_accuracy = evaluate_model(high_test_dataloader, model)

print(f'Low test loss: {low_test_loss:.4f}')
print(f'Low test accuracy: {low_test_accuracy:.4f}')
print(f'High test loss: {high_test_loss:.4f}')
print(f'High test accuracy: {high_test_accuracy:.4f}')

print("Model evaluation completed.")
