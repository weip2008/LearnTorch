import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Load and process the CSV data
def load_training_data(file_path):
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



# Custom dataset for variable-length sequences
class VariableLengthTimeSeriesDataset(Dataset):
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        lengths = [len(seq) for seq in self.data]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        
        batches = []
        for i in range(0, len(sorted_indices), self.batch_size):
            batch_indices = sorted_indices[i:i+self.batch_size]
            batches.append([self.data[idx] for idx in batch_indices])
        
        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]
        return collate_fn(batch)

# Collate function to pad sequences and create masks
def collate_fn(batch):
    sequences = [torch.tensor(item, dtype=torch.float32) for item in batch]
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)
    mask = torch.zeros(padded_sequences.size(0), padded_sequences.size(1), dtype=torch.bool)

    for i, seq_len in enumerate([len(seq) for seq in sequences]):
        mask[i, seq_len:] = True

    return padded_sequences, mask

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

# Function to create subsequent mask
def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Load test data from CSV
print("1. Load test data")
low_data, high_data = load_testing_data('data/SPX_TestingData_2HL_300.csv')
print(f"    Loaded {len(low_data)} low_data and {len(high_data)} high_data.")

# Create datasets and dataloaders
print("2. Create datasets and dataloaders")
batch_size = 32
low_dataset = VariableLengthTimeSeriesDataset(low_data, batch_size)
high_dataset = VariableLengthTimeSeriesDataset(high_data, batch_size)

low_dataloader = DataLoader(low_dataset, batch_size=None, shuffle=False)
high_dataloader = DataLoader(high_dataset, batch_size=None, shuffle=False)

# Load the trained model
print("3. Load trained model")
input_size = 1
d_model = 64
nhead = 4
num_encoder_layers = 3
num_decoder_layers = 3
dim_feedforward = 128
output_size = 1
dropout = 0.1

model = TimeSeriesTransformer(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout)
model.load_state_dict(torch.load('timeseries_transformer_06_2HL.pth'))
model.eval()

# Define tolerance threshold
tolerance_threshold = 0.1  # 10%

# Evaluation loop
print("4. Evaluation Loop")
criterion = nn.MSELoss()
total_loss = 0.0
total_accurate = 0
total_predictions = 0

with torch.no_grad():
    for (low_batch, low_mask), (high_batch, high_mask) in zip(low_dataloader, high_dataloader):
        for batch, mask in [(low_batch, low_mask), (high_batch, high_mask)]:
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
            accurate_predictions = torch.abs(output - tgt_output) / torch.abs(tgt_output) < tolerance_threshold
            total_accurate += accurate_predictions.sum().item()
            total_predictions += tgt_output.numel()

avg_loss = total_loss / (len(low_dataloader) + len(high_dataloader))  # Adjusting for two dataloaders
accuracy_percentage = (total_accurate / total_predictions) * 100

print(f'Average Loss (MSE): {avg_loss:.4f}')
print(f'Accuracy: {accuracy_percentage:.2f}%')

print("Model evaluation completed.")
