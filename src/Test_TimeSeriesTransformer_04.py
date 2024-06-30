import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import matplotlib.pyplot as plt

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

# Load data from CSV
low_data, high_data = load_data('data/SPX_TrainingData_200.csv')

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
low_dataset = VariableLengthTimeSeriesDataset(low_data)
high_dataset = VariableLengthTimeSeriesDataset(high_data)

low_dataloader = DataLoader(low_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
high_dataloader = DataLoader(high_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

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

# Check if a GPU is available and if not, fall back to the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TimeSeriesTransformer(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Function to create subsequent mask
def create_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Training loop
num_epochs = 10
losses = []

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()  # Set the model to training mode
    
    total_loss = 0
    for src, tgt_mask in low_dataloader:
        src, tgt_mask = src.to(device), tgt_mask.to(device)

        optimizer.zero_grad()
        
        # Prepare tgt_input by shifting tgt_mask to the right and prepending a start token
        tgt_input = torch.cat((torch.zeros(tgt_mask.size(0), 1, device=device), tgt_mask[:, :-1]), dim=1)
        
        # Forward pass
        output = model(src, tgt_input, tgt_mask=tgt_mask, src_key_padding_mask=mask.bool(), tgt_key_padding_mask=mask[:, :-1].bool())
        
        # Compute the loss
        tgt_output = tgt_mask[:, 1:]
        loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(low_dataloader)
    epoch_end_time = time.time()
    
    epoch_duration = (epoch_end_time - epoch_start_time) / 60  # Convert duration to minutes
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f} minutes')

    losses.append(avg_loss)
    scheduler.step()

# Save the model
torch.save(model.state_dict(), 'timeseries_transformer.pth')
print("Model saved successfully.")



# Plotting the training loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()
