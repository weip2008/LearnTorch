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
num_features = 10  # Number of features in each time step
batch_size = 4  # Number of sequences in a batch

# Generate synthetic data
src_data = generate_synthetic_data(seq_len, num_features, batch_size)
tgt_data = generate_synthetic_data(seq_len, num_features, batch_size)

# Create sequence mask for the target sequence
tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

# Padding masks (not needed in this simple example as there's no padding)
src_key_padding_mask = None
tgt_key_padding_mask = None

# Define the transformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Apply embedding
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        # Apply transformer
        output = self.transformer(
            src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Apply final linear layer to each time step
        output = self.fc_out(output)
        
        return output

# Instantiate the model
input_size = num_features
d_model = 8
nhead = 2
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 32
output_size = num_features
dropout = 0.1

model = TimeSeriesTransformer(input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_size, dropout)

# Define a simple loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (for demonstration purposes, we'll just run a few iterations)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(src_data, tgt_data, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
    loss = criterion(output, tgt_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# Set the model to evaluation mode
model.eval()

# Generate new synthetic data for testing
test_src_data = generate_synthetic_data(seq_len, num_features, batch_size)

# Make predictions
with torch.no_grad():
    predictions = model(test_src_data, test_src_data, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

print("Predictions:", predictions)