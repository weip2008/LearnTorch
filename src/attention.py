import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply linear layer to input
        x = self.fc(x)
        batch_size, seq_len, _ = x.size()

        # Calculate attention scores
        attn_energies = self.attn(x)
        attn_energies = self.tanh(attn_energies)
        attn_energies = attn_energies.matmul(self.v)

        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_energies)

        # Apply attention weights to input
        context = torch.einsum("ijk,ij->ik", x, attn_weights)

        # Output prediction
        output = self.out(context)
        return output

# Example usage
input_size = 200
hidden_size = 128
output_size = 3

# Create input tensor
x = torch.randn(500, 1, 6, 200)

# Flatten the input tensor
x = x.view(x.size(0), -1).reshape(500,6,200)

# Create Attention model
attention = Attention(input_size, hidden_size, output_size)

# Forward pass
output = attention(x)
print(output.shape)  # Output shape: [500, 3]
