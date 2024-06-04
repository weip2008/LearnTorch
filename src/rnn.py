import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(2)
        x = x.view(batch_size, seq_length, -1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage
input_size = 200
hidden_size = 128
num_layers = 2
output_size = 3

# Create input tensor
x = torch.randn(500, 1, 6, 200)

# Create RNN
rnn = RNN(input_size, hidden_size, num_layers, output_size)

# Forward pass
output = rnn(x)
print(output.shape)  # Output shape: [500, 3]
