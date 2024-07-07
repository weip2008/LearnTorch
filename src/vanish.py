import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])
        return out, h

# Parameters
input_size = 1
hidden_size = 10
output_size = 1
sequence_length = 50  # Long sequence to demonstrate vanishing gradient
batch_size = 1

# Instantiate the model, loss function, and optimizer
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input and target
x = torch.randn(batch_size, sequence_length, input_size)
target = torch.randn(batch_size, output_size)

# Initial hidden state
h0 = torch.zeros(1, batch_size, hidden_size)

# Forward pass
output, hn = model(x, h0)
loss = criterion(output, target)

# Backward pass
optimizer.zero_grad()
loss.backward()

# Inspect the gradient of the initial hidden state
print("Gradients of the initial hidden state (h0):")
print(h0.grad)

# Inspect the gradients of the model parameters
print("\nGradients of the model parameters:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm().item()}")

# Observe that the gradients of the initial hidden state and the parameters are very small
