import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        attention_input = torch.cat((encoder_outputs, h), dim=2)
        attention_energies = self.score(attention_input)
        return nn.functional.softmax(attention_energies, dim=1)
    
    def score(self, attention_input):
        energy = torch.tanh(self.attention(attention_input))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(attention_input.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        hidden = hidden[-1]
        attention_weights = self.attention(hidden, encoder_outputs)
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)
        rnn_output, (hidden, cell) = self.decoder(context.unsqueeze(1))
        output = self.fc(rnn_output.squeeze(1))
        return output

# Input parameters
input_dim = 1200  # Number of input features
hidden_dim = 256  # Number of hidden units in the LSTM
output_dim = 2    # Number of output classes or features

# Model instance
model = Seq2SeqModel(input_dim, hidden_dim, output_dim)

# Example input
input_data = torch.randn(65, 1, 1200)

# Forward pass
output = model(input_data)
print(output.shape)  # Should be (65, 2)
