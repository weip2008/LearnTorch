LSTM, or Long Short-Term Memory, is a type of recurrent neural network (RNN) architecture designed to model sequential data. LSTM networks are particularly effective at capturing long-term dependencies in data, which traditional RNNs struggle with due to the problem of vanishing or exploding gradients during training.

### Key Features of LSTMs:
1. **Memory Cells:** LSTMs have memory cells that can maintain their state over time. These cells enable LSTMs to remember information for long periods.
2. **Gates:** LSTMs use three types of gates to control the flow of information:
   - **Input Gate:** Controls the extent to which new information flows into the memory cell.
   - **Forget Gate:** Controls the extent to which information in the memory cell is retained or forgotten.
   - **Output Gate:** Controls the extent to which information from the memory cell is used to compute the output.

### LSTM Cell Structure:
1. **Forget Gate (\( f_t \)):**
   \[ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \]
   Determines which information to discard from the cell state.

2. **Input Gate (\( i_t \)):**
   \[ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \]
   \[ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \]
   Determines which information to add to the cell state.

3. **Cell State Update (\( C_t \)):**
   \[ C_t = f_t * C_{t-1} + i_t * \tilde{C}_t \]
   Updates the cell state by combining the old cell state and the new candidate values.

4. **Output Gate (\( o_t \)):**
   \[ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \]
   \[ h_t = o_t * \tanh(C_t) \]
   Determines the output based on the cell state.

### Applications of LSTMs:
- **Natural Language Processing (NLP):** Language modeling, text generation, machine translation.
- **Time Series Prediction:** Stock price prediction, weather forecasting.
- **Speech Recognition:** Recognizing spoken words and sentences.
- **Anomaly Detection:** Identifying unusual patterns in sequences of data.

### Example in PyTorch:
Here's a simple example of how to use an LSTM in PyTorch:

[](../src/lstm.py)

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Parameters
input_size = 10
hidden_size = 20
num_layers = 2
output_size = 1
sequence_length = 5

# Model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# Dummy input
x = torch.randn(32, sequence_length, input_size)
output = model(x)
print(output.shape)  # Output: torch.Size([32, 1])
```

In this example:
- `input_size` is the number of features in the input data.
- `hidden_size` is the number of features in the hidden state.
- `num_layers` is the number of stacked LSTM layers.
- `output_size` is the number of output features.

The `LSTMModel` class defines an LSTM layer followed by a fully connected layer. The forward pass initializes the hidden and cell states to zeros and processes the input through the LSTM and fully connected layers to produce the output.


## Gradient Vanish

[](../src/vanish.py)