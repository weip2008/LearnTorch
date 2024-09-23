import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime
import numpy as np
import random

testing_file_path  = 'data/SPX_1m_TestingData_HL_80_400.txt'
model_save_path = 'GRU_model_with_LH_fixlen_data_400.pth'
sample_size = 20

def load_testing_data(training_file_path):
    data = []
    signals = []

    with open(training_file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            signals_part, data_part = line.strip().split(',[')
            
            signal = int(signals_part.strip())
            signals.append(signal)
            
            # Add the beginning bracket to the data part and opening bracket to the target part
            data_part = '[' + data_part
            
            # Convert the string representations to actual lists
            data_row = eval(data_part)
            
            # Append to the respective lists
            data.append(data_row)
            #targets.append(target_row[0])  # Ensure target_row is a 1D array
    
    # Convert lists to numpy arrays
    data_np = np.array(data)
    signals_np = np.array(signals).reshape(-1, 1)  # Reshape to (6883, 1)
    #signals_np = np.array(signals)
    
    return data_np, signals_np


# Example usage
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"1. Load testing data from {testing_file_path}")
testing_data, testing_signals = load_testing_data(testing_file_path)

print("Data shape:", testing_data.shape)
print("Targets shape:", testing_signals.shape)

# Custom dataset class for loading signals and data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.signals[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Instantiate the dataset
print("2. Define dataset and dataloader")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
test_dataset = TimeSeriesDataset(testing_data, testing_signals)

# Create DataLoader for batching
batch_size = 32  # You can change the batch size as needed

# Test dataloader with shuffling
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the GRU model with 2 layers
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model, define the loss function and the optimizer
print("3. Instantiate the model, define the loss function and the optimize")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Define hyperparameters
input_size = 5  # Number of features in each tuple
hidden_size = 64  # Number of features in the hidden state
output_size = 1  # Number of output features (signal)
num_layers = 5    # Number of GRU layers


# Instantiate the model
print(f"Number of layers: {num_layers}")
model = GRUModel(input_size, hidden_size, output_size, num_layers)

# Load the saved model state
print(f"3. Load trained model from {model_save_path}")
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
#model.eval()  # Set the model to evaluation mode


# Function to categorize the model output
def categorize_output(output):
    if 0.6 <= output <= 1.3:
        return 1.0
    elif -1.3 <= output <= -0.6:
        return -1.0
    else:
        return 0.0

# Function to get the model output for a single input row
def get_model_output(single_input):
    single_input_tensor = torch.tensor(single_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need for gradients during testing
        test_output = model(single_input_tensor)
    return test_output.item()  # Return the single output as a scalar


# Randomly select 10 rows from testing data
random_indices = random.sample(range(len(testing_data)), sample_size)
random_datas = testing_data[random_indices]
random_targets = testing_signals[random_indices]


# Print the output for each selected row
print("Randomly selected 10 rows and their corresponding outputs:")
for i in range(sample_size):
    test_data = random_datas[i]
    test_target = random_targets[i].item()  # Get the actual target value
    
    # Call get_model_output to get the predicted output
    test_output = get_model_output(test_data)
    
    # Call categorize_output to categorize the predicted output
    categorized_output = categorize_output(test_output)
    
    # Print the test output, categorized output, and test target
    print(f"Test Output: {test_output:.4f} => Categorized Output: {categorized_output}, \tTarget: {test_target}")


