import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import datetime


testing_file_path  = 'data/SPX_1m_PredictData_HL_43_700.txt'
model_save_path = 'GRU_model_with_LH_fixlen_data_600.pth'
output_results_path = 'data/SPX_1m_HL_43_700_GRU_fixlen_700.txt'

import numpy as np


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
model.eval()  # Set the model to evaluation mode


# Loss function: Binary Cross Entropy Loss
#criterion = nn.BCEWithLogitsLoss()  # Use with sigmoid for binary classification
criterion = nn.MSELoss()


# Training loop
print("4. Start testing loop")
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Evaluate the model on the testing data
test_loss = 0
all_targets = []
all_outputs = []

with torch.no_grad():
    for test_inputs, test_targets in test_dataloader:
        test_outputs = model(test_inputs)
        loss = criterion(test_outputs, test_targets)
        test_loss += loss.item()

        all_targets.extend(test_targets.numpy())
        all_outputs.extend(test_outputs.numpy())

avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss (MSE): {avg_test_loss:.4f}')
# Mean Squared Error (MSE) measures the average squared difference between the predicted values 
# and the actual values.
# A lower MSE indicates that the model’s predictions are closer to the actual values. 
# Test Loss (MSE): 0.01045113 suggests that, on average, the squared difference between the 
# predicted and actual values is quite small.

# Calculate additional metrics manually
all_targets = np.array(all_targets)
all_outputs = np.array(all_outputs)

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(all_targets - all_outputs))
print(f'Mean Absolute Error (MAE): {mae:.4f}')
# MAE measures the average absolute difference between the predicted values and the actual values.
# It gives an idea of how much the predictions deviate from the actual values on average. 
# Mean Absolute Error (MAE): 0.07155589 means on average, the model’s predictions are off by about 0.0716 
# units from the actual values.

# R-squared (R2)
ss_res = np.sum((all_targets - all_outputs) ** 2)
ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f'R-squared (R2): {r2:.4f}')
# R-squared is a statistical measure that represents the proportion of the variance for a 
# dependent variable that’s explained by an independent variable or variables in a regression model.
# R-squared (R2): 0.89939589  indicates that approximately 89.94% of the variance in the target variable
# is explained by the model. This is a high value, suggesting that the model fits the data well.

# MSE and MAE are both measures of prediction error, with lower values indicating better performance.
# R2 is a measure of how well the model explains the variability of the target data, 
#    with values closer to 1 indicating a better fit
    
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Open a text file in write mode
with open(output_results_path, 'w') as file:
    # Loop through all targets and outputs
    for target, output in zip(all_targets, all_outputs):
        # Apply the logic to categorize the output as either 1 or 0
        if 0.6 <= output <= 1.3:
            categorized_output = 1.0
        elif -1.3 <= output <= -0.6:
            categorized_output = -1.0
        else:
            categorized_output = 0.0  # For cases outside defined ranges, do nothing

        # Prepare the output string for each pair
        output_string = f"Target{target} : Output[{output[0]:.4f}] -> Signal[{categorized_output}]\n"
         
        # Print to the screen
        #print(output_string.strip())  # .strip() to avoid extra newlines

        # Write the same output to the file
        file.write(output_string)
        
print(f'Saved categorized signals to file : {output_results_path}')       
print(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        