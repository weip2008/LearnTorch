import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Define the function to load data
def load_data(file_path):
    data = []
    targets = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into data and target parts
            data_part, target_part = line.strip().split('][')
            
            # Add the closing bracket to the data part and opening bracket to the target part
            data_part += ']'
            target_part = '[' + target_part
            
            # Convert the string representations to actual lists
            data_row = eval(data_part)
            target_row = eval(target_part)
            
            # Append to the respective lists
            data.append(data_row)
            targets.append(target_row)
    
    # Convert lists to numpy arrays
    data = np.array(data)
    targets = np.array(targets)
    
    return data, targets

# Define the custom dataset
class FixedLengthDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, h_n = self.gru(x)
        output = self.fc(h_n[-1])
        return output

# Load the testing data
test_file_path = 'data/SPX_TestingData_FixLenGRU_603.txt'
test_data, test_targets = load_data(test_file_path)

# Create a DataLoader for the testing data
test_dataset = FixedLengthDataset(test_data, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load the saved model
model = GRUModel(input_size=5, hidden_size=50, output_size=3)
checkpoint = torch.load('GRU_model_with_fixed_length_data_603.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate the model on the testing data
test_loss = 0.0
all_targets = []
all_outputs = []
criterion = nn.MSELoss()
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        all_targets.extend(targets.numpy())
        all_outputs.extend(outputs.numpy())

avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss (MSE): {avg_test_loss:.8f}')
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
print(f'Mean Absolute Error (MAE): {mae:.8f}')
# MAE measures the average absolute difference between the predicted values and the actual values.
# It gives an idea of how much the predictions deviate from the actual values on average. 
# Mean Absolute Error (MAE): 0.07155589 means on average, the model’s predictions are off by about 0.0716 
# units from the actual values.

# R-squared (R2)
ss_res = np.sum((all_targets - all_outputs) ** 2)
ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f'R-squared (R2): {r2:.8f}')
# R-squared is a statistical measure that represents the proportion of the variance for a 
# dependent variable that’s explained by an independent variable or variables in a regression model.
# R-squared (R2): 0.89939589  indicates that approximately 89.94% of the variance in the target variable
# is explained by the model. This is a high value, suggesting that the model fits the data well.

# MSE and MAE are both measures of prediction error, with lower values indicating better performance.
# R2 is a measure of how well the model explains the variability of the target data, 
#    with values closer to 1 indicating a better fit