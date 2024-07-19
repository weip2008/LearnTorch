import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import csv
import matplotlib.pyplot as plt

# Function to load data (same as before)
def load_data(file_path):
    low_data = []
    high_data = []
    low_target = []
    high_target = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            prefix = list(map(int, row[:1]))  # Convert prefix to a list of integers
            sequence = ', '.join(row[1:])    # Join the rest of the row back into a single string
            
            # Split the sequence into tuples
            sequence = sequence.strip("()")  # Remove leading and trailing parentheses
            items = sequence.split('), (')   # Split based on the tuple separator

            data_list = []
            for item in items:
                item = item.replace('(', '').replace(')', '')  # Remove any remaining parentheses
                data_tuple = tuple(map(float, item.split(', ')))  # Convert to a tuple of floats
                data_list.append(data_tuple)

            # Convert the list of tuples to a numpy array (2D array)
            data_array = np.array(data_list)

            # Extract the target value from the last row, 3rd column
            target_value = data_array[-1, 2]

            if prefix == [1]:
                low_data.append(data_array)
                low_target.append(target_value)
            elif prefix == [0]:
                high_data.append(data_array)
                high_target.append(target_value)

    return low_data, high_data, low_target, high_target

# Load testing data
test_file_path = 'data/SPX_TestingData_2HL_501.csv'
test_low_data, test_high_data, test_low_target, test_high_target = load_data(test_file_path)
print(f'Low data: {len(test_low_data)} sequences')
print(f'High data: {len(test_high_data)} sequences')
print(f'Low targets: {test_low_target}')
print(f'High targets: {test_high_target}')


# Create a custom dataset (same as before)
class VariableLengthDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

# Custom collate function for dynamic batching (same as before)
def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    data, targets = zip(*batch)
    lengths = [seq.shape[0] for seq in data]
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True)
    targets = torch.stack(targets)
    return padded_data, targets, lengths

# Define the GRU model (same as before)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_output, h_n = self.gru(packed_input)
        output = self.fc(h_n[-1])
        return output

# Create datasets and dataloaders for testing data
test_low_dataset = VariableLengthDataset(test_low_data, test_low_target)
test_high_dataset = VariableLengthDataset(test_high_data, test_high_target)
test_low_dataloader = DataLoader(test_low_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_high_dataloader = DataLoader(test_high_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# Load the trained model and optimizer states
checkpoint = torch.load('GRU_training_results_501.pth')
model = GRUModel(input_size=5, hidden_size=50, output_size=1)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Define the loss function
criterion = nn.MSELoss()

# Function to evaluate the model
def evaluate_model(dataloader):
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            outputs = model(inputs, lengths)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# Evaluate the model on the testing data
test_low_loss = evaluate_model(test_low_dataloader)
test_high_loss = evaluate_model(test_high_dataloader)

#print(f'Test Low Data Loss: {test_low_loss:.4f}')
#print(f'Test High Data Loss: {test_high_loss:.4f}')

print(f'Test Low Data Loss: {test_low_loss}')
print(f'Test High Data Loss: {test_high_loss}')

