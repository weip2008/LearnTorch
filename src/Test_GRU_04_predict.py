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

            if prefix == [0]:
                low_data.append(data_array)
                low_target.append(target_value)
            elif prefix == [1]:
                high_data.append(data_array)
                high_target.append(target_value)

    return low_data, high_data, low_target, high_target

# Load testing data
test_file_path = 'data/SPX_TestingData_2HL_505.csv'
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
checkpoint = torch.load('GRU_training_results_504.pth')
model = GRUModel(input_size=5, hidden_size=50, output_size=1)
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Define the loss function
criterion = nn.MSELoss()

# Function to evaluate the model
def evaluate_model(dataloader, threshold=0.1):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            outputs = model(inputs, lengths)
            loss = criterion(outputs.squeeze(), targets)
            total_loss += loss.item()
            
            # Calculate the number of correct predictions
            correct_predictions = torch.abs(outputs.squeeze() - targets) / torch.abs(targets) < threshold
            total_correct += correct_predictions.sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# Evaluate the model on the testing data
test_low_loss, test_low_accuracy = evaluate_model(test_low_dataloader)
test_high_loss, test_high_accuracy = evaluate_model(test_high_dataloader)

print(f'Test Low Data Loss (MSE): {test_low_loss:.6f}')
print(f'Test Low Data Accuracy: {test_low_accuracy:.6f}')
print(f'Test High Data Loss (MSE): {test_high_loss:.6f}')
print(f'Test High Data Accuracy: {test_high_accuracy:.6f}')


# Assuming the model class and checkpoint loading is done as previously described

# Set the model to evaluation mode
#model.eval()

# Example of new input data sequence
# new_data = [
#     [(0.5, 1.2, 0.3, 0.8, 1.0), (0.6, 1.3, 0.4, 0.9, 1.1)]  # Replace with actual data
# ]

# Use the first element from test_low_data as new input data sequence
new_data = [test_low_data[0]]
# Check the shape of the new_data to find out the number of rows and columns
num_rows, num_columns = new_data[0].shape
print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')
print(new_data)

# Convert new data to tensors and pad sequences
new_data_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in new_data]
new_lengths = [len(seq) for seq in new_data_tensor]
padded_new_data = nn.utils.rnn.pad_sequence(new_data_tensor, batch_first=True)

# Make predictions
with torch.no_grad():
    predictions = model(padded_new_data, new_lengths)
    predictions = predictions.squeeze().numpy()  # Convert to numpy for easier handling

# Ensure predictions is a 1-D array for consistency
if predictions.ndim == 0:
    predictions = np.expand_dims(predictions, axis=0)

# Post-process and print predictions
for i, prediction in enumerate(predictions):
    print(f'Prediction for sequence {i}: {prediction}')