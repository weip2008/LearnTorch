import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


test_file_path = 'data\SPX_30m_TestingData_FixLenGRU_150_1002.txt'
model_path = 'GRU_2layer_fixlen_30m_150_1002.pth'
predict_file_path = 'data\SPX_30m_PredictData_FixLenGRU_150_1002.txt'

# Define the function to load data
def load_testing_data(file_path):
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


# Define the GRU model with 2 layers
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # Initialize hidden state
        output, h_n = self.gru(x, h0)
        output = self.fc(h_n[-1])
        return output

# Function to load the trained model
def load_model(model_path, input_size, hidden_size, output_size, num_layers):
    model = GRUModel(input_size, hidden_size, output_size, num_layers)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



# Load the trained model
#model_path = 'GRU_model_with_fixed_length_data_603.pth'
model = load_model(model_path, input_size=3, hidden_size=50, output_size=3, num_layers=2)



# Load the testing data
#testing_file_path = 'data/SPX_TestingData_FixLenGRU_180_900.txt'
test_data, test_targets = load_testing_data(test_file_path)

# Create DataLoader for testing data
test_dataset = FixedLengthDataset(test_data, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

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

# Calculate additional metrics manually
all_targets = np.array(all_targets)
all_outputs = np.array(all_outputs)

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(all_targets - all_outputs))
print(f'Mean Absolute Error (MAE): {mae:.8f}')

# R-squared (R2)
ss_res = np.sum((all_targets - all_outputs) ** 2)
ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f'R-squared (R2): {r2:.8f}')
 

 #====================== Prediction ----------------------
print("5. Predict feture values.")

# # Predict function for new data
# def predict_new_data(new_data, model):
#     model.eval()
#     with torch.no_grad():
#         data_tensor = torch.tensor(new_data, dtype=torch.float32)
#         data_tensor = data_tensor.unsqueeze(0)  # Add batch dimension
#         predictions = model(data_tensor)
#         return predictions.squeeze().numpy()
    
# Function to prepare new data for prediction
def prepare_new_data(data):
    # Convert to numpy array and reshape
    #data = np.array(data)
    # Ensure data is in the shape [1, sequence_length, input_size]
    #data = data[np.newaxis, :]
    #print("Data shape:", data.shape)
    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    return data_tensor



print(f"Load {predict_file_path}")
predict_data, predict_targets = load_testing_data(predict_file_path)

print("Data shape:", predict_data.shape)
print("Targets shape:", predict_targets.shape)
# Prepare the new data
prepared_data = prepare_new_data(predict_data)
#predict_data_tensor = [torch.tensor(seq, dtype=torch.float32) for seq in predict_data]

# Load the saved model
# model = GRUModel(input_size=5, hidden_size=50, output_size=3)
# checkpoint = torch.load('GRU_model_with_fixed_length_data_604.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(prepared_data)
    #predictions = model(predict_data_tensor)
    predictions = predictions.squeeze().numpy()  # Convert to numpy for easier handling

# Ensure predictions is a 1-D array for consistency
if predictions.ndim == 0:
    predictions = np.expand_dims(predictions, axis=0)

# Post-process and print predictions
for i, prediction in enumerate(predictions):
    print("----------------------------------------------------------------")
    print(f'Prediction for sequence {i}: {prediction}')
    print(f'Real  data for sequence {i}: {predict_targets[i]}')
    
