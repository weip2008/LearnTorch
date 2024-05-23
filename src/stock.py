import csv
import ast
import numpy as np
import torch

# Step 1: Read the CSV file
filename = 'data/StockTraningData10.csv'  # replace with your CSV file path
data = []

with open(filename, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        output1 = float(row[0])
        output2 = float(row[1])
        inputs = ast.literal_eval(row[2])
        data.append((output1, output2, inputs))

# Step 2: Parse the data
outputs = []
inputs = []

for output1, output2, input_list in data:
    outputs.append([output1, output2])
    input_values = []
    for _, value1, value2, value3, value4 in input_list:
        input_values.extend([value1, value2, value3, value4])
    inputs.append(input_values)

# Step 3: Format the data
outputs = np.array(outputs, dtype=np.float32)
inputs = np.array(inputs, dtype=np.float32)

# Convert to PyTorch tensors
outputs_tensor = torch.tensor(outputs)
inputs_tensor = torch.tensor(inputs)

# Now you have your inputs_tensor and outputs_tensor ready for NN training
print('Outputs:', outputs_tensor)
print('Inputs:', inputs_tensor)
