import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Define the file path
file_path = 'stockdata/SPY_TraningData_200_08.csv'
labels = ["long","short"]
total=54
columns = 6
window = 100
batch_global = 10


def getDataTensor(file_path):
    global window,columns,batch_global,total
    # Initialize lists to store the outputs and inputs
    outputs = []
    inputs = []

    # Open and read the CSV file
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        
        # Iterate through each row in the CSV file
        for row in csvreader:
            # The first two columns go into outputs and are converted to floats
            outputs.append((float(row[0]), float(row[1])))
            
            # The rest of the columns go into inputs and are converted to floats
            # inputs.append(tuple(float(value) for value in row[2:]))
            inputs.append(tuple(map(float, row[2:])))

    # Convert lists to tuples
    outputs = tuple(outputs)
    inputs = tuple(inputs)

    total = len(inputs)
    window = int(len(inputs[2])/columns)
    print("window:",window)
    for i in range(total):
        if len(inputs[i])/columns!=window:
            raise RuntimeError(f"Input data Error. expected={window}, got {len(inputs[i])/columns}")
    # Convert to PyTorch tensors
    outputs_tensor = torch.tensor(outputs).reshape(total,2)
    inputs_tensor = torch.tensor(inputs).reshape(total,1,columns,window)
    test_output_tensor = torch.tensor([int(y == 1.0) for x, y in outputs])
    # trainingDataset = TensorDataset(inputs_tensor, outputs_tensor)
    # testingDataset = TensorDataset(inputs_tensor, test_output_tensor)
    return inputs_tensor, outputs_tensor

input_data, output_data = getDataTensor(file_path)
# Extract column 3 data
column_data = input_data[:, 0, 3, :]

# Plot the data
for i in range(column_data.shape[0]):
    plt.plot(column_data[i].numpy(), label=f'Row {i}')

# Customize the plot
plt.title('Line Chart of Column 3 Data')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()