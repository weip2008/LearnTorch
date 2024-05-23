import csv
import torch
from torch.utils.data import TensorDataset, DataLoader

# Define the file path
file_path = 'data/StockTraningData05.csv'

def getDataSet(file_path):
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

    # # Print the results (for verification)
    # print("Outputs:")
    # print(outputs)
    # print("\nInputs:")
    # print(inputs)

    # print(len(outputs))
    print(len(inputs))
    # Convert to PyTorch tensors
    outputs_tensor = torch.tensor(outputs)
    inputs_tensor = torch.tensor(inputs)

    dataset = TensorDataset(inputs_tensor, outputs_tensor)
    return dataset

if __name__ == "__main__":
    dataset = getDataSet(file_path)   
    print(type(dataset))