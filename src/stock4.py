"""
Load stock data (weekday,time,price,volume,velocity,acceleration) from csv file;
build a linear model
save the model to a file

important: training dataset and testing dataset are different
"""
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Define the file path
file_train = 'stockdata/SPY_TrainingData_200_09.csv'
file_test = 'stockdata/SPY_TestingData_200_09.csv'
labels = ["long","short"]
total=54
columns = 6
window = 100
batch_global = 10

def getTrainingDataSet(file_path):
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
            inputs.append(tuple(map(float, row[2:])))

    # Convert lists to tuples
    outputs = tuple(outputs)
    inputs = tuple(inputs)

    # # Print the results (for verification)
    # print("Outputs:")
    # print(outputs)
    # print("\nInputs:")
    # print(inputs)

    print("Training Data...")
    print(f'total rows of output dataset: {len(outputs)}')
    print(f'total input dataset: {len(inputs)}, columns X window: {len(inputs[0])}')
    total = len(inputs)
    window = int(len(inputs[2])/columns)
    print("window:",window)
    for i in range(total):
        if len(inputs[i])/columns!=window:
            raise RuntimeError(f"Input data Error. expected={window}, got {len(inputs[i])/columns}")
    # Convert to PyTorch tensors
    outputs_tensor = torch.tensor(outputs).reshape(total,2)
    inputs_tensor = torch.tensor(inputs).reshape(total,1,columns,window)
    trainingDataset = TensorDataset(inputs_tensor, outputs_tensor)
    return trainingDataset

def getTestingDataSet(file_path):
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
            outputs.append(int(row[0]))
            
            # The rest of the columns go into inputs and are converted to floats
            inputs.append(tuple(map(float, row[1:])))

    # Convert lists to tuples
    outputs = tuple(outputs)
    inputs = tuple(inputs)

    print("Test Data...")
    print(f'total number of output data: {len(outputs)}')
    print(f'total input: {len(inputs)}, number of data: {len(inputs[0])}')
    total = len(inputs)
    window = int(len(inputs[2])/columns)
    print("window:",window)
    for i in range(total):
        if len(inputs[i])/columns!=window:
            raise RuntimeError(f"Input data Error. expected={window}, got {len(inputs[i])/columns}")
    # Convert to PyTorch tensors
    outputs_tensor = torch.tensor(outputs).reshape(total)
    inputs_tensor = torch.tensor(inputs).reshape(total,1,columns,window)
    testingDataset = TensorDataset(inputs_tensor, outputs_tensor)
    return testingDataset

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        global window,columns,batch_global
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(columns*window, window),
            nn.ReLU(),  # Rectified Linear Unit
            nn.Linear(window, columns),
            nn.ReLU(),
            nn.Linear(columns, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    global window,columns,batch_global
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # y是

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % batch_global == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader: # y 包含 3个分类结果
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    trainDataset = getTrainingDataSet(file_train)   
    testDataset = getTestingDataSet(file_test)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_global) # the train data include images (input) and its lable index (output)
    test_dataloader = DataLoader(testDataset, batch_size=batch_global) # the train data include images (input) and its lable index (output)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # device = 'gpu'
    model = NeuralNetwork().to(device) # create an model instance without training

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-8) # lr: learning rate

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done with training.")

    torch.save(model.state_dict(), "stock_model.pth")
    print("Saved PyTorch Model State to stock_model.pth")