"""
Load stock data (weekday,time,price,volume,velocity,acceleration) from csv file;
build a linear model
save the model to a file

add linear weights on data before modeling
"""
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F

# Define the file path
file_train = 'stockdata/SPY_TrainingData_50_13.csv'
file_test = 'stockdata/SPY_TestingData_50_13.csv'
labels = ["long","hold","short"]
total=54
columns = 6
window = 50
batch_global = 4

def getTrainingDataSet(file_path):
    global window,columns,batch_global,total
    outputs = []
    inputs = []

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            outputs.append((float(row[0]), float(row[1]), float(row[2])))
            inputs.append(tuple(map(float, row[3:])))

    outputs = tuple(outputs)
    inputs = tuple(inputs)

    print("Training Data...")
    print(f'total number of output data: {len(outputs)}')
    print(f'total input: {len(inputs)}, number of data: {len(inputs[0])}')
    total = len(inputs)
    window = int(len(inputs[0])/columns)
    print("window:",window)
    for i in range(total):
        if len(inputs[i])/columns!=window:
            raise RuntimeError(f"Input data Error. expected={window}, got {len(inputs[i])/columns}")

    outputs_tensor = torch.tensor(outputs).reshape(total,3)
    inputs_tensor = torch.tensor(inputs).reshape(total,1,columns,window)
    trainingDataset = TensorDataset(inputs_tensor, outputs_tensor)
    return trainingDataset

def getTestingDataSet(file_path):
    global window,columns,batch_global,total
    outputs = []
    inputs = []

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            outputs.append(int(row[0]))
            inputs.append(tuple(map(float, row[1:])))

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

    outputs_tensor = torch.tensor(outputs).reshape(total)
    inputs_tensor = torch.tensor(inputs).reshape(total,1,columns,window)
    testingDataset = TensorDataset(inputs_tensor, outputs_tensor)
    return testingDataset

class NeuralNetwork(nn.Module):
    def __init__(self):
        global window,columns,batch_global
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(columns*window, window),
            nn.ReLU(),
            nn.Linear(window, columns),
            nn.ReLU(),
            nn.Linear(columns, 3)
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
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

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
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            pass
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, reduction='none')
        weighted_loss = loss * self.weights
        return weighted_loss.mean()

if __name__ == "__main__":
    trainDataset = getTrainingDataSet(file_train)   
    testDataset = getTestingDataSet(file_test)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_global)
    test_dataloader = DataLoader(testDataset, batch_size=batch_global)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    num_samples = len(train_dataloader.dataset)
    base = 1.01
    weights = np.exp(np.linspace(0, num_samples-1, num_samples) * np.log(base))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.e-4)

    epochs = 20
    best_accurate = 0
    best_model_state = None

    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        current_accurate = test(test_dataloader, model, loss_fn)
        if current_accurate > best_accurate:
            best_accurate = current_accurate
            best_model_state = model.state_dict()

    print("Done with training.")

    if best_model_state is not None:
        model_file_name = f"best_stock_model_{int(best_accurate*100)}.pth"
        torch.save(best_model_state, model_file_name)
        print(f"Saved PyTorch Model State to {model_file_name}")
