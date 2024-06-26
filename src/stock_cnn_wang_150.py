"""
from stock6.py
"""
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F
import os
# from cnn1 import StockCNN

file_train = 'data/SPX_TrainingData_30_150.csv'
file_test = 'data/SPX_TestingData_30_150.csv'
labels = ["long","short"]
total = 65
columns = 5
window = 30
batch_global = 64

class StockCNN(nn.Module):
    def __init__(self):
        super(StockCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * (150 // 8), 128)  # Adjust size based on pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output shape: [batch_size, 16, 75]
        x = self.pool(F.relu(self.conv2(x)))  # Output shape: [batch_size, 32, 37]
        x = self.pool(F.relu(self.conv3(x)))  # Output shape: [batch_size, 64, 18]
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * (150 // 8))  # Flatten the tensor for FC layers
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer
        x = self.fc3(x)
        return x
    
def getTrainingDataSet(file_path):
    global window, columns, batch_global, total
    outputs = []
    inputs = []

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            outputs.append((float(row[0]), float(row[1])))
            inputs.append(tuple(map(float, row[2:])))

    outputs = torch.tensor(outputs).reshape(len(outputs), 2)
    inputs = torch.tensor(inputs).reshape(len(inputs), 1, columns*window)
    trainingDataset = TensorDataset(inputs, outputs)
    return trainingDataset

def getTestingDataSet(file_path):
    global window, columns, batch_global, total
    outputs = []
    inputs = []

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            outputs.append(int(row[0]))
            inputs.append(tuple(map(float, row[1:])))

    outputs = torch.tensor(outputs).reshape(len(outputs))
    inputs = torch.tensor(inputs).reshape(len(inputs), 1, columns*window)
    testingDataset = TensorDataset(inputs, outputs)
    return testingDataset

def train(dataloader, model, loss_fn, optimizer):
    global window, columns, batch_global
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # X = X.reshape([5,1,1200])
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
            # X = X.reshape([5,1,1200])
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    trainDataset = getTrainingDataSet(file_train)
    testDataset = getTestingDataSet(file_test)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_global)
    test_dataloader = DataLoader(testDataset, batch_size=batch_global)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    num_samples = batch_global
    base = 1.01
    weights = np.exp(np.linspace(0, num_samples - 1, num_samples) * np.log(base))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = StockCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-4)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done with training.")

    filepath = os.path.join("outputs", "stock_model.pth")
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to stock_model.pth")
