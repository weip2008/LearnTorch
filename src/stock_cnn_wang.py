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
from cnn1 import StockCNN

file_train = 'stockdata/SPY_TrainingData_200_10.csv'
file_test = 'stockdata/SPY_TestingData_200_10.csv'
labels = ["long","short"]
total = 65
columns = 6
window = 200
batch_global = 5

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
    inputs = torch.tensor(inputs).reshape(len(inputs), columns, window)
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
    inputs = torch.tensor(inputs).reshape(len(inputs), columns, window)
    testingDataset = TensorDataset(inputs, outputs)
    return testingDataset

def train(dataloader, model, loss_fn, optimizer):
    global window, columns, batch_global
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.reshape([5,1,1200])
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
            X = X.reshape([5,1,1200])
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=columns, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * (window // 2), 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * (window // 2))
        x = self.fc1(x)
        return x

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
    optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-8)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done with training.")

    filepath = os.path.join("outputs", "stock_model.pth")
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to stock_model.pth")
