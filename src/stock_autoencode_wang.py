import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn.functional as F
import os

file_train = 'stockdata/SPY_TrainingData_200_10.csv'
file_test = 'stockdata/SPY_TestingData_200_10.csv'
labels = ["long", "short"]
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
        X = X.reshape([5, 1, 1, 1200])
        X, y = X.to(device), y.to(device)
        pred, _ = model(X)  # Get only the classification logits for loss calculation
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
            X = X.reshape([5, 1, 1, 1200])
            X, y = X.to(device), y.to(device)
            pred, _ = model(X)  # Get only the classification logits for loss calculation
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.encoder_fc = nn.Linear(32 * 2 * 150, 256)  # Adjust this based on the flattened size after conv layers
        
        # Latent space
        self.latent_fc = nn.Linear(256, 2)  # Final layer to produce the 2-dimensional output
        
        # Decoder
        self.decoder_fc = nn.Linear(2, 256)
        self.decoder_fc2 = nn.Linear(256, 32 * 2 * 150)  # Adjust based on the shape before flattening
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.decoder_conv1 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x_shape = x.size()  # Get the shape of the tensor
        x = x.view(x_shape[0], -1)  # Flatten the tensor to (batch_size, num_channels * height * width)
        x = F.relu(self.encoder_fc(x))
        
        # Latent space
        latent = self.latent_fc(x)  # Produce the 2-dimensional output
        
        # Decoder
        x = F.relu(self.decoder_fc(latent))
        x = F.relu(self.decoder_fc2(x))
        x = x.view(x_shape[0], 32, 2, 150)  # Reshape back to the expected shape for ConvTranspose2d
        x = F.relu(self.decoder_conv2(x))
        x = torch.sigmoid(self.decoder_conv1(x))

        return latent, x  # Return both the latent and reconstructed output

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

    model = AutoEncoder().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.3e-8)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done with training.")

    filepath = os.path.join("outputs", "stock_model.pth")
    torch.save(model.state_dict(), filepath)
    print("Saved PyTorch Model State to stock_model.pth")
