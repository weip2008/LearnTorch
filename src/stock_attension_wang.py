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
from modellib import *

# Define the file path
file_train = 'stockdata/SPY_TrainingData_200_11.csv' # 价格归一
file_test = 'stockdata/SPY_TestingData_200_11.csv'
file_train = 'stockdata/SPY_TrainingData_200_10.csv' # 原始价格
file_test = 'stockdata/SPY_TestingData_200_10.csv'
labels = ["long","short"]
total=54
columns = 6
window = 100
batch_global = 5

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        attention_input = torch.cat((encoder_outputs, h), dim=2)
        attention_energies = self.score(attention_input)
        return nn.functional.softmax(attention_energies, dim=1)
    
    def score(self, attention_input):
        energy = torch.tanh(self.attention(attention_input))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(attention_input.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        hidden = hidden[-1]
        attention_weights = self.attention(hidden, encoder_outputs)
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)
        rnn_output, (hidden, cell) = self.decoder(context.unsqueeze(1))
        output = self.fc(rnn_output.squeeze(1))
        return output


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
    print(f'total number of output data: {len(outputs)}')
    print(f'total input: {len(inputs)}, window size: {len(inputs[0])}')
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

def train(dataloader, model, loss_fn, optimizer):
    global window,columns,batch_global
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.reshape([5,1,1200])
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
            X = X.reshape([5,1,1200])
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, reduction='none')
        weighted_loss = loss * self.weights
        return weighted_loss.mean()


if __name__ == "__main__":
    batch_global = 5
    trainDataset = getTrainingDataSet(file_train)   
    testDataset = getTestingDataSet(file_test)
    train_dataloader = DataLoader(trainDataset, batch_size=batch_global) # the train data include images (input) and its lable index (output)
    test_dataloader = DataLoader(testDataset, batch_size=batch_global) # the train data include images (input) and its lable index (output)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Generate exponentially increasing weights
    # num_samples = len(train_dataloader.dataset)
    num_samples = batch_global
    base = 1.01  # Adjust the base to control the rate of increase
    weights = np.exp(np.linspace(0, num_samples-1, num_samples) * np.log(base))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Set your hyperparameters
    input_size = window*columns  # This should be the number of features in your input data
    hidden_size = 256    # Number of features in the hidden state
    num_layers = 3       # Number of stacked RNN layers
    output_size = 2      # Number of output features

    device = 'cpu'
    model = Seq2SeqModel(input_size,hidden_size,output_size).to(device) # create an model instance without training

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = WeightedCrossEntropyLoss(weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-8) # lr: learning rate

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done with training.")

    torch.save(model.state_dict(), "stock_model.pth")
    print("Saved PyTorch Model State to stock_model.pth")