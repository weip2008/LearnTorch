import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Define the file path
file_path = 'data/SPY_TraningData06.csv'

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
    print("Outputs:")
    print(outputs)
    # print("\nInputs:")
    # print(inputs)

    # print(len(outputs))
    print(len(inputs),len(inputs[0]))
    # Convert to PyTorch tensors
    outputs_tensor = torch.tensor(outputs).reshape(18,2)
    inputs_tensor = torch.tensor(inputs).reshape(18,1,6,10)
    test_output_tensor = torch.tensor([int(y == 1.0) for x, y in outputs])
    trainingDataset = TensorDataset(inputs_tensor, outputs_tensor)
    testingDataset = TensorDataset(inputs_tensor, test_output_tensor)
    return trainingDataset, testingDataset

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6*10, 10),
            nn.ReLU(),  # Rectified Linear Unit
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
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

        if batch % 100 == 0:
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
            y1 = pred.argmax(1)
            y2 = y1 == y
            correct += y2.type(torch.float).sum().item()
            print(f'correct: {correct}')
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    trainDataset,testDataset = getDataSet(file_path)   

    train_dataloader = DataLoader(trainDataset, batch_size=3) # the train data include images (input) and its lable index (output)
    test_dataloader = DataLoader(testDataset, batch_size=3) # the train data include images (input) and its lable index (output)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # device = 'gpu'
    model = NeuralNetwork().to(device) # create an model instance without training

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # lr: learning rate

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}********************")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done with training.")

    torch.save(model.state_dict(), "stock_model.pth")
    print("Saved PyTorch Model State to stock_model.pth")