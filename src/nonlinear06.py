import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10) # one input
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1) # one output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    x = torch.linspace(-3, 3, 100, dtype=torch.float32)
    y = x**3 + x**2/2 - 4*x - 2

    net = Net()

    criterion = nn.MSELoss() # Mean Square Error 均方差 
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = net(x.unsqueeze(1))
        loss = criterion(outputs, y.unsqueeze(1))
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), 'relu_model.pth')
    print('Done.')

