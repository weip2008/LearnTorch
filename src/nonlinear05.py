import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10) # one input 1 x 10 weights 10 bias
        self.fc2 = nn.Linear(10, 10) # 10 x 10 weights
        self.fc3 = nn.Linear(10, 1) # one output
        self.relu = nn.ReLU()

    def forward(self, x): # show how to link each layer together to get final result
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    x = torch.linspace(-3, 3, 100, dtype=torch.float32)
    print(type(x), x.ndim)
    y = x**3 + x**2/2 - 4*x - 2

    net = Net()

    criterion = nn.MSELoss() # Mean Square Error 均方差 
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    optimizer = optim.SGD(net.parameters(), lr=0.01) # SGD: Stochastics Gradient Descent, adjust weight

    for epoch in range(1000):
        optimizer.zero_grad() # reset gradient
        outputs = net(x.unsqueeze(1)) # prediction for new weights
        loss = criterion(outputs, y.unsqueeze(1)) #
        loss.backward()
        optimizer.step()


    yPred = net(x.unsqueeze(1)).detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Actual function')
    # ax.plot(x, yPred, label='Predicted function')

    for i in range(len(x)):
        ax.add_artist(plt.Circle((x[i], yPred[i]), 0.05, color='red', fill=False))
    plt.legend()
    plt.show()

