import torch
import torch.nn as nn
from nonlinear05 import Net
import numpy as np
import matplotlib.pyplot as plt


f = lambda x: x**3 + x**2/2 - 4*x - 2 # actual nonlinear function

model = Net()
model.load_state_dict(torch.load("relu_model.pth"))

w1, b1 = model.fc1.weight.detach().numpy(), model.fc1.bias.detach().numpy()
w2, b2 = model.fc2.weight.detach().numpy(), model.fc2.bias.detach().numpy()
w3, b3 = model.fc3.weight.detach().numpy(), model.fc3.bias.detach().numpy()

print(w1.shape, b1.shape)
print(w2.shape, b2.shape)
print(w3.shape, b3.shape)

def f1(x):
    # print('f1(x):',w1.shape,x.shape)
    return relu(torch.tensor(np.matmul(w1, x.numpy().reshape(1,-1)) + b1.reshape(-1, 1))) 

relu = lambda x: np.maximum(0,x)

def f2(x):
    # print('f2(x):',w2.shape, x.shape)
    return relu(torch.tensor(np.matmul(w2, x) + b2.reshape(-1, 1))) 

def f3(x):
    # print('f3(x):',w3.shape, x.shape)
    return np.dot(w3, x.numpy()) + b3

def predict(x): 
    return f3(f2(f1(x))) # simluate our forward() method defined in Net() model

x = torch.linspace(-3, 3, 100, dtype=torch.float32)
y = x**3 + x**2/2 - 4*x - 2

fig, ax = plt.subplots()
ax.plot(x, y, label='Actual function')
# ax.plot(x, yPred, label='Predicted function')

for i in range(len(x)):
    ax.add_artist(plt.Circle((x[i], predict(x[i])), 0.05, color='red', fill=False))
plt.legend()
plt.show()