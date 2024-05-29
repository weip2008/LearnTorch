"""
Plot the price, velocity, and acceleration for a single point within a given window..
"""
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Define the file path
file_path = 'stockdata/SPY_TestingData_200_09.csv'

inputs = []
with open(file_path, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    print(csvreader)

    for row in csvreader:
        inputs.append(tuple(map(float, row[1:])))
        break
x = []
y = []
v = []
a = []
const = 5
for i in range(int(len(inputs[0])/6)):
    x.append(i)
    y.append(inputs[0][i*6+2])
    v.append(inputs[0][i*6+4])
    a.append(inputs[0][i*6+5])
fig, ax1 = plt.subplots()
ax1.plot(x, y, 'b-')
ax1.set_xlabel('Index')
ax1.set_ylabel('Price', color='b')

# Creating a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(x, v, 'r-')
ax2.set_ylabel('velocity', color='r')

plt.show()