import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from stock import NeuralNetwork, file_path, getDataSet

model = NeuralNetwork()
model.load_state_dict(torch.load("stock_model.pth"))
index = 0
model.eval()
training_data,test_data = getDataSet(file_path)
for index in range(len(training_data)):
    stock, decision = training_data[index][0], training_data[index][1]
    # print(stock.shape, decision.shape, decision.ndim)
    # print(stock, decision)
    answer = torch.argmax(decision).item()
    # print(index)


    with torch.no_grad():
        stock = stock.reshape(1,6,10)
        pred = model(stock)
        predicted, actual = pred[0].argmax(0), answer
        print(f'model output: {pred[0]}')
        print(f'Predicted: "{predicted}", Actual: "{actual}"')