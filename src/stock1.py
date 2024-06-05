import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from stock import NeuralNetwork, file_path, getDataSet, labels,window

model = NeuralNetwork()
model.load_state_dict(torch.load("stock_model_30_07_83.pth"))
index = 0
model.eval()
training_data,test_data = getDataSet(file_path)
count = 0
for index in range(len(training_data)):
    stock, decision = training_data[index][0], training_data[index][1]
    # print(stock.shape, decision.shape, decision.ndim)
    # print(stock, decision)
    answer = torch.argmax(decision).item()
    # print(index)


    with torch.no_grad():
        stock = stock.reshape(1,6,window)
        pred = model(stock)
        predicted, actual = pred[0].argmax(0), answer
        if predicted==actual: count += 1
        # print(f'model output: {pred[0]}')
        print(f'Predicted: "{labels[predicted]}", Actual: "{labels[actual]}"')

print(f"accuracy: {(100*count/len(training_data)):0.2f}")