"""
从数据中取出一个图像，读入模型数据，使用模型判断该图像的分类。
"""

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from fashion02 import NeuralNetwork, training_data, label_names

model = NeuralNetwork()
model.load_state_dict(torch.load("fashion_model.pth"))
index = 655
model.eval()
img, labelIndex = training_data[index][0], training_data[index][1]
print(img.shape, labelIndex.shape, labelIndex.ndim)
print(img, labelIndex)
index = torch.argmax(labelIndex).item()
print(img.shape, label_names[index], index, sep=', ')


with torch.no_grad():
    pred = model(img)
    predicted, actual = label_names[pred[0].argmax(0)], label_names[index]
    print(f'model output: {pred[0]}')
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

# img1 = img.transpose(0,1).transpose(1,2) # switch color axis with row axis, switch color with column
# print(type(img1), img.shape, img1.shape)
# plt.imshow(img1)

plt.imshow(img.squeeze())
plt.show()