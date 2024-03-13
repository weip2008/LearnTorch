import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from fashion02 import NeuralNetwork, training_data, label_names

model = NeuralNetwork()
model.load_state_dict(torch.load("fashion_model.pth"))

model.eval()
img, labelIndex = training_data[111][0], training_data[111][1]
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