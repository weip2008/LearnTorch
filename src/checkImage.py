import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# Load the MNIST dataset
mnist_train = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())

# Check a sample image
sample_image, sample_label = mnist_train[0]
print(f"Pixel value range: [{sample_image.min()}, {sample_image.max()}]")
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f"Label: {sample_label}")
plt.show()
