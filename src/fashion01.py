"""
从网上下载fashion数据，60000个28X28的服装数据，供分为10类。
取出一个图像，并显示出来。
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

training_data = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

train_dataloader = DataLoader(training_data, batch_size=64)
for x, y in train_dataloader:  #x: image; y:label
    print(f"Shape of x [N, C, H, W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

img_idx = 11
img = x[img_idx]
print(y[img_idx])
index = torch.argmax(y[img_idx]).item()
print(img.shape, img.mT.shape, index, label_names[index]) 
# plt.imshow(img) #TypeError: Invalid shape (1, 28, 28) for image data

# plt.imshow(img.T) # the transpose turn the image 90 degree

# print(img.numpy().shape, img.numpy().squeeze().shape)
# plt.imshow(img.numpy().squeeze()) # convert image to numpy, and squeeze it, it changes the tensor dimension to meet matplotlib requirement

# print(img.numpy().transpose(1,2,0).shape) # this process make the image meet the plot requirement
# plt.imshow(img.numpy().transpose(1,2,0)) # numpy.transpose()

img1 = img.transpose(0,1).transpose(1,2) # (1,28,28) ==> (28,28,1) switch color axis with row axis, switch color with column
print(type(img1), img.shape, img1.shape)
plt.imshow(img1)
plt.show()
plt.imshow(img.squeeze())
plt.show()