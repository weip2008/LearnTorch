import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import imageio

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x

# Define a simple dataset
class WatermarkDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.image = imageio.imread(image_path)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = self.image
        if self.transform:
            image = self.transform(image)
        return image

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the image
image_path = 'data/watermark.jpg'
dataset = WatermarkDataset(image_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the pre-trained model (replace 'unet_watermark_removal.pth' with your model checkpoint)
model = UNet()
model.load_state_dict(torch.load('unet_watermark_removal.pth'))
model.eval()

# Remove watermark
for i, image in enumerate(dataloader):
    with torch.no_grad():
        output = model(image)
    output_image = output[0].permute(1, 2, 0).numpy() * 255
    output_image = output_image.astype(np.uint8)
    imageio.imwrite('data/without_watermark.jpg', output_image)

# Display the images
original_image = imageio.imread(image_path)
result_image = imageio.imread('data/without_watermark.jpg')

Image.fromarray(original_image).show()
Image.fromarray(result_image).show()
