import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformModel(nn.Module):
    def __init__(self):
        super(TransformModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 200, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
input_size = (500, 1, 6, 200)
output_size = (500, 2)
x = torch.randn(input_size)
model = TransformModel()
output = model(x)
print(output.shape)  # Output shape: (500, 3)
