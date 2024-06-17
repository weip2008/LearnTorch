import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Define the CNN model
class HumanDetectionCNN(nn.Module):
    def __init__(self):
        super(HumanDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Display 9 images from the dataset
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get some random training images
dataiter = iter(train_dataset)
images, labels = next(dataiter)

# Show images
imshow(torchvision.utils.make_grid(images))
# Print labels
# print(' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(9)))
print('Label:', train_dataset.classes[labels])

# Filter out only the "person" class (label 1 in CIFAR-10 is for "car" so use label 9 "truck" for binary classification)
person_label = 9

# Create binary datasets for human detection
train_indices = [i for i, label in enumerate(train_dataset.targets) if label == person_label or label == 0]
test_indices = [i for i, label in enumerate(test_dataset.targets) if label == person_label or label == 0]

train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

# Update labels to binary (1 for human, 0 for non-human)
for i in range(len(train_dataset)):
    train_dataset.dataset.targets[train_dataset.indices[i]] = 1 if train_dataset.dataset.targets[train_dataset.indices[i]] == person_label else 0

for i in range(len(test_dataset)):
    test_dataset.dataset.targets[test_dataset.indices[i]] = 1 if test_dataset.dataset.targets[test_dataset.indices[i]] == person_label else 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
model = HumanDetectionCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.float().view(-1, 1)  # Reshape labels for BCELoss
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

# Save the model
torch.save(model.state_dict(), 'human_detection_model.pth')

print("Model training complete and saved as human_detection_model.pth")
