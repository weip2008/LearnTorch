import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)   # 2x2 pooling
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Apply conv1, ReLU, and pool
        x = self.pool(F.relu(self.conv2(x))) # Apply conv2, ReLU, and pool
        x = x.view(-1, 16 * 5 * 5) # Flatten the tensor
        x = F.relu(self.fc1(x))    # Apply fully connected layers with ReLU
        x = F.relu(self.fc2(x))
        x = self.fc3(x)            # Output layer
        return x

if __name__ == '__main__':
    # Define transformations for the training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Define transformations for the training data with data augmentation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # Initialize the CNN, loss function, and optimizer
    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=2.31E-04, momentum=0.9)

    # Train the CNN
    for epoch in range(2):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()   # Zero the parameter gradients
            outputs = net(inputs)   # Forward pass
            loss = criterion(outputs, labels) # Compute loss
            loss.backward()         # Backward pass
            optimizer.step()        # Optimize weights

            running_loss += loss.item()
            if i % 2000 == 1999:    # Print statistics every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    # Save the trained model to a file
    filepath = os.path.join("outputs",'simple_cnn.pth')
    torch.save(net.state_dict(), filepath)
    print('Model saved to simple_cnn.pth')

    print('Finished Training')
