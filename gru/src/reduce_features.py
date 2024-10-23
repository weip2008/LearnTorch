"""
combine more features to smaller features by unsupervised modeling.
for instance, reduce 28X28=784 handwriting images to 64 features.

only help performance on training, it is not very helpful.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Autoencoder model definition
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Initialize model, loss function, and optimizer
input_dim = 28 * 28  # MNIST image size (28x28 pixels)
encoding_dim = 64    # Dimension of reduced features
model = Autoencoder(input_dim, encoding_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training function
def train_autoencoder(dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, _ in dataloader:  # Data, _ because we don't need labels for unsupervised learning
            inputs = data.view(data.size(0), -1)  # Flatten 28x28 images to 784-dimensional vectors
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = model(inputs)
            
            # Compute loss (MSE between input and reconstruction)
            loss = criterion(reconstructed, inputs)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Feature extraction function
def get_encoded_features(dataloader):
    model.eval()  # Set model to evaluation mode
    all_encoded_features = []
    with torch.no_grad():  # Disable gradient computation
        for data, _ in dataloader:
            inputs = data.view(data.size(0), -1)
            _, encoded_features = model(inputs)
            all_encoded_features.append(encoded_features)
    return torch.cat(all_encoded_features, dim=0)

def getDataLoader():
    # Load the MNIST dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader

def defineModel():
    # Initialize model, loss function, and optimizer
    input_dim = 28 * 28  # MNIST image size (28x28 pixels)
    encoding_dim = 64    # Dimension of reduced features
    model = Autoencoder(input_dim, encoding_dim)
    return model

def main():
    train_loader = getDataLoader()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the autoencoder
    train_autoencoder(train_loader, criterion, optimizer, epochs=5)
    
    # Extract reduced features from the dataset
    reduced_features = get_encoded_features(train_loader)
    print("Reduced features shape:", reduced_features.shape)

if __name__ == '__main__':
    main()
