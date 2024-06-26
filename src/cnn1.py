import torch
import torch.nn as nn
import torch.nn.functional as F

class StockCNN(nn.Module):
    def __init__(self):
        super(StockCNN, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Define the max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * (150 // 8), 128)  # Adjust size based on pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * (1200 // 8))
        
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output layer
        x = self.fc3(x)
        return x

# Example usage
if __name__ == "__main__":
    model = StockCNN()
    print(model)
    
    # Example input with batch size 5 and 1200 input features
    example_input = torch.randn(5, 1, 1200)
    output = model(example_input)
    print(output)
