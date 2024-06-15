import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read Data from CSV File
data = pd.read_csv('data/linear_data_with_deviation.csv')

# Separate the features (x) and target (y)
x = data['x'].values
y = data['y'].values

# Convert the data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Step 2: Build and Train the Neural Network Model
# Define the neural network model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 3: Evaluate the Model to Find the Slope and Intercept
# Get the learned parameters
for name, param in model.named_parameters():
    if name == 'linear.weight':
        slope = param.item()
    if name == 'linear.bias':
        intercept = param.item()

print(f'Learned slope: {slope}')
print(f'Learned intercept: {intercept}')

# Plot the original data and the learned linear function
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points with noise')
plt.plot(x, slope * x + intercept, color='red', label=f'Learned line: y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of generated linear data with deviation')
plt.legend()
plt.grid(True)
plt.show()
