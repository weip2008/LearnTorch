import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Example data (time and corresponding height)
# Replace this with your actual measurements
time = np.linspace(0, 2, 100)  # time in seconds
height = 4.9 * time**2  # height in meters (h = 0.5 * g * t^2 with g ≈ 9.8 m/s^2)

# Normalize data
time = (time - time.mean()) / time.std()
height = (height - height.mean()) / height.std()

# Convert to tensors
time_tensor = torch.tensor(time, dtype=torch.float32).view(-1, 1)
height_tensor = torch.tensor(height, dtype=torch.float32).view(-1, 1)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the neural network
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(time_tensor)
    loss = criterion(output, height_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predicted = model(time_tensor).numpy()

# Denormalize data for plotting
time = time * time.std() + time.mean()
height = height * height.std() + height.mean()
predicted = predicted * height.std() + height.mean()

# Plot the results
plt.plot(time, height, 'b^', label='Original data')
plt.plot(time, predicted, 'r', label='Fitted line')
plt.legend()
plt.show()
