import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt

# Define the PolynomialModel class
class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialModel, self).__init__()
        self.degree = degree
        self.fc = nn.Linear(degree + 1, 1)

    def forward(self, x):
        x = torch.tensor([[xi**i for i in range(self.degree + 1)] for xi in x], dtype=torch.float32)
        return self.fc(x)

# Read the x and y values from the CSV file
file_name = 'data.csv'
x = []
y = []
with open(file_name, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))

# Convert x and y to NumPy arrays
x = np.array(x)
y = np.array(y)

# Convert x and y to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Define the degree of the polynomial
degree = 2

# Instantiate the model
model = PolynomialModel(degree)

# Define the loss function and optimizer
criterion = nn.MSELoss()
lr=1e-5
optimizer = optim.SGD(model.parameters(), lr=lr)

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Get the learned coefficients
coefficients = model.fc.weight.data.numpy()[0]
print("Learned coefficients:", coefficients)

# Plot the original data and the polynomial curve
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Original data')
x_plot = np.linspace(min(x), max(x), 100)
y_plot = np.polyval(coefficients[::-1], x)  # Reverse coefficients for np.polyval
plt.plot(x_plot, y_plot, color='red', label='Fitted polynomial curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
