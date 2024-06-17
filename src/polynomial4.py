import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from polynomial2 import *

# Convert numpy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
t_train_tensor = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)

# Define a polynomial model
class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialModel, self).__init__()
        self.degree = degree
        self.linear = nn.Linear(degree + 1, 1)
        nn.init.xavier_uniform_(self.linear.weight)  # Initialize weights

    def forward(self, x):
        x_poly = x ** torch.arange(self.degree + 1, dtype=torch.float32).unsqueeze(0)
        return self.linear(x_poly)

# Initialize the model
degree = 9  # Polynomial degree
model = PolynomialModel(degree)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.14)

# Training the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, t_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plotting the trained model
model.eval()
x_values = torch.linspace(0, 1, 100).unsqueeze(1)
with torch.no_grad():
    y_values = model(x_values)

# Print out coefficients
with torch.no_grad():
    coefficients = model.linear.weight.squeeze().numpy()
    print(f'Fitted Polynomial Coefficients: {coefficients}')
    
plt.figure(figsize=(8, 6))
plt.plot(x_true, y_true, 'g-', label='sin(2πx)')
plt.scatter(x_train, t_train, color='blue', label='Training Data')
plt.plot(x_values, y_values, 'r-', label='Fitted Polynomial')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Polynomial Fitting to sin(2πx)')
plt.legend()
plt.show()
