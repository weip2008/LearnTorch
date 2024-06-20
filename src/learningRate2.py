import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create a toy dataset
torch.manual_seed(0)
X = torch.linspace(0, 1, 100).reshape(-1, 1)
y = 3 * X + 2 + 0.1 * torch.randn(X.size())

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# Function to train the model
def train_model(model, optimizer, criterion, X, y, num_epochs=100):
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history, model

# Hyperparameters
learning_rates = [0.0001, 0.01, 0.1]
num_epochs = 500

# Plotting setup for loss curves
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)

# Dictionary to store the final models
final_models = {}

for lr in learning_rates:
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history, trained_model = train_model(model, optimizer, criterion, X, y, num_epochs)
    plt.plot(loss_history, label=f'Learning Rate = {lr}')
    final_models[lr] = trained_model

# Plotting the loss curves
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Effect of Learning Rate on Training Loss')
plt.legend()

# Plotting the final models and the original data
plt.subplot(2, 1, 2)
plt.scatter(X.numpy(), y.numpy(), label='Data')
x_vals = torch.linspace(0, 1, 100).reshape(-1, 1)

for lr, model in final_models.items():
    with torch.no_grad():
        y_vals = model(x_vals)
    plt.plot(x_vals.numpy(), y_vals.numpy(), label=f'LR = {lr}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Final Model Fits with Different Learning Rates')
plt.legend()

plt.tight_layout()
plt.show()
