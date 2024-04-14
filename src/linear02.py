import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 1 input and 1 output
        
    def forward(self, x):
        return self.linear(x) # w*x + b

if __name__ == '__main__':
    slope1 = 3 # weight
    intercept1 = 10 # bias
    # Create a tensor representing the linear function
    x = torch.tensor(np.arange(0, 11), dtype=torch.float32) # input
    y = slope1 * x + intercept1 # desired model

    # try to use the x, y to find w, b

    x_train = x.numpy().reshape(11,1) # create input data
    y_train = y.numpy().reshape(11,1) # desired output data from our model

    # Create a dataset with some noise
    y_train += np.random.randn(*y_train.shape) * 1 # measured output, use as our training output data, pretent we don't know the model

    # Create a PyTorch DataLoader for the dataset
    batch_size = 5
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    model = LinearRegression()
    optimizer = optim.SGD(model.parameters(), lr=0.01)   #  Stochastic Gradient Descent, lr: Learning Rate

    # Train the model
    num_epochs = 300
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.functional.mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    slope = model.linear.weight.item()
    intercept = model.linear.bias.item()
    print(f"slop: {slope}, intercept: {intercept}")

    # Plot the results
    plt.plot(x_train, y_train, 'ro', label='Random data')
    plt.plot(x_train, model(torch.from_numpy(x_train)).detach().numpy(), label='Fitted line')
    plt.title(f"weight: {slope}, bias: {intercept}")
    plt.legend()
    plt.show()

    x = 13.0
    # Predict the output for x = 13
    x = torch.tensor([x], dtype=torch.float32)
    y_pred = model(x).item()

    print(f'Predict value: {y_pred}, Actual value: {13*slope1 + intercept1}')