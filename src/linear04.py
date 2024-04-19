import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        # Reshape x to have shape (batch_size, 1)
        x = x.reshape(-1, 1)
        return self.linear(x)

if __name__ == '__main__':
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv("data/death.csv")
    # Print the dataframe to check for NaN values
    print(df.isnull().sum())
    # Convert data type to float32
    df['Recent 5-Year Trend (2) in Death Rates'] = pd.to_numeric(df['Recent 5-Year Trend (2) in Death Rates'], errors='coerce').astype(np.float32)

    # Filter out NaN values from the 'Recent 5-Year Trend (2) in Death Rates' column
    df = df.dropna(subset=['Recent 5-Year Trend (2) in Death Rates'])    

    # Filter x to only include matching indices
    df = df.reset_index(drop=True)
    x = df.index.astype(np.float32)
    y = df['Recent 5-Year Trend (2) in Death Rates'].astype(np.float32)

    # Create a PyTorch DataLoader for the dataset
    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer
    model = LinearRegression()
    optimizer = optim.SGD(model.parameters(), lr=1e-8)

    # Train the model, when the loop finished, find minimum of all try
    num_epochs = 1000
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(torch.float32)
            y_batch = y_batch.to(torch.float32)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.functional.mse_loss(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
            if loss.item()<0.5:
                break
    slope = model.linear.weight.item()
    intercept = model.linear.bias.item()
    print(f"slop: {slope}, intercept: {intercept}")

    # Plot the results
    x = x.astype(np.float32)
    x_tensor = torch.from_numpy(x.to_numpy()).float()
    y_tensor = model(x_tensor).detach().numpy()
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, slope*x + intercept, label='Fitted line')
    plt.title(f"slope: {slope}, intercept: {intercept}")
    plt.legend()
    plt.show()
