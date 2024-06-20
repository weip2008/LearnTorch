"""
1. use x.grad get gradient from tensor
2. use df(x) get gradient
"""

import torch

# Define a simple quadratic function f(x) = (x - 2)^2
def f(x):
    return (x - 2) ** 2

def df(x):
    return 2*(x-2)

# Function to perform gradient descent
def gradient_descent(initial_x, learning_rate, num_iterations):
    x = torch.tensor(initial_x, requires_grad=True)
    for i in range(num_iterations):
        # Compute the loss
        loss = f(x)
        
        # Compute gradients (backward pass)
        loss.backward()
        
        print(f'Iteration {i+1}: x = {x.item()}, f(x) = {loss.item()}, grad = {x.grad.item()}')
        
        # Update the variable using gradient descent
        with torch.no_grad():
            # x -= learning_rate * x.grad # calculate gradient by tensor
            x -= learning_rate * df(x) # calculate gradient myself
        
        
        # Zero the gradients after updating
        x.grad.zero_()
        
    return x

# Initial value
initial_x = 10.0

# Number of iterations
num_iterations = 20

# Different learning rates
learning_rates = [0.01, 0.1, 0.5]

# Perform gradient descent with different learning rates
for lr in learning_rates:
    print(f'\nLearning Rate: {lr}')
    gradient_descent(initial_x, lr, num_iterations)