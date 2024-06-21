"""
Demo of Gradient Descending
"""
import torch

# Define a simple quadratic function f(x) = (x - 2)^2
def f(x):
    return (x - 2) ** 2 # g=x-2, f(g)=g**2, f'(x)=f'(g)g'(x)

def df(x):
    return 2*(x-2)

x0 = 20.0
# Initialize a tensor with requires_grad=True to track computation
x = torch.tensor(x0, requires_grad=True)

# Define a learning rate
learning_rate = 0.1

# Perform gradient descent for a few iterations
for i in range(50):
    # Compute the loss
    loss = f(x)
    
    # Compute gradients (backward pass)
    loss.backward()
    
    # Print the gradient
    print(f'Iteration {i+1}: x = {x.item()}, f(x) = {loss.item()}, grad = {x.grad.item()}')
    
    # Update the variable using gradient descent
    with torch.no_grad():
        # x -= learning_rate * x.grad
        x -= learning_rate * df(x)
    
    # Zero the gradients after updating
    x.grad.zero_()