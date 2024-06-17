import torch

# Define a simple quadratic function f(x) = (x - 2)^2
def f(x):
    return (x - 2) ** 2

# Initialize a tensor with requires_grad=True to track computation
x = torch.tensor(10.0, requires_grad=True)

# Define a learning rate
learning_rate = 0.1

# Perform gradient descent for a few iterations
for i in range(20):
    # Compute the loss
    loss = f(x)
    
    # Compute gradients (backward pass)
    loss.backward()
    
    # Print the gradient
    print(f'Iteration {i+1}: x = {x.item()}, f(x) = {loss.item()}, grad = {x.grad.item()}')
    
    # Update the variable using gradient descent
    with torch.no_grad():
        x -= learning_rate * x.grad
    
    # Zero the gradients after updating
    x.grad.zero_()