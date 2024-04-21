import numpy as np
import matplotlib.pyplot as plt
import torch

x = torch.arange(-3,3,0.2)
f = lambda x: x**3 + x**2/2 - 4*x -2

if __name__ == "__main__":
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(x, f(x))

    # Turn on minor ticks for both axes
    ax.minorticks_on()

    # Set the gridlines for both axes
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


    plt.title("Target Function")
    plt.xlabel('x')
    plt.ylabel('$f(x)=x^3 + x^2/2 - 4x -2$')
    plt.show()
