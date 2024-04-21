import numpy as np
import matplotlib.pyplot as plt
import torch
from linear03 import *

x = torch.arange(-3,3,0.2)
f = lambda x: x**3 + x**2/2 - 4*x -2


if __name__ == "__main__":
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(x, f(x))

    x1 = -2.8
    p1 = (x1, f(x1)) # using tuple to represent a point: x=p1[0], y=p1[1]
    x2 = -2.3
    p2 = (x2, f(x2))
    fn = line(p1, p2)
    y1 = fn(x1)
    y2 = fn(x2)
    ax.plot([x1, x2],[y1, y2], 'ro')
    ax.plot(x[0:7], fn(x)[0:7])

    plt.title("Target Function")
    plt.xlabel('x')
    plt.ylabel('$f(x)=x^3 + x^2/2 - 4x -2$')
    plt.show()
