"""
this python code try to build up a AI modeling concept. given 2 points generate a
line as a model, based on this line, generate random diviation as measurements.
the slope and intercept should be the modeling result.

1. create 2 points (1,2); (5,7)
2. create 1 dimension tensor as input data x
3. use line function to create random output data y 
4. plot y, and line.
5. calculate MSE (Mean Square Error)
"""
import matplotlib.pyplot as plt
import torch
import numpy as np

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'({self.x}, {self.y})'

def line(p1, p2):
    x1,y1 = p1.x, p1.y
    x2,y2 = p2.x, p2.y
    slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope*x1

    def f(x):
        return slope*x + intercept
    
    return f

if __name__ == '__main__':
    p1 = Point(1,2)
    p2 = Point(5,7)

    plt.plot([p1.x, p2.x], [p1.y, p2.y], 'ro') # r: red, o: circle mark

    x = torch.tensor(np.arange(0, 8, 0.2)) # input data, 1 input get 1 output
    print(x.size(), x.ndim)
    print(x)
    f = line(p1, p2)
    plt.plot(x, f(x),label='model')

    y = f(x) + np.random.randn(*x.shape) * 0.5 # random output based on input x
    plt.plot(x,y, 'b^',label='measurement') # b: blue, ^: triangle

    mse = np.square(np.subtract(y, f(x))).mean() # MSE: Mean Square Error
    plt.title(f'MSE={mse}')
    plt.legend()
    plt.show()
