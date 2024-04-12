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
    weight1 = -0.8
    bias1 = 2
    x = torch.tensor(np.arange(0, 8, 0.2))
    y1 = weight1*x + bias1
    print(type(x), type(y1))
    plt.plot(x, y1,label='w=-0.8 < 0')

    weight2 = 0.3
    bias2 = -3
    y2 = weight2*x + bias2
    plt.plot(x, y2,label='w=0.3 > 0')
    
    plt.title(f'Different weights (gradient, slope, derivative, partial derivative)')
    plt.xlabel('x')
    plt.legend()
    plt.show()
