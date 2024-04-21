import numpy as np
import matplotlib.pyplot as plt
import torch

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x},{self.y})'    

def line(p1, p2):
    # Get x and y coordinates of the two points
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    
    # Compute slope and y-intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    
    def func(x):
        return slope* x + intercept

    return func, slope, intercept

def relu(x): # rectifier Linear Unit (activate function)
    return np.maximum(0, x)

if __name__ == "__main__":
    x = torch.arange(-3,3,0.2)
    f = lambda x: x**3 + x**2/2 - 4*x -2

    x1 = -2.8
    y1 = f(x1)
    p1 = Point(x1, y1)
    x2 = -2.3
    y2 = f(x2)
    p2 = Point(x2, y2)
    f1,slope,intercept = line(p1, p2)
    f1 = lambda x: -slope*x - intercept
    
    x3 = -2
    p1 = Point(x3, f(x3))
    x4 = -1.4
    p2 = Point(x4, f(x4))
    fn, m2, b2 = line(p1, p2)
    def f2(x):
        mask = (x >= x3) & (x < x4)
        y = np.zeros_like(x)
        y[mask] = m2*x[mask] + b2
        return y

    x5 = -1.5
    p1 = Point(x5, f(x5))
    x6 = -0.7
    p2 = Point(x6, f(x6))
    fn, m3, b3 = line(p1, p2)
    print(m3, b3) # -1.0 0.5
    def f3(x):
        mask = (x >= x5) & (x <= x6)
        y = np.zeros_like(x)
        y[mask] = m3*x[mask] + b3
        return y

    plt.plot(x,f(x))
    plt.plot([x1,x2], [y1,y2], 'ro')
    plt.plot(x, -relu(f1(x)) + relu(f2(x)) + relu(f3(x)))
    plt.show()