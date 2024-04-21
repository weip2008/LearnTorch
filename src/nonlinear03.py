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

    return slope, intercept


if __name__ == "__main__":
    x = torch.arange(-3,3,0.2)
    f = lambda x: x**3 + x**2/2 - 4*x -2

    x1 = -2.8
    p1 = Point(x1, f(x1))
    x2 = -2.3
    p2 = Point(x2, f(x2))
    w1, b1 = line(p1, p2)
    print(w1, b1) # 13.019999999999996 27.62399999999999
    f1 = lambda x: w1*x + b1 

    x3 = -2.1
    p1 = Point(x3, f(x3))
    x4 = -1.5
    p2 = Point(x4, f(x4))
    w2, b2 = line(p1, p2)
    print(w2, b2) # 5.09 9.385
    f2 = lambda x: w2*x + b2 

    x5 = -1.5
    p1 = Point(x5, f(x5))
    x6 = -1.
    p2 = Point(x6, f(x6))
    w3, b3 = line(p1, p2)
    print(w3, b3) # -0.5 1.0
    f3 = lambda x: w3*x + b3 

    x7 = -1.
    p1 = Point(x7, f(x7))
    x8 = .8
    p2 = Point(x8, f(x8))
    w4, b4 = line(p1, p2)
    print(w4, b4) # -3.0 -1.5
    f4 = lambda x: w4*x + b4 

    x9 = .8
    p1 = Point(x9, f(x9))
    x10 = 1.2
    p2 = Point(x10, f(x10))
    w5, b5 = line(p1, p2)
    print(w5, b5) # 3.440000000000001 -7.940000000000001
    f5 = lambda x: w5*x + b5 

    x11 = 1.2
    p1 = Point(x11, f(x11))
    x12 = 1.6
    p2 = Point(x12, f(x12))
    w6, b6 = line(p1, p2)
    print(w6, b6) # 12.879999999999997 -24.931999999999995
    f6 = lambda x: w6*x + b6 

    x13 = 1.8
    p1 = Point(x13, f(x13))
    x14 = 2.6
    p2 = Point(x14, f(x14))
    w7, b7 = line(p1, p2)
    print(w6, b6) # 12.879999999999997 -24.931999999999995
    f7 = lambda x: w7*x + b7 

    fCombine = lambda x: f1(x) + f2(x) + f3(x) + f4(x) + f5(x) + f6(x) + f7(x)

    plt.plot(x,f(x), label='target function')
    plt.plot(x[0:6], f1(x)[0:6], label='f1')
    plt.plot(x[3:9], f2(x)[3:9], label='f2')
    plt.plot(x[7:12], f3(x)[7:12], label='f3')
    plt.plot(x[10:20], f4(x)[10:20], label='f4')
    plt.plot(x[18:22], f5(x)[18:22], label='f5')
    plt.plot(x[20:25], f6(x)[20:25], label='f6')
    plt.plot(x[22:], f7(x)[22:], label='f7')
    
    # plt.plot(x, fCombine(x), label='Combination')

    plt.title("Target Function")
    plt.xlabel('x')
    plt.ylabel('$f(x)=x^3 + x^2/2 - 4x -2$')
    plt.legend()
    plt.show()
