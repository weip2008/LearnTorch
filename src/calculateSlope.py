from turtle import *
from shapes import *
import numpy as np
import math

def calculate_slope_intercept(p1,p2):
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1]-slope*p1[0]
    return slope, intercept

def line(slope, intercept):
    def f(x):
        return slope*x + intercept
    return f

if __name__ == '__main__':
    pen1 = Turtle()

    drawLine2(pen1, (-200-20, 0), ((200+20),0)) # draw x axis
    drawLine2(pen1, (0,-200-20), (0, 200+20)) # draw y axis
    drawText(pen1, 230, 2, 'x')
    drawText(pen1, 0, 223, 'y')

    p1 = 3, 8
    p2  = 5, 10
    slope, intercept = calculate_slope_intercept(p1, p2)
    f = line(slope, intercept)

    p3 = -100, f(-100)
    p4 = 150, f(150)
    pen1.color('red')
    drawLine2(pen1, p3, p4) # draw the line
    pen1.color('black')

    x0 = 50
    dx = 40
    p5 = (x0, f(x0))
    p6 = (x0+dx, f(x0))
    drawLine2(pen1, p5, p6) # draw dx

    p7 = (x0+dx, f(x0+dx))
    drawLine2(pen1, p6, p7) # draw dy

    drawText(pen1, 20, 2, 'θ=arctang(dy/dx)')
    drawText(pen1, 95, 70, 'dy')
    drawText(pen1, 70, 40, 'dx')
    drawText(pen1, 150, 120, 'slope = dy/dx = (95-55) / (90-50)')
    drawText(pen1, p5[0]-50, p5[1],f'{p5}')
    drawText(pen1, p7[0], p7[1],f'{p7}')

    pen1.hideturtle()

    mainloop()