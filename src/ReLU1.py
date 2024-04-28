import matplotlib.pyplot as plt
import numpy as np
import math

relu = lambda x: np.maximum(0,x) # this function activate the value of x>0, 

x = np.arange(-4*math.pi, 4*math.pi, 0.2)# input

y = -3*x -4

plt.plot(x, relu(y), 'ro', label='ReLU function') 
plt.plot(x, y, label='Original function')

plt.title("Activate Function ReLU")
plt.legend()
plt.show()