import matplotlib.pyplot as plt
import numpy as np
import math

# relu = lambda y: np.maximum(0,y) # this function activate the value of x>0, relu: Rectifier Linear Unit
relu = lambda y: np.minimum(0,y) # this function activate the value of x>0, relu: Rectifier Linear Unit

x = np.arange(-4*math.pi, 4*math.pi, 0.2)

y = np.sin(x)

plt.plot(x, relu(y))
plt.plot(x, y, 'ro')

plt.title("Activate Function ReLU")
plt.show()