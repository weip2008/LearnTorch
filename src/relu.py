"""
here relu is an activate function, 
which makes it possible to fit any function using many short straight lines.
"""
import numpy as np
import matplotlib.pyplot as plt

# relu activate function(ReLU: Rectified Linear Unit) can activate part of the linear equation, deactivate other part of the line equation
relu = lambda x: np.maximum(0,x)
relu2 = lambda x: np.minimum(10, x)
relu3 = lambda x: np.where(x == 10, 0, x)

x = np.arange(-2,10,0.4)
fx1 = 3* x
fx2 = fx1 - 4
fx3 = relu(fx2)
fx4 = fx1 + 4
fx5 = relu3(relu2(relu(fx2)))

fig,ax = plt.subplots()

ax.plot(x, fx1, label='y=3x')
ax.plot(x, fx2, 'bo', label='y=3x-4')
ax.plot(x, fx3, label='relu(3x-4)')
ax.plot(x, fx4, label='y=3x+4')
ax.plot(x, fx5, 'p--',label='relu3(relu2(relu(3x-4)))')

ax.text(2.2, 0, "bias=-4 as shifting line to the right")

plt.title("Bias helps ReLU to decide which part of the line is activated")
plt.legend()
plt.show()