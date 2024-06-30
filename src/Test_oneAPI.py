import numpy as np
from numba import njit, prange

# Example function that adds two arrays element-wise
@njit(parallel=True)
def vector_add(a, b):
    c = np.empty_like(a)
    for i in prange(a.size):
        c[i] = a[i] + b[i]
    return c

# Example usage
a = np.random.rand(1000000)
b = np.random.rand(1000000)

c = vector_add(a, b)
print(c)
