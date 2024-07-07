import numpy as np
import time
import numba

# Matrix dimensions
N = 1000

# Generate random matrices
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

# Matrix multiplication without Numba
start_time = time.time()
C_np = np.dot(A, B)
end_time = time.time()
time_without_numba = end_time - start_time

print(f"Time taken without Numba: {time_without_numba:.4f} seconds")

# Matrix multiplication with Numba
@numba.njit(parallel=True, fastmath=True)
def matrix_multiply(A, B):
    N = A.shape[0]
    C = np.zeros((N, N), dtype=np.float32)
    for i in numba.prange(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

start_time = time.time()
C_numba = matrix_multiply(A, B)
end_time = time.time()
time_with_numba = end_time - start_time

print(f"Time taken with Numba: {time_with_numba:.4f} seconds")

# Verify the results are the same
assert np.allclose(C_np, C_numba), "The results are not the same!"

print(f"Speedup: {time_without_numba / time_with_numba:.2f}x")

print("================================================")

start_time = time.time()
C_np = np.dot(A, B)
end_time = time.time()
time_without_numba = end_time - start_time

start_time = time.time()
C_numba = matrix_multiply(A, B)
end_time = time.time()
time_with_numba = end_time - start_time

print(f"Time taken with Numba: {time_with_numba:.4f} seconds")


print("================================================")

start_time = time.time()
C_np = np.dot(A, B)
end_time = time.time()
time_without_numba = end_time - start_time

start_time = time.time()
C_numba = matrix_multiply(A, B)
end_time = time.time()
time_with_numba = end_time - start_time

print(f"Time taken with Numba: {time_with_numba:.4f} seconds")

