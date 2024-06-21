"""
这个结果和numpy算出来的一样。
"""
def convolution(f, g):
    # Lengths of the input sequences
    N = len(f)
    M = len(g)
    
    # Initialize the result with zeros
    result = [0] * (N + M - 1)
    
    # Perform the convolution operation
    for n in range(N + M - 1):
        result[n] = sum(f[k] * g[n - k] for k in range(N) if 0 <= n - k < M)
    
    return result

# Define the sequences
f = [1, 2, 3]
g = [0, 1, 0.5]

# Calculate the convolution
result = convolution(f, g)

# Print the result
print(result)
