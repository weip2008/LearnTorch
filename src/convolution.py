import numpy as np

# 定义两个序列
f = np.array([1, 2, 3])
g = np.array([0, 1, 0.5])

# 计算卷积
convolution_result = np.convolve(f, g)

# 输出结果
print("f:", f)
print("g:", g)
print("卷积结果:", convolution_result)
