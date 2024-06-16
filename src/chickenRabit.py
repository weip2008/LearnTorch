"""
假设有一个笼子里有若干只鸡和兔子。已知鸡有2只脚，兔子有4只脚。通过统计，
笼子里一共有10个头和28只脚。我们需要确定鸡和兔子的数量。
"""

import numpy as np

# 定义矩阵 A 和向量 B
A = np.array([[1, 1],
              [2, 4]])
B = np.array([10, 28])

# 计算逆矩阵 A 的逆
A_inv = np.linalg.inv(A)

print(A_inv)

# 求解 X
X = np.dot(A_inv, B)

print("鸡的数量:", int(X[0]))
print("兔子的数量:", int(X[1]))
