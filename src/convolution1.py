def manual_convolution(f, g):
    N = len(f)
    M = len(g)
    g_flipped = g[::-1]
    
    # 卷积结果的长度
    conv_length = N + M - 1
    result = [0] * conv_length
    
    # 计算卷积
    for i in range(conv_length):
        for j in range(M):
            if i - j >= 0 and i - j < N:
                result[i] += f[i - j] * g_flipped[j]
    
    return result

# 定义两个序列
f = [1, 2, 3]
g = [0, 1, 0.5]

# 计算卷积
convolution_result = manual_convolution(f, g)

# 输出结果
print("f:", f)
print("g:", g)
print("卷积结果:", convolution_result)