import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def weighted_moving_average(prices, weights):
    return np.convolve(prices, weights, mode='valid')

# 示例股票价格序列
stock_prices = np.array([100, 102, 101, 105, 110, 108, 107, 111, 115, 114])

# 定义加权向量（最近价格权重较大）
weights = np.array([0.1, 0.2, 0.3, 0.5])  # 总和应该等于1或不要求

# 计算加权移动平均
smoothed_prices = weighted_moving_average(stock_prices, weights)

# 设置中文字体
font = FontProperties(fname='C:/windows/fonts/SIMLI.TTF', size=14)

# 绘制图表
plt.figure(figsize=(10, 5))
plt.plot(stock_prices, label='原始股票价格')
plt.plot(np.arange(len(weights)-1, len(stock_prices)), smoothed_prices, label='平滑后的股票价格')
plt.legend(prop=font)
plt.xlabel('时间', fontproperties=font)
plt.ylabel('价格', fontproperties=font)
plt.title('股票价格的加权移动平均平滑', fontproperties=font)
plt.show()
