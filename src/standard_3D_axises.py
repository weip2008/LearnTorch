"""

This code creates a 3D plot that visually represents a standard three-dimensional coordinate system

标准三维笛卡尔坐标系模板
定义：
- X轴：水平，左→右
- Y轴：垂直，下→上
- Z轴：深度，外→内（屏幕方向）

视角设定：从右上前方看入，使三轴方向直观符合视觉直觉
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



# 使用标准三维坐标系，从 (0, 0, 0) 开始画，保持比例不变，取消自动轴居中
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 坐标轴方向向量
origin = np.array([0, 0, 0])
x_dir = np.array([1, 0, 0])  # X轴：左右
y_dir = np.array([0, 0, 1])  # Y轴：上下
z_dir = np.array([0, 1, 0])  # Z轴：深度

length = 2.5
ax.quiver(*origin, *x_dir, length=length, color='red')
ax.text(length + 0.2, 0, 0, 'X(L->R)', color='red', fontsize=12)
ax.quiver(*origin, *y_dir, length=length, color='blue')
ax.text(0, 0, length + 0.2, 'Y(D->U)', color='blue', fontsize=12)
ax.quiver(*origin, *z_dir, length=length, color='green')
ax.text(0, length + 0.2, 0, 'Z(DEPTH)', color='green', fontsize=12)

# 添加原点可视标记
ax.scatter(0, 0, 0, color='magenta', s=50)
ax.text(0, 0, 0, 'Origin(0,0,0)', color='magenta', fontsize=10)

# 设置视角
ax.view_init(elev=30, azim=-60)

# 坐标轴范围（从原点出发，不用 set_axes_equal）
ax.set_xlim(0, 3)
ax.set_ylim(0, 6)
ax.set_zlim(0, 5)

ax.set_xlabel('X(L->R)')
ax.set_ylabel('Z(DEPTH)')
ax.set_zlabel('Y(D->U)')
ax.set_title('Standard 3D ')

plt.tight_layout()
plt.show()