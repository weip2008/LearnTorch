import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 1) Set up the figure and 3D axes
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 2) Draw the original coordinate axes
origin = np.array([0, 0, 0])
x_dir = np.array([1, 0, 0])  # X-axis: left->right
y_dir = np.array([0, 0, 1])  # Y-axis: down->up
z_dir = np.array([0, 1, 0])  # Z-axis: depth

length = 2.5
ax.quiver(*origin, *x_dir, length=length, color='red')
ax.text(length + 0.2, 0, 0, 'X(L->R)', color='red', fontsize=12)
ax.quiver(*origin, *y_dir, length=length, color='blue')
ax.text(0, 0, length + 0.2, 'Y(D->U)', color='blue', fontsize=12)
ax.quiver(*origin, *z_dir, length=length, color='green')
ax.text(0, length + 0.2, 0, 'Z(DEPTH)', color='green', fontsize=12)

ax.scatter(0, 0, 0, color='magenta', s=50)
ax.text(0, 0, 0, 'Origin(0,0,0)', color='magenta', fontsize=10)

# 3) Define the vertices for the rectangular cuboid (0->2 in X, 0->4 in Y, 0->5 in Z)
#    Each face is a list of four (x,y,z) corner points.
verts = [
    # Bottom face (z=0)
    [(0, 0, 0), (2, 0, 0), (2, 4, 0), (0, 4, 0)],
    # Top face (z=5)
    [(0, 0, 5), (2, 0, 5), (2, 4, 5), (0, 4, 5)],
    # Front face (y=0)
    [(0, 0, 0), (2, 0, 0), (2, 0, 5), (0, 0, 5)],
    # Back face (y=4)
    [(0, 4, 0), (2, 4, 0), (2, 4, 5), (0, 4, 5)],
    # Left face (x=0)
    [(0, 0, 0), (0, 4, 0), (0, 4, 5), (0, 0, 5)],
    # Right face (x=2)
    [(2, 0, 0), (2, 4, 0), (2, 4, 5), (2, 0, 5)]
]

# 4) Create a 3D polygon collection of those faces
cube_faces = Poly3DCollection(verts, 
                              facecolors='cyan',  # Face color
                              edgecolors='k',     # Edge color
                              linewidths=1,
                              alpha=0.3)          # Transparency
ax.add_collection3d(cube_faces)

# 5) Adjust the viewpoint and axis limits
ax.view_init(elev=30, azim=-60)
ax.set_xlim(0, 3)
ax.set_ylim(0, 6)
ax.set_zlim(0, 5)

ax.set_xlabel('X(L->R)')
ax.set_ylabel('Z(DEPTH)')
ax.set_zlabel('Y(D->U)')
ax.set_title('Standard 3D with a 2x4x5 Cuboid')

plt.tight_layout()
plt.show()
