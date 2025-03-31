import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def main():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Draw the coordinate axes (using your original directions) ----
    origin = np.array([0, 0, 0])
    # Original axes directions:
    #   X: left -> right: [1, 0, 0]
    #   Y: down -> up: [0, 0, 1]
    #   Z: DEPTH:   [0, 1, 0]
    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 0, 1])
    z_dir = np.array([0, 1, 0])
    
    length = 2.5
    ax.quiver(*origin, *x_dir, length=length, color='red')
    ax.text(length + 0.2, 0, 0, 'X (L->R)', color='red', fontsize=12)
    
    ax.quiver(*origin, *y_dir, length=length, color='blue')
    ax.text(0, 0, length + 0.2, 'Y (D->U)', color='blue', fontsize=12)
    
    ax.quiver(*origin, *z_dir, length=length, color='green')
    ax.text(0, length + 0.2, 0, 'Z (DEPTH)', color='green', fontsize=12)

    # Mark the origin
    ax.scatter(0, 0, 0, color='magenta', s=50)
    ax.text(0, 0, 0, 'Origin (0,0,0)', color='magenta', fontsize=10)

    # ---- Define the cuboid vertices ----
    # IMPORTANT: In this coordinate system the axis order is:
    #   1st coordinate: X (L->R)
    #   2nd coordinate: Z (DEPTH)  <-- from z_dir = [0,1,0]
    #   3rd coordinate: Y (D->U)   <-- from y_dir = [0,0,1]
    #
    # To get:
    #   X dimension: 0 to 2,
    #   Z dimension: 0 to 5 (DEPTH),
    #   Y dimension: 0 to 4 (height)
    #
    # We define the vertices in the order (x, z, y):
    verts = [
        # Bottom face (Y = 0)
        [(0, 0, 0), (2, 0, 0), (2, 5, 0), (0, 5, 0)],
        # Top face (Y = 4)
        [(0, 0, 4), (2, 0, 4), (2, 5, 4), (0, 5, 4)],
        # Front face (Z = 0)
        [(0, 0, 0), (2, 0, 0), (2, 0, 4), (0, 0, 4)],
        # Back face (Z = 5)
        [(0, 5, 0), (2, 5, 0), (2, 5, 4), (0, 5, 4)],
        # Left face (X = 0)
        [(0, 0, 0), (0, 5, 0), (0, 5, 4), (0, 0, 4)],
        # Right face (X = 2)
        [(2, 0, 0), (2, 5, 0), (2, 5, 4), (2, 0, 4)]
    ]

    # ---- Create and add the cuboid as an outline (wireframe) ----
    # Instead of using facecolors='none', we set facecolors to fully transparent.
    cube_outline = Poly3DCollection(verts,
                                    facecolors=(0, 0, 0, 0),  # Fully transparent fill
                                    edgecolors='k',           # Black edges
                                    linewidths=2)
    ax.add_collection3d(cube_outline)

    # ---- Set the aspect ratio and axis limits ----
    # Force a 1:1:1 aspect ratio
    ax.set_box_aspect((1, 1, 1))
    # Set axis limits:
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 6)
    ax.set_zlim(0, 5)

    # Label axes (keeping your original labels)
    ax.set_xlabel('X (L->R)')
    ax.set_ylabel('Z (DEPTH)')
    ax.set_zlabel('Y (D->U)')
    ax.set_title('3D Cuboid Outline (X=2, Y=4, Z=5)')

    # Set the view angle
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
