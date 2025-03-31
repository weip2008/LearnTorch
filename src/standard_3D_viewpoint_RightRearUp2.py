import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def main():
    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Draw the coordinate axes ----
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

    # Mark the origin
    ax.scatter(0, 0, 0, color='magenta', s=50)
    ax.text(0, 0, 0, 'Origin(0,0,0)', color='magenta', fontsize=10)

    # ---- Define the 2x4x5 cuboid ----
    # Dimensions: X=2, Y=4, Z=5
    # We specify each face as a list of (x,y,z) corners:
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

    # Build a Poly3DCollection for the cuboid
    cube_faces = Poly3DCollection(
        verts,
        facecolors='cyan',
        edgecolors='k',
        linewidths=1,
        alpha=0.3  # transparency
    )
    ax.add_collection3d(cube_faces)

    # ---- Set the viewpoint and label axes ----
    ax.view_init(elev=30, azim=-60)
    ax.set_xlabel('X(L->R)')
    ax.set_ylabel('Z(DEPTH)')
    ax.set_zlabel('Y(D->U)')
    ax.set_title('3D Cuboid with 1:1:1 Aspect Ratio')

    # ---- Force a 1:1:1 aspect ratio ----
    # For Matplotlib >= 3.3
    ax.set_box_aspect((1, 1, 1))  # Force equal aspect on all axes

    # Optionally set the limits to fit your data nicely
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 6)
    ax.set_zlim(0, 5)

    # If older Matplotlib doesn't have set_box_aspect,
    # uncomment and use the custom function below instead:
    #
    # set_axes_equal(ax)

    plt.tight_layout()
    plt.show()

# Helper function for older Matplotlib versions:
def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

if __name__ == "__main__":
    main()

