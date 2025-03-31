import matplotlib.pyplot as plt
import numpy as np

def main():
    # Create the figure and 3D axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # ---- Draw the custom coordinate axes ----
    # Using your original directions:
    #   X: left → right: [1, 0, 0]
    #   Y: down → up:    [0, 0, 1]
    #   Z: DEPTH:        [0, 1, 0]
    origin = np.array([0, 0, 0])
    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 0, 1])
    z_dir = np.array([0, 1, 0])
    length = 2.5

    ax.quiver(*origin, *x_dir, length=length, color='red')
    ax.text(length + 0.2, 0, 0, 'X (L→R)', color='red', fontsize=12)

    ax.quiver(*origin, *y_dir, length=length, color='blue')
    ax.text(0, 0, length + 0.2, 'Y (D→U)', color='blue', fontsize=12)

    ax.quiver(*origin, *z_dir, length=length, color='green')
    ax.text(0, length + 0.2, 0, 'Z (DEPTH)', color='green', fontsize=12)

    # Mark the origin
    ax.scatter(0, 0, 0, color='magenta', s=50)
    ax.text(0, 0, 0, 'Origin', color='magenta', fontsize=10)

    # ---- Draw parabolic curves ----
    # Parameter for the curves
    t = np.linspace(-2, 2, 200)
    
    # Curve 1: Parabola in the X–Z plane (with constant Y = 0)
    # Here, we interpret the data as (x, z, y)
    x1 = t
    z1 = t**2   # quadratic in DEPTH direction (Z axis)
    y1 = np.zeros_like(t)
    ax.plot3D(x1, z1, y1, color='magenta', linewidth=2, label='Parabola in X–Z plane')

    # Curve 2: Parabola in the X–Y plane (with constant Z = 0)
    # Data interpreted as (x, z, y): here z (DEPTH) is constant = 0, and y varies quadratically.
    x2 = t
    z2 = np.zeros_like(t)
    y2 = t**2   # quadratic in vertical direction (Y axis)
    ax.plot3D(x2, z2, y2, color='orange', linewidth=2, label='Parabola in X–Y plane')

    ax.legend()

    # ---- Adjust view and scaling ----
    ax.set_box_aspect((1, 1, 1))  # Equal aspect ratio
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 5)   # DEPTH (Z axis)
    ax.set_zlim(-1, 5)   # Y (vertical)

    # Set labels to match your custom axes
    ax.set_xlabel('X (L→R)')
    ax.set_ylabel('Z (DEPTH)')
    ax.set_zlabel('Y (D→U)')
    ax.set_title('Parabolic Curves in Custom 3D Coordinate System')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
