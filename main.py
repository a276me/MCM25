import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import grid as g

grid = g.setup_grid()


# Define the coordinates where you want to place cubes (x, y, z)
coordinates = g.find_material(grid, 'stairs')

# Define the size of the cubes and the color
cube_size = 1
cube_color = 'cyan'  # You can change this to any color you like

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot cubes at the selected coordinates with the same color
for coord in coordinates:
    x, y, z = coord
    ax.bar3d(x, y, z, cube_size, cube_size, cube_size, color=cube_color)

# Set labels and show the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
