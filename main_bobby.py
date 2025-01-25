import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the size of the steps and the number of steps
step_width = 3  
step_height = 3  
step_depth = 3  
num_steps = 3 

staircase_coordinates = []
for step in range(num_steps):
    for h in range(step_height):  
        for x in range(step_width):  
            for z in range(step_depth):  
                staircase_coordinates.append((x, step * step_depth + z, step * step_height + h))  # (x, y, z)
                
                # Add an extra layer below for h > 0
                if h > 0: 
                    staircase_coordinates.append((x, step * step_depth + z, step * step_height + h - 2))  # (x, y, z-1)

# Create a figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the staircase using bar3d
cube_color = 'cyan'  # Color for the blocks
cube_size = 1
for coord in staircase_coordinates:
    x, y, z = coord
    ax.bar3d(x, y, z, cube_size, cube_size, cube_size, color=cube_color)

# Set labels and view
ax.set_xlabel('X (Width)')
ax.set_ylabel('Y (Depth)')
ax.set_zlabel('Z (Height)')
ax.set_title('3D Continuous Staircase')

ax.set_zlim([0, max(coord[2] for coord in staircase_coordinates) + cube_size])

plt.show()
