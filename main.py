import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import grid as g
import water_flow as w

size = (20,20,7)
grid = g.setup_grid(size)
cube_size = 1

grid = w.add_water(grid)

for i in range(15):
    grid = w.simulate_water(grid)

    #grid = w.evaporate_water(grid)

    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, size[0]])
    ax.set_ylim([0, size[1]])
    ax.set_zlim([0, size[2]])



    stairs = g.find_material(grid, 'stairs')
    stair_color = 'cyan'  

    water = g.find_material(grid, 'water')
    water_color = 'blue'  

    # Plot cubes at the selected coordinates with the same color
    for coord in stairs:
        x, y, z = coord
        ax.bar3d(x, y, z, cube_size, cube_size, cube_size, color=stair_color)

    for coord in water:
        x, y, z = coord
        ax.bar3d(x, y, z, cube_size, cube_size, cube_size, color=water_color)





    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
