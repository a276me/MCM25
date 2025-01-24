import numpy as np
from Cell import Cell


def generate_stairs(x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    """
    Generate a list of 3D coordinates inside a rectangular volume with a specified resolution.
    
    Parameters:
    - x_min, x_max: Min and max values for x-coordinate
    - y_min, y_max: Min and max values for y-coordinate
    - z_min, z_max: Min and max values for z-coordinate
    - resolution: Step size between consecutive coordinates
    
    Returns:
    - List of tuples containing (x, y, z) coordinates
    """
    # Generate the x, y, and z coordinates within the specified boundaries and resolution
    x_coords = np.arange(x_min, x_max + resolution, resolution)
    y_coords = np.arange(y_min, y_max + resolution, resolution)
    z_coords = np.arange(z_min, z_max + resolution, resolution)

    # Generate a list of all (x, y, z) coordinates within the 3D rectangle
    coordinates_3d = [(x, y, z) for x in x_coords for y in y_coords for z in z_coords]
    
    return coordinates_3d

def find_material(grid, mat):

    coords = []

    for x in range(len(grid)):
        for y in range(len(grid[0])):
            for z in range(len(grid[0,0])):
                if grid[x,y,z].material == mat:
                    coords.append((x,y,z))
    return coords

    
def setup_grid(size=(10,10,10)):

    grid = np.empty(shape=size, dtype=object)

    for x in range(len(grid)):
        for y in range(len(grid[0])):
            for z in range(len(grid[0][0])):
                grid[x,y,z] = Cell(1, (x,y,z))

    for c in generate_stairs(0,5,0,5,0,3,1):
        grid[c].change_material('stairs')
    
    return grid

def find_nearby_coordinates(coord, gshape):
    x, y, z = coord

    # Generate all nearby coordinates including the original
    nearby_coords = [
        (x + dx, y + dy, z + dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0, 1]
        if (dx != 0 or dy != 0 or dz != 0)  # Exclude the original coordinate
    ]

    # Filter to include only coordinates where all components are > 0
    valid_coords = [c for c in nearby_coords if c[0] > 0 and c[1] > 0 and c[2] > 0]
    valid_coords = [c for c in valid_coords if c[0] < gshape[0] and c[1] < gshape[1] and c[2] < gshape[2]]


    return valid_coords

def find_nearby_coordinates_same_level(coord, gshape):
    x, y, z = coord

    # Generate all nearby coordinates including the original
    nearby_coords = [
        (x + dx, y + dy, z + dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [ 0 ]
        if (dx != 0 or dy != 0 or dz != 0)  # Exclude the original coordinate
    ]

    # Filter to include only coordinates where all components are > 0
    valid_coords = [c for c in nearby_coords if c[0] > 0 and c[1] > 0 and c[2] > 0]
    valid_coords = [c for c in valid_coords if c[0] < gshape[0] and c[1] < gshape[1] and c[2] < gshape[2]]


    return valid_coords

def find_nearby_lower_coordinates(coord, gshape):
    x, y, z = coord

    # Generate all nearby coordinates
    nearby_coords = [
        (x + dx, y + dy, z + dz)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dz in [-1, 0]  # Only consider lower z-coordinates (dz = -1 or 0)
        if (dx != 0 or dy != 0 or dz != 0)  # Exclude the original coordinate
    ]

    # Filter to include only coordinates where all components are > 0
    valid_coords = [
        c for c in nearby_coords if c[0] > 0 and c[1] > 0 and c[2] > 0 and c[2] < z
    ]
    valid_coords = [c for c in valid_coords if c[0] < gshape[0] and c[1] < gshape[1] and c[2] < gshape[2]]


    return valid_coords
