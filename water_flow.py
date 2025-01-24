import numpy as np
import grid as g
import random as r

def add_water(grid): # adds 1 layer of water to top layer of grid
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            grid[x,y,-1].change_material('water')

    return grid

def simulate_water(grid):
    newgrid = np.copy(grid)
    waters = g.find_material(grid, 'water')
    for c in waters:
        moved = False
        # make water fall straight down
        if grid[c[0], c[1], c[2]-1].material == 'air' and newgrid[c[0], c[1], c[2]-1].material == 'air' and c[2]-1 >= 0:
            newgrid[c[0], c[1], c[2]-1].material ='water'
            newgrid[c[0], c[1], c[2]].material = 'air'
            moved = True
        
        # make water flow into nearby lower cells
        if len(g.find_nearby_lower_coordinates(c, grid.shape)) > 0 and moved==False:
            coords = g.find_nearby_lower_coordinates(c, grid.shape)
            r.shuffle(coords)
            for i in coords:
                if newgrid[i].material == 'air' and grid[i].material == 'air':
                    newgrid[i].material = 'water'
                    newgrid[c].material = 'air'
                    moved = True
                    break

        # make water flow into other same level cells
        # if c == (4,4,4): print('asd')
        if len(g.find_nearby_coordinates_same_level(c, grid.shape)) > 0 and moved == False:
            coords = g.find_nearby_coordinates_same_level(c, grid.shape)
            
            if len(coords) == 0: break
            r.shuffle(coords)
            for i in coords:
                if i in waters: continue
                if newgrid[i].material == 'air' and grid[i].material == 'air':
                    #print(f'moved {c} to {i}')
                    newgrid[i].material = 'water'
                    newgrid[c].material = 'air'
                    moved = True
                    break
        
        if moved:
            for i in g.find_nearby_coordinates_same_level(c, grid.shape):
                grid[i].calculate_erosion(0.5)
            

        # if c == (4,4,4):
        #     print(g.find_nearby_lower_coordinates(c, grid.shape))
        #     print(g.find_nearby_coordinates_same_level(c, grid.shape))




    return newgrid

def evaporate_water(grid):
    print(g.find_material(grid, 'water'))
    for c in g.find_material(grid, 'water'):
        grid[c[0], c[1], c[2]].change_material('air')
            
    return grid
