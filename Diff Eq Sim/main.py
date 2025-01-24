import numpy as np
import matplotlib.pyplot as plt

# Variables
stair_height = 20
stair_width = 80
stair_depth = 20
num_of_stairs = 3
top_left_coord = (10)

# Setup Grid
space = np.zeros(shape=(stair_depth*num_of_stairs+2, stair_width+2))

# Boundary Conditions
space[0,:] = 0
space[-1,:] = 0
space[:,0] = 0
space[:,-1] = 0

space[1:stair_depth+1, 1:stair_width+1] = 1

plt.imshow(space)
plt.show()

