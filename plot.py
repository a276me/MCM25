import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
stair_width = 30
stair_height = 30
Lx, Ly = 100.0, stair_width*3  # Domain size
Nx, Ny = int(Lx), int(Ly)    # Number of grid points
alpha = 0.01         # Thermal diffusivity
T_max = 2.0          # Maximum time
dt = 1               # Time step
dx = 1 #Lx / (Nx - 1)   # Grid spacing in x
dy = 1 #Ly / (Ny - 1)   # Grid spacing in y
Nt = int(T_max / dt) # Number of time steps

const = 1 # constant flux
q = -1 # constant flux value

# Initial condition: temperature distribution
u = np.zeros((Nx, Ny))

for i in range(3): 
    u[:,stair_width*i:stair_width*(i+1)] = stair_height*i


data = u
# Generate X and Y coordinates
X = np.arange(data.shape[1])
Y = np.arange(data.shape[0])
X, Y = np.meshgrid(X, Y)

# Create a figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for x, xs in enumerate(data):
    for y, ys in enumerate(xs):
        ax.bar3d(x, y, 0, 1, 1, data[x][y], shade=True)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot with Solid Volume Beneath')

# Show the plot
plt.show()