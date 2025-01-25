import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from findiff import FinDiff

import monte_carlo as mc

# we use equation Erotion rate = - Del (q) where q is sediment flux vector
# the sediment flux vector is a x C x Velocity, where velocity is proportional to gradient of terrain
# C has units of kg/m^3 which is density of water
# a has units of kg/kg for how much mass on target is loss per mass of water

# boundary conditions are constant gradient at max and and min y and neumann at max and min x

# Parameters
stair_width = 200
stair_length = 50 
stair_height = 30


Lx, Ly = stair_width, stair_length * 3 # Domain size
Nx, Ny = int(Lx), int(Ly)    # Number of grid points
alpha = 0.000001         # Weather Errosion
beta = 0.01      # Step Errosion coefficient
T_max = 2.0          # Maximum time
dt = 0.1               # Time step
dx = 1 #Lx / (Nx - 1)   # Grid spacing in x
dy = 1 #Ly / (Ny - 1)   # Grid spacing in y
Nt = int(T_max / dt) # Number of time steps

const = 1 # constant flux
q = -stair_height # constant flux value in cm^-2

# Boundary condition functions
def boundary_conditions(u):
    # Apply custom boundary conditions
    # u[0, :] = u[1, :]  # Zero flux at x = 0
    # u[-1, :] = u[-2, :]  # Zero flux at x = max
    # u[:, 0] = u[:, 1] #+ q * dy  # Constant flux at y = 0
    # u[:, -1] = u[:, -2] #- q * dy  # Constant flux at y = max

    u[0, :] = 0  # Zero flux at x = 0
    u[-1, :] = 0  # Zero flux at x = max
    u[:, 0] = 0 #+ q * dy  # Constant flux at y = 0
    u[:, -1] = 0 #- q * dy  # Constant flux at y = max

    return u

def laplacian(u, dx, dy):
    """
    Compute the Laplacian of u using finite differences.
    """
    laplacian = np.zeros_like(u)
    laplacian[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    return laplacian

def derivatives(f, h=1):
    # Define derivative operators
    d_dx = FinDiff(0, h, 1)  # First derivative with respect to x
    d_dy = FinDiff(1, h, 1)  # First derivative with respect to y
    d2_dx2 = FinDiff(0, h, 2)  # Second derivative with respect to x
    d2_dy2 = FinDiff(1, h, 2)  # Second derivative with respect to y

    # Compute derivatives
    df_dx = d_dx(f)
    df_dy = d_dy(f)

    return np.array([df_dx, df_dy])

# Initial condition: temperature distribution
u = np.zeros((Nx, Ny))
force_map = np.zeros((Nx, Ny))

for i in range(3): 
    u[:,stair_length*i:stair_length*(i+1)] = stair_height*i

force_map[:, stair_length:2*stair_length] = mc.monte_carlo(stair_width, stair_length, 10, mc.custom_pressures)
force_map = force_map / force_map.max()

# u = np.random.random(size=(Nx, Ny))
# u[int(Nx/4):int(3*Nx/4), int(Ny/4):int(3*Ny/4)] = 10.0  # Hot square in the center

def update_grid():
    global u

    # du = alpha * laplacian(u, dx, dy) - beta * force_map
    du = -alpha * np.sqrt((derivatives(u)**2).sum()) - beta * force_map

    du = du*dt
    u_new = u + du
    # Apply constant flux boundary condition at y = 0
    u = boundary_conditions(u_new)

def update(frame):
    global u, force_map

    for i in range(5):
        update_grid()
    force_map[:, stair_length:2*stair_length] = mc.monte_carlo(stair_width, stair_length, 10, mc.custom_pressures)
    force_map = force_map / force_map.max()

    X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
    ax.clear()
    ax.plot_surface(X, Y, np.transpose(u), cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    return [ax]

# Set up the figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([10, 9, 6])

# Create the animation
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=False)

# Display the animation
plt.show()