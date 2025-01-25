import numpy as np
import matplotlib.pyplot as plt

# we use equation Erotion rate = - Del (q) where q is sediment flux vector
# the sediment flux vector is a x C x Velocity, where velocity is proportional to gradient of terrain
# C has units of kg/m^3 which is density of water
# a has units of kg/kg for how much mass on target is loss per mass of water

# boundary conditions are constant gradient

# Parameters
Lx, Ly = 10.0, 10.0  # Domain size
Nx, Ny = 50, 50      # Number of grid points
alpha = 0.01         # Thermal diffusivity
T_max = 2.0          # Maximum time
dt = 0.01            # Time step
dx = Lx / (Nx - 1)   # Grid spacing in x
dy = Ly / (Ny - 1)   # Grid spacing in y
Nt = int(T_max / dt) # Number of time steps

# Initial condition: temperature distribution
u = np.zeros((Nx, Ny))
u[int(Nx/4):int(3*Nx/4), int(Ny/4):int(3*Ny/4)] = 100.0  # Hot square in the center

# Time-stepping loop
for n in range(Nt):
    u_new = u.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 +
                (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
            )
    u = u_new

    # Visualization every 100 time steps
    if n % 100 == 0:
        plt.imshow(u, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Temperature')
        plt.title(f'Temperature Distribution at t = {n*dt:.2f}')
        plt.show()




