import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from findiff import FinDiff
from matplotlib.animation import FuncAnimation
import random

import monte_carlo as mc

# we use equation Erotion rate = - Del (q) where q is sediment flux vector
# the sediment flux vector is a x C x Velocity, where velocity is proportional to gradient of terrain
# C has units of kg/m^3 which is density of water
# a has units of kg/kg for how much mass on target is loss per mass of water

# boundary conditions are constant gradient at max and and min y and neumann at max and min x



# Boundary condition functions

def laplacian(u, dx, dy):
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

class ModelI:

    def __init__(self, alpha=1, beta=0.1, stair_dim=(200, 50, 30), direction=1, n_peaks=2, step_frequency=5, dt=0.1, randomize_per_dt=5, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.stair_width, self.stair_length, self.q = stair_dim
        self.dx = 1 
        self.dy = 1 
        self.dt = dt
        self.step_frequency = step_frequency
        self.direction = direction
        self.peaks = n_peaks
        self.randomize_per_dt = randomize_per_dt
        self.dt_till_random = 0
    
    def renormal_func(self, u):
        return u*np.exp(-1*(0.1*u)**2)

    def update_grid(self,):
        
        # print(force_map.shape)
        # print(self.u.shape)
        #print(laplacian(self.u, self.dx, self.dy).std())
        # force_map[:, self.stair_length:2*self.stair_length] = mc.monte_carlo(self.stair_width,
        #                                                      self.stair_length, self.steps_per_dt, self.direction, self.distribution)
        
        if self.dt_till_random <= 0:
            self.force_map = np.zeros((self.Nx, self.Ny))
            self.force_map[:,self.stair_length:2*self.stair_length] = mc.monte_carlo(self.stair_width,
                                    self.stair_length, 500, self.direction, self.peaks) / 500
            # self.force_map += 0.1*np.random.rand(self.Nx, self.Ny)
            self.dt_till_random = self.randomize_per_dt
 
        du = self.alpha * self.renormal_func(laplacian(self.u, self.dx, self.dy)) - self.beta * self.force_map * self.step_frequency
        #du += 0.01*np.random.rand(self.Nx, self.Ny)
        # du = -alpha * np.sqrt((derivatives(u)**2).sum()) - beta * force_map

        du = du*self.dt
        self.u = self.u + du

        self.apply_boundary_conditions()
        self.dt_till_random -= 1
        
    def apply_boundary_conditions(self,):
        # Apply custom boundary conditions
        self.u[0, :] = self.u[1, :]  # Zero flux at x = 0
        self.u[-1, :] = self.u[-2, :]  # Zero flux at x = max
        self.u[:, 0] = self.u[:, 1] - self.q * self.dy  # Constant flux at y = 0
        self.u[:, -1] = self.u[:, -2] + self.q * self.dy  # Constant flux at y = max

        # self.u[0, :] = self.u[1, :]  # Zero flux at x = 0 
        # self.u[-1, :] = self.u[-2, :]  # Zero flux at x = max 
        # self.u[:, 0] = 0
        # self.u[:, -1] = 3*self.q


    def run_simulation(self, iterations, save=False, save_file='./DiffEqSim/sim.npy'):

        self.randomize_per_dt = int(iterations / 10)

        self.Lx, self.Ly = self.stair_width, self.stair_length*3 # Domain size
        self.Nx, self.Ny = int(self.Lx), int(self.Ly) 
        self.u = np.zeros((self.Nx, self.Ny))
        Y, X = np.meshgrid( np.linspace(0, self.Ly, self.Ny), np.linspace(0, self.Lx, self.Nx),)

        for i in range(3): 
            self.u[:,self.stair_length*i:self.stair_length*(i+1)] = self.q*i

        
        self.saves = np.zeros(shape=(iterations, 3, self.u.shape[0], self.u.shape[1]))
        self.saves[:,1,:,:] = X
        self.saves[:,2,:,:] = Y

        for i in range(iterations):
            self.saves[i][0] = self.u
            self.update_grid()
            print(f'simulated {i}/{iterations}')
        
        #print(self.saves.shape)
        if save: np.save(save_file, self.saves, allow_pickle=True)
        return self.saves


if __name__ == '__main__':

    model = ModelI(step_frequency=10)
    DATA = model.run_simulation(10000)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the initial surface
    surface = ax.plot_surface(DATA[0][1], DATA[0][2], DATA[0][0], cmap='viridis', edgecolor='none')

    # Add labels
    ax.set_title('3D Animated Surface', fontsize=16)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Function to update the surface for animation
    def update(frame):
        global surface
        # Clear the previous surface
        surface.remove()
        surface = ax.plot_surface(DATA[frame][1], DATA[frame][2], DATA[frame][0], cmap='viridis', edgecolor='none')

    # Create animation
    ani = FuncAnimation(fig, update, frames=100, interval=50)

    # Show the animation
    plt.show()

    # # Parameters
    # stair_width = 200
    # stair_length = 50 
    # stair_height = 30


    # Lx, Ly = stair_width, stair_length * 3 # Domain size
    # Nx, Ny = int(Lx), int(Ly)    # Number of grid points
    # alpha = 0.0001         # Weather Errosion
    # beta = 0.1      # Step Errosion coefficient
    # T_max = 2.0          # Maximum time
    # dt = 0.1               # Time step
    # dx = 1 #Lx / (Nx - 1)   # Grid spacing in x
    # dy = 1 #Ly / (Ny - 1)   # Grid spacing in y
    # Nt = int(T_max / dt) # Number of time steps

    # const = 1 # constant flux
    # q = -stair_height # constant flux value in cm^-2


    # # Initial condition: temperature distribution
    # u = np.zeros((Nx, Ny))
    # force_map = np.zeros((Nx, Ny))

    # for i in range(3): 
    #     u[:,stair_length*i:stair_length*(i+1)] = stair_height*i

    # force_map[:, stair_length:2*stair_length] = mc.monte_carlo(stair_width, stair_length, 10, mc.downward, True, mc.gaussian_2d)
    # # force_map = force_map / force_map.max()

    # # u = np.random.random(size=(Nx, Ny))
    # # u[int(Nx/4):int(3*Nx/4), int(Ny/4):int(3*Ny/4)] = 10.0  # Hot square in the center


    # # Set up the figure and axis
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_box_aspect([10, 9, 6])

    # # Create the animation
    # from matplotlib.animation import FuncAnimation
    # ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=False)

    # # Display the animation
    # plt.show()
