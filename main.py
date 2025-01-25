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



# Boundary condition functions
def boundary_conditions(u, q):
    # Apply custom boundary conditions
    # u[0, :] = u[1, :]  # Zero flux at x = 0
    # u[-1, :] = u[-2, :]  # Zero flux at x = max
    # u[:, 0] = u[:, 1] #+ q * dy  # Constant flux at y = 0
    # u[:, -1] = u[:, -2] #- q * dy  # Constant flux at y = max

    u[0, :] = u[1, :]  # Zero flux at x = 0 
    u[-1, :] = u[-2, :]  # Zero flux at x = max 
    u[:, 0] = 0
    u[:, -1] = 3*q

    return u

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

def update_grid(u, alpha, beta, stair_dim, dt, direction, distribution, steps_per_dt):
    
    stair_width, stair_height, q = stair_dim

    force_map = np.zeros((Nx, Ny))
    force_map[:, stair_length:2*stair_length] = mc.monte_carlo(stair_width, stair_length, steps_per_dt, direction, distribution)
    
    du = alpha * laplacian(u, dx, dy) - beta * force_map

    # du = -alpha * np.sqrt((derivatives(u)**2).sum()) - beta * force_map

    du = du*dt
    u_new = u + du
    u = boundary_conditions(u_new, q)
    return u

def update(frame):
    global u, force_map

    for i in range(5):
        u = update_grid(u)
    
    force_map = force_map

    X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
    ax.clear()
    ax.plot_surface(X, Y, np.transpose(u), cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')
    return [ax]

def run_simulation(stair_dim, iterations, dt, direction, distribution, steps_per_dt, **args):
    
    stair_width, stair_height, stair_length = stair_dim

    Lx, Ly = stair_width, stair_length * 3 # Domain size
    Nx, Ny = int(Lx), int(Ly)    # Number of grid points

    alpha = 0.0001  # Weather Errosion
    beta = 0.1      # Step Errosion coefficient

    u = np.zeros((Nx, Ny))
    force_map = np.zeros((Nx, Ny))

    saves = np.zeros(shape=(iterations, u.shape[0], u.shape[1]))

    for i in range(3): 
        u[:,stair_length*i:stair_length*(i+1)] = stair_height*i
    
    for i in range(iterations):
        saves[i] = u
        u = update_grid(u,alpha, beta, stair_dim, dt, direction, distribution, steps_per_dt)
        print(f'simulated {i}/{iterations}')
        

class ModelI:

    def __init__(self, alpha=0.05, beta=0.1, stair_dim=(200, 50, 30), direction=1, distribution=mc.gaussian_2d, steps_per_dt=5, dt=1, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.stair_width, self.stair_length, self.q = stair_dim
        self.dx = 1 
        self.dy = 1 
        self.dt = dt
        self.steps_per_dt = steps_per_dt
        self.direction = direction
        self.distribution = distribution

    def update_grid(self,):
        force_map = np.zeros((self.Nx, self.Ny))
        # force_map[:, self.stair_length:2*self.stair_length] = mc.monte_carlo(self.stair_width,
        #                                                      self.stair_length, self.steps_per_dt, self.direction, self.distribution)
        force_map = mc.monte_carlo(self.stair_width,
                                    self.stair_length, self.steps_per_dt, self.direction, self.distribution)
 
        du = self.alpha * laplacian(self.u, self.dx, self.dy) - self.beta * force_map

        # du = -alpha * np.sqrt((derivatives(u)**2).sum()) - beta * force_map

        du = du*self.dt
        self.u = self.u + du

        self.apply_boundary_conditions()
        
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


    def run_simulation(self, iterations, save_file='./DiffEqSim/sim.npy'):

        self.Lx, self.Ly = self.stair_width, self.stair_length # Domain size
        self.Nx, self.Ny = int(self.Lx), int(self.Ly) 
        self.u = np.zeros((self.Nx, self.Ny))
        Y, X = np.meshgrid( np.linspace(0, self.Ly, self.Ny), np.linspace(0, self.Lx, self.Nx),)

        # for i in range(3): 
        #     self.u[:,self.stair_length*i:self.stair_length*(i+1)] = self.q*i

        self.saves = np.zeros(shape=(iterations, 3, self.u.shape[0], self.u.shape[1]))
        self.saves[:,1,:,:] = X
        self.saves[:,2,:,:] = Y

        for i in range(iterations):
            self.saves[i][0] = self.u
            self.update_grid()
            print(f'simulated {i}/{iterations}')
        
        print(self.saves.shape)
        np.save(save_file, self.saves, allow_pickle=True)
        return self.saves




if __name__ == '__main__':

    model = ModelI()
    model.run_simulation(1000)

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