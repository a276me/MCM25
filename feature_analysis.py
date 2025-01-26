from matplotlib import pyplot as plt
import numpy as np

# analyze different features of the stairs depth

def calculate_lost_volume(matrix):
    return -np.sum(matrix)

def project_to_side(matrix):
    return np.sum(matrix,axis=1)/total_steps

def plot_lost(simulations):
    """
    Plot the lost volume from simulations.
    
    Parameters:
        simulations (numpy array): Array where each row represents
                                   simulation data and column 0 holds the lost volume.
    """
    plt.plot(simulations[:, 0], label="Lost Volume")
    plt.xlabel("Simulation Index")
    plt.ylabel("Lost Volume")
    plt.title("Lost Volume Across Simulations")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    plot_lost()