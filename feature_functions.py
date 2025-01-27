import numpy as np
from scipy.signal import find_peaks

def laplacian(u, dx, dy):
    laplacian = np.zeros_like(u)
    laplacian[1:-1, 1:-1] = (
        (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
    return laplacian

def cut_stairs(u):
    return u[:, u.shape[1]//3 +1:u.shape[1]*2//3 -1]

def get_inner_edge_max(u):
    return u[:,u.shape[1]-1].max()

def get_inner_edge_avg(u):
    return u[:,u.shape[1]-1].mean()

def roughness(u):
    return laplacian(u,1,1).std()

def get_avg_side_func(u):
    avg = np.zeros(shape=(u.shape[1]))

    for line in u:
        avg += line

    return avg / avg.max()

def skew_index(u):
    er = 1-get_avg_side_func(u)
    er = er/er.sum()
    return (er*[i for i in range(u.shape[1])]).mean()
    

def get_avg_front_func(u):
    avg = np.zeros(shape=(u.shape[0]))

    for i in range(u.shape[1]):
        avg += u[:,i]

    return avg / avg.max()

    # return u[:,u.shape[1]]

def correlation_coefficient(x, y, sigma_y):
    # Convert inputs to NumPy arrays for element-wise operations
    x = np.array(x)
    y = np.array(y)
    sigma_y = np.array(sigma_y)
    
    # Calculate the mean of x and y
    mean_x = np.mean(x)
    mean_y = np.average(y, weights=1/sigma_y**2)  # Weighted average for y
    
    # Compute the numerator and denominator of the formula
    numerator = np.sum((x - mean_x) * (y - mean_y) / sigma_y**2)
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum(((y - mean_y)**2 / sigma_y**2)))
    
    # Return the correlation coefficient
    return numerator / denominator

def peak_index(f):
    f = f-f.min()
    f = f/f.max()
    p, a = find_peaks(f,prominence=0.05)
    s, a = find_peaks(1-f,prominence=0.05)

    return len(p)+len(s)

def width_peak_index(u):
    return peak_index(get_avg_front_func(u))