# cell size: 0.005m

from matplotlib import pyplot as plt
import numpy as np

# Read the data from the text file into a matrix


def read_matrix_from_txt(file_path):
    with open(file_path, 'r') as file:
        matrix = [list(map(float, line.split())) for line in file]
    return np.array(matrix)

# Plot the matrix as a heatmap
def plot_matrix(matrix):
    plt.imshow(matrix, cmap='viridis', aspect='equal')
    plt.colorbar(label='Value')
    plt.title('Matrix Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    plt.show()

def replace_zeros_with_neighbors(matrix):
    """Replaces zeros in the matrix with the average of surrounding non-zero elements."""
    rows, cols = matrix.shape
    new_matrix = matrix.copy()

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                # Collect surrounding elements
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and (di != 0 or dj != 0)):
                            if matrix[ni, nj] != 0:
                                neighbors.append(matrix[ni, nj])

                # Replace zero with the average of surrounding non-zero neighbors
                if neighbors:
                    new_matrix[i, j] = np.mean(neighbors)

    return new_matrix

def subtract_plane(data):
    """
    Subtracts a best-fit plane from a 2D array of height data.
    
    Parameters
    ----------
    data : 2D numpy.ndarray
        A 2D array of height values, shape (rows, cols).
    
    Returns
    -------
    data_subtracted : 2D numpy.ndarray
        The original data with the best-fit plane subtracted.
    """
    # Get the shape of the input data
    rows, cols = data.shape
    
    # Create a coordinate grid
    # Y will vary from 0 to rows-1 (down the rows),
    # X will vary from 0 to cols-1 (across the columns).
    Y, X = np.mgrid[:rows, :cols]
    
    # Flatten the coordinates and data for fitting
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = data.ravel()
    
    # Build the design matrix G where each row is [x, y, 1].
    # This corresponds to fitting the model z = a*x + b*y + c
    G = np.column_stack((X_flat, Y_flat, np.ones_like(X_flat)))
    
    # Solve for the least-squares fit of the plane coefficients (a, b, c)
    # using numpy.linalg.lstsq
    (a, b, c), residuals, rank, s = np.linalg.lstsq(G, Z_flat, rcond=None)
    
    # Construct the fitted plane for each point in the 2D grid
    plane = a * X + c
    
    # Subtract the fitted plane from the original data
    data_subtracted = data - plane
    
    return data_subtracted

def subtract_linear_fit(data):
    """
    Subtracts a best-fit plane from a 2D array of height data.
    
    Parameters
    ----------
    data : 2D numpy.ndarray
        A 2D array of height values, shape (rows, cols).
    
    Returns
    -------
    data_subtracted : 2D numpy.ndarray
        The original data with the best-fit plane subtracted.
    """
    # Get the shape of the input data
    rows, cols = data.shape
    
    # Create a coordinate grid
    # Y will vary from 0 to rows-1 (down the rows),
    # X will vary from 0 to cols-1 (across the columns).
    Y, X = np.mgrid[:rows, :cols]
    
    # Flatten the coordinates and data for fitting
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = data.ravel()
    
    # Build the design matrix G where each row is [x, y, 1].
    # This corresponds to fitting the model z = a*x + b*y + c
    G = np.column_stack((X_flat, Y_flat, np.ones_like(X_flat)))
    
    # Solve for the least-squares fit of the plane coefficients (a, b, c)
    # using numpy.linalg.lstsq
    (a, b, c), residuals, rank, s = np.linalg.lstsq(G, Z_flat, rcond=None)
    
    # Construct the fitted plane for each point in the 2D grid
    plane = a * X + b * Y+ c
    
    # Subtract the fitted plane from the original data
    data_subtracted = data - plane
    
    return data_subtracted

def plot_contours(data, n_levels=10):
    """
    Plots contour lines of a 2D NumPy array.
    
    Parameters
    ----------
    data : 2D numpy.ndarray
        A 2D array of height values, shape (rows, cols).
    n_levels : int
        Number of contour levels to plot.
    """
    rows, cols = data.shape
    
    # Create a coordinate grid
    Y, X = np.mgrid[:rows, :cols]
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Create contour lines
    # You can use contourf for filled contours, or contour for just contour lines
    contour = ax.contour(X, Y, data, levels=n_levels, cmap='viridis')
    
    # Optional: label the contour lines
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Add a color bar
    plt.colorbar(contour, ax=ax)
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Contour Plot of 2D Data')
    ax.set_aspect('equal')
    # Show the plot
    plt.show()

file_path = 'DataExtract/stairs_data.txt'
data = read_matrix_from_txt(file_path)
data = data[5:136,1:341]

sample_stairs_data = replace_zeros_with_neighbors(data)*100
sample_step_data =sample_stairs_data[40:70]
sample_step_data = sample_step_data[::-1]
sample_step_data = subtract_plane(sample_step_data)
#sample_step_data = subtract_linear_fit(sample_step_data)
sample_stairs_dimension = (30,340,12)

if __name__ == "__main__":
    #plot_matrix(sample_stairs_data[85:135])
    plot_matrix(sample_step_data)
    plot_contours(sample_step_data)
    plt.plot(sample_step_data.mean(axis=0))
    plt.show()
    #print(sample_step_data.shape)
    
