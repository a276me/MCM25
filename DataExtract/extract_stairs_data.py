# cell size: 0.005m

import numpy as np
import matplotlib.pyplot as plt

# Read the data from the text file into a matrix


def read_matrix_from_txt(file_path):
    with open(file_path, 'r') as file:
        matrix = [list(map(float, line.split())) for line in file]
    return np.array(matrix)

# Plot the matrix as a heatmap


def plot_matrix(matrix):
    plt.imshow(matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Matrix Heatmap')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def plot_matrix_3d(matrix):

    # Create the plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(matrix[1],matrix[2],matrix[0], cmap='viridis')

    # Set labels
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Height')

    # Show the plot
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


file_path = 'DataExtract/stairs_data.txt'
data = read_matrix_from_txt(file_path)
data = data[10:270]
sample_stairs_data = replace_zeros_with_neighbors(data)

if __name__ == "__main__":
    #plot_matrix(sample_stairs_data[85:135])
    plot_matrix(sample_stairs_data)
    plot_matrix_3d(sample_stairs_data)
