import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.ndimage import gaussian_filter
import foot

def gaussian_2d(size, center, sigma_x, sigma_y):
    width, height= size
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Gaussian formula
    gaussian = np.exp(-(((x - center[0]) ** 2) / (2 * sigma_x ** 2) +
                        ((y - center[1]) ** 2) / (2 * sigma_y ** 2)))
    return gaussian

def sample_points_from_distribution(distribution, num_samples):
    """
    Sample points from a 2D distribution using probabilities.
    """
    # Normalize the distribution to probabilities
    probabilities = distribution / distribution.sum()

    # Generate cumulative distribution for efficient sampling
    cumulative_prob = probabilities.ravel().cumsum()

    # Sample random numbers uniformly in [0, 1]
    random_values = np.random.rand(num_samples)

    # Find the indices in the cumulative distribution
    sampled_indices = np.searchsorted(cumulative_prob, random_values)

    # Convert the flat indices back to 2D coordinates
    sampled_y, sampled_x = np.unravel_index(sampled_indices, distribution.shape)

    return np.column_stack((sampled_x, sampled_y))


def add_footprint_to_grid(grid, footprint, pressure):
    for point in footprint:
        x, y = point
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:  # Check bounds
            grid[y, x] += pressure

def foot_angle():
    return np.random.uniform(0, 90)

def rotate_foot_points(points, angle, center):
    angle_rad = np.radians(angle)  # Convert angle to radians
    cx, cy = center
    rotated_points = []
    
    for x, y in points:
        # Translate point to origin
        x -= cx
        y -= cy
        
        # Rotate using 2D rotation matrix
        new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        
        # Translate point back
        new_x += cx
        new_y += cy
        
        rotated_points.append((int(round(new_x)), int(round(new_y))))
    
    return rotated_points

#current assume cm
stair_width = 200
stair_length = 50 
stair_height = 50
custom_pressures = {
    (23, 27): 129.00,
    (20, 22): 114.67,
    (16, 19): 78.28,
    (13, 15): 102.28,
    (8, 12): 118.85,
    (5, 7): 88.17,
    (2, 4): 43.60,
    (0, 1): 0.00
}
human_width = 50
foot_length = 27

iterations = 100

size = np.array((stair_width, stair_length))


# We will assume that a single person prefers to walk either in the center
# Or to the sides of the stairs


def monte_carlo(stair_width, stair_height, iterations, custom_pressures):
    base_grid  = gaussian_2d(size, (stair_width/2, stair_height/2), human_width, foot_length)

    step_points = sample_points_from_distribution(base_grid , iterations)

    # Initialize heatmap grid (aggregated pressure values)
    heatmap_grid = np.zeros_like(base_grid)

    # Add footprints to the heatmap
    for point in step_points:
        x, y = point
        foot_center = (x, y)
        foot_points = foot.generate_foot(center=foot_center, size=foot_length) 

        angle = foot_angle()
        rotated_foot_points = rotate_foot_points(foot_points, angle, foot_center)
        
        pressure = 0
        for key, value in custom_pressures.items():
            if key[0] <= y <= key[1]:
                pressure = value
                break

        add_footprint_to_grid(heatmap_grid, rotated_foot_points, pressure)

    # Smooth the heatmap for better visualization
    smoothed_heatmap = gaussian_filter(heatmap_grid, sigma=2)

    # Prepare data for 3D surface plot (swap x and y)
    x = np.arange(smoothed_heatmap.shape[0])  # Stair width
    y = np.arange(smoothed_heatmap.shape[1])  # Stair depth
    x, y = np.meshgrid(x, y)  # Create grid for plotting (swap axes)
    z = smoothed_heatmap.T  # Transpose Z to match swapped X and Y

    return z    

if __name__ == '__main__':

    # Run the Monte Carlo simulation
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')  # 3D plot
    x, y, z = monte_carlo(stair_width, stair_height, 1, custom_pressures)
    surf = ax.plot_surface(x, y, z, cmap="viridis")  # Use a colormap like viridis

    # Add color bar and labels
    fig.colorbar(surf, ax=ax, label="Pressure Intensity")
    ax.set_title("Aggregated Foot Pressure Surface Plot (With Random Angles)")
    ax.set_xlabel("Stair Length")
    ax.set_ylabel("Stair Width")
    ax.set_zlabel("Pressure")

    ax.set_box_aspect([1,3,0.3])
    plt.show()

