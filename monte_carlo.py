import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.ndimage import gaussian_filter
import foot

def uniform_2d(size, center, sigma_x, sigma_y):
    return np.ones((size[1], size[0])) # Uniform distribution

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

def foot_angle(is_downward = True):
    if is_downward:
        return np.random.uniform(0-45,0+45)
    return np.random.uniform(180-45,180+45)

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
        
        rotated_points.append(((round(new_x)), (round(new_y))))
    
    return rotated_points

def calculate_pressure(y, direction, upward, downward):
    """
    Determine the pressure value based on the direction and y-coordinate.
    """
    pressure_map = upward if direction == 0 else downward
    for key, value in pressure_map.items():
        if key[0] <= y <= key[1]:
            return value
    return 0

def process_footprint(grid, step_points, direction, upward, downward, foot_length):
    """
    Process each footprint, apply rotation, and update the grid with pressure values.
    """
    for point in step_points:
        x, y = point
        foot_center = (x, y)
        foot_points = foot.generate_foot(center=foot_center, size=foot_length)

        # Rotate the foot based on direction
        angle = foot_angle(direction)
        rotated_foot_points = rotate_foot_points(foot_points, angle, foot_center)

        # Determine the pressure value
        pressure = calculate_pressure(y, direction, upward, downward)

        # Add the rotated footprint to the grid
        add_footprint_to_grid(grid, rotated_foot_points, pressure)

    return grid

def monte_carlo(stair_width, stair_length, iterations, upward_percentage, distribution_type=gaussian_2d):

    percentages = [upward_percentage, 1 - upward_percentage]
    
    upward = {
    (23, 27): 129.00,
    (20, 22): 114.67,
    (16, 19): 78.28,
    (13, 15): 102.28,
    (8, 12): 118.85,
    (5, 7): 88.17,
    (2, 4): 43.60,
    (0, 1): 35
    }

    downward = {
    (23, 27): 74.50,
    (20, 22): 82.67,
    (16, 19): 59.14,
    (13, 15): 128.50,
    (8, 12): 144.43,
    (5, 7): 94.71,
    (2, 4): 27,
    (0, 1): 20
    }

    size = np.array((stair_width, stair_length))

    human_width = 50
    foot_length = 27

    base_grid = distribution_type(size, (stair_width / 2, stair_length / 2), human_width, foot_length)

    # Initialize heatmap grid
    heatmap_grid = np.zeros_like(base_grid)

    # Process each direction (upward and downward)
    for direction, percentage in enumerate(percentages):
        num_steps = int(percentage * iterations)
        step_points = sample_points_from_distribution(base_grid, num_steps)
        heatmap_grid = process_footprint(heatmap_grid, step_points, direction, upward, downward, foot_length)

    # Smooth the heatmap for better visualization
    smoothed_heatmap = gaussian_filter(heatmap_grid, sigma=2)
    z = smoothed_heatmap.T  # Transpose Z to match swapped X and Y

    return z

if __name__ == '__main__':

    # Run the Monte Carlo simulation
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')  # 3D plot
    z = monte_carlo(200, 50, 10000, 0, distribution_type=uniform_2d)
    surf = ax.plot_surface(np.arange(z.shape[1]), np.arange(z.shape[0])[:, None], z, cmap="viridis")

    # Add color bar and labels
    fig.colorbar(surf, ax=ax, label="Pressure Intensity")
    ax.set_title("Aggregated Foot Pressure Surface Plot (With Random Angles)")
    ax.set_xlabel("Stair Length")
    ax.set_ylabel("Stair Width")
    ax.set_zlabel("Pressure")

    ax.set_box_aspect([1,3,0.3])
    plt.show()

