import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.ndimage import gaussian_filter
import foot
import testditribution as td
import time

def uniform_2d(size, sigma_x, sigma_y):
    return np.ones((size[1], size[0])) # Uniform distribution

def multi_modal_2d(size, sigma_x, sigma_y, n_peak):
    """
    Generate multi-peak 2D distribution with automatic peak placement
    Parameters:
        size: (width, height) of the output array
        sigma_x: standard deviation in x-direction
        sigma_y: standard deviation in y-direction
        n_peak: number of desired peaks
    """
    width, height = size
    distribution = np.zeros(size[::-1])  # Create empty array
    
    # Calculate maximum number of possible peaks based on sigma_x
    n_max = size[0] / sigma_x  # Maximum number of peaks allowed
    n_actual = min(n_peak, n_max)
    
    # Calculate horizontal positions
    left_bound = 30
    right_bound = width - 30
    y_center = height // 2
    
    # if n_actual == 1:
    #     x_centers = [width // 2]
    # else:
    #     # Evenly distribute peaks between bounds
    #     x_centers = np.linspace(left_bound, right_bound, n_actual).astype(int)
    
    for pos in range(left_bound, right_bound):
        x_centers = [pos]
        distribution += gaussian_2d(
            size=size,
            center=(pos, y_center),
            sigma_x=sigma_x,
            sigma_y=sigma_y
        )
        pos += size[0] / n_actual * sigma_x 

    # # Create Gaussian peaks
    # for x in x_centers:
    #     # Calculate distance from nearest edge
    #     edge_distance = min(x - left_bound, right_bound - x)
    #     max_edge_distance = (right_bound - left_bound) // 2
    #     strength = 0.1 + 0.9 * (edge_distance / max_edge_distance)  # Strength reduction near edges
        
    #     # Add Gaussian to distribution
    #     distribution += gaussian_2d(
    #         size=size,
    #         center=(x, y_center),
    #         sigma_x=sigma_x,
    #         sigma_y=sigma_y
    #     ) * strength
    
    # Normalize distribution
    distribution /= distribution.max()
    
    return distribution

# def multi_modal_2d(size, center, sigma_x, sigma_y, human_width, n_peak):
    width, height = size
    centers = []
    left_edge = 30 
    right_edge = width - 30
    middle = width // 2  

    n_max = width / human_width  # Maximum number of peaks allowed

    if n_peak * 2 > n_max:
        # Original code for generating peaks when n_peak is too large
        current_x = left_edge
        while current_x < middle:
            centers.append((current_x, height // 2))
            current_x += 60

        current_x = right_edge
        while current_x > middle:
            centers.append((current_x, height // 2))
            current_x -= 60
    else:
        # Generate n_peak peaks on both left and right sides
        # Left side peaks (from left_edge towards middle)
        left_centers = []
        for i in range(n_peak):
            cx = left_edge + i * human_width
            left_centers.append((cx, height // 2))
        
        # Right side peaks (from right_edge towards middle)
        right_centers = []
        for i in range(n_peak):
            cx = right_edge - i * human_width
            right_centers.append((cx, height // 2))

        # Calculate the average x-coordinate of left and right peaks
        left_avg_x = np.mean([cx for cx, cy in left_centers])
        right_avg_x = np.mean([cx for cx, cy in right_centers])

        # Desired average positions: 1/3 and 2/3 of the width
        desired_left_avg = width / 3
        desired_right_avg = 2 * width / 3

        # Compute the shift needed for left and right peaks
        left_shift = desired_left_avg - left_avg_x
        right_shift = desired_right_avg - right_avg_x

        # Shift all left and right peaks
        left_centers = [(cx + left_shift, cy) for cx, cy in left_centers]
        right_centers = [(cx + right_shift, cy) for cx, cy in right_centers]

        # Combine left and right centers
        centers = left_centers + right_centers

    # Generate Gaussian distributions for all peaks
    distribution = np.zeros((height, width))
    for cx, cy in centers:
        distance_from_edge = min(abs(cx - left_edge), abs(cx - right_edge)) 
        max_distance = middle - left_edge  
        strength = 1.0 - (distance_from_edge / max_distance) * 0.9  # Strength decreases near edges
        middle_gauss = gaussian_2d(size, (cx, cy), sigma_x, sigma_y) * strength
        distribution += middle_gauss

    # Blend near the middle for smooth transition
    blend_width = 50  
    for x in range(width):
        dist_from_middle = abs(x - middle)
        if dist_from_middle < blend_width:
            blend_weight = 1.0 - (dist_from_middle / blend_width)
            distribution[:, x] = (1 - blend_weight) * distribution[:, x] + blend_weight * middle_gauss[:, x]

    # Normalize the distribution
    distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min())

    return distribution

def gaussian_2d(size, center, sigma_x, sigma_y):
    width, height= size
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Gaussian formula
    gaussian = np.exp(-(((x - center[0]) ** 2) / (2 * sigma_x ** 2) +
                        ((y - center[1]) ** 2) / (2 * sigma_y ** 2)))
    return gaussian

def x_distribute(trials, n):
    x_pos = []

    for i in range (trials):
        y=[td.generate_normal_points_int(170,25,0,340)]
        for j in range(n):
            #pos = generate_normal_points_int(std_dev=200)
            pos = np.random.uniform(low=0, high=341, size=1).astype(int)
            #pos = generate_left_skewed_points_int()
            if td.find_min_difference(pos,y)>50:
                y.append(pos)
                x_pos.append(pos)

    return x_pos

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
    sampled_x = x_distribute(trials = num_samples, n = 5) 
    sampled_y, x = np.unravel_index(sampled_indices, distribution.shape)

    return np.column_stack((sampled_x, sampled_y))


def add_footprint_to_grid(grid, footprint, pressure):

    for point in footprint:
        x, y = point
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:  # Check bounds
            grid[y, x] += pressure
    
    return grid

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

def process_footprint(grid, iterations, direction, upward, downward, foot_length, n):
    """
    Process each footprint, apply rotation, and update the grid with pressure values.
    """
    for i in range(iterations):
        X=[np.random.normal(grid.shape[1] // 2, 25)]
        y = np.random.normal(grid.shape[0] // 2, foot_length)
        for j in range(n):
            #pos = generate_normal_points_int(std_dev=200)
            pos = np.random.uniform(low=0, high=grid.shape[1])
            #pos = generate_left_skewed_points_int()
            if td.find_min_difference(pos,X)>50:
                X.append(pos)
            
        for x in X:
            foot_center = (x, y)
            foot_points = foot.generate_foot(center=foot_center, size=foot_length)
            # Rotate the foot based on direction
            angle = foot_angle(direction)
            rotated_foot_points = rotate_foot_points(foot_points, angle, foot_center)

            # Determine the pressure value
            pressure = calculate_pressure(y, direction, upward, downward)

            # Add the rotated footprint to the grid
            grid = add_footprint_to_grid(grid, rotated_foot_points, pressure)

    return grid

def monte_carlo(stair_width, stair_length, iterations, upward_percentage, n):

    # if stair_width <= 100:
    #     distribution_type = gaussian_2d
    # else:
    #     distribution_type = multi_modal_2d

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

    size = np.array((stair_length,stair_width))

    human_width = 50
    foot_length = 27

    # Initialize heatmap grid
    heatmap_grid = np.zeros(size)

    # Process each direction (upward and downward)
    for direction, percentage in enumerate(percentages):
        num_steps = int(percentage * iterations/n)
        heatmap_grid += process_footprint(heatmap_grid, num_steps, direction, upward, downward, foot_length,n)

    # Smooth the heatmap for better visualization
    smoothed_heatmap = gaussian_filter(heatmap_grid, sigma=2)
    z = smoothed_heatmap.T

    return z  # Return both the initial distribution and the final heatmap

if __name__ == '__main__':
    stair_width = 300
    stair_length = 50
    iterations = 1000
    upward_percentage = 0.5
    # number_percentage = 0 # Percentage of gaussian-modal distribution
    n_peak = 2
    # Get the initial distribution and final heatmap
    t1 = time.time()
    z = monte_carlo(stair_width, stair_length, iterations, upward_percentage, n_peak)
    print("Time taken: ", (time.time() - t1), " sec")

    # Plot the final heatmap
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')  # 3D plot
    surf = ax.plot_surface(np.arange(z.shape[1]), np.arange(z.shape[0])[:, None], z, cmap="viridis")

    # Add color bar and labels
    fig.colorbar(surf, ax=ax, label="Pressure Intensity")
    ax.set_title("Aggregated Foot Pressure Surface Plot (With Random Angles)")
    ax.set_xlabel("Stair Length")
    ax.set_ylabel("Stair Width")
    ax.set_zlabel("Pressure")

    ax.set_box_aspect([1, 3, 0.3])
    plt.show()