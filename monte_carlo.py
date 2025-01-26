import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from scipy.ndimage import gaussian_filter
import foot

def uniform_2d(size, center, sigma_x, sigma_y):
    return np.ones((size[1], size[0])) # Uniform distribution

def multi_modal_2d(size, center, sigma_x, sigma_y):
    width, height = size
    centers = []
    left_edge = 30 
    right_edge = width - 30
    middle = width // 2  

    current_x = left_edge
    while current_x < middle:
        centers.append((current_x, height // 2))
        current_x += 60

    current_x = right_edge
    while current_x > middle:
        centers.append((current_x, height // 2))
        current_x -= 60

    adjusted_sigma_x = sigma_x * 0.75  # 比原sigma_x更小，避免峰间重叠

    # 叠加所有高斯分布
    distribution = np.zeros((height, width))
    for cx, cy in centers:
        # 计算当前峰距离边缘的位置
        distance_from_edge = min(abs(cx - left_edge), abs(cx - right_edge))  # 距离最近边缘的距离
        max_distance = middle - left_edge  # 最大距离（从边缘到中间）

        # 强度从两侧的1.0逐渐减小到中间的0.1
        strength = 1.0 - (distance_from_edge / max_distance) * 0.9
        # strength = max(strength, 0.3)  # 最小强度为0.1

        # 生成高斯分布并叠加
        middle_gauss = gaussian_2d(size, (cx, cy), adjusted_sigma_x, sigma_y) * strength
        distribution += middle_gauss

    # 对中间区域进行调整
    blend_width = 50  # 渐变区域宽度
    for x in range(width):
        # 计算距离中间的距离
        dist_from_middle = abs(x - middle)
        if dist_from_middle < blend_width:
            # 混合权重：距离中间越近，中间分布的权重越高
            blend_weight = 1.0 - (dist_from_middle / blend_width)
            distribution[:, x] = (1 - blend_weight) * distribution[:, x] + blend_weight * middle_gauss[:, x]

    distribution = (distribution - distribution.min()) / (distribution.max() - distribution.min())
    distribution += gaussian_2d(size, (middle, height // 2), adjusted_sigma_x, sigma_y) * 0.6


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
    z = smoothed_heatmap.T

    return base_grid, z  # Return both the initial distribution and the final heatmap

if __name__ == '__main__':
    stair_width = 300
    stair_length = 50
    iterations = 10000
    upward_percentage = 0.5

    # Get the initial distribution and final heatmap
    base_grid, z = monte_carlo(stair_width, stair_length, iterations, upward_percentage, distribution_type=multi_modal_2d)

    # Plot the initial distribution
    plt.figure(figsize=(10, 6))
    plt.imshow(base_grid.T, cmap="viridis", origin="lower", extent=[0, stair_length, 0, stair_width])
    plt.colorbar(label="Probability Density")
    plt.title("Initial Distribution (Base Grid)")
    plt.xlabel("Stair Width")
    plt.ylabel("Stair Length")
    plt.show()

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