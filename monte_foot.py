import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def gaussian_2d(size, center, sigma_x, sigma_y):
    width, height = size
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
    probabilities = distribution / distribution.sum()  # Normalize
    cumulative_prob = probabilities.ravel().cumsum()
    random_values = np.random.rand(num_samples)
    sampled_indices = np.searchsorted(cumulative_prob, random_values)
    sampled_y, sampled_x = np.unravel_index(sampled_indices, distribution.shape)
    return np.column_stack((sampled_x, sampled_y))

def add_footprint(distribution, point, foot_width, foot_length, intensity=1):
    """
    Add an elliptical footprint to the distribution.
    """
    x, y = point
    height, width = distribution.shape
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Ellipse formula for the footprint
    ellipse = (((x_grid - x) / (foot_width / 2)) ** 2 +
               ((y_grid - y) / (foot_length / 2)) ** 2) <= 1

    # Increase the intensity in the footprint area
    distribution[ellipse] += intensity
    return distribution

# Parameters
stair_width = 200
stair_depth = 30
human_width = 50
foot_length = 27
num_samples = 10000
foot_width = 20  # Width of the front foot
foot_length = 15  # Length of the front foot

# Create a base distribution for where people step
size = (stair_width, stair_depth)
oneperson = gaussian_2d(size, (stair_width / 2, stair_depth / 2), human_width, foot_length)

# Sample points representing where people step
step_points = sample_points_from_distribution(oneperson, num_samples)

# Initialize a new distribution for footprints
footprint_distribution = np.zeros_like(oneperson)

# Add footprints for each sampled point
for point in step_points:
    footprint_distribution = add_footprint(footprint_distribution, point, foot_width, foot_length)

# Smooth the distribution for better visualization
footprint_distribution_smoothed = gaussian_filter(footprint_distribution, sigma=2)

# Plot the original distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Distribution of Steps")
plt.imshow(oneperson, cmap="viridis")
plt.colorbar()

# Plot the footprint distribution
plt.subplot(1, 2, 2)
plt.title("Footprint Heatmap")
plt.imshow(footprint_distribution_smoothed, cmap="hot")
plt.colorbar()

plt.show()
