import numpy as np
import matplotlib.pyplot as plt

def gaussian_2d(size, center, sigma_x, sigma_y):
    """
    Generate a 2D Gaussian distribution.

    Parameters:
    - size: Tuple of grid size (height, width).
    - mean_x: Mean of the Gaussian in the x-direction.
    - mean_y: Mean of the Gaussian in the y-direction.
    - sigma_x: Standard deviation of the Gaussian in the x-direction.
    - sigma_y: Standard deviation of the Gaussian in the y-direction.

    Returns:
    - A 2D array containing the Gaussian distribution.
    """
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


#current assume cm
stair_width = 200
stair_depth = 30
stair_height = 30

size = np.array((stair_width, stair_depth))

human_width = 50
foot_length = 20

iterations = 100

# We will assume that a single person prefers to walk either in the center
# Or to the sides of the stairs

oneperson = gaussian_2d(size, size/2, human_width, foot_length)

step_points = sample_points_from_distribution(oneperson, iterations)




plt.imshow(oneperson)
plt.show()

