import numpy as np
import matplotlib.pyplot as plt

# Function to generate points inside an ellipse
def ellipse_points(center, width, height, resolution):
    x, y = center
    points = []
    for i in range(-int(width / 2), int(width / 2), resolution):
        for j in range(-int(height / 2), int(height / 2), resolution):
            if (i / (width / 2))**2 + (j / (height / 2))**2 <= 1:
                points.append((int(x + i), int(y + j)))
    return points

# Function to generate points inside a circle
def circle_points(center, radius, resolution):
    x, y = center
    points = []
    for i in range(-radius, radius, resolution):
        for j in range(-radius, radius, resolution):
            if i**2 + j**2 <= radius**2:
                points.append((int(x + i), int(y + j)))
    return points

# Function to generate a foot shape
# Parameters: center (x, y), size (total length of the foot), and resolution of points
def generate_foot(center=(0, 0), size=200, resolution=5):
    cx, cy = center
    scale_factor = size / 200.0  # Normalize size based on default foot length of 200
    foot_points = []

    # Heel (circle)
    foot_points += circle_points(center=(cx, cy), radius=int(40 * scale_factor), resolution=resolution)

    # Arch (ellipse)
    foot_points += ellipse_points(center=(cx, cy + int(50 * scale_factor)), width=int(80 * scale_factor), height=int(100 * scale_factor), resolution=resolution)

    # Ball of the foot (ellipse)
    foot_points += ellipse_points(center=(cx, cy + int(140 * scale_factor)), width=int(100 * scale_factor), height=int(70 * scale_factor), resolution=resolution)

    # Toes (circles)
    foot_points += circle_points(center=(cx + int(20 * scale_factor), cy + int(190 * scale_factor)), radius=int(25 * scale_factor), resolution=resolution)  # Big toe
    foot_points += circle_points(center=(cx, cy + int(190 * scale_factor)), radius=int(20 * scale_factor), resolution=resolution)   # Second toe
    foot_points += circle_points(center=(cx - int(20 * scale_factor), cy + int(185 * scale_factor)), radius=int(15 * scale_factor), resolution=resolution) # Third toe
    foot_points += circle_points(center=(cx - int(35 * scale_factor), cy + int(175 * scale_factor)), radius=int(10 * scale_factor), resolution=resolution) # Fourth toe
    foot_points += circle_points(center=(cx - int(45 * scale_factor), cy + int(165 * scale_factor)), radius=int(8 * scale_factor), resolution=resolution)  # Fifth toe

    return foot_points

# Example usage
resolution = 1  # Resolution of points
size = 20     # Total length of the foot
center = (0, 0) # Center of the foot

foot_points = generate_foot(center=center, size=size, resolution=resolution)

# Extract x and y coordinates
x_coords, y_coords = zip(*foot_points)

# Plot the points for visualization
plt.figure(figsize=(6, 10))
plt.scatter(x_coords, y_coords, s=10, color="blue")
plt.axis("equal")
plt.title("Foot Shape Filled with Points (Upright)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()

# Output the list of coordinates
print("Coordinates that fill the foot shape:")
print(foot_points)
