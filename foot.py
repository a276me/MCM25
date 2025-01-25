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
size = 27   # Total length of the foot
center = (0, 0) # Center of the foot

foot_points = generate_foot(center=center, size=size, resolution=resolution)

# Extract x and y coordinates
x_coords, y_coords = zip(*foot_points)

filtered_points = [point for point in foot_points if 0 <= point[1] <= 30]

custom_pressures = {
    (23, 27): 129.00,
    (20, 22): 114.67,
    (16, 19): 78.28,
    (13,15): 102.28,
    (8,12): 118.85,
    (5, 7): 88.17,
    (2, 4 ): 43.60,
    (1, 2): 0.00
}

filtered_points_with_pressure = []
for point in filtered_points:
    y = point[1]
    pressure = None
    for key, value in custom_pressures.items():
        if isinstance(key, tuple) and key[0] <= y <= key[1]: 
            pressure = value
            break
        elif isinstance(key, int) and y == key: 
            pressure = value
            break
    if pressure is not None:
        filtered_points_with_pressure.append({"x": point[0], "y": point[1], "pressure": pressure})

# Extract x, y, and pressure coordinates for visualization
filtered_x_coords = [p["x"] for p in filtered_points_with_pressure]
filtered_y_coords = [p["y"] for p in filtered_points_with_pressure]
pressures = [p["pressure"] for p in filtered_points_with_pressure]

# Plot the points for visualization
plt.figure(figsize=(8, 10))
plt.scatter(filtered_x_coords, filtered_y_coords, s=10, color="blue", label="Points")

# Annotate each point with its pressure value
for p in filtered_points_with_pressure:
    plt.text(p["x"] + 0.3, p["y"] + 0.3, f'{p["pressure"]:.2f}', fontsize=8, color="red")

plt.axis("equal")
plt.title("Foot Shape with Custom Pressure Values (Top 10 Rows: Height 10 to 20)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()

# Output the filtered list of coordinates with pressure
print("Filtered points with custom pressure values ")
print(filtered_points)
