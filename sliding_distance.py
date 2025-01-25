import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
h = 8e-6  # Linear wear (mm)
A_forefoot = 800  # Forefoot contact area (mm²)
A_full = 2000  # Full foot contact area (mm²)
F = 70 * 9.81  # Normal load (N)
H_rubber = 6.87  # Hardness of rubber sole (N/mm²)
H_eva = 4.91  # Hardness of EVA sole (N/mm²)
k_dry = 3e-4  # Wear coefficient (dry conditions)
k_wet = 7e-4  # Wear coefficient (wet conditions)

# Function to calculate sliding distance
def calculate_sliding_distance(h, A, H, k, F):
    Q = h * A  # Volume wear (mm³) 
    s = (Q * H) / (k * F)  # Sliding distance (mm)
    return s

# Calculate sliding distances for different conditions
s_rubber_dry_forefoot = calculate_sliding_distance(h, A_forefoot, H_rubber, k_dry, F)
s_rubber_dry_full = calculate_sliding_distance(h, A_full, H_rubber, k_dry, F)
s_eva_wet_forefoot = calculate_sliding_distance(h, A_forefoot, H_eva, k_wet, F)
s_eva_wet_full = calculate_sliding_distance(h, A_full, H_eva, k_wet, F)

sliding_distance = np.mean([s_rubber_dry_forefoot, s_eva_wet_forefoot]) 
print(sliding_distance)