from matplotlib import pyplot as plt
import numpy as np

def generate_normal_points_int(mean=500, std_dev=100, lower_bound=0, upper_bound=1000):
    """
    Generate random integer points from a normal distribution within a specified range.
    
    Parameters:
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.
        num_points (int): Number of points to generate.
        lower_bound (int): Minimum value of the points.
        upper_bound (int): Maximum value of the points.

    Returns:
        list: List of randomly generated integer points.
    """
    points = np.random.normal(loc=mean, scale=std_dev, size=100)
    points = np.clip(points, lower_bound, upper_bound)  # Ensure values are within bounds
    return points[0].astype(int)

def generate_left_skewed_points_int(mean=500, std_dev=100, lower_bound=0, upper_bound=1000):
    """
    Generate random integer points from a left-skewed distribution within a specified range.
    
    Parameters:
        mean (float): The mean of the normal distribution used as a base.
        std_dev (float): The standard deviation of the normal distribution used as a base.
        num_points (int): Number of points to generate.
        lower_bound (int): Minimum value of the points.
        upper_bound (int): Maximum value of the points.

    Returns:
        list: List of randomly generated integer points.
    """
    # Generate random points from a normal distribution
    points = np.random.normal(loc=mean, scale=std_dev, size=100)
    # Apply a skew transformation (e.g., square the points to skew to the left)
    points = mean - (mean - points) ** 2 / mean
    points = np.clip(points, lower_bound, upper_bound)  # Ensure values are within bounds
    return points[0].astype(int)

def find_min_difference(num, arr):
    """
    Find the minimum difference between a number and elements in an array.
    
    Parameters:
        num (int or float): The reference number.
        arr (list): A list of numbers to compare.
    
    Returns:
        tuple: The minimum difference and the corresponding array element.
    """
    if not arr:
        return 10000
    
    # Calculate the absolute differences
    differences = [abs(num - x) for x in arr]
    min_diff = min(differences)
    closest_value = arr[differences.index(min_diff)]
    
    return min_diff
# Example usage
if __name__ == "__main__":
    n = 20
    
    x = np.zeros(341)
    
    for i in range (10000):
        y=[ generate_normal_points_int(170,25,0,340)]
        for j in range(n):
            #pos = generate_normal_points_int(std_dev=200)
            pos = np.random.uniform(low=0, high=341, size=1).astype(int)
            #pos = generate_left_skewed_points_int()
            if find_min_difference(pos,y)>50:
                y.append(pos)
        
        for k in y:
            x[k]+=1
    
    plt.plot(x)
    plt.show()