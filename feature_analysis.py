import numpy as np

# analyze different features of the stairs depth

def calculate_lost_volume(matrix):
    return -np.sum(matrix)

def project_to_side(matrix,total_steps):
    return np.sum(matrix,axis=1)/total_steps