import numpy as np
import matplotlib.pyplot as plt

# we use equation Erotion rate = - Del (q) where q is sediment flux vector
# the sediment flux vector is a x C x Velocity, where velocity is proportional to gradient of terrain
# C has units of kg/m^3 which is density of water
# a has units of kg/kg for how much mass on target is loss per mass of water

# boundary conditions are constant gradient

def simulate_water(grid):



