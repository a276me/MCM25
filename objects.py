from monte_carlo import gaussian_2d, uniform_2d
from main import ModelI
class Material:
    def __init__(self, wearing_coefficient, hardness, sliding_distance, diffusion_ratio):
        # product of the unit is 1/kPa
        self.wearing_coefficient = wearing_coefficient
        self.hardness = hardness
        # unit: cm
        self.sliding_distance = sliding_distance
        # unit: m^3/N ~ cm/kPa
        self.step_erosion_coefficient = sliding_distance * wearing_coefficient / hardness
        # ratio of diffusion of liquid to diffusion of stairs, unit: 1
        self.diffusion_ratio = diffusion_ratio

class Stairs:
    def __init__(self, width, length, height, material: Material):
        # dimensions of the stairs, unit: cm
        self.width = width
        self.length = length
        self.height = height
        self.material = material
        self.dimensions = (width, length, height)
        

class Environment:
    def __init__(self, env_coef):
        # diffusion of liquid that causes erosion
        self.env_coef = env_coef
        

class StairsUsage:
    def __init__(self, frequency, direction_percentage, is_paired):
        # the total number of steps in a year, unit: 1/year
        self.frequency = frequency
        self.direction_percentage = direction_percentage
        self.is_paired = is_paired
        self.distribution = uniform_2d if is_paired else gaussian_2d


class SimulationSettings:
    def __init__(self, total_time, environment: Environment, stairs: Stairs, stairs_usage: StairsUsage, dt):
        # number of iterations, unit: 1
        self.iterations = int(total_time/dt)
        # total length of durations, unit: year
        self.total_time = total_time
        # total number of steps ever stepped onto this stair, unit: 1
        self.total_steps = stairs_usage.frequency * total_time
        
        # beta coefficient in the diff eq, unit: (cm/year)/kPa
        # beta * force_map = change_in_height_per_year
        # step_erosion_coefficient * force_map * frequency = change_in_height_per_year
        self.beta = stairs.material.step_erosion_coefficient * stairs_usage.frequency
        
        # alpha coefficient in diff eq, unit: (cm^2/year)
        self.alpha = environment.env_coef * stairs.material.diffusion_ratio
        
        self.environment = environment
        self.stairs = stairs
        self.stairs_usage = stairs_usage
        
        self.model = ModelI(stair_dim=stairs.dimensions,
                            alpha=self.alpha,
                            beta=self.beta,
                            direction=stairs_usage.direction_percentage,
                            distribution=stairs_usage.distribution,
                            step_frequency=stairs_usage.frequency,
                            dt=dt)
        
        self.simulation = self.model.run_simulation(self.iterations)
        self.final_shape = self.simulation[self.iterations-1]