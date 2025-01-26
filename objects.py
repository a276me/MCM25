from monte_carlo import gaussian_2d, uniform_2d
from main import ModelI
class Material:
    def __init__(self, wearing_coefficient, hardness, sliding_distance):
        # product of the unit is 1/kPa
        self.wearing_coefficient = wearing_coefficient
        self.hardness = hardness
        # unit: mc
        self.sliding_distance = sliding_distance
        # unit: m^3/N ~ cm/kPa
        self.step_erosion_coefficient = sliding_distance * wearing_coefficient / hardness

class Stairs:
    def __init__(self, width, length, height, material: Material):
        # dimensions of the stairs, unit: cm
        self.width = width
        self.length = length
        self.height = height
        self.material = material
        self.dimensions = (width, length, height)
        

class Environment:
    def __init__(self, weather_erosion_coefficient_per_rain, rain_frequency):
        # amount of material eroded per rain, unit: cm^2
        self.weather_erosion_coefficient_per_rain = weather_erosion_coefficient_per_rain
        # number of rains in a year, unit: 1/year
        self.rain_frequency = rain_frequency
        

class StairsUsage:
    def __init__(self, frequency, direction_percentage, is_paired):
        # the total number of steps in a year, unit: 1/year
        self.frequency = frequency
        self.direction_percentage = direction_percentage
        self.is_paired = is_paired
        self.distribution = uniform_2d if is_paired else gaussian_2d


class SimulationSettings:
    def __init__(self, total_time, environment: Environment, stairs: Stairs, stairs_usage: StairsUsage):
        # number of iterations, unit: 1
        self.iterations = int(total_time*environment.rain_frequency)
        # duration of a iteration, unit: year
        self.period = 1/environment.rain_frequency
        # total length of durations, unit: year
        self.total_time = total_time
        # how many number of steps are stepped onto the stairs when a rainfall happens on average
        self.steps_per_rain = environment.rain_frequency/stairs_usage.frequency
        # total number of steps ever stepped onto this stair, unit: 1
        self.total_steps = stairs_usage.frequency * total_time
        
        # beta coefficient in the diff eq, unit: (cm/year)/kPa
        # beta * force_map = change_in_height_per_year
        # step_erosion_coefficient * force_map * frequency = change_in_height_per_year
        self.beta = stairs.material.step_erosion_coefficient * stairs_usage.frequency
        
        # alpha coefficient in diff eq, unit: (cm^2/year)
        self.alpha = environment.weather_erosion_coefficient_per_rain * environment.rain_frequency
        
        self.environment = environment
        self.stairs = stairs
        self.stairs_usage = stairs_usage
        
        self.model = ModelI(stair_dim=stairs.dimensions,
                            alpha=self.alpha,
                            beta=self.beta,
                            direction=stairs_usage.direction_percentage,
                            distribution=stairs_usage.distribution,
                            steps_per_dt=self.steps_per_rain,
                            dt=self.period)
        
        self.simulation = self.model.run_simulation(self.iterations)
        self.final_shape = self.simulation[self.iterations-1]


