from DataExtract.extract_stairs_data import sample_step_data, sample_stairs_dimension, plot_contours,plot_matrix
from objects import Environment, Stairs, Material, StairsUsage,SimulationSettings
import feature_functions as f

hardness_1 = 850200
hardness_2 = 5000000
env = Environment()
mat = Material(hardness=hardness_1,wearing_coefficient=1.5E-0,sliding_distance=0.013939961305623166,diffusion_ratio=2E-3)
str = Stairs(sample_stairs_dimension[0],sample_stairs_dimension[1],sample_stairs_dimension[2],mat)
actual_usage = StairsUsage(frequency=5E6,peaks=10,direction_percentage=0.9)
sim = SimulationSettings(total_time=158,environment=env, stairs=str, stairs_usage=actual_usage,dt=0.1)
print(sim.alpha,sim.beta)
Z_hat = f.cut_stairs(sim.final_shape[0])
Z_hat = sim.final_shape[0]
Z_hat = f.cut_stairs(sim.simulation[500][0])
Z_obs=sample_step_data.T
initial_height = f.get_inner_edge_max(Z_obs)
Z_obs += sample_stairs_dimension[2] - initial_height
if __name__ == "__main__":
    plot_matrix(Z_obs)
    plot_matrix(Z_hat)
    print("Actual:")
    print("Surface Roughness Index:", f.roughness(Z_obs))
    print("Inner Edge Index:", f.get_inner_edge_avg(Z_obs))
    print("Side Skewness Index:", f.skew_index(Z_obs))
    print("Width Peak Index:", f.width_peak_index(Z_obs))
    print("Predicted:")
    print("Surface Roughness Index:", f.roughness(Z_hat))
    print("Inner Edge Index:", f.get_inner_edge_avg(Z_hat))
    print("Side Skewness Index:", f.skew_index(Z_hat))
    print("Width Peak Index:", f.width_peak_index(Z_hat))
    