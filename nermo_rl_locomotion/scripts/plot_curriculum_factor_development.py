import json
from pathlib import Path
import pandas
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import get_monitor_files

PARENT_DIR = Path(__file__).parent

"""
    Script for plotting the development of curriculum factors in trainings with the ProportionalSecondaryRewardCurriculum and the LinearCurriculum
"""

# Activate seaborn
seaborn.set()

plt.style.use((PARENT_DIR / f"../styles/subfigure.mplstyle").resolve().as_posix())


# Training paths within an experiment with the ProportionalSecondaryRewardCurriculum
# trainings in 2022-03-02_train_psrc_07_exp_adaptions
# training_paths = [
#     "/home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models/NermoCommandedVelocity-v0/2022-03-02_train_psrc_07_exp_adaptions/ppo_5000000_timesteps_seed_1044604204",
#     "/home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models/NermoCommandedVelocity-v0/2022-03-02_train_psrc_07_exp_adaptions/ppo_5000000_timesteps_seed_1779062764",
#     "/home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models/NermoCommandedVelocity-v0/2022-03-02_train_psrc_07_exp_adaptions/ppo_5000000_timesteps_seed_3902921476"
# ]
# trainings in 2022-03-02_train_psrc_07_exp_adaptions_semi_high_velocity
training_paths = [
    "/home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models/NermoFixedVelocity-v0/2022-03-02_train_psrc_07_exp_adaptions_semi_high_velocity/ppo_5000000_timesteps_seed_592523906",
    "/home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models/NermoFixedVelocity-v0/2022-03-02_train_psrc_07_exp_adaptions_semi_high_velocity/ppo_5000000_timesteps_seed_2165862853",
    "/home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models/NermoFixedVelocity-v0/2022-03-02_train_psrc_07_exp_adaptions_semi_high_velocity/ppo_5000000_timesteps_seed_2387668938"
]
# Steps for which the ProportionalSecondaryRewardCurriculum was configured
adaption_steps = [0, 100000, 300000, 700000, 1500000]
training_steps = 5e6
# Index of the curriculum factor in the array of curriculum factors
factor_index = 0

# Read factors from the monitor files
experiment_factors = []
for path in training_paths:
    monitor_files = get_monitor_files(path)
    training_factors = []
    for file_path in monitor_files:
        with open(file_path, "rt") as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            header = json.loads(first_line[1:])
            data_frame = pandas.read_csv(file_handler, index_col=None)
            data_frame["t"] += header["t_start"]
            env_factors = data_frame.get("curriculum_factors")
            env_factors = [factor.replace("[", "") for factor in env_factors]
            env_factors = [factor.replace("]", "") for factor in env_factors]
            env_factors = [list(map(float, factor.split())) for factor in env_factors]
            env_factors = [factor[factor_index] for factor in env_factors]
            env_factors = list(dict.fromkeys(env_factors))
            training_factors.append(env_factors)
    experiment_factors.append(training_factors)


# factors for the trainings in 2022-03-02_train_psrc_07_exp_adaptions
# curriculum_factors = [
#     [[0, 0.0342038, 0.05863679, 0.07294864, 0.07666339], [0, 0.04484876, 0.05486394, 0.05304228, 0.08614718], [0, 0.04876868, 0.0326548, 0.06341298, 0.07027752], [0, 0.03927257, 0.04104999, 0.07803469, 0.06708666]], # seed 1044...
#     [[0, 0.03438703, 0.04790676, 0.05882542, 0.07334447], [0, 0.04726951, 0.04844092, 0.07050533, 0.07484327], [0, 0.02109371, 0.03219345, 0.07475243, 0.08215598], [0, 0.04327164, 0.04937751, 0.06908187, 0.08266653]], # seed 1779...
#     [[0, 0.03552816, 0.038755, 0.05283824, 0.07897427], [0, 0.03925478, 0.03573433, 0.06561351, 0.07456918], [0, 0.05104758, 0.04526928, 0.06800758, 0.07540364], [0, 0.04273559, 0.03630898, 0.06248287, 0.07386959]] # seed 3902...
# ]

avg_curriculum_factors_per_training = [np.mean(training_factors, axis=0) for training_factors in experiment_factors]
avg_curriculum_factors_per_training = [np.append(cs, cs[-1]) for cs in avg_curriculum_factors_per_training]
print(f"Avg. factors per training: {avg_curriculum_factors_per_training}")
avg_curriculum_factors = np.mean(avg_curriculum_factors_per_training, axis=0)
print(f"Avg. factors for the experiment: {avg_curriculum_factors}")
median_curriculum_factors = np.median(avg_curriculum_factors_per_training, axis=0)
print(f"Median factors for the experiment: {median_curriculum_factors}")
min_curriculum_factors = np.min(avg_curriculum_factors_per_training, axis=0)
print(f"Min. factors for the experiment: {min_curriculum_factors}")
max_curriculum_factors = np.max(avg_curriculum_factors_per_training, axis=0)
print(f"Max. factors for the experiment: {max_curriculum_factors}")

x = adaption_steps + [training_steps]
plt.fill_between(x, min_curriculum_factors, max_curriculum_factors, step="post", alpha=0.33, color="tab:green", linewidth=0)
plt.plot(x, median_curriculum_factors, drawstyle='steps-post', color="tab:green", label="Adaptive Curriculum")

plt.xlim(0, training_steps)
plt.xlabel("Steps")
plt.ylabel("Curriculum factor $k_e$")

# Optionally manually plot the development of the curriculum factor with the LinearCurriculum
if True:
    # trainings in 2022-03-05_train_linear_curriculum_to_007
    # plt.plot([0, 1500000, 5000000], [0, 0.07, 0.07], label="Linear Curriculum", color="tab:orange")
    # trainings in 2022-03-06_train_linear_curriculum_to_005_semi_high_velocity
    plt.plot([0, 1500000, 5000000], [0, 0.05, 0.05], label="Linear Curriculum", color="tab:orange")
    plt.legend()

plt.tight_layout()
plt.show()