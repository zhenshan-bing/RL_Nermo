import numpy as np
import seaborn
import matplotlib.pyplot as plt

# Activate seaborn
seaborn.set()

fig, ax = plt.subplots()

# Data for the stance phases (time in msec) depicted in the diagram of Hruska et al. 1979 - Format for each phase: (t_begin, t_end)
foot_stance_phases = {
    "lh": [(74.42, 502.335), (613.965, 1041.88), (1162.8125, 1562.82), (1674.45, 2000)],
    "lf": [(0, 191.6315), (297.68, 725.595), (874.435, 1265.14), (1395.375, 1720.9625), (1897.71, 2000)],
    "rf": [(37.21, 409.31), (539.545, 967.46), (1097.695, 1488.4), (1637.24, 1972.13)],
    "rh": [(0, 130.235), (334.89, 818.62), (911.645, 1339.56), (1451.19, 1786.08), (1916.315, 2000)]
}
# Convert phases to format (t_begin, duration) required for matplotlib's `broken_barh` 
for foot, stance_phases in foot_stance_phases.items():
    foot_stance_phases[foot] = [(phase[0], phase[1] - phase[0]) for phase in foot_stance_phases[foot]]

# Scale phase times to seconds
time_scaling_factor = 1/1000
for foot, stance_phases in foot_stance_phases.items():
    foot_stance_phases[foot] = [(phase[0] * time_scaling_factor, phase[1] * time_scaling_factor) for phase in foot_stance_phases[foot]]


hildebrand_foot_order=["rh", "rf", "lf", "lh"]
t_start = 0
t_end = 2
time_unit = "sec"

for i, foot in enumerate(hildebrand_foot_order):
    ax.broken_barh(list(foot_stance_phases[foot]), ((i * 5) + 4, 2), facecolors='tab:blue')

ax.set_ylim(1, 24)
ax.set_xlim(t_start, t_end)
ax.set_xlabel(f"Time [{time_unit}]")
ax.set_yticks([5, 10, 15, 20])
ax.set_yticklabels([label.upper() for label in hildebrand_foot_order])
ax.grid(True)


plt.tight_layout()
plt.show()
