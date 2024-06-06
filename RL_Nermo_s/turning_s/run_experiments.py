"""
Run multiple experiments after each other.
"""
import os
import json

import numpy as np

EXPERIMENTS = [
{
    "env": "NermoCommandedTurningRadiusEnv-v0",
    "env_kwargs": {
        "frame_skip": 4,
        "terminate_upon_falling": True,
        "knees_on_ground_as_falling": False,
        "falling_penalty": -1,
        "terminate_in_nonfinite_simulation_state": True,
        "energy_penalty_weight": 0,
        "jerkiness_penalty_weight": 0,
        "exclude_head_actuation": True,
        "actuator_velocity_limit": 6,
        #possible observation_space_elements: ["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data", "foot_contact_forces", "joint_velocities", "latest_action"],
        "observation_space_elements": ["actuator_positions", "imu_gyro_data", "imu_accelerometer_data"],
        "omit_positions_of_excluded_actuators": True,
        "use_correct_sensors": True,
        "use_legacy_joint_velocities": False,
        "exclude_tail_actuation": True,
        "knee_on_ground_penalty": -0.01,
        "not_moving_penalty": -0.01,
        "wrong_direction_penalty": -0.01,
        "smoothed_yaw_window_size": 100,
        "deterministic_radius": False,
    },
    "experiment_name": "commanded_turning_radius"
},
{
    "env": "NermoDesiredTurningRadiusEnv-v0",
    "env_kwargs": {
        "frame_skip": 4,
        "terminate_upon_falling": True,
        "knees_on_ground_as_falling": False,
        "falling_penalty": -1,
        "terminate_in_nonfinite_simulation_state": True,
        "energy_penalty_weight": 0,
        "jerkiness_penalty_weight": 0,
        "exclude_head_actuation": True,
        "actuator_velocity_limit": 6,
        #possible observation_space_elements: ["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data", "foot_contact_forces", "joint_velocities", "latest_action"],
        "observation_space_elements": ["actuator_positions", "imu_gyro_data", "imu_accelerometer_data"],
        "omit_positions_of_excluded_actuators": True,
        "use_correct_sensors": True,
        "use_legacy_joint_velocities": False,
        "exclude_tail_actuation": True,
        "knee_on_ground_penalty": -0.01,
        "not_moving_penalty": -0.01,
        "wrong_direction_penalty": -0.01,
        "smoothed_yaw_window_size": 100,
        "desired_turning_radius":-1.0
    },
    "experiment_name": "fixed_turning_radius"
},
]
ALGOS = ["ppo"]
SEEDS = [-1]

for seed in SEEDS:
    for experiment in EXPERIMENTS:
        for algo in ALGOS:
            args = [
                "--env",
                experiment.get("env"),
                "--env-kwargs",
                ("\'" + json.dumps(experiment.get("env_kwargs", {}), separators=(',', ':')).replace('"', '\"') + "\'"),
                "--experiment-name",
                experiment.get("experiment_name"),
                "--algo",
                algo,
                "--seed",
                seed,
                "--vec-env",
                experiment.get("vec_env", "auto")

            ]
            args = list(map(str, args))

            command = f"python train.py {' '.join(args)}"
            print(f"Running 'python train.py {' '.join(args)}'")
            os.system(command)
