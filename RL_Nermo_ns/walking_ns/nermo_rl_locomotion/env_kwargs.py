# The default environment kwargs are configured here and not in `envs/nermo_envs.py`
# because the defaults in `nermo_envs.py` are set such that legacy agents trained before a more flexible training framework was developed can be loaded.

BASE_ENV_KWARGS = {
    "frame_skip": 4,
    "penalize_orthogonal_movement": True,
    "terminate_upon_falling": True,
    "knees_on_ground_as_falling": True,
    "falling_penalty": -0.5,
    "terminate_in_nonfinite_simulation_state": True,
    "energy_penalty_weight": 0.0,
    "jerkiness_penalty_weight": 0.0,
    "knee_on_ground_penalty": -0.5,
    "exclude_head_actuation": True,
    "exclude_tail_actuation": True,
    "exclude_lateral_spine_actuation": False,
    "actuator_velocity_limit": 6,
    # possible observation_space_elements: ["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data", "foot_contact_forces", "joint_velocities", "latest_action"],
    # Niklas: use O-Medium for simplicity
    "observation_space_elements": ["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data"],
    "omit_positions_of_excluded_actuators": True,
    "use_correct_sensors": True,
    "use_legacy_joint_velocities": False,
    # "reward_curriculum_class_name": "LinearTimeCurriculum",
    # "reward_curriculum_kwargs": {"initial_curriculum_factors": [0, 1, 1], "max_curriculum_factors": [0.07, 1, 1],
    #                         "increase_per_step": [0.07 / 3000000*8, 0, 0]}
}

ENV_KWARGS = {
    "NermoMaxVelocity-v0": {
        **BASE_ENV_KWARGS,
        "energy_penalty_weight": -0.0,
    },
    "NermoFixedVelocity-v0": {
        **BASE_ENV_KWARGS,
        "desired_velocity": 0.2,
        "velocity_tracking_reward_weight": 0.032,  # 0.032,
        "velocity_step_granularity": 0.2/4,
        "energy_penalty_weight": -1
    },
    "NermoCommandedVelocity-v0": {
        **BASE_ENV_KWARGS,
        "velocity_tracking_reward_weight": 0.032,
        "min_velocity": 0.08,
        "max_velocity": 0.20,
        "velocity_step_granularity": 0.04,
        "command_sample_frequency": 10,
    },

}

EVAL_KWARGS = {
    "falling_penalty": -1,
    "energy_penalty_weight": 0,
    "jerkiness_penalty_weight": 0,
    "knee_on_ground_penalty": -1,
    # "not_moving_penalty": -0.01,
    #"wrong_direction_penalty": -0.01,
    "reward_curriculum_class_name": None,
    "reward_curriculum_kwargs": None,
    #Note: Here we set deterministic to True, to select a deterministic radius in the evaluation environment
    #Only needed for the classes NermoCommandedTurningRadiusEnv, NermoCommandedTurningRadiusAndFixedVelocityEnv and NermoCommandedTurningRadiusAndCommandedVelocityEnv 
    #"deterministic_radius": True
}


# Keywords for logging metrics with the NermoMonitor (cf. `utils/nermo_monitor.py`)
#directed_env_monitor_keywords = [
#    "primary_reward",
    #"secondary_reward",
#    "invariable_penalties",
    #"secondary_rewards",
    ##"curriculum_factors",
#    "speed",
    #"power"
# ]

# Keywords for logging metrics with the NermoMonitor (cf. `utils/nermo_monitor.py`)
directed_env_monitor_keywords = [
    "primary_reward",
    "secondary_reward",
    "invariable_penalties",
    "secondary_rewards",
    "curriculum_factors",
    "speed",
    "power",
    "velocity_in_desired_direction",
    "velocity_in_orthogonal_direction"
]
MONITOR_KWARGS = {
    "NermoMaxVelocity-v0": {
        "info_keywords": directed_env_monitor_keywords
    },
    "NermoFixedVelocity-v0": {
        "info_keywords": directed_env_monitor_keywords + ["velocity_deviation", "abs_velocity_deviation"]
    },
    "NermoCommandedVelocity-v0": {
        "info_keywords": directed_env_monitor_keywords + ["desired_velocity", "velocity_deviation", "abs_velocity_deviation"]
    },
    "nermo-short-episodes-v0": {
        "info_keywords": directed_env_monitor_keywords
    },
    "nermo-long-episodes-v0": {
        "info_keywords": directed_env_monitor_keywords
    },
    
}

