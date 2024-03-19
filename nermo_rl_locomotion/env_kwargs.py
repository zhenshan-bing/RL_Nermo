# The default environment kwargs are configured here and not in `envs/nermo_envs.py`
# because the defaults in `nermo_envs.py` are set such that legacy agents trained before a more flexible training framework was developed can be loaded.

BASE_ENV_KWARGS = {
    "frame_skip": 4,
    #"penalize_orthogonal_movement": True,
    "terminate_upon_falling": True,
    "knees_on_ground_as_falling": False,
    "falling_penalty": -1,
    "terminate_in_nonfinite_simulation_state": True,
    "energy_penalty_weight": 0,
    "jerkiness_penalty_weight": 0,
    "knee_on_ground_penalty": -0.008,
    "exclude_head_actuation": True,
    "exclude_tail_actuation": True,
    "exclude_lateral_spine_actuation": False,
    "actuator_velocity_limit": 6,
    #possible observation_space_elements: ["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data", "foot_contact_forces", "joint_velocities", "latest_action"],
    #Niklas: use O-Medium for simplicity
    "observation_space_elements": ["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data"],
    "omit_positions_of_excluded_actuators": True,
    "use_correct_sensors": True,
    "use_legacy_joint_velocities": False
}

ENV_KWARGS = {
    "NermoMaxVelocity-v0": {
        **BASE_ENV_KWARGS,
        "energy_penalty_weight": 0,
    },
    "NermoFixedVelocity-v0": {
        **BASE_ENV_KWARGS,
        "desired_velocity": 0.2,
        "velocity_tracking_reward_weight": 0.032,
        "velocity_step_granularity": 0.05,
        "energy_penalty_weight": -0.1
    },
    "NermoCommandedVelocity-v0": {
        **BASE_ENV_KWARGS,
        "velocity_tracking_reward_weight": 0.032,
        "min_velocity": 0.1,
        "max_velocity": 0.25,
        "velocity_step_granularity": 0.05,
        "command_sample_frequency": 10,
    },
    "nermo-short-episodes-v0": BASE_ENV_KWARGS,
    "nermo-long-episodes-v0": BASE_ENV_KWARGS,

    #below are the new environments added in my (Niklas Ganter) master thesis
    #This was the first environment developed for turning behavior
    #Goal: Learn a desired turning radius with a desired yaw rate
    "NermoDesiredTurningRadiusEnv-v0": {
        **BASE_ENV_KWARGS,
        "exclude_tail_actuation": True,
        "exclude_lateral_spine_actuation": False,
        "knee_on_ground_penalty": -0.01,
        "desired_turning_radius": 2.0,
        "desired_yaw_rate": 0.05,
        "max_deviation": 0.2
    },
    #Goal: Learn to command a turning radius {-2, -1, 1, 2}[m] with a desired yaw rate
    "NermoCommandedTurningRadiusEnv-v0": {
        **BASE_ENV_KWARGS,
        "exclude_lateral_spine_actuation": False,
        "knee_on_ground_penalty": -0.01,
        "not_moving_penalty": -0.01, 
        "wrong_direction_penalty": -0.01,
        #This variable is to decide, if a randomly sampled radius should be picked for training or specific one (for evaluation)
        "deterministic_radius": False,
        #we insert the variable "desired_velocity" here instead of "desired_yaw_rate", because the values for velocity are more intuitive.
        #still, there is no direct velocity tracking in the envrionment, the velocity is just used to calculate the desired yaw rate according the 
        #relationship v = w*r
        "desired_velocity":0.157,
        "smoothed_yaw_window_size": 200
    },
    #Goal: Learn to command a turning radius {-2, -1, 1, 2}[m] with a desired yaw rate and speed
    #Note: Actually the yaw rate and speed are related by: v=w*r, but it turned out that the results get better when we reward both, yaw rate and speed
    "NermoCommandedTurningRadiusAndFixedVelocityEnv-v0": {
        **BASE_ENV_KWARGS,
        "exclude_tail_actuation": True,
        "exclude_lateral_spine_actuation": False,
        "knee_on_ground_penalty": -0.01,
        "not_moving_penalty": -0.01, 
        "wrong_direction_penalty": -0.01,
        "desired_velocity": 0.15,
        "deterministic_radius": False,
        "smoothed_speed_window_size": 100,
        "smoothed_yaw_window_size": 100
    },
    #Goal: Learn to command a turning radius with a commanded yaw rate and speed
    "NermoCommandedTurningRadiusAndCommandedVelocityEnv-v0": {
        **BASE_ENV_KWARGS,
        "exclude_tail_actuation": True,
        "exclude_lateral_spine_actuation": False,
        "deterministic_radius": False,
        "smoothed_speed_window_size": 100,
        "knee_on_ground_penalty": -0.01,
        "not_moving_penalty": -0.01,
        "wrong_direction_penalty": -0.01
    }
}


EVAL_KWARGS = {
    "falling_penalty": 0,
    "energy_penalty_weight": 0,
    "jerkiness_penalty_weight": 0,
    "knee_on_ground_penalty": 0,
    #"not_moving_penalty": -0.01,
    #"wrong_direction_penalty": -0.01,
    "reward_curriculum_class_name": None,
    "reward_curriculum_kwargs": None,

    # "max_deviation": 2,  # to exclude stopping when agent is too far away from desired circle
    #Note: Here we set deterministic to True, to select a deterministic radius in the evaluation environment
    #Only needed for the classes NermoCommandedTurningRadiusEnv, NermoCommandedTurningRadiusAndFixedVelocityEnv and NermoCommandedTurningRadiusAndCommandedVelocityEnv 
    # "deterministic_radius": True
}


# Keywords for logging metrics with the NermoMonitor (cf. `utils/nermo_monitor.py`)
directed_env_monitor_keywords = [
    "primary_reward",
    #"secondary_reward",
    "invariable_penalties",
    #"secondary_rewards",
    ##"curriculum_factors",
    "speed",
    #"power"
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
    "NermoDesiredTurningRadiusEnv-v0": {
        "info_keywords": directed_env_monitor_keywords + ["invariable_penalties", "spine_angle", "displacement_reward", "yaw_reward", "power", "position", "desired_turning_radius", "desired_yaw_rate", "circle_center", "yaw_rate", "smoothed_yaw_rate"]
    },
    "NermoCommandedTurningRadiusEnv-v0": {
        "info_keywords": directed_env_monitor_keywords + ["not_moving_penalty", "wrong_direction_penalty", "knee_on_ground_penalty", "falling_penalty", "invariable_penalties", "spine_angle", "displacement_reward", "yaw_reward", "power", "position", "desired_turning_radius", "desired_yaw_rate", "circle_center", "yaw_rate", "smoothed_yaw_rate"]
    },
    "NermoCommandedTurningRadiusAndFixedVelocityEnv-v0": {
        "info_keywords": directed_env_monitor_keywords + ["not_moving_penalty", "wrong_direction_penalty", "knee_on_ground_penalty", "falling_penalty", "invariable_penalties", "spine_angle", "displacement_reward", "yaw_reward", "desired_velocity", "velocity_tracking_reward", "power", "position", "desired_turning_radius", "desired_yaw_rate", "circle_center", "yaw_rate", "smoothed_yaw_rate"]
    },
    "NermoCommandedTurningRadiusAndCommandedVelocityEnv-v0": {
       "info_keywords": directed_env_monitor_keywords + ["not_moving_penalty", "wrong_direction_penalty", "knee_on_ground_penalty", "falling_penalty", "invariable_penalties", "spine_angle", "displacement_reward", "yaw_reward", "desired_velocity", "velocity_tracking_reward", "power", "position", "desired_turning_radius", "desired_yaw_rate", "circle_center", "yaw_rate", "smoothed_yaw_rate"]
    }
}
