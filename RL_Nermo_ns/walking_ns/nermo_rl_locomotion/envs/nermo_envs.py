from collections import deque
from pathlib import Path
import math
import timeit
import random
import nermo_rl_locomotion
from nermo_rl_locomotion.utils.utils import logistic_kernel, custom_kernel_1, custom_kernel_2, custom_kernel_3, custom_kernel_4, speed_kernel
from nermo_rl_locomotion.utils import reward_curricula
import numpy as np
from mujoco_py.generated import const
from gym import spaces, utils
from gym.envs.mujoco.mujoco_env import MujocoEnv
import mujoco_py
import sys

# Build paths relative to the absolute path of this script (or rather the script's parent dir)
# Source: https://stackoverflow.com/a/55051039
BASE_PATH = Path(__file__).parent
MUJOCO_MODEL_PATH = (BASE_PATH / "../../models/dynamic_4l.xml").resolve()

"""
    Gym training environments for NeRmo. 
    The default args for training should be configured in `env_kwargs.py`. 
    The defaults set here were configured such that 
    legacy agents trained with earlier iterations of the environments can be loaded without an issue.
"""

class NermoBaseEnv(MujocoEnv, utils.EzPickle):
    def __init__(
            self,
            model_path=None,
            frame_skip=4,
            terminate_upon_falling=True,
            knees_on_ground_as_falling=True,
            falling_penalty=-0,
            terminate_in_nonfinite_simulation_state=False,
            energy_penalty_weight=-0.00,
            jerkiness_penalty_weight=-0.0,
            knee_on_ground_penalty=0,
            exclude_head_actuation=True,
            exclude_tail_actuation=True,
            exclude_lateral_spine_actuation=False,
            actuator_velocity_limit=None,
            observation_space_elements=["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data", "latest_action"],
            omit_positions_of_excluded_actuators=False,
            fixed_cam_id=None,
            default_render_width=1044,
            default_render_height=810,
            reward_curriculum_class_name=None,
            reward_curriculum_kwargs=None,
            use_correct_sensors=False,
            use_legacy_joint_velocities=True,
            action_smoothing_window=None,
            action_offset_range=None,
            seed_generator=0,

    ):
        utils.EzPickle.__init__(self)

        # define seed for random generator in the turning environments with commanded radii
        self.seed_generator = seed_generator

        # define Return (named reward_return, because self.return is reserved by python)
        self.reward_return = 0

        # Termination Criteria
        self.terminate_upon_falling = terminate_upon_falling
        self.knees_on_ground_as_falling = knees_on_ground_as_falling
        self.falling_penalty = falling_penalty
        self.terminate_in_nonfinite_simulation_state = terminate_in_nonfinite_simulation_state

        # Penalty weights
        self.energy_penalty_weight = energy_penalty_weight
        self.jerkiness_penalty_weight = jerkiness_penalty_weight
        self.knee_on_ground_penalty = knee_on_ground_penalty

        # Observation space configuration
        self.observation_space_elements = observation_space_elements
        self.omit_positions_of_excluded_actuators = omit_positions_of_excluded_actuators

        # Action space configuration
        # Exclusion of actuators from the action space
        self.exclude_head_actuation = exclude_head_actuation
        self.exclude_tail_actuation = exclude_tail_actuation
        self.exclude_lateral_spine_actuation = exclude_lateral_spine_actuation
        self.actuator_indices_to_exclude = []       # set by _set_action_space()
        # Optionally use offset to the current position within the specified range as the action ...
        # instead of the absolute position (experimental/not recommended)
        self.action_offset_range = action_offset_range 
        # Optional smoothing of actions (experimental/not recommended)
        self.action_smoothing_window = action_smoothing_window

        # Limit for actuators velocities
        self.actuator_velocity_limit = actuator_velocity_limit

        # Rendering defaults
        self.fixed_cam_id = fixed_cam_id
        self.default_render_width = default_render_width
        self.default_render_height = default_render_height

        # Whether to use faulty legacy implementations (for correct loading of legacy agents).
        # New agents should always use the correct implementations
        # (the defaults are properly configured in `env_kwargs.py`)

        self.use_correct_sensors = use_correct_sensors
        # Exists because the built-in method for reading sensor values with mujoco-py is erroneous

        self.use_legacy_joint_velocities = use_legacy_joint_velocities
        
        # Reward curriculum
        if reward_curriculum_class_name is not None:
            curriculum_class = getattr(reward_curricula, reward_curriculum_class_name)
            self.reward_curriculum: reward_curricula.BaseCurriculum = curriculum_class(**reward_curriculum_kwargs)
        else:
            self.reward_curriculum = None

        # Set initial state
        self.total_steps = 0
        self.episode = 0
        self.episode_step = 0

        self.previous_action = None
        self.latest_action = None
        if self.action_smoothing_window is not None:
            self.action_history = deque([], maxlen=self.action_smoothing_window)
        else:
            self.action_history = None

        self.previous_joint_postions = None
        self.previous_actuator_positions = None

        # For calculating angle_displacements in the turning environments
        self.turning_angle_old = 0
        self.turning_angle_new = 0

        # Initialize MuJoCo environment
        MujocoEnv.__init__(self, model_path if model_path is not None else MUJOCO_MODEL_PATH.as_posix(), frame_skip)

        self.previous_mouse_xy_position = None
        self.current_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)


        self.power = 0
        self.speed = 0

        self.info = {}

    def _set_action_space(self):
        self.actuator_indices_to_exclude = []
        if self.exclude_head_actuation:
            neck_actuator_id = self.model.actuator_name2id("neck")
            head_actuator_id = self.model.actuator_name2id("head")
            self.actuator_indices_to_exclude.append(neck_actuator_id)
            self.actuator_indices_to_exclude.append(head_actuator_id)

        if self.exclude_tail_actuation:
            tail_actuator_id = self.model.actuator_name2id("m1_tail")
            self.actuator_indices_to_exclude.append(tail_actuator_id)

        if self.exclude_lateral_spine_actuation:
            spine_actuator_id = self.model.actuator_name2id("spine")
            self.actuator_indices_to_exclude.append(spine_actuator_id)
 
        if self.action_offset_range is None:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            
            for index in sorted(self.actuator_indices_to_exclude, reverse=True):
                bounds = np.delete(bounds, index, axis=0)
            for i in range(8):
                bounds = np.delete(bounds, i+1, axis=0)
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            n_active_actuators = self.model.nu - len(self.actuator_indices_to_exclude)
            self.action_space = spaces.Box(low=self.action_offset_range[0], high=self.action_offset_range[1], shape=(n_active_actuators,), dtype=np.float32)

        return self.action_space
    
    def step(self, action, i=None):
        self.info = {}

        self.previous_action = self.latest_action
        # self.latest_action = action
        if self.action_history is not None:
            self.action_history.append(action)
            if self.action_smoothing_window is not None:
                action = np.mean(self.action_history, axis=0)
        for i in range(8):
            action = np.insert(action, 2*i+1, 0)
        for index in sorted(self.actuator_indices_to_exclude):
            action = np.insert(action, index, 0)

        # get previous positions before simulation step in order to calculate velocities
        self.previous_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)

        self.do_simulation(action, self.frame_skip)

        # get current positions before simulation step in order to calculate velocities
        self.current_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)
        observation = self._get_observation()

        self._compute_velocities()
        self._compute_power_consumption()

        if self.reward_curriculum is not None:
            self.reward_curriculum.step_curriculum(self._primary_reward, self._secondary_rewards)

        reward = self._get_reward()

        # Calculate return by summing up the collected rewards
        self.reward_return += reward
        done = self._is_done()

        self.episode_step += 1
        self.total_steps += 1

        self.info = {
            "episode_num": self.episode,
            "step": self.episode_step,
            "primary_reward": self._primary_reward,
            "secondary_rewards": self._secondary_rewards,
            "secondary_reward": self._secondary_reward,
            "invariable_penalties": self._invariable_penalties,
            "return": self.reward_return,
            "curriculum_factors": self.reward_curriculum.curriculum_factors if self.reward_curriculum is not None else None,
            "has_fallen": self._has_fallen(),
            "position": self.current_mouse_xy_position,
            "speed": self.speed,
            "power": self.power,
            "foot_contacts": {
                "lf": self._has_contact(["foot_fl"]),
                "rf": self._has_contact(["foot_fr"]),
                "rh": self._has_contact(["foot_rr"]),
                "lh": self._has_contact(["foot_rl"])
            },
            "foot_positions": {
                "lf": self.get_sensor('fl_foot_pos', dimensions=3, use_correct_sensors=True),
                "rf": self.get_sensor('fr_foot_pos', dimensions=3, use_correct_sensors=True),
                "rh": self.get_sensor('rr_foot_pos', dimensions=3, use_correct_sensors=True),
                "lh": self.get_sensor('rl_foot_pos', dimensions=3, use_correct_sensors=True)
            },
            **self.info
        }
        return observation, reward, done, self.info

    def do_simulation(self, ctrl, n_frames):
        self.latest_action = ctrl
        if self.action_offset_range is None:
            self.sim.data.ctrl[:] = ctrl
            self.info = {
                "ctrl": ctrl,
                "actuator_force": self.sim.data.actuator_force,
                ** self.info
            }
        else:
            self.sim.data.ctrl[:] += ctrl
        for _ in range(n_frames):
            if self.actuator_velocity_limit is not None:
                # Limit actuator velocities (data sheet claims 6.5 rad/s https://www.pololu.com/file/0J1754/DSM44.pdf but without load)
                # Note: velocity value may be computed to be above the specified limit within the next simulation step due to acceleration so the limit should be set slightly below the real maximum
                # Note: gather empirical speed and torque values for DSM44 servos from real world for more realistic actuator model
                # self.data.actuator_velocity[:] = np.clip(self.data.actuator_velocity, -self.actuator_velocity_limit, self.actuator_velocity_limit)
                self.data.qvel[:] = np.clip(self.data.qvel, -self.actuator_velocity_limit, self.actuator_velocity_limit)

            self.info = {
                "qvel": self.data.qvel[:],
                **self.info
            }

            self.sim.step()
            self.info = {
                "actuator_velocity": self.data.actuator_velocity[:],
                **self.info
            }

            #render only when running enjoy.py script
            script_name = sys.argv[0].split('/')[-1]  # Get the script name from the path
            if script_name == "enjoy.py":
                self.render()
            else:
                pass

    def reset(self):
        self.episode += 1
        self.episode_step = 0

        self.previous_action = None
        self.latest_action = None
        if self.action_history is not None:
            self.action_history = deque([], maxlen=self.action_history.maxlen)
        else:
            self.action_history = None

        self.previous_joint_postions = None
        self.previous_actuator_positions = None

        self.reward_return = 0

        return super().reset()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        # Run some initial simulation steps for the model to settle 
        # as it is initialized floating above the ground it needs some time to reach the designated intital configuration
        for _ in range(100):
            # example for initializing with special configuration of actuators
            self.sim.data.ctrl[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            #self.sim.data.ctrl[:] = np.array(self.action_space.sample())
            self.sim.step()
        
        return self._get_observation()

    def viewer_setup(self):
        if self.fixed_cam_id is not None:
            self.viewer.cam.fixedcamid = self.fixed_cam_id
            self.viewer.cam.type = const.CAMERA_FIXED  # Niklas change from CAMERA_FIXED
    
    def render(
        self,
        mode="human",
        width=None,
        height=None,
        camera_id=None,
        camera_name=None,
    ):
        width = self.default_render_width if width is None else width
        height = self.default_render_height if height is None else height
        return super().render(mode=mode, width=width, height=height, camera_id=camera_id, camera_name=camera_name)

    def _get_observation(self):
        actuator_positions = np.concatenate(
            (
                self.get_sensor('thigh_joint_fl'),
                self.get_sensor('leg_joint_fl'),
                self.get_sensor('thigh_joint_fr'),
                self.get_sensor('leg_joint_fr'),
                self.get_sensor('thigh_joint_rl'),
                self.get_sensor('leg_joint_rl'),
                self.get_sensor('thigh_joint_rr'),
                self.get_sensor('leg_joint_rr'),
                self.get_sensor('m1_tail') if not (self.exclude_tail_actuation and self.omit_positions_of_excluded_actuators) else [],
                self.get_sensor('neck') if not (self.exclude_head_actuation and self.omit_positions_of_excluded_actuators) else [],
                self.get_sensor('head') if not (self.exclude_head_actuation and self.omit_positions_of_excluded_actuators) else [],
                self.get_sensor('spine') if not (self.exclude_lateral_spine_actuation and self.omit_positions_of_excluded_actuators) else [],
            )
        )

        knee_joint_positions = np.concatenate(
            (
                self.get_sensor('knee_fl'),
                self.get_sensor('knee_fr'),
                self.get_sensor('knee_rl'),
                self.get_sensor('knee_rr'),
            )
        )

        imu_gyro_data = self.get_sensor('imu_gyro', dimensions=3)
        imu_accelerometer_data = self.get_sensor('imu_acc', dimensions=3)

        foot_contact_forces = np.concatenate(
            (
                self.get_sensor('fl_t1'),
                self.get_sensor('fr_t1'),
                self.get_sensor('rl_t1'),
                self.get_sensor('rr_t1'),
            )
        )

        if self.use_legacy_joint_velocities:
            actuator_velocities = [0]
        elif self.previous_actuator_positions is not None:
            actuator_velocities = (actuator_positions - self.previous_actuator_positions) / self.dt
        else:
            actuator_velocities = [0] * len(actuator_positions)
        self.previous_actuator_positions = actuator_positions

        if self.use_legacy_joint_velocities:
            knee_joint_velocities = [0]
        elif self.previous_joint_postions is not None:
            knee_joint_velocities = (knee_joint_positions - self.previous_joint_postions) / self.dt
        else:
            knee_joint_velocities = [0] * len(knee_joint_positions)
        self.previous_joint_postions = knee_joint_positions

        # Dictionary with all available observations
        observation_space_dict = {
            "actuator_positions": actuator_positions, 
            "knee_joint_positions": knee_joint_positions,
            "imu_gyro_data": imu_gyro_data,
            "imu_accelerometer_data": imu_accelerometer_data,
            "foot_contact_forces": foot_contact_forces,
            "joint_velocities": np.concatenate((actuator_velocities, knee_joint_velocities)),
            "latest_action": self.latest_action if self.latest_action is not None else [0] * len(self.action_space.sample())
        }

        if self.action_history is not None:
            flattened_action_history = [control_value for action in self.action_history for control_value in action]
            # Create a padding for the action history to ensure that the observation space always has the same dimension (even if only the first action is in the action history)
            virtual_action_history = ([0] * len(self.action_space.sample()) * (self.action_history.maxlen - len(self.action_history)))
            observation_space_dict["action_history"] = virtual_action_history + flattened_action_history
        else:
            observation_space_dict["action_history"] = np.concatenate(
                (
                    self.previous_action if self.previous_action is not None else [0] * len(self.action_space.sample()),
                    self.latest_action if self.latest_action is not None else [0] * len(self.action_space.sample())
                )
            )

        # Filter observations to contain only the specified elements
        observation_space_dict = {
            key: observation_space_dict[key] for key in self.observation_space_elements
        }

        # Return a flattened observations space (1D array of numbers)
        return np.concatenate(list(observation_space_dict.values()))

    @property
    def _primary_reward(self):
        return 0

    @property
    def _secondary_rewards(self):
        # based on the energy penalty in "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots" p. 3-4
        energy_penalty = self.energy_penalty_weight * self.dt * self.power

        # Encourage smoothness of actions / discourage successive actions being far apart
        if self.previous_action is not None and self.latest_action is not None:
            jerkiness_penalty = self.jerkiness_penalty_weight * np.linalg.norm(self.latest_action - self.previous_action)
        else:
            jerkiness_penalty = 0

        return [energy_penalty, jerkiness_penalty]

    @property
    def _secondary_reward(self):
        if self.reward_curriculum is None:
            return np.sum(self._secondary_rewards)
        else:
            return np.dot(self.reward_curriculum.curriculum_factors, self._secondary_rewards)

    @property
    def _invariable_penalties(self):
        falling_penalty = int(self._has_fallen()) * self.falling_penalty
        knee_on_ground_penalty = int(self._knee_on_ground()) * self.knee_on_ground_penalty
        self.info={
            **self.info,
            "knee_on_ground_penalty": knee_on_ground_penalty,
            "falling_penalty": falling_penalty
        }

        return falling_penalty + knee_on_ground_penalty
    
    def _get_reward(self):
        return self._primary_reward + self._secondary_reward + self._invariable_penalties

    def _compute_velocities(self):
        displacement = self.current_mouse_xy_position - self.previous_mouse_xy_position
        distance_traveled = np.linalg.norm(displacement)
        self.speed = distance_traveled / self.dt
        
    def _compute_power_consumption(self):
        # Important: The actuator torque is equal to the actuator force as all actuators have gear=1 (could be generalized to different gears via processing `self.sim.model.actuator_gear`)
        # Power = sum_{j=1}^{num_actuators}: |torque_of_actuator_j * angular_velocity_of_actuator_j|
        self.power = np.dot(np.abs(self.data.actuator_force), np.abs(self.data.actuator_velocity))


    def _knee_on_ground(self):
        # changed knee name, added corresponding geoms in Mujoco files
        knee_geoms = ["knee_fl_touch", "knee_fr_touch", "knee_rr_touch", "knee_rl_touch"]

        return self._has_contact(knee_geoms)
    
    def _has_fallen(self):
        geoms_that_should_not_have_contact = ["mouse_body", "spine_mouse"]
        if self.knees_on_ground_as_falling:
            geoms_that_should_not_have_contact += ["knee_fl_touch", "knee_fr_touch", "knee_rr_touch", "knee_rl_touch"]

        return self._has_contact(geoms_that_should_not_have_contact)

    def _has_contact(self, geoms):
        # Inspired by: https://gist.github.com/machinaut/209c44e8c55245c0d0f0094693053158
        for i in range(self.data.ncon):
            contact = self.data.contact[i]  # get contact list
            # every contact contains two geom
            if self.model.geom_id2name(contact.geom1) in geoms or self.model.geom_id2name(contact.geom2) in geoms:
                return True
        return False

    def _is_done(self):
        # ------- Episode Termination -------
        # An episode terminates:
        #   * after X timesteps (X is configured upon environment creation)
        #   * when NeRmo falls (i.e., body is in contact with the ground)
        #   * when the simulation state contains numbers that are not finite
        #  cf. https://github.com/openai/gym/blob/a5c0608efbadfd0a1097aaff284885be18129427/gym/envs/mujoco/ant_v3.py#L69
        if self.terminate_upon_falling and self._has_fallen():
            return True
        if self.terminate_in_nonfinite_simulation_state and not np.isfinite(self.state_vector()).all():
            return True
        return False

    def get_sensor(self, name, dimensions=1, use_correct_sensors=None):
        if use_correct_sensors is None:
            use_correct_sensors = self.use_correct_sensors

        '''
        # We need to maintain our own mapping from sensor names to indices as mujoco_py fails ...
            in the presence of sensors with multidimensional outputs 
            (https://github.com/openai/mujoco-py/issues/193#issuecomment-458697015)
        # Normal way of accessing sensor data: https://github.com/openai/mujoco-py/issues/26#issuecomment-446236206
        # Issues: 
        #   - sensor_name2id in the original get_sensor does not consider sensors with multidimensional outputs ...
            and always increments the id by 1
        #   - the original get_sensor method only returns one value for each sensor name 
            (https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi#L2774)
        # Solution: https://github.com/openai/mujoco-py/issues/193#issuecomment-458697015 
            + maintain own map from sensor name to sensor index
        '''
        # Yulong changed: actuator joint sensor names
        name2index_map = {
            "thigh_joint_fl": 0,
            "leg_joint_fl": 1,
            "thigh_joint_fr": 2,
            "leg_joint_fr": 3,
            "thigh_joint_rl": 4,
            "leg_joint_rl": 5,
            "thigh_joint_rr": 6,
            "leg_joint_rr": 7,
            "m1_tail": 8,
            "neck": 9,
            "head": 10,
            "spine": 11,
            "knee_fl": 12,
            "knee_fr": 13,
            "knee_rl": 14,
            "knee_rr": 15,
            "fl_t1": 16,
            "fr_t1": 17,
            "rl_t1": 18,
            "rr_t1": 19,
            "com_pos": 20,
            "com_quat": 23,
            "com_vel": 27,
            "imu_acc": 30,
            "imu_gyro": 33,
            "fl_foot_pos": 36,
            "fr_foot_pos": 39,
            "rl_foot_pos": 42,
            "rr_foot_pos": 45,
        }

        if use_correct_sensors:
            sensor_index = name2index_map[name]
        else:
            sensor_index = self.sim.model.sensor_name2id(name)

        return self.sim.data.sensordata[sensor_index:sensor_index+dimensions].copy()

class NermoDirectedEnv(NermoBaseEnv):
    def __init__(self, *args, desired_direction=[0, -1], penalize_orthogonal_movement=False, orthogonal_penalty_weight=-1, **kwargs):
        self.desired_direction = desired_direction[:2]
        self.desired_direction /= np.linalg.norm(desired_direction)

        self.penalize_orthogonal_movement = penalize_orthogonal_movement
        self.orthogonal_penalty_weight = orthogonal_penalty_weight
        
        # Construct vector that is orthogonal to desired direction based on https://stackoverflow.com/a/33658815
        self.orthogonal_direction = np.array([1.0, 1.0])
        self.orthogonal_direction -= self.orthogonal_direction.dot(self.desired_direction) * self.desired_direction
        self.orthogonal_direction /= np.linalg.norm(self.orthogonal_direction)

        self.velocity_in_desired_direction = 0
        self.displacement_in_desired_direction = 0
        self.displacement_in_orthogonal_direction = 0
        self.velocity_in_orthogonal_direction = 0
        self.smoothed_yaw_rate_vector = np.array([0])
        self.smoothed_yaw_rate = 0

        super().__init__(*args, **kwargs)

    def _compute_velocities(self):
        super()._compute_velocities()
        displacement = self.current_mouse_xy_position - self.previous_mouse_xy_position
        self.displacement_in_desired_direction = np.dot(displacement, self.desired_direction)
        self.velocity_in_desired_direction = self.displacement_in_desired_direction / self.dt

        self.displacement_in_orthogonal_direction = np.abs(np.dot(displacement, self.orthogonal_direction))
        self.velocity_in_orthogonal_direction = self.displacement_in_orthogonal_direction / self.dt
        self.yaw_rate = abs(self.get_sensor("imu_gyro", 3, use_correct_sensors=True)[2])
        # smoothing the yaw_rate (in desired window size)
        if self.smoothed_yaw_rate_vector.size < 50:
            self.smoothed_yaw_rate_vector = np.append(self.smoothed_yaw_rate_vector, self.yaw_rate)
        else:
            self.smoothed_yaw_rate_vector = np.append(self.smoothed_yaw_rate_vector, self.yaw_rate)[1:]
        self.smoothed_yaw_rate = np.mean(self.smoothed_yaw_rate_vector)

        self.info = {
            **self.info,
            "velocity_in_desired_direction": self.velocity_in_desired_direction,
            "velocity_in_orthogonal_direction": self.velocity_in_orthogonal_direction,
            "spine_angle": self.get_sensor("spine", 1, use_correct_sensors=True)[0],
            "angle_m1_rl": self.get_sensor("leg_joint_rl", 1, use_correct_sensors=True)[0],
            "angle_m1_rr": self.get_sensor("leg_joint_rr", 1, use_correct_sensors=True)[0],
            "angle_m1_r_max": 0.2,
            "angle_m1_r_min": -2.6,
            "angle_m2_rl": self.get_sensor("thigh_joint_rl", 1, use_correct_sensors=True)[0],
            "angle_m2_rr": self.get_sensor("thigh_joint_rr", 1, use_correct_sensors=True)[0],
            "angle_m2_r_max": 1,
            "angle_m2_r_min": -2.6,
            "angle_m1_fl": self.get_sensor("leg_joint_fl", 1, use_correct_sensors=True)[0],
            "angle_m1_fr": self.get_sensor("leg_joint_fr", 1, use_correct_sensors=True)[0],
            "angle_m1_f_max": 0.2,
            "angle_m1_f_min": -2.6,
            "angle_m2_fl": self.get_sensor("thigh_joint_fl", 1, use_correct_sensors=True)[0],
            "angle_m2_fr": self.get_sensor("thigh_joint_fr", 1, use_correct_sensors=True)[0],
            "angle_m2_f_max": 1,
            "angle_m2_f_min": -2.6,
            "smoothed_yaw_rate": self.smoothed_yaw_rate
        }

    @property
    def _secondary_rewards(self):
        """
        Compute optional penalty for movement orthogonal to the desired direction
        """        
        motion_in_orthogonal_direction_penalty = self.displacement_in_orthogonal_direction * self.orthogonal_penalty_weight * int(self.penalize_orthogonal_movement)
        return super()._secondary_rewards + [motion_in_orthogonal_direction_penalty]

class NermoMaxVelocityEnv(NermoDirectedEnv):
    @property
    def _primary_reward(self):
        """
        Encourage moving as fast as possible in the given direction
        """
        return self.displacement_in_desired_direction + super()._primary_reward

class NermoDesiredVelocityEnv(NermoDirectedEnv):
    def __init__(self, *args, velocity_tracking_reward_weight=0.008, velocity_step_granularity=0.01, **kwargs):
        self.velocity_tracking_reward_weight = velocity_tracking_reward_weight
        self.velocity_step_granularity = velocity_step_granularity
        self.desired_velocity = 0

        super().__init__(*args, **kwargs)

    @property
    def _primary_reward(self):
        """
        Encourage moving at the desired velocity in the given direction
        """
        velocity_deviation = self.velocity_in_desired_direction - self.desired_velocity
        velocity_step_deviation_percentage = velocity_deviation / self.velocity_step_granularity
        # Reward computation inspired by the usage of a kernel in "Learning Agile and Dynamic Motor Skills for Legged Robots" (Hwangbo+18)
        velocity_tracking_reward = logistic_kernel(velocity_step_deviation_percentage) * self.velocity_tracking_reward_weight

        self.info = {
            **self.info,
            "desired_velocity": self.desired_velocity,
            "velocity_deviation": velocity_deviation,
            "abs_velocity_deviation": np.abs(velocity_deviation),
            "displacement_in_desired_direction": self.displacement_in_desired_direction,
            "velocity_tracking_reward": velocity_tracking_reward
        } 
        return velocity_tracking_reward + super()._primary_reward


class NermoFixedVelocityEnv(NermoDesiredVelocityEnv):
    def __init__(self, *args, desired_velocity=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.desired_velocity = desired_velocity


class NermoCommandedVelocityEnv(NermoDesiredVelocityEnv):
    def __init__(self, *args, min_velocity=0.0, max_velocity=0.3, command_sample_frequency=5, round_sampled_command=True, **kwargs):
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.command_sample_frequency = command_sample_frequency
        self.round_sampled_command = round_sampled_command
        super().__init__(*args, **kwargs)
        if self.command_sample_frequency is not None:
            self._sample_velocity_command()

    def step(self, action):
        if self.command_sample_frequency is not None and (self.episode_step * self.dt) % self.command_sample_frequency == 0:
            self._sample_velocity_command()
        
        return super().step(action)

    def set_velocity_command(self, commanded_velocity):
        self.desired_velocity = commanded_velocity
        # self.desired_velocity = 0.06

    def _sample_velocity_command(self):
        if self.round_sampled_command:
            # Sample from larger range to make sure min and max velocity commands have the same probability after rounding
            sampled_command = random.uniform(self.min_velocity - self.velocity_step_granularity, self.max_velocity + self.velocity_step_granularity)
            # Clip sample to make sure extremes outside of the range are not possible
            sampled_command = np.clip(sampled_command, self.min_velocity, self.max_velocity)
            # Round to nearest float in the given command granularity based on round_nearest in https://stackoverflow.com/a/28425782
            sampled_command = round(sampled_command / self.velocity_step_granularity) * self.velocity_step_granularity
        else:
            sampled_command = random.uniform(self.min_velocity, self.max_velocity)

        self.set_velocity_command(sampled_command)

    def _get_observation(self):
        # Include commanded velocity in observation space
        return np.append(super()._get_observation(), [self.desired_velocity])

