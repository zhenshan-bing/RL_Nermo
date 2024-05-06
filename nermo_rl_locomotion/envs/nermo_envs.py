from collections import deque
from pathlib import Path
import math
import timeit
import random
import nermo_rl_locomotion
from nermo_rl_locomotion.utils.utils import logistic_kernel, custom_kernel_1, custom_kernel_2, custom_kernel_3, custom_kernel_4, custom_kernel_dist, custom_kernel_dist_30
from nermo_rl_locomotion.utils import reward_curricula
import numpy as np
from mujoco_py.generated import const
from gym import spaces, utils
from gym.envs.mujoco.mujoco_env import MujocoEnv

# Build paths relative to the absolute path of this script (or rather the script's parent dir)
# Source: https://stackoverflow.com/a/55051039
BASE_PATH = Path(__file__).parent
MUJOCO_MODEL_PATH = (BASE_PATH / "../../models/dynamic_4l.xml").resolve()

"""
    Gym training environments for NeRmo. 
    The default args for training should be configured in `env_kwargs.py`. 
    The defaults set here were configured such that legacy agents trained with earlier iterations of the environments can be loaded without an issue.
"""

class NermoBaseEnv(MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        model_path=None,
        frame_skip=4,
        terminate_upon_falling=True,
        knees_on_ground_as_falling=False,
        falling_penalty=-1,
        terminate_in_nonfinite_simulation_state=False,
        energy_penalty_weight=-0.008,
        jerkiness_penalty_weight=0,
        knee_on_ground_penalty=0,
        exclude_head_actuation=True,
        exclude_tail_actuation=True,
        exclude_lateral_spine_actuation=False,
        actuator_velocity_limit=None,
        observation_space_elements=["actuator_positions", "knee_joint_positions", "imu_gyro_data", "imu_accelerometer_data", "foot_contact_forces"],
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
        #seed generator for reproducable sampling in the turning environments with commanded radii; is set automatically with the random_seed from the experiment manager
        seed_generator=0
    ):
        utils.EzPickle.__init__(self)

        #define seed for random generator in the turning environments with commanded radii
        self.seed_generator = seed_generator

        #define Return (named reward_return, because self.return is reserved by python)
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
        # Optionally use offset to the current position within the specified range as the action instead of the absolute position (experimental/not recommended)
        self.action_offset_range = action_offset_range 
        # Optional smoothing of actions (experimental/not recommended)
        self.action_smoothing_window = action_smoothing_window

        # Limit for actuators velocities
        self.actuator_velocity_limit = actuator_velocity_limit

        # Rendering defaults
        self.fixed_cam_id = fixed_cam_id
        self.default_render_width = default_render_width
        self.default_render_height = default_render_height

        # Whether to use faulty legacy implementations (for correct loading of legacy agents). New agents should always use the correct implementations (the defaults are properly configured in `env_kwargs.py`)
        self.use_correct_sensors = use_correct_sensors  # Exists because the built-in method for reading sensor values with mujoco-py is erroneous
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

        self.previous_joint_positions = None
        self.previous_actuator_positions = None

        #For calculating angle_dispacements in the turning environments
        self.turning_angle_old = 0
        self.turning_angle_new = 0

        # Initialize MuJoCo environment
        MujocoEnv.__init__(self, model_path if model_path is not None else MUJOCO_MODEL_PATH.as_posix(), frame_skip)

        self.previous_mouse_xy_position = None
        self.current_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)

        self.previous_yaw = None
        self.current_yaw = self.quat2euler(self.get_sensor("com_quat", 4, use_correct_sensors=True))[0, 2]

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
            spine_vel_actuator_id = self.model.actuator_name2id("spine_vel")
            self.actuator_indices_to_exclude.append(spine_vel_actuator_id)
 
        if self.action_offset_range is None:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            
            for index in sorted(self.actuator_indices_to_exclude, reverse=True):
                bounds = np.delete(bounds, index, axis=0)

            # for i in range(8):
            #     bounds = np.delete(bounds, i+1, axis=0)

            if self.exclude_lateral_spine_actuation:
                for i in range(8):
                    bounds = np.delete(bounds, i+1, axis=0)
                    # print(bounds)
            else:
                for i in range(9):
                    bounds = np.delete(bounds, i+1, axis=0)

            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            n_active_actuators = self.model.nu - len(self.actuator_indices_to_exclude)
            self.action_space = spaces.Box(low=self.action_offset_range[0], high=self.action_offset_range[1], shape=(n_active_actuators,), dtype=np.float32)

        # print(self.action_space)
        return self.action_space
    
    def step(self, action, i=None):
        self.info = {}

        self.previous_action = self.latest_action
        self.latest_action = action
        if self.action_history is not None:
            self.action_history.append(action)
            if self.action_smoothing_window is not None:
                action = np.mean(self.action_history, axis=0)

        # for i in range(8):
        #     action = np.insert(action, 2*i+1, 0)

        if self.exclude_lateral_spine_actuation:
            for i in range(8):
                action = np.insert(action, 2*i+1, 0)
        else:
            for i in range(9):
                action = np.insert(action, 2*i+1, 0)
        
        for index in sorted(self.actuator_indices_to_exclude):
            action = np.insert(action, index, 0)

        #get previous positions before simulation step in order to calculate velocities
        self.previous_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)
        self.previous_yaw = self.quat2euler(self.get_sensor("com_quat", 4, use_correct_sensors=True))[0, 2]
        # print(self.previous_yaw)

        self.do_simulation(action, self.frame_skip)

        #get current positions before simulation step in order to calculate velocities
        self.current_mouse_xy_position = self.get_sensor("com_pos", 2, use_correct_sensors=True)
        self.current_yaw = self.quat2euler(self.get_sensor("com_quat", 4, use_correct_sensors=True))[0, 2]
        
        observation = self._get_observation()

        self._compute_velocities()
        self._compute_power_consumption()

        if self.reward_curriculum is not None:
            self.reward_curriculum.step_curriculum(self._primary_reward, self._secondary_rewards)
        
        reward = self._get_reward()
        
        #Calculate return by summing up the collected rewards
        self.reward_return += reward
        done = self._is_done()

        self.episode_step += 1
        self.total_steps += 1

        # lf_reletive
        site_name = "leg_link_fl"  # Replace with the name of the site you want to get the position of
        site_index = self.model.site_name2id(site_name)
        # print(self.model.site_id2name(site_index))
        lf_site_position_world_frame = self.data.site_xpos[site_index]
        lf_foot_position = self.get_sensor('fl_foot_pos', dimensions=3, use_correct_sensors=True)
        lf_reletive = np.array(lf_foot_position)-np.array(lf_site_position_world_frame)
        # print(lf_reletive)

        # rf_reletive
        site_name = "leg_link_fr"  # Replace with the name of the site you want to get the position of
        site_index = self.model.site_name2id(site_name)
        rf_site_position_world_frame = self.data.site_xpos[site_index]
        rf_foot_position = self.get_sensor('fr_foot_pos', dimensions=3, use_correct_sensors=True)
        rf_reletive = np.array(rf_foot_position)-np.array(rf_site_position_world_frame)

        # rh_reletive
        site_name = "leg_link_rr"  # Replace with the name of the site you want to get the position of
        site_index = self.model.site_name2id(site_name)
        rh_site_position_world_frame = self.data.site_xpos[site_index]
        rh_foot_position = self.get_sensor('rr_foot_pos', dimensions=3, use_correct_sensors=True)
        rh_reletive = np.array(rh_foot_position)-np.array(rh_site_position_world_frame)

        # lh_reletive
        site_name = "leg_link_rl"  # Replace with the name of the site you want to get the position of
        site_index = self.model.site_name2id(site_name)
        lh_site_position_world_frame = self.data.site_xpos[site_index]
        lh_foot_position = self.get_sensor('rl_foot_pos', dimensions=3, use_correct_sensors=True)
        lh_reletive = np.array(lh_foot_position)-np.array(lh_site_position_world_frame)

        self.info = {
            "episode_num": self.episode,
            "step": self.episode_step,
            # For the turning environments, _primary_reward should not be called here, because it already gets called in the method _get_reward. 
            # When calling the function _primary_reward twice, the cummulative values calculated there, are not true any more --> more the info entry in the specific classes and do not call the function again. Instead write the value manually
            # "primary_reward": self._primary_reward,
            # "secondary_rewards": self._secondary_rewards,
            "secondary_reward": self._secondary_reward,
            "invariable_penalties": self._invariable_penalties,
            "return": self.reward_return,
            # "curriculum_factors": self.reward_curriculum.curriculum_factors if self.reward_curriculum is not None else None,
            # "has_fallen": self._has_fallen(),
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
            "site_positions": {
                "lf": lf_site_position_world_frame,
                "rf": rf_site_position_world_frame,
                "rh": rh_site_position_world_frame,
                "lh": lh_site_position_world_frame
            },
            "current_yaw": self.current_yaw,
            "relative_foot_positions": {
                "lf": lf_reletive,
                "rf": rf_reletive,
                "rh": rh_reletive,
                "lh": lh_reletive
            },
            "current_roll": self.quat2euler(self.get_sensor("com_quat", 4, use_correct_sensors=True))[0, 0],
            "current_pitch": self.quat2euler(self.get_sensor("com_quat", 4, use_correct_sensors=True))[0, 1],
            "rare_orientation_yaw": self.quat2euler(self.get_sensor("rare_orientation", 4, use_correct_sensors=True))[0, 2],
            "rare_orientation_roll": self.quat2euler(self.get_sensor("rare_orientation", 4, use_correct_sensors=True))[0, 0],
            "rare_orientation_pitch": self.quat2euler(self.get_sensor("rare_orientation", 4, use_correct_sensors=True))[0, 1],
            **self.info
        }
        # print("rel",self.info["relative_foot_positions"])
        # print("abs",self.info["foot_positions"])

        


        #For Turning environments:
        self.turning_angle_old = self.turning_angle_new

        return observation, reward, done, self.info

    def do_simulation(self, ctrl, n_frames):
        if self.action_offset_range is None:
            self.sim.data.ctrl[:] = ctrl
            self.info = {
                "ctrl": ctrl,
                "actuator_force": self.sim.data.actuator_force,
                **self.info
            }
        else:
            self.sim.data.ctrl[:] += ctrl
        for _ in range(n_frames):
            if self.actuator_velocity_limit is not None:
                # Limit actuator velocities (data sheet claims 6.5 rad/s https://www.pololu.com/file/0J1754/DSM44.pdf but without load)
                # Note: velocity value may be computed to be above the specified limit within the next simulation step due to acceleration so the limit should be set slightly below the real maximum
                # TODO: gather empirical speed and torque values for DSM44 servos from real world for more realistic actuator model
                self.data.actuator_velocity[:] = np.clip(self.data.actuator_velocity, -self.actuator_velocity_limit, self.actuator_velocity_limit)
                self.data.qvel[:] = np.clip(self.data.qvel, -self.actuator_velocity_limit, self.actuator_velocity_limit)
            self.sim.step()
            self.info = {
                "actuator_velocity": self.data.actuator_velocity[:],
                **self.info
            }

            self.render()

    def reset(self):
        self.episode += 1
        self.episode_step = 0

        self.previous_action = None
        self.latest_action = None
        if self.action_history is not None:
            self.action_history = deque([], maxlen=self.action_history.maxlen)
        else:
            self.action_history = None

        self.previous_joint_positions = None
        self.previous_actuator_positions = None

        self.reward_return = 0

        return super().reset()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        # Run some initial simulation steps for the model to settle 
        # as it is initialized floating above the ground it needs some time to reach the designated intital configuration
        for _ in range(100):
            #example for initializing with special configuration of actuators
            self.sim.data.ctrl[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            self.sim.step()
        
        return self._get_observation()

    def viewer_setup(self):
        if self.fixed_cam_id is not None:
            self.viewer.cam.fixedcamid = self.fixed_cam_id
            self.viewer.cam.type = const.CAMERA_FIXED #Niklas change from CAMERA_FIXED            
    
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
                self.get_sensor('m1_tail') if not (
                            self.exclude_tail_actuation and self.omit_positions_of_excluded_actuators) else [],
                self.get_sensor('neck') if not (
                            self.exclude_head_actuation and self.omit_positions_of_excluded_actuators) else [],
                self.get_sensor('head') if not (
                            self.exclude_head_actuation and self.omit_positions_of_excluded_actuators) else [],
                self.get_sensor('spine') if not (
                            self.exclude_lateral_spine_actuation and self.omit_positions_of_excluded_actuators) else [],
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
        elif self.previous_joint_positions is not None:
            knee_joint_velocities = (knee_joint_positions - self.previous_joint_positions) / self.dt
        else:
            knee_joint_velocities = [0] * len(knee_joint_positions)
        self.previous_joint_positions = knee_joint_positions

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
            contact = self.data.contact[i]
            if self.model.geom_id2name(contact.geom1) in geoms or self.model.geom_id2name(contact.geom2) in geoms:
                return True
        return False

    def _is_done(self):
        # ------- Episode Termination -------
        # An episode terminates:
        #   * after X timesteps (X is configured upon environment creation)
        #   * when NeRmo falls (i.e., body is in contact with the ground)
        #   * when the simulation state contains numbers that are not finite (cf. https://github.com/openai/gym/blob/a5c0608efbadfd0a1097aaff284885be18129427/gym/envs/mujoco/ant_v3.py#L69)
        if self.terminate_upon_falling and self._has_fallen():
            return True
        if self.terminate_in_nonfinite_simulation_state and not np.isfinite(self.state_vector()).all():
            return True
        return False

    def get_sensor(self, name, dimensions=1, use_correct_sensors=None):
        if use_correct_sensors is None:
            use_correct_sensors = self.use_correct_sensors

        # We need to maintain our own mapping from sensor names to indices as mujoco_py fails in the presence of sensors with multidimensional outputs (https://github.com/openai/mujoco-py/issues/193#issuecomment-458697015)
        # Normal way of accessing sensor data: https://github.com/openai/mujoco-py/issues/26#issuecomment-446236206
        # Issues: 
        #   - sensor_name2id in the original get_sensor does not consider sensors with multidimensional outputs and always increments the id by 1
        #   - the original get_sensor method only returns one value for each sensor name (https://github.com/openai/mujoco-py/blob/master/mujoco_py/generated/wrappers.pxi#L2774)
        # Solution: https://github.com/openai/mujoco-py/issues/193#issuecomment-458697015 + maintain own map from sensor name to sensor index
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
            "rare_orientation": 48,
        }

        if use_correct_sensors:
            sensor_index = name2index_map[name]
        else:
            sensor_index = self.sim.model.sensor_name2id(name)

        return self.sim.data.sensordata[sensor_index:sensor_index+dimensions].copy()

    def mat2euler(self, mat):
        _FLOAT_EPS = np.finfo(np.float64).eps,
        _EPS4 = np.finfo(np.float64).eps * 4.0
        """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
        mat = np.asarray(mat, dtype=np.float64)
        assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

        cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
        condition = cy > _EPS4
        euler = np.empty(mat.shape[:-1], dtype=np.float64)
        euler[..., 2] = np.where(condition,
                                 -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                                 -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
        euler[..., 1] = np.where(condition,
                                 -np.arctan2(-mat[..., 0, 2], cy),
                                 -np.arctan2(-mat[..., 0, 2], cy))
        euler[..., 0] = np.where(condition,
                                 -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                                 0.0)
        return euler

    def quat2mat(self, quat):
        _FLOAT_EPS = np.finfo(np.float64).eps,
        _EPS4 = np.finfo(np.float64).eps * 4.0
        """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
        quat = np.asarray(quat, dtype=np.float64)
        assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        Nq = np.sum(quat * quat, axis=-1)
        # if Nq!=0:
        s = 2.0 / Nq
        X, Y, Z = x * s, y * s, z * s
        wX, wY, wZ = w * X, w * Y, w * Z
        xX, xY, xZ = x * X, x * Y, x * Z
        yY, yZ, zZ = y * Y, y * Z, z * Z

        mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
        mat[..., 0, 0] = 1.0 - (yY + zZ)
        mat[..., 0, 1] = xY - wZ
        mat[..., 0, 2] = xZ + wY
        mat[..., 1, 0] = xY + wZ
        mat[..., 1, 1] = 1.0 - (xX + zZ)
        mat[..., 1, 2] = yZ - wX
        mat[..., 2, 0] = xZ - wY
        mat[..., 2, 1] = yZ + wX
        mat[..., 2, 2] = 1.0 - (xX + yY)
        return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))
        # else:
        #     return np.eye(3)

    def quat2euler(self, quat):
        """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
        return self.mat2euler(self.quat2mat(quat))


class NermoDirectedEnv(NermoBaseEnv):
    def __init__(self, *args, desired_direction=[0, -1], penalize_orthogonal_movement=False,
                 orthogonal_penalty_weight=-1, **kwargs):
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

        # self.foot_displacement_in_desired_direction = 0

        self.displacement_in_orthogonal_direction = 0
        self.velocity_in_orthogonal_direction = 0
        self.smoothed_yaw_rate_vector = np.array([0])
        self.smoothed_yaw_rate = 0

        super().__init__(*args, **kwargs)

    def _compute_velocities(self):
        super()._compute_velocities()
        # foot_displacement = self.current_foot_xy_position - self.previous_foot_xy_position
        # self.foot_displacement_in_desired_direction = np.dot(foot_displacement, self.desired_direction)

        displacement = self.current_mouse_xy_position - self.previous_mouse_xy_position
        self.displacement_in_desired_direction = np.dot(displacement, self.desired_direction)
        self.velocity_in_desired_direction = self.displacement_in_desired_direction / self.dt

        self.displacement_in_orthogonal_direction = np.abs(np.dot(displacement, self.orthogonal_direction))
        self.velocity_in_orthogonal_direction = self.displacement_in_orthogonal_direction / self.dt
        # self.yaw_rate = abs(self.get_sensor("imu_gyro", 3, use_correct_sensors=True)[2])
        # # smoothing the yaw_rate (in desired window size)
        # if self.smoothed_yaw_rate_vector.size < 50:
        #     self.smoothed_yaw_rate_vector = np.append(self.smoothed_yaw_rate_vector, self.yaw_rate)
        # else:
        #     self.smoothed_yaw_rate_vector = np.append(self.smoothed_yaw_rate_vector, self.yaw_rate)[1:]
        # self.smoothed_yaw_rate = np.mean(self.smoothed_yaw_rate_vector)

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
        motion_in_orthogonal_direction_penalty = self.displacement_in_orthogonal_direction * self.orthogonal_penalty_weight * int(
            self.penalize_orthogonal_movement)

        return super()._secondary_rewards + [motion_in_orthogonal_direction_penalty]

        # return super()._secondary_rewards + [(logistic_kernel(self.smoothed_yaw_rate/0.1)-0.25)*0.01]

    # def _is_done(self):
    #     # ------- Episode Termination ------- Add termination conditions: * when Nermo turned a half circle (180° ---
    #     # Pi) * when Nermo is too far away (too high deviation) from desired circular path Pi is rounded here to make
    #     # sure to actually stop the episode (otherwise values could jump from 3.14... to 0.0...)
    #     orientation = self.quat2euler(self.get_sensor("com_quat", 4, use_correct_sensors=True))
    #     z_orientation = orientation[0, 2]
    #     if abs(z_orientation) > 30 / 180 * 3.14:
    #         yaw_is_too_big = 1
    #     else:
    #         yaw_is_too_big = 0
    #
    #     if yaw_is_too_big == 1:
    #         return True
    #
    #     # if self.is_too_far_away == 1:
    #     #     return True
    #     # if self.current_mouse_xy_position[1] <= -2: #for generalization test
    #     #   return True
    #     return super()._is_done()


class NermoMaxVelocityEnv(NermoDirectedEnv):
    @property
    def _primary_reward(self):
        """
        Encourage moving as fast as possible in the given direction
        """
        return self.displacement_in_desired_direction + super()._primary_reward
        # return self.displacement_in_desired_direction * 100 - 0.1 * logistic_kernel(self.foot_displacement_in_desired_direction) + super()._primary_reward

        # return self.foot_displacement_in_desired_direction * 100 + super()._primary_reward


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
        velocity_tracking_reward = logistic_kernel(
            velocity_step_deviation_percentage) * self.velocity_tracking_reward_weight
        # velocity_tracking_reward = speed_kernel(velocity_deviation)
        # velocity_tracking_reward = speed_kernel(velocity_deviation) * self.velocity_tracking_reward_weight

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
    def __init__(self, *args, min_velocity=0.0, max_velocity=0.3, command_sample_frequency=5,
                 round_sampled_command=True, **kwargs):
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.command_sample_frequency = command_sample_frequency
        self.round_sampled_command = round_sampled_command
        super().__init__(*args, **kwargs)
        if self.command_sample_frequency is not None:
            self._sample_velocity_command()

    def step(self, action):
        if self.command_sample_frequency is not None and (
                self.episode_step * self.dt) % self.command_sample_frequency == 0:
            self._sample_velocity_command()

        return super().step(action)

    def set_velocity_command(self, commanded_velocity):
        self.desired_velocity = commanded_velocity

    def _sample_velocity_command(self):
        if self.round_sampled_command:
            # Sample from larger range to make sure min and max velocity commands have the same probability after rounding
            sampled_command = random.uniform(self.min_velocity - self.velocity_step_granularity,
                                             self.max_velocity + self.velocity_step_granularity)
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

#This class is the base class for all turning environments: Giving a desired turning radius and a desired yaw rate
class NermoDesiredTurningRadiusEnv(NermoBaseEnv):
    def __init__(self, desired_turning_radius=1, max_deviation = 0.1, desired_yaw_rate=0.157, smoothed_yaw_rate_window_size = 100,smoothed_yaw_window_size = 100, *args, **kwargs):
        self.desired_turning_radius = desired_turning_radius

        #get the center point of the desired circle trajectory
        self.circle_center = np.array([self.desired_turning_radius, 0])
        self.cummulative_angle_displacement = 0
        self.cummulative_displacement_reward = 0

        #variable for checking, if Nermo is too far away from the desired path (for checking earlier episode termination)
        self.is_too_far_away = 0
        self.max_deviation = self.desired_turning_radius/5
        # self.max_deviation = 0.1

        #yaw related variables
        self.desired_yaw_rate = desired_yaw_rate
        self.yaw_rate = 0
        self.cummulative_yaw_reward = 0

        #for smoothed yaw rate:
        self.smoothed_yaw_rate_vector = np.array([0])
        self.smoothed_yaw_rate = 0
        #we smooth the yaw rate over 200 timesteps to minimize the oszillations from the gyro #changed on June 8th to 100
        self.smoothed_yaw_rate_window_size = smoothed_yaw_rate_window_size

        self.smoothed_yaw_vector = np.array([0])
        self.smoothed_yaw = 0
        self.smoothed_yaw_window_size = smoothed_yaw_window_size
        
        super().__init__(*args, **kwargs)


    def _compute_velocities(self):
        super()._compute_velocities()
        #For all turning environments, get 3rd entry for rotation around z-axis
        # yaw_displacement = self.current_yaw-self.previous_yaw
        self.yaw_rate = self.get_sensor("imu_gyro", 3, use_correct_sensors=True)[2]
        # self.yaw_rate = yaw_displacement/self.dt
        # self.yaw_rate = self.angle_displacement/self.dt
        #smoothing the yaw_rate (in desired window size)
        if self.smoothed_yaw_rate_vector.size < self.smoothed_yaw_rate_window_size:
            self.smoothed_yaw_rate_vector = np.append(self.smoothed_yaw_rate_vector, self.yaw_rate)
        else:
            self.smoothed_yaw_rate_vector = np.append(self.smoothed_yaw_rate_vector, self.yaw_rate)[1:]
        self.smoothed_yaw_rate = np.mean(self.smoothed_yaw_rate_vector)

        if self.smoothed_yaw_vector.size < self.smoothed_yaw_window_size:
            self.smoothed_yaw_vector = np.append(self.smoothed_yaw_vector, abs(self.current_yaw))
        else:
            self.smoothed_yaw_vector = np.append(self.smoothed_yaw_vector, abs(self.current_yaw))[1:]
        self.smoothed_yaw = np.mean(self.smoothed_yaw_vector)

        self.info = {
            **self.info,
            "desired_yaw_rate": self.desired_yaw_rate,
            "yaw_rate": self.yaw_rate,
            "smoothed_yaw_rate": self.smoothed_yaw_rate,
            "smoothed_yaw": self.smoothed_yaw,
        }

    def _is_done(self):
        # ------- Episode Termination -------
        # Add termination conditions:
        #   * when Nermo turned a half circle (180° --- Pi)
        #   * when Nermo is too far away (too high deviation) from desired circular path
        #Pi is rounded here to make sure to actually stop the episode (otherwise values could jump from 3.14... to 0.0...)
        if self.cummulative_angle_displacement > 3.13: #* abs(self.desired_turning_radius) > 3.13:
            return True
        # if self.is_too_far_away == 1:
        #     return True
        #if self.current_mouse_xy_position[1] <= -2: #for generalization test
         #   return True
        # if abs(abs(self.smoothed_yaw)-3.14) < 0.05:
        #     return True
        return super()._is_done()

    def reset(self):
        #reset variables 
        self.cummulative_angle_displacement = 0
        self.cummulative_displacement_reward = 0
        self.cummulative_yaw_reward = 0
        self.turning_angle_new = 0
        self.smoothed_yaw_rate_vector = np.array([0])
        self.smoothed_yaw_vector = np.array([0])
        self.smoothed_yaw_rate = 0
        self.turning_angle_old = 0
        self.is_too_far_away = 0
        self.current_yaw = 0
        self.smoothed_yaw = 0

        return super().reset()

    def calculate_angle_displacement_and_radius_deviation(self):
        #calculate the current turning angle
        #if-statement to check, if angle is already greater than 90° (atan only defined between 0,90°)
        if self.desired_turning_radius > 0:
            if self.current_mouse_xy_position[0] > self.desired_turning_radius:
                self.turning_angle_new = math.atan((self.current_mouse_xy_position[0] - self.desired_turning_radius) / (-self.current_mouse_xy_position[1])) + math.pi/2   
            
            else:
                self.turning_angle_new = math.atan(-self.current_mouse_xy_position[1] / (self.desired_turning_radius - self.current_mouse_xy_position[0]))
        else:
            if self.current_mouse_xy_position[0] < self.desired_turning_radius:
                self.turning_angle_new = math.atan((self.desired_turning_radius - self.current_mouse_xy_position[0]) / (-self.current_mouse_xy_position[1])) + math.pi/2   
            
            else:
                self.turning_angle_new = math.atan(-self.current_mouse_xy_position[1] / (self.current_mouse_xy_position[0] - self.desired_turning_radius))
        
        self.angle_displacement = self.turning_angle_new - self.turning_angle_old
        self.cummulative_angle_displacement += self.angle_displacement

        self.radius_deviation = abs(np.linalg.norm(self.circle_center - self.current_mouse_xy_position) - abs(self.desired_turning_radius))
        
        #radius deviation should not be greater than 0.1m
        if self.radius_deviation > self.max_deviation:
            self.is_too_far_away = 1

    @property
    def _primary_reward(self):
        """
        Encourage moving at the desired velocity with desired turning radius
        """
        
        self.calculate_angle_displacement_and_radius_deviation()

        #add yaw component: reward yaw rate in desired turning direction
        yaw_reward = custom_kernel_1(self.desired_yaw_rate - self.smoothed_yaw_rate) / abs(self.desired_turning_radius)
        self.cummulative_yaw_reward += yaw_reward

        displacement_reward = abs(self.desired_turning_radius) * self.angle_displacement * 10
        self.cummulative_displacement_reward += displacement_reward

        self.info = {
            **self.info,
            "primary_reward": yaw_reward + displacement_reward,
            "angle_displacement": self.angle_displacement,
            "cummulative_angle_displacement": self.cummulative_angle_displacement,
            "displacement_reward": displacement_reward,
            "cummulative_displacement_reward": self.cummulative_displacement_reward,
            "yaw_reward": yaw_reward,
            "cummulative_yaw_reward": self.cummulative_yaw_reward,
            "desired_turning_radius": self.desired_turning_radius,
            "circle_center": self.circle_center,
            "spine_angle": self.get_sensor("spine", 1, use_correct_sensors=True)[0],
            "desired_yaw_rate": self.desired_yaw_rate
        } 
        
        # return yaw_reward + displacement_reward
        return - (self.smoothed_yaw - 3.14) ** 2/3.14/3.14

class NermoCommandedTurningRadiusEnv(NermoDesiredTurningRadiusEnv):
    def __init__(self, desired_turning_radius=None, seed_generator=12345, not_moving_penalty = -0.01, wrong_direction_penalty = -0.01, deterministic_radius=False, desired_velocity=0.157, *args, **kwargs):
        #I now choose desired_velocity instead of desired_yaw_rate as input, since it is more intuitive
        #always trained with desired_velocity=0.157 so far
        self.desired_velocity = desired_velocity
        #define possible desired radi
        self.desired_radii =[desired_turning_radius]
        # print("desired turning radius: "+str(self.desired_radii))
        # self.desired_radii =[-0.6] #[-2, -1, 1, 2]#[-2,-1,-0.5,0.5,1,2] # [0.5,1,1.5,2] #[-2, -1 , 1, 2]
        # self.desired_radii =[-1, -1/2, 1/2, 1]#[-2,-1,-0.5,0.5,1,2] # [0.5,1,1.5,2] #[-2, -1 , 1, 2]
        #seed for sampling random radii
        self.seed_generator = seed_generator
        #create pseudo random generator
        self.random_desired_radius = np.random.default_rng(self.seed_generator)
        #deterministic radius for evaluation and enjoy.py
        self.deterministic_radius = deterministic_radius
        #penalty values
        self.not_moving_penalty = not_moving_penalty
        self.wrong_direction_penalty = wrong_direction_penalty
        super().__init__(*args, **kwargs)

    def reset(self):
        #sample new desired_turning_radius every new episode
        self.desired_turning_radius = float(self.random_desired_radius.choice(self.desired_radii, 1))
        if self.deterministic_radius == True:
            # use this construction when using the enjoy script (each new episode, the next radius from the list of desired radii is taken)
            # self.desired_turning_radius = self.desired_radii[math.floor((self.episode % 8)/2)] #uncomment when calling the enjoy script: this construction is due to the fact, that reset() is called twice in enjoy.py because of the Monitor script. So we double the number of the modulo operator and divide by two, in order to get the right entry from the desired radii list
            #use this construction for training
            self.desired_turning_radius = self.desired_radii[self.episode % 1]
        # self.desired_turning_radius = -0.8

        # desired yaw rate is dependent on desired turning radius
        self.desired_yaw_rate = self.desired_velocity / self.desired_turning_radius
        #reset also circle center:
        self.circle_center = np.array([self.desired_turning_radius, 0])
        # self.max_deviation = abs(self.desired_turning_radius/10)
        self.max_deviation = 0.15
        #For debugging 
        # if self.deterministic_radius == True:
        # print("episode:  "+str(self.episode)+"   desired turning radius: " + str(self.desired_turning_radius) + "max_deviation: "+str(self.max_deviation) + "   desired yaw rate: " + str(self.desired_yaw_rate) + "   circle center: " + str(self.circle_center))
        
        return super().reset()

    @property
    def _primary_reward(self):
        """
        Encourage moving at the desired yaw velocity with turning radius
        """
        self.calculate_angle_displacement_and_radius_deviation()

        yaw_reward = custom_kernel_1(self.desired_yaw_rate - self.smoothed_yaw_rate) #/ abs(self.desired_turning_radius)
        self.cummulative_yaw_reward += yaw_reward 

        displacement_reward = abs(self.desired_turning_radius) * self.angle_displacement
        self.cummulative_displacement_reward += displacement_reward
        self.info = {
            **self.info,
            "primary_reward": yaw_reward + displacement_reward,
            "angle_displacement": self.angle_displacement,
            "cummulative_angle_displacement": self.cummulative_angle_displacement,
            "displacement_reward": displacement_reward,
            "cummulative_displacement_reward": self.cummulative_displacement_reward,
            "yaw_reward": yaw_reward,
            "cummulative_yaw_reward": self.cummulative_yaw_reward,
            "desired_turning_radius": self.desired_turning_radius,
            "desired_velocity": self.desired_velocity,
            "desired_yaw_rate": self.desired_yaw_rate,
            "circle_center": self.circle_center,
            "seed": self.seed_generator,
            "spine_angle": self.get_sensor("spine", 1, use_correct_sensors=True)[0],
        } 
        
        return yaw_reward + displacement_reward
    
    @property
    def _invariable_penalties(self):
        """
        Add not_moving_penalty and wrong_direction_penalty
        """        
        #adding the not_moving_penalty tp prevent Nermo from not moving forward
        # print("angle displacement: "+str(self.angle_displacement))
        if self.angle_displacement > 0:
            not_moving_penalty = 0
        else:
            not_moving_penalty = self.not_moving_penalty

        #add a penalty for moving in the wrong direction
        if math.copysign(1, self.current_yaw * self.desired_turning_radius) < 1:
            wrong_direction_penalty = self.wrong_direction_penalty
        else:
            wrong_direction_penalty = 0

        self.info = {
            **self.info,
            "not_moving_penalty": not_moving_penalty,
            "wrong_direction_penalty": wrong_direction_penalty
        } 
        return super()._invariable_penalties + not_moving_penalty + wrong_direction_penalty

    #make the current desired velocity an input for the observation space
    def _get_observation(self):
        #observation is scaled to -1 .. 1: 1/desired_turning_radius, to get a linear function
        # return np.append(super()._get_observation(), [(1 / self.desired_turning_radius)])
        return np.append(super()._get_observation(), [(1/self.desired_turning_radius)])

class NermoCommandedTurningRadiusAndFixedVelocityEnv(NermoCommandedTurningRadiusEnv):
    def __init__(self, smoothed_speed_window_size=200, desired_velocity=0.15, *args, **kwargs):
        #for smoothed speed:
        self.smoothed_speed_vector = np.array([0])
        self.smoothed_speed = 0
        #we smooth the yaw rate over 200 timesteps to minimize the oszillations in the data
        self.smoothed_speed_window_size = smoothed_speed_window_size
        self.cummulative_velocity_reward = 0
        super().__init__(*args, **kwargs)
        self.desired_velocity = desired_velocity
        
    def _compute_velocities(self):
        super()._compute_velocities()
        #smoothing the speed values
        if self.smoothed_speed_vector.size < self.smoothed_speed_window_size:
            self.smoothed_speed_vector = np.append(self.smoothed_speed_vector, self.speed)
        else:
            self.smoothed_speed_vector = np.append(self.smoothed_speed_vector, self.speed)[1:]
        self.smoothed_speed = np.mean(self.smoothed_speed_vector)

        self.info = {
            **self.info,
            "smoothed_speed": self.smoothed_speed
        }

    def reset(self):
        self.smoothed_speed_vector = np.array([0])
        self.smoothed_speed = 0
        self.cummulative_velocity_reward = 0
        return super().reset()

    @property
    def _primary_reward(self):
        """
        Encourage moving at the desired yaw velocity with turning radius
        """
        self.calculate_angle_displacement_and_radius_deviation()

        # yaw_reward = custom_kernel_4(self.desired_yaw_rate - self.smoothed_yaw_rate)
        yaw_reward = custom_kernel_dist(self.smoothed_yaw - self.cummulative_angle_displacement)*1
        # yaw_reward = custom_kernel_dist_30(self.smoothed_yaw - self.cummulative_angle_displacement)*1.5
        self.cummulative_yaw_reward += yaw_reward 

        displacement_reward = abs(self.desired_turning_radius) * self.angle_displacement * 2
        self.cummulative_displacement_reward += displacement_reward

        #add a reward for tracking the right velocity
        velocity_tracking_reward = custom_kernel_4(self.smoothed_speed - self.desired_velocity)

        dist_reward = custom_kernel_dist(self.radius_deviation)
        self.cummulative_velocity_reward += velocity_tracking_reward

        self.info = {
            **self.info,
            "primary_reward": yaw_reward + displacement_reward + velocity_tracking_reward,
            "angle_displacement": self.angle_displacement,
            "cummulative_angle_displacement": self.cummulative_angle_displacement,
            "displacement_reward": displacement_reward,
            "cummulative_displacement_reward": self.cummulative_displacement_reward,
            "yaw_reward": yaw_reward,
            "cummulative_yaw_reward": self.cummulative_yaw_reward,
            "velocity_tracking_reward": velocity_tracking_reward,
            "cummulative_velocity_reward": self.cummulative_velocity_reward,
            "desired_turning_radius": self.desired_turning_radius,
            "desired_velocity": self.desired_velocity,
            "desired_yaw_rate": self.desired_yaw_rate,
            "circle_center": self.circle_center,
            "seed": self.seed_generator,
            "spine_angle": self.get_sensor("spine", 1, use_correct_sensors=True)[0],
            "smoothed_speed_window_size": self.smoothed_speed_window_size,
            "smoothed_yaw_rate_window_size": self.smoothed_yaw_rate_window_size,
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
        } 
        
        return displacement_reward + yaw_reward #dist_reward *  #+ velocity_tracking_reward

class NermoCommandedTurningRadiusAndCommandedVelocityEnv(NermoCommandedTurningRadiusAndFixedVelocityEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #test with more radii
        self.desired_radii = [-2, -1 , -0.5, 0.5, 1, 2]
        #test with 0.1m/s and 0.2m/s desired velocity
        self.desired_velocities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        #create pseudo random generator for sampling the desired velocity
        self.random_desired_velocity = np.random.default_rng(self.seed_generator + 1)

    #draft for curriculum learning - experimental
    # def get_reward_curriculum(self,total_steps):
    #     if total_steps < 187500: #*8: 1.5M
    #         return 1
    #     elif (total_steps >= 187500) and (total_steps <= 375000): #*8: 3M
    #         factor = (total_steps - 187500) / 187500 * 4 + 1
    #         return factor
    #     else:
    #         return 5

    def reset(self):
        self.info={}
        #sample new desired_velocity every new episode, this time uniformly sampled
        self.smoothed_speed_vector = np.array([0])
        self.smoothed_speed = 0
        self.cummulative_velocity_reward = 0

        #sample new desired_turning_radius every new episode, this time uniformly
        self.desired_velocity = float(self.random_desired_velocity.uniform(0.1, 0.25, 1))
        #self.desired_velocity = float(self.random_desired_velocity.choice([0.1, 0.15, 0.2, 0.25]))
        self.desired_turning_radius = float(self.random_desired_radius.choice([self.random_desired_radius.uniform(-2,-0.5,1), self.random_desired_radius.uniform(0.5,2,1)], 1))
        #self.desired_turning_radius = float(self.random_desired_radius.choice(self.desired_radii))
        if self.deterministic_radius == True:
            #use this construction when using the enjoy script: We use 4 different radii for testing
            self.desired_turning_radius = self.desired_radii[math.floor((self.episode % 12)/2)] #uncomment when calling the enjoy script: this construction is due to the fact, that reset() is called twice in enjoy.py because of the Monitor script. So we double the number of the modulo operator and divide by two, in order to get the right entry from the desired radii list
            self.desired_velocity = self.desired_velocities[math.floor((self.episode % 24)/2)]
            
            #You can also manually enter desired turning radius and velocity for enjoying the agent
            #self.desired_turning_radius = -0.5
            #self.desired_velocity = 0.25
            
            #use this construction for training (12 evaluation episodes)
            #self.desired_turning_radius =  self.desired_radii[self.episode % 6]
            #for evaluation I choose the mean velocity of 0.1m/s and 0.2m/s
            #self.desired_velocity = self.desired_velocities[self.episode % 12]
        #desired yaw rate is dependent on desired turning radius
        self.desired_yaw_rate = self.desired_velocity / self.desired_turning_radius
        #reset also circle center:
        self.circle_center = np.array([self.desired_turning_radius, 0])
        self.max_deviation = 0.1 
        if self.deterministic_radius == True:
            print("desired turning radius: " + str(self.desired_turning_radius) + "   desired yaw rate: " + str(self.desired_yaw_rate) + "   circle center: " + str(self.circle_center) + "     desired velocity: " + str(self.desired_velocity))
        

        #inheritance from NermoDesiredTurningRadiusEnv, because we want to sample the radii uniformly now
        return super(NermoCommandedTurningRadiusEnv, self).reset()

    def _get_observation(self):
        #append observation of desired turning radius and velocity and scale to -1..1
        return np.append(super(NermoCommandedTurningRadiusEnv, self)._get_observation(), [(1 / self.desired_turning_radius) / 2, (self.desired_velocity - 0.175) * 40 / 3])
        #Try to not scale reward instead of -1 .. 1
        #return np.append(super()._get_observation(), [self.desired_velocity])
        #return np.append(super()._get_observation(), [(self.desired_velocity - 0.15) * 10])

    #one possibility to command radius and velocity in training or enjoy script, by sampling a new value every x seconds
    #def step(self, action):
    #     if (self.episode_step * self.dt) % 10 == 0:
    #         self.desired_turning_radius = -self.desired_turning_radius
    #         self.desired_velocity = self.desired_velocity - 0.15
    #     return super().step(action)

    @property
    def _primary_reward(self):
        """
        Encourage moving at the desired yaw velocity with turning radius
        """
        self.calculate_angle_displacement_and_radius_deviation()
        #draft for integration of curriculum learning - experimental
        #curriculum_weight = self.get_reward_curriculum(self.total_steps)

        yaw_reward = custom_kernel_4((self.desired_yaw_rate - self.smoothed_yaw_rate)) #/ abs(self.desired_turning_radius)
        self.cummulative_yaw_reward += yaw_reward 

        displacement_reward = abs(self.desired_turning_radius) * self.angle_displacement
        self.cummulative_displacement_reward += displacement_reward

        #add a reward for tracking the right velocity
        velocity_tracking_reward = custom_kernel_4((self.smoothed_speed - self.desired_velocity))
        self.cummulative_velocity_reward += velocity_tracking_reward

        #add a reward for tracking the position accurately
        #position_tracking_reward = position_tracking_kernel(self.radius_deviation)
        
        self.info = {
            **self.info,
            "primary_reward": yaw_reward + displacement_reward + velocity_tracking_reward, #+ position_tracking_reward,
            "angle_displacement": self.angle_displacement,
            "cummulative_angle_displacement": self.cummulative_angle_displacement,
            "displacement_reward": displacement_reward,
            "cummulative_displacement_reward": self.cummulative_displacement_reward,
            "yaw_reward": yaw_reward,
            "cummulative_yaw_reward": self.cummulative_yaw_reward,
            "velocity_tracking_reward": velocity_tracking_reward,
            "cummulative_velocity_reward": self.cummulative_velocity_reward,
            #"position_tracking_reward": position_tracking_reward,
            "desired_turning_radius": self.desired_turning_radius,
            "desired_velocity": self.desired_velocity,
            "desired_yaw_rate": self.desired_yaw_rate,
            "smoothed_yaw_rate": self.smoothed_yaw_rate,
            "circle_center": self.circle_center,
            "seed": self.seed_generator,
            "total_steps": self.total_steps, 
            #"curriculum_factor": curriculum_weight,
            "angle_values" : {
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
            },
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
            "spine_angle": self.get_sensor("spine", 1, use_correct_sensors=True)[0]
        } 
        
        return  yaw_reward + velocity_tracking_reward + displacement_reward #+ position_tracking_reward