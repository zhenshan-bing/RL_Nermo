import csv
import json
import numbers
import time
import timeit
from typing import Any, Dict, List, Optional, Tuple, Union
from nermo_rl_locomotion.envs.nermo_envs import NermoBaseEnv
from nermo_rl_locomotion.utils.evaluation_util import compute_avg_gait_metrics

import numpy as np
import pandas
import gym
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class NermoMonitor(gym.Wrapper):
    """
        Custom Monitor for monitoring our specific gym environments based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/monitor.py
    """

    def __init__(
        self,
        env: NermoBaseEnv,
        filename: Optional[str] = None,
        info_keywords: Tuple[str, ...] = (),
        verbose: int = 0,
        real_time_info_interval = None,
        virtual_episode_length = None,
        gait_metric_kwargs: Optional[Dict[str, Any]] = {"phase_gap_tolerance": 3, "min_phase_steps": 5},
    ):
        super().__init__(env=env)
        self.monitoring_start = time.time()

        self.filename = filename
        self.results_writer = None

        # Keywords in the `info` dict returned by the environments `step` function which should be logged
        self.info_keywords = info_keywords

        self.verbose = verbose

        # Interval at which some real-time monitoring info should be printed (None to deactivate)
        self.real_time_info_interval = real_time_info_interval

        # Virtual episode length for environments without a time limit (especially for monitoring the model-based controller (cf. https://gitlab.lrz.de/ge25qiq/mouse_controller))
        self.virtual_episode_length = virtual_episode_length

        # Arguments for the computation of the gait metrics
        self.gait_metric_kwargs = gait_metric_kwargs

        # Initialization of the internal state
        self.total_steps = 0
        self.episode_num = 0
        self.episode_step_rewards = []
        self.episode_step_infos = []
        self.episode_steps = 0
        self.episode_start = timeit.default_timer()
        self.needs_reset = True

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = []
        self.episode_times = []

    def reset(self, **kwargs) -> GymObs:
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = []
        self.episode_times = []

        self.episode_step_rewards = []
        self.episode_step_infos = []

        self.needs_reset = False
        self.episode_steps = 0
        return self.env.reset(**kwargs)

    def start_virtual_episode(self):
        self.episode_steps = 0

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        if self.episode_steps == 0:
            self.episode_start = time.time()

        observation, reward, done, info = self.env.step(action)
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_step_rewards.append(reward)
        self.episode_step_infos.append(info)
        
        # Optionally print real-time diagnostics (for enjoy.py)
        if self.verbose > 0 and self.real_time_info_interval is not None:
            if self.episode_steps != 0 and self.episode_steps % self.real_time_info_interval == 0:
                recent_infos = self.episode_step_infos[-self.real_time_info_interval:]
                avg_speed = np.mean([info.get("speed", 0) for info in recent_infos])
                avg_velocity = np.mean([info.get("velocity_in_desired_direction", 0) for info in recent_infos])
                avg_power = np.mean([info.get("power", 0) for info in recent_infos])
                diagnostics = {
                    "speed": f"{avg_speed:.2f}",
                    "velocity": f"{avg_velocity:.2f}",
                    "des_vel": f"{info.get('desired_velocity', -1):.2f}",
                    "des_radius": f"{info.get('desired_turning_radius', -1):.2f}",
                    "smoothed_yaw_rate": f"{info.get('smoothed_yaw_rate', -1):.2f}",
                    "des_yaw_rate": f"{info.get('desired_yaw_rate', -1):.2f}",
                    #"vel": f"{avg_velocity:.2f}",
                    "pow": f"{avg_power:.2f}"
                }
                # Overwrite current line in terminal with newest diagnostics based on https://stackoverflow.com/a/8436827
                print(f"\r{diagnostics}".rstrip(), end="")

        if done or (self.episode_steps == self.virtual_episode_length):
            self.episode_num += 1
            episode_end = time.time()
            measured_episode_length = episode_end - self.episode_start
            self.needs_reset = done
            passed_episode_rewards = self.episode_step_rewards[-self.episode_steps:]
            episode_reward = np.sum(passed_episode_rewards)
            mean_step_reward = np.mean(passed_episode_rewards)
            std_step_reward = np.std(passed_episode_rewards)

            episode_info = {
                "r": round(episode_reward, 6),
                "l": self.episode_steps,
                "t": round(episode_end - self.monitoring_start, 6),
                #"measured_episode_length": round(measured_episode_length, 3),
                #"mean_step_reward": round(mean_step_reward, 6),
                #"std_step_reward": round(std_step_reward, 6),
                #"final_position": info.get("position")
            }
            # Compute metrics based on info keywords
            for key in self.info_keywords:
                if isinstance(info[key], (numbers.Number, np.number)) or key == "secondary_rewards":
                    info_key_values = [step_info[key] for step_info in self.episode_step_infos[-self.episode_steps:]]
                    episode_info[f"sum_{key}"] = np.round(np.sum(info_key_values, axis=0), 6)
                    episode_info[f"mean_{key}"] = np.round(np.mean(info_key_values, axis=0), 6)
                    episode_info[F"std_{key}"] = np.round(np.std(info_key_values, axis=0), 6)
                else:
                    episode_info[key] = info[key]

            passed_episode_infos = self.episode_step_infos[-self.episode_steps:]
            
            episode_info = {
                **episode_info,
                #uncomment for printing avg_gait_metrics to console
                **compute_avg_gait_metrics(passed_episode_infos, self.env.dt, **self.gait_metric_kwargs)
            }

            info["episode"] = episode_info

            self.episode_rewards.append(episode_reward)
            self.episode_infos.append(episode_info)
            self.episode_times.append(episode_end - self.monitoring_start)
            

            if self.filename is not None and self.results_writer is None:
                self.results_writer = ResultsWriter(
                    self.filename,
                    header={"t_start": self.monitoring_start, "env_id": self.env.spec and self.env.spec.id},
                    extra_keys=tuple([key for key in episode_info.keys() if key not in ["r", "l", "t"]]),
                )

            if self.results_writer is not None:
                self.results_writer.write_row(episode_info)

            if self.verbose > 0:
                print(f"\n------ Summary of episode {self.episode_num} ------")
                print(json.dumps(episode_info, indent=4, cls=NumpyEncoder))

            self.episode_steps = 0

        return observation, reward, done, info

            
    def close(self) -> None:
        """
        Closes the environment
        """
        if self.verbose > 0 and self.episode_num > 1:
            summary = {}
            for key in self.episode_infos[-1]:
                if isinstance(self.episode_infos[-1][key], (numbers.Number, np.number)):
                    info_key_values = [info[key] for info in self.episode_infos]
                    summary[f"mean_{key}"] = round(np.mean(info_key_values), 6)

            print(f"------ Summary of the {self.episode_num} episodes ------")
            print(json.dumps(summary, indent=4, cls=NumpyEncoder))

        super(NermoMonitor, self).close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types based on https://stackoverflow.com/a/57915246"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void): 
            return None

        return json.JSONEncoder.default(self, obj)