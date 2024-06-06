# Adapted from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/v1.3.0/utils/utils.py
import argparse
import importlib
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import gym
from sb3_contrib import QRDQN, TQC
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

# For custom activation fn
from torch import nn as nn  # noqa: F401 pylint: disable=unused-import

ALGOS: "dict[str, BaseAlgorithm]" = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    "qrdqn": QRDQN,
    "tqc": TQC,
}

MULTI_PROCESSING_CAPABLE_ALGOS = ALGOS  # Since stable-baselines3 version 1.4.0 all algorithms except for HER support multiprocessing

OFF_POLICY_ALGOS = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]


def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, gym.spaces.Dict)
    try:
        return gym.wrappers.FlattenObservation(env)
    except AttributeError:
        keys = env.observation_space.spaces.keys()
        return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams: Dict[str, Any]) -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper
    for multiple, specify a list:
    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper
    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback
    for multiple, specify a list:
    callback:
        - utils.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback
    :param hyperparams:
    :return:
    """

    def get_module_name(callback_name):
        return ".".join(callback_name.split(".")[:-1])

    def get_class_name(callback_name):
        return callback_name.split(".")[-1]

    callbacks = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}
            callback_module = importlib.import_module(get_module_name(callback_name))
            callback_class = getattr(callback_module, get_class_name(callback_name))
            callbacks.append(callback_class(**kwargs))

    return callbacks

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.
    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)

# (https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use)
def triweight_kernel(u, clipped=False):
    if clipped:
        u = np.clip(u, -1, 1)
    return (1 - u**2)**3

def logistic_kernel(u):
    return 1 / (np.exp(u) + 2 + np.exp(-u))
def speed_kernel(u):
    #return 1 / (20*np.exp(50*u) + 40 + 20*np.exp(-50*u))
    return 1 / (2*np.exp(60*u) + 2*np.exp(-60*u))

def custom_kernel_1(x):
    return 1/(1*(np.exp(100*x)+390+np.exp(-100*x)))

def custom_kernel_2(x):
    return 1/(1*(np.exp(250*x)+390+np.exp(-250*x)))

def custom_kernel_3(x):
    return 1/(np.exp(250*x)+180+np.exp(-250*x))

def custom_kernel_4(x):
    return 1/(np.exp(250*x)+80+np.exp(-250*x))

def custom_kernel_5(x):
    return 1/(1*(np.exp(320*x)+180+np.exp(-320*x)))

def custom_kernel_6(x,i):
    return 1/(1*(np.exp(300*x)+400/i+np.exp(-300*x)))

def position_tracking_kernel(x):
    y = 2/(3*np.exp(200*x)+500+3*np.exp(-200*x))
    return y

def custom_kernel_7(x,i):
    return 1/(1*(np.exp(250*x)+400/i+np.exp(-250*x)))