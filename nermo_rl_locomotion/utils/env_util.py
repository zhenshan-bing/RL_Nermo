import os
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecNormalize, SubprocVecEnv
from gym.wrappers import RescaleAction

from ba_nermo_rl_locomotion.utils.utils import get_wrapper_class


def make_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_class: gym.Wrapper = Monitor,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = DummyVecEnv,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    Largely based on: (stable_baselines3.common.env_util import make_vec_env)[https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/env_util.py#L38]

    :param env_id: The environment ID or the environment class
    :param n_envs: The number of environments you wish to have in parallel
    :param env_kwargs: Optional keyword arguments to pass to the env constructor
    :param seed: The initial seed for the random number generator
    :param start_index: Start rank index
    :param monitor_class: The monitor wrapper with which the environment should be wrapped (None to omit). 
    :param monitor_kwargs: Keyword arguments to pass to the constructor of the monitor class
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor

    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)

            if monitor_class is not None:
                # Wrap the env in a Monitor wrapper
                # to have additional training information
                if monitor_kwargs["filename"] is not None:
                    monitor_kwargs["filename"] = monitor_kwargs["filename"] + str(rank)

                env = monitor_class(env, **monitor_kwargs)
            
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

def make_env(
    env_id: str,
    env_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    monitor_class: gym.Wrapper = Monitor,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create a Gym env that can be potentially wrapped and monitored
    Largely based on: (stable_baselines3.common.env_util import make_vec_env)[https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/env_util.py#L38]
    Inspired by: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments

    :param env_id: The environment ID.
    :param env_kwargs: Keyword arguments to pass to the env constructor.
    :param seed: The initial seed for the random number generator.
    :param monitor_class: The monitor wrapper with which the environment should be wrapped (None to omit).   
    :param monitor_kwargs: Keyword arguments to pass to the monitor class constructor.
    :param wrapper: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    env = gym.make(env_id, **env_kwargs)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    
    if monitor_class is not None:
        # Wrap the env in a Monitor wrapper
        env = monitor_class(env, **monitor_kwargs)
    
    # Optionally, wrap the environment with the provided wrapper
    if wrapper is not None:
        env = wrapper(env, **wrapper_kwargs)
    
    return env


def create_enjoy_env(
    env_id: str,
    env_kwargs: Dict[str, Any] = {},
    experiment_details: Dict[str, Any] = {},
    stats_path: Optional[str] = None,
    seed: int = 0,
    normalize_reward: bool = False,
    monitor_class: gym.Wrapper = Monitor,
    monitor_kwargs: Dict[str, Any] = {},
) -> Tuple[VecEnv, int]:
    """
    Create environment for enjoying a trained agent
    :param env_id: Keyword arguments to pass to the env constructor.
    :param env_kwargs: Keyword arguments to pass to the env constructor.
    :param experiment_details: Details of the training experiment for for which the env should be created.
    :param seed: Seed for random number generator.
    :param stats_path: Path to the stats of the experiment (e.g., for loading the stored running average for the normalization).
    :param normalize_reward: Whether to normalize the reward.
    :param monitor_class: The monitor wrapper with which the environment should be wrapped (None to omit).   
    :param monitor_kwargs: Keyword arguments to pass to the monitor class constructor.
    
    :return: the DummyVecEnv wrapped environment for enjoying the agent (potentially with normalization and frame stacking), the env without any VecEnv wrapped around it
    """
    hyperparams = experiment_details.get("hyperparams", {})

    # Construct wrappers that should be applied to the env
    rescale_actions = experiment_details.get("rescale_actions", False)
    if rescale_actions:
        # For legacy loading of the env_wrapper for rescaling actions (structure of experiment_details.json changed after ExperimentManger was introduced)
        env_wrapper = RescaleAction
        wrapper_kwargs = {"min_action": -1, "max_action": 1}
    else:
        env_wrapper = get_wrapper_class(hyperparams)
        wrapper_kwargs = {}
    
    env = make_env(env_id, env_kwargs=env_kwargs, seed=seed, wrapper=env_wrapper, wrapper_kwargs=wrapper_kwargs, monitor_class=monitor_class, monitor_kwargs=monitor_kwargs)

    vec_env = DummyVecEnv([lambda: env])

    # Load saved stats for normalizing observations if needed
    if hyperparams.get("normalize", False):
        # Special case, instead of both normalizing
        # both observation and reward, we can normalize one of the two.
        # in that case `hyperparams["normalize"]` is a string
        # that can be evaluated as python,
        # ex: "dict(norm_obs=False, norm_reward=True)"
        if isinstance(hyperparams["normalize"], str):
            normalize_kwargs = eval(hyperparams["normalize"])
            normalize_kwargs["norm_reward"] = normalize_reward
        else:
            normalize_kwargs = {
                "norm_obs": True,
                "norm_reward": normalize_reward
            }
        print("Loading running average")
        print(f"with params: {normalize_kwargs}")
        if stats_path is not None:
            normalizations_path = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(normalizations_path):
                vec_env = VecNormalize.load(normalizations_path, vec_env)
                vec_env.norm_obs = normalize_kwargs["norm_obs"]
                vec_env.norm_reward = normalize_kwargs["norm_reward"]
                # Deactivate training
                vec_env.training = False
            else:
                raise ValueError(f"VecNormalize stats {normalizations_path} not found")
        else:
            print("VecNormalize stats could not be loaded as no path to the stats was provided")

    # Optionally stack frames
    n_stack = hyperparams.get("frame_stack", 0)
    if n_stack > 0:
        print(f"Stacking {n_stack} frames")
        vec_env = VecFrameStack(vec_env, n_stack)
    
    return vec_env, env