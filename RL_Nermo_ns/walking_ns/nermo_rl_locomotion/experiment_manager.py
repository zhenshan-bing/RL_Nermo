# Adapted from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/exp_manager.py

import argparse
import os
import pickle as pkl
import time
import warnings
import multiprocessing
import json
from nermo_rl_locomotion.utils.env_util import make_vec_env
from mujoco_py.builder import MujocoException
from stable_baselines3.common.monitor import Monitor
import yaml
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gym

import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_intermediate_values, plot_parallel_coordinate, plot_contour, plot_edf, plot_slice, plot_pareto_front

# For using HER with GoalEnv
from stable_baselines3 import HerReplayBuffer  # noqa: F401
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn, set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize
)

# For custom activation fn
from torch import nn as nn  # noqa: F401

import nermo_rl_locomotion   # Required to find the env_id
from nermo_rl_locomotion.utils.callbacks import SaveVecNormalizeCallback, TrialEvalCallback
from nermo_rl_locomotion.utils.hyperparams_opt import HYPERPARAMS_SAMPLER
from nermo_rl_locomotion.utils.utils import ALGOS, MULTI_PROCESSING_CAPABLE_ALGOS, get_callback_list, get_wrapper_class, linear_schedule


# Build paths relative to the absolute path of this script (or rather the script's parent dir)
# Source: https://stackoverflow.com/a/55051039
BASE_PATH = Path(__file__).parent
LEARNED_MODEL_PATH = (BASE_PATH / "../trained_agents/models").resolve()
HYPERPARAMS_PATH = (BASE_PATH / "../hyperparams/").resolve()
HYPERPARAMETER_OPTIMIZATION_PATH = (BASE_PATH / "../hyperparams/optimization_studies").resolve()


class ExperimentManager(object):
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.
    Please take a look at `train.py` and `optimize_hyperparameters.py` for details on each argument.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        env_id: str,
        experiment_name: str,
        experiment_description: str,
        algo: str,
        n_timesteps: int = -1,
        hyperparams: Optional[Dict[str, Any]] = None,
        seed: int = 0,
        trained_agent: str = "",
        env_kwargs: Optional[Dict[str, Any]] = None,
        eval_env_kwargs: Optional[Dict[str, Any]] = None,
        vec_env_type: str = "auto",
        eval_freq: float = 10000,
        n_eval_episodes: int = 4,
        n_eval_envs: int = 1,
        checkpoint_freq: float = -1,
        save_replay_buffer: bool = False,
        optimize_hyperparameters: bool = False,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
        n_trials: int = 1,
        n_jobs: int = 1,
        sampler: str = "tpe",
        pruner: str = "median",
        optimization_log_folder: Path = HYPERPARAMETER_OPTIMIZATION_PATH,
        save_best_models: bool = True,
        n_startup_trials: int = 0,
        n_evaluations: int = 1,
        no_optim_plots: bool = False,
        print_n_best_trials: int = 1,
        save_n_best_hyperparameters: int = 0,
        truncate_last_trajectory: bool = False,
        log_folder: Path = LEARNED_MODEL_PATH,
        monitor_class: gym.Wrapper = Monitor,
        monitor_kwargs: Dict[str, Any] = {},
        tensorboard_log: bool = True,
        log_interval: int = 0,
        verbose: int = 1,
        device: str = "auto"
    ):
        super(ExperimentManager, self).__init__()
        # Experiment info
        self.experiment_name = experiment_name
        self.experiment_description = experiment_description
        # Environment
        self.env_id = env_id
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.eval_env_kwargs = eval_env_kwargs
        # Algorithm & Training config
        self.algo = algo
        self.custom_hyperparams = hyperparams
        self.n_timesteps = n_timesteps  # Can be overwritten from hyperparams file
        self.seed = seed
        # Params that are loaded from the hyperparams file for the algorithm and the env
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None

        # Which VecEnv class to use
        if vec_env_type == "auto":
            vec_env_type = "subproc" if algo in MULTI_PROCESSING_CAPABLE_ALGOS else "dummy"
        self.vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]
        self.vec_env_kwargs = {}
        self.n_envs = multiprocessing.cpu_count() if vec_env_type == "subproc" else 1           # may be updated when reading hyperparams

        # Callbacks
        self.specified_callbacks = []
        self.callbacks = []
        self.checkpoint_freq = checkpoint_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs

        self.n_actions = None  # For DDPG/TD3 action noise objects
        self._hyperparams = {}

        self.trained_agent = trained_agent
        self.continue_training = trained_agent.endswith(".zip") and os.path.isfile(trained_agent)
        self.truncate_last_trajectory = truncate_last_trajectory

        # Hyperparameter optimization config
        self.optimize_hyperparameters = optimize_hyperparameters
        self.storage = storage
        self.study_name = study_name
        self.no_optim_plots = no_optim_plots
        self.optimization_log_folder = optimization_log_folder
        # maximum number of trials for finding the best hyperparams
        self.n_trials = n_trials
        # number of parallel jobs when doing hyperparameter search
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.save_best_models = save_best_models
        self.deterministic_eval = True
        self.print_n_best_trials = print_n_best_trials
        self.save_n_best_hyperparameters = save_n_best_hyperparameters

        # Logging
        self.log_folder = log_folder
        self.monitor_class = monitor_class
        self.monitor_kwargs = monitor_kwargs
        self.verbose = verbose
        self.args = args
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer
        self.tensorboard_log = tensorboard_log

        # Compute configuration
        self.device = device

    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.
        :return: the initialized RL model
        """
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks = self._preprocess_hyperparams(hyperparams)
        
        if not self.optimize_hyperparameters:
            self.algorithm_description = f"{self.algo}_{self.n_timesteps}_timesteps_seed_{self.seed}/"
            base_experiment_save_path = (self.log_folder / f"{self.env_id}/{self.experiment_name}/").resolve()
            self.log_path = (base_experiment_save_path / self.algorithm_description).resolve()
            self.tensorboard_log = (base_experiment_save_path / "tensorboard").resolve() if self.tensorboard_log else None
            
        else:
            report_name = (
                f"{self.study_name if self.study_name is not None else 'report'}"
                f"_{self.n_trials}-trials_{self.n_timesteps}-timesteps"
                f"_{self.sampler}_{self.pruner}_{int(time.time())}"
            )

            self.log_path = (self.optimization_log_folder / self.algo / self.env_id / report_name).resolve()

        # Do not create a log folder, if an optimization study is only loaded and not resumed
        if not (self.optimize_hyperparameters and self.n_trials < 1):
            print(f"Log path: {self.log_path}")
            self._create_log_folder()
            self._save_config(saved_hyperparams)

        self._create_callbacks()
        

        if self.seed is not None:
            set_random_seed(self.seed)

        # Create env to have access to action space for action noise
        env = self.create_envs(self.n_envs, no_log=self.optimize_hyperparameters)

        self._hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, env)


        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, env)
        elif self.optimize_hyperparameters:
            return None
        else:
            # Create model to train an agent from scratch
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                verbose=self.verbose,
                device=self.device,
                **self._hyperparams,
            )

        return model

    def learn(self, model: BaseAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs = {"tb_log_name": self.algorithm_description}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            try:
                model.env.close()
            except EOFError:
                pass

    def save_trained_model(self, model: BaseAlgorithm, name: str = "model") -> None:
        """
        Save trained model optionally with its replay buffer
        and ``VecNormalize`` statistics
        :param model:
        """
        print(f"Saving to {self.log_path}")
        model.save((self.log_path / name).resolve())

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer((self.log_path / "replay_buffer.pkl").resolve())

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save((self.log_path / "vecnormalize.pkl").resolve())

    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be use later
        to reproduce an experiment.
        :param saved_hyperparams:
        """
        # Save experiment details
        config = {
            "experiment_name": self.experiment_name,
            "experiment_description": self.experiment_description,
            "started_at": datetime.now().replace(microsecond=0).isoformat(),
            "env_id": self.env_id,
            "env_kwargs": self.env_kwargs,
            "eval_env_kwargs": self.eval_env_kwargs,
            "algorithm": self.algo,
            "n_timesteps": self.n_timesteps,
            "num_envs": self.n_envs,
            "hyperparams": saved_hyperparams,
            "eval_frequency": self.eval_freq,
            "checkpoint_frequency": self.checkpoint_freq
        }

        with open((self.log_path / "config.json").resolve(), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # Save command line arguments that were used to start the experiment
        with open((self.log_path / "args.json").resolve(), "w", encoding="utf-8") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            json.dump(ordered_args, f, ensure_ascii=False, indent=4)

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load hyperparameters from yaml file
        with open((HYPERPARAMS_PATH / f"{self.algo}.yml").resolve(), "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            else:
                raise ValueError(f"Hyperparameters not found for {self.algo}-{self.env_id}")

        if self.custom_hyperparams is not None:
            # Overwrite hyperparams if needed
            hyperparams.update(self.custom_hyperparams)
        # Sort hyperparams that will be saved
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

        if self.verbose > 0:
            print("Default hyperparameters for environment (ones being tuned will be overridden):")
            pprint(saved_hyperparams)

        return hyperparams, saved_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        # Create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def _preprocess_normalization(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        if "normalize" in hyperparams.keys():
            self.normalize = hyperparams["normalize"]

            # Special case, instead of both normalizing
            # both observation and reward, we can normalize one of the two.
            # in that case `hyperparams["normalize"]` is a string
            # that can be evaluated as python,
            # ex: "dict(norm_obs=False, norm_reward=True)"
            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True

            # Use the same discount factor as for the algorithm
            if "gamma" in hyperparams:
                self.normalize_kwargs["gamma"] = hyperparams["gamma"]

            del hyperparams["normalize"]
        return hyperparams

    def _preprocess_hyperparams(
        self, hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:
        self.n_envs = hyperparams.get("n_envs", self.n_envs)

        if self.verbose > 0:
            print(f"Using {self.n_envs} environments")

        # Convert schedule strings to objects
        hyperparams = self._preprocess_schedules(hyperparams)

        # Pre-process train_freq
        if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
            hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

        # Should we overwrite the number of timesteps?
        if self.n_timesteps > 0:
            if self.verbose:
                print(f"Overwriting n_timesteps with n={self.n_timesteps}")
        else:
            self.n_timesteps = int(hyperparams["n_timesteps"])

        # Pre-process normalize config
        hyperparams = self._preprocess_normalization(hyperparams)

        # Pre-process policy/buffer keyword arguments
        # Convert to python object if needed
        for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

        # Delete keys so the dict can be pass to the model constructor
        if "n_envs" in hyperparams.keys():
            del hyperparams["n_envs"]
        del hyperparams["n_timesteps"]

        if "frame_stack" in hyperparams.keys():
            self.frame_stack = hyperparams["frame_stack"]
            del hyperparams["frame_stack"]

        # obtain a class object from a wrapper name string in hyperparams
        # and delete the entry
        env_wrapper = get_wrapper_class(hyperparams)
        if "env_wrapper" in hyperparams.keys():
            del hyperparams["env_wrapper"]

        callbacks = get_callback_list(hyperparams)
        if "callback" in hyperparams.keys():
            self.specified_callbacks = hyperparams["callback"]
            del hyperparams["callback"]

        return hyperparams, env_wrapper, callbacks

    def _preprocess_action_noise(
        self, hyperparams: Dict[str, Any], saved_hyperparams: Dict[str, Any], env: VecEnv
    ) -> Dict[str, Any]:
        # Parse noise string
        # Note: only off-policy algorithms are supported
        if hyperparams.get("noise_type") is not None:
            noise_type = hyperparams["noise_type"].strip()
            noise_std = hyperparams["noise_std"]

            # Save for later (hyperparameter optimization)
            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std {noise_std}")

            del hyperparams["noise_type"]
            del hyperparams["noise_std"]

        return hyperparams

    def _create_log_folder(self):
        os.makedirs(self.log_path, exist_ok=True)

    def _create_callbacks(self):
        if self.checkpoint_freq > 0 and not self.optimize_hyperparameters:
            # Interpret float in range (0, 1) as a percentage of the timesteps
            if 0 < self.checkpoint_freq < 1:
                self.checkpoint_freq = int(self.checkpoint_freq * self.n_timesteps)
            else:
                self.checkpoint_freq = int(self.checkpoint_freq)
            # Account for the number of parallel environments
            self.checkpoint_freq = max(self.checkpoint_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.checkpoint_freq,
                    save_path=self.log_path,
                    name_prefix="model",
                    verbose=1,
                )
            )

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            # Interpret float in range (0, 1) as a percentage of the timesteps
            if 0 < self.eval_freq < 1:
                self.eval_freq = int(self.eval_freq * self.n_timesteps)
            else:
                self.eval_freq = int(self.eval_freq)
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)

            if self.verbose > 0:
                print("Creating test environment")

            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.log_path)
            eval_callback = EvalCallback(
                self.create_envs(self.n_eval_envs, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.log_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.log_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
            )

            self.callbacks.append(eval_callback)

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.
        :param env:
        :param eval_env:
        :return:
        """
        # Pretrained model, load normalization
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    print("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env

    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.
        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.log_path

        # Always use DummyVecEnv for eval envs
        if eval_env:
            vec_env_class = DummyVecEnv
        else:
            vec_env_class = self.vec_env_class
        
        # Use special environment arguments for the eval_env, if provided
        if eval_env and self.eval_env_kwargs is not None:
            env_kwargs = self.eval_env_kwargs
        else:
            env_kwargs = self.env_kwargs

        # Use the default Monitor class for the Eval env to avoid warnings
        monitor_class = Monitor if eval_env else self.monitor_class

        monitor_kwargs = {
            "filename": f"{log_dir.as_posix()}/" if log_dir is not None else None,
            **self.monitor_kwargs
        }

        # On most env, SubprocVecEnv does not help and is quite memory hungry
        # therefore we use DummyVecEnv by default
        env = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            seed=self.seed,
            monitor_class=monitor_class,
            monitor_kwargs=monitor_kwargs,
            wrapper_class=self.env_wrapper,
            vec_env_cls=vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
        )

        # Wrap the env into a VecNormalize wrapper if needed
        # and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        # Optional Frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f"Stacking {n_stack} frames")

        return env

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]


        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            seed=self.seed,
            tensorboard_log=self.tensorboard_log,
            verbose=self.verbose,
            device=self.device,
            **hyperparams,
        )

        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

        if os.path.exists(replay_buffer_path):
            print("Loading replay buffer")
            # `truncate_last_traj` will be taken into account only if we use HER replay buffer
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model

    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            # TODO: try with multivariate=True
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
        elif sampler_method == "skopt":
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = MedianPruner(n_startup_trials=self.n_trials, n_warmup_steps=self.n_evaluations)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    def objective(self, trial: optuna.Trial) -> float:

        kwargs = self._hyperparams.copy()

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions
        # Hack when using HerReplayBuffer
        trial.using_her_replay_buffer = kwargs.get("replay_buffer_class") == HerReplayBuffer
        if trial.using_her_replay_buffer:
            trial.her_kwargs = kwargs.get("replay_buffer_kwargs", {})
        # Sample candidate hyperparameters
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial)
        kwargs.update(sampled_hyperparams)

        model = ALGOS[self.algo](
            env=self.create_envs(self.n_envs, no_log=True),
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            device=self.device,
            verbose=0,
            **kwargs,
        )

        model.trial = trial

        eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)

        optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        optuna_eval_freq = max(optuna_eval_freq // model.get_env().num_envs, 1)
        path = None

        if self.save_best_models:
            path = (self.log_path / f"trial_{str(trial.number)}").resolve()

            os.makedirs(path, exist_ok=True)
        
            with open((path / "sampled_hyperprams.json").resolve(), "w", encoding="utf-8") as f:
                default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
                json.dump(trial.params, f, ensure_ascii=False, indent=4, default=default)

        callbacks = get_callback_list({"callback": self.specified_callbacks})
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            best_model_save_path=path,
            log_path=path,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=optuna_eval_freq,
            deterministic=self.deterministic_eval,
        )
        callbacks.append(eval_callback)

        try:
            model.learn(self.n_timesteps, callback=callbacks)
            # Free memory
            model.env.close()
            eval_env.close()
        except (AssertionError, ValueError, MujocoException) as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            print("============")
            print("Sampled hyperparams:")
            pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned()
        except (RuntimeError, EOFError) as e:
            # Alow trials to continue in case of a RuntimeError (e.g., caused by CUDA)
            print(e)

        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameter_optimization(self) -> None:
        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        if self.tensorboard_log is not None:
            warnings.warn("Tensorboard log is deactivated when running hyperparameter optimization")
            self.tensorboard_log = None

        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        if self.n_trials > 0:
            n_trials_before = len(study.trials)
            try:
                study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
            except KeyboardInterrupt:
                pass

        print(f"Total number of finished trials: {len(study.trials)}")

        best_trial = study.best_trial
        print(f"Best trial: {best_trial.number}")

        print(f"Value: {best_trial.value}")

        print("Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")


        # Based on: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/scripts/parse_study.py
        sorted_trials = study.trials
        sorted_trials.sort(
            key = lambda t: t.value if t.value is not None else float("-inf"),
            reverse = True
        )
        
        if self.print_n_best_trials > 1:
            print(f"{self.print_n_best_trials} best trials:")
            for rank in range(0, min(self.print_n_best_trials, len(sorted_trials))):
                trial = sorted_trials[rank]
                print(f"# {rank + 1} - number: {trial.number} - value: {trial.value:.2f}")
                pprint(trial.params)


        if self.save_n_best_hyperparameters > 0:
            path = (self.log_path / "best_hyperparams").resolve()
            os.makedirs(path, exist_ok=True)
            for rank in range(0, min(self.save_n_best_hyperparameters, len(sorted_trials))):
                trial = sorted_trials[rank]

                with open((path / f"{rank + 1}_hyperparams_trial-{trial.number}_value-{trial.value:.0f}.json").resolve(), "w", encoding="utf-8") as f:
                    json.dump(trial.params, f, indent=4)
            print(f"Saved best hyperparameters to {path}")

        if self.n_trials > 0:
            if self.verbose:
                print(f"Writing report to {self.log_path}")

            # Write report
            study.trials_dataframe().to_csv(f"{self.log_path}/report.csv")

            # Save python object to inspect/re-use it later
            with open(f"{self.log_path}/study.pkl", "wb+") as f:
                pkl.dump(study, f)

        # Skip plots
        if self.no_optim_plots:
            print("Skipped plots")
            return

        print("Generating plots ...")
        fig1 = plot_optimization_history(study)
        fig2 = plot_optimization_history(study, target=lambda t: t.duration.total_seconds(), target_name="Duration")
        fig3 = plot_param_importances(study)
        fig4 = plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="Duration")
        fig5 = plot_slice(study)
        fig6 = plot_slice(study, target=lambda t: t.duration.total_seconds(), target_name="Duration")
        fig7 = plot_intermediate_values(study)
        # fig8 = plot_contour(study)
        fig9 = plot_edf(study)

        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
        fig6.show()
        fig7.show()
        # fig8.show()
        fig9.show()
