# Adapted from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/train.py
from pathlib import Path
import timeit
import argparse
import json

import numpy as np
import torch as th

from ba_nermo_rl_locomotion.experiment_manager import ExperimentManager
from ba_nermo_rl_locomotion.utils.utils import ALGOS, StoreDict
from ba_nermo_rl_locomotion.utils.nermo_monitor import NermoMonitor
# Configure environment parameters here:
from ba_nermo_rl_locomotion.env_kwargs import ENV_KWARGS, EVAL_KWARGS, MONITOR_KWARGS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="NermoMaxVelocity-v0", help="Environment ID")
    parser.add_argument("--env-kwargs", type=json.loads, default="{}", help='Overwrite the specified keyword arguments for the environment (pass in json format e.g. {"energy_penalty_weight": -0.1}')
    parser.add_argument("-en", "--experiment-name", type=str, help="Name for the experiment (should be unique within the specified env)")
    parser.add_argument("-ed", "--experiment-desc", type=str, default="", help="Detailed description for the experiment")
    parser.add_argument("-a", "--algo", help="RL algorithm (hyperparameters for each environment are defined in `hyperparameters/<algo>.yml`)", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="The number of timesteps to train with (-1 to use the number specified in the hyperparams file)", default=-1, type=int)
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite specified hyperparameter from the hyperparams file (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument("-s", "--seed", help="Random generator seed (-1 to choose a random seed)", type=int, default=-1)
    parser.add_argument("--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument("--vec-env", help="VecEnv type (auto to chose the type automatically depending on whether the algorithm is multiprocessing capable or not)", type=str, default="auto", choices=["auto", "dummy", "subproc"])
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). Can be a float in the range (0, 1) or and integer. A float x in (0, 1) will be interpreted as n = x * n_timesteps (where n_timesteps is the number of timesteps used for training)",
        default=100000,
        type=float,
    )
    parser.add_argument("--n-eval-episodes", help="Number of episodes to use for evaluation", default=12, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument("--checkpoint-freq", help="Save the model every n steps (if negative, no checkpoint). Can be a float in the range (0, 1) or and integer. A float x in (0, 1) will be interpreted as n = x * n_timesteps (where n_timesteps is the number of timesteps used for training)", default=100000, type=float)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )
    parser.add_argument("--device", help="Device on which the learning algorithm should be run. When set to auto, the code will run on the GPU (via cuda) if possible.", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument("--log-interval", help="Override log interval (if negative, no change)", default=-1, type=int)
    args = parser.parse_args()

    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    print("=" * 10, args.env, "=" * 10)
    print(f"Algo: {args.algo}")
    print(f"Seed: {args.seed}")
    
    env_kwargs = {
        **ENV_KWARGS[args.env],
        **args.env_kwargs,
        #add seed for random generator of commanded desired turning radius
        "seed_generator": args.seed
    }

    eval_env_kwargs = {
        **env_kwargs,
        **EVAL_KWARGS,
    }
    
    exp_manager = ExperimentManager(
        args,
        args.env,
        args.experiment_name,
        args.experiment_desc,
        args.algo,
        n_timesteps=args.n_timesteps,
        hyperparams=args.hyperparams,
        seed=args.seed,
        trained_agent=args.trained_agent,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        vec_env_type=args.vec_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        n_eval_envs=args.n_eval_envs,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer=args.save_replay_buffer,
        monitor_class=NermoMonitor,
        monitor_kwargs=MONITOR_KWARGS[args.env],
        device=args.device,
        log_interval=args.log_interval,
        verbose=args.verbose,
    )

    model = exp_manager.setup_experiment()
    
    start_training = timeit.default_timer()
    exp_manager.learn(model)
    stop_training = timeit.default_timer()

    exp_manager.save_trained_model(model, name="final_model")

    training_time_seconds = stop_training - start_training
    training_time_hours = training_time_seconds / 3600
    print(f"Training time: {training_time_seconds} s = {training_time_hours} h")
