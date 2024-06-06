import argparse
import os
from pathlib import Path

import numpy as np
import seaborn
import torch as th

from nermo_rl_locomotion.experiment_manager import ExperimentManager
from nermo_rl_locomotion.utils.utils import ALGOS, StoreDict
# Configure environment parameters here:
from nermo_rl_locomotion.env_kwargs import ENV_KWARGS

seaborn.set()


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="NermoMaxVelocity-v0", help="Environment ID")
    parser.add_argument("-a", "--algo", help="RL algorithm (hyperparameters for each environment are defined in `hyperparameters/<algo>.yml` but the ones being tuned will be overridden)", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="The number of timesteps the algorithm should be run for each hyperparameter sample (i.e., trial) (-1 to use the number specified in the hyperparams file)", default=-1, type=int)
    parser.add_argument("-s", "--seed", help="Random generator seed (-1 to choose a random seed)", type=int, default=-1)
    parser.add_argument("--vec-env", help="VecEnv type (auto to chose the type automatically depending on whether the algorithm is multiprocessing capable or not)", type=str, default="auto", choices=["auto", "dummy", "subproc"])
    parser.add_argument("--n-eval-episodes", help="Number of episodes to use for evaluation", default=3, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters during this run. Set to 0 to only load and show results of the study with the specified study name.",
        type=int,
        default=10,
    )
    parser.add_argument("--n-jobs", help="Number of parallel jobs when optimizing hyperparameters", type=int, default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument("--n-startup-trials", help="Number of trials before using optuna sampler", type=int, default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--storage", help="Database storage path if distributed optimization should be used", type=str, default=None
    )
    parser.add_argument("--study-name", help="Study name for distributed optimization. Running with a study name that already exists in the given storage allows to resume that study instead of creating a new one.", type=str, default=None)
    parser.add_argument(
        "--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots at the end of the optimization run."
    )
    parser.add_argument("--print-n-best-trials", help="Show final values and params for the n best trials. The best trial will always be printed.", type=int, default=5)
    parser.add_argument(
        "--save-n-best-hyperparameters",
        help="Save the hyperparameters for the n best trials in a dedicated directory <log_path>/best_hyperparams",
        type=int,
        default=0,
    )
    parser.add_argument("--device", help="Device on which the learning algorithm should be run. When set to auto, the code will run on the GPU (via cuda) if possible.", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
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

    exp_manager = ExperimentManager(
        args,
        args.env,
        "",
        "",
        args.algo,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        env_kwargs=ENV_KWARGS[args.env],
        vec_env_type=args.vec_env,
        n_eval_episodes=args.n_eval_episodes,
        n_eval_envs=args.n_eval_envs,
        optimize_hyperparameters=True,
        storage=args.storage,
        study_name=args.study_name,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        sampler=args.sampler,
        pruner=args.pruner,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        no_optim_plots=args.no_optim_plots,
        print_n_best_trials=args.print_n_best_trials,
        save_n_best_hyperparameters=args.save_n_best_hyperparameters,
        device=args.device,
        verbose=args.verbose,
    )

    exp_manager.setup_experiment()

    exp_manager.hyperparameter_optimization()
