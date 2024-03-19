import argparse
import os
from pathlib import Path

import numpy as np
import seaborn
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
from scipy import stats

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy

# from ba_nermo_rl_locomotion.experiment_manager import BASE_PATH


PARENT_DIR = Path(__file__).parent

# Activate seaborn
#seaborn.set()

parser = argparse.ArgumentParser("Plot evolution of the agent during the training in different experiments")
parser.add_argument("-p", "--base-path", help="Path under which the experiments are stored", type=str, required=True)
parser.add_argument("-es", "--experiments", nargs="+", help="Names of the experiments for which to plot the results", type=str, required=True)
parser.add_argument("-ls", "--labels", nargs="+", help="Graph labels to use for the experiments instead of the experiment name. Experiment names are used, if labels are not provided", type=str, required=False, default=None)
parser.add_argument("-cs", "--colors", nargs="+", help="Colors to use for the experiments instead of the default color cycle. Default color cycle is used, if colors are not provided", type=str, required=False, default=None)
parser.add_argument("-ms", "--metrics", nargs="+", help="Metrics that are to be plotted", type=str, required=False, default=["return", "primary_return", "secondary_return", "secondary_returns", "velocity", "speed", "power", "orthogonal_velocity", "yaw_rate"])
parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], type=str, default="steps")
parser.add_argument("--window-length", help="Length of the window that is used to smooth the data. Must be an odd number. Set to 0 in order to disable smoothing", type=int, default=51)
parser.add_argument("-a", "--aggregate", help="The function to use for aggregating the data from the trainings within the experiment", choices=["mean", "median", "max", "min"], type=str, default="median")
parser.add_argument("-r", "--range", help="The type of the range to display for the data in addition to the aggregate", choices=["min-max", "std-dev", "std-err", "iqr"], type=str, default=None)
parser.add_argument("--save-as", help="The format in which the plots should be stored. If omitted, figures are not stored", choices=["png", "jpg", "pdf", "svg"], type=str, default=None)
parser.add_argument("--name-prefix", help="Prefix for the filenames under which the figures are stored", type=str, default=None)
parser.add_argument("--do-not-show", action="store_true", default=False, help="Do not display the plots on the screen")
parser.add_argument("--style-sheet", help="The matplotlib style sheet to use", choices=["subfigure"], type=str, default=None)

args = parser.parse_args()

#latex-similar formatting
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('axes', axisbelow=True)

if args.style_sheet is not None:
    plt.style.use((PARENT_DIR / f"../styles/{args.style_sheet}.mplstyle").resolve().as_posix())

x_axis = {
    "steps": X_TIMESTEPS,
    "episodes": X_EPISODES,
    "time": X_WALLTIME
}[args.x_axis]
x_label = {
    "steps": "Steps",
    "episodes": "Episodes",
    "time": "Wall-clock time [h]",
}[args.x_axis]


base_path = Path(args.base_path)
for i, experiment in enumerate(args.experiments):
    experiment_path = (base_path / f"{experiment}").resolve()
    print(f"Experiment path: {experiment_path}")

    figures = {
        "return": {
            "values": [],
            "y_label": "Episode return",
            "key": "r"
        },
        "primary_return": {
            "values": [],
            "y_label": "Primary return $G_T^p$",
            "key": "sum_primary_reward"
        },
        "secondary_return": {
            "values": [],
            "y_label": "Secondary return $G_T^s$",
            "key": "sum_secondary_reward"
        },
        #add secondary returns in order to see influence of each component
        "secondary_returns": {
            "values": [],
            "y_label": "Secondary return $G_T^s$",
            "key": "sum_secondary_rewards"
        },
        "invariable_penalties": {
            "values": [],
            "y_label": "Total invariable penalties $G_T^i$",
            "key": "sum_invariable_penalties"
        },
        "velocity": {
            "values": [],
            "y_label": "Velocity $\overline{v}$ [m/s]",
            "key": "mean_velocity_in_desired_direction"
        },
        "speed": {
            "values": [],
            "y_label": "Speed $\overline{s}$ [m/s]",
            "key": "mean_speed"
        },
        "power": {
            "values": [],
            "y_label": "Power $\overline{P}$ [W]",
            "key": "mean_power"
        },
        "orthogonal_velocity": {
            "values": [],
            "y_label": "Orthogonal velocity $\overline{v}_o$ [m/s]",
            "key": "mean_velocity_in_orthogonal_direction"
        },
        "episode_length": {
            "values": [],
            "y_label": "Episode length",
            "key": "l"
        },
        "velocity_deviation": {
            "values": [],
            "y_label": "Velocity tracking error $\overline{e}_v$",
            "key": "mean_velocity_deviation",
        },
        "abs_velocity_deviation": {
            "values": [],
            "y_label": "Absolute velocity tracking error $\\frac{1}{T} \sum_{t=0}^{T-1} |e_v^t|$",
            "key": "mean_abs_velocity_deviation",
        },
        "yaw_rate": {
            "values": [],
            "y_label": "Yaw rate",
            "key": "mean_yaw_rate",
        },
        "curriculum_factor": {
            "values": [],
            "y_label": "Curriculum factor",
            "key": "mean_curriculum_factor",
        },
        "displacement_reward": {
            "values": [],
            "y_label": "Displacement reward",
            "key": "mean_displacement_reward",
        },
        "yaw_reward": {
            "values": [],
            "y_label": "Yaw rate tracking reward",
            "key": "mean_yaw_reward",
        },
        "velocity_tracking_reward": {
            "values": [],
            "y_label": "Velocity tracking reward",
            "key": "mean_velocity_tracking_reward",
        }
    }

    #choose which figures should be shown as stated in args.metrics
    figures = {
        key: figures[key] for key in args.metrics
    }

    xs = []

    trainings = os.listdir(experiment_path)
    if "tensorboard" in trainings:
        trainings.remove("tensorboard")

    print(f"Trainings: {trainings}")

    # Load data from the different trainings in the experiment in the figures dictionary
    for training in trainings:
        training_path = (experiment_path / training).resolve()

        # Load data from the monitor files created during training
        training_results_df = load_results(training_path)

        x, _ = ts2xy(training_results_df, x_axis)
        xs.append(x)

        for figure_name in figures:
            values = training_results_df.get(figures[figure_name]["key"])
            if values is not None:
                figures[figure_name]["values"].append(np.array(values))

    # The shortest training within the experiment determines the number of steps until which the values are considered
    x = min(xs, key=lambda x: len(x))

    for figure_name in figures:
        #get values from figure dictionary
        values = figures[figure_name]["values"]
        values = [v[0:len(x)] for v in values]

        if len(values) == 0:
            continue

        plt.figure(figure_name)

        # Apply filter to smooth data
        if args.window_length > 0 and args.window_length % 2 != 0:
            values = savgol_filter(values, args.window_length, 3, axis=1)

        # Compute aggregate
        if args.aggregate == "mean":
            aggregate_over_trainings = np.mean(values, axis=0)
        elif args.aggregate == "median":
            aggregate_over_trainings = np.median(values, axis=0)
        elif args.aggregate == "max":
            aggregate_over_trainings = np.max(values, axis=0)
        elif args.aggregate == "min":
            aggregate_over_trainings = np.min(values, axis=0)

        # Compute range
        if args.range == "min-max":
            range_lower = np.min(values, axis=0)
            range_upper = np.max(values, axis=0)
        elif args.range == "std-dev":
            mean_per_episode = np.mean(values, axis=0)
            std_dev_per_episode = np.std(values, axis=0, ddof=1)    # ddof = 1 to compute sample standard deviation
            range_lower = mean_per_episode - std_dev_per_episode
            range_upper = mean_per_episode + std_dev_per_episode
        elif args.range == "std-err":
            mean_per_episode = np.mean(values, axis=0)
            std_err_per_episode = stats.sem(values, axis=0)
            range_lower = mean_per_episode - std_err_per_episode
            range_upper = mean_per_episode + std_err_per_episode
        elif args.range == "iqr":
            range_lower = np.quantile(values, 0.25, axis=0)
            range_upper = np.quantile(values, 0.75, axis=0)

        # Plot aggregate (solid line) and optionally the range (shaded area)
        color_args = {}
        if args.colors is not None:
            color_args["color"] = color = args.colors[i]

        if args.range is not None:
            plt.fill_between(x, range_lower, range_upper, alpha=0.33, linewidth=0, **color_args, label='_nolegend_')
        plt.plot(x, aggregate_over_trainings, label=f"{experiment}", linewidth=1, **color_args)
        print("Last entry: " + f"{experiment}" + ": " + str(aggregate_over_trainings[-1]))

for figure_name in figures:
    if len(figures[figure_name]["values"]) == 0:
        continue
    # Label the figures
    y_label = figures[figure_name]["y_label"]
    plt.figure(figure_name)
    plt.xlabel(x_label, fontsize=23)
    plt.ylabel(y_label, fontsize=23)
    plt.grid(color="whitesmoke")
    if args.labels is not None:
        for count, label in enumerate(args.labels):
            label = label.replace("_", " ")
            label = label.replace("pos", "+")
            label = label.replace("neg", "-")
            args.labels[count] = label
        plt.legend(args.labels, loc="lower right")
    else:
        plt.legend()
    plt.tight_layout()

    if args.save_as is not None:
        os.makedirs(f"plots/{args.save_as}", exist_ok=True)
        filename = f"{figure_name}_{args.aggregate}"
        if args.range is not None:
            filename = f"{filename}_{args.range}"
        if args.name_prefix is not None:
            filename = f"{args.name_prefix}_{filename}"
        plt.savefig(f"plots/{args.save_as}/{filename}.{args.save_as}")

if not args.do_not_show:
    plt.show()
