# Adaptded from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/enjoy.py
from re import X
import threading
from os import name
from pathlib import Path
import timeit
import json
from nermo_rl_locomotion.env_kwargs import MONITOR_KWARGS
from nermo_rl_locomotion.utils.evaluation_util import generate_hildebrand_plot, plot_metrics_over_episode, cal_and_save_feet_metic
from nermo_rl_locomotion.utils.nermo_monitor import NermoMonitor
from nermo_rl_locomotion.utils.utils import ALGOS, OFF_POLICY_ALGOS
import numpy as np
import argparse
import tikzplotlib

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder
from mujoco_py import GlfwContext
import matplotlib.pyplot as plt

import nermo_rl_locomotion
from nermo_rl_locomotion.utils.env_util import create_enjoy_env, make_env
import scipy.fftpack

# Build paths relative to the absolute path of this script (or rather the script's parent dir)
# Source: https://stackoverflow.com/a/55051039
BASE_PATH = Path(__file__).parent
LEARNED_MODEL_PATH = (BASE_PATH / "../trained_agents/models/").resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tp", "--training-path", type=str, help=f"Path to the folder of the training from which the model(s) should be loaded. The path can be absolute or relative to {LEARNED_MODEL_PATH}", required=True)
    parser.add_argument("-mtl", "--models-to-load", nargs="+", type=str, default=["best_model"], help="Names of the models that should be loaded from the training path")
    parser.add_argument("-s", "--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--non-deterministic", action="store_true", default=False, help="Pick actions using a non-deterministic version of the policy")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward, if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--video-length", type=int, default=None, help="Record a video of the agent for n steps (do not specify in order to not record a video at all and render the agent behavior to the screen instead)")
    parser.add_argument("--video-resolution", nargs=2, type=int, default=[1440, 810], help="Resolution \"width height\" of the video that is to be recorded. The higher the resolution, the longer the recording takes.")
    parser.add_argument("--video-base-path", type=str, default=None, help=f"Path under which the recorded videos should be saved (do not specify in order to store the videos within a 'videos' folder at the specified training path). Note: can only be set when the training path is relative to {LEARNED_MODEL_PATH}")
    parser.add_argument("--cam-ids", nargs="+", type=int, default=[2], help="Ids of the MuJoCo cameras for which a video should be recorded (one video for each camera). For rendering to the screen the first camera in the given list is used for the initial point of view.")
    parser.add_argument("--n-episodes", type=int, default=-1, help="Number of rendered episodes to enjoy (-1 to loop until interrupted by ctrl+c or until the videos have been recorded)")
    parser.add_argument("--no-monitor-file", action="store_true", default=False, help="Do not write the aggregated episode information of the monitor to a file")
    parser.add_argument("--show-eval-plots", action="store_true", default=False, help="Plot diagrams for the locomotion evaluation after each episode")
    parser.add_argument("--style-sheet", help="The matplotlib style sheet to use for the eval plots", choices=["subfigure"], type=str, default=None)
    args = parser.parse_args()

    #falls später einmal gesammelte Positionplots gemacht werden sollen (ein Radius für verschieded Geschwindigkeiten z.B.)
    #XPOS = []
    #YPOS = []
    #add input possibility for radius and velocity for real time changes
    #first value is radius, second velocity
    #set default values
    #user_input = [-2, 0.15]
    user_input = 0
    command = 0
    radius_maze = -100
    velocity_maze = 0.2
    #create Function for checking changed user input
    # def checkChangeValue():
    #     global user_input
    #     while True:
    #         #rad, vel = input().split()
    #         command = input()
    #         #scale the values to the desired observation value
    #         #user_input[0] = 1 / float(rad)
    #         #user_input[1] = (float(vel) - 0.15) * 10
            
    #         if command == "a":
    #             user_input = 2
    #         elif command == "d":
    #             user_input = -1
    #         else:
    #             user_input = float(command)
    #         print(user_input)
    # #check user input with thread
    # threading.Thread(target=checkChangeValue).start()

    command_sequence = None
    # command_sequence = [(0.15, 10), (0.25, 10)]
    # command_sequence = [(0, 20)]
    # command_sequence = [(0.05, 20)]
    # command_sequence = [(0.09, 20)]
    # command_sequence = [(0.1, 20)]
    # command_sequence = [(0.2, 20)]
    # command_sequence = [(0.25, 20)]
    # command_sequence = [(0.12, 20)]
    # command_sequence = [(0.0, 2), (0.05, 2), (0.1, 2), (0.15, 2), (0.2, 2), (0.25, 5), (0.3, 5)]
    # command_sequence = [(0.0, 20), (0.05, 20), (0.1, 20), (0.15, 20), (0.2, 20), (0.25, 20), (0.3, 20)]
    command_sequence_name = "random_commands"
    command_sequence_name = "10_cm_per_s"
    # command_sequence_name = "15_to_25_cm_per_s"


    args.cam_ids = args.cam_ids if args.video_length is not None else [args.cam_ids[0]]
    #args.cam_ids = args.cam_ids if args.video_length is not None else [5] #1

    # Do not show eval plots when recording videos
    show_eval_plots = args.show_eval_plots and (args.video_length is None)
    if args.style_sheet is not None:
        plt.style.use((BASE_PATH / f"./styles/{args.style_sheet}.mplstyle").resolve().as_posix())

    render = args.video_length is None and (not args.no_rendering)

    # Construct absolute path of the training from which to load the models
    training_path = Path(args.training_path)
    if not training_path.is_absolute():
        training_path = (LEARNED_MODEL_PATH / training_path).resolve()
    
    # Load experiment_details
    experiment_details_path = (training_path / f"experiment_details.json").resolve()
    config_path = (training_path / f"config.json").resolve()
    if experiment_details_path.is_file():
        with open(experiment_details_path) as json_file:
            experiment_details = json.load(json_file)
    elif config_path.is_file():
        with open(config_path) as json_file:
            experiment_details = json.load(json_file)
    else:
        experiment_details = {}

    algo = experiment_details.get("algorithm", "ppo")

    env_id = experiment_details.get("env_id", "nermo-short-episodes-v0")

    if experiment_details.get("make_env_kwargs") is not None:
        # For legacy loading of the env_kwargs (structure of experiment_details.json changed after ExperimentManger was introduced)
        training_env_kwargs = experiment_details.get("make_env_kwargs", {}).get("env_kwargs", {})
    else:
        training_env_kwargs = experiment_details.get("env_kwargs", {})

    eval_env_kwargs = experiment_details.get("eval_env_kwargs", {})
    
    enjoy_env_kwargs = {
        **training_env_kwargs,
        **eval_env_kwargs,
        "terminate_upon_falling": False,
        "default_render_width": args.video_resolution[0],
        "default_render_height": args.video_resolution[1]
    }

    if env_id == "NermoCommandedVelocity-v0" and command_sequence is not None and len(command_sequence) > 0:
        enjoy_env_kwargs = {
            **enjoy_env_kwargs,
            "command_sample_frequency": None
        }
    else:
        command_sequence = None

    for model_name in args.models_to_load:
        # Load trained model
        model_path = (training_path / f"{model_name}").resolve()
        print(f"\nLoading model {model_name} from {model_path}")

        kwargs = dict(seed=args.seed)
        if algo in OFF_POLICY_ALGOS:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))

        for cam_id in args.cam_ids:
            if render or args.video_length is not None:
                print(f"\n{'Rendering' if args.video_length is None else 'Recording'} agent from camera with id {cam_id} ...")
            else:
                print(f"Evaluating agent ...")

            model = ALGOS[algo].load(model_path, **kwargs)

            # Reset the random seed to get to see the same behavior for every camera angle
            if args.seed is not None:
                set_random_seed(args.seed)

            # Configure the selected camera id
            enjoy_env_kwargs["fixed_cam_id"] = cam_id
            
            # Configure custom NermoMonitor for logging performance metrics
            monitor_path = (training_path / "enjoy").resolve()
            if not args.no_monitor_file:
                monitor_path.mkdir(parents=True, exist_ok=True)
            filename = f"{monitor_path.as_posix()}/{model_name}_seed_{args.seed}" if not args.no_monitor_file else None
            if command_sequence is not None and filename is not None:
                filename = f"{filename}_{command_sequence_name}"
            monitor_kwargs = {
                **MONITOR_KWARGS[env_id],
                "filename": filename,
                "verbose": 1,
                "real_time_info_interval": 100
            }

            # Create enjoy_env (with the specified camera angle)
            enjoy_env, raw_env = create_enjoy_env(
                env_id=env_id,
                env_kwargs=enjoy_env_kwargs,
                experiment_details=experiment_details,
                stats_path=training_path.as_posix(),
                seed=args.seed,
                normalize_reward=args.norm_reward,
                monitor_class=NermoMonitor,
                monitor_kwargs=monitor_kwargs
            )

            # Wrap env with video recorder, if a video should be recorded
            if args.video_length is not None:
                # Fix for making offscreen rendering possible based on https://github.com/openai/mujoco-py/issues/390#issuecomment-525385434
                # Try to comment out the line in case this it an error on your machine
                GlfwContext(offscreen=True)  # Create a window to init GLFW.

                if args.video_base_path is not None:
                    relative_training_path = training_path.relative_to(LEARNED_MODEL_PATH)
                    video_folder = Path(f"{args.video_base_path}/{relative_training_path}/{model_name}")
                else:
                    video_folder = (training_path / f"videos/{model_name}").resolve()

                if not video_folder.exists():
                    video_folder.mkdir(parents=True, exist_ok=True)

                name_prefix = (
                    f"{model_name}_cam_{cam_id if cam_id is not None else 'free'}"
                    f"_seed_{args.seed}"
                )

                if command_sequence_name is not None and command_sequence is not None:
                    name_prefix = f"{command_sequence_name}_" + name_prefix
                    with open((video_folder / f"{command_sequence_name}_command_sequence").resolve(), "w+") as f:
                        f.write("\n".join(str(item) for item in command_sequence))

                enjoy_env = VecVideoRecorder(
                    enjoy_env,
                    video_folder=video_folder,
                    record_video_trigger=lambda x: x == 0,
                    name_prefix=name_prefix,
                    video_length=args.video_length
                )

            # Initialize variables for episode evaluation 
            total_steps = 0
            episode_num = 0

            infos = []
            episode_infos = []

            command_index = 0
            command_steps = 0
            command = None
            if command_sequence is not None and len(command_sequence) > 0:
                command = command_sequence[command_index]
                raw_env.set_velocity_command(command[0])
            
            # Let agent act in environment til loop is interrupted (Ctrl + c) or til video_length steps have been recorded 
            observations = enjoy_env.reset()
            real_time = 0
            try:
                while (args.n_episodes == -1 or episode_num < args.n_episodes):
                    action, _states = model.predict(observations, deterministic=(not args.non_deterministic))
                    observations, rewards, dones, infos = enjoy_env.step(action)

                    observation, reward, done, info = observations[0], rewards[0], dones[0], infos[0]
                    # print(info.get("ctrl"))
                    #for commanding: give observations at special points
                    real_time += raw_env.dt
                    real_time_rounded = round(real_time,1)
                    if real_time_rounded == 0.7:
                        radius_maze = -1
                        velocity_maze = 0.15
                    if real_time_rounded == 5.1:
                        velocity_maze = 0.15
                        radius_maze = 0.4
                    if real_time_rounded == 22.0:
                        radius_maze = -1
                        velocity_maze = 0.2
                    if real_time_rounded == 23.5:
                        radius_maze = 0.5
                        velocity_maze = 0.2
                    if real_time_rounded == 27.0:
                        radius_maze = 100
                        velocity_maze = 0.2
                    # print(observation.shape)
                    # observation[15] = 0.2
                    #observation[19] = (1/radius_maze)/2
                    #observation[20] = (velocity_maze - 0.175) * 40 / 3
                    #observation[19] = user_input[0]
                    #observation[20] = user_input[1]
                    #print(real_time_rounded)
                    episode_infos.append(info)

                    # if render:
                    #     enjoy_env.render()

                    total_steps += 1
                    command_steps += 1
                    
                    #print(str(real_time) + "    " + str(radius_maze))
                    if command is not None:
                        if command_steps * raw_env.dt == command[1]:
                            command_index += 1
                            command = command_sequence[command_index % len(command_sequence)]
                            raw_env.set_velocity_command(command[0])
                            command_steps = 0

                    if done:
                        episode_num += 1
                        # print(type(np.array(episode_infos)))
                        # foot_contacts = [info['foot_contacts'] for info in episode_infos]
                        # np.savetxt((None / f"foot_contacts.csv").resolve(), foot_contacts , delimiter=",")
                        #for automatically saving the position plots
                        #image_save_path = (training_path / ".." / str(raw_env.desired_velocity) / (str(raw_env.desired_turning_radius) + ".svg")).resolve()
                        cal_and_save_feet_metic(
                                episode_infos=episode_infos,
                                metrics=["foot_positions"], #only z-positions
                                #caution: positions are negative
                                x_scale_factor = raw_env.dt,
                                x_label="time [s]",
                                y_label="z-Position [m]",
                                figure_name=f"foot_positions_episode_{episode_num}",
                                show_legend=True,
                                smoothing_window=None,
                                csv_path=training_path,
                                model_name=model_name,
                        )
                        if show_eval_plots:
                            smoothing_window = 51
                            x_scale_factor = raw_env.dt
                            x_label = "Time [sec]"
                            smoothed_only = True
                            # if info.get("velocity_in_desired_direction") is not None:
                            #     plot_metrics_over_episode(
                            #         episode_infos=episode_infos,
                            #         metrics=["speed", "smoothed_speed"],
                            #         metric_labels = ["Speed $s_t$", "Smoothed speed $s_s$"],
                            #         show_legend = True,
                            #         x_scale_factor=x_scale_factor,
                            #         x_label=x_label,
                            #         y_label="Speed [m/s]",
                            #         figure_name=f"speed_episode_{episode_num}",
                            #         smoothing_window=None,
                            #         smoothed_only=not smoothed_only,
                            #     )
                            # plot_metrics_over_episode(
                            #         episode_infos=episode_infos,
                            #         metrics=["speed"],
                            #         metric_labels = ["Speed $s_t$"],
                            #         show_legend = True,
                            #         x_scale_factor=x_scale_factor,
                            #         x_label=x_label,
                            #         y_label="Speed [m/s]",
                            #         figure_name=f"speed_episode_{episode_num}",
                            #         smoothing_window=None,
                            #         smoothed_only=not smoothed_only,
                            #     )
                            if info.get("velocity_in_desired_direction") is not None:
                                vel_metrics=["velocity_in_desired_direction"]
                                metric_labels = None
                                show_legend = True
                                if info.get("desired_velocity") is not None:
                                    vel_metrics.append("desired_velocity")
                                    metric_labels = ["Velocity $v_t$", "Desired velocity $v_d$"]
                                    show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=vel_metrics,
                                    x_scale_factor=x_scale_factor,
                                    x_label=x_label,
                                    y_label="Velocity [m/s]",
                                    figure_name=f"velocity_episode_{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=smoothing_window,
                                    smoothed_only=smoothed_only,
                                )
                            if info.get("angle_m1_rl") is not None:                    
                                angle_metrics = ["angle_m1_rl", "angle_m1_rr", "angle_m1_r_max", "angle_m1_r_min"]
                                metric_labels = ["M1 RL", "M1 RR", "M1 Rear Bounds", None]
                                show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=angle_metrics,
                                    x_scale_factor=x_scale_factor,
                                    x_label=x_label,
                                    y_label="Motor Angles M1 Rear [rad]",
                                    figure_name=f"rear_feet_angles_m1{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=None,
                                    smoothed_only=not smoothed_only,
                                    csv_path=training_path,
                                    model_name=model_name,
                                    colors=["#ff6666", "#8080ff", "#000000", "#000000"],
                                    legend_loc="upper right"
                                )
                            if info.get("angle_m2_rl") is not None:                    
                                angle_metrics = ["angle_m2_rl", "angle_m2_rr", "angle_m2_r_max", "angle_m2_r_min"]
                                metric_labels = ["M2 RL", "M2 RR", "M2 Rear Bounds", None]
                                show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=angle_metrics,
                                    x_scale_factor=x_scale_factor,
                                    x_label=x_label,
                                    y_label="Motor Angles M2 Rear [rad]",
                                    figure_name=f"rear_feet_angles_m2{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=None,
                                    smoothed_only=not smoothed_only,
                                    csv_path=training_path,
                                    model_name=model_name,
                                    colors=["#ff6666", "#8080ff", "#000000", "#000000"],
                                    legend_loc="lower right"
                                )
                            if info.get("angle_m1_fl") is not None:                    
                                angle_metrics = ["angle_m1_fl", "angle_m1_fr", "angle_m1_f_max", "angle_m1_f_min"]
                                metric_labels = ["M1 FL", "M1 FR", "M1 Front Bounds", None]
                                show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=angle_metrics,
                                    x_scale_factor=x_scale_factor,
                                    x_label=x_label,
                                    y_label="Motor Angles M1 Front [rad]",
                                    figure_name=f"front_feet_angles_m1_{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=None,
                                    smoothed_only=not smoothed_only,
                                    csv_path=training_path,
                                    model_name=model_name,
                                    colors=["#ff6666", "#8080ff", "#000000", "#000000"],
                                    legend_loc="lower right"
                                )
                            if info.get("angle_m2_fl") is not None:                    
                                angle_metrics = ["angle_m2_fl", "angle_m2_fr", "angle_m2_f_max", "angle_m2_f_min"]
                                metric_labels = ["M2 FL", "M2 FR", "M2 Front Bounds", None]
                                show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=angle_metrics,
                                    x_scale_factor=x_scale_factor,
                                    x_label=x_label,
                                    y_label="Motor Angles M2 Front [rad]",
                                    figure_name=f"front_feet_angles_m2{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=None,
                                    smoothed_only=not smoothed_only,
                                    csv_path=training_path,
                                    model_name=model_name,
                                    colors=["#ff6666", "#8080ff", "#000000", "#000000"],
                                    legend_loc="lower right"
                                )
                            # if info.get("angle_values") is not None:                    
                            #     angle_metrics = ["angle_values"]
                            #     metric_labels = ["angle_values"]
                            #     show_legend = True
                            #     plot_metrics_over_episode(
                            #         episode_infos=episode_infos,
                            #         metrics=angle_metrics,
                            #         x_scale_factor=x_scale_factor,
                            #         x_label=x_label,
                            #         y_label="Motor Angles [rad]",
                            #         figure_name=f"angle_values{episode_num}",
                            #         metric_labels=metric_labels,
                            #         show_legend=show_legend,
                            #         smoothing_window=None,
                            #         smoothed_only=not smoothed_only,
                                # )
                            if info.get("spine_angle") is not None:                    
                                angle_metrics = ["spine_angle"]
                                metric_labels = ["Spine motor angle"]
                                show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=angle_metrics,
                                    x_scale_factor=x_scale_factor,  
                                    x_label=x_label,
                                    y_label="Spine motor angle [rad]",
                                    figure_name=f"spine_angle{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=None,
                                    smoothed_only=not smoothed_only,
                                    csv_path=training_path,
                                    model_name=model_name
                                )


                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=["return"],
                            #     x_scale_factor=x_scale_factor,
                            #     x_label=x_label,
                            #     y_label="Return",
                            #     figure_name=f"return_episode_{episode_num}",
                            #     smoothing_window=smoothing_window,
                            #     smoothed_only=smoothed_only,
                            # )
                            
                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=["curriculum_factors"],
                            #     x_scale_factor=x_scale_factor,
                            #     x_label=x_label,
                            #     y_label="curriculum_factors",
                            #     figure_name=f"curriculum_factors_{episode_num}",
                            #     smoothing_window=smoothing_window,
                            #     smoothed_only=smoothed_only,
                            # )
                            #Split reward plot in different components (turning radius env)
                            # if info.get("displacement_reward") is not None:
                            #     metrics=["displacement_reward"]
                            #     metric_labels = ["Displacement Reward"]
                            #     show_legend = True
                            #     if info.get("yaw_reward") is not None:
                            #         metrics.append("yaw_reward")
                            #         metric_labels = ["Displacement Reward", "Yaw-keeping Reward"]
                            #         show_legend = True
                            #         if info.get("velocity_tracking_reward") is not None:
                            #             metrics.append("velocity_tracking_reward")
                            #             metric_labels = ["Displacement Reward", "Yaw Tracking Reward", "Velocity Tracking Reward"]
                            #             show_legend = True
                            #     plot_metrics_over_episode(
                            #         episode_infos=episode_infos,
                            #         metrics=metrics,
                            #         x_scale_factor=x_scale_factor,
                            #         x_label=x_label,
                            #         y_label="Return components",
                            #         figure_name=f"Return Components_{episode_num}",
                            #         metric_labels=metric_labels,
                            #         show_legend=show_legend,
                            #         smoothing_window=None,
                            #         smoothed_only=not smoothed_only,
                            #     )
                            # #Plot development of reward components (turning radius env)
                            # if info.get("angle_displacement") is not None:
                            #     metrics=["primary_reward", "displacement_reward", "yaw_reward"]
                            #     metric_labels = ["Reward", "Angle displacement reward", "Yaw reward"]
                            #     show_legend = True
                            # if info.get("velocity_tracking_reward") is not None:
                            #     metrics.append("velocity_tracking_reward")
                            #     metric_labels.append("Velocity tracking reward")
                            #     show_legend = True
                            # if info.get("position_tracking_reward") is not None:
                            #     metrics.append("position_tracking_reward")
                            #     metric_labels.append("Position tracking reward")
                            #     show_legend = True
                            #     plot_metrics_over_episode(
                            #         episode_infos=episode_infos,
                            #         metrics=metrics,
                            #         x_scale_factor=x_scale_factor,
                            #         x_label=x_label,
                            #         y_label="Reward components",
                            #         figure_name=f"Reward Components_{episode_num}",
                            #         metric_labels=metric_labels,
                            #         show_legend=show_legend,
                            #         smoothing_window=None,
                            #         smoothed_only=not smoothed_only,
                            #     )
                            if info.get("yaw_rate") is not None:
                                vel_metrics=["yaw_rate"]
                                metric_labels = None
                                show_legend = False
                                if info.get("desired_yaw_rate") is not None:
                                    vel_metrics.append("desired_yaw_rate")
                                    metric_labels = ["Yaw rate $w_t$", "Desired yaw rate $w_d$"]
                                    show_legend = True
                                if info.get("smoothed_yaw_rate") is not None:
                                    vel_metrics.append("smoothed_yaw_rate")
                                    metric_labels = ["Yaw rate", "Desired yaw rate", "Smoothed yaw rate"]
                                    show_legend = True
                                plot_metrics_over_episode(
                                    episode_infos=episode_infos,
                                    metrics=vel_metrics,
                                    x_scale_factor=x_scale_factor,
                                    x_label=x_label,
                                    y_label="Yaw rate [rad/s]",
                                    figure_name=f"yaw_episode_{episode_num}",
                                    metric_labels=metric_labels,
                                    show_legend=show_legend,
                                    smoothing_window=None,
                                    smoothed_only=not smoothed_only,
                                )
                            if info.get("desired_turning_radius") is not None:
                                    metrics = ["desired_turning_radius"]
                            if info.get("position") is not None:
                                    metrics = ["position"]
                                    # metrics.append("position")
                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=metrics,
                            #     x_scale_factor=x_scale_factor,
                            #     #caution: positions are negative
                            #     x_label="x-Position [m]",
                            #     y_label="y-Position [m]",
                            #     figure_name=f"position_episode_{episode_num}",
                            #     show_legend=True,
                            #     smoothing_window=None,
                            #     smoothed_only=not smoothed_only,
                            #     #for automatically saving the position plots
                            #     #image_save_path=image_save_path
                            # )
                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=["ctrl"],
                            #     x_scale_factor=x_scale_factor,
                            #     # caution: positions are negative
                            #     x_label=x_label,
                            #     y_label=None,
                            #     figure_name=None,
                            #     show_legend=True,
                            #     smoothing_window=None,
                            #     smoothed_only=not smoothed_only,
                            #     # for automatically saving the position plots
                            #     # image_save_path=image_save_path
                            # )

                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=["actuator_force"],
                            #     x_scale_factor=x_scale_factor,
                            #     # caution: positions are negative
                            #     x_label=x_label,
                            #     y_label=None,
                            #     figure_name=None,
                            #     show_legend=True,
                            #     smoothing_window=None,
                            #     smoothed_only=not smoothed_only,
                            #     # for automatically saving the position plots
                            #     # image_save_path=image_save_path
                            # )
                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=["actuator_velocity"],
                            #     x_scale_factor=x_scale_factor,
                            #     # caution: positions are negative
                            #     x_label=x_label,
                            #     y_label=None,
                            #     figure_name=None,
                            #     show_legend=True,
                            #     smoothing_window=None,
                            #     smoothed_only=not smoothed_only,
                            #     # for automatically saving the position plots
                            #     # image_save_path=image_save_path
                            # )

                            #falls später einmal gesammelte Positionplots gemacht werden sollen (ein Radius für verschieded Geschwindigkeiten z.B.)
                            #XPOS.append(x)
                            #YPOS.append(y)
                            
                            plot_metrics_over_episode(
                                episode_infos=episode_infos,
                                metrics=["foot_positions"], #only z-positions
                                x_scale_factor=x_scale_factor,
                                #caution: positions are negative
                                x_label="time [s]",
                                y_label="z-Position [m]",
                                figure_name=f"foot_positions_episode_{episode_num}",
                                show_legend=True,
                                smoothing_window=None,
                                smoothed_only=not smoothed_only,
                                csv_path=training_path,
                                model_name=model_name,
                            )
                            # plot_metrics_over_episode(
                            #     episode_infos=episode_infos,
                            #     metrics=["tangential_foot_velocities"],
                            #     x_scale_factor=x_scale_factor,
                            #     x_label="time [s]",
                            #     y_label="tangential velocity [m/s]",
                            #     figure_name=f"tangential_foot_velocities_episode_{episode_num}",
                            #     show_legend=True,
                            #     smoothing_window=None,
                            #     smoothed_only=not smoothed_only,
                            # )
                            plot_metrics_over_episode(
                                episode_infos=episode_infos,
                                metrics=["power"],
                                x_scale_factor=x_scale_factor,
                                x_label=x_label,
                                y_label="Power [W]",
                                figure_name=f"power_episode_{episode_num}",
                                show_legend=True,
                                smoothing_window=smoothing_window,
                                smoothed_only=smoothed_only,
                            )
                            # generate_hildebrand_plot(
                            #     episode_infos=episode_infos,
                            #     hildebrand_steps_start=50,
                            #     hildebrand_steps=200,
                            #     hildebrand_foot_order= ["rh", "rf", "lf", "lh"] ,
                            #     phase_gap_tolerance=3,
                            #     min_phase_steps=5,
                            #     x_scale_factor=x_scale_factor,
                            #     x_label=x_label,
                            #     episode_num=episode_num,
                            # )
                            plt.show(block=True)
                            
                        
                        episode_infos = []

                        if args.video_length is not None and total_steps >= args.video_length:
                            break

                        observations = enjoy_env.reset()
                        episode_start = timeit.default_timer()
                    
                    if args.video_length is not None and total_steps >= args.video_length:
                        break
                    
                    # 
                    if episode_num == 1:
                        break
                #falls später einmal gesammelte Positionplots gemacht werden sollen (ein Radius für verschieded Geschwindigkeiten z.B.)
                #print(XPOS)
            except KeyboardInterrupt:
                #falls später einmal gesammelte Positionplots gemacht werden sollen (ein Radius für verschieded Geschwindigkeiten z.B.)
                #print(XPOS)
                print("\nInterrupted rendering loop")
            
            enjoy_env.close()
