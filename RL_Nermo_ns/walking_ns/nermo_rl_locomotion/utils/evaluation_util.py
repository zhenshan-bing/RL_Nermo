from typing import Dict, List

import numpy as np
import seaborn
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import savgol_filter
from math import pi
import circle_fit as cf
import tikzplotlib
import pandas as pd
import scipy.fftpack
from scipy.signal import correlate, find_peaks, savgol_filter

# Activate seaborn
#seaborn.set()

#latex-similar formatting
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def find_small_change_intervals(y_values, threshold=0.03):
    constant_intervals = []
    start_index = None

    for i in range(0, len(y_values)):
        # Check if the change in y-value is smaller than the threshold
        if abs(y_values[i] - y_values[i - 1]) < threshold:
            if start_index is None:
                start_index = i - 1
        else:
            # If the threshold is exceeded, record the interval if it has been started
            if start_index is not None:
                if i - 1 -start_index> 20:
                    constant_intervals.append((start_index, i - 1))
                    start_index = None
            if start_index ==0:
                # if i - 1 -start_index> 20:
                constant_intervals.append((start_index, i - 1))
                start_index = None

    # Handling the last interval
    if start_index is not None:
        constant_intervals.append((start_index, len(y_values) - 1))

    return constant_intervals

def cal_and_save_feet_metic(episode_infos: List[Dict[str, any]], metrics: List[str], x_scale_factor=1, x_label=None, y_label=None, smoothing_window=51, smoothed_only = False, figure_name=None, metric_labels=None, show_legend=False, image_save_path=None, csv_path=None, model_name=None, colors=None, legend_loc=None):
    # metric == "foot_positions"
    metric_values = [info["foot_positions"] for info in episode_infos]
    t = [i * x_scale_factor for i in range(len(episode_infos))]
    lf=[]
    rf=[]
    rh=[]
    lh=[]
    feet=[lf, rf, rh, lh]
    feets=['lf', 'rf', 'rh', 'lh']
    feet_y_positions = np.empty((2000, 0))
    #append values from "metric_values" to according array
    #first loop in metric_values dict to get position dict from one time step of all feet
    for positions in metric_values:
        #second loop for enumerating positions of each foot in one time step
        for count, foot in enumerate(positions):
            feet[count].append(positions[foot][1]) #[2] for y-positions

    for j in range(len(feet)):
        column = -np.array(feet[j]).reshape(-1, 1)

        # Concatenate along the second axis (columns)
        feet_y_positions = np.concatenate((feet_y_positions, column), axis=1)
    print(feet_y_positions.shape)
    # plt.plot(t, feet[0], label = "lf")
    # plt.plot(t, feet[1], label = "rf")
    # plt.plot(t, feet[2], label = "rh")
    # plt.plot(t, feet[3], label = "lh")
    #create subplots
    try:
        gait_data = pd.read_csv('gait_data_0206_s_max.csv')
    except FileNotFoundError:
        # gait_data = pd.DataFrame(columns=['desired_velocity', 'velocity', 'power', 'foot', 'stride_time', 'frequency', 'stride_length', 'stance_time', 'swing_time', 'duty_factor'])
        gait_data = pd.DataFrame(columns=['velocity', 'power', 'foot', 'stride_time', 'frequency', 'stride_length', 'stance_time', 'swing_time', 'duty_factor'])

    for j in range(len(feets)):
        y_values = feet_y_positions[:,j]
        # peaks, _ = find_peaks(-y_values)  # Finding troughs in y_values
        # troughs, _ = find_peaks(y_values)  # Finding peaks in y_values
        constant_intervals = find_small_change_intervals(y_values, 0.002)
        print(constant_intervals)
        interval_timestamps = [(t[start], t[end]) for start, end in constant_intervals]
        # print(interval_timestamps)


        # calculate frequency
        peridlength = []
        for i in range(10,len(constant_intervals)):
            peridlength.append(t[constant_intervals[i][0]]-t[constant_intervals[i-1][0]])   

        print(feets[j]+"_"+"stride time: ",np.mean(peridlength))
        print(feets[j]+"_"+"freq: ",1/np.mean(peridlength))

        # calculate stride length
        stride_length = []
        index = []
        for i in range(4,len(constant_intervals)-4):
            stride_length.append(feet_y_positions[constant_intervals[i][0],j]-feet_y_positions[constant_intervals[i-1][1],j])
            # index.append(constant_intervals[i][0])
            # index.append(constant_intervals[i-1][1])
            
        # print(stride_length)
        # print(index)
        print(feets[j]+"_"+"stride length: ",np.mean(stride_length))

        # calculate stance time
        stance_time = []
        index = []
        for i in range(4,len(constant_intervals)-4):
            stance_time.append(t[constant_intervals[i][1]]-t[constant_intervals[i][0]])
            # index.append(constant_intervals[i][1])
        #     index.append([constant_intervals[i][1],constant_intervals[i][0]])
        # print(index)
        print(feets[j]+"_"+"stance time: ",np.mean(stance_time))

        # calculate swing time
        swing_time = []
        index = []
        for i in range(4,len(constant_intervals)-4):
            swing_time.append(t[constant_intervals[i][0]]-t[constant_intervals[i-1][1]])
        #     index.append([constant_intervals[i][0],constant_intervals[i-1][1]])
        # print(index)
        print(feets[j]+"_"+"swing time: ",np.mean(swing_time))

        Duty_factor = np.mean(stance_time)/(np.mean(stance_time)+np.mean(swing_time))
        print(feets[j]+"_"+"Duty factor: ",Duty_factor)


        velocity = [info["velocity_in_desired_direction"] for info in episode_infos]
        mean_velocity = np.mean(velocity[200:])
        power = [info["power"] for info in episode_infos]
        mean_power = np.mean(power[200:])
        # desired_velocity= [info["desired_velocity"] for info in episode_infos]
        # mean_desired_velocity = np.mean(desired_velocity[200:])
        spine_motor= [info["spine_angle"] for info in episode_infos]
        # print(spine_motor)
        offset = 400
        # Sample spacing (timestep)
        T = 0.01
        t_offset = t[offset:]
        N = len(t_offset)
        # if spine_motor is not None: 
        spine_motor = spine_motor[offset:]
        #center oscillation (subtract mean of oscillation)
        mean_offset = np.mean(spine_motor)
        spine_motor = spine_motor - mean_offset
        #FFT
        yf = scipy.fftpack.fft(spine_motor)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        yf_plot = 2.0/N * np.abs(yf[:N//2])
        peaks, _ = find_peaks(yf_plot, height=0.01)
        amp = yf_plot[peaks]
        if amp.size > 0:
            index = np.argmax(amp)
            # index = np.argmax(amp)
            amp = amp[index]
            # Proceed with operations that use 'index'
        else:
            # Handle the case when amp is empty
            amp=0
        


        data = {
            # 'desired_velocity': mean_desired_velocity,
            'velocity': mean_velocity,
            'power': mean_power,
            'spine_amplitude': amp,
            'foot': feets[j],
            'stride_time': np.mean(peridlength),
            'frequency': 1 / np.mean(peridlength),
            'stride_length': np.mean(stride_length),
            'stance_time': np.mean(stance_time),
            'swing_time': np.mean(swing_time),
            'duty_factor': Duty_factor
        }

        # append the dictionary to the DataFrame
        # data_df = pd.DataFrame([data])
        # gait_data = pd.concat([gait_data, data_df], ignore_index=True)
        # print(gait_data)
        # gait_data.to_csv('gait_data_0206_s_max.csv', index=False)


def plot_metrics_over_episode(episode_infos: List[Dict[str, any]], metrics: List[str], x_scale_factor=1, x_label=None, y_label=None, smoothing_window=51, smoothed_only = False, figure_name=None, metric_labels=None, show_legend=False, image_save_path=None, csv_path=None, model_name=None, colors=None, legend_loc=None):
    t = [i * x_scale_factor for i in range(len(episode_infos))]
    
    plt.figure(figure_name)

    for i, metric in enumerate(metrics):
        metric_values = [info[metric] for info in episode_infos]
        label = metric_labels[i] if metric_labels is not None else metric
        
        if not smoothed_only:
            #special treatment for "position", since the array is 2-D and not over time, but x-y-Position
            if metric == "position":
                x=[]
                y=[]
                for pos in metric_values:
                    #change signs for intuitive visualization like in simulation
                    x.append(-pos[0])
                    y.append(-pos[1])
                plt.scatter(x, y, label = "Position robot", s=1)
                #plot and fit a circle to xy-position data
                plot_fitted_circle(x, y) 
                #plot_desired_circle(1.0)
                #return x,y
                #tikzplotlib.save("test_1.tex")


            elif metric == "ctrl":
                x=[]
                y = []
                for ctrl in metric_values:
                    x.append(ctrl[0])
                    y.append(ctrl[2])
                a = np.ones(500)
                fig, axs = plt.subplots(2, 1, num="ctrl")
                axs[0].plot(t[500:1000], x[500:1000], 'tab:blue')
                axs[0].plot(t[500:1000], 0.2*a, 'k--', linewidth=1)
                axs[0].plot(t[500:1000], -2.6*a, 'k--', linewidth=1)
                axs[0].set_title("leg ctrl")
                axs[1].plot(t[500:1000], y[500:1000], 'tab:orange')
                axs[1].plot(t[500:1000], 1*a, 'k--', linewidth=1)
                axs[1].plot(t[500:1000], -2.6*a, 'k--', linewidth=1)
                axs[1].set_title("thigh ctrl")

            # elif metric == "z_orientation":
            #     x=[]
            #     for z_orientation in metric_values:
            #         x.append(z_orientation)
            #     plt.plot(t,x)
            elif metric == "actuator_force":
                x = []
                y = []
                a = np.ones(500)
                for actuator_force in metric_values:
                    x.append(actuator_force[0])
                    y.append(actuator_force[2])
                fig, axs = plt.subplots(2, 1, num="actuator_force")
                axs[0].plot(t[500:1000], x[500:1000], 'tab:blue')
                axs[0].plot(t[500:1000], 0.157 * a, 'k--', linewidth=1)
                axs[0].plot(t[500:1000], -0.157 * a, 'k--', linewidth=1)
                axs[0].set_title("leg actuator force")
                axs[1].plot(t[500:1000], y[500:1000], 'tab:orange')
                axs[1].plot(t[500:1000], 0.157 * a, 'k--', linewidth=1)
                axs[1].plot(t[500:1000], -0.157 * a, 'k--', linewidth=1)
                axs[1].set_title("thigh actuator force")

            elif metric == "actuator_velocity":
                x=[]
                y = []
                for actuator_velocity in metric_values:
                    x.append(actuator_velocity[0])
                    y.append(actuator_velocity[2])
                a = np.ones(500)
                fig, axs = plt.subplots(2, 1, num="actuator_velocity")
                axs[0].plot(t[500:1000], x[500:1000], 'tab:blue')
                axs[0].plot(t[500:1000], 6 * a, 'k--', linewidth=1)
                axs[0].plot(t[500:1000], -6 * a, 'k--', linewidth=1)
                axs[0].set_title("leg actuator velocity")
                axs[1].plot(t[500:1000], y[500:1000], 'tab:orange')
                axs[1].plot(t[500:1000], 6 * a, 'k--', linewidth=1)
                axs[1].plot(t[500:1000], -6 * a, 'k--', linewidth=1)
                axs[1].set_title("thigh actuator velocity")


            elif metric == "desired_turning_radius":
                fig = plt.gcf()
                fig.set_size_inches(4, 2.9) #fig.set_size_inches(5, 3.5)
                plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=17)    # fontsize of the tick labels
                plt.rc('legend', fontsize=10)    # legend fontsize
                plt.rc('axes', axisbelow=True)
                plt.axis('equal')
                plt.grid(color="whitesmoke")
                plot_desired_circle(np.mean(metric_values))
                #plt.ylim(0, 2.1)
                #plt.xlim(-1, 1)

            elif metric == "spine_angle":
                plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
                plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
                plt.rc('legend', fontsize=17)    # legend fontsize
                plt.plot(t, metric_values, label = label)
                plt.plot(t, np.mean(metric_values) * np.ones(len(metric_values)), label = "Mean spine motor angle")
                np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                np.savetxt((csv_path / f"spine_angle_{model_name}.csv").resolve(),metric_values,delimiter=",")
                
            #special treatment also for "foot_positions": we want to get the z-values from each foot to evaluate foot contact slip and foot clearance from the ground
            elif metric == "foot_positions":
                lf=[]
                rf=[]
                rh=[]
                lh=[]
                feet=[lf, rf, rh, lh]
                #append values from "metric_values" to according array
                #first loop in metric_values dict to get position dict from one time step of all feet
                for positions in metric_values:
                    #second loop for enumerating positions of each foot in one time step
                    for count, foot in enumerate(positions):
                        feet[count].append(positions[foot][2]) #[2] for z-positions
                # plt.plot(t, feet[0], label = "lf")
                # plt.plot(t, feet[1], label = "rf")
                # plt.plot(t, feet[2], label = "rh")
                # plt.plot(t, feet[3], label = "lh")
                #create subplots
                fig, axs = plt.subplots(2, 2)
                plt.figure(figure_name)
                axs[0, 0].plot(t, feet[0], 'tab:blue')
                np.savetxt((csv_path / f"feet_lf_z_{model_name}.csv").resolve(), feet[0], delimiter=",")
                axs[0, 0].plot(t, np.mean(feet[0]) * np.ones(len(feet[0])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 0].set_title("lf")
                axs[0, 1].plot(t, feet[1], 'tab:orange')
                np.savetxt((csv_path / f"feet_rf_z_{model_name}.csv").resolve(), feet[1], delimiter=",")
                axs[0, 1].plot(t, np.mean(feet[1]) * np.ones(len(feet[1])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 1].set_title('rf')
                axs[1, 0].plot(t, feet[2], 'tab:green')
                np.savetxt((csv_path / f"feet_rh_z_{model_name}.csv").resolve(), feet[2], delimiter=",")
                axs[1, 0].plot(t, np.mean(feet[2]) * np.ones(len(feet[2])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 0].set_title('rh')
                axs[1, 1].plot(t, feet[3], 'tab:purple')
                np.savetxt((csv_path / f"feet_lh_z_{model_name}.csv").resolve(), feet[3], delimiter=",")
                axs[1, 1].plot(t, np.mean(feet[3]) * np.ones(len(feet[3])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 1].set_title('lh')

                for ax in axs.flat:
                    ax.set(xlabel=x_label, ylabel=y_label)

                # #print means to console:
                # print("Mean z-position lf: " +  str(np.mean(feet[0])) + "m")
                # print("Mean z-position rf: " +  str(np.mean(feet[1])) + "m")
                # print("Mean z-position rh: " +  str(np.mean(feet[2]))+ "m")
                # print("Mean z-position lh: " +  str(np.mean(feet[3])) + "m")

                lf=[]
                rf=[]
                rh=[]
                lh=[]
                feet=[lf, rf, rh, lh]
                for positions in metric_values:
                    #second loop for enumerating positions of each foot in one time step
                    for count, foot in enumerate(positions):
                        feet[count].append(positions[foot][1]) #[2] for z-positions
                fig, axs = plt.subplots(2, 2)
                plt.figure(figure_name)
                axs[0, 0].plot(t, feet[0], 'tab:blue')
                np.savetxt((csv_path / f"feet_lf_y_{model_name}.csv").resolve(), feet[0], delimiter=",")
                axs[0, 0].plot(t, np.mean(feet[0]) * np.ones(len(feet[0])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 0].set_title("lf")
                axs[0, 1].plot(t, feet[1], 'tab:orange')
                np.savetxt((csv_path / f"feet_rf_y_{model_name}.csv").resolve(), feet[1], delimiter=",")
                axs[0, 1].plot(t, np.mean(feet[1]) * np.ones(len(feet[1])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 1].set_title('rf')
                axs[1, 0].plot(t, feet[2], 'tab:green')
                np.savetxt((csv_path / f"feet_rh_y_{model_name}.csv").resolve(), feet[2], delimiter=",")
                axs[1, 0].plot(t, np.mean(feet[2]) * np.ones(len(feet[2])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 0].set_title('rh')
                axs[1, 1].plot(t, feet[3], 'tab:purple')
                np.savetxt((csv_path / f"feet_lh_y_{model_name}.csv").resolve(), feet[3], delimiter=",")
                axs[1, 1].plot(t, np.mean(feet[3]) * np.ones(len(feet[3])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 1].set_title('lh')

                for ax in axs.flat:
                    ax.set(xlabel=x_label, ylabel=y_label)

                lf=[]
                rf=[]
                rh=[]
                lh=[]
                feet=[lf, rf, rh, lh]
                #append values from "metric_values" to according array
                #first loop in metric_values dict to get position dict from one time step of all feet
                for positions in metric_values:
                    #second loop for enumerating positions of each foot in one time step
                    for count, foot in enumerate(positions):
                        feet[count].append(positions[foot][0]) #[x] for z-positions
                # plt.plot(t, feet[0], label = "lf")
                # plt.plot(t, feet[1], label = "rf")
                # plt.plot(t, feet[2], label = "rh")
                # plt.plot(t, feet[3], label = "lh")
                #create subplots
                fig, axs = plt.subplots(2, 2)
                plt.figure(figure_name)
                axs[0, 0].plot(t, feet[0], 'tab:blue')
                np.savetxt((csv_path / f"feet_lf_x_{model_name}.csv").resolve(), feet[0], delimiter=",")
                axs[0, 0].plot(t, np.mean(feet[0]) * np.ones(len(feet[0])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 0].set_title("lf")
                axs[0, 1].plot(t, feet[1], 'tab:orange')
                np.savetxt((csv_path / f"feet_rf_x_{model_name}.csv").resolve(), feet[1], delimiter=",")
                axs[0, 1].plot(t, np.mean(feet[1]) * np.ones(len(feet[1])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 1].set_title('rf')
                axs[1, 0].plot(t, feet[2], 'tab:green')
                np.savetxt((csv_path / f"feet_rh_x_{model_name}.csv").resolve(), feet[2], delimiter=",")
                axs[1, 0].plot(t, np.mean(feet[2]) * np.ones(len(feet[2])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 0].set_title('rh')
                axs[1, 1].plot(t, feet[3], 'tab:purple')
                np.savetxt((csv_path / f"feet_lh_x_{model_name}.csv").resolve(), feet[3], delimiter=",")
                axs[1, 1].plot(t, np.mean(feet[3]) * np.ones(len(feet[3])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 1].set_title('lh')

                for ax in axs.flat:
                    ax.set(xlabel=x_label, ylabel=y_label)

            elif metric == "tangential_foot_velocities":
                lf=[]
                rf=[]
                rh=[]
                lh=[]
                feet=[lf, rf, rh, lh]
                #append values from "metric_values" to according array
                #first loop in metric_values dict to get tangential_velocities dict from one time step of all feet
                for tangential_velocities in metric_values:
                    #second loop for enumerating positions of each foot in one time step
                    for count, foot in enumerate(tangential_velocities):
                        feet[count].append(tangential_velocities[foot]) 

                #uncomment to create one plot with all tangential_foot_velocities     
                #plt.plot(t, feet[0], label = "lf")
                #plt.plot(t, feet[1], label = "rf")
                #plt.plot(t, feet[2], label = "rh")
                #plt.plot(t, feet[3], label = "lh")

                #create subplots
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].plot(t, feet[0], 'tab:blue')
                axs[0, 0].plot(t, np.mean(feet[0]) * np.ones(len(feet[0])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 0].set_title("lf")
                axs[0, 1].plot(t, feet[1], 'tab:orange')
                axs[0, 1].plot(t, np.mean(feet[1]) * np.ones(len(feet[1])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[0, 1].set_title('rf')
                axs[1, 0].plot(t, feet[2], 'tab:green')
                axs[1, 0].plot(t, np.mean(feet[2]) * np.ones(len(feet[2])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 0].set_title('rh')
                axs[1, 1].plot(t, feet[3], 'tab:purple')
                axs[1, 1].plot(t, np.mean(feet[3]) * np.ones(len(feet[3])), 'tab:red') #add this plot to indicate the mean over a episode
                axs[1, 1].set_title('lh')

                for ax in axs.flat:
                    ax.set(xlabel=x_label, ylabel=y_label)

                #print means to console:
                print("Mean tangential velocity lf: " +  str(np.mean(feet[0])) + "m/s")
                print("Mean tangential velocity rf: " +  str(np.mean(feet[1])) + "m/s")
                print("Mean tangential velocity rh: " +  str(np.mean(feet[2]))+ "m/s")
                print("Mean tangential velocity lh: " +  str(np.mean(feet[3])) + "m/s")

            elif metric == "angle_values":
                fig, axs = plt.subplots(2, 2)
                angle_m1_rl = []
                angle_m1_rr = []
                angle_m1_r_max = []
                angle_m1_r_min = []
                angle_m2_rl = []
                angle_m2_rr = []
                angle_m2_r_max = []
                angle_m2_r_min = []
                angle_m1_fl = []
                angle_m1_fr = []
                angle_m1_f_max = []
                angle_m1_f_min = []
                angle_m2_fl = []
                angle_m2_fr = []
                angle_m2_f_max = []
                angle_m2_f_min = []
                for angle_value in  metric_values:
                    angle_m1_rl.append(angle_value["angle_m1_rl"])
                    angle_m1_rr.append(angle_value["angle_m1_rr"])
                    angle_m1_r_max.append(angle_value["angle_m1_r_max"])
                    angle_m1_r_min.append(angle_value["angle_m1_r_min"])
                    angle_m2_rl.append(angle_value["angle_m2_rl"])
                    angle_m2_rr.append(angle_value["angle_m2_rr"])
                    angle_m2_r_max.append(angle_value["angle_m2_r_max"])
                    angle_m2_r_min.append(angle_value["angle_m2_r_min"])
                    angle_m1_fl.append(angle_value["angle_m1_fl"])
                    angle_m1_fr.append(angle_value["angle_m1_fr"])
                    angle_m1_f_max.append(angle_value["angle_m1_f_max"])
                    angle_m1_f_min.append(angle_value["angle_m1_f_min"])
                    angle_m2_fl.append(angle_value["angle_m2_fl"])
                    angle_m2_fr.append(angle_value["angle_m2_fr"])
                    angle_m2_f_max.append(angle_value["angle_m2_f_max"])
                    angle_m2_f_min.append(angle_value["angle_m2_f_min"])

                axs[0, 0].plot(t, angle_m1_rl, label = "left")
                axs[0, 0].plot(t, angle_m1_rr, label = "right")
                axs[0, 0].plot(t, angle_m1_r_max, label = "upper bound")
                axs[0, 0].plot(t, angle_m1_r_min, label = "lower bound")
                axs[0, 0].set_title("m1_rear")
                axs[0,0].legend()

                axs[0, 1].plot(t, angle_m2_rl, label = "left")
                axs[0, 1].plot(t, angle_m2_rr, label = "right")
                axs[0, 1].plot(t, angle_m2_r_max, label = "upper bound")
                axs[0, 1].plot(t, angle_m2_r_min, label = "lower bound")
                axs[0, 1].set_title("m2_rear")
                axs[0, 1].legend()

                axs[1, 0].plot(t, angle_m1_fl, label = "left")
                axs[1, 0].plot(t, angle_m1_fr, label = "right")
                axs[1, 0].plot(t, angle_m1_f_max, label = "upper bound")
                axs[1, 0].plot(t, angle_m1_f_min, label = "lower bound")
                axs[1, 0].set_title("m1_front")
                axs[1, 0].legend()

                axs[1, 1].plot(t, angle_m2_fl, label = "left")
                axs[1, 1].plot(t, angle_m2_fr, label = "right")
                axs[1, 1].plot(t, angle_m2_f_max, label = "upper bound")
                axs[1, 1].plot(t, angle_m2_f_min, label = "lower bound")
                axs[1, 1].set_title("m2_front")
                axs[1, 1].legend()       

                for ax in axs.flat:
                    ax.set(xlabel=x_label, ylabel=y_label)         
            
            else:
                #saving angle values of the current simulated episode as csv file for postprocessing with fft 
                if metric=="angle_m1_rl":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m1_rl_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m2_rl":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m2_rl_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m1_fl":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m1_fl_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m2_fl":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m2_fl_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m1_rr":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m1_rr_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m2_rr":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m2_rr_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m1_fr":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m1_fr_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if metric=="angle_m2_fr":
                    np.savetxt((csv_path / f"timestamp.csv"),t,delimiter=",")
                    np.savetxt((csv_path / f"angle_m2_fr_{model_name}.csv").resolve(),metric_values,delimiter=",")
                if colors==None:
                    plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
                    plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
                    plt.xlim(2, 5)
                    plt.plot(t, metric_values, label = label) 
                if colors!=None:
                    plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
                    plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
                    plt.xlim(2, 5)
                    plt.plot(t, metric_values, label = label, c=colors[i])

        if smoothing_window is not None:
            smoothed_metric_values = savgol_filter(metric_values, smoothing_window, 3)
            smoothed_label = label if smoothed_only else f"{label} smoothed"
            #Example for saving angle values as csv file for postprocessing with fft
            # if metric=="angle_m1_rl":
            #         np.savetxt("timestamp.csv",t,delimiter=",")
            #         np.savetxt("angle_m1_rl_smoothed.csv",smoothed_metric_values,delimiter=",")
            # if metric=="angle_m2_rl":
            #     np.savetxt("timestamp.csv",t,delimiter=",")
            #     np.savetxt("angle_m2_rl_smoothed.csv",smoothed_metric_values,delimiter=",")
            # if metric=="angle_m1_fl":
            #     np.savetxt("timestamp.csv",t,delimiter=",")
            #     np.savetxt("angle_m1_fl_smoothed.csv",smoothed_metric_values,delimiter=",")
            # if metric=="angle_m2_fl":
            #     np.savetxt("timestamp.csv",t,delimiter=",")
            #     np.savetxt("angle_m2_fl_smoothed.csv",smoothed_metric_values,delimiter=",")
            if metric == "velocity_in_desired_direction":
                # plt.rc('xtick', labelsize=25)  # fontsize of the tick labels
                # plt.rc('ytick', labelsize=25)  # fontsize of the tick labels
                # plt.rc('legend', fontsize=17)  # legend fontsize
                plt.plot(t, smoothed_metric_values, label=label)
                plt.plot(t, np.mean(smoothed_metric_values[200:]) * np.ones(len(metric_values)), label="mean velocity = "+str(np.mean(smoothed_metric_values[200:])))
            elif metric == "power":
                # plt.rc('xtick', labelsize=25)  # fontsize of the tick labels
                # plt.rc('ytick', labelsize=25)  # fontsize of the tick labels
                # plt.rc('legend', fontsize=17)  # legend fontsize
                plt.plot(t, smoothed_metric_values, label=label)
                plt.plot(t, np.mean(smoothed_metric_values[200:]) * np.ones(len(metric_values)), label="mean power = "+str(np.mean(smoothed_metric_values[200:])))
            else:
                plt.plot(t, smoothed_metric_values, label = smoothed_label)
    
    plt.grid(color="whitesmoke")
    plt.xlabel(x_label, fontsize=17) #17 for position plot #25 normale plots
    plt.ylabel(y_label, fontsize=17) #17 for position plot

    # plt.rc('xtick', labelsize=25)    # fontsize of the tick labels, 17 for position plot
    # plt.rc('ytick', labelsize=25)    # fontsize of the tick labels, 17 for position plot
    # plt.rc('legend', fontsize=17)    # legend fontsize (10 for position plot)

    plt.rc('xtick', labelsize=17)    # fontsize of the tick labels, 17 for position plot
    plt.rc('ytick', labelsize=17)    # fontsize of the tick labels, 17 for position plot
    plt.rc('legend', fontsize=10)    # legend fontsize (10 for position plot)

    if show_legend:
        if legend_loc==None:
            plt.legend(loc="upper right")
        else:
            plt.legend(loc=legend_loc)
    
    plt.tight_layout()
    #plt.savefig("filename.pdf", bbox_inches = 'tight', pad_inches = 0)

def plot_fitted_circle(x, y):
    #combine x and y positions to a common array
    x_y_positions = np.column_stack([x,y])
    #calculate center point, radius and variance of fitted circle
    xc, yc, r, s = cf.least_squares_circle(x_y_positions)
    #plot half of the fitted circle
    theta_fit = np.linspace(0, pi, 1800)
    x_fit = xc + r*np.cos(theta_fit)
    y_fit = yc + r*np.sin(theta_fit)
    #label = "Center point of fitted circle with r = " + str(round(r,2))
    #print("x-y-pos: " + str(xc) + str(yc))

    plt.plot(x_fit, y_fit, ":", label = "Fitted circle (r = " + str(round(r,2)) + "m)", color="red") #print in red
    #plt.scatter(xc, yc, label = label, c="#FF0000")

def plot_desired_circle(radius):
    theta = np.linspace(0, np.pi, 1000)
    #theta = np.linspace(0, 0.02, 1000) #for generalization plot
    x  = radius*np.cos(theta) - radius
    if radius > 0:
        y = radius*np.sin(theta)
    else:
        y = - radius*np.sin(theta)
    plt.plot(x, y, label = ("Desired turning radius: " + str(radius) + "m"), color="#30d93b")


def generate_hildebrand_plot(episode_infos: List[Dict[str, any]], hildebrand_steps_start=0, hildebrand_steps=200, hildebrand_foot_order=["rh", "rf", "lf", "lh"], phase_gap_tolerance=0, min_phase_steps=0, x_scale_factor=1, x_label="Steps", episode_num=0):
    foot_stance_phases, foot_swing_phases = build_foot_stride_phases(episode_infos, phase_gap_tolerance=phase_gap_tolerance, min_phase_steps=min_phase_steps)
    for foot_phases in [foot_stance_phases, foot_swing_phases]:
        for foot, phases in foot_phases.items():
            foot_phases[foot] = [(phase[0] * x_scale_factor, phase[1] * x_scale_factor) for phase in phases]

    hildebrand_steps_start = hildebrand_steps_start * x_scale_factor
    hildebrand_steps = hildebrand_steps * x_scale_factor
    
    fig, ax = plt.subplots(num = f"stance_phases_episode_{episode_num}")

    for i, foot in enumerate(hildebrand_foot_order):
        included_contacts = [(contact[0], contact[1]) for contact in foot_stance_phases[foot] if (contact[0] + contact[1]) >= hildebrand_steps_start and contact[0] <= (hildebrand_steps_start + hildebrand_steps)]
        ax.broken_barh(included_contacts, ((i * 5) + 4, 2), facecolors='black')

    ax.set_ylim(1, 24)
    ax.set_xlim(hildebrand_steps_start, hildebrand_steps_start + hildebrand_steps)
    ax.set_xlabel(x_label, fontsize=25)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_yticklabels([label.upper() for label in hildebrand_foot_order])
    ax.grid(False)

    plt.tight_layout()

    fig, ax = plt.subplots(num=f"swing_phases_episode_{episode_num}")

    for i, foot in enumerate(hildebrand_foot_order):
        included_contacts = [(contact[0], contact[1]) for contact in foot_swing_phases[foot] if (contact[0] + contact[1]) >= hildebrand_steps_start and contact[0] <= (hildebrand_steps_start + hildebrand_steps)]
        ax.broken_barh(included_contacts, ((i * 5) + 4, 2), facecolors='black')

    ax.set_ylim(1, 24)
    ax.set_xlim(hildebrand_steps_start, hildebrand_steps_start + hildebrand_steps)
    ax.set_xlabel(x_label, fontsize=25)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_yticklabels([label.upper() for label in hildebrand_foot_order])
    ax.grid(False)

    plt.tight_layout()


def compute_avg_gait_metrics(episode_infos: List[Dict[str, any]], env_dt, phase_gap_tolerance=0, min_phase_steps=0):
    foot_stance_phases, foot_swing_phases = build_foot_stride_phases(episode_infos, phase_gap_tolerance=phase_gap_tolerance, min_phase_steps=min_phase_steps)

    foot_stance_positions = {}
    for foot in foot_stance_phases.keys():
        foot_positions = [info["foot_positions"][foot] for info in episode_infos]
        foot_stance_positions[foot] = [(foot_positions[phase[0]-1][:2], foot_positions[(phase[0]-1) + phase[1]][:2]) for phase in foot_stance_phases[foot]]
    
    for foot_phases in [foot_stance_phases, foot_swing_phases]:
        for foot, phases in foot_phases.items():
            foot_phases[foot] = [(phase[0] * env_dt, phase[1] * env_dt) for phase in phases]

    gait_metrics = {}

    for foot in foot_stance_phases.keys():
        stance_durations = [stance_phase[1] for stance_phase in foot_stance_phases[foot]]
        swing_durations = [swing_phase[1] for swing_phase in foot_swing_phases[foot]]

        stride_lengths = [np.linalg.norm(stance_b[1] - stance_a[1]) for stance_a, stance_b in zip(foot_stance_positions[foot], foot_stance_positions[foot][1:])]
        contact_slips = [np.linalg.norm(stance[1] - stance[0]) for stance in foot_stance_positions[foot]]

        avg_stance_time = np.mean(stance_durations)
        avg_swing_time = np.mean(swing_durations)
        avg_stride_time = avg_stance_time + avg_swing_time
        avg_stride_frequency = 1 / avg_stride_time
        avg_duty_factor = avg_stance_time / avg_stride_time
        avg_stride_length = np.mean(stride_lengths)
        avg_contact_slip = np.mean(contact_slips)
        gait_metrics[f"{foot}_stance_time_avg"] = round(avg_stance_time, 2)
        gait_metrics[f"{foot}_stance_time_std"] = round(np.std(stance_durations), 2)
        gait_metrics[f"{foot}_swing_time_avg"] = round(avg_swing_time, 2)
        gait_metrics[f"{foot}_swing_time_std"] = round(np.std(swing_durations), 2)
        gait_metrics[f"{foot}_stride_time_avg"] = round(avg_stride_time, 2)
        gait_metrics[f"{foot}_stride_frequency_avg"] = round(avg_stride_frequency, 2)
        gait_metrics[f"{foot}_duty_factor_avg"] = round(avg_duty_factor, 2)
        gait_metrics[f"{foot}_stride_length_avg"] = round(avg_stride_length, 4)
        gait_metrics[f"{foot}_contact_slip_avg"] = round(avg_contact_slip, 4)

    return gait_metrics

def build_foot_stride_phases(episode_infos: List[Dict[str, any]], phase_gap_tolerance=0, min_phase_steps=0):
    foot_stance_phases = {
        "lf": [(1, 0)],
        "rf": [(1, 0)],
        "rh": [(1, 0)],
        "lh": [(1, 0)]
    }

    foot_swing_phases = {
        "lf": [(1, 0)],
        "rf": [(1, 0)],
        "rh": [(1, 0)],
        "lh": [(1, 0)]
    }
    # foot_contact_rf = []

    for step, info in enumerate(episode_infos):
        for foot, has_contact in info.get("foot_contacts").items():
            # if foot == 'rf':
            #     foot_contact_rf.append(has_contact)
            if has_contact:
                if sum(foot_stance_phases[foot][-1]) + phase_gap_tolerance < step:
                    # Beginning of a new stance phase
                    foot_stance_phases[foot].append((step+1, 0))
                else:
                    # Extend the current stance phase up to the current step
                    foot_stance_phases[foot][-1] = (foot_stance_phases[foot][-1][0], (step + 1) - foot_stance_phases[foot][-1][0])
            else:
                if sum(foot_swing_phases[foot][-1]) + phase_gap_tolerance < step:
                    # Beginning of a new swing phase
                    foot_swing_phases[foot].append((step+1, 0))
                else:
                    # Extend the current swing phase up to the current step
                    foot_swing_phases[foot][-1] = (foot_swing_phases[foot][-1][0], (step + 1) - foot_swing_phases[foot][-1][0])
    # print(foot_contact_rf)
    # np.savetxt("foot_contact_rf.csv", np.array(foot_contact_rf) , delimiter=",")
    for foot_phases in [foot_stance_phases, foot_swing_phases]:
        for foot, phases in foot_phases.items():
            foot_phases[foot] = [phase for phase in phases if phase[1] >= min_phase_steps]

    return foot_stance_phases, foot_swing_phases


# Smoothing used in tensorboard based on https://stackoverflow.com/a/49357445
def exponential_moving_average(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed
