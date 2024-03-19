from stat import SF_APPEND
from turtle import position
from matplotlib.figure import Figure
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as ticker
import scipy.fftpack
from scipy.signal import correlate, find_peaks, savgol_filter
from pathlib import Path
import argparse
import os
import timeit
import gym
import numpy as np
import json
import ba_nermo_rl_locomotion
from sqlalchemy import false, true
from ba_nermo_rl_locomotion.utils.evaluation_util import plot_fitted_circle, plot_desired_circle
import matplotlib

BASE_PATH = Path(__file__).parent
LEARNED_MODEL_PATH = (BASE_PATH / "../trained_agents/models/").resolve()
TWOPI = 2.0*np.pi
scale = 1.0
desired_rad = 1.0

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#Function to identify main frequency, if various frequencies are identified by fft of all motors
#we use the frequency with the most occurencies over all motors
def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

#Functions for generating motor pattern
def motor_pattern(i, scale, freq, amp, mean_offset, recovered_time_shift):
    ft_result = scale * amp * np.sin(freq * TWOPI*i + recovered_time_shift*0.01*freq*TWOPI) + mean_offset
    return ft_result

def plot_episode_position(x,y,desired_radius):
    fig = plt.gcf()
    fig.set_size_inches(4, 2.9) #fig.set_size_inches(5, 3.5)
    plt.rc('axes', axisbelow=True)
    plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=17)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.axis('equal')
    plt.grid(color="whitesmoke")
    #plot_desired_circle(desired_radius)
    plt.scatter(x, y, label = "Position robot", s=1)
    #plot and fit a circle to xy-position data
    plot_fitted_circle(x, y)
    plt.legend()
    plt.legend(loc="upper right")
    plt.ylim(0, 1.7)
    #plt.xlim(-1,1)
    plt.xlabel("x-Position [m]", fontsize=17)
    plt.ylabel("y-Position [m]", fontsize=17)
    plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tp", "--training-path", type=str, help=f"Path to the folder of the training from which the model(s) should be loaded. The path must be relative to {LEARNED_MODEL_PATH}", required=True)
    parser.add_argument("-mtl", "--model-to-load", type=str, help="Name of the models that should be loaded from the training path", required=True)
    args = parser.parse_args()

    training_path = args.training_path
    training_path = (LEARNED_MODEL_PATH / training_path).resolve()
    model_name = args.model_to_load

    #File checking
    print("\n Searching for according csv files...")
    motor_files_found = false
    spine_file_found = false
    if os.path.isfile((training_path/f"angle_m1_rl_{model_name}.csv")):
        if os.path.isfile((training_path/f"angle_m1_rr_{model_name}.csv")):
            if os.path.isfile((training_path/f"angle_m2_rl_{model_name}.csv")):
                if os.path.isfile((training_path/f"angle_m2_rr_{model_name}.csv")):
                    if os.path.isfile((training_path/f"angle_m1_fl_{model_name}.csv")):
                        if os.path.isfile((training_path/f"angle_m1_fr_{model_name}.csv")):
                            if os.path.isfile((training_path/f"angle_m2_fl_{model_name}.csv")):
                                if os.path.isfile((training_path/f"angle_m2_fr_{model_name}.csv")):
                                    motor_files_found=true

    if motor_files_found:
        print("\t All leg motor angles were found.")
    else:
        print("\t Leg motor angles not found.")

    if os.path.isfile((training_path/f"spine_angle_{model_name}.csv")):
        print("\t Spine angles were found.")
    else:
        print("\t Spine angles not found.")

    #Load csv files
    t = genfromtxt((training_path/f"timestamp.csv"), delimiter=',')
    m1_fl = genfromtxt((training_path/f"angle_m1_fl_{model_name}.csv"), delimiter=',')
    m2_fl = genfromtxt((training_path/f"angle_m2_fl_{model_name}.csv"), delimiter=',')
    m1_fr = genfromtxt((training_path/f"angle_m1_fr_{model_name}.csv"), delimiter=',')
    m2_fr = genfromtxt((training_path/f"angle_m2_fr_{model_name}.csv"), delimiter=',')
    m1_rl = genfromtxt((training_path/f"angle_m1_rl_{model_name}.csv"), delimiter=',')
    m2_rl = genfromtxt((training_path/f"angle_m2_rl_{model_name}.csv"), delimiter=',')
    m1_rr = genfromtxt((training_path/f"angle_m1_rr_{model_name}.csv"), delimiter=',')
    m2_rr = genfromtxt((training_path/f"angle_m2_rr_{model_name}.csv"), delimiter=',')
    spine = genfromtxt((training_path/f"spine_angle_{model_name}.csv"), delimiter=',')

    angle_values={
        "m1_fl": m1_fl, 
        "m2_fl": m2_fl,
        "m1_fr": m1_fr, 
        "m2_fr": m2_fr, 
        "m1_rl": m1_rl, 
        "m2_rl": m2_rl,
        "m1_rr": m1_rr,
        "m2_rr": m2_rr, 
        "spine": spine
    }

    #helpful for plotting: We create the same again, but name it differently
    angle_values_original={
        "m1_fl": m1_fl, 
        "m2_fl": m2_fl,
        "m1_fr": m1_fr, 
        "m2_fr": m2_fr, 
        "m1_rl": m1_rl, 
        "m2_rl": m2_rl,
        "m1_rr": m1_rr,
        "m2_rr": m2_rr, 
        "spine": spine
    }

    # Offset because of initial accelaration until stationary gait is reached
    offset = 200
    # Sample spacing (timestep)
    T = 0.01
    #copy original timestamps for plotting reasons later
    t_long = t
    # cut the interval
    t = t[offset:]
    # Number of samplepoints
    N = len(t)
    plt.figure(num=0, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.7)
    #plt.suptitle("Motor angles", fontsize=18, y=0.95)
    #dictionary for storing the gait parameters: frequency, amplitude, phase shift
    gait_paramaters = {}
    for n, motor in enumerate (angle_values):
        angle_values[motor] = angle_values[motor][offset:]
        #center oscillation (subtract mean of oscillation)
        mean_offset = np.mean(angle_values[motor])
        angle_values[motor] = angle_values[motor] - mean_offset
        #FFT
        yf = scipy.fftpack.fft(angle_values[motor])
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        yf_plot = 2.0/N * np.abs(yf[:N//2])
        peaks, _ = find_peaks(yf_plot, height=0.01)
        #extract the frequency with the highest amplitude
        amp = yf_plot[peaks]
        index = np.argmax(amp)
        amp = amp[index]

        freq = xf[peaks]
        freq = freq[index]
        
        #save data of first motor for plotting reasons later:
        if n==0:
            amp_m1_fl = amp
            freq_m1_fl = freq
            xf_m1_fl = xf
            yf_plot_m1_fl = yf_plot
        #phase shift calculation: m1_fl as benchmark
        xcorr = correlate(angle_values["m1_fl"], angle_values[motor])
        # delta time array to match xcorr
        nsamples = angle_values["m1_fl"].size
        dt = np.arange(1-nsamples, nsamples)
        recovered_time_shift = dt[xcorr.argmax()]
        #save the gait parameters
        gait_paramaters[motor] = {"freq": freq, "amp": amp, "mean_offset": mean_offset, "recovered_time_shift": recovered_time_shift}
        #plot
        ax = plt.subplot(3, 3, n + 1)
        ax.plot(xf, yf_plot)
        ax.plot(freq, amp,"x")
        ax.set(xlabel="Frequency [Hz]", ylabel="Amplitude")
        ax.set_xlim(left=0, right=6)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_title("Shoulder FL") if motor=="m1_fl" else None
        ax.set_title("Shoulder FR") if motor=="m1_fr" else None
        ax.set_title("Elbow FL") if motor=="m2_fl" else None
        ax.set_title("Elbow FR") if motor=="m2_fr" else None
        ax.set_title("Hip RL") if motor=="m1_rl" else None
        ax.set_title("Hip RR") if motor=="m1_rr" else None
        ax.set_title("Knee RL") if motor=="m2_rl" else None
        ax.set_title("Knee RR") if motor=="m2_rr" else None
        ax.set_title("Spine") if motor=="spine" else None
        #ax.set_title(f"{motor}")
    #plt.savefig("fft_neg_0_5_rad.pdf", bbox_inches = 'tight', pad_inches = 0)

    #plot original angle values and fitted curves
    plt.figure(num=1, figsize=(7, 8))
    plt.subplots_adjust(hspace=0.5, wspace=0.7)
    for n, motor in enumerate (angle_values):
        #cut away offset again
        angle_values_original[motor] = angle_values_original[motor][offset:]
        motor_values_fitted = []
        for i in t_long:
            motor_value_fitted = motor_pattern(i, scale, gait_paramaters[motor]["freq"], gait_paramaters[motor]["amp"], gait_paramaters[motor]["mean_offset"], gait_paramaters[motor]["recovered_time_shift"])
            motor_values_fitted.append(motor_value_fitted)
        ax = plt.subplot(3, 3, n + 1)
        #save data of first plot for later
        if n==0:
            angle_values_original_m1_fl = angle_values_original[motor]
            motor_values_fitted_m1_fl = motor_values_fitted
        #plot original values of motor angles (RL learned gait)
        ax.plot(t, angle_values_original[motor])
        #plot values of fitted motor angles (fitted gait)
        ax.plot(t_long, motor_values_fitted)
        ax.set(xlabel="Time [s]", ylabel="Motor angles [rad]")
        ax.set_xlim(left=7, right=9)
        ax.set_title("Shoulder FL") if motor=="m1_fl" else None
        ax.set_title("Shoulder FR") if motor=="m1_fr" else None
        ax.set_title("Elbow FL") if motor=="m2_fl" else None
        ax.set_title("Elbow FR") if motor=="m2_fr" else None
        ax.set_title("Hip RL") if motor=="m1_rl" else None
        ax.set_title("Hip RR") if motor=="m1_rr" else None
        ax.set_title("Knee RL") if motor=="m2_rl" else None
        ax.set_title("Knee RR") if motor=="m2_rr" else None
        ax.set_title("Spine") if motor=="spine" else None
        #ax.set_title(f"{motor}")
    plt.figlegend(["RL learned motor oscillations", "Motor oscillations based on extracted gait parameters"], loc = 'lower center', ncol=5, labelspacing=0.)
    #plt.show()
    #plt.savefig("motor_oscillations_neg_0_5_rad.pdf", bbox_inches = 'tight', pad_inches = 0)

    #plots to visualize impact of scaling the amplitude (only demonstrated for one motor):
    #fft plot
    plt.figure(num=2, figsize=(3, 3))
    plt.rc('xtick', labelsize=17)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=17)    # fontsize of the tick labels
    plt.plot(xf_m1_fl, yf_plot_m1_fl)
    plt.plot(freq_m1_fl, amp_m1_fl,"x")
    plt.xlim(0, 6.0)
    plt.xlabel("Frequency [Hz]", fontsize=17)
    plt.ylabel("Amplitude", fontsize=17)
    plt.tight_layout()

    #motor angle plot with source data (RL) and fitted data (with and without scaling)
    plt.figure(num=3, figsize=(5, 3))
    plt.rc('axes', axisbelow=True)
    plt.grid(color="whitesmoke")
    #plot original values of motor angles (RL learned gait)
    plt.plot(t, angle_values_original_m1_fl)
    #plot values of fitted motor angles (fitted gait) without scaling --> remove scaling again
    plt.plot(t_long, ((motor_values_fitted_m1_fl - gait_paramaters["m1_fl"]["mean_offset"]) / scale) + gait_paramaters["m1_fl"]["mean_offset"])
    #plot values of fitted motor angles (fitted gait) with scaling
    plt.plot(t_long, motor_values_fitted_m1_fl)
    plt.xlim(13.5, 15.5)
    plt.ylim(0.25, 2.0)
    plt.legend(["RL learned gait", "Fitted gait without scaling", "Fitted gait with scaling"], loc="upper right")
    plt.xlabel("Time [s]", fontsize=17)
    plt.ylabel("Motor position [rad]", fontsize=17)
    plt.tight_layout()


    #Check if all frequencies are the same:
    # extracting value to compare
    test_val = gait_paramaters["m1_fl"]["freq"]
    all_frequencies = []
    for motor in gait_paramaters:
        all_frequencies.append(gait_paramaters[motor]["freq"])
        print("Frequency of " + str(motor) + ": " + str(gait_paramaters[motor]["freq"]))
        # if gait_paramaters[motor]["freq"] != test_val:
        #     exit("Error: Frequencies are not the same!")
    main_freq = most_frequent(all_frequencies)
    for motor in gait_paramaters:
        gait_paramaters[motor]["freq"] = main_freq
        #print("Frequency of " + str(motor) + ": " + str(gait_paramaters[motor]["freq"]))
    #save gait parameters
    with open("gait_parameters.txt", 'w') as convert_file:
        convert_file.write(json.dumps(gait_paramaters, indent=3, default=str))

    print(gait_paramaters)
    #Simulate the gait parameters
    #start gym environment
    env = gym.make('NermoCommandedTurningRadiusAndCommandedVelocityEnv-v0')
    env.reset()
    i = 0
    episode_start = timeit.default_timer()
    smoothed_speed=[]
    x_pos=[]
    y_pos=[]
    while True:
        #env.render()
        motor_values=[]
        for motor in gait_paramaters:
            motor_value = motor_pattern(i, scale, gait_paramaters[motor]["freq"], gait_paramaters[motor]["amp"], gait_paramaters[motor]["mean_offset"], gait_paramaters[motor]["recovered_time_shift"])
            motor_values.append(motor_value)
        observation, reward, done, info = env.step(np.array(motor_values)) 
        x_pos.append(-info["position"][0])
        y_pos.append(-info["position"][1])
        smoothed_speed.append(info["smoothed_speed"])
        if done:
            episode_end = timeit.default_timer()
            measured_episode_length = episode_end - episode_start
            print(f"Measured episode length: {measured_episode_length}") 
            # break
            plt.figure(num=4)
            plot_episode_position(x_pos,y_pos, desired_rad)
            #plt.savefig("filename.pdf", bbox_inches = 'tight', pad_inches = 0)
            #plt.figure(num=5)
            timesteps = np.linspace(0,(len(smoothed_speed)-1) * 0.01, len(smoothed_speed))
            #plt.plot(timesteps, smoothed_speed)
            #plt.xlabel("Time [s]")
            #plt.ylabel("Speed [m/s]")
            print("mean speed: " + str(np.mean(smoothed_speed)))
            plt.show()
            i = 0
            env.reset()
            smoothed_speed=[]
            x_pos=[]
            y_pos=[]
            episode_start = timeit.default_timer()
            break
        i += 0.01
    env.close()
    