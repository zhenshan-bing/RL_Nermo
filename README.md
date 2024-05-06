# Pushing physical limit and uncovering motion templates of spine-based quadruped locomotion via reinforcement learning


# Setup & Installation
## Prerequisites
* Ubuntu 22.04
* Python 3.8


## Install MuJoCo 2.1.0
[Download MuJoCo 2.1.0 release from from the OpenAi Mujoco Website](https://mujoco.org/download)

- Download [mujoco210 linux](https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz) and extract it to `~/.mujoco` as `mujoco210`



## Install Python Packages and Special System Dependencies

- Add the Mujoco installation to `LD_LIBRARY_PATH` via adding the following to `.bashrc`:
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco200/bin
    ```
    Note: run `exec $SHELL` for the changes in `.bashrc` to have an effect
- Install build dependencies for mujoco_py on Ubuntu:
    ```bash
    sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
    sudo apt install patchelf
    ```
    cf. [https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting)
- Install setuptools for the installation in the next step: 
    ```bash 
    pip install setuptools
    ```
- Chose among the following installation options:
    * Option 1: Install minimal project requirements (for training, for hyperparameter optimization and for enjoying trained agents)
        ```bash
        pip install -e .
        pip install 'gym==0.20.0'
        ```
    * Option 2: Install with requirements for recording videos
        ```bash
        pip install -e .[recording]
        pip install 'gym==0.20.0'
        ```
    Note: the correct gym version for our purposes needs to be installed separately due to conflicting version requirements with [Stable Baselines 3 v1.4.0](https://github.com/DLR-RM/stable-baselines3/releases/tag/v1.4.0)
    


# Usage
## Train an agent
```bash
$ python3 train.py -h
usage: train.py [-h] [--env ENV] [--env-kwargs ENV_KWARGS] [-en EXPERIMENT_NAME] [-ed EXPERIMENT_DESC] [-a {a2c,ddpg,dqn,ppo,sac,td3,qrdqn,tqc}] [-n N_TIMESTEPS] [-params HYPERPARAMS [HYPERPARAMS ...]] [-s SEED]
                [--trained-agent TRAINED_AGENT] [--vec-env {auto,dummy,subproc}] [--eval-freq EVAL_FREQ] [--n-eval-episodes N_EVAL_EPISODES] [--n-eval-envs N_EVAL_ENVS] [--checkpoint-freq CHECKPOINT_FREQ] [--save-replay-buffer]
                [--device {auto,cuda,cpu}] [--num-threads NUM_THREADS] [--verbose VERBOSE] [--log-interval LOG_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Environment ID
  --env-kwargs ENV_KWARGS
                        Overwrite the specified keyword arguments for the environment (pass in json format e.g. {"energy_penalty_weight": -0.1}
  -en EXPERIMENT_NAME, --experiment-name EXPERIMENT_NAME
                        Name for the experiment (should be unique within the specified env)
  -ed EXPERIMENT_DESC, --experiment-desc EXPERIMENT_DESC
                        Detailed description for the experiment
  -a {a2c,ddpg,dqn,ppo,sac,td3,qrdqn,tqc}, --algo {a2c,ddpg,dqn,ppo,sac,td3,qrdqn,tqc}
                        RL algorithm (hyperparameters for each environment are defined in `hyperparameters/<algo>.yml`)
  -n N_TIMESTEPS, --n-timesteps N_TIMESTEPS
                        The number of timesteps to train with (-1 to use the number specified in the hyperparams file)
  -params HYPERPARAMS [HYPERPARAMS ...], --hyperparams HYPERPARAMS [HYPERPARAMS ...]
                        Overwrite specified hyperparameter from the hyperparams file (e.g. learning_rate:0.01 train_freq:10)
  -s SEED, --seed SEED  Random generator seed (-1 to choose a random seed)
  --trained-agent TRAINED_AGENT
                        Path to a pretrained agent to continue training
  --vec-env {auto,dummy,subproc}
                        VecEnv type (auto to chose the type automatically depending on whether the algorithm is multiprocessing capable or not)
  --eval-freq EVAL_FREQ
                        Evaluate the agent every n steps (if negative, no evaluation). Can be a float in the range (0, 1) or and integer. A float x in (0, 1) will be interpreted as n = x * n_timesteps (where n_timesteps is the
                        number of timesteps used for training)
  --n-eval-episodes N_EVAL_EPISODES
                        Number of episodes to use for evaluation
  --n-eval-envs N_EVAL_ENVS
                        Number of environments for evaluation
  --checkpoint-freq CHECKPOINT_FREQ
                        Save the model every n steps (if negative, no checkpoint). Can be a float in the range (0, 1) or and integer. A float x in (0, 1) will be interpreted as n = x * n_timesteps (where n_timesteps is the
                        number of timesteps used for training)
  --save-replay-buffer  Save the replay buffer too (when applicable)
  --device {auto,cuda,cpu}
                        Device on which the learning algorithm should be run. When set to auto, the code will run on the GPU (via cuda) if possible.
  --num-threads NUM_THREADS
                        Number of threads for PyTorch (-1 to use default)
  --verbose VERBOSE     Verbose mode (0: no output, 1: INFO)
  --log-interval LOG_INTERVAL
                        Override log interval (if negative, no change)
```

Note:
* Most arguments have reasonable defaults that can be inspected in `nermo_rl_locomotion/train.py`
* Available environments are registered in `nermo_rl_locomotion/__init__.py`  
* Hyperparameters for each environment are defined in `hyperparameters/<algo>.yml`
* Environment args currently need to be specified in the code of `nermo_rl_locomotion/env_kwargs.py`
* Trainings can be monitored in realtime using [TensorBoard](https://www.tensorflow.org/tensorboard)


## Run multiple experiments after another
For training multiple agents after another without manually calling `train.py` every time u can configure the experiments in `run_experiments.py` and call as follows:
```bash
python3 run_experiments.py
```

## Load and enjoy a trained agent
```bash
$ python3 enjoy.py -h
usage: enjoy.py [-h] -tp TRAINING_PATH [-mtl MODELS_TO_LOAD [MODELS_TO_LOAD ...]] [-s SEED] [--non-deterministic] [--norm-reward] [--video-length VIDEO_LENGTH]
                [--video-resolution VIDEO_RESOLUTION VIDEO_RESOLUTION] [--video-base-path VIDEO_BASE_PATH] [--cam-ids CAM_IDS [CAM_IDS ...]]
                [--n-episodes N_EPISODES] [--no-rendering] [--no-monitor-file] [--show-eval-plots] [--style-sheet {subfigure}]

optional arguments:
  -h, --help            show this help message and exit
  -tp TRAINING_PATH, --training-path TRAINING_PATH
                        Path to the folder of the training from which the model(s) should be loaded. The path can be absolute or relative to
                        /home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models
  -mtl MODELS_TO_LOAD [MODELS_TO_LOAD ...], --models-to-load MODELS_TO_LOAD [MODELS_TO_LOAD ...]
                        Names of the models that should be loaded from the training path
  -s SEED, --seed SEED  Random generator seed
  --non-deterministic   Pick actions using a non-deterministic version of the policy
  --norm-reward         Normalize reward, if applicable (trained with VecNormalize)
  --video-length VIDEO_LENGTH
                        Record a video of the agent for n steps (do not specify in order to not record a video at all and render the agent behavior to the
                        screen instead)
  --video-resolution VIDEO_RESOLUTION VIDEO_RESOLUTION
                        Resolution "width height" of the video that is to be recorded. The higher the resolution, the longer the recording takes.
  --video-base-path VIDEO_BASE_PATH
                        Path under which the recorded videos should be saved (do not specify in order to store the videos within a 'videos' folder at the
                        specified training path). Note: can only be set when the training path is relative to
                        /home/r8iy/Documents/dev_project_folder/ba_nermo_rl_locomotion/trained_agents/models
  --cam-ids CAM_IDS [CAM_IDS ...]
                        Ids of the MuJoCo cameras for which a video should be recorded (one video for each camera). For rendering to the screen the first camera
                        in the given list is used for the initial point of view.
  --n-episodes N_EPISODES
                        Number of rendered episodes to enjoy (-1 to loop until interrupted by ctrl+c or until the videos have been recorded)
  --no-rendering        Do not render the agent (useful when only measurements via monitor are needed)
  --no-monitor-file     Do not write the aggregated episode information of the monitor to a file
  --show-eval-plots     Plot diagrams for the locomotion evaluation after each episode
  --style-sheet {subfigure}
                        The matplotlib style sheet to use for the eval plots
```

