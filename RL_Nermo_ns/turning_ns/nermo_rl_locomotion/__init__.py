from gym.envs.registration import register

#In this file, all new created gym environments need to be registered

register(
    id='NermoDesiredTurningRadiusEnv-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoDesiredTurningRadiusEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

register(
    id='NermoCommandedTurningRadiusEnv-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoCommandedTurningRadiusEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)
