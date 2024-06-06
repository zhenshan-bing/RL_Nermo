from gym.envs.registration import register

#In this file, all new created gym environments need to be registered

register(
    id='NermoMaxVelocity-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoMaxVelocityEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

register(
    id='NermoFixedVelocity-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoFixedVelocityEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

register(
    id='NermoCommandedVelocity-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoCommandedVelocityEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

register(
    id='NermoDirectedYawEnv-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoDirectedYawEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

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

register(
    id='NermoCommandedTurningRadiusAndFixedVelocityEnv-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoCommandedTurningRadiusAndFixedVelocityEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

register(
    id='NermoCommandedTurningRadiusAndCommandedVelocityEnv-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoCommandedTurningRadiusAndCommandedVelocityEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
    #max_episode_steps=2850,    # for testing with maze environment
)

# Legacy envs
register(
    id='nermo-short-episodes-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoMaxVelocityEnv',
    max_episode_steps=2000,     # max_episode_steps * frame_skip * mujoco_timestep = 2000 * 4 * 0.0025s = 20s
)

register(
    id='nermo-long-episodes-v0',
    entry_point='nermo_rl_locomotion.envs.nermo_envs:NermoMaxVelocityEnv',
    max_episode_steps=8000,     # max_episode_steps * frame_skip * mujoco_timestep = 8000 * 4 * 0.0025s = 80s
)