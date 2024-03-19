import timeit
import gym

import ba_nermo_rl_locomotion
    
if __name__ == '__main__':
    env = gym.make('NermoCommandedTurningRadiusAndCommandedVelocityEnv-v0')
    env.reset()
    i = 0
    
    episode_start = timeit.default_timer()
    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
        # if i % 10000 == 0:
        #     # print(f"observation {i}: {observation}")
        #     # print(f"reward {i}: {reward}")
        print(env.current_yaw)
        if done:
            episode_end = timeit.default_timer()
            measured_episode_length = episode_end - episode_start

            print(f"Measured episode length: {measured_episode_length}")
            # break
            env.reset()
            episode_start = timeit.default_timer()
        i += 1
    env.close()