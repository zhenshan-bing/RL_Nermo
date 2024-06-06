import timeit
import gym

import nermo_rl_locomotion
import numpy as np
import matplotlib.pyplot as plt
import itertools

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))
if __name__ == '__main__':
    env = gym.make('NermoFixedVelocity-v0')
    env.reset()
    i = 0
    
    episode_start = timeit.default_timer()
    ctrl=[]
    act_pos=[]
    t=[]
    while True:
        # env.render()
        act=env.action_space.sample()
        observation, reward, done, info = env.step(act)  # take a random action
        # print(env.get_sensor("thigh_joint_rl_vel", 1, use_correct_sensors=True)[0])
        #print(env.sim.model.actuator_ctrlrange.copy().astype(np.float32).T[0][1])
        # print("action",act)
        # print("---------")
        # print(env.sim.data.ctrl)
        # print(env.sim.data.actuator_force)
        # print("act_vel------")
        # print(env.sim.data.actuator_velocity)
        ctrl.append(env.sim.data.ctrl[0])
        act_pos.append(env.sim.data.actuator_length[0])
        t.append(i)
        # print(env.sim.data.actuator_force)
        # print(env.model.actuator_id2name(1))
        # print(env.model.actuator_id2name(3))
        # print(env.model.actuator_id2name(5))
        # print(env.model.actuator_id2name(7))
        # print(env.model.actuator_id2name(9))
        #print(observation)
        #print(env.action_space.sample())
        # observation, reward, done, info = env.step([-1.7571829, -10.0780842, 0.07053261, -0.06784979, 0.38229555, -0.8142879, -0.88462114, -0.6766941, -0.870562])
        # print(env.cal_q2_range(0))
        if i % 1000 == 0:
            #print(f"observation {i}: {observation}")
            #print(f"observation {i}: {info}")
            print(f"action {i}: {env.action_space.sample()[1]}")
            #print(f"reward {i}: {reward}")
            #print(env.get_sensor("com_quat", 4, use_correct_sensors=True))
        if done:
            episode_end = timeit.default_timer()
            measured_episode_length = episode_end - episode_start
            #print(env.get_sensor("com_quat", 4, use_correct_sensors=True))
            #print(quat2euler(env.get_sensor("com_quat", 4, use_correct_sensors=True))/3.14*180)
            #print(env.info)
            # print(env.sim.data.sensordata)
            # print(f"Measured episode length: {measured_episode_length}")
            # break
            env.reset()
            episode_start = timeit.default_timer()
            plt.plot(t,ctrl)
            plt.show()
            plt.plot(t,act_pos)
            plt.show()
        i += 1
    env.close()