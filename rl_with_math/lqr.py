import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.linalg
import logging
import math
import gym
from keras import layers

LOGGER =ogging.getLogger(__name__)

configuration = {
    'T': 150,
    
}

class DynamicsPriorGMM:
    def __init__(self) -> None:
        pass


class LQRFLMagent(object):
    def __init__(self):
        self.env = gym.make("Pendulum-v1", g=9.80665)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self. action_bound = self.env.action_space.high[0]

        self.prior = DynamicsPriorGMM()


MAX_ITER = 60
agent = LQRFLMagent()

agent.update(MAX_ITER)
T = configuration('T')
Kt = agent.prev_control_data.Kt
kt = agent.prev_control_data.kt

x0 = agent.init_state

play_iter = 5
save_gain = []

for pn in range(play_iter):
    print(f"     play number: {pn+1}")
    if pn < 2:
        bad_init = True
        while bad_init:
            state = env.reset()
            x0_err = state - x0
            if np.sqrt(x0_err.T.dot(x0_err)) < 0.1:
                bad_init = False
    else:
        state = env.reset()

for time in range(T+1):
    env.render()
    action = Kt[time, :, :].dot(state) + kt[time, :]
    action = np.clip(action, -agent.action_bound, agent.action_bound)
    ang = math.atan2(state[1], state[0])
    print(f'Time: {time}, Angle: {ang*180.0/np.pi}, Action: {action}')
    save_gain.append([time, Kt[time, 0, 0], Kt[time, 0 ,1], Kt[time, 0, 2], kt[time, 0]])

    state, reward, _, _ = env.step(action)

np.savetxt('./moedel/kalman_gain.txt', save_gain)