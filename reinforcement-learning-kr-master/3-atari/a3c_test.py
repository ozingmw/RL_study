import random
import threading
import gym
import numpy as np
import tensorflow as tf
import keras
import keras.layers
import keras.optimizers

from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation

def A3C_model(action_size, state_size):
    input = keras.layers.Input(shape=state_size)
    conv1 = keras.layers.Conv2D(32, 8, 4, activation='relu')(input)
    conv2 = keras.layers.Conv2D(64, 4, 2, activation='relu')(conv1)
    conv3 = keras.layers.Conv2D(64, 3, 1, activation='relu')(conv2)
    flatten = keras.layers.Flatten()(conv3)
    shared_fc = keras.layers.Dense(512, activation='relu')(flatten)

    policy = keras.layers.Dense(action_size, activation='softmax')(shared_fc)
    value = keras.layers.Dense(1, activation='linear')

    model = keras.Model(inputs=input, outputs=[policy, value])

    return model

class A3CAgent():
    def __init__(self, action_size, state_size):
        self.global_model = A3C_model(action_size, state_size)

    def train(self):
        runners = [Runner(env) for _ in range(16)]
        for i, runner in enumerate(runners):
            print(f'Start worker #{i}')
            runner.start()

class Runner(threading.Thread):
    global_episode = 0

    def __init__(self, env):
        threading.Thread.__init__(self)

        self.env = env
        self.local_model = A3C_model(3, (84, 84, 4))
        self.agent_waiting = 20

    def run(self):
        global episode
        step_per_episode = 0
        
        while episode < max_episode:
            obs = self.env.reset()

            # for _ in range(random.randint(1, self.agent_waiting)):
            #     obs, _, _, done = self.env.step(1)

            # history = np.stack([obs, obs, obs, obs], axis=2)
            # history = np.reshape([history], (1, 84, 84, 4))

            while True:
                step_per_episode += 1
                action = self.env.action_space.sample()
                
                obs, reward, done, info = self.env.step(action)

                if done:
                    episode += 1
                    print(f'episode: {step_per_episode}')

                    step_per_episode = 0
                    break

global episode
episode = 0
max_episode = 10000


env = GrayScaleObservation(gym.make('BreakoutDeterministic-v4'), keep_dim=True)
env = ResizeObservation(env, 84)
