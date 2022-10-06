import os
import random
import threading
import time
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
    value = keras.layers.Dense(1, activation='linear')(shared_fc)

    model = keras.Model(inputs=input, outputs=[policy, value])

    return model

class A3CAgent():
    def __init__(self, env_name, action_size):
        self._make_env(env_name)
        self.action_size = action_size
        self.state_size = (84, 84, 4)
        
        self.discount_factor = 0.99
        self.lr = 1e-4
        self.threads = 16
        
        self.global_model = A3C_model(self.action_size, self.state_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr, clipnorm=40.)
        
        self.model_path = os.path.join(os.getcwd(), 'save_model')

    def train(self):
        runners = [Runner(self.env, self.action_size, self.state_size) 
                   for _ in range(self.threads)]
        for i, runner in enumerate(runners):
            print(f'Start worker #{i}')
            runner.start()

        while True:
            self.global_model.save_weights(self.model_path)
            time.sleep(10*60)

    def _make_env(self, env_name):
        self.env = GrayScaleObservation(gym.make(env_name), keep_dim=True)
        self.env = ResizeObservation(self.env, 84)

class Runner(threading.Thread):
    global_episode = 0

    def __init__(self, env, action_size, state_size):
        threading.Thread.__init__(self)
        self.env = env
        self.action_size = action_size
        self.local_model = A3C_model(action_size, state_size)
        self.agent_waiting = 20
        self.t = 0
        self.t_max = 20

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.local_model(history)[0][0]
        policy = np.array(policy)
        policy = np.round(policy, 6)
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def train_model(self, ):
        pass

    def run(self):
        global episode
        step_per_episode = 0
        
        while episode < max_episode:
            obs = self.env.reset()
            score, life = 0, 5
            dead = False

            for _ in range(random.randint(1, self.agent_waiting)):
                obs, _, _, done = self.env.step(1)

            history = np.stack([obs, obs, obs, obs], axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while True:
                step_per_episode += 1
                self.t += 1
                
                action = self.get_action(history)
                if dead:
                    action, dead = 1, False

                obs, reward, done, info = self.env.step(action)

                next_state = obs
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                if life > info['lives']:
                    dead = True
                    life = info['lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                if dead:
                    history = np.stack((next_state, next_state, next_state, next_state))
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.t = 0

                if done:
                    episode += 1
                    print(f'episode: {episode}')

                    step_per_episode = 0
                    break

global episode
episode = 0
max_episode = 10000

env_name = 'BreakoutDeterministic-v4'
global_agent = A3CAgent(env_name, action_size=3)
global_agent.train()