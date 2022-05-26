import tensorflow as tf
from tensorflow import keras

import gym
import numpy as np


env = gym.make("FrozenLake-v1", is_slippery=False)
Q_table = np.zeros([env.observation_space.n, env.action_space.n])
discount = 0.95
episodes = 2000
total_reward = 0

for episode in range(episodes):
    obs = env.reset()
    epsilon = 1. / ((episode / 100) + 1)
    while True:
        if np.random.rand(1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.random.choice(np.flatnonzero(Q_table[obs, :] == Q_table[obs, :].max()))
        next_obs, reward, done, info = env.step(action)
        Q_table[obs, action] = reward + discount * np.max(Q_table[next_obs, :])
        print(f"\repisode : {episode+1} / {episodes}, total_reward : {total_reward}", end="")
        obs = next_obs
        if done:
            total_reward += reward
            break

print(f"\naccuracy : {total_reward/episodes:.3f}")