import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name="4x4")
Q_table = np.zeros([env.observation_space.n, env.action_space.n])
discount = 0.95
learning_rate = 0.75
episodes = 2000
total_reward = 0
reward_list = []

for episode in range(episodes):
    obs = env.reset()
    epsilon = 1. / ((episode / 100) + 1)
    while True:
        action = np.argmax(Q_table[obs, :] + np.random.randn(1, env.action_space.n) / (episode + 1))
        next_obs, reward, done, info = env.step(action)
        Q_table[obs, action] = (1-learning_rate) * Q_table[obs, action] + learning_rate * (reward + discount * np.max(Q_table[next_obs, :]))
        print(f"\repisode : {episode+1} / {episodes}, total_reward : {total_reward}", end="")
        obs = next_obs
        if done:
            total_reward += reward
            reward_list.append(total_reward)
            break

print(f"\naccuracy : {total_reward/episodes:.3f}")
sns.set_style("darkgrid")
plt.plot(reward_list)
plt.show()